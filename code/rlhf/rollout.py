"""
经验收集（Rollout）模块

实现 RLHF-PPO 训练中的经验收集（rollout / experience generation）过程。

在 RLHF 中，一次 rollout 的流程:
    1. 从数据集中采样 prompt
    2. 用当前策略（actor）生成 response
    3. 用奖励模型（reward model）对 response 打分
    4. 用参考模型（reference model）计算参考对数概率（用于 KL 惩罚）
    5. 用价值模型（critic）估计状态价值（用于 GAE 计算）
    6. 打包所有信息为 RolloutBatch，供 PPO 训练使用

四个模型的协作:
    - Actor (策略模型): 生成 response
    - Critic (价值模型): 估计状态价值 V(s)
    - Reward Model: 评估 response 质量
    - Reference Model: 提供 KL 惩罚的基准

这四个模型是 RLHF 的主要显存瓶颈，工程上需要精心管理。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class RolloutBatch:
    """
    Rollout 数据批次。

    包含 PPO 训练所需的所有信息。

    Attributes:
        prompt_ids: prompt 的 token id, [batch_size, prompt_len]
        response_ids: response 的 token id, [batch_size, response_len]
        old_log_probs: 采样时策略的对数概率, [batch_size, response_len]
        ref_log_probs: 参考模型的对数概率, [batch_size, response_len]
        rewards: 奖励模型给出的分数, [batch_size]
        values: critic 的价值估计, [batch_size, response_len]
        attention_mask: 注意力掩码, [batch_size, total_len]
        response_mask: response 部分的掩码, [batch_size, response_len]
    """
    prompt_ids: torch.Tensor
    response_ids: torch.Tensor
    old_log_probs: torch.Tensor
    ref_log_probs: torch.Tensor
    rewards: torch.Tensor
    values: torch.Tensor
    attention_mask: torch.Tensor
    response_mask: torch.Tensor

    def to(self, device: torch.device) -> "RolloutBatch":
        """将所有张量移到指定设备。"""
        return RolloutBatch(
            prompt_ids=self.prompt_ids.to(device),
            response_ids=self.response_ids.to(device),
            old_log_probs=self.old_log_probs.to(device),
            ref_log_probs=self.ref_log_probs.to(device),
            rewards=self.rewards.to(device),
            values=self.values.to(device),
            attention_mask=self.attention_mask.to(device),
            response_mask=self.response_mask.to(device),
        )


class SimplePolicy(nn.Module):
    """
    简化版策略模型（教学用）。

    在真实 RLHF 中，这是一个完整的语言模型（如 LLaMA、GPT）。
    这里我们用一个简化版本来演示 rollout 流程。

    Attributes:
        embedding: token 嵌入层
        transformer: Transformer 编码器
        lm_head: 语言模型头，输出 logits
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播，返回 logits。

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        # 因果掩码（确保只能看到前面的 token）
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float("-inf"),
            diagonal=1
        )

        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        hidden = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.lm_head(hidden)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 50
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        自回归生成 response。

        Args:
            prompt_ids: prompt 的 token id, [batch_size, prompt_len]
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数（越大越随机）
            top_k: top-k 采样的 k 值

        Returns:
            response_ids: 生成的 token id, [batch_size, max_new_tokens]
            log_probs: 每个 token 的对数概率, [batch_size, max_new_tokens]
        """
        batch_size = prompt_ids.shape[0]
        device = prompt_ids.device
        current_ids = prompt_ids.clone()

        generated_ids = []
        generated_log_probs = []

        for _ in range(max_new_tokens):
            # 截断过长的序列
            if current_ids.shape[1] > 200:
                current_ids = current_ids[:, -200:]

            logits = self.forward(current_ids)
            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :]  # [batch_size, vocab_size]

            # 温度缩放
            next_logits = next_logits / max(temperature, 1e-8)

            # Top-k 采样
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_logits, k=min(top_k, self.vocab_size))
                # 将非 top-k 的位置设为负无穷
                next_logits = torch.full_like(next_logits, float("-inf"))
                next_logits.scatter_(1, top_k_indices, top_k_logits)

            # 计算概率分布
            probs = F.softmax(next_logits, dim=-1)
            log_prob_dist = F.log_softmax(next_logits, dim=-1)

            # 采样
            next_token = torch.multinomial(probs, num_samples=1)  # [batch_size, 1]

            # 获取采样 token 的对数概率
            token_log_prob = log_prob_dist.gather(1, next_token).squeeze(-1)

            generated_ids.append(next_token.squeeze(-1))
            generated_log_probs.append(token_log_prob)

            # 拼接到当前序列
            current_ids = torch.cat([current_ids, next_token], dim=1)

        response_ids = torch.stack(generated_ids, dim=1)  # [batch_size, max_new_tokens]
        log_probs = torch.stack(generated_log_probs, dim=1)  # [batch_size, max_new_tokens]

        return response_ids, log_probs


class SimpleCritic(nn.Module):
    """
    简化版 Critic 模型（价值网络）。

    Critic 估计状态价值 V(s_t)，即从状态 s_t 开始的期望累积奖励。
    在 RLHF 中，s_t 是生成到第 t 个 token 时的状态。

    Critic 通常与 Actor 共享 backbone，只是头部不同:
    - Actor 的头部输出 logits（用于生成 token）
    - Critic 的头部输出标量价值（用于估计 V(s)）
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        # 价值头: 输出标量价值
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播，返回每个位置的状态价值。

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            values: [batch_size, seq_len]
        """
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device) * float("-inf"),
            diagonal=1
        )

        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        hidden = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        values = self.value_head(hidden).squeeze(-1)  # [batch_size, seq_len]
        return values


class SimpleRewardModel(nn.Module):
    """
    简化版奖励模型（教学用）。

    在 rollout 中使用，对完整的 (prompt + response) 给出标量奖励。
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算标量奖励。

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]

        Returns:
            reward: [batch_size]
        """
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embedding(input_ids) + self.pos_embedding(positions)

        if attention_mask is not None:
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        hidden = self.transformer(x, src_key_padding_mask=padding_mask)
        # 取最后一个有效位置
        if attention_mask is not None:
            last_pos = attention_mask.sum(dim=1) - 1
            last_pos = last_pos.clamp(min=0).long()
            batch_idx = torch.arange(input_ids.shape[0], device=input_ids.device)
            pooled = hidden[batch_idx, last_pos]
        else:
            pooled = hidden[:, -1, :]

        reward = self.reward_head(pooled).squeeze(-1)
        return reward


def collect_rollouts(
    actor: SimplePolicy,
    critic: SimpleCritic,
    reward_model: SimpleRewardModel,
    ref_model: SimplePolicy,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    kl_coef: float = 0.1
) -> RolloutBatch:
    """
    收集一批 rollout 经验。

    这是 RLHF 训练循环的核心步骤之一。

    流程:
        1. Actor 根据 prompt 生成 response
        2. Reward Model 对 (prompt + response) 打分
        3. Reference Model 计算参考对数概率
        4. Critic 估计状态价值
        5. 打包为 RolloutBatch

    Args:
        actor: 当前策略模型
        critic: 价值模型
        reward_model: 奖励模型
        ref_model: 参考模型（冻结的 SFT 模型）
        prompt_ids: prompt token id, [batch_size, prompt_len]
        max_new_tokens: 最大生成长度
        temperature: 生成温度
        kl_coef: KL 惩罚系数

    Returns:
        RolloutBatch: 包含所有 PPO 训练所需数据
    """
    device = prompt_ids.device
    batch_size, prompt_len = prompt_ids.shape

    # 1. Actor 生成 response
    response_ids, old_log_probs = actor.generate(
        prompt_ids, max_new_tokens=max_new_tokens, temperature=temperature
    )
    response_len = response_ids.shape[1]

    # 2. 拼接 prompt 和 response
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # [B, prompt_len + response_len]
    total_len = full_ids.shape[1]

    # 3. 创建 attention mask
    attention_mask = torch.ones(batch_size, total_len, device=device)
    response_mask = torch.ones(batch_size, response_len, device=device)

    # 4. Reward Model 打分
    with torch.no_grad():
        rewards = reward_model(full_ids, attention_mask)  # [batch_size]

    # 5. Reference Model 计算参考对数概率
    with torch.no_grad():
        ref_logits = ref_model(full_ids, attention_mask)
        # 取 response 部分的对数概率
        ref_response_logits = ref_logits[:, prompt_len - 1:-1, :]  # 对齐: 输入 token t 预测 token t+1
        ref_log_probs_all = F.log_softmax(ref_response_logits, dim=-1)
        ref_log_probs = ref_log_probs_all.gather(
            dim=-1, index=response_ids.unsqueeze(-1)
        ).squeeze(-1)  # [batch_size, response_len]

    # 6. Critic 估计价值
    with torch.no_grad():
        all_values = critic(full_ids, attention_mask)
        # 取 response 部分的价值
        values = all_values[:, prompt_len:]  # [batch_size, response_len]

    return RolloutBatch(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        old_log_probs=old_log_probs,
        ref_log_probs=ref_log_probs,
        rewards=rewards,
        values=values,
        attention_mask=attention_mask,
        response_mask=response_mask,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("Rollout (经验收集) 模块演示")
    print("=" * 60)

    # 设置参数
    vocab_size = 200
    d_model = 64
    n_heads = 4
    n_layers = 2
    max_seq_len = 128
    batch_size = 4
    prompt_len = 10
    max_new_tokens = 15

    # 1. 创建四个模型
    print("\n--- 1. 创建四个模型 ---")
    actor = SimplePolicy(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    critic = SimpleCritic(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    reward_model = SimpleRewardModel(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    ref_model = SimplePolicy(vocab_size, d_model, n_heads, n_layers, max_seq_len)

    # 冻结参考模型
    for param in ref_model.parameters():
        param.requires_grad = False

    print(f"Actor 参数量:   {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic 参数量:  {sum(p.numel() for p in critic.parameters()):,}")
    print(f"Reward 参数量:  {sum(p.numel() for p in reward_model.parameters()):,}")
    print(f"Ref 参数量:     {sum(p.numel() for p in ref_model.parameters()):,}")
    total = sum(
        sum(p.numel() for p in m.parameters())
        for m in [actor, critic, reward_model, ref_model]
    )
    print(f"四模型总参数量: {total:,}")

    # 2. 生成 prompt
    print(f"\n--- 2. 生成 prompt ---")
    prompt_ids = torch.randint(1, vocab_size, (batch_size, prompt_len))
    print(f"Prompt 形状: {prompt_ids.shape}")

    # 3. 收集 Rollout
    print(f"\n--- 3. 收集 Rollout ---")
    rollout_batch = collect_rollouts(
        actor=actor,
        critic=critic,
        reward_model=reward_model,
        ref_model=ref_model,
        prompt_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        kl_coef=0.1,
    )

    print(f"Prompt IDs 形状:       {rollout_batch.prompt_ids.shape}")
    print(f"Response IDs 形状:     {rollout_batch.response_ids.shape}")
    print(f"Old Log Probs 形状:    {rollout_batch.old_log_probs.shape}")
    print(f"Ref Log Probs 形状:    {rollout_batch.ref_log_probs.shape}")
    print(f"Rewards 形状:          {rollout_batch.rewards.shape}")
    print(f"Values 形状:           {rollout_batch.values.shape}")
    print(f"Attention Mask 形状:   {rollout_batch.attention_mask.shape}")
    print(f"Response Mask 形状:    {rollout_batch.response_mask.shape}")

    print(f"\n奖励分数: {rollout_batch.rewards.tolist()}")
    print(f"对数概率范围: [{rollout_batch.old_log_probs.min():.4f}, "
          f"{rollout_batch.old_log_probs.max():.4f}]")
    print(f"价值范围: [{rollout_batch.values.min():.4f}, "
          f"{rollout_batch.values.max():.4f}]")

    # 4. KL 散度检查
    print(f"\n--- 4. KL 散度检查 ---")
    kl = rollout_batch.old_log_probs - rollout_batch.ref_log_probs
    print(f"KL 散度 (per token): 均值={kl.mean():.4f}, 标准差={kl.std():.4f}")
    print(f"KL 散度 (per sequence): {kl.sum(dim=1).tolist()}")

    print("\n" + "=" * 60)
    print("Rollout 演示完成!")
    print("=" * 60)
