"""
奖励模型（Reward Model）训练模块

实现基于 Bradley-Terry 偏好模型的奖励模型训练。

核心数学原理:
    给定 prompt x 和两个 response y_w (chosen) 与 y_l (rejected)，
    Bradley-Terry 模型假设人类偏好的概率为:

        P(y_w > y_l | x) = sigma(r(x, y_w) - r(x, y_l))

    其中 sigma 是 sigmoid 函数，r 是奖励函数。

    最大化似然等价于最小化损失:
        L_RM = -log sigma(r(x, y_w) - r(x, y_l))

    这等价于一个二分类问题：奖励模型需要学会给 chosen response
    更高的分数。

奖励模型在 RLHF 中的作用:
    1. 将人类偏好编码为标量奖励信号
    2. 为 PPO 训练提供优化目标
    3. 替代人类在线评估，实现可扩展的对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class RewardModel(nn.Module):
    """
    基于 Transformer 的奖励模型。

    架构:
        输入 token ids -> Transformer Backbone -> 池化 -> 线性层 -> 标量奖励

    奖励模型通常基于预训练语言模型初始化（如 SFT 模型的 backbone），
    然后在人类偏好数据上微调。最后一层的语言模型头（lm_head）被替换为
    一个输出标量的线性层（reward head）。

    Attributes:
        backbone: Transformer 编码器（简化版）
        reward_head: 输出标量奖励的线性层
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: 词汇表大小
            d_model: 模型隐藏维度
            n_heads: 注意力头数
            n_layers: Transformer 层数
            max_seq_len: 最大序列长度
            dropout: Dropout 概率
        """
        super().__init__()

        self.d_model = d_model

        # Token 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-Norm
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # 奖励头：将隐藏状态映射为标量奖励
        # 使用两层 MLP 增强表达能力
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier 初始化，奖励头的最后一层使用较小的初始化。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # 奖励头最后一层使用较小的初始化
        # 确保初始奖励分数接近0，有利于训练稳定性
        nn.init.normal_(self.reward_head[-1].weight, std=0.01)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播：输入 token ids，输出标量奖励。

        Args:
            input_ids: token id 序列, [batch_size, seq_len]
            attention_mask: 注意力掩码, [batch_size, seq_len]
                            1 = 有效 token, 0 = 填充 token

        Returns:
            reward: 标量奖励, [batch_size]
        """
        batch_size, seq_len = input_ids.shape

        # Token 嵌入 + 位置嵌入
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # 创建 padding mask (PyTorch TransformerEncoder 需要的格式)
        if attention_mask is not None:
            # TransformerEncoder 的 src_key_padding_mask: True = 忽略
            padding_mask = (attention_mask == 0)
        else:
            padding_mask = None

        # Transformer 编码
        hidden_states = self.transformer(x, src_key_padding_mask=padding_mask)

        # 池化策略：取最后一个有效 token 的隐藏状态
        # 这是因为在因果语言模型中，最后一个 token 聚合了所有前面的信息
        if attention_mask is not None:
            # 找到每个序列的最后一个有效位置
            last_positions = attention_mask.sum(dim=1) - 1  # [batch_size]
            last_positions = last_positions.clamp(min=0)
            # 提取对应位置的隐藏状态
            batch_indices = torch.arange(batch_size, device=input_ids.device)
            pooled = hidden_states[batch_indices, last_positions]  # [batch_size, d_model]
        else:
            # 没有 mask 时取最后一个位置
            pooled = hidden_states[:, -1, :]

        # 通过奖励头得到标量奖励
        reward = self.reward_head(pooled).squeeze(-1)  # [batch_size]
        return reward


def compute_reward_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    margin: float = 0.0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算 Bradley-Terry 偏好损失。

    数学推导:
        给定 chosen 奖励 r_w 和 rejected 奖励 r_l：

        似然：P(w > l) = sigma(r_w - r_l)
        对数似然：log P(w > l) = log sigma(r_w - r_l)
        损失（负对数似然）：L = -log sigma(r_w - r_l)

        可选 margin: L = -log sigma(r_w - r_l - margin)
        margin 鼓励 chosen 和 rejected 之间有更大的分数差距。

    梯度分析:
        dL/d(r_w) = -(1 - sigma(r_w - r_l))
        dL/d(r_l) = (1 - sigma(r_w - r_l))

        当 r_w >> r_l 时，sigma ≈ 1，梯度接近0（已学好）
        当 r_w ≈ r_l 时，sigma ≈ 0.5，梯度最大（最大学习信号）
        当 r_w << r_l 时，sigma ≈ 0，梯度接近-1（强烈推动学习）

    Args:
        chosen_rewards: chosen response 的奖励分数, [batch_size]
        rejected_rewards: rejected response 的奖励分数, [batch_size]
        margin: 分数差距的 margin

    Returns:
        loss: 标量损失
        metrics: 训练指标字典
    """
    # Bradley-Terry 损失
    # L = -log(sigma(r_w - r_l - margin))
    # 等价于 binary cross entropy，target=1
    diff = chosen_rewards - rejected_rewards - margin
    loss = -F.logsigmoid(diff).mean()

    # 计算训练指标
    with torch.no_grad():
        # 准确率：chosen 分数高于 rejected 的比例
        accuracy = (chosen_rewards > rejected_rewards).float().mean().item()
        # 平均分数差
        reward_diff = diff.mean().item()
        # 各自的平均分数
        chosen_mean = chosen_rewards.mean().item()
        rejected_mean = rejected_rewards.mean().item()

    metrics = {
        "loss": loss.item(),
        "accuracy": accuracy,
        "reward_diff": reward_diff,
        "chosen_reward_mean": chosen_mean,
        "rejected_reward_mean": rejected_mean,
    }

    return loss, metrics


class RewardModelTrainer:
    """
    奖励模型训练器。

    封装了训练循环、评估、日志记录等逻辑。

    训练流程:
        1. 从偏好数据集加载 (prompt, chosen, rejected) 三元组
        2. 分别计算 chosen 和 rejected 的奖励分数
        3. 使用 Bradley-Terry 损失更新奖励模型
        4. 定期评估和保存
    """

    def __init__(
        self,
        model: RewardModel,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        margin: float = 0.0
    ):
        """
        Args:
            model: 奖励模型实例
            lr: 学习率
            weight_decay: 权重衰减
            margin: Bradley-Terry 损失的 margin
        """
        self.model = model
        self.margin = margin
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        执行一步训练。

        Args:
            batch: 包含 chosen 和 rejected 的编码数据

        Returns:
            训练指标字典
        """
        self.model.train()

        # 分别计算 chosen 和 rejected 的奖励
        chosen_rewards = self.model(
            batch["chosen_input_ids"],
            batch.get("chosen_attention_mask")
        )
        rejected_rewards = self.model(
            batch["rejected_input_ids"],
            batch.get("rejected_attention_mask")
        )

        # 计算损失
        loss, metrics = compute_reward_loss(
            chosen_rewards, rejected_rewards, self.margin
        )

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return metrics

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: "DataLoader"
    ) -> Dict[str, float]:
        """
        在验证集上评估。

        Args:
            dataloader: 验证集 DataLoader

        Returns:
            评估指标字典
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        total_diff = 0.0
        num_batches = 0

        for batch in dataloader:
            chosen_rewards = self.model(
                batch["chosen_input_ids"],
                batch.get("chosen_attention_mask")
            )
            rejected_rewards = self.model(
                batch["rejected_input_ids"],
                batch.get("rejected_attention_mask")
            )
            loss, metrics = compute_reward_loss(
                chosen_rewards, rejected_rewards, self.margin
            )
            total_loss += metrics["loss"]
            total_acc += metrics["accuracy"]
            total_diff += metrics["reward_diff"]
            num_batches += 1

        return {
            "eval_loss": total_loss / max(num_batches, 1),
            "eval_accuracy": total_acc / max(num_batches, 1),
            "eval_reward_diff": total_diff / max(num_batches, 1),
        }


if __name__ == "__main__":
    from preference_dataset import (
        PreferenceDataset,
        generate_synthetic_preferences,
        create_preference_collator,
    )
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("奖励模型训练演示")
    print("=" * 60)

    # 1. 准备数据
    print("\n--- 1. 准备偏好数据 ---")
    raw_data = generate_synthetic_preferences(num_samples=200)
    train_data = raw_data[:160]
    val_data = raw_data[160:]

    train_dataset = PreferenceDataset(train_data, max_length=64)
    val_dataset = PreferenceDataset(val_data, max_length=64)

    collate_fn = create_preference_collator()
    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=16, collate_fn=collate_fn
    )
    print(f"训练集: {len(train_dataset)} 条, 验证集: {len(val_dataset)} 条")

    # 2. 创建奖励模型
    print("\n--- 2. 创建奖励模型 ---")
    model = RewardModel(
        vocab_size=256,  # 简单字符编码
        d_model=128,
        n_heads=4,
        n_layers=2,
        max_seq_len=64,
        dropout=0.1
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 3. 训练
    print("\n--- 3. 训练奖励模型 ---")
    trainer = RewardModelTrainer(model, lr=1e-3, margin=0.0)

    num_epochs = 5
    for epoch in range(num_epochs):
        epoch_metrics = {"loss": 0, "accuracy": 0, "count": 0}
        for batch in train_loader:
            metrics = trainer.train_step(batch)
            epoch_metrics["loss"] += metrics["loss"]
            epoch_metrics["accuracy"] += metrics["accuracy"]
            epoch_metrics["count"] += 1

        avg_loss = epoch_metrics["loss"] / epoch_metrics["count"]
        avg_acc = epoch_metrics["accuracy"] / epoch_metrics["count"]

        # 验证
        eval_metrics = trainer.evaluate(val_loader)

        print(
            f"  Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_loss:.4f}, Train Acc: {avg_acc:.4f} | "
            f"Val Loss: {eval_metrics['eval_loss']:.4f}, "
            f"Val Acc: {eval_metrics['eval_accuracy']:.4f}"
        )

    # 4. 测试奖励模型
    print("\n--- 4. 奖励模型推理测试 ---")
    model.eval()
    test_batch = next(iter(val_loader))
    with torch.no_grad():
        chosen_r = model(
            test_batch["chosen_input_ids"],
            test_batch["chosen_attention_mask"]
        )
        rejected_r = model(
            test_batch["rejected_input_ids"],
            test_batch["rejected_attention_mask"]
        )

    print(f"Chosen 奖励:   {chosen_r[:5].tolist()}")
    print(f"Rejected 奖励: {rejected_r[:5].tolist()}")
    print(f"Chosen > Rejected: {(chosen_r > rejected_r).float().mean().item():.2%}")

    print("\n" + "=" * 60)
    print("奖励模型训练演示完成!")
    print("=" * 60)
