"""
PPO (Proximal Policy Optimization) 训练器模块

实现 RLHF 中的 PPO 训练循环。这是 RLHF 流水线的核心组件。

PPO 算法概述:
    PPO 是一种策略梯度算法，通过限制策略更新的幅度来保证训练稳定性。
    在 RLHF 中，PPO 用于优化语言模型策略，使其生成的文本获得更高的奖励。

核心数学:
    1. 重要性采样比率:
       r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)

    2. PPO 裁剪目标函数:
       L^CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]

    3. RLHF 中的总奖励:
       R_total = R_RM(x, y) - beta * KL(pi_theta || pi_ref)

    4. 完整损失:
       L = L^CLIP + c1 * L^VF + c2 * H(pi)
       其中 L^VF 是价值函数损失，H 是策略熵

PPO 在 RLHF 中的完整训练流程:
    for each iteration:
        1. Rollout: 用当前策略生成 response
        2. Evaluate: 奖励模型打分 + 参考模型 KL + Critic 价值
        3. Compute: 计算 GAE 优势和回报
        4. Optimize: 用 PPO 目标函数更新 Actor 和 Critic（多个 epoch）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """
    PPO 训练配置。

    包含所有 PPO 训练需要的超参数。
    这些超参数的选择对训练效果有重大影响。
    """
    # PPO 核心参数
    clip_epsilon: float = 0.2       # PPO 裁剪参数 epsilon
    gamma: float = 1.0              # 折扣因子（RLHF 中通常为 1.0）
    lam: float = 0.95               # GAE lambda 参数
    ppo_epochs: int = 4             # 每批数据的 PPO 更新轮数
    mini_batch_size: int = 8        # mini-batch 大小

    # 损失权重
    vf_coef: float = 0.5            # 价值函数损失权重
    entropy_coef: float = 0.01      # 熵正则化权重
    max_grad_norm: float = 1.0      # 梯度裁剪阈值

    # KL 惩罚
    init_kl_coef: float = 0.1       # 初始 KL 系数
    kl_target: float = 6.0          # 目标 KL 散度
    kl_horizon: float = 10000.0     # KL 自适应调整速度

    # 价值函数裁剪
    clip_value_loss: bool = True     # 是否裁剪价值损失
    vf_clip_range: float = 0.2       # 价值损失裁剪范围

    # 学习率
    actor_lr: float = 1e-5          # Actor 学习率
    critic_lr: float = 1e-5         # Critic 学习率


class PPOTrainer:
    """
    PPO 训练器。

    封装了 RLHF-PPO 训练的完整逻辑。

    关键设计:
    1. Actor-Critic 分离: Actor 和 Critic 有独立的优化器
    2. GAE 优势估计: 平衡偏差和方差
    3. 裁剪目标函数: 限制策略更新幅度
    4. 价值函数裁剪: 稳定价值网络训练
    5. 自适应 KL 控制: 动态调整 KL 惩罚系数

    Attributes:
        actor: 策略模型
        critic: 价值模型
        config: PPO 配置
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        config: PPOConfig = None
    ):
        """
        Args:
            actor: 策略模型（语言模型）
            critic: 价值模型
            config: PPO 配置
        """
        self.actor = actor
        self.critic = critic
        self.config = config or PPOConfig()

        # 优化器
        self.actor_optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=self.config.actor_lr,
            weight_decay=0.01
        )
        self.critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=self.config.critic_lr,
            weight_decay=0.01
        )

        # 自适应 KL 系数
        self.kl_coef = self.config.init_kl_coef
        self.kl_target = self.config.kl_target

        # 训练统计
        self.global_step = 0

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        使用 GAE 计算优势和回报。

        GAE 递推公式:
            delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            A_t = delta_t + gamma * lambda * A_{t+1}
            R_t = A_t + V(s_t)

        Args:
            rewards: 即时奖励 (含 KL 惩罚), [batch_size, seq_len]
            values: Critic 价值估计, [batch_size, seq_len]
            mask: 有效 token 掩码

        Returns:
            advantages: 白化后的优势估计
            returns: 目标回报
        """
        gamma = self.config.gamma
        lam = self.config.lam
        batch_size, seq_len = rewards.shape
        device = rewards.device

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros(batch_size, device=device)

        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_value = torch.zeros(batch_size, device=device)
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + gamma * next_value - values[:, t]

            if mask is not None:
                delta = delta * mask[:, t]
                if t < seq_len - 1:
                    last_gae = last_gae * mask[:, t + 1]

            last_gae = delta + gamma * lam * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values

        # 白化优势（仅对有效位置）
        if mask is not None:
            valid_advantages = advantages[mask.bool()]
            if valid_advantages.numel() > 1:
                adv_mean = valid_advantages.mean()
                adv_std = valid_advantages.std() + 1e-8
                advantages = (advantages - adv_mean) / adv_std
        else:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def compute_policy_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 PPO 裁剪策略损失。

        数学推导:
            r_t(theta) = exp(log pi_theta - log pi_old) = pi_theta / pi_old

            L^CLIP = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)

            当 A_t > 0（好的动作）:
                L = min(r_t * A_t, (1+eps) * A_t)
                如果 r_t > 1+eps，被裁剪为 (1+eps)*A_t
                直觉：即使动作很好，也不允许策略变化太大

            当 A_t < 0（差的动作）:
                L = min(r_t * A_t, (1-eps) * A_t)
                如果 r_t < 1-eps，被裁剪为 (1-eps)*A_t
                直觉：即使动作很差，也不允许过度惩罚

        Args:
            log_probs: 当前策略的对数概率, [batch_size, seq_len]
            old_log_probs: 旧策略的对数概率, [batch_size, seq_len]
            advantages: 优势估计, [batch_size, seq_len]
            mask: 有效 token 掩码

        Returns:
            loss: 策略损失（标量）
            metrics: 训练指标
        """
        eps = self.config.clip_epsilon

        # 重要性采样比率
        # r_t = exp(log_pi_new - log_pi_old) = pi_new / pi_old
        ratio = torch.exp(log_probs - old_log_probs)

        # 未裁剪目标
        surr1 = ratio * advantages

        # 裁剪目标
        clipped_ratio = torch.clamp(ratio, 1.0 - eps, 1.0 + eps)
        surr2 = clipped_ratio * advantages

        # PPO 目标：取两者较小值（更保守的估计）
        # 取负号因为我们要最大化目标，但优化器做的是最小化
        policy_loss = -torch.min(surr1, surr2)

        if mask is not None:
            policy_loss = (policy_loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            policy_loss = policy_loss.mean()

        # 计算训练指标
        with torch.no_grad():
            clipped = ((ratio - 1.0).abs() > eps).float()
            if mask is not None:
                clip_frac = (clipped * mask).sum() / mask.sum().clamp(min=1)
                approx_kl = ((ratio - 1.0) - (ratio.log()) * mask).sum() / mask.sum().clamp(min=1)
            else:
                clip_frac = clipped.mean()
                approx_kl = ((ratio - 1.0) - ratio.log()).mean()

        metrics = {
            "policy_loss": policy_loss.item(),
            "clip_fraction": clip_frac.item(),
            "approx_kl": approx_kl.item(),
            "ratio_mean": ratio.mean().item(),
            "ratio_std": ratio.std().item(),
        }

        return policy_loss, metrics

    def compute_value_loss(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算价值函数损失。

        两种模式:
        1. 标准 MSE 损失: L_VF = (V(s) - R)^2
        2. 裁剪价值损失（类似 PPO 裁剪）:
           V_clipped = V_old + clip(V - V_old, -eps, eps)
           L_VF = max((V - R)^2, (V_clipped - R)^2)

        裁剪价值损失的目的：防止价值函数的更新过大，
        与 PPO 策略裁剪的思想一致。

        Args:
            values: 当前 Critic 的价值预测, [batch_size, seq_len]
            old_values: 旧 Critic 的价值预测, [batch_size, seq_len]
            returns: GAE 计算的目标回报, [batch_size, seq_len]
            mask: 有效 token 掩码

        Returns:
            loss: 价值损失（标量）
            metrics: 训练指标
        """
        if self.config.clip_value_loss:
            # 裁剪价值损失
            vf_clip = self.config.vf_clip_range
            values_clipped = old_values + torch.clamp(
                values - old_values, -vf_clip, vf_clip
            )
            vf_loss1 = (values - returns) ** 2
            vf_loss2 = (values_clipped - returns) ** 2
            value_loss = torch.max(vf_loss1, vf_loss2)
        else:
            # 标准 MSE 损失
            value_loss = (values - returns) ** 2

        if mask is not None:
            value_loss = (value_loss * mask).sum() / mask.sum().clamp(min=1)
        else:
            value_loss = value_loss.mean()

        # 乘以系数
        value_loss = self.config.vf_coef * value_loss

        metrics = {
            "value_loss": value_loss.item(),
            "value_mean": values.mean().item(),
            "return_mean": returns.mean().item(),
        }

        return value_loss, metrics

    def compute_entropy_loss(
        self,
        logits: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算熵正则化损失。

        熵 H(pi) = -sum_a pi(a) log pi(a)

        熵正则化的作用:
        1. 鼓励探索：防止策略过早坍缩到少数 token
        2. 提高生成多样性：避免模式坍缩
        3. 平滑优化：提供更好的梯度信号

        我们最大化熵（即最小化负熵）。

        Args:
            logits: 模型输出 logits, [batch_size, seq_len, vocab_size]
            mask: 有效 token 掩码

        Returns:
            loss: 熵损失（标量，负熵）
            metrics: 训练指标
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # [batch_size, seq_len]

        if mask is not None:
            mean_entropy = (entropy * mask).sum() / mask.sum().clamp(min=1)
        else:
            mean_entropy = entropy.mean()

        # 负号：最大化熵 = 最小化负熵
        entropy_loss = -self.config.entropy_coef * mean_entropy

        metrics = {
            "entropy": mean_entropy.item(),
            "entropy_loss": entropy_loss.item(),
        }

        return entropy_loss, metrics

    def prepare_rewards(
        self,
        rm_rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        response_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        准备 token 级别的奖励信号。

        在 RLHF 中，奖励由两部分组成:
        1. KL 惩罚（逐 token）: r_kl_t = -beta * (log pi_theta(a_t|s_t) - log pi_ref(a_t|s_t))
        2. 奖励模型分数（仅在序列末尾）: r_rm

        最终每个 token 的奖励:
            r_t = -beta * kl_t         (中间 token)
            r_T = R_RM - beta * kl_T   (最后一个 token)

        Args:
            rm_rewards: 奖励模型分数, [batch_size]
            old_log_probs: 策略对数概率, [batch_size, seq_len]
            ref_log_probs: 参考对数概率, [batch_size, seq_len]
            response_mask: response 掩码, [batch_size, seq_len]

        Returns:
            token_rewards: 逐 token 奖励, [batch_size, seq_len]
        """
        # 逐 token 的 KL 惩罚
        kl_per_token = old_log_probs - ref_log_probs  # [batch_size, seq_len]
        kl_penalty = -self.kl_coef * kl_per_token

        # 初始化 token 奖励
        token_rewards = kl_penalty * response_mask

        # 在序列末尾加上奖励模型分数
        # 找到每个序列的最后一个有效 token 位置
        seq_lengths = response_mask.sum(dim=1).long()  # [batch_size]
        batch_size = rm_rewards.shape[0]

        for i in range(batch_size):
            last_pos = seq_lengths[i] - 1
            if last_pos >= 0:
                token_rewards[i, last_pos] += rm_rewards[i]

        return token_rewards

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rm_rewards: torch.Tensor,
        old_values: torch.Tensor,
        response_mask: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Dict[str, float]:
        """
        执行一个完整的 PPO 训练步骤。

        流程:
        1. 准备 token 级别奖励（KL 惩罚 + RM 分数）
        2. 计算 GAE 优势和目标回报
        3. 多轮 PPO 更新:
           a. Actor 前向传播，得到新的 log_probs 和 logits
           b. Critic 前向传播，得到新的 values
           c. 计算策略损失、价值损失、熵损失
           d. 更新参数

        Args:
            prompt_ids: [batch_size, prompt_len]
            response_ids: [batch_size, response_len]
            old_log_probs: [batch_size, response_len]
            ref_log_probs: [batch_size, response_len]
            rm_rewards: [batch_size]
            old_values: [batch_size, response_len]
            response_mask: [batch_size, response_len]
            attention_mask: [batch_size, total_len]

        Returns:
            训练统计字典
        """
        # 1. 准备 token 级别奖励
        token_rewards = self.prepare_rewards(
            rm_rewards, old_log_probs, ref_log_probs, response_mask
        )

        # 2. 计算 GAE 优势和回报
        with torch.no_grad():
            advantages, returns = self.compute_advantages(
                token_rewards, old_values, response_mask
            )

        # 3. 多轮 PPO 更新
        all_metrics = {
            "policy_loss": 0, "value_loss": 0, "entropy_loss": 0,
            "clip_fraction": 0, "approx_kl": 0, "entropy": 0,
        }
        num_updates = 0

        full_ids = torch.cat([prompt_ids, response_ids], dim=1)
        prompt_len = prompt_ids.shape[1]

        for epoch in range(self.config.ppo_epochs):
            # Actor 前向传播
            self.actor.train()
            actor_logits = self.actor(full_ids, attention_mask)
            # 提取 response 部分的 logits
            response_logits = actor_logits[:, prompt_len - 1:-1, :]
            # 计算新的对数概率
            new_log_probs = F.log_softmax(response_logits, dim=-1)
            new_log_probs = new_log_probs.gather(
                dim=-1, index=response_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Critic 前向传播
            self.critic.train()
            all_values = self.critic(full_ids, attention_mask)
            new_values = all_values[:, prompt_len:]

            # 计算策略损失
            policy_loss, policy_metrics = self.compute_policy_loss(
                new_log_probs, old_log_probs, advantages, response_mask
            )

            # 计算价值损失
            value_loss, value_metrics = self.compute_value_loss(
                new_values, old_values, returns, response_mask
            )

            # 计算熵损失
            entropy_loss, entropy_metrics = self.compute_entropy_loss(
                response_logits, response_mask
            )

            # 总损失
            total_loss = policy_loss + value_loss + entropy_loss

            # 更新 Actor
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.config.max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.max_grad_norm
            )

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            # 累积指标
            for k, v in policy_metrics.items():
                if k in all_metrics:
                    all_metrics[k] += v
            for k, v in value_metrics.items():
                if k in all_metrics:
                    all_metrics[k] += v
            for k, v in entropy_metrics.items():
                if k in all_metrics:
                    all_metrics[k] += v
            num_updates += 1

            # 早停：如果 KL 散度过大，停止更新
            if policy_metrics["approx_kl"] > 1.5 * self.kl_target:
                break

        # 平均指标
        for k in all_metrics:
            all_metrics[k] /= max(num_updates, 1)

        # 更新 KL 系数
        kl_mean = (old_log_probs - ref_log_probs).mean().item()
        self._update_kl_coef(abs(kl_mean))

        all_metrics["kl_coef"] = self.kl_coef
        all_metrics["kl_mean"] = kl_mean
        all_metrics["reward_mean"] = rm_rewards.mean().item()
        all_metrics["ppo_epochs_actual"] = num_updates

        self.global_step += 1
        return all_metrics

    def _update_kl_coef(self, kl_value: float):
        """
        自适应更新 KL 系数。

        当 KL > target: 增大 beta（加强约束）
        当 KL < target: 减小 beta（放松约束）
        """
        proportional_error = (kl_value - self.kl_target) / self.kl_target
        multiplier = 1.0 + (1.0 / self.config.kl_horizon) * proportional_error
        self.kl_coef = max(self.kl_coef * multiplier, 1e-6)


if __name__ == "__main__":
    from rollout import SimplePolicy, SimpleCritic, SimpleRewardModel, collect_rollouts

    print("=" * 60)
    print("PPO 训练器演示")
    print("=" * 60)

    # 配置
    vocab_size = 200
    d_model = 64
    n_heads = 4
    n_layers = 2
    max_seq_len = 128
    batch_size = 4
    prompt_len = 10
    max_new_tokens = 15

    # 1. 创建模型
    print("\n--- 1. 创建模型 ---")
    actor = SimplePolicy(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    critic = SimpleCritic(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    reward_model = SimpleRewardModel(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    ref_model = SimplePolicy(vocab_size, d_model, n_heads, n_layers, max_seq_len)

    for param in ref_model.parameters():
        param.requires_grad = False
    for param in reward_model.parameters():
        param.requires_grad = False

    print(f"Actor 参数量: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic 参数量: {sum(p.numel() for p in critic.parameters()):,}")

    # 2. 创建 PPO 训练器
    print("\n--- 2. 创建 PPO 训练器 ---")
    config = PPOConfig(
        clip_epsilon=0.2,
        ppo_epochs=4,
        actor_lr=1e-4,
        critic_lr=1e-4,
        init_kl_coef=0.1,
        kl_target=6.0,
        entropy_coef=0.01,
        vf_coef=0.5,
    )
    trainer = PPOTrainer(actor, critic, config)
    print(f"PPO 配置: epsilon={config.clip_epsilon}, "
          f"epochs={config.ppo_epochs}, KL_target={config.kl_target}")

    # 3. 训练循环
    print("\n--- 3. PPO 训练循环 ---")
    num_iterations = 5

    for iteration in range(num_iterations):
        # 3.1 采样 prompt
        prompt_ids = torch.randint(1, vocab_size, (batch_size, prompt_len))

        # 3.2 收集 rollout
        rollout = collect_rollouts(
            actor=actor,
            critic=critic,
            reward_model=reward_model,
            ref_model=ref_model,
            prompt_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
        )

        # 3.3 PPO 更新
        metrics = trainer.train_step(
            prompt_ids=rollout.prompt_ids,
            response_ids=rollout.response_ids,
            old_log_probs=rollout.old_log_probs.detach(),
            ref_log_probs=rollout.ref_log_probs.detach(),
            rm_rewards=rollout.rewards.detach(),
            old_values=rollout.values.detach(),
            response_mask=rollout.response_mask,
            attention_mask=rollout.attention_mask,
        )

        print(
            f"  Iter {iteration+1}/{num_iterations} | "
            f"Policy Loss: {metrics['policy_loss']:.4f} | "
            f"Value Loss: {metrics['value_loss']:.4f} | "
            f"Entropy: {metrics['entropy']:.4f} | "
            f"KL: {metrics['kl_mean']:.4f} | "
            f"Beta: {metrics['kl_coef']:.6f} | "
            f"Reward: {metrics['reward_mean']:.4f}"
        )

    print("\n--- 4. 训练总结 ---")
    print(f"总步数: {trainer.global_step}")
    print(f"最终 KL 系数: {trainer.kl_coef:.6f}")
    print(f"最终策略损失: {metrics['policy_loss']:.4f}")
    print(f"最终裁剪比例: {metrics['clip_fraction']:.4f}")

    print("\n" + "=" * 60)
    print("PPO 训练器演示完成!")
    print("=" * 60)
