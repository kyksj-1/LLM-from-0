"""
GRPO (Group Relative Policy Optimization) 损失函数实现

GRPO 由 DeepSeek 提出，用于 DeepSeek-Math 和 DeepSeek-R1 的训练。
核心创新：去掉 Critic（价值）模型，使用组内相对排序估计优势函数。

数学基础:
- 对同一 prompt 采样 G 个回答，获得奖励 {r_1, ..., r_G}
- 组内归一化优势: A_i = (r_i - mean(r)) / (std(r) + eps)
- GRPO 目标: PPO 裁剪目标 + KL 正则化

参考:
- Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning.
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List


def compute_group_advantages(
    rewards: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """
    计算组内归一化优势

    核心思想：对于同一个 prompt 的一组回答，通过组内均值和标准差
    将奖励转化为相对优势。这替代了传统 PPO 中 Critic 模型的角色。

    数学公式:
        A_i = (r_i - mean(r_1, ..., r_G)) / (std(r_1, ..., r_G) + eps)

    直觉：如果一个回答的奖励高于组平均，它的优势为正（应该被鼓励）；
    如果低于平均，优势为负（应该被抑制）。

    Args:
        rewards: 一组回答的奖励，形状 [group_size] 或 [batch_size, group_size]
        epsilon: 防止除零的小常数

    Returns:
        归一化后的优势，形状与输入相同
    """
    if rewards.dim() == 1:
        # 单个 prompt 的一组回答
        mean = rewards.mean()
        std = rewards.std()
        advantages = (rewards - mean) / (std + epsilon)
    elif rewards.dim() == 2:
        # 一批 prompt，每个有 G 个回答
        mean = rewards.mean(dim=-1, keepdim=True)
        std = rewards.std(dim=-1, keepdim=True)
        advantages = (rewards - mean) / (std + epsilon)
    else:
        raise ValueError(f"rewards 的维度应为 1 或 2，但得到 {rewards.dim()}")

    return advantages


def grpo_loss(
    policy_logps: torch.Tensor,
    old_policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    rewards: torch.Tensor,
    response_lengths: torch.Tensor,
    clip_epsilon: float = 0.2,
    beta: float = 0.01,
    advantage_epsilon: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    GRPO 损失函数

    GRPO 保留了 PPO 的裁剪目标函数，但用组内归一化优势替代 Critic 模型
    的优势估计，并添加 KL 正则化。

    完整推导:
    1. 采样: 对每个 prompt x，从 pi_old 采样 G 个回答 {y_1, ..., y_G}
    2. 奖励: 使用外部函数（如规则验证器）给每个回答打分 {r_1, ..., r_G}
    3. 优势估计: A_i = (r_i - mean(r)) / (std(r) + eps)
    4. 裁剪目标:
       L_clip = min(rho_i * A_i, clip(rho_i, 1-eps, 1+eps) * A_i)
       其中 rho_i = pi_theta(y_i|x) / pi_old(y_i|x)
    5. KL 正则化:
       D_KL = pi_ref(y_i|x) / pi_theta(y_i|x) - log(pi_ref(y_i|x)/pi_theta(y_i|x)) - 1
    6. 最终目标: L = -L_clip + beta * D_KL

    Args:
        policy_logps: 当前策略的对数概率，形状 [batch_size, group_size]
                      每个元素是 log pi_theta(y_i|x)
        old_policy_logps: 旧策略的对数概率，形状 [batch_size, group_size]
        ref_logps: 参考策略的对数概率，形状 [batch_size, group_size]
        rewards: 每个回答的奖励，形状 [batch_size, group_size]
        response_lengths: 每个回答的长度，形状 [batch_size, group_size]
        clip_epsilon: PPO 裁剪范围
        beta: KL 正则化系数
        advantage_epsilon: 优势归一化的防除零常数

    Returns:
        (loss, metrics): 损失标量和监控指标字典
    """
    batch_size, group_size = policy_logps.shape

    # 步骤 1: 计算组内归一化优势
    advantages = compute_group_advantages(rewards, epsilon=advantage_epsilon)

    # 步骤 2: 计算重要性采样比率 rho
    # rho = pi_theta / pi_old = exp(log pi_theta - log pi_old)
    log_ratios = policy_logps - old_policy_logps
    ratios = torch.exp(log_ratios)

    # 步骤 3: PPO 裁剪目标
    # L_1 = rho * A
    # L_2 = clip(rho, 1-eps, 1+eps) * A
    # L_clip = min(L_1, L_2)
    surrogate1 = ratios * advantages
    clipped_ratios = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate2 = clipped_ratios * advantages
    clip_objective = torch.min(surrogate1, surrogate2)

    # 按回答长度归一化
    # 这是 GRPO 的实现细节：确保不同长度的回答有相同的权重
    clip_objective = clip_objective.sum() / response_lengths.sum()

    # 步骤 4: KL 正则化（使用近似 KL 散度）
    # D_KL(pi_ref || pi_theta) approx= pi_ref/pi_theta - log(pi_ref/pi_theta) - 1
    # 这个近似在 pi_theta 接近 pi_ref 时精确
    log_ref_ratio = ref_logps - policy_logps
    approx_kl = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
    kl_penalty = approx_kl.sum() / response_lengths.sum()

    # 步骤 5: 总损失 = -裁剪目标 + KL 惩罚
    loss = -clip_objective + beta * kl_penalty

    # 监控指标
    with torch.no_grad():
        clip_fraction = ((ratios - 1.0).abs() > clip_epsilon).float().mean()
        metrics = {
            "loss": loss.detach(),
            "clip_objective": clip_objective.detach(),
            "kl_penalty": kl_penalty.detach(),
            "mean_advantage": advantages.mean().detach(),
            "std_advantage": advantages.std().detach(),
            "mean_ratio": ratios.mean().detach(),
            "clip_fraction": clip_fraction.detach(),
            "mean_reward": rewards.mean().detach(),
            "reward_std": rewards.std().detach(),
        }

    return loss, metrics


class GRPOLoss(nn.Module):
    """
    GRPO 损失函数的 nn.Module 封装

    使用示例:
        criterion = GRPOLoss(clip_epsilon=0.2, beta=0.01)
        loss, metrics = criterion(
            policy_logps, old_logps, ref_logps,
            rewards, response_lengths
        )
    """

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        beta: float = 0.01,
    ):
        """
        Args:
            clip_epsilon: PPO 裁剪参数
            beta: KL 正则化系数
        """
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.beta = beta

    def forward(
        self,
        policy_logps: torch.Tensor,
        old_policy_logps: torch.Tensor,
        ref_logps: torch.Tensor,
        rewards: torch.Tensor,
        response_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播"""
        return grpo_loss(
            policy_logps,
            old_policy_logps,
            ref_logps,
            rewards,
            response_lengths,
            clip_epsilon=self.clip_epsilon,
            beta=self.beta,
        )


def simulate_math_grpo(
    group_size: int = 16,
    num_prompts: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    模拟数学推理场景下的 GRPO 训练

    在这个模拟中：
    - 每个 prompt 是一道数学题
    - 模型采样 G 个回答
    - 奖励：答对 +1，答错 -1

    Args:
        group_size: 每个 prompt 的采样数量
        num_prompts: prompt 的数量

    Returns:
        模拟的 GRPO 训练数据
    """
    # 模拟对数概率
    policy_logps = -torch.rand(num_prompts, group_size) * 50 - 10
    old_policy_logps = policy_logps.detach() + torch.randn_like(policy_logps) * 0.1
    ref_logps = -torch.rand(num_prompts, group_size) * 50 - 10

    # 模拟奖励：假设每个 prompt 有一些回答正确（+1），一些错误（-1）
    # 正确率大约 30-50%
    correct_mask = torch.rand(num_prompts, group_size) > 0.5
    rewards = correct_mask.float() * 2 - 1  # +1 或 -1

    # 模拟回答长度
    response_lengths = torch.randint(10, 200, (num_prompts, group_size)).float()

    return {
        "policy_logps": policy_logps,
        "old_policy_logps": old_policy_logps,
        "ref_logps": ref_logps,
        "rewards": rewards,
        "response_lengths": response_lengths,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("GRPO (Group Relative Policy Optimization) 损失函数演示")
    print("=" * 60)

    torch.manual_seed(42)

    # 场景 1: 基本 GRPO 计算
    print("\n--- 场景 1: 基本 GRPO 损失计算 ---")
    data = simulate_math_grpo(group_size=16, num_prompts=4)

    loss, metrics = grpo_loss(
        data["policy_logps"],
        data["old_policy_logps"],
        data["ref_logps"],
        data["rewards"],
        data["response_lengths"],
        clip_epsilon=0.2,
        beta=0.01,
    )
    print(f"GRPO 损失: {loss.item():.4f}")
    print(f"裁剪目标: {metrics['clip_objective'].item():.4f}")
    print(f"KL 惩罚: {metrics['kl_penalty'].item():.4f}")
    print(f"平均奖励: {metrics['mean_reward'].item():.4f}")
    print(f"奖励标准差: {metrics['reward_std'].item():.4f}")
    print(f"裁剪比例: {metrics['clip_fraction'].item():.4f}")

    # 场景 2: 组内归一化优势的可视化
    print("\n--- 场景 2: 组内归一化优势 ---")
    rewards_example = torch.tensor([1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
    advantages = compute_group_advantages(rewards_example)
    print(f"原始奖励: {rewards_example.tolist()}")
    print(f"归一化优势: {[f'{a:.3f}' for a in advantages.tolist()]}")
    print(f"均值: {advantages.mean().item():.6f} (应接近 0)")
    print(f"标准差: {advantages.std().item():.6f} (应接近 1)")

    # 场景 3: 不同组大小的影响
    print("\n--- 场景 3: 组大小对优势估计的影响 ---")
    for G in [4, 8, 16, 32, 64]:
        # 模拟多次，计算优势估计的方差
        variances = []
        for _ in range(100):
            rewards_sample = torch.where(
                torch.rand(G) > 0.5,
                torch.ones(G),
                -torch.ones(G),
            )
            adv = compute_group_advantages(rewards_sample)
            variances.append(adv.var().item())

        avg_var = sum(variances) / len(variances)
        print(f"  G={G:3d} -> 优势方差平均={avg_var:.4f}")

    # 场景 4: KL 惩罚系数的影响
    print("\n--- 场景 4: beta (KL 系数) 的影响 ---")
    for b in [0.001, 0.01, 0.05, 0.1, 0.5]:
        loss, m = grpo_loss(
            data["policy_logps"],
            data["old_policy_logps"],
            data["ref_logps"],
            data["rewards"],
            data["response_lengths"],
            beta=b,
        )
        print(f"  beta={b:.3f} -> 损失={loss.item():.4f}, "
              f"KL惩罚={m['kl_penalty'].item():.4f}")

    # 场景 5: 使用模块封装
    print("\n--- 场景 5: GRPOLoss 模块 ---")
    criterion = GRPOLoss(clip_epsilon=0.2, beta=0.01)
    loss_mod, _ = criterion(
        data["policy_logps"],
        data["old_policy_logps"],
        data["ref_logps"],
        data["rewards"],
        data["response_lengths"],
    )
    print(f"模块化损失: {loss_mod.item():.4f}")

    print("\n演示完成。")
