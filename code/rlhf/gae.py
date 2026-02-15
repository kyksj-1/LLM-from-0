"""
GAE（Generalized Advantage Estimation）优势估计模块

实现广义优势估计算法，这是 PPO 训练中计算优势函数的核心组件。

核心数学原理:
    定义 TD 残差（Temporal Difference Residual）:
        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

    GAE 定义为 TD 残差的指数加权和:
        A_t^GAE = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

    展开递推公式:
        A_t^GAE = delta_t + gamma * lambda * A_{t+1}^GAE

    GAE 统一了两个极端:
    - lambda = 0: A_t = delta_t = r_t + gamma * V(s_{t+1}) - V(s_t) (TD(0), 低方差高偏差)
    - lambda = 1: A_t = sum_{l=0}^{T-t} gamma^l * r_{t+l} - V(s_t) (MC, 高方差低偏差)

    在 RLHF 场景中:
    - s_t 是文本生成到第 t 个 token 时的状态
    - a_t 是第 t 个生成的 token
    - r_t 通常只在序列末尾有非零值（来自奖励模型）
    - V(s_t) 是 critic 模型（价值网络）的输出
"""

import torch
from typing import Tuple, Optional


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 GAE（广义优势估计）。

    算法流程（从后往前递推）:
        1. 计算 TD 残差: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        2. 递推: A_t = delta_t + gamma * lambda * A_{t+1}
        3. 回报: R_t = A_t + V(s_t)

    在 RLHF 中的典型设置:
        - gamma = 1.0 (不衰减，因为 episode 较短)
        - lambda = 0.95 (平衡偏差和方差)
        - 奖励在序列末尾 (来自奖励模型减去 KL 惩罚)

    Args:
        rewards: 即时奖励, [batch_size, seq_len]
            在 RLHF 中，这包含 KL 惩罚和最终奖励:
            r_t = -beta * kl_t (中间 token)
            r_T = R_RM - beta * kl_T (最后一个 token)
        values: critic 预测的状态价值, [batch_size, seq_len]
        gamma: 折扣因子, 默认 1.0
        lam: GAE 的 lambda 参数, 控制偏差-方差权衡
        mask: 有效 token 掩码, [batch_size, seq_len]
            1 = 有效 token (response 部分)
            0 = 无效 token (prompt 部分或 padding)

    Returns:
        advantages: GAE 优势估计, [batch_size, seq_len]
        returns: 回报 (优势 + 价值), [batch_size, seq_len]
    """
    batch_size, seq_len = rewards.shape
    device = rewards.device

    # 初始化
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(batch_size, device=device)

    # 从后往前递推计算 GAE
    # A_t = delta_t + gamma * lambda * A_{t+1}
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            # 最后一个时间步：没有下一个状态
            # delta_T = r_T + gamma * 0 - V(s_T) = r_T - V(s_T)
            next_value = torch.zeros(batch_size, device=device)
        else:
            next_value = values[:, t + 1]

        # TD 残差
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_value - values[:, t]

        # 如果有 mask，当前时间步无效则 delta 和 last_gae 置零
        if mask is not None:
            delta = delta * mask[:, t]
            # 如果下一个时间步无效（mask=0），也要截断 GAE 传播
            if t < seq_len - 1:
                next_mask = mask[:, t + 1]
            else:
                next_mask = torch.zeros(batch_size, device=device)
            last_gae = last_gae * next_mask

        # GAE 递推: A_t = delta_t + gamma * lambda * A_{t+1}
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae

    # 回报 = 优势 + 价值
    # R_t = A_t + V(s_t)
    # 这样定义是因为: A_t = R_t - V(s_t)
    returns = advantages + values

    return advantages, returns


def compute_td_lambda_return(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算 TD(lambda) 回报。

    与 GAE 的关系:
        R_t^{TD(lambda)} = A_t^{GAE} + V(s_t)

    也可以直接通过递推定义:
        R_t^{TD(lambda)} = (1-lambda) * sum_{n=1}^{T-t} lambda^{n-1} * R_t^{(n)}

    其中 R_t^{(n)} 是 n 步回报:
        R_t^{(n)} = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * V(s_{t+n})

    Args:
        rewards: 即时奖励, [batch_size, seq_len]
        values: critic 预测的价值, [batch_size, seq_len]
        gamma: 折扣因子
        lam: lambda 参数
        mask: 有效 token 掩码

    Returns:
        returns: TD(lambda) 回报, [batch_size, seq_len]
    """
    advantages, returns = compute_gae(rewards, values, gamma, lam, mask)
    return returns


def analyze_bias_variance(
    rewards: torch.Tensor,
    values: torch.Tensor,
    lambdas: list = None,
    gamma: float = 1.0,
    mask: Optional[torch.Tensor] = None
) -> dict:
    """
    分析不同 lambda 值对 GAE 的偏差-方差影响。

    直觉:
    - lambda 越小 -> GAE 越依赖 critic (V)，偏差高（如果 V 不准）但方差低
    - lambda 越大 -> GAE 越依赖实际回报（MC），偏差低但方差高
    - 最优的 lambda 在两者之间取得平衡

    这是一个重要的超参数选择问题。本函数通过对比不同 lambda 下
    优势估计的统计特性来帮助理解这个权衡。

    Args:
        rewards: 奖励序列
        values: 价值估计
        lambdas: 要分析的 lambda 值列表
        gamma: 折扣因子
        mask: 有效 token 掩码

    Returns:
        分析结果字典
    """
    if lambdas is None:
        lambdas = [0.0, 0.5, 0.9, 0.95, 0.99, 1.0]

    results = {}
    for lam in lambdas:
        advantages, returns = compute_gae(rewards, values, gamma, lam, mask)
        if mask is not None:
            valid_adv = advantages[mask.bool()]
        else:
            valid_adv = advantages.flatten()

        results[f"lambda={lam}"] = {
            "mean": valid_adv.mean().item(),
            "std": valid_adv.std().item(),
            "min": valid_adv.min().item(),
            "max": valid_adv.max().item(),
            "abs_mean": valid_adv.abs().mean().item(),
        }

    return results


if __name__ == "__main__":
    print("=" * 60)
    print("GAE (Generalized Advantage Estimation) 演示")
    print("=" * 60)

    # 1. 基本 GAE 计算
    print("\n--- 1. 基本 GAE 计算 ---")
    batch_size, seq_len = 2, 10

    # 模拟 RLHF 场景：中间 token 有 KL 惩罚，最后 token 有奖励模型分数
    rewards = torch.zeros(batch_size, seq_len)
    rewards[:, :-1] = -0.01  # KL 惩罚（中间 token）
    rewards[:, -1] = torch.tensor([1.5, -0.3])  # 奖励模型分数（最后 token）

    # Critic 的价值预测
    values = torch.randn(batch_size, seq_len) * 0.1

    advantages, returns = compute_gae(rewards, values, gamma=1.0, lam=0.95)

    print(f"奖励 (样本 0): {rewards[0].tolist()}")
    print(f"价值 (样本 0): [" + ", ".join(f"{v:.3f}" for v in values[0].tolist()) + "]")
    print(f"优势 (样本 0): [" + ", ".join(f"{v:.3f}" for v in advantages[0].tolist()) + "]")
    print(f"回报 (样本 0): [" + ", ".join(f"{v:.3f}" for v in returns[0].tolist()) + "]")

    # 2. 带 mask 的 GAE
    print("\n--- 2. 带 mask 的 GAE（区分 prompt 和 response）---")
    mask = torch.zeros(batch_size, seq_len)
    mask[:, 4:] = 1.0  # 前4个是 prompt，后6个是 response

    advantages_masked, returns_masked = compute_gae(
        rewards, values, gamma=1.0, lam=0.95, mask=mask
    )
    print(f"Mask:           {mask[0].tolist()}")
    print(f"优势 (masked): [" + ", ".join(f"{v:.3f}" for v in advantages_masked[0].tolist()) + "]")
    print("注意: prompt 部分 (mask=0) 的优势为 0")

    # 3. 偏差-方差分析
    print("\n--- 3. 不同 lambda 的偏差-方差分析 ---")
    # 更大的样本
    big_rewards = torch.zeros(32, 20)
    big_rewards[:, -1] = torch.randn(32) + 1.0  # 奖励模型分数
    big_rewards[:, :-1] = -0.02 * torch.rand(32, 19)  # KL 惩罚

    big_values = torch.randn(32, 20) * 0.5

    analysis = analyze_bias_variance(big_rewards, big_values, gamma=1.0)
    print(f"{'Lambda':<12} {'Mean':>10} {'Std':>10} {'|Mean|':>10}")
    print("-" * 45)
    for key, stats in analysis.items():
        print(f"{key:<12} {stats['mean']:>10.4f} {stats['std']:>10.4f} {stats['abs_mean']:>10.4f}")

    print("\n解读:")
    print("  lambda=0 (TD(0)): 标准差最小（低方差），但可能有偏差")
    print("  lambda=1 (MC):    标准差最大（高方差），但无偏")
    print("  lambda=0.95:      在两者之间取得平衡（RLHF 常用值）")

    # 4. 极端情况测试
    print("\n--- 4. 边界情况测试 ---")
    # 全零奖励
    zero_rewards = torch.zeros(2, 5)
    zero_values = torch.ones(2, 5) * 0.5
    adv, ret = compute_gae(zero_rewards, zero_values, gamma=1.0, lam=0.95)
    print(f"全零奖励时的优势: [" + ", ".join(f"{v:.4f}" for v in adv[0].tolist()) + "]")
    print(f"理论上应接近 -V(s_t) 加上前向传播的衰减")

    # 单步 episode
    single_reward = torch.tensor([[2.0]])
    single_value = torch.tensor([[0.5]])
    adv_single, ret_single = compute_gae(single_reward, single_value)
    print(f"\n单步 episode:")
    print(f"  reward=2.0, value=0.5")
    print(f"  优势 = r - V = {adv_single.item():.4f} (应为 1.5)")
    print(f"  回报 = A + V = {ret_single.item():.4f} (应为 2.0)")

    print("\n" + "=" * 60)
    print("GAE 演示完成!")
    print("=" * 60)
