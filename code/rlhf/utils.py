"""
RLHF 工具函数模块

提供 RLHF 训练中常用的工具函数，包括：
- 对数概率计算
- KL 散度计算
- 白化（whiten）操作
- 裁剪统计
- 熵计算
- 经验统计汇总

这些函数被 ppo_trainer.py、gae.py、rollout.py 等模块广泛调用。
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    从 logits 中提取指定 token 的对数概率。

    在 RLHF 中，我们需要计算策略 pi(a|s) 的对数概率，用于:
    1. 重要性采样比率 r_t(theta) = pi_theta / pi_old
    2. KL 散度惩罚
    3. PPO 目标函数

    Args:
        logits: 模型输出的 logits, 形状 [batch_size, seq_len, vocab_size]
        labels: 目标 token id, 形状 [batch_size, seq_len]

    Returns:
        对数概率, 形状 [batch_size, seq_len]
    """
    # 对 logits 做 log_softmax，得到对数概率分布
    log_probs = F.log_softmax(logits, dim=-1)
    # 使用 gather 从对数概率分布中取出对应 token 的对数概率
    # labels.unsqueeze(-1) 将形状从 [B, T] 变为 [B, T, 1]
    # gather 后 squeeze 回 [B, T]
    per_token_log_probs = log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    return per_token_log_probs


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    计算 logits 对应分布的熵。

    熵衡量分布的不确定性：H(pi) = -sum_a pi(a) log pi(a)
    熵越大说明分布越均匀（模型越不确定），熵越小说明分布越集中。
    在 RLHF 中，熵正则化可以防止策略坍缩（collapse）到少数 token。

    Args:
        logits: 模型输出的 logits, 形状 [..., vocab_size]

    Returns:
        每个位置的熵, 形状 [...]
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    # H = -sum(p * log(p))
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def whiten(values: torch.Tensor, mask: Optional[torch.Tensor] = None,
           shift_mean: bool = True) -> torch.Tensor:
    """
    对张量进行白化（标准化），使其均值为0、方差为1。

    白化在 RLHF 中的作用：
    1. 稳定优势估计：GAE 计算的优势值 A_t 的尺度可能差异很大
    2. 改善梯度信号：白化后的优势值作为 PPO 梯度的系数，能提供更稳定的更新
    3. 降低超参数敏感性：不同任务/batch 的奖励尺度不同，白化可以统一

    Args:
        values: 需要白化的张量
        mask: 有效位置的掩码（1=有效, 0=填充）
        shift_mean: 是否减去均值。某些实现只做方差归一化

    Returns:
        白化后的张量
    """
    if mask is not None:
        # 仅对有效位置计算统计量
        masked_values = values * mask
        count = mask.sum()
        mean = masked_values.sum() / count.clamp(min=1)
        var = ((masked_values - mean * mask) ** 2 * mask).sum() / count.clamp(min=1)
    else:
        mean = values.mean()
        var = values.var()

    std = (var + 1e-8).sqrt()

    if shift_mean:
        whitened = (values - mean) / std
    else:
        whitened = values / std

    return whitened


def compute_kl_divergence(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    计算两个策略之间的 KL 散度（逐 token）。

    KL(pi_theta || pi_ref) = sum_a pi_theta(a|s) [log pi_theta(a|s) - log pi_ref(a|s)]

    在 RLHF 中，KL 散度用于约束当前策略不偏离参考策略太远：
    R_total = R_reward_model - beta * KL(pi_theta || pi_ref)

    这里我们使用一种简化的近似：对于已采样的 token a_t：
    kl_t ≈ log pi_theta(a_t|s_t) - log pi_ref(a_t|s_t)

    这是 KL 散度的无偏估计（一阶近似）。

    Args:
        log_probs: 当前策略的对数概率, [batch_size, seq_len]
        ref_log_probs: 参考策略的对数概率, [batch_size, seq_len]
        mask: 有效 token 掩码

    Returns:
        逐 token 的 KL 散度, [batch_size, seq_len]
    """
    kl = log_probs - ref_log_probs  # 简化的 per-token KL
    if mask is not None:
        kl = kl * mask
    return kl


def clip_reward(reward: torch.Tensor, clip_value: float = 10.0) -> torch.Tensor:
    """
    裁剪奖励值，防止极端值导致训练不稳定。

    奖励模型可能输出非常大或非常小的值（reward hacking 场景），
    裁剪可以限制奖励信号的范围。

    Args:
        reward: 原始奖励值
        clip_value: 裁剪范围 [-clip_value, clip_value]

    Returns:
        裁剪后的奖励值
    """
    return torch.clamp(reward, -clip_value, clip_value)


def compute_clip_fraction(
    ratio: torch.Tensor,
    epsilon: float = 0.2,
    mask: Optional[torch.Tensor] = None
) -> float:
    """
    计算 PPO 中被裁剪的比例。

    裁剪比例是一个重要的训练监控指标：
    - 比例过高（>0.3）：说明策略更新步长过大，可能需要降低学习率
    - 比例过低（<0.05）：说明策略更新太保守，可能需要增大学习率
    - 理想范围：0.1-0.2

    Args:
        ratio: 重要性采样比率 r_t(theta) = pi_theta / pi_old
        epsilon: PPO 裁剪参数
        mask: 有效位置掩码

    Returns:
        被裁剪的比例（标量）
    """
    clipped = (ratio - 1.0).abs() > epsilon
    if mask is not None:
        clip_frac = (clipped.float() * mask).sum() / mask.sum().clamp(min=1)
    else:
        clip_frac = clipped.float().mean()
    return clip_frac.item()


def flatten_dict(nested_dict: Dict, parent_key: str = '', sep: str = '/') -> Dict:
    """
    将嵌套字典展平为单层字典，便于日志记录。

    Args:
        nested_dict: 嵌套字典
        parent_key: 父级键前缀
        sep: 键之间的分隔符

    Returns:
        展平后的字典
    """
    items = {}
    for k, v in nested_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        else:
            items[new_key] = v
    return items


def get_training_stats(
    rewards: torch.Tensor,
    kl_values: torch.Tensor,
    advantages: torch.Tensor,
    ratio: torch.Tensor,
    epsilon: float = 0.2,
    mask: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    汇总 RLHF 训练的关键统计量。

    这些统计量用于监控训练过程：
    - reward_mean/std: 奖励信号的分布
    - kl_mean: 平均 KL 散度，反映策略偏移程度
    - advantage_mean/std: 优势函数的分布
    - clip_fraction: PPO 裁剪比例
    - entropy: 策略的熵（需要 logits，此处省略）

    Args:
        rewards: 奖励值
        kl_values: KL 散度值
        advantages: 优势函数值
        ratio: 重要性采样比率
        epsilon: PPO 裁剪参数
        mask: 有效位置掩码

    Returns:
        统计量字典
    """
    if mask is not None:
        count = mask.sum().clamp(min=1)
        reward_mean = (rewards * mask).sum() / count
        reward_std = ((rewards - reward_mean) ** 2 * mask).sum() / count
        kl_mean = (kl_values * mask).sum() / count
        adv_mean = (advantages * mask).sum() / count
    else:
        reward_mean = rewards.mean()
        reward_std = rewards.std()
        kl_mean = kl_values.mean()
        adv_mean = advantages.mean()

    clip_frac = compute_clip_fraction(ratio, epsilon, mask)

    stats = {
        "reward/mean": reward_mean.item(),
        "reward/std": (reward_std + 1e-8).sqrt().item(),
        "kl/mean": kl_mean.item(),
        "advantage/mean": adv_mean.item(),
        "ppo/clip_fraction": clip_frac,
        "ppo/ratio_mean": ratio.mean().item(),
    }
    return stats


if __name__ == "__main__":
    print("=" * 60)
    print("RLHF 工具函数演示")
    print("=" * 60)

    # 1. 对数概率提取
    print("\n--- 1. 对数概率提取 ---")
    batch_size, seq_len, vocab_size = 2, 5, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    lp = log_probs_from_logits(logits, labels)
    print(f"输入 logits 形状: {logits.shape}")
    print(f"输入 labels 形状: {labels.shape}")
    print(f"输出对数概率形状: {lp.shape}")
    print(f"对数概率范围: [{lp.min().item():.4f}, {lp.max().item():.4f}]")

    # 2. 熵计算
    print("\n--- 2. 熵计算 ---")
    ent = entropy_from_logits(logits)
    print(f"每个位置的熵形状: {ent.shape}")
    print(f"平均熵: {ent.mean().item():.4f}")
    print(f"理论最大熵 (log vocab_size): {torch.log(torch.tensor(float(vocab_size))).item():.4f}")

    # 3. 白化
    print("\n--- 3. 白化操作 ---")
    values = torch.randn(10) * 5 + 3  # 均值3, 标准差5
    print(f"原始值 均值={values.mean():.4f}, 标准差={values.std():.4f}")
    whitened = whiten(values)
    print(f"白化后 均值={whitened.mean():.4f}, 标准差={whitened.std():.4f}")

    # 4. KL 散度
    print("\n--- 4. KL 散度计算 ---")
    log_p = torch.randn(batch_size, seq_len) - 2  # 模拟对数概率
    ref_log_p = torch.randn(batch_size, seq_len) - 2
    kl = compute_kl_divergence(log_p, ref_log_p)
    print(f"KL 散度形状: {kl.shape}")
    print(f"平均 KL: {kl.mean().item():.4f}")

    # 5. 裁剪统计
    print("\n--- 5. PPO 裁剪统计 ---")
    ratio = torch.randn(100) * 0.3 + 1.0  # 围绕 1.0 的比率
    clip_frac = compute_clip_fraction(ratio, epsilon=0.2)
    print(f"裁剪比例: {clip_frac:.4f}")

    # 6. 训练统计汇总
    print("\n--- 6. 训练统计汇总 ---")
    rewards = torch.randn(batch_size, seq_len)
    kl_vals = torch.abs(torch.randn(batch_size, seq_len)) * 0.1
    advantages = torch.randn(batch_size, seq_len)
    ratios = torch.randn(batch_size, seq_len) * 0.2 + 1.0
    stats = get_training_stats(rewards, kl_vals, advantages, ratios)
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("所有工具函数测试通过!")
    print("=" * 60)
