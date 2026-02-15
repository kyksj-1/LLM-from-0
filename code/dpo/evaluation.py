"""
对齐评估工具

本模块实现了偏好优化后的模型评估工具，包括：
- 偏好胜率（Win Rate）计算
- 隐式奖励分布分析
- KL 散度监控
- 回答质量多维度评估

评估是偏好优化的关键环节：仅看训练损失下降不够，
需要从多个角度验证模型是否真正"对齐"了人类偏好。

参考:
- Rafailov et al. (2023). Direct Preference Optimization.
- Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
import logging
import math

logger = logging.getLogger(__name__)


def compute_win_rate(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Dict[str, float]:
    """
    计算偏好胜率（Win Rate）

    胜率 = 模型认为偏好回答比非偏好回答更好的比例
    即隐式奖励差 > 0 的样本占总样本的比例。

    这是评估偏好优化效果最直接的指标。

    数学公式:
        win_rate = E[1(r_hat(y_w) > r_hat(y_l))]
    其中:
        r_hat(y) = beta * log(pi_theta(y|x) / pi_ref(y|x))

    Args:
        policy_chosen_logps: 策略在偏好回答上的 log P，形状 [N]
        policy_rejected_logps: 策略在非偏好回答上的 log P，形状 [N]
        ref_chosen_logps: 参考模型在偏好回答上的 log P，形状 [N]
        ref_rejected_logps: 参考模型在非偏好回答上的 log P，形状 [N]
        beta: KL 惩罚系数

    Returns:
        评估指标字典
    """
    with torch.no_grad():
        # 计算隐式奖励
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

        # 奖励差
        reward_diff = chosen_rewards - rejected_rewards

        # 胜率
        win_rate = (reward_diff > 0).float().mean().item()

        # 奖励统计
        results = {
            "win_rate": win_rate,
            "mean_chosen_reward": chosen_rewards.mean().item(),
            "mean_rejected_reward": rejected_rewards.mean().item(),
            "mean_reward_margin": reward_diff.mean().item(),
            "std_reward_margin": reward_diff.std().item(),
        }

    return results


def compute_kl_divergence(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
) -> Dict[str, float]:
    """
    计算策略和参考模型之间的 KL 散度

    KL 散度衡量策略偏离参考模型的程度。
    过大的 KL 说明策略可能过度优化（reward hacking）；
    过小的 KL 说明偏好优化可能效果不足。

    数学公式:
        KL(pi || pi_ref) = E[log(pi(y|x)/pi_ref(y|x))]

    经验法则:
        KL < 1: 偏离较小，可能对齐效果不足
        KL in [1, 10]: 合理范围
        KL > 10: 偏离过大，可能过度优化

    Args:
        policy_logps: 策略模型的 log P，形状 [N]
        ref_logps: 参考模型的 log P，形状 [N]

    Returns:
        KL 散度相关指标
    """
    with torch.no_grad():
        log_ratio = policy_logps - ref_logps

        # KL(pi || pi_ref) = E[log(pi/pi_ref)]
        kl = log_ratio.mean().item()

        # 逆向 KL: KL(pi_ref || pi) = E[log(pi_ref/pi)]
        reverse_kl = -log_ratio.mean().item()

        # JS 散度: JS = 0.5 * (KL(pi||m) + KL(pi_ref||m)) where m = 0.5*(pi+pi_ref)
        # 近似: JS ~= 0.25 * Var(log_ratio)
        js_approx = 0.25 * log_ratio.var().item()

        results = {
            "kl_divergence": kl,
            "reverse_kl": reverse_kl,
            "js_divergence_approx": js_approx,
            "mean_log_ratio": log_ratio.mean().item(),
            "std_log_ratio": log_ratio.std().item(),
            "max_log_ratio": log_ratio.max().item(),
            "min_log_ratio": log_ratio.min().item(),
        }

    return results


def compute_reward_distribution(
    policy_logps: torch.Tensor,
    ref_logps: torch.Tensor,
    beta: float = 0.1,
    num_bins: int = 20,
) -> Dict[str, Any]:
    """
    分析隐式奖励的分布

    通过分析奖励分布可以了解:
    - 模型是否在所有样本上均匀提升
    - 是否存在奖励极端值（可能的过拟合信号）
    - 奖励分布是否合理（应近似正态分布）

    Args:
        policy_logps: 策略模型的 log P，形状 [N]
        ref_logps: 参考模型的 log P，形状 [N]
        beta: KL 惩罚系数
        num_bins: 直方图的桶数

    Returns:
        分布分析结果
    """
    with torch.no_grad():
        rewards = beta * (policy_logps - ref_logps)

        # 基本统计
        stats = {
            "mean": rewards.mean().item(),
            "std": rewards.std().item(),
            "median": rewards.median().item(),
            "min": rewards.min().item(),
            "max": rewards.max().item(),
            "skewness": _compute_skewness(rewards),
            "kurtosis": _compute_kurtosis(rewards),
        }

        # 直方图
        hist = torch.histc(rewards, bins=num_bins)
        stats["histogram"] = hist.tolist()
        stats["histogram_range"] = (rewards.min().item(), rewards.max().item())

        # 分位数
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        for q in quantiles:
            stats[f"quantile_{q}"] = torch.quantile(rewards.float(), q).item()

    return stats


def _compute_skewness(x: torch.Tensor) -> float:
    """计算偏度（Skewness）"""
    mean = x.mean()
    std = x.std()
    if std.item() < 1e-10:
        return 0.0
    return ((x - mean) ** 3).mean().item() / (std.item() ** 3)


def _compute_kurtosis(x: torch.Tensor) -> float:
    """计算峰度（Kurtosis）"""
    mean = x.mean()
    std = x.std()
    if std.item() < 1e-10:
        return 0.0
    return ((x - mean) ** 4).mean().item() / (std.item() ** 4) - 3.0


def evaluate_alignment(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Dict[str, Any]:
    """
    综合评估偏好对齐效果

    汇总多个维度的评估指标，提供全面的对齐质量报告。

    评估维度:
    1. 胜率: 模型是否能正确区分好坏回答
    2. KL 散度: 模型偏离参考模型的程度
    3. 奖励分布: 隐式奖励是否合理

    Args:
        与 compute_win_rate 相同

    Returns:
        综合评估结果
    """
    results = {}

    # 1. 胜率
    win_rate_metrics = compute_win_rate(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps, beta
    )
    results["win_rate"] = win_rate_metrics

    # 2. KL 散度
    # 对偏好回答和非偏好回答分别计算
    all_policy_logps = torch.cat([policy_chosen_logps, policy_rejected_logps])
    all_ref_logps = torch.cat([ref_chosen_logps, ref_rejected_logps])
    kl_metrics = compute_kl_divergence(all_policy_logps, all_ref_logps)
    results["kl_divergence"] = kl_metrics

    # 3. 偏好回答的奖励分布
    chosen_dist = compute_reward_distribution(
        policy_chosen_logps, ref_chosen_logps, beta
    )
    results["chosen_reward_distribution"] = chosen_dist

    # 4. 非偏好回答的奖励分布
    rejected_dist = compute_reward_distribution(
        policy_rejected_logps, ref_rejected_logps, beta
    )
    results["rejected_reward_distribution"] = rejected_dist

    # 5. 可分离性分析
    with torch.no_grad():
        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

        # Cohen's d: 两组奖励的标准化差异
        pooled_std = torch.sqrt(
            (chosen_rewards.var() + rejected_rewards.var()) / 2
        )
        if pooled_std > 1e-10:
            cohens_d = (chosen_rewards.mean() - rejected_rewards.mean()) / pooled_std
        else:
            cohens_d = torch.tensor(0.0)

        results["separability"] = {
            "cohens_d": cohens_d.item(),
            # Cohen's d 解读: 0.2=小, 0.5=中, 0.8=大
            "interpretation": (
                "excellent" if abs(cohens_d.item()) > 0.8
                else "good" if abs(cohens_d.item()) > 0.5
                else "moderate" if abs(cohens_d.item()) > 0.2
                else "poor"
            ),
        }

    return results


def format_evaluation_report(
    eval_results: Dict[str, Any],
    model_name: str = "Policy Model",
) -> str:
    """
    将评估结果格式化为可读的报告

    Args:
        eval_results: evaluate_alignment 的返回结果
        model_name: 模型名称

    Returns:
        格式化的评估报告字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"对齐评估报告: {model_name}")
    lines.append("=" * 60)

    # 胜率
    wr = eval_results["win_rate"]
    lines.append(f"\n--- 偏好胜率 ---")
    lines.append(f"  Win Rate: {wr['win_rate']:.4f} ({wr['win_rate']*100:.1f}%)")
    lines.append(f"  Mean Chosen Reward: {wr['mean_chosen_reward']:.4f}")
    lines.append(f"  Mean Rejected Reward: {wr['mean_rejected_reward']:.4f}")
    lines.append(f"  Reward Margin: {wr['mean_reward_margin']:.4f} "
                 f"(+/- {wr['std_reward_margin']:.4f})")

    # KL 散度
    kl = eval_results["kl_divergence"]
    lines.append(f"\n--- KL 散度 ---")
    lines.append(f"  KL(pi || pi_ref): {kl['kl_divergence']:.4f}")
    lines.append(f"  Reverse KL: {kl['reverse_kl']:.4f}")
    lines.append(f"  Log Ratio Range: [{kl['min_log_ratio']:.4f}, "
                 f"{kl['max_log_ratio']:.4f}]")

    # 可分离性
    sep = eval_results["separability"]
    lines.append(f"\n--- 可分离性 ---")
    lines.append(f"  Cohen's d: {sep['cohens_d']:.4f}")
    lines.append(f"  解读: {sep['interpretation']}")

    # 奖励分布
    cd = eval_results["chosen_reward_distribution"]
    rd = eval_results["rejected_reward_distribution"]
    lines.append(f"\n--- 奖励分布 ---")
    lines.append(f"  偏好回答: mean={cd['mean']:.4f}, std={cd['std']:.4f}, "
                 f"range=[{cd['min']:.4f}, {cd['max']:.4f}]")
    lines.append(f"  非偏好回答: mean={rd['mean']:.4f}, std={rd['std']:.4f}, "
                 f"range=[{rd['min']:.4f}, {rd['max']:.4f}]")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("对齐评估工具演示")
    print("=" * 60)

    torch.manual_seed(42)
    num_samples = 200
    beta = 0.1

    # 场景 1: 评估一个"好"的模型（偏好回答奖励明显更高）
    print("\n--- 场景 1: 评估对齐良好的模型 ---")
    policy_chosen = torch.randn(num_samples) + 2.0
    policy_rejected = torch.randn(num_samples) - 1.0
    ref_chosen = torch.randn(num_samples)
    ref_rejected = torch.randn(num_samples)

    results = evaluate_alignment(
        policy_chosen, policy_rejected,
        ref_chosen, ref_rejected, beta
    )
    report = format_evaluation_report(results, "对齐良好的模型")
    print(report)

    # 场景 2: 评估一个"差"的模型（无法区分好坏回答）
    print("\n--- 场景 2: 评估未对齐的模型 ---")
    policy_chosen_bad = torch.randn(num_samples)  # 和参考模型几乎一样
    policy_rejected_bad = torch.randn(num_samples)

    results_bad = evaluate_alignment(
        policy_chosen_bad, policy_rejected_bad,
        ref_chosen, ref_rejected, beta
    )
    report_bad = format_evaluation_report(results_bad, "未对齐的模型")
    print(report_bad)

    # 场景 3: 评估一个"过度优化"的模型
    print("\n--- 场景 3: 评估过度优化的模型 ---")
    policy_chosen_over = torch.randn(num_samples) + 10.0  # 极端偏离
    policy_rejected_over = torch.randn(num_samples) - 10.0

    results_over = evaluate_alignment(
        policy_chosen_over, policy_rejected_over,
        ref_chosen, ref_rejected, beta
    )
    report_over = format_evaluation_report(results_over, "过度优化的模型")
    print(report_over)

    # 场景 4: KL 散度分析
    print("\n--- 场景 4: KL 散度解读指南 ---")
    print("KL 散度范围     | 解读")
    print("-" * 50)
    print("KL < 1          | 偏离小，可能对齐效果不足")
    print("1 <= KL <= 10   | 合理范围")
    print("KL > 10         | 偏离大，可能过度优化")
    print("\n注意: 具体阈值需要根据任务和模型规模调整。")

    # 场景 5: 奖励分布分析
    print("\n--- 场景 5: 奖励分布分析 ---")
    dist = compute_reward_distribution(
        policy_chosen, ref_chosen, beta
    )
    print(f"偏好回答奖励分布:")
    print(f"  均值: {dist['mean']:.4f}")
    print(f"  标准差: {dist['std']:.4f}")
    print(f"  偏度: {dist['skewness']:.4f} "
          f"({'右偏' if dist['skewness'] > 0 else '左偏'})")
    print(f"  峰度: {dist['kurtosis']:.4f} "
          f"({'尖峰' if dist['kurtosis'] > 0 else '平峰'})")
    print(f"  10%分位: {dist['quantile_0.1']:.4f}")
    print(f"  50%分位(中位数): {dist['quantile_0.5']:.4f}")
    print(f"  90%分位: {dist['quantile_0.9']:.4f}")

    print("\n演示完成。")
