"""
ORPO (Odds Ratio Preference Optimization) 损失函数实现

ORPO 的核心创新：将 SFT 和偏好优化统一到一个训练阶段，
无需参考模型，使用优势比（Odds Ratio）来衡量偏好。

数学基础:
- 赔率(Odds): odds(y|x) = P(y|x) / (1 - P(y|x))
- 优势比(OR): OR(y_w, y_l) = odds(y_w) / odds(y_l)
- ORPO 损失: L = L_NLL + lambda * L_OR
  其中 L_NLL 是语言建模损失，L_OR 是优势比偏好损失

参考:
- Hong et al. (2024). ORPO: Monolithic Preference Optimization without Reference Model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math


def compute_odds(log_probs: torch.Tensor, seq_lengths: torch.Tensor) -> torch.Tensor:
    """
    计算回答的赔率（Odds）

    赔率定义为: odds(y|x) = P(y|x) / (1 - P(y|x))
    其中 P(y|x) 是回答的平均 token 概率的指数。

    在实践中，直接计算 P(y|x) 可能导致数值下溢，
    因此我们使用对数概率进行计算。

    Args:
        log_probs: 整个回答的对数概率（已求和），形状 [batch_size]
        seq_lengths: 每个回答的长度，形状 [batch_size]

    Returns:
        每个样本的对数赔率 log_odds，形状 [batch_size]
    """
    # 平均 token 对数概率
    avg_log_prob = log_probs / seq_lengths.float()

    # P(y|x) = exp(avg_log_prob)
    # odds(y|x) = P / (1 - P) = exp(avg_log_prob) / (1 - exp(avg_log_prob))
    # log_odds = avg_log_prob - log(1 - exp(avg_log_prob))

    # 使用 log1p(-exp(x)) 来数值稳定地计算 log(1 - exp(x))
    # 当 avg_log_prob 很小（接近 0，即 P 接近 1）时需要特别小心
    log_one_minus_p = torch.log1p(-torch.exp(avg_log_prob.clamp(max=-1e-7)))
    log_odds = avg_log_prob - log_one_minus_p

    return log_odds


def orpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    policy_chosen_nll: torch.Tensor,
    lambda_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    ORPO 损失函数

    ORPO 将 SFT 和偏好优化合二为一:
    L_ORPO = L_NLL + lambda * L_OR

    其中:
    - L_NLL: 偏好回答上的标准语言建模损失（等价于 SFT）
    - L_OR: 优势比损失，鼓励模型对偏好回答给出更高的赔率

    Args:
        policy_chosen_logps: 策略在偏好回答上的对数概率之和
                             形状 [batch_size]
        policy_rejected_logps: 策略在非偏好回答上的对数概率之和
                               形状 [batch_size]
        chosen_lengths: 偏好回答的长度，形状 [batch_size]
        rejected_lengths: 非偏好回答的长度，形状 [batch_size]
        policy_chosen_nll: 偏好回答上的 NLL 损失（已计算好）
                          形状 [batch_size] 或标量
        lambda_weight: 优势比损失的权重

    Returns:
        (loss, metrics): 总损失和监控指标
    """
    # 步骤 1: 计算赔率的对数
    chosen_log_odds = compute_odds(policy_chosen_logps, chosen_lengths)
    rejected_log_odds = compute_odds(policy_rejected_logps, rejected_lengths)

    # 步骤 2: 计算对数优势比
    # log OR = log(odds_w / odds_l) = log_odds_w - log_odds_l
    log_odds_ratio = chosen_log_odds - rejected_log_odds

    # 步骤 3: 优势比损失
    # L_OR = -log sigmoid(log OR)
    # 这鼓励优势比 > 1（即偏好回答的赔率更高）
    or_loss = -F.logsigmoid(log_odds_ratio).mean()

    # 步骤 4: 语言建模损失（SFT 部分）
    nll_loss = policy_chosen_nll.mean()

    # 步骤 5: 总损失
    total_loss = nll_loss + lambda_weight * or_loss

    # 监控指标
    with torch.no_grad():
        or_accuracy = (log_odds_ratio > 0).float().mean()
        metrics = {
            "loss": total_loss.detach(),
            "nll_loss": nll_loss.detach(),
            "or_loss": or_loss.detach(),
            "log_odds_ratio": log_odds_ratio.mean().detach(),
            "or_accuracy": or_accuracy.detach(),
            "chosen_log_odds": chosen_log_odds.mean().detach(),
            "rejected_log_odds": rejected_log_odds.mean().detach(),
        }

    return total_loss, metrics


class ORPOLoss(nn.Module):
    """
    ORPO 损失函数的 nn.Module 封装

    使用示例:
        criterion = ORPOLoss(lambda_weight=1.0)
        loss, metrics = criterion(chosen_logps, rejected_logps,
                                  chosen_lens, rejected_lens, nll)
    """

    def __init__(self, lambda_weight: float = 1.0):
        """
        Args:
            lambda_weight: 优势比损失的权重，控制偏好信号的强度
        """
        super().__init__()
        self.lambda_weight = lambda_weight

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        chosen_lengths: torch.Tensor,
        rejected_lengths: torch.Tensor,
        policy_chosen_nll: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播"""
        return orpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            chosen_lengths,
            rejected_lengths,
            policy_chosen_nll,
            lambda_weight=self.lambda_weight,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("ORPO (Odds Ratio Preference Optimization) 损失函数演示")
    print("=" * 60)

    torch.manual_seed(42)
    batch_size = 8

    # 模拟数据
    # 偏好回答通常有更高的对数概率
    chosen_logps = -torch.rand(batch_size) * 50 - 10    # 对数概率（负值）
    rejected_logps = -torch.rand(batch_size) * 50 - 30  # 对数概率更低

    chosen_lengths = torch.randint(10, 50, (batch_size,))
    rejected_lengths = torch.randint(10, 50, (batch_size,))

    # NLL 损失
    nll = -chosen_logps / chosen_lengths.float()  # 平均 token NLL

    # 场景 1: 标准 ORPO
    print("\n--- 场景 1: 标准 ORPO ---")
    loss, metrics = orpo_loss(
        chosen_logps, rejected_logps,
        chosen_lengths, rejected_lengths,
        nll, lambda_weight=1.0
    )
    print(f"总损失: {loss.item():.4f}")
    print(f"NLL 损失: {metrics['nll_loss'].item():.4f}")
    print(f"OR 损失: {metrics['or_loss'].item():.4f}")
    print(f"对数优势比: {metrics['log_odds_ratio'].item():.4f}")
    print(f"OR 准确率: {metrics['or_accuracy'].item():.4f}")

    # 场景 2: 不同 lambda 的影响
    print("\n--- 场景 2: lambda 对损失的影响 ---")
    for lam in [0.1, 0.5, 1.0, 2.0, 5.0]:
        loss, _ = orpo_loss(
            chosen_logps, rejected_logps,
            chosen_lengths, rejected_lengths,
            nll, lambda_weight=lam
        )
        print(f"  lambda={lam:.1f} -> 总损失={loss.item():.4f}")

    # 场景 3: 模型已学好（偏好回答概率远高于非偏好）
    print("\n--- 场景 3: 策略已学好 ---")
    good_chosen = -torch.rand(batch_size) * 10 - 5     # 高概率
    bad_rejected = -torch.rand(batch_size) * 100 - 50  # 很低概率

    loss_good, metrics_good = orpo_loss(
        good_chosen, bad_rejected,
        chosen_lengths, rejected_lengths,
        nll, lambda_weight=1.0
    )
    print(f"总损失: {loss_good.item():.4f}")
    print(f"OR 准确率: {metrics_good['or_accuracy'].item():.4f}")

    # 场景 4: 使用 nn.Module
    print("\n--- 场景 4: ORPOLoss 模块 ---")
    criterion = ORPOLoss(lambda_weight=1.0)
    loss_mod, _ = criterion(
        chosen_logps, rejected_logps,
        chosen_lengths, rejected_lengths, nll
    )
    print(f"模块化损失: {loss_mod.item():.4f}")

    print("\n演示完成。")
