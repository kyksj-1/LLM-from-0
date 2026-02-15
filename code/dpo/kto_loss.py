"""
KTO (Kahneman-Tversky Optimization) 损失函数实现

KTO 的核心创新：不需要成对偏好数据，只需要二元反馈（好/坏）。
基于行为经济学中的前景理论（Prospect Theory），利用人类对损失的
不对称感知来设计损失函数。

数学基础:
- 前景理论: 人对损失的敏感度 > 对等量收益的敏感度 (损失厌恶)
- KTO 好回答: v(x, y_w) = sigma(beta * log(pi/pi_ref) - z_ref)
- KTO 坏回答: v(x, y_l) = sigma(z_ref - beta * log(pi/pi_ref))
- 损失: L = E[lambda_y * (1 - v(x, y))]

参考:
- Ethayarajh et al. (2024). KTO: Model Alignment as Prospect Theoretic Optimization.
- Kahneman & Tversky (1979). Prospect Theory: An Analysis of Decision under Risk.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def kto_loss(
    policy_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    is_desirable: torch.Tensor,
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    KTO 损失函数

    与 DPO 不同，KTO 不需要成对的 (y_w, y_l) 数据。
    每个样本只需要 (x, y, label)，其中 label 表示 y 是好是坏。

    数学推导:
    1. 定义隐式奖励: r_hat(x,y) = beta * log(pi_theta(y|x) / pi_ref(y|x))
    2. 计算参考点: z_ref = E[KL(pi_theta || pi_ref)]（动态基线）
    3. 好回答的价值: v_w = sigma(r_hat - z_ref)
       - 当隐式奖励超过基线时，价值趋近 1
    4. 坏回答的价值: v_l = sigma(z_ref - r_hat)
       - 当隐式奖励低于基线时，价值趋近 1
    5. 损失: L = E_w[lambda_w * (1 - v_w)] + E_l[lambda_l * (1 - v_l)]
       - lambda_l > lambda_w 体现损失厌恶

    Args:
        policy_logps: 策略模型的对数概率 log pi_theta(y|x)
                      形状 [batch_size]
        reference_logps: 参考模型的对数概率 log pi_ref(y|x)
                         形状 [batch_size]
        is_desirable: 布尔张量，True 表示好回答，False 表示坏回答
                      形状 [batch_size]
        beta: KL 惩罚系数
        desirable_weight: 好回答的损失权重 (lambda_w)
        undesirable_weight: 坏回答的损失权重 (lambda_l)
                           通常 > desirable_weight，体现损失厌恶

    Returns:
        (loss, metrics): 损失标量和监控指标字典
    """
    # 步骤 1: 计算对数概率比（隐式奖励的核心部分）
    log_ratios = policy_logps - reference_logps

    # 步骤 2: 计算 KL 散度作为参考点
    # z_ref 是当前批次中所有样本的平均 KL 散度
    # 这个参考点帮助模型判断"好到什么程度算好"
    # KL(pi || pi_ref) approx = pi_ref/pi - log(pi_ref/pi) - 1
    # 简化近似: z_ref = mean(beta * |log_ratio|)
    z_ref = beta * log_ratios.detach().abs().mean()

    # 步骤 3: 分离好回答和坏回答
    desirable_mask = is_desirable.bool()
    undesirable_mask = ~desirable_mask

    # 步骤 4: 计算好回答的价值函数
    # v_w = sigma(beta * log_ratio - z_ref)
    # 好回答的隐式奖励应该超过参考点
    desirable_logits = beta * log_ratios - z_ref

    # 步骤 5: 计算坏回答的价值函数
    # v_l = sigma(z_ref - beta * log_ratio)
    # 坏回答的隐式奖励应该低于参考点
    undesirable_logits = z_ref - beta * log_ratios

    # 步骤 6: 计算损失
    # L_w = lambda_w * (1 - sigma(desirable_logits))
    # L_l = lambda_l * (1 - sigma(undesirable_logits))
    # 使用 -log_sigmoid 作为 (1 - sigma) 的替代，更数值稳定
    # 注意: 1 - sigma(z) = sigma(-z)

    # 初始化损失为 0
    total_loss = torch.tensor(0.0, device=policy_logps.device)
    num_desirable = desirable_mask.sum().item()
    num_undesirable = undesirable_mask.sum().item()

    # 好回答的损失
    if num_desirable > 0:
        desirable_loss = (
            desirable_weight
            * (1.0 - torch.sigmoid(desirable_logits[desirable_mask])).mean()
        )
        total_loss = total_loss + desirable_loss

    # 坏回答的损失
    if num_undesirable > 0:
        undesirable_loss = (
            undesirable_weight
            * (1.0 - torch.sigmoid(undesirable_logits[undesirable_mask])).mean()
        )
        total_loss = total_loss + undesirable_loss

    # 监控指标
    with torch.no_grad():
        implicit_rewards = beta * log_ratios
        metrics = {
            "loss": total_loss.detach(),
            "z_ref": z_ref.detach(),
            "num_desirable": torch.tensor(float(num_desirable)),
            "num_undesirable": torch.tensor(float(num_undesirable)),
            "mean_implicit_reward": implicit_rewards.mean().detach(),
        }
        if num_desirable > 0:
            metrics["desirable_rewards"] = implicit_rewards[desirable_mask].mean()
        if num_undesirable > 0:
            metrics["undesirable_rewards"] = implicit_rewards[undesirable_mask].mean()

    return total_loss, metrics


class KTOLoss(nn.Module):
    """
    KTO 损失函数的 nn.Module 封装

    使用示例:
        criterion = KTOLoss(beta=0.1, undesirable_weight=1.5)
        loss, metrics = criterion(policy_logps, ref_logps, labels)
    """

    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
    ):
        """
        初始化 KTO 损失模块

        Args:
            beta: KL 惩罚系数
            desirable_weight: 好回答的损失权重
            undesirable_weight: 坏回答的损失权重
                               设置 > desirable_weight 体现损失厌恶
        """
        super().__init__()
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_logps: torch.Tensor,
        reference_logps: torch.Tensor,
        is_desirable: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播"""
        return kto_loss(
            policy_logps,
            reference_logps,
            is_desirable,
            beta=self.beta,
            desirable_weight=self.desirable_weight,
            undesirable_weight=self.undesirable_weight,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("KTO (Kahneman-Tversky Optimization) 损失函数演示")
    print("=" * 60)

    torch.manual_seed(42)
    batch_size = 16

    # 构造模拟数据
    # 一半是好回答（is_desirable=True），一半是坏回答（is_desirable=False）
    policy_logps = torch.randn(batch_size)
    reference_logps = torch.randn(batch_size)
    is_desirable = torch.tensor(
        [True] * (batch_size // 2) + [False] * (batch_size // 2)
    )

    # 场景 1: 标准 KTO（对称权重）
    print("\n--- 场景 1: 标准 KTO（对称权重）---")
    loss, metrics = kto_loss(
        policy_logps, reference_logps, is_desirable,
        beta=0.1, desirable_weight=1.0, undesirable_weight=1.0
    )
    print(f"损失: {loss.item():.4f}")
    print(f"参考点 z_ref: {metrics['z_ref'].item():.4f}")
    print(f"好回答平均隐式奖励: {metrics['desirable_rewards'].item():.4f}")
    print(f"坏回答平均隐式奖励: {metrics['undesirable_rewards'].item():.4f}")

    # 场景 2: 带损失厌恶的 KTO（非对称权重）
    print("\n--- 场景 2: 损失厌恶 KTO（lambda_l=1.5）---")
    loss_la, metrics_la = kto_loss(
        policy_logps, reference_logps, is_desirable,
        beta=0.1, desirable_weight=1.0, undesirable_weight=1.5
    )
    print(f"损失: {loss_la.item():.4f}")
    print(f"损失增加（因为对坏回答惩罚更重）: {loss_la.item() - loss.item():.4f}")

    # 场景 3: 策略已经学好（好回答概率高，坏回答概率低）
    print("\n--- 场景 3: 策略已学好 ---")
    good_policy = torch.randn(batch_size)
    good_policy[:batch_size // 2] += 3.0   # 好回答概率大幅高于参考
    good_policy[batch_size // 2:] -= 3.0   # 坏回答概率大幅低于参考
    good_ref = torch.zeros(batch_size)

    loss_good, metrics_good = kto_loss(
        good_policy, good_ref, is_desirable, beta=0.1
    )
    print(f"损失: {loss_good.item():.4f} (应该较小)")
    print(f"好回答隐式奖励: {metrics_good['desirable_rewards'].item():.4f}")
    print(f"坏回答隐式奖励: {metrics_good['undesirable_rewards'].item():.4f}")

    # 场景 4: 使用 nn.Module 封装
    print("\n--- 场景 4: KTOLoss 模块 ---")
    criterion = KTOLoss(beta=0.1, undesirable_weight=1.5)
    loss_mod, _ = criterion(policy_logps, reference_logps, is_desirable)
    print(f"模块化损失: {loss_mod.item():.4f}")

    print("\n演示完成。")
