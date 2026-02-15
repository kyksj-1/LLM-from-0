"""
DPO 及变体的损失函数实现

本模块实现了 DPO（Direct Preference Optimization）及其核心变体 IPO 的损失函数。
包含完整的数学推导注释，帮助理解每个损失函数的设计动机。

数学基础:
- DPO 损失: L = -E[log sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x)))]
- IPO 损失: L = E[(log pi(y_w|x)/pi_ref(y_w|x) - log pi(y_l|x)/pi_ref(y_l|x) - 1/(2*beta))^2]

参考:
- Rafailov et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model.
- Azar et al. (2023). A General Theoretical Paradigm to Understand Learning from Human Feedback.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    根据模型输出的 logits 和真实标签计算对数概率

    这是所有偏好优化方法的基础函数。给定模型在每个位置的输出分布（logits）
    和真实的 token 序列（labels），计算整个序列的对数概率。

    数学公式:
        log P(y|x) = sum_{t=1}^{T} log P(y_t | x, y_{<t})

    Args:
        logits: 模型输出的 logits，形状 [batch_size, seq_len, vocab_size]
        labels: 真实标签，形状 [batch_size, seq_len]
        mask: 可选的掩码，形状 [batch_size, seq_len]，1 表示有效位置

    Returns:
        每个样本的对数概率，形状 [batch_size]
    """
    # 将 logits 转换为对数概率分布
    # log_softmax 比先 softmax 再 log 更数值稳定
    log_probs = F.log_softmax(logits, dim=-1)

    # 提取真实 token 对应的对数概率
    # gather 操作：从 log_probs 中按 labels 索引取值
    # labels.unsqueeze(-1) 将 [batch, seq] 变为 [batch, seq, 1]
    per_token_log_probs = log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)  # [batch_size, seq_len]

    if mask is not None:
        # 只计算有效位置的对数概率（忽略 padding）
        per_token_log_probs = per_token_log_probs * mask
        # 对每个样本求和
        return per_token_log_probs.sum(dim=-1)
    else:
        return per_token_log_probs.sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    DPO (Direct Preference Optimization) 损失函数

    完整推导过程:
    1. RLHF 目标: max E[r(x,y)] - beta * KL(pi || pi_ref)
    2. 最优策略: pi*(y|x) = pi_ref(y|x) * exp(r(x,y)/beta) / Z(x)
    3. 隐式奖励: r(x,y) = beta * log(pi(y|x)/pi_ref(y|x)) + beta * log Z(x)
    4. 奖励差: r(y_w) - r(y_l) = beta * [log(pi(y_w)/pi_ref(y_w)) - log(pi(y_l)/pi_ref(y_l))]
       (Z(x) 在做差时消去)
    5. 代入 Bradley-Terry 模型: P(y_w > y_l) = sigma(r(y_w) - r(y_l))
    6. 最终损失: L = -E[log sigma(beta * (log_ratio_w - log_ratio_l))]

    Args:
        policy_chosen_logps: 策略模型在偏好回答上的对数概率 log pi_theta(y_w|x)
                             形状 [batch_size]
        policy_rejected_logps: 策略模型在非偏好回答上的对数概率 log pi_theta(y_l|x)
                               形状 [batch_size]
        reference_chosen_logps: 参考模型在偏好回答上的对数概率 log pi_ref(y_w|x)
                                形状 [batch_size]
        reference_rejected_logps: 参考模型在非偏好回答上的对数概率 log pi_ref(y_l|x)
                                  形状 [batch_size]
        beta: KL 惩罚系数（温度参数），控制策略偏离参考模型的程度
              beta 越大，对偏好差异越敏感
        label_smoothing: 标签平滑系数 (0 表示不使用)

    Returns:
        (loss, metrics): 损失标量和包含监控指标的字典
    """
    # 步骤 1: 计算隐式奖励
    # r_hat(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x))
    # 这里先计算对数概率比（不乘 beta），最终在 logits 中乘
    chosen_log_ratios = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratios = policy_rejected_logps - reference_rejected_logps

    # 步骤 2: 计算隐式奖励差
    # delta_r = r_hat(y_w) - r_hat(y_l) = beta * (log_ratio_w - log_ratio_l)
    logits = beta * (chosen_log_ratios - rejected_log_ratios)

    # 步骤 3: 计算 DPO 损失
    # L = -E[log sigma(logits)]
    if label_smoothing > 0:
        # 标签平滑: 混合正反标签的损失
        # 这有助于防止过度自信的预测
        loss = (
            -label_smoothing * F.logsigmoid(-logits)
            - (1 - label_smoothing) * F.logsigmoid(logits)
        )
    else:
        loss = -F.logsigmoid(logits)

    loss = loss.mean()

    # 计算监控指标
    with torch.no_grad():
        chosen_rewards = beta * chosen_log_ratios
        rejected_rewards = beta * rejected_log_ratios
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

    metrics = {
        "loss": loss.detach(),
        "chosen_rewards": chosen_rewards.mean().detach(),
        "rejected_rewards": rejected_rewards.mean().detach(),
        "reward_accuracy": reward_accuracy.detach(),
        "reward_margin": reward_margin.detach(),
        "logits_mean": logits.mean().detach(),
    }

    return loss, metrics


def ipo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    IPO (Identity Preference Optimization) 损失函数

    IPO 的核心改进: 使用均方误差代替交叉熵损失，防止隐式奖励差趋向无穷大。

    数学形式:
        L_IPO = E[(log(pi(y_w)/pi_ref(y_w)) - log(pi(y_l)/pi_ref(y_l)) - 1/(2*beta))^2]

    直觉: IPO 希望隐式奖励差收敛到 1/(2*beta)，而不是无限增大。
    这提供了内置的正则化效果，防止 DPO 常见的过拟合问题。

    Args:
        与 dpo_loss 相同

    Returns:
        (loss, metrics): 损失标量和监控指标字典
    """
    # 计算对数概率比
    chosen_log_ratios = policy_chosen_logps - reference_chosen_logps
    rejected_log_ratios = policy_rejected_logps - reference_rejected_logps

    # 隐式奖励差（不含 beta 缩放）
    log_ratio_diff = chosen_log_ratios - rejected_log_ratios

    # IPO 目标: 让奖励差接近 1/(2*beta)
    target = 1.0 / (2.0 * beta)

    # 均方误差损失
    loss = ((log_ratio_diff - target) ** 2).mean()

    # 监控指标
    with torch.no_grad():
        chosen_rewards = beta * chosen_log_ratios
        rejected_rewards = beta * rejected_log_ratios
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

    metrics = {
        "loss": loss.detach(),
        "chosen_rewards": chosen_rewards.mean().detach(),
        "rejected_rewards": rejected_rewards.mean().detach(),
        "reward_accuracy": reward_accuracy.detach(),
        "log_ratio_diff": log_ratio_diff.mean().detach(),
        "target": torch.tensor(target),
    }

    return loss, metrics


class DPOLoss(nn.Module):
    """
    DPO 损失函数的 nn.Module 封装

    提供了更方便的接口，可以直接接收模型 logits 并计算损失。
    """

    def __init__(
        self,
        beta: float = 0.1,
        loss_type: str = "dpo",
        label_smoothing: float = 0.0,
    ):
        """
        初始化 DPO 损失模块

        Args:
            beta: KL 惩罚系数
            loss_type: 损失类型，可选 "dpo" 或 "ipo"
            label_smoothing: 标签平滑系数（仅对 DPO 有效）
        """
        super().__init__()
        self.beta = beta
        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播计算损失

        Args:
            与对应损失函数相同

        Returns:
            (loss, metrics)
        """
        if self.loss_type == "dpo":
            return dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=self.beta,
                label_smoothing=self.label_smoothing,
            )
        elif self.loss_type == "ipo":
            return ipo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                beta=self.beta,
            )
        else:
            raise ValueError(f"不支持的损失类型: {self.loss_type}")


if __name__ == "__main__":
    print("=" * 60)
    print("DPO / IPO 损失函数演示")
    print("=" * 60)

    torch.manual_seed(42)
    batch_size = 8
    beta = 0.1

    # 场景 1: 策略已经完美区分偏好
    print("\n--- 场景 1: 策略完美区分偏好 ---")
    policy_chosen = torch.randn(batch_size) + 2.0   # 偏好回答概率更高
    policy_rejected = torch.randn(batch_size) - 2.0  # 非偏好回答概率更低
    ref_chosen = torch.randn(batch_size)
    ref_rejected = torch.randn(batch_size)

    loss, metrics = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta
    )
    print(f"DPO 损失: {loss.item():.4f}")
    print(f"奖励准确率: {metrics['reward_accuracy'].item():.4f}")
    print(f"奖励边际: {metrics['reward_margin'].item():.4f}")

    # 场景 2: 策略与参考模型相同
    print("\n--- 场景 2: 策略 = 参考模型 ---")
    shared_chosen = torch.randn(batch_size)
    shared_rejected = torch.randn(batch_size)

    loss, metrics = dpo_loss(
        shared_chosen, shared_rejected, shared_chosen, shared_rejected, beta=beta
    )
    print(f"DPO 损失: {loss.item():.4f} (理论值: log(2) = {torch.log(torch.tensor(2.0)).item():.4f})")

    # 场景 3: 对比 DPO 和 IPO
    print("\n--- 场景 3: DPO vs IPO 对比 ---")
    policy_chosen = torch.randn(batch_size) + 1.0
    policy_rejected = torch.randn(batch_size) - 0.5
    ref_chosen = torch.randn(batch_size)
    ref_rejected = torch.randn(batch_size)

    dpo_l, dpo_m = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta
    )
    ipo_l, ipo_m = ipo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta
    )
    print(f"DPO 损失: {dpo_l.item():.4f}")
    print(f"IPO 损失: {ipo_l.item():.4f}")
    print(f"IPO 目标奖励差: {ipo_m['target'].item():.4f}")
    print(f"IPO 实际奖励差: {ipo_m['log_ratio_diff'].item():.4f}")

    # 场景 4: beta 对损失的影响
    print("\n--- 场景 4: beta 对 DPO 损失的影响 ---")
    betas = [0.01, 0.05, 0.1, 0.5, 1.0]
    for b in betas:
        loss, _ = dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=b
        )
        print(f"  beta={b:.2f} -> 损失={loss.item():.4f}")

    # 场景 5: 使用 nn.Module 封装
    print("\n--- 场景 5: DPOLoss 模块 ---")
    dpo_module = DPOLoss(beta=0.1, loss_type="dpo")
    loss, metrics = dpo_module(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected
    )
    print(f"模块化 DPO 损失: {loss.item():.4f}")

    print("\n演示完成。")
