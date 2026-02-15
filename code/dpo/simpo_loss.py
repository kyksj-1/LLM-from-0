"""
SimPO (Simple Preference Optimization) 损失函数实现

SimPO 的核心创新：
1. 去除参考模型（不需要额外维护 pi_ref）
2. 使用长度归一化的对数概率作为隐式奖励（消除长度偏差）
3. 引入目标奖励差 gamma（类似 SVM 的间隔）

数学基础:
- SimPO 隐式奖励: r(x,y) = (beta/|y|) * log pi_theta(y|x)
- 损失: L = -E[log sigma(r(y_w) - r(y_l) - gamma)]
  其中 gamma > 0 是目标奖励差（margin）

参考:
- Meng et al. (2024). SimPO: Simple Preference Optimization with a Reference-Free Reward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def simpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    SimPO 损失函数

    SimPO 是 DPO 的简化版本，有三个关键改进:
    1. 去掉参考模型 pi_ref
    2. 长度归一化消除长度偏差
    3. 目标奖励差 gamma 防止过早收敛

    数学推导:
    - DPO 隐式奖励: r_DPO = beta * log(pi/pi_ref)
    - SimPO 隐式奖励: r_SimPO = (beta/|y|) * log pi_theta(y|x)
      (直接用策略自身的概率作为奖励信号)
    - SimPO 损失: L = -E[log sigma(r_w - r_l - gamma)]
      其中 r_w = (beta/|y_w|) * sum_t log pi(y_w_t|x,y_w_{<t})
           r_l = (beta/|y_l|) * sum_t log pi(y_l_t|x,y_l_{<t})

    为什么长度归一化有效？
    - 对数概率 log P(y|x) = sum_t log P(y_t|...) 随长度线性递减
    - 不归一化时，短回答天然有更高的"奖励"，产生长度偏差
    - 除以 |y| 后，奖励反映的是"平均每个 token 的置信度"

    Args:
        policy_chosen_logps: 策略在偏好回答上的对数概率之和，形状 [batch_size]
        policy_rejected_logps: 策略在非偏好回答上的对数概率之和，形状 [batch_size]
        chosen_lengths: 偏好回答的长度，形状 [batch_size]
        rejected_lengths: 非偏好回答的长度，形状 [batch_size]
        beta: 温度参数（SimPO 中通常设为较大的值，如 2.0）
        gamma: 目标奖励差（margin），确保偏好回答和非偏好回答之间有间隔
               类似 SVM 中的 margin

    Returns:
        (loss, metrics): 损失标量和监控指标字典
    """
    # 步骤 1: 计算长度归一化的隐式奖励
    # r(x, y) = (beta / |y|) * log pi_theta(y|x)
    chosen_rewards = beta * policy_chosen_logps / chosen_lengths.float()
    rejected_rewards = beta * policy_rejected_logps / rejected_lengths.float()

    # 步骤 2: 计算奖励差 - gamma
    # gamma 是目标间隔，要求偏好回答的奖励至少比非偏好回答高 gamma
    logits = chosen_rewards - rejected_rewards - gamma

    # 步骤 3: 计算损失
    # L = -log sigma(logits)
    loss = -F.logsigmoid(logits).mean()

    # 监控指标
    with torch.no_grad():
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()

        metrics = {
            "loss": loss.detach(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "reward_accuracy": reward_accuracy.detach(),
            "reward_margin": reward_margin.detach(),
            "effective_margin": (reward_margin - gamma).detach(),
        }

    return loss, metrics


class SimPOLoss(nn.Module):
    """
    SimPO 损失函数的 nn.Module 封装

    使用示例:
        criterion = SimPOLoss(beta=2.0, gamma=0.5)
        loss, metrics = criterion(chosen_logps, rejected_logps,
                                  chosen_lengths, rejected_lengths)
    """

    def __init__(self, beta: float = 2.0, gamma: float = 0.5):
        """
        Args:
            beta: 温度参数
            gamma: 目标奖励差
        """
        super().__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        chosen_lengths: torch.Tensor,
        rejected_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """前向传播"""
        return simpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            chosen_lengths,
            rejected_lengths,
            beta=self.beta,
            gamma=self.gamma,
        )


if __name__ == "__main__":
    print("=" * 60)
    print("SimPO (Simple Preference Optimization) 损失函数演示")
    print("=" * 60)

    torch.manual_seed(42)
    batch_size = 8

    # 模拟数据
    chosen_logps = -torch.rand(batch_size) * 30 - 10     # 偏好回答的 log prob
    rejected_logps = -torch.rand(batch_size) * 30 - 20   # 非偏好回答的 log prob
    chosen_lengths = torch.randint(20, 100, (batch_size,))
    rejected_lengths = torch.randint(20, 100, (batch_size,))

    # 场景 1: 标准 SimPO
    print("\n--- 场景 1: 标准 SimPO ---")
    loss, metrics = simpo_loss(
        chosen_logps, rejected_logps,
        chosen_lengths, rejected_lengths,
        beta=2.0, gamma=0.5
    )
    print(f"损失: {loss.item():.4f}")
    print(f"偏好回答奖励: {metrics['chosen_rewards'].item():.4f}")
    print(f"非偏好回答奖励: {metrics['rejected_rewards'].item():.4f}")
    print(f"奖励准确率: {metrics['reward_accuracy'].item():.4f}")
    print(f"奖励边际: {metrics['reward_margin'].item():.4f}")
    print(f"有效边际 (margin - gamma): {metrics['effective_margin'].item():.4f}")

    # 场景 2: gamma 的影响
    print("\n--- 场景 2: gamma 的影响 ---")
    for g in [0.0, 0.25, 0.5, 1.0, 2.0]:
        loss, m = simpo_loss(
            chosen_logps, rejected_logps,
            chosen_lengths, rejected_lengths,
            beta=2.0, gamma=g
        )
        print(f"  gamma={g:.2f} -> 损失={loss.item():.4f}, "
              f"有效边际={m['effective_margin'].item():.4f}")

    # 场景 3: 展示长度归一化的效果
    print("\n--- 场景 3: 长度归一化消除长度偏差 ---")
    # 构造：短回答（总 logp 高）vs 长回答（总 logp 低但每 token logp 高）
    short_logps = torch.tensor([-10.0])  # 短回答，总 log prob 较高
    long_logps = torch.tensor([-30.0])   # 长回答，总 log prob 较低
    short_len = torch.tensor([10])
    long_len = torch.tensor([50])

    # 不归一化的奖励
    raw_short_reward = short_logps.item()
    raw_long_reward = long_logps.item()
    print(f"不归一化: 短回答奖励={raw_short_reward:.2f}, "
          f"长回答奖励={raw_long_reward:.2f}")
    print(f"  -> 短回答 '看起来更好' (但可能只是因为更短)")

    # 归一化的奖励
    norm_short_reward = short_logps.item() / short_len.item()
    norm_long_reward = long_logps.item() / long_len.item()
    print(f"长度归一化: 短回答奖励={norm_short_reward:.2f}, "
          f"长回答奖励={norm_long_reward:.2f}")
    print(f"  -> 长回答每 token 置信度更高")

    # 场景 4: 对比 SimPO 和 DPO (需要参考模型 vs 不需要)
    print("\n--- 场景 4: SimPO 不需要参考模型 ---")
    print("SimPO 优势:")
    print("  - 不需要加载和存储参考模型 (节省约 50% 显存)")
    print("  - 不需要参考模型的前向传播 (节省约 30% 计算)")
    print("  - 长度归一化消除长度偏差")
    print("  - gamma 参数提供额外的控制手段")

    # 场景 5: 使用 nn.Module
    print("\n--- 场景 5: SimPOLoss 模块 ---")
    criterion = SimPOLoss(beta=2.0, gamma=0.5)
    loss_mod, _ = criterion(
        chosen_logps, rejected_logps, chosen_lengths, rejected_lengths
    )
    print(f"模块化损失: {loss_mod.item():.4f}")

    print("\n演示完成。")
