"""
MoE 辅助损失实现：负载均衡损失 + Router z-loss

本模块实现了 MoE 训练中用于负载均衡的辅助损失函数。

核心公式:
- 负载均衡损失: L_aux = alpha * N * sum(f_i * P_i)
  其中 f_i 是专家被选中的频率，P_i 是平均路由概率
- Router z-loss: L_z = beta * mean(log(sum(exp(logits)))^2)
  防止路由 logits 数值过大

参考:
- Shazeer et al. (2017). Outrageously Large Neural Networks.
- Fedus et al. (2022). Switch Transformers.
- Zoph et al. (2022). ST-MoE (PaLM 中的 Router z-loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class LoadBalanceLoss(nn.Module):
    """
    负载均衡辅助损失

    目标: 鼓励每个专家被大致等量的 token 选择。
    当所有专家被均匀选择时，损失取最小值。

    L_aux = alpha * N * sum_{i=1}^{N} f_i * P_i

    其中:
    - f_i = (1/T) * sum_t 1[i in TopK(x_t)]  (专家被选中的频率)
    - P_i = (1/T) * sum_t softmax(W_g * x_t)_i  (平均路由概率)
    - alpha 是损失系数（通常 0.01）
    - N 是专家总数

    Args:
        num_experts: 专家总数
        alpha: 负载均衡损失系数
    """

    def __init__(self, num_experts: int, alpha: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算负载均衡损失

        Args:
            logits: [batch, seq_len, num_experts] 路由 logits（softmax 前）
            top_k_indices: [batch, seq_len, top_k] 选中的专家索引

        Returns:
            标量损失值
        """
        # 展平 batch 和 seq_len 维度
        flat_logits = logits.view(-1, self.num_experts)     # [T, N]
        flat_indices = top_k_indices.view(
            -1, top_k_indices.shape[-1]
        )                                                    # [T, K]
        T = flat_logits.shape[0]

        # ---- 计算 f_i: 每个专家被选中的频率 ----
        # 将 indices 转换为 one-hot: [T, K, N]
        one_hot = F.one_hot(flat_indices, self.num_experts).float()
        # 对 K 维求和（一个 token 选了多少次该专家）: [T, N]
        expert_selected = one_hot.sum(dim=1)
        # 对 T 维求均值: [N]
        f = expert_selected.mean(dim=0)

        # ---- 计算 P_i: 平均路由概率 ----
        # 对 logits 做 softmax: [T, N]
        probs = F.softmax(flat_logits, dim=-1)
        # 对 T 维求均值: [N]
        P = probs.mean(dim=0)

        # ---- 计算负载均衡损失 ----
        loss = self.alpha * self.num_experts * torch.sum(f * P)

        return loss


class RouterZLoss(nn.Module):
    """
    Router z-loss

    防止路由 logits 的绝对值过大，提升训练稳定性。
    由 PaLM/ST-MoE 引入。

    L_z = beta * (1/T) * sum_t (log(sum_i exp(h_i(x_t))))^2

    这个损失惩罚 log-partition-function 的平方，
    从而防止 logits 的数值不稳定。

    Args:
        beta: z-loss 系数（通常 0.001）
    """

    def __init__(self, beta: float = 0.001):
        super().__init__()
        self.beta = beta

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        计算 Router z-loss

        Args:
            logits: [batch, seq_len, num_experts] 路由 logits

        Returns:
            标量损失值
        """
        flat_logits = logits.view(-1, logits.shape[-1])  # [T, N]

        # log-partition-function: log(sum(exp(h_i)))
        log_z = torch.logsumexp(flat_logits, dim=-1)  # [T]

        # 惩罚 log_z 的平方
        z_loss = self.beta * torch.mean(log_z ** 2)

        return z_loss


class MoEAuxiliaryLoss(nn.Module):
    """
    MoE 综合辅助损失：负载均衡 + Router z-loss

    L_total = L_main + L_load_balance + L_z_loss

    Args:
        num_experts: 专家总数
        alpha: 负载均衡损失系数
        beta: Router z-loss 系数
    """

    def __init__(
        self,
        num_experts: int,
        alpha: float = 0.01,
        beta: float = 0.001,
    ):
        super().__init__()
        self.load_balance = LoadBalanceLoss(num_experts, alpha)
        self.z_loss = RouterZLoss(beta)

    def forward(
        self,
        logits: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算综合辅助损失

        Args:
            logits: [batch, seq_len, num_experts] 路由 logits
            top_k_indices: [batch, seq_len, top_k] 选中的专家索引

        Returns:
            字典包含:
            - total: 总辅助损失
            - load_balance: 负载均衡损失
            - z_loss: Router z-loss
        """
        lb_loss = self.load_balance(logits, top_k_indices)
        z_loss = self.z_loss(logits)

        return {
            "total": lb_loss + z_loss,
            "load_balance": lb_loss,
            "z_loss": z_loss,
        }


def compute_expert_utilization(
    top_k_indices: torch.Tensor,
    num_experts: int,
) -> Dict[str, torch.Tensor]:
    """
    计算专家利用率指标

    Args:
        top_k_indices: [batch, seq_len, top_k] 选中的专家索引
        num_experts: 专家总数

    Returns:
        字典包含:
        - frequency: [num_experts] 各专家被选中的频率
        - utilization: [num_experts] 各专家的利用率（相对于理想均匀分布）
        - cv: 变异系数（越小越均匀）
        - max_utilization: 最大利用率
        - min_utilization: 最小利用率
    """
    flat_indices = top_k_indices.view(-1, top_k_indices.shape[-1])  # [T, K]
    T = flat_indices.shape[0]
    K = flat_indices.shape[1]

    # 各专家被选中的次数
    counts = torch.zeros(num_experts, device=top_k_indices.device)
    for k in range(K):
        for i in range(num_experts):
            counts[i] += (flat_indices[:, k] == i).sum().float()

    # 频率（被选中次数 / 总选中次数）
    frequency = counts / (T * K)

    # 利用率（相对于理想均匀分布）
    ideal_frequency = 1.0 / num_experts
    utilization = frequency / ideal_frequency

    # 变异系数
    cv = utilization.std() / (utilization.mean() + 1e-8)

    return {
        "frequency": frequency,
        "utilization": utilization,
        "cv": cv.item(),
        "max_utilization": utilization.max().item(),
        "min_utilization": utilization.min().item(),
    }


if __name__ == "__main__":
    # 模拟路由数据
    batch_size = 4
    seq_len = 32
    num_experts = 8
    top_k = 2

    # 模拟路由 logits 和选择结果
    logits = torch.randn(batch_size, seq_len, num_experts)

    # Top-K 选择
    top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)

    print("=" * 60)
    print("辅助损失计算演示")
    print("=" * 60)

    # 综合辅助损失
    aux_loss = MoEAuxiliaryLoss(num_experts, alpha=0.01, beta=0.001)
    losses = aux_loss(logits, top_k_indices)
    print(f"总辅助损失: {losses['total']:.6f}")
    print(f"  负载均衡损失: {losses['load_balance']:.6f}")
    print(f"  Router z-loss: {losses['z_loss']:.6f}")

    print("\n" + "=" * 60)
    print("专家利用率分析")
    print("=" * 60)

    util = compute_expert_utilization(top_k_indices, num_experts)
    print(f"各专家频率: {util['frequency'].tolist()}")
    print(f"各专家利用率: {[f'{u:.2f}' for u in util['utilization'].tolist()]}")
    print(f"变异系数 (CV): {util['cv']:.4f}")
    print(f"最大利用率: {util['max_utilization']:.2f}")
    print(f"最小利用率: {util['min_utilization']:.2f}")

    # 对比：均匀分布 vs 不均匀分布的损失
    print("\n" + "=" * 60)
    print("均匀 vs 不均匀分布的辅助损失对比")
    print("=" * 60)

    # 均匀分布的 logits（所有专家概率相近）
    uniform_logits = torch.zeros(batch_size, seq_len, num_experts)
    _, uniform_indices = uniform_logits.topk(top_k, dim=-1)

    # 极不均匀的 logits（偏好某些专家）
    skewed_logits = torch.zeros(batch_size, seq_len, num_experts)
    skewed_logits[:, :, 0] = 10.0  # 专家 0 的分数远高于其他
    skewed_logits[:, :, 1] = 5.0   # 专家 1 次之
    _, skewed_indices = skewed_logits.topk(top_k, dim=-1)

    lb_loss = LoadBalanceLoss(num_experts, alpha=0.01)

    uniform_loss = lb_loss(uniform_logits, uniform_indices)
    skewed_loss = lb_loss(skewed_logits, skewed_indices)

    print(f"均匀分布的负载均衡损失: {uniform_loss:.6f}")
    print(f"不均匀分布的负载均衡损失: {skewed_loss:.6f}")
    print(f"比值 (不均匀/均匀): {skewed_loss / (uniform_loss + 1e-8):.2f}x")
    print("(不均匀分布的损失应显著更高)")
