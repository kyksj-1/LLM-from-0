"""
MoE 路由器实现：Top-K Router, Expert Choice Router

本模块实现了混合专家模型（MoE）的核心路由机制。
路由器决定每个 token 由哪些专家处理以及对应的权重。

数学基础:
- Top-K 路由: g(x) = softmax(TopK(W_g * x, K))
- Expert Choice: 每个专家选择 Top-C 个 token

参考:
- Shazeer et al. (2017). Outrageously Large Neural Networks.
- Fedus et al. (2022). Switch Transformers.
- Zhou et al. (2022). Mixture-of-Experts with Expert Choice Routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TopKRouter(nn.Module):
    """
    Top-K 路由器

    为每个 token 选择 K 个专家，返回归一化的路由权重和对应的专家索引。
    支持可选的噪声注入以增强训练时的探索性。

    Args:
        d_model: 模型维度（输入特征维度）
        num_experts: 专家总数
        top_k: 每个 token 激活的专家数量
        noise_std: 训练时路由 logits 的噪声标准差（0 表示无噪声）
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # 路由线性层: 将输入映射到专家分数
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：计算路由权重和专家选择

        Args:
            x: [batch, seq_len, d_model] 输入张量

        Returns:
            weights: [batch, seq_len, top_k] 归一化路由权重
            indices: [batch, seq_len, top_k] 选中的专家索引
            logits:  [batch, seq_len, num_experts] 原始路由 logits（用于辅助损失）
        """
        # 计算路由 logits: [batch, seq_len, num_experts]
        logits = self.gate(x)

        # 训练时注入噪声，增加探索性
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # 选择 Top-K 个专家
        # top_k_logits: [batch, seq_len, top_k]
        # top_k_indices: [batch, seq_len, top_k]
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)

        # 在选中的 K 个专家上做 softmax 归一化
        weights = F.softmax(top_k_logits, dim=-1)

        return weights, top_k_indices, logits


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice 路由器

    与 Token Choice（Top-K）相反，Expert Choice 让每个专家选择要处理的 token。
    天然实现负载均衡，无需辅助损失。

    注意：Expert Choice 不适合自回归推理（需要看到全序列），
    主要用于编码器或非自回归场景。

    Args:
        d_model: 模型维度
        num_experts: 专家总数
        capacity_factor: 容量因子，决定每个专家处理的 token 数量
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        capacity_factor: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

        # 路由线性层
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播：每个专家选择自己要处理的 token

        Args:
            x: [batch * seq_len, d_model] 展平后的输入（需要预先展平）

        Returns:
            dispatch_weights: [num_experts, capacity] 各专家对所选 token 的权重
            dispatch_indices: [num_experts, capacity] 各专家选中的 token 索引
            logits: [T, num_experts] 原始路由 logits
        """
        T = x.shape[0]  # 总 token 数量

        # 每个专家的容量
        capacity = int(T * self.capacity_factor / self.num_experts)
        capacity = max(capacity, 1)

        # 计算路由分数: [T, num_experts]
        logits = self.gate(x)
        scores = F.softmax(logits, dim=0)  # 沿 token 维度 softmax

        # 每个专家选择 Top-C 个 token
        # scores 转置为 [num_experts, T]，每行选 Top-C
        scores_t = scores.t()  # [num_experts, T]
        top_c_scores, top_c_indices = torch.topk(
            scores_t, capacity, dim=-1
        )  # [num_experts, capacity]

        return top_c_scores, top_c_indices, logits


class NoisyTopKRouter(nn.Module):
    """
    带噪声的 Top-K 路由器（Shazeer et al., 2017 原始设计）

    在 softmax 之前加入可学习的噪声，提升探索性。
    噪声量由额外的线性层控制。

    Args:
        d_model: 模型维度
        num_experts: 专家总数
        top_k: 每个 token 激活的专家数量
    """

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 路由线性层
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        # 噪声控制层：决定每个专家的噪声大小
        self.noise_gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            weights, indices, logits
        """
        # 计算路由 logits
        logits = self.gate(x)  # [batch, seq_len, num_experts]

        # 训练时加入可控噪声
        if self.training:
            noise_logits = self.noise_gate(x)
            noise_std = F.softplus(noise_logits)  # 保证噪声为正
            noise = torch.randn_like(logits) * noise_std
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        # Top-K 选择（基于加噪 logits）
        top_k_logits, top_k_indices = torch.topk(
            noisy_logits, self.top_k, dim=-1
        )

        # 归一化权重（基于原始 logits 中选中专家的值）
        # 收集原始 logits 中对应位置的值
        original_top_k = torch.gather(logits, -1, top_k_indices)
        weights = F.softmax(original_top_k, dim=-1)

        return weights, top_k_indices, logits


if __name__ == "__main__":
    # 演示各种路由器的使用
    batch_size = 2
    seq_len = 8
    d_model = 64
    num_experts = 8

    x = torch.randn(batch_size, seq_len, d_model)

    print("=" * 60)
    print("Top-K Router 演示")
    print("=" * 60)

    router = TopKRouter(d_model, num_experts, top_k=2, noise_std=0.1)
    weights, indices, logits = router(x)
    print(f"输入形状: {x.shape}")
    print(f"路由权重形状: {weights.shape}")
    print(f"专家索引形状: {indices.shape}")
    print(f"路由 logits 形状: {logits.shape}")
    print(f"第一个 token 的选中专家: {indices[0, 0].tolist()}")
    print(f"对应权重: {weights[0, 0].tolist()}")

    print("\n" + "=" * 60)
    print("Expert Choice Router 演示")
    print("=" * 60)

    ec_router = ExpertChoiceRouter(d_model, num_experts, capacity_factor=1.0)
    flat_x = x.view(-1, d_model)  # 展平
    ec_weights, ec_indices, ec_logits = ec_router(flat_x)
    print(f"展平输入形状: {flat_x.shape}")
    print(f"每个专家选择的 token 权重形状: {ec_weights.shape}")
    print(f"每个专家选择的 token 索引形状: {ec_indices.shape}")
    capacity = ec_indices.shape[1]
    print(f"每个专家的容量: {capacity}")

    print("\n" + "=" * 60)
    print("Noisy Top-K Router 演示")
    print("=" * 60)

    noisy_router = NoisyTopKRouter(d_model, num_experts, top_k=2)
    nw, ni, nl = noisy_router(x)
    print(f"路由权重形状: {nw.shape}")
    print(f"专家索引形状: {ni.shape}")
