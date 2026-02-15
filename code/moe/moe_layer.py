"""
MoE 层完整实现：标准 MoE 层 + DeepSeekMoE 层

本模块将路由器和专家组合成完整的 MoE 层，
可以直接替换 Transformer Block 中的 FFN 层。

核心公式:
- 标准 MoE: MoE(x) = sum_{i in TopK} g_i(x) * E_i(x)
- DeepSeekMoE: MoE(x) = sum(shared_experts) + sum_{i in TopK} g_i(x) * E_i(x)

参考:
- Shazeer et al. (2017). Outrageously Large Neural Networks.
- DeepSeek-AI (2024). DeepSeekMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class TopKRouter(nn.Module):
    """内部使用的 Top-K 路由器（简化版本）"""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        return weights, top_k_indices, logits


class MoELayer(nn.Module):
    """
    标准 MoE 层

    替换 Transformer Block 中的 FFN 层。
    每个 token 通过 Top-K 路由选择 K 个专家，
    各专家输出加权求和。

    MoE(x) = sum_{i in TopK(g(x), K)} g_i(x) * Expert_i(x)

    Args:
        d_model: 模型维度
        num_experts: 专家总数
        top_k: 每个 token 激活的专家数
        d_ff: 每个专家的 FFN 隐藏维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        d_ff = d_ff or 4 * d_model

        # 路由器
        self.router = TopKRouter(d_model, num_experts, top_k)

        # 专家网络（每个是标准 FFN）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(d_ff, d_model, bias=False),
            )
            for _ in range(num_experts)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model] MoE 层输出
            aux_info: 字典，包含 logits 和 indices（用于辅助损失计算）
        """
        batch, seq_len, d = x.shape
        T = batch * seq_len

        # 路由决策
        weights, indices, logits = self.router(x)
        # weights: [batch, seq_len, top_k]
        # indices: [batch, seq_len, top_k]
        # logits: [batch, seq_len, num_experts]

        # 展平处理
        flat_x = x.view(T, d)                            # [T, d]
        flat_weights = weights.view(T, self.top_k)        # [T, K]
        flat_indices = indices.view(T, self.top_k)        # [T, K]

        # 初始化输出
        output = torch.zeros(T, d, device=x.device, dtype=x.dtype)

        # 遍历每个 Top-K 位置
        for k in range(self.top_k):
            # 当前 Top-K 位置的专家索引和权重
            expert_indices = flat_indices[:, k]        # [T]
            expert_weights = flat_weights[:, k:k+1]    # [T, 1]

            # 遍历每个专家，收集属于该专家的 token
            for i in range(self.num_experts):
                mask = (expert_indices == i)
                if mask.any():
                    # 获取属于专家 i 的 token
                    expert_input = flat_x[mask]        # [n_i, d]
                    expert_output = self.experts[i](expert_input)  # [n_i, d]
                    # 加权累加到输出
                    output[mask] += expert_weights[mask] * expert_output

        output = self.dropout(output)
        output = output.view(batch, seq_len, d)

        aux_info = {
            "logits": logits,
            "indices": indices,
            "weights": weights,
        }

        return output, aux_info


class DeepSeekMoELayer(nn.Module):
    """
    DeepSeekMoE 层：细粒度路由专家 + 共享专家

    核心公式:
    MoE(x) = sum_{s=1}^{N_s} Shared_s(x) + sum_{j in TopK} g_j(x) * Routed_j(x)

    特点:
    1. 大量细粒度路由专家（如 64-256 个小 FFN）
    2. 少量共享专家（如 1-2 个标准 FFN，始终激活）
    3. Top-K 路由选择路由专家（K 通常为 6-8）

    Args:
        d_model: 模型维度
        num_routed_experts: 路由专家数量
        num_shared_experts: 共享专家数量
        top_k: 每个 token 激活的路由专家数
        routed_expert_d_ff: 每个路由专家的隐藏维度
        shared_expert_d_ff: 每个共享专家的隐藏维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        num_routed_experts: int = 64,
        num_shared_experts: int = 2,
        top_k: int = 6,
        routed_expert_d_ff: Optional[int] = None,
        shared_expert_d_ff: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_routed = num_routed_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k

        # 路由专家的隐藏维度（细粒度，较小）
        routed_d_ff = routed_expert_d_ff or max(d_model // 2, 64)

        # 共享专家的隐藏维度（较大，与标准 FFN 接近）
        shared_d_ff = shared_expert_d_ff or 4 * d_model

        # 路由器（只路由到路由专家）
        self.router = TopKRouter(d_model, num_routed_experts, top_k)

        # 路由专家（细粒度）
        self.routed_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, routed_d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(routed_d_ff, d_model, bias=False),
            )
            for _ in range(num_routed_experts)
        ])

        # 共享专家（始终激活）
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, shared_d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(shared_d_ff, d_model, bias=False),
            )
            for _ in range(num_shared_experts)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            aux_info: 字典，包含路由信息
        """
        batch, seq_len, d = x.shape
        T = batch * seq_len

        # ===== 共享专家（始终激活）=====
        shared_output = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_output = shared_output + expert(x)

        # ===== 路由专家（稀疏激活）=====
        weights, indices, logits = self.router(x)

        flat_x = x.view(T, d)
        flat_weights = weights.view(T, self.top_k)
        flat_indices = indices.view(T, self.top_k)

        routed_output = torch.zeros(T, d, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            expert_indices = flat_indices[:, k]
            expert_weights = flat_weights[:, k:k+1]

            for i in range(self.num_routed):
                mask = (expert_indices == i)
                if mask.any():
                    expert_input = flat_x[mask]
                    expert_out = self.routed_experts[i](expert_input)
                    routed_output[mask] += expert_weights[mask] * expert_out

        routed_output = routed_output.view(batch, seq_len, d)

        # ===== 组合输出 =====
        output = shared_output + routed_output
        output = self.dropout(output)

        aux_info = {
            "logits": logits,
            "indices": indices,
            "weights": weights,
        }

        return output, aux_info

    def count_parameters(self) -> Dict[str, int]:
        """统计各部分参数量"""
        routed_params = sum(
            p.numel() for e in self.routed_experts for p in e.parameters()
        )
        shared_params = sum(
            p.numel() for e in self.shared_experts for p in e.parameters()
        )
        router_params = sum(p.numel() for p in self.router.parameters())

        return {
            "routed_experts_total": routed_params,
            "shared_experts_total": shared_params,
            "router": router_params,
            "total": routed_params + shared_params + router_params,
            "active_per_token": (
                shared_params
                + self.top_k * routed_params // self.num_routed
                + router_params
            ),
        }


if __name__ == "__main__":
    d_model = 256
    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, d_model)

    print("=" * 60)
    print("标准 MoE 层 (8 专家, Top-2)")
    print("=" * 60)

    moe = MoELayer(d_model=d_model, num_experts=8, top_k=2)
    output, aux = moe(x)
    total_params = sum(p.numel() for p in moe.parameters())
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    print(f"总参数量: {total_params:,}")
    print(f"路由 logits 形状: {aux['logits'].shape}")

    print("\n" + "=" * 60)
    print("DeepSeekMoE 层 (64 路由专家 + 2 共享专家, Top-6)")
    print("=" * 60)

    ds_moe = DeepSeekMoELayer(
        d_model=d_model,
        num_routed_experts=64,
        num_shared_experts=2,
        top_k=6,
        routed_expert_d_ff=128,
        shared_expert_d_ff=1024,
    )
    output, aux = ds_moe(x)
    param_info = ds_moe.count_parameters()
    print(f"输入: {x.shape}")
    print(f"输出: {output.shape}")
    print(f"总参数量: {param_info['total']:,}")
    print(f"  路由专家总参数: {param_info['routed_experts_total']:,}")
    print(f"  共享专家总参数: {param_info['shared_experts_total']:,}")
    print(f"  路由器参数: {param_info['router']:,}")
    print(f"  每 token 激活参数 (估算): {param_info['active_per_token']:,}")

    ratio = param_info['total'] / param_info['active_per_token']
    print(f"  参数效率比 (总/激活): {ratio:.1f}x")
