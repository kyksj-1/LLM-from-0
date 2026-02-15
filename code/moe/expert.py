"""
MoE 专家网络实现：标准专家、细粒度专家、共享专家

本模块实现了 MoE 架构中的各种专家网络。
每个专家本质上是一个 FFN（前馈网络），但在 MoE 中以不同的粒度和模式组织。

核心概念:
- 标准专家: 完整的 FFN，隐藏维度与普通 Transformer 相同
- 细粒度专家 (DeepSeekMoE): 将标准 FFN 拆分为多个小 FFN
- 共享专家 (DeepSeekMoE): 始终激活的专家，处理通用知识

参考:
- Shazeer (2020). GLU Variants Improve Transformer.
- DeepSeek-AI (2024). DeepSeekMoE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Expert(nn.Module):
    """
    标准专家网络（FFN with SiLU/SwiGLU）

    每个专家是一个独立的前馈网络，结构与标准 Transformer FFN 相同。
    支持两种模式：标准 FFN（两层 + 激活）和 SwiGLU（门控 + 上投影 + 下投影）。

    Args:
        d_model: 输入/输出维度
        d_ff: FFN 隐藏维度（None 时默认 4 * d_model）
        use_swiglu: 是否使用 SwiGLU 激活（True 时更接近现代 LLM）
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        use_swiglu: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_swiglu = use_swiglu

        if use_swiglu:
            # SwiGLU: output = (Swish(x @ W_gate) * (x @ W_up)) @ W_down
            # 隐藏维度调整为 2/3 * 4d 保持参数量接近标准 FFN
            d_ff = d_ff or int(8 * d_model / 3)
            self.w_gate = nn.Linear(d_model, d_ff, bias=False)
            self.w_up = nn.Linear(d_model, d_ff, bias=False)
            self.w_down = nn.Linear(d_ff, d_model, bias=False)
        else:
            # 标准 FFN: output = SiLU(x @ W1) @ W2
            d_ff = d_ff or 4 * d_model
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_ff, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_model] 任意前导维度的输入

        Returns:
            [..., d_model] 与输入形状相同的输出
        """
        if self.use_swiglu:
            # SwiGLU 前向传播
            gate = F.silu(self.w_gate(x))  # Swish 门控
            up = self.w_up(x)              # 上投影
            out = self.w_down(gate * up)   # 门控混合 + 下投影
        else:
            # 标准 FFN 前向传播
            out = self.w2(F.silu(self.w1(x)))

        return self.dropout(out)


class FineGrainedExpert(nn.Module):
    """
    细粒度专家 (DeepSeekMoE 风格)

    将标准大专家拆分为更小的专家，每个专家的隐藏维度较小。
    在相同总参数量下，提供更精确的路由选择。

    例如: 标准 MoE 8 专家(d_ff=4096) → 细粒度 MoE 64 专家(d_ff=512)

    Args:
        d_model: 输入/输出维度
        d_ff: 细粒度专家的隐藏维度（通常远小于标准 FFN）
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        # 细粒度专家使用简单的两层结构（不用 SwiGLU 以减少参数）
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_model]

        Returns:
            [..., d_model]
        """
        return self.dropout(self.w_down(F.silu(self.w_up(x))))


class SharedExpert(nn.Module):
    """
    共享专家 (DeepSeekMoE 风格)

    始终激活的专家，不参与路由。处理所有 token 都需要的通用知识。
    结构与标准 Expert 相同，但在 MoE 层中独立于路由机制。

    Args:
        d_model: 输入/输出维度
        d_ff: 隐藏维度（通常与标准 FFN 相同或更大）
        use_swiglu: 是否使用 SwiGLU
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        d_ff: Optional[int] = None,
        use_swiglu: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        # 共享专家本质上就是一个标准的 Expert
        self.expert = Expert(d_model, d_ff, use_swiglu, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        共享专家前向传播（始终激活，处理所有 token）

        Args:
            x: [..., d_model]

        Returns:
            [..., d_model]
        """
        return self.expert(x)


class ExpertGroup(nn.Module):
    """
    专家组：管理多个专家的集合

    提供统一的接口来管理和调用多个专家网络。
    支持标准专家和细粒度专家两种模式。

    Args:
        d_model: 输入/输出维度
        num_experts: 专家数量
        d_ff: 每个专家的隐藏维度
        expert_type: 专家类型 ("standard" 或 "fine_grained")
        use_swiglu: 是否使用 SwiGLU（仅 standard 模式有效）
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        d_ff: Optional[int] = None,
        expert_type: str = "standard",
        use_swiglu: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.expert_type = expert_type

        if expert_type == "standard":
            self.experts = nn.ModuleList([
                Expert(d_model, d_ff, use_swiglu, dropout)
                for _ in range(num_experts)
            ])
        elif expert_type == "fine_grained":
            # 细粒度专家需要指定 d_ff
            assert d_ff is not None, "细粒度专家需要指定 d_ff"
            self.experts = nn.ModuleList([
                FineGrainedExpert(d_model, d_ff, dropout)
                for _ in range(num_experts)
            ])
        else:
            raise ValueError(f"不支持的专家类型: {expert_type}")

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """
        调用指定索引的专家

        Args:
            x: [..., d_model] 输入
            expert_idx: 专家索引

        Returns:
            [..., d_model] 专家输出
        """
        return self.experts[expert_idx](x)

    def count_parameters(self) -> dict:
        """
        统计参数量

        Returns:
            包含各专家和总参数量的字典
        """
        expert_params = [
            sum(p.numel() for p in e.parameters())
            for e in self.experts
        ]
        return {
            "per_expert": expert_params[0] if expert_params else 0,
            "total": sum(expert_params),
            "num_experts": self.num_experts,
        }


if __name__ == "__main__":
    d_model = 256
    batch_size = 2
    seq_len = 16

    x = torch.randn(batch_size, seq_len, d_model)

    print("=" * 60)
    print("标准专家 (SwiGLU)")
    print("=" * 60)
    expert = Expert(d_model, use_swiglu=True)
    out = expert(x)
    params = sum(p.numel() for p in expert.parameters())
    print(f"输入: {x.shape} -> 输出: {out.shape}")
    print(f"参数量: {params:,}")

    print("\n" + "=" * 60)
    print("细粒度专家")
    print("=" * 60)
    fg_expert = FineGrainedExpert(d_model, d_ff=128)
    out = fg_expert(x)
    params = sum(p.numel() for p in fg_expert.parameters())
    print(f"输入: {x.shape} -> 输出: {out.shape}")
    print(f"参数量: {params:,}")

    print("\n" + "=" * 60)
    print("共享专家")
    print("=" * 60)
    shared = SharedExpert(d_model, d_ff=1024)
    out = shared(x)
    params = sum(p.numel() for p in shared.parameters())
    print(f"输入: {x.shape} -> 输出: {out.shape}")
    print(f"参数量: {params:,}")

    print("\n" + "=" * 60)
    print("专家组对比")
    print("=" * 60)

    # 标准专家组: 8 个大专家
    standard_group = ExpertGroup(
        d_model, num_experts=8, expert_type="standard"
    )
    std_info = standard_group.count_parameters()
    print(f"标准专家组 (8 专家):")
    print(f"  每个专家参数: {std_info['per_expert']:,}")
    print(f"  总参数: {std_info['total']:,}")

    # 细粒度专家组: 64 个小专家
    fg_group = ExpertGroup(
        d_model, num_experts=64, d_ff=128, expert_type="fine_grained"
    )
    fg_info = fg_group.count_parameters()
    print(f"\n细粒度专家组 (64 专家):")
    print(f"  每个专家参数: {fg_info['per_expert']:,}")
    print(f"  总参数: {fg_info['total']:,}")

    ratio = fg_info['total'] / std_info['total']
    print(f"\n参数量比值 (细粒度 / 标准): {ratio:.2f}")
