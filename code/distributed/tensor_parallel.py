"""
张量并行 (Tensor Parallelism) 简化实现

本模块实现 Megatron-LM 风格的张量并行，包括:
    - 列并行线性层 (Column Parallel Linear)
    - 行并行线性层 (Row Parallel Linear)
    - 张量并行的多头注意力
    - 张量并行的 FFN

核心思想:
    将模型的单个层的权重矩阵按维度切分到多个设备上，
    每个设备计算部分结果后通过通信合并。

注意: 本实现在单进程中模拟多设备的张量并行，
    实际使用时应结合 torch.distributed。

作者: LLM Tutorial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math


class ColumnParallelLinear(nn.Module):
    """
    列并行线性层

    将权重矩阵 W [in_features, out_features] 按列（输出维度）切分，
    每个设备持有 W_i [in_features, out_features / world_size]。

    前向传播:
        输入 X 在每个设备上完整复制
        输出 Y_i = X @ W_i，每个设备持有部分输出

    通信:
        前向: 无需通信（输入已复制）
        反向: All-Reduce 输入的梯度
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = False,
    ):
        """
        Args:
            in_features: 输入特征维度
            out_features: 总输出特征维度（将被切分）
            world_size: 张量并行度（设备数量）
            rank: 当前设备的编号
            bias: 是否使用偏置
        """
        super().__init__()
        assert out_features % world_size == 0, (
            f"out_features ({out_features}) 必须能被 world_size ({world_size}) 整除"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.local_out_features = out_features // world_size

        # 每个设备只持有部分权重
        self.weight = nn.Parameter(
            torch.randn(self.local_out_features, in_features) * (1.0 / math.sqrt(in_features))
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.local_out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量 [batch_size, seq_len, in_features]

        Returns:
            局部输出 [batch_size, seq_len, local_out_features]
        """
        output = F.linear(x, self.weight, self.bias)
        return output


class RowParallelLinear(nn.Module):
    """
    行并行线性层

    将权重矩阵 W [in_features, out_features] 按行（输入维度）切分，
    每个设备持有 W_i [in_features / world_size, out_features]。

    前向传播:
        输入 X 需要按列切分，每个设备持有 X_i [batch, seq, in_features/world_size]
        输出 Y_i = X_i @ W_i，然后通过 All-Reduce 求和得到完整结果

    通信:
        前向: All-Reduce 输出
        反向: 无需额外通信（梯度自然分片）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        world_size: int,
        rank: int,
        bias: bool = False,
    ):
        """
        Args:
            in_features: 总输入特征维度（将被切分）
            out_features: 输出特征维度
            world_size: 张量并行度
            rank: 当前设备编号
            bias: 是否使用偏置
        """
        super().__init__()
        assert in_features % world_size == 0, (
            f"in_features ({in_features}) 必须能被 world_size ({world_size}) 整除"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        self.local_in_features = in_features // world_size

        # 每个设备只持有部分权重
        self.weight = nn.Parameter(
            torch.randn(out_features, self.local_in_features) * (1.0 / math.sqrt(in_features))
        )
        if bias:
            # 偏置只在 rank 0 上有效
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 局部输入 [batch_size, seq_len, local_in_features]

        Returns:
            局部输出 [batch_size, seq_len, out_features]
            注意: 实际分布式环境需要 All-Reduce 求和
        """
        output = F.linear(x, self.weight)
        if self.bias is not None and self.rank == 0:
            output = output + self.bias
        return output


class TensorParallelMultiHeadAttention(nn.Module):
    """
    张量并行的多头注意力

    按注意力头数切分:
        - 每个设备负责 n_heads / world_size 个注意力头
        - QKV 投影使用列并行（按头数切分）
        - 输出投影使用行并行

    Megatron-LM 的设计精髓:
        1. QKV 列并行: 每个设备独立计算部分头的 QKV
        2. 注意力计算: 各设备完全独立（无通信）
        3. 输出投影行并行: All-Reduce 合并结果
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        world_size: int,
        rank: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 总注意力头数
            world_size: 张量并行度
            rank: 当前设备编号
            dropout: Dropout 概率
        """
        super().__init__()
        assert n_heads % world_size == 0, (
            f"n_heads ({n_heads}) 必须能被 world_size ({world_size}) 整除"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.world_size = world_size
        self.rank = rank
        self.local_heads = n_heads // world_size
        self.head_dim = d_model // n_heads

        # QKV 投影使用列并行
        # 每个设备只计算 local_heads 个头的 QKV
        local_qkv_dim = self.local_heads * self.head_dim
        self.wq = ColumnParallelLinear(d_model, d_model, world_size, rank)
        self.wk = ColumnParallelLinear(d_model, d_model, world_size, rank)
        self.wv = ColumnParallelLinear(d_model, d_model, world_size, rank)

        # 输出投影使用行并行
        self.wo = RowParallelLinear(d_model, d_model, world_size, rank)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        is_causal: bool = True,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入 [batch_size, seq_len, d_model]
            is_causal: 是否使用因果掩码

        Returns:
            输出 [batch_size, seq_len, d_model]
            注意: 实际需要在 wo 后进行 All-Reduce
        """
        B, S, _ = x.shape

        # 列并行 QKV 投影（各设备独立）
        q = self.wq(x)  # [B, S, local_heads * head_dim]
        k = self.wk(x)
        v = self.wv(x)

        # 重塑为多头形式
        q = q.view(B, S, self.local_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.local_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.local_heads, self.head_dim).transpose(1, 2)

        # 注意力计算（各设备完全独立，无通信）
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            causal_mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)

        # 重塑回 [B, S, local_heads * head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, self.local_heads * self.head_dim)

        # 行并行输出投影（需要 All-Reduce）
        out = self.wo(out)

        return out


class TensorParallelFFN(nn.Module):
    """
    张量并行的前馈网络 (SwiGLU 风格)

    切分策略:
        - W_gate 和 W_up 使用列并行（按输出维度切分）
        - W_down 使用行并行（按输入维度切分）
        - 中间的激活函数在各设备上独立计算

    通信:
        - 仅在 W_down (行并行) 后需要 All-Reduce
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        world_size: int,
        rank: int,
    ):
        """
        Args:
            d_model: 模型维度
            d_ff: FFN 隐藏维度
            world_size: 张量并行度
            rank: 当前设备编号
        """
        super().__init__()

        self.d_model = d_model
        self.d_ff = d_ff
        self.world_size = world_size
        self.rank = rank

        # 列并行: W_gate 和 W_up
        self.w_gate = ColumnParallelLinear(d_model, d_ff, world_size, rank)
        self.w_up = ColumnParallelLinear(d_model, d_ff, world_size, rank)

        # 行并行: W_down
        self.w_down = RowParallelLinear(d_ff, d_model, world_size, rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU 前向传播

        SwiGLU(x) = (Swish(x @ W_gate) * (x @ W_up)) @ W_down

        Args:
            x: 输入 [batch_size, seq_len, d_model]

        Returns:
            输出 [batch_size, seq_len, d_model]
        """
        # 列并行: 各设备独立计算
        gate = F.silu(self.w_gate(x))  # Swish 激活
        up = self.w_up(x)

        # 逐元素相乘（各设备独立）
        hidden = gate * up

        # 行并行: 需要 All-Reduce
        out = self.w_down(hidden)

        return out


class TensorParallelTransformerBlock(nn.Module):
    """
    张量并行的 Transformer Block

    Megatron-LM 风格: 每个 Block 仅需 2 次 All-Reduce
        1. 注意力的输出投影后
        2. FFN 的 W_down 后
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        world_size: int,
        rank: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            d_model: 模型维度
            n_heads: 注意力头数
            d_ff: FFN 隐藏维度
            world_size: 张量并行度
            rank: 当前设备编号
            dropout: Dropout 概率
        """
        super().__init__()

        # 注意力层（张量并行）
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = TensorParallelMultiHeadAttention(
            d_model, n_heads, world_size, rank, dropout
        )

        # FFN 层（张量并行）
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = TensorParallelFFN(d_model, d_ff, world_size, rank)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (Pre-Norm)

        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model]
        """
        # Attention + 残差
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = residual + x  # 注意: 实际环境中此处的 x 来自 All-Reduce

        # FFN + 残差
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x  # 注意: 同上

        return x


def simulate_tensor_parallel(
    d_model: int = 256,
    n_heads: int = 8,
    d_ff: int = 512,
    world_size: int = 4,
    batch_size: int = 2,
    seq_len: int = 16,
):
    """
    模拟张量并行的执行过程

    创建 world_size 个张量并行 Block，各自处理相同的输入，
    然后手动将各设备的结果合并，验证正确性。

    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: FFN 隐藏维度
        world_size: 张量并行度
        batch_size: 批次大小
        seq_len: 序列长度
    """
    print("=" * 60)
    print(f"张量并行模拟 (TP={world_size})")
    print("=" * 60)
    print(f"  d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  每个设备的注意力头数: {n_heads // world_size}")
    print(f"  每个设备的 FFN 隐藏维度: {d_ff // world_size}")

    # 创建每个 rank 的 Block
    blocks = []
    for rank in range(world_size):
        block = TensorParallelTransformerBlock(
            d_model, n_heads, d_ff, world_size, rank
        )
        blocks.append(block)

    # 创建输入
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, d_model)

    # 各设备独立计算
    outputs = []
    for rank, block in enumerate(blocks):
        with torch.no_grad():
            out = block(x)
        outputs.append(out)
        print(f"\n  Rank {rank} 输出形状: {out.shape}")
        print(f"  Rank {rank} 输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")

    # 在实际环境中，行并行后的 All-Reduce 会使所有设备的输出一致
    # 这里模拟 All-Reduce: 将所有设备的输出求和
    combined = sum(outputs)
    print(f"\n  合并后输出形状: {combined.shape}")
    print(f"  合并后输出范围: [{combined.min().item():.4f}, {combined.max().item():.4f}]")

    # 分析参数量
    total_params = sum(p.numel() for p in blocks[0].parameters())
    full_model_params = total_params * world_size  # 近似
    print(f"\n  每个设备的参数量: {total_params:,}")
    print(f"  完整模型参数量 (近似): {full_model_params:,}")
    print(f"  参数切分比例: 1/{world_size}")

    # 通信量分析
    activation_size = batch_size * seq_len * d_model * 4  # FP32
    print(f"\n  每次 All-Reduce 的数据量: {activation_size / 1024:.1f} KB")
    print(f"  每个 Block 的 All-Reduce 次数: 2 (前向) + 2 (反向) = 4")
    print(f"  每个 Block 的总通信量: {4 * activation_size / 1024:.1f} KB")


if __name__ == "__main__":
    simulate_tensor_parallel()
