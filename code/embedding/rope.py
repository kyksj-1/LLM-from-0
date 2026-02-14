"""
RoPE（旋转位置编码）完整实现

包含：
- 频率计算
- 旋转变换
- 在注意力中的应用
- 长度扩展方法（Position Interpolation, NTK-aware）

作者: LLM学习教程
模块: 模块2 - Embedding

参考: Su et al. (2021) - RoFormer: Enhanced Transformer with Rotary Position Embedding
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    scaling_factor: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的 cos 和 sin 缓存

    频率公式: θ_i = 1 / (base^(2i/dim))
    缓存: cos(m * θ_i), sin(m * θ_i) for m in [0, max_seq_len)

    Args:
        dim: 嵌入维度（必须为偶数）
        max_seq_len: 最大序列长度
        base: 频率基数（默认10000）
        scaling_factor: 位置缩放因子（用于长度外推）

    Returns:
        (cos_cache, sin_cache): 各 [max_seq_len, dim]
    """
    # 计算频率: θ_i = 1 / base^(2i/dim)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # 位置索引（可选缩放）
    t = torch.arange(max_seq_len).float() / scaling_factor

    # 外积: [max_seq_len] x [dim/2] -> [max_seq_len, dim/2]
    freqs = torch.outer(t, inv_freq)

    # 复制一份得到 [max_seq_len, dim]
    # 每个频率对应两个维度（cos和sin的配对）
    emb = torch.cat([freqs, freqs], dim=-1)

    cos_cache = emb.cos()
    sin_cache = emb.sin()

    return cos_cache, sin_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将输入的后半部分取负并与前半部分交换

    rotate_half([x1, x2, x3, x4]) = [-x3, -x4, x1, x2]

    这对应于二维旋转中的操作:
    [cos θ, -sin θ] [x1]   [x1 cos θ - x2 sin θ]
    [sin θ,  cos θ] [x2] = [x1 sin θ + x2 cos θ]

    实现中通过 x * cos + rotate_half(x) * sin 来完成旋转

    Args:
        x: [..., dim] 的张量（dim必须为偶数）

    Returns:
        旋转后的张量
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Query 和 Key 应用 RoPE

    核心公式:
    q_rotated = q * cos(mθ) + rotate_half(q) * sin(mθ)
    k_rotated = k * cos(nθ) + rotate_half(k) * sin(nθ)

    数学性质（RoPE的核心优势）:
    q_rotated^T @ k_rotated = q^T @ R(m-n) @ k
    即注意力分数只依赖相对位置 (m-n)

    Args:
        q: Query [batch, n_heads, seq_len, head_dim]
        k: Key [batch, n_heads, seq_len, head_dim]
        cos: cos缓存 [seq_len, head_dim] 或 [1, 1, seq_len, head_dim]
        sin: sin缓存 [seq_len, head_dim] 或 [1, 1, seq_len, head_dim]
        position_ids: 可选的位置ID [batch, seq_len]

    Returns:
        (q_embed, k_embed): 应用RoPE后的Query和Key
    """
    if position_ids is not None:
        # 按位置ID索引cos/sin
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        # 使用前seq_len个位置
        seq_len = q.shape[-2]
        cos = cos[:seq_len]
        sin = sin[:seq_len]

        # 扩展维度以匹配 [batch, heads, seq_len, dim]
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)

    # 应用旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE 模块

    管理频率缓存，提供统一的 RoPE 接口。
    支持多种长度扩展方法。
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling_type: str = 'none',
        scaling_factor: float = 1.0
    ):
        """
        Args:
            dim: head_dim（每个注意力头的维度，必须为偶数）
            max_seq_len: 最大序列长度
            base: 频率基数
            scaling_type: 长度扩展方法
                - 'none': 不扩展
                - 'linear': Position Interpolation
                - 'ntk': NTK-aware Interpolation
                - 'yarn': YaRN
            scaling_factor: 扩展倍数
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor

        # 根据扩展方法计算频率
        if scaling_type == 'ntk':
            # NTK-aware: 修改 base
            adjusted_base = base * (scaling_factor ** (dim / (dim - 2)))
            cos_cache, sin_cache = precompute_freqs_cis(dim, max_seq_len, adjusted_base)
        elif scaling_type == 'linear':
            # Position Interpolation: 缩放位置索引
            cos_cache, sin_cache = precompute_freqs_cis(
                dim, max_seq_len, base, scaling_factor
            )
        else:
            # 标准 RoPE
            cos_cache, sin_cache = precompute_freqs_cis(dim, max_seq_len, base)

        self.register_buffer('cos_cached', cos_cache)
        self.register_buffer('sin_cached', sin_cache)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 Q 和 K 应用 RoPE

        Args:
            q: [batch, n_heads, seq_len, head_dim]
            k: [batch, n_heads, seq_len, head_dim]
            position_ids: [batch, seq_len]（可选）

        Returns:
            (q_rotated, k_rotated)
        """
        seq_len = q.shape[-2]

        # 动态扩展缓存
        if seq_len > self.max_seq_len:
            self._extend_cache(seq_len)

        return apply_rotary_pos_emb(
            q, k,
            self.cos_cached, self.sin_cached,
            position_ids
        )

    def _extend_cache(self, new_len: int):
        """动态扩展缓存长度"""
        if self.scaling_type == 'ntk':
            adjusted_base = self.base * (self.scaling_factor ** (self.dim / (self.dim - 2)))
            cos_cache, sin_cache = precompute_freqs_cis(self.dim, new_len, adjusted_base)
        elif self.scaling_type == 'linear':
            cos_cache, sin_cache = precompute_freqs_cis(
                self.dim, new_len, self.base, self.scaling_factor
            )
        else:
            cos_cache, sin_cache = precompute_freqs_cis(self.dim, new_len, self.base)

        self.cos_cached = cos_cache.to(self.cos_cached.device)
        self.sin_cached = sin_cache.to(self.sin_cached.device)
        self.max_seq_len = new_len


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 50)
    print("RoPE 旋转位置编码演示")
    print("=" * 50)

    # 基本参数
    batch_size = 2
    n_heads = 8
    seq_len = 16
    head_dim = 64

    # 1. 基本RoPE
    print("\n1. 基本RoPE使用")
    rope = RotaryPositionalEmbedding(head_dim, max_seq_len=2048)
    q = torch.randn(batch_size, n_heads, seq_len, head_dim)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim)

    q_rot, k_rot = rope(q, k)
    print(f"   Q: {q.shape} -> {q_rot.shape}")
    print(f"   K: {k.shape} -> {k_rot.shape}")

    # 2. 验证相对位置性质
    print("\n2. 验证相对位置性质")
    # 创建简单的Q和K
    q_test = torch.randn(1, 1, 4, head_dim)
    k_test = torch.randn(1, 1, 4, head_dim)

    q_rot_test, k_rot_test = rope(q_test, k_test)

    # 计算注意力分数矩阵
    scores = torch.matmul(q_rot_test, k_rot_test.transpose(-2, -1))
    print(f"   注意力分数矩阵:\n{scores[0, 0].detach()}")

    # 验证: score(i,j) 应该只依赖于 i-j
    # score(0,1) 应近似等于 score(1,2) 和 score(2,3)（如果Q和K相同）
    q_same = torch.randn(1, 1, 1, head_dim).expand(1, 1, 4, head_dim)
    k_same = torch.randn(1, 1, 1, head_dim).expand(1, 1, 4, head_dim)
    q_rot_same, k_rot_same = rope(q_same.clone(), k_same.clone())
    scores_same = torch.matmul(q_rot_same, k_rot_same.transpose(-2, -1))
    print(f"\n   相同Q/K的注意力分数（对角线应相等）:")
    print(f"   score(0,1) = {scores_same[0,0,0,1]:.4f}")
    print(f"   score(1,2) = {scores_same[0,0,1,2]:.4f}")
    print(f"   score(2,3) = {scores_same[0,0,2,3]:.4f}")

    # 3. 长度扩展方法对比
    print("\n3. 长度扩展方法对比")
    methods = {
        'none': RotaryPositionalEmbedding(head_dim, max_seq_len=128),
        'linear (2x)': RotaryPositionalEmbedding(head_dim, max_seq_len=256, scaling_type='linear', scaling_factor=2.0),
        'ntk (2x)': RotaryPositionalEmbedding(head_dim, max_seq_len=256, scaling_type='ntk', scaling_factor=2.0),
    }

    long_q = torch.randn(1, 1, 256, head_dim)
    long_k = torch.randn(1, 1, 256, head_dim)

    for name, rope_method in methods.items():
        try:
            q_out, k_out = rope_method(long_q, long_k)
            print(f"   {name:15s}: 成功, 输出形状 {q_out.shape}")
        except Exception as e:
            print(f"   {name:15s}: 失败 - {e}")

    print("\n演示完成!")
