"""
位置编码统一实现：Sinusoidal + RoPE + ALiBi

本模块实现了三种主流位置编码方案的统一接口，方便对比和切换。

三种方案的核心差异:
- Sinusoidal: 绝对位置编码，加在输入嵌入上 (PE_{pos,2i} = sin(pos/10000^{2i/d}))
- RoPE: 相对位置编码，作用在 Q/K 上 (旋转矩阵 R_m)
- ALiBi: 无位置编码，在注意力分数上加偏置 (score -= m * |i-j|)

参考:
- Vaswani et al. (2017). Attention Is All You Need.
- Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- Press et al. (2022). Train Short, Test Long: Attention with Linear Biases.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


# ============================================================================
# Sinusoidal 位置编码
# ============================================================================

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    生成 Sinusoidal 位置编码

    公式:
        PE_{(pos, 2i)} = sin(pos / 10000^{2i/d})
        PE_{(pos, 2i+1)} = cos(pos / 10000^{2i/d})

    关键性质:
        PE_{pos+k} 可以表示为 PE_{pos} 的线性变换（旋转矩阵）

    Args:
        max_len: 最大序列长度
        d_model: 嵌入维度

    Returns:
        位置编码矩阵 [max_len, d_model]
    """
    # 位置索引 [max_len, 1]
    position = torch.arange(max_len).unsqueeze(1).float()

    # 计算分母: 10000^(2i/d) = exp(2i * ln(10000) / d)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )  # [d_model/2]

    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维: sin
    pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维: cos

    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal 位置编码模块

    将位置编码加到输入嵌入上: x = x + PE

    Args:
        d_model: 嵌入维度
        max_len: 最大序列长度
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = sinusoidal_positional_encoding(max_len, d_model)
        # [1, max_len, d_model] 方便广播
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# RoPE (Rotary Position Embedding)
# ============================================================================

def precompute_rope_freqs(dim: int, max_len: int, base: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    预计算 RoPE 的频率参数

    频率公式: θ_i = 1 / base^{2i/dim}

    Args:
        dim: 头维度（必须为偶数）
        max_len: 最大序列长度
        base: 频率基数

    Returns:
        (cos_freqs, sin_freqs): 各 [max_len, dim]
    """
    # θ_i = 1 / 10000^{2i/d}
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    # 位置 × 频率: [max_len] × [dim/2] → [max_len, dim/2]
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, inv_freq)

    # 复制使得 [max_len, dim]
    emb = torch.cat([freqs, freqs], dim=-1)

    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将输入向量分成两半并旋转

    rotate_half([x1, x2]) = [-x2, x1]
    对应二维旋转的反时针方向

    Args:
        x: [..., dim] (dim 为偶数)
    Returns:
        [..., dim]
    """
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对 Query 和 Key 应用 RoPE

    核心操作:
        q_rotated = q * cos + rotate_half(q) * sin
        k_rotated = k * cos + rotate_half(k) * sin

    数学等价于对每对维度 (x_{2i}, x_{2i+1}) 应用旋转矩阵 R(m*θ_i)

    Args:
        q: [batch, n_heads, seq_len, head_dim]
        k: [batch, n_heads, seq_len, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]

    Returns:
        (q_embed, k_embed): 旋转后的 Q 和 K
    """
    # 扩展维度: [seq_len, dim] → [1, 1, seq_len, dim]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class RotaryPositionalEmbedding(nn.Module):
    """
    RoPE 模块

    将旋转位置编码应用到输入张量上。
    RoPE 的核心性质: (R_m q)^T (R_n k) = q^T R_{m-n} k
    即注意力分数只依赖于相对位置 m-n。

    Args:
        dim: 头维度
        max_seq_len: 最大序列长度
        base: 频率基数
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len

        cos_cached, sin_cached = precompute_rope_freqs(dim, max_seq_len, base)
        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        对输入应用 RoPE

        Args:
            x: [..., seq_len, dim]
            seq_len: 可选的序列长度覆盖

        Returns:
            旋转后的张量
        """
        if seq_len is None:
            seq_len = x.shape[-2]

        # 动态扩展缓存
        if seq_len > self.max_seq_len:
            cos, sin = precompute_rope_freqs(self.dim, seq_len, device=x.device)
        else:
            cos = self.cos_cached[:seq_len]
            sin = self.sin_cached[:seq_len]

        return x * cos + rotate_half(x) * sin


# ============================================================================
# ALiBi (Attention with Linear Biases)
# ============================================================================

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    计算 ALiBi 的头斜率

    斜率公式: m_h = 1 / 2^{h * 8/n_heads}

    例如 8 个头: [1/2, 1/4, 1/8, ..., 1/256]

    Args:
        n_heads: 注意力头数量

    Returns:
        [n_heads] 的斜率张量
    """
    return 1.0 / (2.0 ** (torch.arange(1, n_heads + 1) * 8.0 / n_heads))


def get_alibi_bias(n_heads: int, seq_len: int, device: str = "cpu") -> torch.Tensor:
    """
    生成 ALiBi 偏置矩阵

    偏置公式: B_{ij} = -m_h * |i - j|
    每个头有不同的斜率 m_h，产生不同的距离衰减速率。

    Args:
        n_heads: 注意力头数量
        seq_len: 序列长度
        device: 计算设备

    Returns:
        [n_heads, seq_len, seq_len] 的偏置矩阵
    """
    slopes = get_alibi_slopes(n_heads).to(device)

    # 相对距离矩阵: |i - j|
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    distance = (rows - cols).abs().float()

    # [n_heads, seq_len, seq_len]
    bias = -slopes.unsqueeze(1).unsqueeze(2) * distance.unsqueeze(0)

    return bias


class ALiBiAttentionBias(nn.Module):
    """
    ALiBi 偏置模块

    不修改 Q/K/V，只在注意力分数上加线性偏置。
    优势: 完美外推（可处理任意长度序列），零额外参数。

    Args:
        n_heads: 注意力头数量
        max_seq_len: 最大序列长度（用于预计算）
    """

    def __init__(self, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        bias = get_alibi_bias(n_heads, max_seq_len)
        self.register_buffer("bias", bias)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        将 ALiBi 偏置加到注意力分数上

        Args:
            scores: [batch, n_heads, seq_len, seq_len] 注意力分数

        Returns:
            加偏置后的注意力分数
        """
        seq_len = scores.shape[-1]
        return scores + self.bias[:, :seq_len, :seq_len]


# ============================================================================
# 统一的 Transformer Embedding 层
# ============================================================================

class TransformerEmbedding(nn.Module):
    """
    完整的 Transformer Embedding 层

    组成:
    1. Token Embedding (查找表)
    2. Position Embedding (可选，取决于位置编码类型)
    3. Dropout

    RoPE 和 ALiBi 不在 Embedding 层应用，而是在注意力层中应用。

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        max_seq_len: 最大序列长度
        dropout: Dropout 概率
        pos_type: 位置编码类型 ("sinusoidal", "learnable", "rope", "alibi")
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pos_type: str = "rope",
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_type = pos_type

        if pos_type == "learnable":
            self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        elif pos_type == "sinusoidal":
            pe = sinusoidal_positional_encoding(max_seq_len, d_model)
            self.register_buffer("pos_embedding", pe)
        # RoPE 和 ALiBi 不在此处应用
        else:
            self.pos_embedding = None

        self.dropout = nn.Dropout(dropout)

        # 初始化: N(0, d^{-0.5})
        nn.init.normal_(self.token_embedding.weight, mean=0, std=d_model ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] 词元 ID
        Returns:
            [batch, seq_len, d_model] 嵌入向量
        """
        seq_len = x.shape[1]
        x = self.token_embedding(x)

        if self.pos_type == "learnable":
            positions = torch.arange(seq_len, device=x.device)
            x = x + self.pos_embedding(positions)
        elif self.pos_type == "sinusoidal":
            x = x + self.pos_embedding[:seq_len]

        return self.dropout(x)
