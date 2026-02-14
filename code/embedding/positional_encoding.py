"""
位置编码实现：Sinusoidal + 可学习 + ALiBi

包含多种位置编码方案的从零实现和对比工具

作者: LLM学习教程
模块: 模块2 - Embedding
"""

import torch
import torch.nn as nn
import math
from typing import Optional


def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    生成 Sinusoidal 位置编码

    公式:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        max_len: 最大序列长度
        d_model: 嵌入维度

    Returns:
        位置编码矩阵 [max_len, d_model]
    """
    # 位置索引 [max_len, 1]
    position = torch.arange(max_len).unsqueeze(1).float()

    # 计算频率因子: 10000^(2i/d) = exp(2i * log(10000) / d)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() *
        (-math.log(10000.0) / d_model)
    )  # [d_model/2]

    # 初始化编码矩阵
    pe = torch.zeros(max_len, d_model)

    # 偶数维度用 sin，奇数维度用 cos
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal 位置编码模块

    原始 Transformer (Vaswani et al., 2017) 使用的位置编码方案。
    固定编码，不需要学习参数。
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算并注册为 buffer（不参与梯度更新）
        pe = sinusoidal_positional_encoding(max_len, d_model)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model] 加上位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码

    GPT / BERT 使用的位置编码方案。
    位置编码是可训练的参数。

    优点: 简单灵活
    缺点: 无法外推到训练时未见过的位置
    """

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        return self.dropout(x + self.pos_embedding(positions))


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    计算 ALiBi 的每个注意力头的斜率

    斜率公式: m_h = 1 / 2^(h * 8 / n_heads)

    Args:
        n_heads: 注意力头数量

    Returns:
        [n_heads] 的斜率张量
    """
    # 使用几何序列: 2^(-8/n_heads), 2^(-16/n_heads), ...
    ratio = 2.0 ** (-8.0 / n_heads)
    slopes = torch.tensor([ratio ** i for i in range(1, n_heads + 1)])
    return slopes


def get_alibi_bias(n_heads: int, seq_len: int) -> torch.Tensor:
    """
    生成 ALiBi 偏置矩阵

    bias[h, i, j] = -m_h * |i - j|

    Args:
        n_heads: 注意力头数量
        seq_len: 序列长度

    Returns:
        [n_heads, seq_len, seq_len] 的偏置矩阵
    """
    slopes = get_alibi_slopes(n_heads)  # [n_heads]

    # 相对距离矩阵 [seq_len, seq_len]
    positions = torch.arange(seq_len)
    distance = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs().float()

    # 偏置 = -斜率 * 距离
    # [n_heads, 1, 1] * [1, seq_len, seq_len] -> [n_heads, seq_len, seq_len]
    bias = -slopes.unsqueeze(1).unsqueeze(2) * distance.unsqueeze(0)

    return bias


class TransformerEmbedding(nn.Module):
    """
    统一的 Transformer Embedding 层

    支持多种位置编码方案:
    - 'sinusoidal': 固定正弦位置编码
    - 'learnable': 可学习位置编码
    - 'rope': RoPE（在注意力层内部应用，此处不处理）
    - 'alibi': ALiBi（在注意力层内部应用，此处不处理）
    - 'none': 无位置编码
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        pos_type: str = 'rope'
    ):
        super().__init__()

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # 位置编码
        self.pos_type = pos_type
        if pos_type == 'sinusoidal':
            self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        elif pos_type == 'learnable':
            self.pos_encoding = LearnablePositionalEncoding(d_model, max_seq_len, dropout)
        else:
            # RoPE, ALiBi, None: 位置信息在注意力层处理
            self.pos_encoding = nn.Dropout(dropout)

        # 嵌入缩放（原始Transformer使用 sqrt(d_model) 缩放）
        self.scale = math.sqrt(d_model)

        # 初始化
        nn.init.normal_(self.token_embedding.weight, std=d_model ** -0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len] 的词元ID

        Returns:
            [batch, seq_len, d_model] 的嵌入向量
        """
        # Token Embedding (带缩放)
        x = self.token_embedding(x) * self.scale

        # 应用位置编码
        if self.pos_type in ['sinusoidal', 'learnable']:
            x = self.pos_encoding(x)
        else:
            x = self.pos_encoding(x)  # 仅 dropout

        return x


# ============ 使用示例 ============

if __name__ == "__main__":
    print("=" * 50)
    print("位置编码演示")
    print("=" * 50)

    d_model = 64
    max_len = 100

    # 1. Sinusoidal 位置编码
    print("\n1. Sinusoidal 位置编码")
    pe = sinusoidal_positional_encoding(max_len, d_model)
    print(f"   形状: {pe.shape}")
    print(f"   值范围: [{pe.min():.4f}, {pe.max():.4f}]")

    # 验证相对位置性质
    # PE(pos+k) 应该可以由 PE(pos) 通过线性变换得到
    pos = 10
    k = 5
    pe_pos = pe[pos]
    pe_pos_k = pe[pos + k]

    # 验证: 相邻位置的内积应该相同
    dot_product_1 = torch.dot(pe[0], pe[k])
    dot_product_2 = torch.dot(pe[10], pe[10 + k])
    print(f"   <PE(0), PE({k})>  = {dot_product_1:.4f}")
    print(f"   <PE(10), PE({10+k})> = {dot_product_2:.4f}")
    print(f"   差异: {abs(dot_product_1 - dot_product_2):.6f} (应接近0)")

    # 2. ALiBi 偏置
    print("\n2. ALiBi 偏置")
    n_heads = 8
    seq_len = 16
    alibi = get_alibi_bias(n_heads, seq_len)
    print(f"   形状: {alibi.shape}")
    print(f"   Head 0 斜率: {get_alibi_slopes(n_heads)[0]:.6f}")
    print(f"   Head 7 斜率: {get_alibi_slopes(n_heads)[7]:.6f}")

    # 3. 统一 Embedding 层
    print("\n3. 统一 Embedding 层")
    vocab_size = 1000
    batch_size = 2

    for pos_type in ['sinusoidal', 'learnable', 'rope']:
        emb = TransformerEmbedding(vocab_size, d_model, pos_type=pos_type)
        x = torch.randint(0, vocab_size, (batch_size, 32))
        out = emb(x)
        print(f"   {pos_type:12s}: 输入 {x.shape} -> 输出 {out.shape}")
