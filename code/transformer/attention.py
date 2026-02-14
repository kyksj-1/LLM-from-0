"""
注意力机制实现：Self-Attention, Multi-Head Attention, Causal Mask

本模块实现了 Transformer 的核心注意力计算。

数学基础:
- Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- MultiHead = Concat(head_1, ..., head_h) W_O
- 缩放因子: Var(q^T k) = d_k, 除以 sqrt(d_k) 使方差归一

参考:
- Vaswani et al. (2017). Attention Is All You Need.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    dropout: Optional[nn.Dropout] = None,
    is_causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    缩放点积注意力

    Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        query: [batch, n_heads, seq_q, d_k]
        key:   [batch, n_heads, seq_k, d_k]
        value: [batch, n_heads, seq_k, d_v]
        mask:  可选掩码
        dropout: 可选 Dropout
        is_causal: 是否使用因果掩码

    Returns:
        (output, attention_weights)
    """
    d_k = query.shape[-1]
    scale = math.sqrt(d_k)

    # QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale

    # 因果掩码: 防止看到未来 token
    if is_causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # 自定义掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax 归一化: 使权重非负且和为1
    attn_weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # 加权求和
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力

    将 d_model 维投影到 h 个子空间，每个子空间维度 d_k = d_model/h。
    每个头独立计算注意力，然后拼接并投影回 d_model 维。

    MultiHead(Q,K,V) = Concat(head_1,...,head_h) W_O
    head_i = Attention(Q W_Q^i, K W_K^i, V W_V^i)

    Args:
        d_model: 模型维度
        n_heads: 注意力头数量
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        assert d_model % n_heads == 0, "d_model必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V 投影 (合并为单个矩阵提高效率)
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        # 输出投影
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, seq_q, d_model]
            key:   [batch, seq_k, d_model]
            value: [batch, seq_k, d_model]
            mask:  可选掩码
            is_causal: 是否使用因果掩码

        Returns:
            [batch, seq_q, d_model]
        """
        batch_size = query.shape[0]
        seq_q = query.shape[1]
        seq_k = key.shape[1]

        # 线性投影
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)

        # 重塑为多头: [batch, seq, d_model] → [batch, n_heads, seq, head_dim]
        q = q.view(batch_size, seq_q, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力计算
        out, attn_weights = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, is_causal=is_causal
        )

        # 重塑回: [batch, n_heads, seq_q, head_dim] → [batch, seq_q, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)

        # 输出投影
        return self.wo(out)
