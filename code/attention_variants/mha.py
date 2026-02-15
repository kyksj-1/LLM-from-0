"""
标准多头注意力 (Multi-Head Attention, MHA) 完整实现

本模块实现了标准的多头注意力机制，是所有注意力变体的基准参考实现。
包含完整的 KV Cache 支持，用于自回归推理。

数学基础:
- Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- MultiHead = Concat(head_1, ..., head_h) W_O
- 每个头独立拥有 Q, K, V 投影

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
        query: [batch, n_heads, seq_q, d_k] 查询张量
        key:   [batch, n_heads, seq_k, d_k] 键张量
        value: [batch, n_heads, seq_k, d_v] 值张量
        mask:  可选掩码，形状可广播到 [batch, n_heads, seq_q, seq_k]
        dropout: 可选 Dropout 层
        is_causal: 是否使用因果掩码（防止关注未来位置）

    Returns:
        (output, attention_weights) 元组
        - output: [batch, n_heads, seq_q, d_v]
        - attention_weights: [batch, n_heads, seq_q, seq_k]
    """
    d_k = query.shape[-1]
    scale = math.sqrt(d_k)

    # 计算注意力分数: QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / scale

    # 因果掩码: 防止看到未来 token
    if is_causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool),
            diagonal=seq_k - seq_q + 1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    # 自定义掩码
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax 归一化: 使权重非负且和为1
    attn_weights = torch.softmax(scores, dim=-1)

    # 处理全 -inf 行（避免 nan）
    attn_weights = attn_weights.nan_to_num(0.0)

    if dropout is not None:
        attn_weights = dropout(attn_weights)

    # 加权求和
    output = torch.matmul(attn_weights, value)

    return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    标准多头注意力 (MHA)

    将 d_model 维投影到 h 个子空间，每个子空间维度 d_k = d_model/h。
    每个头独立计算注意力，然后拼接并投影回 d_model 维。

    MHA 是所有注意力变体（MQA, GQA, MLA）的基准对照。

    Args:
        d_model: 模型维度
        n_heads: 注意力头数量
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q, K, V 投影矩阵（每个头独立）
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        # 输出投影
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        Args:
            x: [batch, seq_len, d_model] 输入张量
            mask: 可选注意力掩码
            is_causal: 是否使用因果掩码
            kv_cache: 可选的 KV 缓存元组 (cached_k, cached_v)
                      用于自回归推理加速

        Returns:
            (output, new_kv_cache) 元组
            - output: [batch, seq_len, d_model]
            - new_kv_cache: 更新后的 KV 缓存 (k, v)
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影
        q = self.wq(x)  # [batch, seq, d_model]
        k = self.wk(x)
        v = self.wv(x)

        # 重塑为多头形式: [batch, seq, d_model] -> [batch, n_heads, seq, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 处理 KV Cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)  # 沿序列维度拼接
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)

        # 注意力计算
        out, _ = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout=self.dropout, is_causal=is_causal
        )

        # 重塑回: [batch, n_heads, seq, head_dim] -> [batch, seq, d_model]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # 输出投影
        return self.wo(out), new_kv_cache

    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters())

    def kv_cache_elements_per_token(self) -> int:
        """计算每层每 token 的 KV Cache 元素数"""
        return 2 * self.n_heads * self.head_dim


if __name__ == "__main__":
    # 演示: 标准多头注意力的使用
    print("=" * 60)
    print("标准多头注意力 (MHA) 演示")
    print("=" * 60)

    # 模型配置
    d_model = 512
    n_heads = 8
    batch_size = 2
    seq_len = 16

    # 创建 MHA 模块
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    print(f"\n模型配置:")
    print(f"  d_model = {d_model}")
    print(f"  n_heads = {n_heads}")
    print(f"  head_dim = {mha.head_dim}")
    print(f"  参数量 = {mha.count_parameters():,}")
    print(f"  每层每 token KV Cache 元素数 = {mha.kv_cache_elements_per_token()}")

    # 前向传播
    x = torch.randn(batch_size, seq_len, d_model)
    output, kv_cache = mha(x, is_causal=True)
    print(f"\n前向传播:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  KV Cache K 形状: {kv_cache[0].shape}")
    print(f"  KV Cache V 形状: {kv_cache[1].shape}")

    # 模拟自回归推理：使用 KV Cache
    print(f"\n自回归推理（使用 KV Cache）:")
    new_token = torch.randn(batch_size, 1, d_model)  # 新的单个 token
    output_new, kv_cache_new = mha(new_token, is_causal=True, kv_cache=kv_cache)
    print(f"  新 token 输入形状: {new_token.shape}")
    print(f"  新 token 输出形状: {output_new.shape}")
    print(f"  更新后 KV Cache K 形状: {kv_cache_new[0].shape}")
    print(f"  KV Cache 增长: {kv_cache[0].shape[2]} -> {kv_cache_new[0].shape[2]} tokens")
