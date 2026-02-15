"""
多查询注意力 (Multi-Query Attention, MQA) 完整实现

MQA 的核心思想：所有 Query 头共享同一组 Key 和 Value。
这大幅减少了 KV Cache 的大小，提升推理效率。

数学基础:
- Q_h = X W_Q^(h), h = 1..H  (每个头独立的 Query)
- K = X W_K                    (所有头共享的 Key)
- V = X W_V                    (所有头共享的 Value)
- head_h = softmax(Q_h K^T / sqrt(d_k)) V
- KV Cache 压缩比: H 倍 (H 为头数)

参考:
- Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from mha import scaled_dot_product_attention


class MultiQueryAttention(nn.Module):
    """
    多查询注意力 (MQA)

    与标准 MHA 的区别：
    - Q 仍然有 n_heads 个独立的投影
    - K 和 V 只有 1 个投影，被所有 Q 头共享
    - KV Cache 大小从 2 * n_heads * d_head 降为 2 * d_head

    被 PaLM (Google, 2022) 和 StarCoder 采用。

    Args:
        d_model: 模型维度
        n_heads: Query 头数量
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()

        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q 投影: n_heads 个独立的头（参数量 = d_model * d_model）
        self.wq = nn.Linear(d_model, d_model, bias=False)

        # K, V 投影: 只有 1 个头（参数量 = d_model * head_dim）
        self.wk = nn.Linear(d_model, self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.head_dim, bias=False)

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
            kv_cache: 可选的 KV 缓存 (cached_k, cached_v)

        Returns:
            (output, new_kv_cache) 元组
        """
        batch_size, seq_len, _ = x.shape

        # Q 投影: [batch, seq, d_model] -> [batch, n_heads, seq, head_dim]
        q = self.wq(x)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # K, V 投影: [batch, seq, d_model] -> [batch, 1, seq, head_dim]
        k = self.wk(x).unsqueeze(1)  # [batch, 1, seq, head_dim]
        v = self.wv(x).unsqueeze(1)  # [batch, 1, seq, head_dim]

        # 处理 KV Cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)

        # 将 K, V 广播到所有 Q 头
        # [batch, 1, seq_k, head_dim] -> [batch, n_heads, seq_k, head_dim]
        k_expanded = k.expand(batch_size, self.n_heads, -1, self.head_dim)
        v_expanded = v.expand(batch_size, self.n_heads, -1, self.head_dim)

        # 注意力计算
        out, _ = scaled_dot_product_attention(
            q, k_expanded, v_expanded,
            mask=mask, dropout=self.dropout, is_causal=is_causal
        )

        # 重塑输出
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.wo(out), new_kv_cache

    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters())

    def kv_cache_elements_per_token(self) -> int:
        """计算每层每 token 的 KV Cache 元素数"""
        # MQA 只有 1 组 KV
        return 2 * self.head_dim


if __name__ == "__main__":
    # 演示: 多查询注意力的使用和与 MHA 的对比
    print("=" * 60)
    print("多查询注意力 (MQA) 演示")
    print("=" * 60)

    d_model = 512
    n_heads = 8
    batch_size = 2
    seq_len = 16

    # 创建 MQA 模块
    mqa = MultiQueryAttention(d_model=d_model, n_heads=n_heads)
    print(f"\n模型配置:")
    print(f"  d_model = {d_model}")
    print(f"  n_heads (Q) = {n_heads}")
    print(f"  n_kv_heads = 1 (MQA 特征)")
    print(f"  head_dim = {mqa.head_dim}")
    print(f"  参数量 = {mqa.count_parameters():,}")
    print(f"  每层每 token KV Cache 元素数 = {mqa.kv_cache_elements_per_token()}")

    # 与 MHA 对比
    from mha import MultiHeadAttention
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
    print(f"\n与 MHA 对比:")
    print(f"  MHA 参数量 = {mha.count_parameters():,}")
    print(f"  MQA 参数量 = {mqa.count_parameters():,}")
    print(f"  参数节省 = {1 - mqa.count_parameters() / mha.count_parameters():.1%}")
    print(f"  MHA KV Cache = {mha.kv_cache_elements_per_token()} 元素/层/token")
    print(f"  MQA KV Cache = {mqa.kv_cache_elements_per_token()} 元素/层/token")
    print(f"  KV Cache 压缩比 = {mha.kv_cache_elements_per_token() / mqa.kv_cache_elements_per_token():.0f}x")

    # 前向传播
    x = torch.randn(batch_size, seq_len, d_model)
    output, kv_cache = mqa(x, is_causal=True)
    print(f"\n前向传播:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  KV Cache K 形状: {kv_cache[0].shape}")
    print(f"  KV Cache V 形状: {kv_cache[1].shape}")

    # 自回归推理
    new_token = torch.randn(batch_size, 1, d_model)
    output_new, kv_cache_new = mqa(new_token, is_causal=True, kv_cache=kv_cache)
    print(f"\n自回归推理:")
    print(f"  新 token 输出形状: {output_new.shape}")
    print(f"  KV Cache 增长: {kv_cache[0].shape[2]} -> {kv_cache_new[0].shape[2]} tokens")
