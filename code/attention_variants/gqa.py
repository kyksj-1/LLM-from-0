"""
分组查询注意力 (Grouped-Query Attention, GQA) 完整实现

GQA 是 MHA 和 MQA 的折中方案：将 Q 头分成 G 组，每组共享一组 KV。
当 G = H 时退化为 MHA，当 G = 1 时退化为 MQA。

数学基础:
- Q_h = X W_Q^(h), h = 1..H
- K_g = X W_K^(g), V_g = X W_V^(g), g = 0..G-1
- 映射: g(h) = floor(h * G / H)
- head_h = softmax(Q_h K_{g(h)}^T / sqrt(d_k)) V_{g(h)}
- KV Cache 压缩比: H/G 倍

参考:
- Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer
  Models from Multi-Head Checkpoints.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from mha import scaled_dot_product_attention


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (GQA)

    将 H 个 Query 头分成 G 组，每组内的 Q 头共享同一组 KV。
    这是 MHA (G=H) 和 MQA (G=1) 的统一推广。

    被 Llama 2/3、Mistral、Gemma 2 等广泛采用。

    Args:
        d_model: 模型维度
        n_heads: Query 头数量 (H)
        n_kv_heads: KV 头/组数量 (G)，必须能整除 n_heads
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        assert n_heads % n_kv_heads == 0, "n_heads 必须能被 n_kv_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # 每组内的 Q 头数

        # Q 投影: H 个独立头
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)

        # K, V 投影: G 组
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)

        # 输出投影
        self.wo = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        将 KV 头重复以匹配 Q 头的数量

        这是 GQA 实现的核心操作：将每组的 KV 复制 n_rep 份。

        Args:
            x: [batch, n_kv_heads, seq_len, head_dim]
            n_rep: 每组内的 Q 头数 (H / G)

        Returns:
            [batch, n_heads, seq_len, head_dim]
        """
        if n_rep == 1:
            return x  # 已经是 MHA，无需重复

        batch, n_kv_heads, seq_len, head_dim = x.shape
        # 先扩展维度，再 reshape
        # [batch, n_kv, seq, head] -> [batch, n_kv, n_rep, seq, head]
        x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        # [batch, n_kv * n_rep, seq, head] = [batch, n_heads, seq, head]
        return x.reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)

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

        # K, V 投影: [batch, seq, d_model] -> [batch, n_kv_heads, seq, head_dim]
        k = self.wk(x)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 处理 KV Cache
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        new_kv_cache = (k, v)

        # 将 KV 头重复以匹配 Q 头数
        k_expanded = self.repeat_kv(k, self.n_rep)
        v_expanded = self.repeat_kv(v, self.n_rep)

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
        return 2 * self.n_kv_heads * self.head_dim


if __name__ == "__main__":
    # 演示: 分组查询注意力的使用
    print("=" * 60)
    print("分组查询注意力 (GQA) 演示")
    print("=" * 60)

    d_model = 512
    n_heads = 8
    batch_size = 2
    seq_len = 16

    # 测试不同的分组配置
    configs = [
        ("MHA (G=8)", 8),
        ("GQA-4 (G=4)", 4),
        ("GQA-2 (G=2)", 2),
        ("MQA (G=1)", 1),
    ]

    print(f"\n模型配置: d_model={d_model}, n_heads={n_heads}")
    print(f"{'配置':<15} {'参数量':>10} {'KV Cache 元素':>15} {'压缩比':>8}")
    print("-" * 55)

    for name, n_kv_heads in configs:
        gqa = GroupedQueryAttention(
            d_model=d_model, n_heads=n_heads, n_kv_heads=n_kv_heads
        )
        kv_cache_size = gqa.kv_cache_elements_per_token()
        base_kv = 2 * n_heads * (d_model // n_heads)  # MHA 的 KV Cache
        compression = base_kv / kv_cache_size
        print(f"{name:<15} {gqa.count_parameters():>10,} {kv_cache_size:>15} {compression:>7.0f}x")

    # 前向传播演示（GQA-4）
    print(f"\n--- GQA-4 前向传播演示 ---")
    gqa = GroupedQueryAttention(d_model=d_model, n_heads=n_heads, n_kv_heads=4)

    x = torch.randn(batch_size, seq_len, d_model)
    output, kv_cache = gqa(x, is_causal=True)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"KV Cache K 形状: {kv_cache[0].shape}")
    print(f"  (注意 n_kv_heads=4 而非 n_heads=8)")

    # 自回归推理
    new_token = torch.randn(batch_size, 1, d_model)
    output_new, kv_cache_new = gqa(new_token, is_causal=True, kv_cache=kv_cache)
    print(f"\n自回归推理:")
    print(f"  新 token 输出形状: {output_new.shape}")
    print(f"  KV Cache 增长: {kv_cache[0].shape[2]} -> {kv_cache_new[0].shape[2]} tokens")
