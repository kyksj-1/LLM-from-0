"""
多头潜在注意力 (Multi-head Latent Attention, MLA) 完整实现

MLA 是 DeepSeek-V2 提出的注意力变体，通过低秩压缩 KV 表示来大幅
降低 KV Cache 的大小，同时保持每个头的独立性。

数学基础:
- KV 压缩: c_KV = W_DKV * x, 其中 d_c << n_h * d_h
- KV 解压缩: k_h = W_UK_h * c_KV, v_h = W_UV_h * c_KV
- 解耦 RoPE: q = [内容部分 ; RoPE(位置部分)]
                k = [内容部分 ; RoPE(位置部分)]
- KV Cache: 仅存储 c_KV 和 RoPE(位置 key)

参考:
- DeepSeek-AI (2024). DeepSeek-V2: A Strong, Economical, and Efficient
  Mixture-of-Experts Language Model.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


def apply_rotary_emb(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码 (RoPE)

    将输入张量的相邻维度配对，应用旋转变换。

    Args:
        x: [..., d] 输入张量，d 必须为偶数
        freqs: [..., d/2] 频率张量（角度）

    Returns:
        旋转后的张量，形状与输入相同
    """
    d = x.shape[-1]
    assert d % 2 == 0, "RoPE 要求维度为偶数"

    # 将 x 拆分为实部和虚部（相邻元素配对）
    x_re = x[..., 0::2]  # 偶数索引
    x_im = x[..., 1::2]  # 奇数索引

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    # 旋转: (x_re + i*x_im) * (cos + i*sin)
    out_re = x_re * cos - x_im * sin
    out_im = x_re * sin + x_im * cos

    # 交错合并
    out = torch.stack([out_re, out_im], dim=-1).flatten(-2)
    return out


def precompute_freqs(dim: int, max_seq_len: int, base: float = 10000.0) -> torch.Tensor:
    """
    预计算 RoPE 频率

    Args:
        dim: 旋转维度（必须为偶数）
        max_seq_len: 最大序列长度
        base: RoPE 基础频率

    Returns:
        [max_seq_len, dim/2] 频率张量
    """
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # [max_seq_len, dim/2]
    return freqs


class MultiLatentAttention(nn.Module):
    """
    多头潜在注意力 (MLA) - DeepSeek-V2

    核心创新：
    1. 将 KV 压缩到低维潜在空间 (d_c << n_h * d_h)
    2. 推理时只缓存低维的 c_KV，大幅降低 KV Cache
    3. 使用解耦 RoPE 兼容位置编码
    4. 每个头通过独立的解压缩矩阵恢复 K/V，保持头的多样性

    Args:
        d_model: 模型维度
        n_heads: 注意力头数量
        head_dim: 每个头的维度
        d_compress: KV 压缩维度 (d_c)
        d_rope: RoPE 维度 (d_r)，用于解耦位置编码
        max_seq_len: 最大序列长度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int = 128,
        d_compress: int = 512,
        d_rope: int = 64,
        max_seq_len: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_compress = d_compress
        self.d_rope = d_rope

        # ---- KV 压缩/解压缩 ----
        # 下投影（压缩）: d_model -> d_c
        self.w_dkv = nn.Linear(d_model, d_compress, bias=False)
        # 上投影（解压缩）: d_c -> n_heads * head_dim (K 和 V 各一个)
        self.w_uk = nn.Linear(d_compress, n_heads * head_dim, bias=False)
        self.w_uv = nn.Linear(d_compress, n_heads * head_dim, bias=False)

        # ---- Q 压缩/解压缩 ----
        # Q 也做低秩压缩（减少训练时的激活值显存）
        self.w_dq = nn.Linear(d_model, d_compress, bias=False)
        self.w_uq = nn.Linear(d_compress, n_heads * head_dim, bias=False)

        # ---- 解耦 RoPE ----
        # 位置信息通过独立的投影和 RoPE 处理
        self.w_qr = nn.Linear(d_model, d_rope, bias=False)  # Q 的位置投影
        self.w_kr = nn.Linear(d_model, d_rope, bias=False)  # K 的位置投影

        # 预计算 RoPE 频率
        self.register_buffer(
            "rope_freqs",
            precompute_freqs(d_rope, max_seq_len),
            persistent=False,
        )

        # ---- 输出投影 ----
        self.wo = nn.Linear(n_heads * head_dim, d_model, bias=False)

        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else None

        # 注意力缩放因子: 考虑内容维度和位置维度
        self.scale = (head_dim + d_rope) ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        start_pos: int = 0,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        Args:
            x: [batch, seq_len, d_model] 输入张量
            mask: 可选注意力掩码
            is_causal: 是否使用因果掩码
            kv_cache: 可选的 KV 缓存 (cached_c_kv, cached_k_rope)
                      注意：MLA 缓存的是压缩表示，不是完整的 K/V
            start_pos: 序列起始位置（用于 RoPE）

        Returns:
            (output, new_kv_cache) 元组
            - output: [batch, seq_len, d_model]
            - new_kv_cache: (c_kv, k_rope) 压缩后的 KV 缓存
        """
        batch_size, seq_len, _ = x.shape

        # ---- 步骤 1: KV 压缩 ----
        # [batch, seq, d_model] -> [batch, seq, d_c]
        c_kv = self.w_dkv(x)

        # ---- 步骤 2: 解耦 RoPE 的位置部分 ----
        # K 的位置投影 + RoPE
        k_rope = self.w_kr(x)  # [batch, seq, d_rope]
        freqs = self.rope_freqs[start_pos: start_pos + seq_len]  # [seq, d_rope/2]
        k_rope = apply_rotary_emb(k_rope, freqs.unsqueeze(0))

        # ---- 步骤 3: 处理 KV Cache ----
        if kv_cache is not None:
            cached_c_kv, cached_k_rope = kv_cache
            c_kv_full = torch.cat([cached_c_kv, c_kv], dim=1)
            k_rope_full = torch.cat([cached_k_rope, k_rope], dim=1)
        else:
            c_kv_full = c_kv
            k_rope_full = k_rope

        new_kv_cache = (c_kv_full, k_rope_full)

        # ---- 步骤 4: KV 解压缩 ----
        # [batch, seq_k, d_c] -> [batch, seq_k, n_heads * head_dim]
        k_content = self.w_uk(c_kv_full)
        v = self.w_uv(c_kv_full)

        # 重塑: [batch, seq_k, n_heads * head_dim] -> [batch, n_heads, seq_k, head_dim]
        seq_k = c_kv_full.shape[1]
        k_content = k_content.view(batch_size, seq_k, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_k, self.n_heads, self.head_dim).transpose(1, 2)

        # ---- 步骤 5: Q 压缩 + 解压缩 ----
        c_q = self.w_dq(x)  # [batch, seq, d_c]
        q_content = self.w_uq(c_q)  # [batch, seq, n_heads * head_dim]
        q_content = q_content.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # Q 的位置投影 + RoPE
        q_rope = self.w_qr(x)  # [batch, seq, d_rope]
        q_rope = apply_rotary_emb(q_rope, freqs.unsqueeze(0))

        # ---- 步骤 6: 拼接内容和位置部分 ----
        # Q: [batch, n_heads, seq_q, head_dim + d_rope]
        q_rope_expanded = q_rope.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        q = torch.cat([q_content, q_rope_expanded], dim=-1)

        # K: [batch, n_heads, seq_k, head_dim + d_rope]
        k_rope_expanded = k_rope_full.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        k = torch.cat([k_content, k_rope_expanded], dim=-1)

        # ---- 步骤 7: 注意力计算 ----
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            seq_q_len = scores.shape[-2]
            seq_k_len = scores.shape[-1]
            causal_mask = torch.triu(
                torch.ones(seq_q_len, seq_k_len, device=x.device, dtype=torch.bool),
                diagonal=seq_k_len - seq_q_len + 1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = attn_weights.nan_to_num(0.0)

        if self.dropout_layer is not None:
            attn_weights = self.dropout_layer(attn_weights)

        # [batch, n_heads, seq_q, seq_k] @ [batch, n_heads, seq_k, head_dim]
        out = torch.matmul(attn_weights, v)

        # ---- 步骤 8: 输出投影 ----
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.wo(out)

        return out, new_kv_cache

    def count_parameters(self) -> int:
        """统计参数量"""
        return sum(p.numel() for p in self.parameters())

    def kv_cache_elements_per_token(self) -> int:
        """
        计算每层每 token 的 KV Cache 元素数

        MLA 缓存: c_KV (d_c 维) + k_rope (d_r 维)
        """
        return self.d_compress + self.d_rope


if __name__ == "__main__":
    # 演示: 多头潜在注意力的使用
    print("=" * 60)
    print("多头潜在注意力 (MLA) 演示 - DeepSeek-V2 风格")
    print("=" * 60)

    # 模型配置（缩小版，用于演示）
    d_model = 512
    n_heads = 8
    head_dim = 64
    d_compress = 128  # KV 压缩维度
    d_rope = 32       # RoPE 维度
    batch_size = 2
    seq_len = 16

    mla = MultiLatentAttention(
        d_model=d_model,
        n_heads=n_heads,
        head_dim=head_dim,
        d_compress=d_compress,
        d_rope=d_rope,
    )

    print(f"\n模型配置:")
    print(f"  d_model = {d_model}")
    print(f"  n_heads = {n_heads}")
    print(f"  head_dim = {head_dim}")
    print(f"  d_compress (d_c) = {d_compress}")
    print(f"  d_rope (d_r) = {d_rope}")
    print(f"  参数量 = {mla.count_parameters():,}")
    print(f"  每层每 token KV Cache 元素数 = {mla.kv_cache_elements_per_token()}")

    # 对比 MHA 的 KV Cache
    mha_kv = 2 * n_heads * head_dim
    mla_kv = mla.kv_cache_elements_per_token()
    print(f"\nKV Cache 对比:")
    print(f"  MHA: {mha_kv} 元素/层/token")
    print(f"  MLA: {mla_kv} 元素/层/token")
    print(f"  压缩比: {mha_kv / mla_kv:.1f}x")

    # 前向传播
    x = torch.randn(batch_size, seq_len, d_model)
    output, kv_cache = mla(x, is_causal=True)
    print(f"\n前向传播:")
    print(f"  输入形状: {x.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  KV Cache c_kv 形状: {kv_cache[0].shape}")
    print(f"  KV Cache k_rope 形状: {kv_cache[1].shape}")

    # 自回归推理
    print(f"\n自回归推理（使用 KV Cache）:")
    new_token = torch.randn(batch_size, 1, d_model)
    output_new, kv_cache_new = mla(
        new_token, is_causal=True, kv_cache=kv_cache, start_pos=seq_len
    )
    print(f"  新 token 输出形状: {output_new.shape}")
    print(f"  KV Cache c_kv 增长: {kv_cache[0].shape[1]} -> {kv_cache_new[0].shape[1]} tokens")
    print(f"  KV Cache k_rope 增长: {kv_cache[1].shape[1]} -> {kv_cache_new[1].shape[1]} tokens")
