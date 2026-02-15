"""
KV Cache 实现与管理

本模块提供了通用的 KV Cache 管理器，支持不同的注意力变体
（MHA, MQA, GQA, MLA），并提供显存分析工具。

KV Cache 是自回归推理的核心优化：缓存已计算的 Key 和 Value，
避免在生成每个新 token 时重复计算之前所有 token 的 KV。

显存公式:
- MHA: M_kv = 2 * n_layers * n_heads * d_head * seq_len * dtype_size
- GQA: M_kv = 2 * n_layers * n_kv_heads * d_head * seq_len * dtype_size
- MLA: M_kv = n_layers * (d_c + d_r) * seq_len * dtype_size
"""

import torch
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class KVCacheConfig:
    """
    KV Cache 配置

    描述模型的关键参数，用于计算 KV Cache 的理论大小。
    """
    n_layers: int           # Transformer 层数
    n_heads: int            # Q 头数
    n_kv_heads: int         # KV 头数 (MHA: =n_heads, GQA: <n_heads, MQA: =1)
    head_dim: int           # 每头维度
    max_seq_len: int        # 最大序列长度
    dtype_bytes: int = 2    # 数据类型字节数 (FP16=2, FP32=4, FP8=1)
    d_compress: int = 0     # MLA 压缩维度 (仅 MLA 使用)
    d_rope: int = 0         # MLA RoPE 维度 (仅 MLA 使用)
    attn_type: str = "mha"  # 注意力类型: "mha", "mqa", "gqa", "mla"


class KVCacheAnalyzer:
    """
    KV Cache 显存分析器

    计算不同注意力变体在不同配置下的 KV Cache 大小，
    帮助理解推理系统的显存瓶颈。
    """

    @staticmethod
    def elements_per_token_per_layer(config: KVCacheConfig) -> int:
        """
        计算每层每 token 的 KV Cache 元素数

        Args:
            config: KV Cache 配置

        Returns:
            KV Cache 元素数（不含 dtype 大小）
        """
        if config.attn_type == "mla":
            # MLA: 缓存压缩向量 + RoPE key
            return config.d_compress + config.d_rope
        else:
            # MHA/GQA/MQA: 缓存 K 和 V（每个 KV 头各一份）
            return 2 * config.n_kv_heads * config.head_dim

    @staticmethod
    def total_bytes(
        config: KVCacheConfig,
        seq_len: int,
        batch_size: int = 1,
    ) -> int:
        """
        计算 KV Cache 的总字节数

        Args:
            config: KV Cache 配置
            seq_len: 当前序列长度
            batch_size: 批大小

        Returns:
            总字节数
        """
        elements = KVCacheAnalyzer.elements_per_token_per_layer(config)
        return elements * config.n_layers * seq_len * batch_size * config.dtype_bytes

    @staticmethod
    def total_gb(
        config: KVCacheConfig,
        seq_len: int,
        batch_size: int = 1,
    ) -> float:
        """
        计算 KV Cache 的总 GB 数

        Args:
            config: KV Cache 配置
            seq_len: 当前序列长度
            batch_size: 批大小

        Returns:
            总 GB 数
        """
        return KVCacheAnalyzer.total_bytes(config, seq_len, batch_size) / (1024 ** 3)

    @staticmethod
    def compare_variants(
        d_model: int = 4096,
        n_heads: int = 32,
        n_layers: int = 32,
        head_dim: int = 128,
        seq_len: int = 4096,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """
        对比不同注意力变体的 KV Cache 大小

        Args:
            d_model: 模型维度
            n_heads: Q 头数
            n_layers: 层数
            head_dim: 每头维度
            seq_len: 序列长度
            batch_size: 批大小

        Returns:
            各变体的 KV Cache GB 数
        """
        configs = {
            "MHA": KVCacheConfig(
                n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_heads,
                head_dim=head_dim, max_seq_len=seq_len, attn_type="mha",
            ),
            "GQA-8": KVCacheConfig(
                n_layers=n_layers, n_heads=n_heads, n_kv_heads=8,
                head_dim=head_dim, max_seq_len=seq_len, attn_type="gqa",
            ),
            "GQA-4": KVCacheConfig(
                n_layers=n_layers, n_heads=n_heads, n_kv_heads=4,
                head_dim=head_dim, max_seq_len=seq_len, attn_type="gqa",
            ),
            "MQA": KVCacheConfig(
                n_layers=n_layers, n_heads=n_heads, n_kv_heads=1,
                head_dim=head_dim, max_seq_len=seq_len, attn_type="mqa",
            ),
            "MLA": KVCacheConfig(
                n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_heads,
                head_dim=head_dim, max_seq_len=seq_len,
                d_compress=512, d_rope=64, attn_type="mla",
            ),
        }

        results = {}
        for name, config in configs.items():
            results[name] = KVCacheAnalyzer.total_gb(config, seq_len, batch_size)

        return results


class SimpleKVCache:
    """
    简单的 KV Cache 实现

    为自回归推理提供 KV 缓存管理。支持预分配内存和动态增长两种模式。

    用法示例:
        cache = SimpleKVCache(n_layers=32)
        for layer_idx in range(n_layers):
            k, v = compute_kv(x)
            k, v = cache.update(layer_idx, k, v)
            # k, v 现在包含了之前所有 token 的 KV
    """

    def __init__(self, n_layers: int):
        """
        Args:
            n_layers: Transformer 层数
        """
        self.n_layers = n_layers
        # 每层独立存储 K 和 V
        self.cache_k: List[Optional[torch.Tensor]] = [None] * n_layers
        self.cache_v: List[Optional[torch.Tensor]] = [None] * n_layers

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新指定层的 KV Cache

        将新的 K/V 追加到缓存中，返回包含历史的完整 K/V。

        Args:
            layer_idx: 层索引
            new_k: [batch, n_kv_heads, seq_new, head_dim] 新的 Key
            new_v: [batch, n_kv_heads, seq_new, head_dim] 新的 Value

        Returns:
            (full_k, full_v) 包含历史的完整 K/V
        """
        if self.cache_k[layer_idx] is None:
            # 第一次调用，直接存储
            self.cache_k[layer_idx] = new_k
            self.cache_v[layer_idx] = new_v
        else:
            # 追加到已有缓存
            self.cache_k[layer_idx] = torch.cat(
                [self.cache_k[layer_idx], new_k], dim=2
            )
            self.cache_v[layer_idx] = torch.cat(
                [self.cache_v[layer_idx], new_v], dim=2
            )

        return self.cache_k[layer_idx], self.cache_v[layer_idx]

    def get_seq_len(self, layer_idx: int = 0) -> int:
        """获取当前缓存的序列长度"""
        if self.cache_k[layer_idx] is None:
            return 0
        return self.cache_k[layer_idx].shape[2]

    def clear(self):
        """清空所有缓存"""
        self.cache_k = [None] * self.n_layers
        self.cache_v = [None] * self.n_layers

    def memory_bytes(self) -> int:
        """计算当前缓存占用的字节数"""
        total = 0
        for k, v in zip(self.cache_k, self.cache_v):
            if k is not None:
                total += k.nelement() * k.element_size()
            if v is not None:
                total += v.nelement() * v.element_size()
        return total

    def memory_mb(self) -> float:
        """计算当前缓存占用的 MB 数"""
        return self.memory_bytes() / (1024 ** 2)


if __name__ == "__main__":
    print("=" * 60)
    print("KV Cache 分析演示")
    print("=" * 60)

    # 场景: 类似 Llama 2 70B 的配置
    print("\n--- 场景: Llama 2 70B 级别配置 ---")
    print(f"n_layers=80, n_heads=64, head_dim=128, FP16")

    results = KVCacheAnalyzer.compare_variants(
        d_model=8192,
        n_heads=64,
        n_layers=80,
        head_dim=128,
        seq_len=4096,
        batch_size=1,
    )

    print(f"\n{'注意力类型':<10} {'KV Cache (GB)':>15} {'相对 MHA':>10}")
    print("-" * 40)
    mha_gb = results["MHA"]
    for name, gb in results.items():
        ratio = gb / mha_gb
        print(f"{name:<10} {gb:>14.3f} {ratio:>9.1%}")

    # 场景: 不同序列长度
    print("\n--- 场景: KV Cache 随序列长度变化 ---")
    print(f"模型: 32 层, 32 头, head_dim=128, batch=1, FP16")

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    configs = {
        "MHA": KVCacheConfig(
            n_layers=32, n_heads=32, n_kv_heads=32,
            head_dim=128, max_seq_len=32768, attn_type="mha",
        ),
        "GQA-8": KVCacheConfig(
            n_layers=32, n_heads=32, n_kv_heads=8,
            head_dim=128, max_seq_len=32768, attn_type="gqa",
        ),
        "MLA": KVCacheConfig(
            n_layers=32, n_heads=32, n_kv_heads=32,
            head_dim=128, max_seq_len=32768,
            d_compress=512, d_rope=64, attn_type="mla",
        ),
    }

    header = f"{'Seq Len':>8}"
    for name in configs:
        header += f" {name + ' (GB)':>12}"
    print(f"\n{header}")
    print("-" * (8 + 12 * len(configs) + len(configs)))

    for sl in seq_lengths:
        row = f"{sl:>8}"
        for name, config in configs.items():
            gb = KVCacheAnalyzer.total_gb(config, sl, batch_size=1)
            row += f" {gb:>11.3f}"
        print(row)

    # 演示: SimpleKVCache 的使用
    print("\n--- SimpleKVCache 使用演示 ---")
    cache = SimpleKVCache(n_layers=4)

    batch_size = 2
    n_kv_heads = 4
    head_dim = 64

    # 模拟 Prefill: 处理 8 个 token
    for layer_idx in range(4):
        k = torch.randn(batch_size, n_kv_heads, 8, head_dim)
        v = torch.randn(batch_size, n_kv_heads, 8, head_dim)
        full_k, full_v = cache.update(layer_idx, k, v)

    print(f"Prefill 后:")
    print(f"  缓存序列长度: {cache.get_seq_len()}")
    print(f"  缓存占用: {cache.memory_mb():.2f} MB")

    # 模拟 Decode: 逐 token 生成
    for step in range(4):
        for layer_idx in range(4):
            k = torch.randn(batch_size, n_kv_heads, 1, head_dim)
            v = torch.randn(batch_size, n_kv_heads, 1, head_dim)
            full_k, full_v = cache.update(layer_idx, k, v)

    print(f"\nDecode 4 步后:")
    print(f"  缓存序列长度: {cache.get_seq_len()}")
    print(f"  缓存占用: {cache.memory_mb():.2f} MB")
