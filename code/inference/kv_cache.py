"""
KV Cache 实现

本模块实现了 LLM 推理中的 KV Cache 管理器，支持：
- 预分配模式：一次性分配最大长度的缓存空间
- 动态增长模式：按需追加缓存空间
- 显存分析：计算不同配置下的 KV Cache 大小

KV Cache 是自回归推理的核心优化技术。在 Decode 阶段，每生成一个新 token
需要与之前所有 token 的 Key 和 Value 做注意力运算。KV Cache 缓存已计算的
K 和 V 张量，避免重复计算，将生成 T 个 token 的复杂度从 O(T^2 d) 降为 O(Td)。

显存公式:
    M_KV = 2 * L * n_kv_heads * head_dim * seq_len * batch * dtype_size
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class KVCacheConfig:
    """KV Cache 配置"""
    n_layers: int       # Transformer 层数
    n_kv_heads: int     # KV 注意力头数
    head_dim: int       # 每头维度
    max_seq_len: int    # 最大序列长度
    dtype: torch.dtype = torch.float16  # 数据类型


class KVCache:
    """
    KV Cache 管理器

    支持预分配和动态增长两种模式。预分配模式在初始化时就分配最大长度的空间，
    适合延迟敏感的场景；动态增长模式按需扩展，适合显存受限的场景。

    Args:
        config: KV Cache 配置
        batch_size: 批大小
        device: 设备（cpu 或 cuda）
        mode: 缓存模式，"preallocate" 或 "dynamic"
    """

    def __init__(
        self,
        config: KVCacheConfig,
        batch_size: int = 1,
        device: str = "cpu",
        mode: str = "preallocate",
    ):
        self.config = config
        self.batch_size = batch_size
        self.device = device
        self.mode = mode
        self.seq_len = 0  # 当前缓存的序列长度

        if mode == "preallocate":
            # 预分配：一次性分配最大长度的空间
            # shape: [n_layers][batch, n_kv_heads, max_seq_len, head_dim]
            self.k_cache = [
                torch.zeros(
                    batch_size, config.n_kv_heads, config.max_seq_len, config.head_dim,
                    dtype=config.dtype, device=device
                )
                for _ in range(config.n_layers)
            ]
            self.v_cache = [
                torch.zeros(
                    batch_size, config.n_kv_heads, config.max_seq_len, config.head_dim,
                    dtype=config.dtype, device=device
                )
                for _ in range(config.n_layers)
            ]
        elif mode == "dynamic":
            # 动态增长：初始为空列表
            self.k_cache = [None for _ in range(config.n_layers)]
            self.v_cache = [None for _ in range(config.n_layers)]
        else:
            raise ValueError(f"不支持的模式: {mode}，请选择 'preallocate' 或 'dynamic'")

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将新的 K, V 追加到缓存，并返回完整的 K, V

        Args:
            layer_idx: 层索引
            new_k: 新的 Key 张量 [batch, n_kv_heads, n_new, head_dim]
            new_v: 新的 Value 张量 [batch, n_kv_heads, n_new, head_dim]

        Returns:
            (cached_k, cached_v): 完整的 K 和 V 张量
        """
        n_new = new_k.shape[2]  # 新增的 token 数

        if self.mode == "preallocate":
            # 预分配模式：写入预留的位置
            start = self.seq_len if layer_idx == 0 else self.seq_len
            end = start + n_new

            self.k_cache[layer_idx][:, :, start:end, :] = new_k
            self.v_cache[layer_idx][:, :, start:end, :] = new_v

            # 返回有效部分
            cached_k = self.k_cache[layer_idx][:, :, :end, :]
            cached_v = self.v_cache[layer_idx][:, :, :end, :]

        elif self.mode == "dynamic":
            # 动态增长模式：拼接新的 KV
            if self.k_cache[layer_idx] is None:
                self.k_cache[layer_idx] = new_k
                self.v_cache[layer_idx] = new_v
            else:
                self.k_cache[layer_idx] = torch.cat(
                    [self.k_cache[layer_idx], new_k], dim=2
                )
                self.v_cache[layer_idx] = torch.cat(
                    [self.v_cache[layer_idx], new_v], dim=2
                )
            cached_k = self.k_cache[layer_idx]
            cached_v = self.v_cache[layer_idx]

        # 只在最后一层更新 seq_len（避免重复计数）
        if layer_idx == self.config.n_layers - 1:
            self.seq_len += n_new

        return cached_k, cached_v

    def get(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """获取指定层的当前 KV 缓存"""
        if self.mode == "preallocate":
            if self.seq_len == 0:
                return None, None
            return (
                self.k_cache[layer_idx][:, :, :self.seq_len, :],
                self.v_cache[layer_idx][:, :, :self.seq_len, :],
            )
        else:
            return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def memory_usage_bytes(self) -> int:
        """计算当前 KV Cache 的实际显存占用（字节）"""
        dtype_size = 2 if self.config.dtype == torch.float16 else 4
        if self.mode == "preallocate":
            # 预分配模式：按最大长度计算
            per_layer = (2 * self.batch_size * self.config.n_kv_heads
                        * self.config.max_seq_len * self.config.head_dim * dtype_size)
            return per_layer * self.config.n_layers
        else:
            # 动态模式：按实际长度计算
            per_layer = (2 * self.batch_size * self.config.n_kv_heads
                        * self.seq_len * self.config.head_dim * dtype_size)
            return per_layer * self.config.n_layers

    def reset(self):
        """重置缓存"""
        self.seq_len = 0
        if self.mode == "preallocate":
            for i in range(self.config.n_layers):
                self.k_cache[i].zero_()
                self.v_cache[i].zero_()
        else:
            self.k_cache = [None for _ in range(self.config.n_layers)]
            self.v_cache = [None for _ in range(self.config.n_layers)]


class SimpleAttentionWithCache(nn.Module):
    """
    带 KV Cache 的简单注意力层

    用于演示 KV Cache 如何集成到注意力计算中。
    """

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = n_heads // self.n_kv_heads  # GQA 时的重复倍数

        # 投影矩阵
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """GQA: 将 KV 头复制以匹配 Q 头数"""
        if self.n_rep == 1:
            return x
        bs, n_kv_heads, seq_len, head_dim = x.shape
        return (
            x[:, :, None, :, :]
            .expand(bs, n_kv_heads, self.n_rep, seq_len, head_dim)
            .reshape(bs, n_kv_heads * self.n_rep, seq_len, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        """
        带 KV Cache 的注意力前向传播

        Args:
            x: 输入张量 [batch, seq_len, d_model]
            kv_cache: KV Cache 管理器（None 则不使用缓存）
            layer_idx: 当前层索引

        Returns:
            输出张量 [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape

        # 计算 Q, K, V
        q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 转置为 [batch, heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 使用 KV Cache
        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # GQA: 复制 KV 头
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # 计算注意力
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # 因果掩码（只对新 token 相关的部分）
        kv_len = k.shape[2]
        q_len = q.shape[2]
        # 创建因果掩码
        causal_mask = torch.triu(
            torch.ones(q_len, kv_len, device=x.device, dtype=torch.bool),
            diagonal=kv_len - q_len + 1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(out)


def benchmark_kv_cache(
    d_model: int = 512,
    n_heads: int = 8,
    n_kv_heads: int = 8,
    n_layers: int = 6,
    max_seq_len: int = 512,
    gen_length: int = 128,
    device: str = "cpu",
) -> dict:
    """
    基准测试：对比有无 KV Cache 的推理速度

    Args:
        d_model: 隐藏维度
        n_heads: Q 头数
        n_kv_heads: KV 头数
        n_layers: 层数
        max_seq_len: 最大序列长度
        gen_length: 生成的 token 数
        device: 设备

    Returns:
        包含计时结果的字典
    """
    # 创建简单的多层注意力模型
    layers = nn.ModuleList([
        SimpleAttentionWithCache(d_model, n_heads, n_kv_heads)
        for _ in range(n_layers)
    ]).to(device)

    # 初始 prompt
    prompt_len = 32
    prompt = torch.randn(1, prompt_len, d_model, device=device)

    # --- 无 KV Cache: 每步重新计算全部 ---
    start_time = time.perf_counter()
    sequence = prompt.clone()
    for step in range(gen_length):
        x = sequence
        for layer_idx, layer in enumerate(layers):
            x = layer(x)
        # 取最后一个 token 的输出作为新 token 的嵌入（简化）
        new_token = x[:, -1:, :]
        sequence = torch.cat([sequence, new_token], dim=1)
    time_no_cache = time.perf_counter() - start_time

    # --- 有 KV Cache: 增量计算 ---
    cache_config = KVCacheConfig(
        n_layers=n_layers,
        n_kv_heads=n_kv_heads,
        head_dim=d_model // n_heads,
        max_seq_len=max_seq_len,
        dtype=torch.float32,  # 与模型默认 dtype 一致
    )
    kv_cache = KVCache(cache_config, batch_size=1, device=device, mode="preallocate")

    start_time = time.perf_counter()
    # Prefill 阶段：处理完整 prompt
    x = prompt
    for layer_idx, layer in enumerate(layers):
        x = layer(x, kv_cache=kv_cache, layer_idx=layer_idx)
    last_token = x[:, -1:, :]

    # Decode 阶段：每次只输入新 token
    for step in range(gen_length):
        x = last_token
        for layer_idx, layer in enumerate(layers):
            x = layer(x, kv_cache=kv_cache, layer_idx=layer_idx)
        last_token = x  # 输出就是下一个 token 的嵌入
    time_with_cache = time.perf_counter() - start_time

    results = {
        "gen_length": gen_length,
        "time_no_cache_ms": time_no_cache * 1000,
        "time_with_cache_ms": time_with_cache * 1000,
        "speedup": time_no_cache / time_with_cache if time_with_cache > 0 else float("inf"),
        "cache_memory_bytes": kv_cache.memory_usage_bytes(),
    }
    return results


if __name__ == "__main__":
    print("=" * 60)
    print("KV Cache 实现演示")
    print("=" * 60)

    # 1. 基本功能演示
    print("\n--- 基本功能演示 ---")
    config = KVCacheConfig(n_layers=2, n_kv_heads=4, head_dim=64, max_seq_len=256)

    # 预分配模式
    cache_pre = KVCache(config, batch_size=1, mode="preallocate")
    print(f"预分配模式初始显存: {cache_pre.memory_usage_bytes() / 1024:.2f} KB")

    # 模拟 Prefill: 追加 16 个 token 的 KV
    new_k = torch.randn(1, 4, 16, 64)
    new_v = torch.randn(1, 4, 16, 64)
    for i in range(config.n_layers):
        k, v = cache_pre.update(i, new_k, new_v)
    print(f"Prefill 后 KV 形状: K={k.shape}, V={v.shape}")
    print(f"当前序列长度: {cache_pre.seq_len}")

    # 模拟 Decode: 每次追加 1 个 token
    new_k = torch.randn(1, 4, 1, 64)
    new_v = torch.randn(1, 4, 1, 64)
    for i in range(config.n_layers):
        k, v = cache_pre.update(i, new_k, new_v)
    print(f"Decode 1 步后 KV 形状: K={k.shape}, V={v.shape}")
    print(f"当前序列长度: {cache_pre.seq_len}")

    # 动态增长模式
    cache_dyn = KVCache(config, batch_size=1, mode="dynamic")
    print(f"\n动态模式初始显存: {cache_dyn.memory_usage_bytes()} bytes")
    new_k = torch.randn(1, 4, 16, 64)
    new_v = torch.randn(1, 4, 16, 64)
    for i in range(config.n_layers):
        k, v = cache_dyn.update(i, new_k, new_v)
    print(f"动态模式 Prefill 后显存: {cache_dyn.memory_usage_bytes() / 1024:.2f} KB")

    # 2. 显存分析
    print("\n--- 不同模型的 KV Cache 显存分析 ---")
    model_configs = {
        "7B (MHA, 32 heads)": KVCacheConfig(
            n_layers=32, n_kv_heads=32, head_dim=128, max_seq_len=4096
        ),
        "7B (GQA, 8 KV heads)": KVCacheConfig(
            n_layers=32, n_kv_heads=8, head_dim=128, max_seq_len=4096
        ),
        "70B (GQA, 8 KV heads)": KVCacheConfig(
            n_layers=80, n_kv_heads=8, head_dim=128, max_seq_len=4096
        ),
    }

    for name, cfg in model_configs.items():
        cache = KVCache(cfg, batch_size=1, mode="preallocate")
        mem_gb = cache.memory_usage_bytes() / (1024 ** 3)
        print(f"  {name}: {mem_gb:.3f} GB")

    # 3. 速度基准测试
    print("\n--- KV Cache 速度基准测试 ---")
    for gen_len in [32, 64, 128]:
        results = benchmark_kv_cache(
            d_model=256, n_heads=4, n_kv_heads=4,
            n_layers=4, max_seq_len=512,
            gen_length=gen_len, device="cpu"
        )
        print(f"\n  生成长度={gen_len}:")
        print(f"    无 Cache: {results['time_no_cache_ms']:.1f} ms")
        print(f"    有 Cache: {results['time_with_cache_ms']:.1f} ms")
        print(f"    加速比: {results['speedup']:.2f}x")
