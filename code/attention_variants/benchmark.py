"""
注意力机制性能基准测试工具

本模块提供了对比 MHA/MQA/GQA/MLA 四种注意力变体的基准测试框架，
测量参数量、KV Cache 大小、前向传播时间等关键指标。

使用方法:
    python benchmark.py

注意: 精确的 GPU 时间测量需要 CUDA 设备。CPU 上的时间仅供定性对比。
"""

import torch
import torch.nn as nn
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

from mha import MultiHeadAttention
from mqa import MultiQueryAttention
from gqa import GroupedQueryAttention
from mla import MultiLatentAttention


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    d_model: int = 1024
    n_heads: int = 16
    head_dim: int = 64       # d_model // n_heads
    n_kv_heads_gqa: int = 4  # GQA 的 KV 头数
    d_compress: int = 256    # MLA 压缩维度
    d_rope: int = 32         # MLA RoPE 维度
    batch_size: int = 4
    seq_len: int = 512
    n_warmup: int = 5        # 预热次数
    n_trials: int = 20       # 测试次数


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    n_params: int
    kv_cache_elements: int
    forward_time_ms: float
    forward_time_std_ms: float


def create_attention_modules(config: BenchmarkConfig) -> Dict[str, nn.Module]:
    """
    创建四种注意力模块用于对比测试

    Args:
        config: 基准测试配置

    Returns:
        {名称: 模块} 字典
    """
    modules = {}

    # 标准 MHA
    modules["MHA"] = MultiHeadAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
    )

    # MQA
    modules["MQA"] = MultiQueryAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
    )

    # GQA
    modules["GQA"] = GroupedQueryAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads_gqa,
    )

    # MLA
    modules["MLA"] = MultiLatentAttention(
        d_model=config.d_model,
        n_heads=config.n_heads,
        head_dim=config.head_dim,
        d_compress=config.d_compress,
        d_rope=config.d_rope,
    )

    return modules


def measure_forward_time(
    module: nn.Module,
    x: torch.Tensor,
    n_warmup: int = 5,
    n_trials: int = 20,
    device: str = "cpu",
) -> Tuple[float, float]:
    """
    测量模块前向传播时间

    Args:
        module: 待测模块
        x: 输入张量
        n_warmup: 预热次数（不计入结果）
        n_trials: 测试次数
        device: 设备类型

    Returns:
        (平均时间 ms, 标准差 ms)
    """
    module = module.to(device)
    x = x.to(device)
    module.eval()

    times = []

    with torch.no_grad():
        # 预热
        for _ in range(n_warmup):
            _ = module(x, is_causal=True)
            if device == "cuda":
                torch.cuda.synchronize()

        # 正式测量
        for _ in range(n_trials):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _ = module(x, is_causal=True)

            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)  # 转换为 ms

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5

    return avg, std


def run_benchmark(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """
    运行完整的基准测试

    Args:
        config: 基准测试配置

    Returns:
        测试结果列表
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # 创建注意力模块
    modules = create_attention_modules(config)

    # 创建输入
    x = torch.randn(config.batch_size, config.seq_len, config.d_model)

    results = []

    for name, module in modules.items():
        print(f"\n测试 {name}...")

        # 参数量
        n_params = sum(p.numel() for p in module.parameters())

        # KV Cache 大小
        kv_elements = module.kv_cache_elements_per_token()

        # 前向传播时间
        avg_time, std_time = measure_forward_time(
            module, x, config.n_warmup, config.n_trials, device
        )

        result = BenchmarkResult(
            name=name,
            n_params=n_params,
            kv_cache_elements=kv_elements,
            forward_time_ms=avg_time,
            forward_time_std_ms=std_time,
        )
        results.append(result)

    return results


def print_results(results: List[BenchmarkResult], config: BenchmarkConfig):
    """
    格式化打印基准测试结果

    Args:
        results: 测试结果列表
        config: 基准测试配置
    """
    print("\n" + "=" * 80)
    print("基准测试结果")
    print("=" * 80)
    print(f"配置: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"batch={config.batch_size}, seq_len={config.seq_len}")
    print("-" * 80)

    # 表头
    header = (
        f"{'方法':<8} "
        f"{'参数量':>12} "
        f"{'KV Cache/token':>15} "
        f"{'KV 压缩比':>10} "
        f"{'前向时间(ms)':>15} "
        f"{'相对速度':>10}"
    )
    print(header)
    print("-" * 80)

    # 找到 MHA 的基准值
    mha_result = next(r for r in results if r.name == "MHA")

    for r in results:
        kv_ratio = mha_result.kv_cache_elements / r.kv_cache_elements
        speed_ratio = mha_result.forward_time_ms / r.forward_time_ms if r.forward_time_ms > 0 else 0

        print(
            f"{r.name:<8} "
            f"{r.n_params:>12,} "
            f"{r.kv_cache_elements:>15} "
            f"{kv_ratio:>9.1f}x "
            f"{r.forward_time_ms:>10.2f} +/- {r.forward_time_std_ms:.2f} "
            f"{speed_ratio:>9.2f}x"
        )


def benchmark_sequence_lengths(config: BenchmarkConfig):
    """
    测试不同序列长度下的 KV Cache 大小

    Args:
        config: 基准测试配置
    """
    print("\n" + "=" * 80)
    print("KV Cache 大小随序列长度变化 (GB, FP16, 32 层, batch=1)")
    print("=" * 80)

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    n_layers = 32
    dtype_bytes = 2  # FP16

    modules = create_attention_modules(config)

    header = f"{'Seq Len':>8}"
    for name in modules:
        header += f" {name:>10}"
    print(header)
    print("-" * (8 + 10 * len(modules) + len(modules)))

    for sl in seq_lengths:
        row = f"{sl:>8}"
        for name, module in modules.items():
            elements = module.kv_cache_elements_per_token()
            # 总字节 = 元素数 * 层数 * 序列长度 * dtype字节
            total_bytes = elements * n_layers * sl * dtype_bytes
            total_gb = total_bytes / (1024 ** 3)
            row += f" {total_gb:>9.3f}"
        print(row)


if __name__ == "__main__":
    print("=" * 80)
    print("注意力机制基准测试")
    print("=" * 80)

    # 默认配置
    config = BenchmarkConfig()
    print(f"\n配置:")
    print(f"  d_model = {config.d_model}")
    print(f"  n_heads = {config.n_heads}")
    print(f"  head_dim = {config.head_dim}")
    print(f"  GQA n_kv_heads = {config.n_kv_heads_gqa}")
    print(f"  MLA d_compress = {config.d_compress}")
    print(f"  MLA d_rope = {config.d_rope}")
    print(f"  batch_size = {config.batch_size}")
    print(f"  seq_len = {config.seq_len}")

    # 运行基准测试
    results = run_benchmark(config)
    print_results(results, config)

    # KV Cache 随序列长度变化
    benchmark_sequence_lengths(config)
