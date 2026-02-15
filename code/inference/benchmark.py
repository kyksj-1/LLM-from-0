"""
推理性能基准测试

本模块提供 LLM 推理相关的性能基准测试工具，包括：
- 延迟测试（TTFT / TPOT）
- 吞吐量测试（tokens/s）
- 显存使用分析
- 量化加速对比

这些工具帮助理解推理系统的性能特征和优化效果。
"""

import torch
import torch.nn as nn
import time
import math
from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    d_model: int = 512          # 隐藏维度
    n_heads: int = 8            # 注意力头数
    n_layers: int = 6           # 层数
    vocab_size: int = 32000     # 词汇表大小
    max_seq_len: int = 1024     # 最大序列长度
    prompt_len: int = 128       # 输入 prompt 长度
    gen_len: int = 64           # 生成长度
    batch_size: int = 1         # 批大小
    n_warmup: int = 2           # 预热轮数
    n_repeats: int = 5          # 重复测量次数


class SimpleLLM(nn.Module):
    """
    简化版 LLM，用于基准测试

    包含标准 Transformer Decoder 的核心组件:
    - Token Embedding
    - 多层 Attention + FFN
    - LM Head

    注意：这是一个简化实现，不包含 KV Cache 等优化，
    用于对比测量基础推理性能。
    """

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.config = config

        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer 层
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 4,
                dropout=0.0,
                batch_first=True,
            )
            for _ in range(config.n_layers)
        ])

        # LM Head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        seq_len = input_ids.shape[1]
        x = self.embed(input_ids)

        # 因果掩码
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )

        for layer in self.layers:
            x = layer(x, x, tgt_mask=causal_mask)

        logits = self.lm_head(x)
        return logits


def measure_latency(
    model: nn.Module,
    input_ids: torch.Tensor,
    n_warmup: int = 2,
    n_repeats: int = 5,
    use_cuda: bool = False,
) -> Dict[str, float]:
    """
    测量模型单次前向传播的延迟

    Args:
        model: 模型
        input_ids: 输入 [batch, seq_len]
        n_warmup: 预热次数
        n_repeats: 测量次数
        use_cuda: 是否使用 CUDA 同步计时

    Returns:
        延迟统计（ms）
    """
    model.eval()

    # 预热
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(input_ids)

    # 测量
    latencies = []
    with torch.no_grad():
        for _ in range(n_repeats):
            if use_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()
                start = time.perf_counter()
                _ = model(input_ids)
                torch.cuda.synchronize()
            else:
                start = time.perf_counter()
                _ = model(input_ids)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # 转为 ms

    return {
        "mean_ms": sum(latencies) / len(latencies),
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "std_ms": (
            sum((x - sum(latencies)/len(latencies))**2 for x in latencies)
            / len(latencies)
        ) ** 0.5,
    }


def benchmark_prefill_vs_decode(config: BenchmarkConfig) -> dict:
    """
    对比 Prefill 和 Decode 阶段的性能差异

    Args:
        config: 基准测试配置

    Returns:
        性能对比结果
    """
    device = "cpu"
    model = SimpleLLM(config).to(device)
    model.eval()

    results = {}

    # Prefill: 处理完整 prompt
    prompt_ids = torch.randint(0, config.vocab_size,
                               (config.batch_size, config.prompt_len))
    prefill_latency = measure_latency(model, prompt_ids, config.n_warmup, config.n_repeats)
    results["prefill"] = {
        "latency_ms": prefill_latency["mean_ms"],
        "tokens_processed": config.prompt_len * config.batch_size,
        "tokens_per_second": (
            config.prompt_len * config.batch_size
            / (prefill_latency["mean_ms"] / 1000)
        ),
    }

    # Decode: 处理单 token（模拟，未使用 KV Cache）
    # 实际 decode 会短得多（因为只处理 1 token + 读取 cache）
    # 这里用 prompt_len+1 个 token 模拟一次完整的 decode step
    decode_ids = torch.randint(0, config.vocab_size,
                               (config.batch_size, config.prompt_len + 1))
    decode_latency = measure_latency(model, decode_ids, config.n_warmup, config.n_repeats)
    # 单 token decode 的近似延迟
    single_decode_ms = (decode_latency["mean_ms"] - prefill_latency["mean_ms"])
    single_decode_ms = max(single_decode_ms, 0.1)  # 防止负值

    results["decode"] = {
        "estimated_per_token_ms": single_decode_ms,
        "tokens_per_second": config.batch_size / (single_decode_ms / 1000),
    }

    # 端到端估算
    total_time = (prefill_latency["mean_ms"]
                  + single_decode_ms * config.gen_len)
    results["end_to_end"] = {
        "estimated_total_ms": total_time,
        "prefill_fraction": prefill_latency["mean_ms"] / total_time,
        "decode_fraction": (single_decode_ms * config.gen_len) / total_time,
    }

    return results


def benchmark_batch_size_effect(config: BenchmarkConfig) -> List[dict]:
    """
    测量不同 batch size 对推理性能的影响

    Args:
        config: 基准测试配置

    Returns:
        不同 batch size 的性能列表
    """
    model = SimpleLLM(config)
    model.eval()

    results = []
    for bs in [1, 2, 4, 8, 16]:
        input_ids = torch.randint(0, config.vocab_size, (bs, config.prompt_len))
        latency = measure_latency(model, input_ids, config.n_warmup, config.n_repeats)

        results.append({
            "batch_size": bs,
            "latency_ms": latency["mean_ms"],
            "throughput_tokens_per_sec": (
                bs * config.prompt_len / (latency["mean_ms"] / 1000)
            ),
            "latency_per_sample_ms": latency["mean_ms"] / bs,
        })

    return results


def benchmark_sequence_length_effect(config: BenchmarkConfig) -> List[dict]:
    """
    测量不同序列长度对推理性能的影响

    Args:
        config: 基准测试配置

    Returns:
        不同序列长度的性能列表
    """
    model = SimpleLLM(config)
    model.eval()

    results = []
    for seq_len in [32, 64, 128, 256, 512]:
        if seq_len > config.max_seq_len:
            continue
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
        latency = measure_latency(model, input_ids, config.n_warmup, config.n_repeats)

        results.append({
            "seq_len": seq_len,
            "latency_ms": latency["mean_ms"],
            "ms_per_token": latency["mean_ms"] / seq_len,
        })

    return results


def estimate_memory_usage(
    n_params: int,
    batch_size: int,
    seq_len: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype_bytes: int = 2,
) -> Dict[str, float]:
    """
    估算推理时的总显存使用

    Args:
        n_params: 模型参数量
        batch_size: 批大小
        seq_len: 序列长度
        n_layers: 层数
        n_kv_heads: KV 头数
        head_dim: 每头维度
        dtype_bytes: 数据类型字节数

    Returns:
        各部分显存使用（GB）
    """
    # 模型权重
    model_memory = n_params * dtype_bytes

    # KV Cache
    kv_cache_memory = (2 * n_layers * n_kv_heads * head_dim
                       * seq_len * batch_size * dtype_bytes)

    # 激活值（近似：前向传播中的中间张量）
    # 推理时不需要保存反向传播的激活，但仍需要一些中间缓冲区
    activation_memory = batch_size * seq_len * (n_kv_heads * head_dim) * dtype_bytes * 4

    total = model_memory + kv_cache_memory + activation_memory

    return {
        "model_weights_gb": model_memory / (1024**3),
        "kv_cache_gb": kv_cache_memory / (1024**3),
        "activations_gb": activation_memory / (1024**3),
        "total_gb": total / (1024**3),
        "kv_cache_fraction": kv_cache_memory / total,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("推理性能基准测试")
    print("=" * 60)

    # 使用较小的配置以便在 CPU 上运行
    config = BenchmarkConfig(
        d_model=256,
        n_heads=4,
        n_layers=4,
        vocab_size=1000,
        max_seq_len=512,
        prompt_len=64,
        gen_len=32,
        batch_size=1,
        n_warmup=1,
        n_repeats=3,
    )

    # 1. Prefill vs Decode 对比
    print("\n--- Prefill vs Decode 性能对比 ---")
    perf = benchmark_prefill_vs_decode(config)
    print(f"  Prefill (prompt_len={config.prompt_len}):")
    print(f"    延迟: {perf['prefill']['latency_ms']:.1f} ms")
    print(f"    吞吐: {perf['prefill']['tokens_per_second']:.0f} tokens/s")
    print(f"  Decode (每 token):")
    print(f"    估算延迟: {perf['decode']['estimated_per_token_ms']:.1f} ms/token")
    print(f"  端到端 (生成 {config.gen_len} tokens):")
    print(f"    总延迟: {perf['end_to_end']['estimated_total_ms']:.0f} ms")
    print(f"    Prefill 占比: {perf['end_to_end']['prefill_fraction']*100:.1f}%")
    print(f"    Decode 占比: {perf['end_to_end']['decode_fraction']*100:.1f}%")

    # 2. Batch Size 影响
    print("\n--- Batch Size 对性能的影响 ---")
    bs_results = benchmark_batch_size_effect(config)
    print(f"{'Batch':<8} {'延迟(ms)':<12} {'吞吐(tok/s)':<15} {'每样本延迟(ms)'}")
    print("-" * 50)
    for r in bs_results:
        print(f"{r['batch_size']:<8} {r['latency_ms']:<12.1f} "
              f"{r['throughput_tokens_per_sec']:<15.0f} "
              f"{r['latency_per_sample_ms']:.1f}")

    # 3. 序列长度影响
    print("\n--- 序列长度对性能的影响 ---")
    sl_results = benchmark_sequence_length_effect(config)
    print(f"{'Seq Len':<10} {'延迟(ms)':<12} {'ms/token'}")
    print("-" * 35)
    for r in sl_results:
        print(f"{r['seq_len']:<10} {r['latency_ms']:<12.1f} {r['ms_per_token']:.3f}")

    # 4. 显存使用估算
    print("\n--- 显存使用估算 ---")
    model_configs = [
        ("Llama 2 7B (MHA, FP16)", 7e9, 32, 32, 128, 2),
        ("Llama 2 7B (MHA, INT8)", 7e9, 32, 32, 128, 1),
        ("Llama 2 7B (MHA, INT4)", 7e9, 32, 32, 128, 0.5),
        ("Llama 2 70B (GQA, FP16)", 70e9, 80, 8, 128, 2),
        ("Llama 3 8B (GQA, FP16)", 8e9, 32, 8, 128, 2),
    ]

    print(f"{'模型':<30} {'权重':<8} {'KV Cache':<10} {'总计':<8} {'KV占比'}")
    print("-" * 65)
    for name, params, n_layers, n_kv_heads, head_dim, dtype_bytes in model_configs:
        # dtype_bytes 为 0.5 时代表 INT4
        effective_dtype = max(dtype_bytes, 1)  # KV Cache 最少 1 byte
        mem = estimate_memory_usage(
            int(params), batch_size=1, seq_len=4096,
            n_layers=n_layers, n_kv_heads=n_kv_heads,
            head_dim=head_dim, dtype_bytes=effective_dtype,
        )
        # INT4 权重单独计算
        if dtype_bytes == 0.5:
            mem["model_weights_gb"] = params * 0.5 / (1024**3)
            mem["total_gb"] = (mem["model_weights_gb"]
                              + mem["kv_cache_gb"] + mem["activations_gb"])

        print(f"{name:<30} {mem['model_weights_gb']:<8.1f} "
              f"{mem['kv_cache_gb']:<10.2f} "
              f"{mem['total_gb']:<8.1f} "
              f"{mem['kv_cache_fraction']*100:.0f}%")

    # 5. 算术强度分析
    print("\n--- Decode 阶段算术强度分析 ---")
    d = 4096  # Llama 7B 的 d_model
    print(f"矩阵-向量乘法 (d={d}):")
    for bs in [1, 8, 32, 128, 512]:
        flops = 2 * d * d * bs
        bytes_moved = (d * d + d * bs + d * bs) * 2  # FP16
        ai = flops / bytes_moved
        a100_roofline = 312e12  # A100 FP16 峰值
        a100_bw = 2e12  # A100 带宽
        ridge_point = a100_roofline / a100_bw  # ~156 FLOPs/Byte
        bound = "访存密集" if ai < ridge_point else "计算密集"
        print(f"  batch={bs:<4d}: AI={ai:.1f} FLOPs/Byte, "
              f"拐点={ridge_point:.0f}, {bound}")
