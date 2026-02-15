"""
推理优化工具函数

本模块提供推理优化相关的通用工具函数，包括：
- 显存计算
- 性能指标
- 可视化辅助
"""

import torch
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple


@dataclass
class ModelConfig:
    """
    模型配置

    用于 KV Cache 显存计算和性能分析的模型参数。
    """
    n_layers: int           # Transformer 层数
    n_heads: int            # Query 注意力头数
    n_kv_heads: int         # KV 注意力头数（MHA: =n_heads, GQA: <n_heads, MQA: =1）
    head_dim: int           # 每头维度
    d_model: int            # 隐藏维度
    vocab_size: int         # 词汇表大小
    max_seq_len: int        # 最大序列长度
    dtype_bytes: int = 2    # 数据类型字节数（FP16=2, FP32=4, INT8=1, FP8=1）

    @property
    def total_params(self) -> int:
        """估算模型总参数量（近似公式 12*L*d^2）"""
        # 注意力层: Q(d^2) + K(d*d_kv) + V(d*d_kv) + O(d^2)
        d_kv = self.n_kv_heads * self.head_dim
        attn_params = self.d_model ** 2 + 2 * self.d_model * d_kv + self.d_model ** 2
        # FFN 层（SwiGLU 风格）: gate(d*d_ff) + up(d*d_ff) + down(d_ff*d)
        d_ff = int(8 * self.d_model / 3)
        ffn_params = 3 * self.d_model * d_ff
        # 总体
        per_layer = attn_params + ffn_params
        total = per_layer * self.n_layers + self.vocab_size * self.d_model
        return total

    @property
    def model_size_gb(self) -> float:
        """模型大小（GB）"""
        return self.total_params * self.dtype_bytes / (1024 ** 3)


# 预置模型配置
LLAMA2_7B = ModelConfig(
    n_layers=32, n_heads=32, n_kv_heads=32, head_dim=128,
    d_model=4096, vocab_size=32000, max_seq_len=4096, dtype_bytes=2
)

LLAMA2_70B = ModelConfig(
    n_layers=80, n_heads=64, n_kv_heads=8, head_dim=128,
    d_model=8192, vocab_size=32000, max_seq_len=4096, dtype_bytes=2
)

LLAMA3_8B = ModelConfig(
    n_layers=32, n_heads=32, n_kv_heads=8, head_dim=128,
    d_model=4096, vocab_size=128000, max_seq_len=8192, dtype_bytes=2
)


def compute_kv_cache_size(
    config: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    计算 KV Cache 的显存大小

    公式: M_KV = 2 * L * n_kv_heads * head_dim * S * b * dtype_size

    Args:
        config: 模型配置
        seq_len: 序列长度
        batch_size: 批大小

    Returns:
        包含各项显存信息的字典（单位为字节和 GB）
    """
    # 每层每 token 的 KV Cache 元素数
    elements_per_token_per_layer = 2 * config.n_kv_heads * config.head_dim

    # 总元素数
    total_elements = (elements_per_token_per_layer
                      * config.n_layers
                      * seq_len
                      * batch_size)

    # 总字节数
    total_bytes = total_elements * config.dtype_bytes

    return {
        "elements_per_token_per_layer": elements_per_token_per_layer,
        "total_elements": total_elements,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / (1024 ** 3),
    }


def compute_arithmetic_intensity(
    m: int, n: int, k: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> float:
    """
    计算矩阵乘法的算术强度

    对于 Y = W @ X:
    - W: [m, k], X: [k, n*batch_size]
    - FLOPs = 2 * m * k * n * batch_size
    - Bytes = (m*k + k*n*batch_size + m*n*batch_size) * dtype_bytes

    Args:
        m: 输出维度
        n: 每个样本的列数（Decode 阶段 n=1）
        k: 内部维度
        batch_size: 批大小
        dtype_bytes: 数据类型字节数

    Returns:
        算术强度（FLOPs/Byte）
    """
    total_n = n * batch_size
    flops = 2 * m * k * total_n
    bytes_accessed = (m * k + k * total_n + m * total_n) * dtype_bytes
    return flops / bytes_accessed


class Timer:
    """
    简单的计时器，用于性能测量

    支持 GPU 同步计时。
    """

    def __init__(self, name: str = "", use_cuda: bool = False):
        self.name = name
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.elapsed = 0.0

    def __enter__(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.use_cuda:
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start

    def __str__(self):
        return f"{self.name}: {self.elapsed*1000:.2f} ms"


def format_bytes(n_bytes: int) -> str:
    """将字节数格式化为可读的字符串"""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 ** 2:
        return f"{n_bytes / 1024:.2f} KB"
    elif n_bytes < 1024 ** 3:
        return f"{n_bytes / 1024**2:.2f} MB"
    else:
        return f"{n_bytes / 1024**3:.2f} GB"


if __name__ == "__main__":
    print("=" * 60)
    print("推理优化工具函数演示")
    print("=" * 60)

    # 1. KV Cache 显存计算
    print("\n--- KV Cache 显存分析 ---")
    configs = {
        "Llama 2 7B (MHA)": LLAMA2_7B,
        "Llama 2 70B (GQA)": LLAMA2_70B,
        "Llama 3 8B (GQA)": LLAMA3_8B,
    }

    for name, config in configs.items():
        kv_info = compute_kv_cache_size(config, seq_len=4096, batch_size=1)
        print(f"\n{name}:")
        print(f"  模型参数量: {config.total_params / 1e9:.2f}B")
        print(f"  模型大小: {config.model_size_gb:.2f} GB")
        print(f"  每层每token KV元素数: {kv_info['elements_per_token_per_layer']}")
        print(f"  KV Cache (seq=4096, batch=1): {format_bytes(kv_info['total_bytes'])}")

    # 2. 算术强度分析
    print("\n--- 算术强度分析（Llama 2 7B, d=4096）---")
    d = 4096
    for batch_size in [1, 8, 32, 128]:
        ai = compute_arithmetic_intensity(d, 1, d, batch_size=batch_size)
        print(f"  Batch size={batch_size}: AI = {ai:.2f} FLOPs/Byte")

    # 3. 计时器使用示例
    print("\n--- 计时器演示 ---")
    x = torch.randn(1024, 1024)
    with Timer("矩阵乘法 [1024x1024]") as t:
        y = x @ x
    print(f"  {t}")
