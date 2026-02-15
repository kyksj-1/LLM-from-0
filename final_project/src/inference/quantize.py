"""
模型量化

知识依赖:
- 模块 14（推理加速）: INT8/INT4 量化原理

参考实现:
- code/inference/quantization.py
- code/inference/gptq_simplified.py

量化原理:
    将 FP16/BF16 的模型权重压缩为 INT8 或 INT4，减少显存占用和带宽需求。
    Decode 阶段是访存密集型（Arithmetic Intensity 很低），
    量化通过减少数据传输量来加速推理。

    量化公式 (对称量化):
    scale = max(|W|) / (2^(bits-1) - 1)
    W_int = round(W / scale)
    W_dequant = W_int * scale

    量化误差 = W - W_dequant

量化策略:
- 训练后量化 (PTQ): 不需要额外训练，直接量化权重
  - Per-tensor: 整个张量一个 scale（简单但误差大）
  - Per-channel: 每个输出通道一个 scale（更精确）
- GPTQ: 基于 Hessian 信息的逐列量化，误差更小
- AWQ: 基于激活值重要性的加权量化
"""

import torch
import torch.nn as nn
from typing import Optional, Dict


def quantize_int8(model: nn.Module) -> nn.Module:
    """
    将模型线性层量化为 INT8

    Args:
        model: FP16/BF16 模型

    Returns:
        INT8 量化后的模型

    实现步骤:
        遍历所有 nn.Linear 层:
        1. 计算 per-channel scale: scale = max(|W|, dim=1) / 127
        2. 量化: W_int8 = round(W / scale).clamp(-128, 127).to(torch.int8)
        3. 保存 scale 用于反量化
        4. 替换原始线性层为自定义的 QuantizedLinear

    注意: 这是教学版简化实现，生产环境请用 bitsandbytes 或 auto-gptq
    """
    raise NotImplementedError(
        "TODO: 实现 INT8 量化\n"
        "参考: 模块 14 的量化章节\n"
        "参考实现: code/inference/quantization.py"
    )


def quantize_int4(model: nn.Module) -> nn.Module:
    """
    将模型线性层量化为 INT4

    INT4 进一步压缩: 每个权重只用 4 bit 存储。
    需要分组量化 (group-wise): 每 128 个元素共享一个 scale。

    Args:
        model: FP16/BF16 模型

    Returns:
        INT4 量化后的模型
    """
    raise NotImplementedError(
        "TODO: 实现 INT4 量化\n"
        "参考: 模块 14 的 GPTQ/AWQ 章节\n"
        "参考实现: code/inference/gptq_simplified.py\n"
        "提示: 生产环境推荐直接使用 auto-gptq 或 autoawq 库"
    )


def estimate_quantized_size(
    n_params: int, bits: int = 8, group_size: int = 128
) -> Dict[str, float]:
    """
    估算量化后的模型大小

    Args:
        n_params: 参数量
        bits: 量化位数 (4 或 8)
        group_size: 分组大小（INT4 时使用）

    Returns:
        {"original_gb": float, "quantized_gb": float, "compression_ratio": float}
    """
    raise NotImplementedError(
        "TODO: 估算量化后大小\n"
        "公式: quantized_bytes = n_params * bits / 8 + n_groups * 2 (scale 存储)"
    )
