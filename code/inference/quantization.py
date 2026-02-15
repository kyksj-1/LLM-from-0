"""
线性量化基础实现

本模块实现了 LLM 量化推理中的基础量化方法，包括：
- 对称量化（Symmetric Quantization）
- 非对称量化（Asymmetric Quantization）
- 不同粒度：per-tensor / per-channel / per-group
- 量化误差分析

量化的核心公式:
    量化:   x_q = round(x / scale) + zero_point
    反量化: x_hat = scale * (x_q - zero_point)

量化误差来自 round 操作，可以建模为均匀分布的加性噪声:
    E[epsilon^2] = scale^2 / 12
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QuantConfig:
    """量化配置"""
    n_bits: int = 8             # 量化位宽
    symmetric: bool = True      # 是否对称量化
    granularity: str = "tensor" # 粒度: "tensor", "channel", "group"
    group_size: int = 128       # per-group 量化的组大小


def symmetric_quantize(
    x: torch.Tensor,
    n_bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对称线性量化

    将浮点张量映射到 [-2^(b-1)+1, 2^(b-1)-1] 的整数范围。
    零点固定为 0，浮点数 0 精确映射到整数 0。

    公式:
        scale = max(|x|) / (2^(b-1) - 1)
        x_q = round(x / scale), 裁剪到 [-qmax, qmax]

    Args:
        x: 输入浮点张量
        n_bits: 量化位宽

    Returns:
        (x_q, scale): 量化后的整数张量和缩放因子
    """
    qmax = 2 ** (n_bits - 1) - 1
    qmin = -qmax

    # 计算缩放因子
    x_abs_max = x.abs().max()
    if x_abs_max == 0:
        scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
    else:
        scale = x_abs_max / qmax

    # 量化
    x_q = torch.round(x / scale).clamp(qmin, qmax)

    return x_q, scale


def asymmetric_quantize(
    x: torch.Tensor,
    n_bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    非对称线性量化

    将浮点张量映射到 [0, 2^b - 1] 的整数范围。
    允许零点偏移，能更好地利用量化范围。

    公式:
        scale = (x_max - x_min) / (2^b - 1)
        zero_point = round(-x_min / scale)
        x_q = round(x / scale) + zero_point

    Args:
        x: 输入浮点张量
        n_bits: 量化位宽

    Returns:
        (x_q, scale, zero_point): 量化值、缩放因子、零点
    """
    qmax = 2 ** n_bits - 1
    qmin = 0

    x_min = x.min()
    x_max = x.max()

    # 处理退化情况
    if x_min == x_max:
        scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        zero_point = torch.tensor(0, dtype=torch.int32, device=x.device)
        x_q = torch.zeros_like(x)
        return x_q, scale, zero_point

    # 计算参数
    scale = (x_max - x_min) / qmax
    zero_point = torch.round(-x_min / scale).clamp(qmin, qmax).to(torch.int32)

    # 量化
    x_q = torch.round(x / scale + zero_point.float()).clamp(qmin, qmax)

    return x_q, scale, zero_point


def dequantize_symmetric(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    对称量化的反量化

    Args:
        x_q: 量化值
        scale: 缩放因子

    Returns:
        近似恢复的浮点张量
    """
    return x_q.float() * scale


def dequantize_asymmetric(
    x_q: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """
    非对称量化的反量化

    Args:
        x_q: 量化值
        scale: 缩放因子
        zero_point: 零点

    Returns:
        近似恢复的浮点张量
    """
    return (x_q.float() - zero_point.float()) * scale


def per_channel_quantize(
    weight: torch.Tensor,
    n_bits: int = 8,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    按输出通道量化（per-channel）

    每个输出通道（weight 的每一行）有独立的 scale 和 zero_point。
    适合权重量化，因为不同输出通道的数值范围可能差异很大。

    Args:
        weight: 权重矩阵 [out_features, in_features]
        n_bits: 量化位宽
        symmetric: 是否对称量化

    Returns:
        (w_q, scales, zero_points): 量化权重、每通道缩放因子、零点
    """
    out_features = weight.shape[0]
    w_q = torch.zeros_like(weight)
    scales = torch.zeros(out_features, dtype=weight.dtype, device=weight.device)
    zero_points = None if symmetric else torch.zeros(
        out_features, dtype=torch.int32, device=weight.device
    )

    for i in range(out_features):
        row = weight[i]
        if symmetric:
            row_q, scale = symmetric_quantize(row, n_bits)
            w_q[i] = row_q
            scales[i] = scale
        else:
            row_q, scale, zp = asymmetric_quantize(row, n_bits)
            w_q[i] = row_q
            scales[i] = scale
            zero_points[i] = zp

    return w_q, scales, zero_points


def per_group_quantize(
    weight: torch.Tensor,
    n_bits: int = 8,
    group_size: int = 128,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    按组量化（per-group）

    将权重矩阵的每一行按 group_size 分组，每组有独立的量化参数。
    这是现代 LLM 量化（如 GPTQ、AWQ）最常用的粒度。

    Args:
        weight: 权重矩阵 [out_features, in_features]
        n_bits: 量化位宽
        group_size: 每组的元素数
        symmetric: 是否对称量化

    Returns:
        (w_q, scales, zero_points): 量化权重、每组缩放因子、零点
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, (
        f"in_features ({in_features}) 必须能被 group_size ({group_size}) 整除"
    )
    n_groups = in_features // group_size

    w_q = torch.zeros_like(weight)
    scales = torch.zeros(out_features, n_groups, dtype=weight.dtype, device=weight.device)
    zero_points = None if symmetric else torch.zeros(
        out_features, n_groups, dtype=torch.int32, device=weight.device
    )

    for i in range(out_features):
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            group = weight[i, start:end]

            if symmetric:
                group_q, scale = symmetric_quantize(group, n_bits)
                w_q[i, start:end] = group_q
                scales[i, g] = scale
            else:
                group_q, scale, zp = asymmetric_quantize(group, n_bits)
                w_q[i, start:end] = group_q
                scales[i, g] = scale
                zero_points[i, g] = zp

    return w_q, scales, zero_points


def quantized_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    scales: torch.Tensor,
    zero_points: Optional[torch.Tensor] = None,
    granularity: str = "tensor",
    group_size: int = 128,
) -> torch.Tensor:
    """
    使用量化权重的线性层计算

    先反量化权重，再做矩阵乘法。
    注意：实际硬件上可以用整数运算直接计算，这里用反量化方式模拟。

    Args:
        x: 输入 [batch, ..., in_features]
        w_q: 量化权重 [out_features, in_features]
        scales: 缩放因子
        zero_points: 零点（对称量化为 None）
        granularity: 量化粒度
        group_size: 组大小

    Returns:
        输出 [batch, ..., out_features]
    """
    out_features, in_features = w_q.shape

    if granularity == "tensor":
        # Per-tensor: 全局一个 scale
        if zero_points is None:
            w_deq = dequantize_symmetric(w_q, scales)
        else:
            w_deq = dequantize_asymmetric(w_q, scales, zero_points)

    elif granularity == "channel":
        # Per-channel: 每行一个 scale
        if zero_points is None:
            w_deq = w_q.float() * scales.unsqueeze(1)
        else:
            w_deq = (w_q.float() - zero_points.float().unsqueeze(1)) * scales.unsqueeze(1)

    elif granularity == "group":
        # Per-group: 分组反量化
        n_groups = in_features // group_size
        w_deq = torch.zeros_like(w_q, dtype=torch.float32)
        for g in range(n_groups):
            start = g * group_size
            end = start + group_size
            if zero_points is None:
                w_deq[:, start:end] = w_q[:, start:end].float() * scales[:, g:g+1]
            else:
                w_deq[:, start:end] = (
                    (w_q[:, start:end].float() - zero_points[:, g:g+1].float())
                    * scales[:, g:g+1]
                )

    return x.float() @ w_deq.T


def compute_quantization_error(
    original: torch.Tensor,
    quantized: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor] = None,
) -> dict:
    """
    计算量化误差指标

    Args:
        original: 原始浮点张量
        quantized: 量化后的张量
        scale: 缩放因子
        zero_point: 零点

    Returns:
        包含各种误差指标的字典
    """
    # 反量化
    if zero_point is None:
        restored = dequantize_symmetric(quantized, scale)
    else:
        restored = dequantize_asymmetric(quantized, scale, zero_point)

    error = original.float() - restored

    return {
        "mse": (error ** 2).mean().item(),
        "rmse": (error ** 2).mean().sqrt().item(),
        "max_abs_error": error.abs().max().item(),
        "mean_abs_error": error.abs().mean().item(),
        "snr_db": (10 * torch.log10(
            (original.float() ** 2).mean() / (error ** 2).mean()
        )).item() if (error ** 2).mean() > 0 else float("inf"),
        "theoretical_mse": (scale.item() ** 2) / 12 if scale.numel() == 1 else None,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("线性量化基础实现演示")
    print("=" * 60)

    torch.manual_seed(42)

    # 1. 对称 vs 非对称量化
    print("\n--- 对称 vs 非对称量化对比 ---")
    x = torch.randn(1000)  # 正态分布，以 0 为中心

    for n_bits in [8, 4]:
        # 对称量化
        x_q_sym, scale_sym = symmetric_quantize(x, n_bits)
        err_sym = compute_quantization_error(x, x_q_sym, scale_sym)

        # 非对称量化
        x_q_asym, scale_asym, zp_asym = asymmetric_quantize(x, n_bits)
        err_asym = compute_quantization_error(x, x_q_asym, scale_asym, zp_asym)

        print(f"\nINT{n_bits} 量化 (正态分布数据):")
        print(f"  对称:   MSE={err_sym['mse']:.6f}, SNR={err_sym['snr_db']:.1f} dB")
        print(f"  非对称: MSE={err_asym['mse']:.6f}, SNR={err_asym['snr_db']:.1f} dB")

    # 2. 非负数据的量化（如 ReLU 后的激活）
    print("\n--- 非负数据量化（ReLU 输出）---")
    x_relu = F.relu(torch.randn(1000))  # 非负分布

    x_q_sym, s_sym = symmetric_quantize(x_relu, 8)
    err_sym = compute_quantization_error(x_relu, x_q_sym, s_sym)

    x_q_asym, s_asym, zp_asym = asymmetric_quantize(x_relu, 8)
    err_asym = compute_quantization_error(x_relu, x_q_asym, s_asym, zp_asym)

    print(f"  对称:   MSE={err_sym['mse']:.6f} (浪费了一半范围)")
    print(f"  非对称: MSE={err_asym['mse']:.6f} (更好地利用范围)")

    # 3. 不同粒度的量化对比
    print("\n--- 量化粒度对比 ---")
    # 模拟一个权重矩阵（带有不同范围的行）
    weight = torch.randn(64, 256)
    # 人为制造不同行有不同尺度
    weight[0:16] *= 0.1   # 小权重行
    weight[16:32] *= 1.0  # 正常行
    weight[32:48] *= 5.0  # 大权重行
    weight[48:64] *= 0.5  # 中等行

    for n_bits in [8, 4]:
        print(f"\nINT{n_bits}:")

        # Per-tensor
        w_q, scale = symmetric_quantize(weight, n_bits)
        err = compute_quantization_error(weight, w_q, scale)
        print(f"  Per-tensor:  MSE={err['mse']:.6f}")

        # Per-channel
        w_q_ch, scales_ch, _ = per_channel_quantize(weight, n_bits, symmetric=True)
        # 反量化计算误差
        w_restored_ch = w_q_ch.float() * scales_ch.unsqueeze(1)
        mse_ch = ((weight - w_restored_ch) ** 2).mean().item()
        print(f"  Per-channel: MSE={mse_ch:.6f}")

        # Per-group (group_size=128)
        w_q_grp, scales_grp, _ = per_group_quantize(weight, n_bits, group_size=128)
        n_groups = 256 // 128
        w_restored_grp = torch.zeros_like(weight, dtype=torch.float32)
        for g in range(n_groups):
            s, e = g * 128, (g + 1) * 128
            w_restored_grp[:, s:e] = w_q_grp[:, s:e].float() * scales_grp[:, g:g+1]
        mse_grp = ((weight - w_restored_grp) ** 2).mean().item()
        print(f"  Per-group (g=128): MSE={mse_grp:.6f}")

    # 4. Outlier 对量化的影响
    print("\n--- Outlier 对量化的影响 ---")
    x_normal = torch.randn(1000)
    x_with_outlier = x_normal.clone()
    x_with_outlier[0] = 100.0  # 添加一个极端离群值

    x_q_normal, s_normal = symmetric_quantize(x_normal, 8)
    x_q_outlier, s_outlier = symmetric_quantize(x_with_outlier, 8)

    err_normal = compute_quantization_error(x_normal, x_q_normal, s_normal)
    err_outlier = compute_quantization_error(x_with_outlier, x_q_outlier, s_outlier)

    print(f"  无 outlier: scale={s_normal.item():.4f}, MSE={err_normal['mse']:.6f}")
    print(f"  有 outlier: scale={s_outlier.item():.4f}, MSE={err_outlier['mse']:.6f}")
    print(f"  Outlier 导致 scale 增大 {s_outlier.item()/s_normal.item():.1f}x，"
          f"MSE 增大 {err_outlier['mse']/err_normal['mse']:.1f}x")

    # 5. 量化矩阵乘法的精度
    print("\n--- 量化矩阵乘法精度 ---")
    weight = torch.randn(128, 256)
    x_input = torch.randn(4, 256)

    # FP32 参考结果
    y_ref = x_input @ weight.T

    for n_bits, label in [(8, "INT8"), (4, "INT4")]:
        # Per-channel 量化
        w_q, scales, _ = per_channel_quantize(weight, n_bits)
        y_quant = quantized_linear(x_input, w_q, scales, granularity="channel")

        error = (y_ref - y_quant).abs()
        rel_error = error / (y_ref.abs() + 1e-8)
        print(f"\n  {label} per-channel:")
        print(f"    平均绝对误差: {error.mean().item():.6f}")
        print(f"    最大绝对误差: {error.max().item():.6f}")
        print(f"    平均相对误差: {rel_error.mean().item()*100:.2f}%")
