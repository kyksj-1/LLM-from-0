"""
GPTQ 简化实现

本模块实现了 GPTQ（Generalized Post-Training Quantization）的核心算法，
这是一种基于 Hessian 矩阵的逐层权重量化方法。

核心思想:
    量化不是简单地 round 每个权重，而是在量化一个权重后，调整剩余未量化权重
    来补偿量化误差。调整的依据来自 Hessian 矩阵 H = 2*X*X^T。

算法流程:
    1. 计算校准数据的 Hessian 矩阵 H
    2. 对 H 做 Cholesky 分解
    3. 按列（或分组）依次量化权重:
       - 量化当前列 w_j
       - 计算量化误差 delta = (w_j - quant(w_j)) / H_jj
       - 用 delta 更新后续列: W[:, j+1:] -= delta * H[j, j+1:]

参考论文:
    Frantar et al., "GPTQ: Accurate Post-Training Quantization for Generative
    Pre-trained Transformers", ICLR 2023
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class GPTQConfig:
    """GPTQ 量化配置"""
    n_bits: int = 4             # 量化位宽
    group_size: int = 128       # 分组大小（-1 表示 per-channel）
    block_size: int = 128       # 批量更新的列数
    symmetric: bool = True      # 对称量化
    damp_percent: float = 0.01  # Hessian 对角线阻尼系数


def compute_hessian(
    X: torch.Tensor,
) -> torch.Tensor:
    """
    计算 Hessian 矩阵 H = 2 * X * X^T

    Hessian 矩阵衡量了每个权重对输出的影响程度。
    H_ij 大的权重对输出影响大，量化时需要更谨慎。

    Args:
        X: 校准数据的层输入 [n_samples, in_features]

    Returns:
        H: Hessian 矩阵 [in_features, in_features]
    """
    n_samples = X.shape[0]
    # H = (2/n) * X^T @ X，除以 n_samples 做归一化
    H = (2.0 / n_samples) * (X.T @ X)
    return H


def quantize_value(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """
    将浮点值量化到最近的整数

    Args:
        x: 待量化的浮点值
        scale: 缩放因子
        zero_point: 零点
        qmin, qmax: 量化范围

    Returns:
        量化后的值（仍为浮点表示，因为后续需要计算误差）
    """
    x_q = torch.round(x / scale + zero_point).clamp(qmin, qmax)
    # 反量化回浮点，用于误差计算
    x_deq = (x_q - zero_point) * scale
    return x_deq


def compute_scale_zp(
    w: torch.Tensor,
    n_bits: int,
    symmetric: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    """
    计算一组权重的量化参数

    Args:
        w: 权重 [n_elements]
        n_bits: 位宽
        symmetric: 是否对称

    Returns:
        (scale, zero_point, qmin, qmax)
    """
    if symmetric:
        qmax = 2 ** (n_bits - 1) - 1
        qmin = -qmax
        w_max = w.abs().max()
        scale = w_max / qmax if w_max > 0 else torch.tensor(1.0)
        zero_point = torch.tensor(0.0)
    else:
        qmax = 2 ** n_bits - 1
        qmin = 0
        w_min = w.min()
        w_max = w.max()
        scale = (w_max - w_min) / qmax if (w_max - w_min) > 0 else torch.tensor(1.0)
        zero_point = torch.round(-w_min / scale)

    return scale, zero_point, qmin, qmax


def gptq_quantize_layer(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    config: GPTQConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPTQ 逐层量化

    按列顺序量化权重，每量化一列后使用 Hessian 信息更新后续列
    以补偿量化误差。

    Args:
        weight: 权重矩阵 [out_features, in_features]
        hessian: Hessian 矩阵 [in_features, in_features]
        config: GPTQ 配置

    Returns:
        (quantized_weight, scales, zero_points):
            量化后的权重、缩放因子、零点
    """
    out_features, in_features = weight.shape
    device = weight.device
    dtype = weight.dtype

    # 复制权重，避免修改原始数据
    W = weight.clone().float()
    H = hessian.clone().float()

    # 添加阻尼，防止 Hessian 奇异
    damp = config.damp_percent * torch.diag(H).mean()
    H += damp * torch.eye(in_features, device=device)

    # 计算 Hessian 逆的 Cholesky 分解
    # H^{-1} 的 Cholesky 分解: H^{-1} = L L^T
    # 实际上我们用的是 H 的 Cholesky，然后求逆对角元素
    try:
        H_cho = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_cho)
    except RuntimeError:
        # Cholesky 分解失败，回退到简单量化
        print("警告: Hessian Cholesky 分解失败，使用简化方法")
        H_inv = torch.diag(1.0 / torch.diag(H).clamp(min=1e-8))

    # 确定量化分组
    if config.group_size > 0:
        n_groups = math.ceil(in_features / config.group_size)
    else:
        n_groups = 1

    # 存储量化结果
    Q = torch.zeros_like(W)
    scales = torch.zeros(out_features, max(n_groups, 1), device=device)
    zero_points = torch.zeros(out_features, max(n_groups, 1), device=device)

    # 按块处理（Lazy Batch Updates）
    for block_start in range(0, in_features, config.block_size):
        block_end = min(block_start + config.block_size, in_features)
        block_len = block_end - block_start

        # 当前块的权重
        W_block = W[:, block_start:block_end].clone()
        # 当前块的 Hessian 逆对角
        H_inv_diag = torch.diag(H_inv)[block_start:block_end]
        # 当前块到后续列的 Hessian 逆
        H_inv_block = H_inv[block_start:block_end, block_start:block_end]

        # 累积误差（用于延迟更新）
        Err = torch.zeros(out_features, block_len, device=device)

        for j in range(block_len):
            col_idx = block_start + j

            # 确定当前列所属的量化组
            if config.group_size > 0:
                group_idx = col_idx // config.group_size
                group_start = group_idx * config.group_size
                group_end = min(group_start + config.group_size, in_features)
            else:
                group_idx = 0
                group_start = 0
                group_end = in_features

            # 如果是新组的第一列，计算该组的量化参数
            if config.group_size > 0 and col_idx == group_start:
                w_group = W[:, group_start:group_end]
                for row in range(out_features):
                    s, zp, _, _ = compute_scale_zp(
                        w_group[row], config.n_bits, config.symmetric
                    )
                    scales[row, group_idx] = s
                    zero_points[row, group_idx] = zp

            # 量化当前列
            w_col = W[:, col_idx]

            for row in range(out_features):
                s = scales[row, group_idx]
                zp = zero_points[row, group_idx]
                qmin = -(2 ** (config.n_bits - 1) - 1) if config.symmetric else 0
                qmax = 2 ** (config.n_bits - 1) - 1 if config.symmetric else 2 ** config.n_bits - 1

                # 量化并反量化
                w_deq = quantize_value(w_col[row:row+1], s, zp, qmin, qmax)
                Q[row, col_idx] = w_deq.item()

                # 量化误差
                err = (w_col[row] - w_deq.item())
                Err[row, j] = err

            # 使用 Hessian 信息更新块内后续列
            if H_inv_diag[j].abs() > 1e-10:
                delta = Err[:, j:j+1] / H_inv_diag[j]
                W[:, col_idx+1:block_end] -= (
                    delta @ H_inv_block[j:j+1, j+1:block_len]
                )

        # 延迟更新：将累积误差传播到块之后的列
        if block_end < in_features:
            W[:, block_end:] -= (
                Err @ H_inv[block_start:block_end, block_end:]
            )

    return Q, scales, zero_points


def gptq_quantize_simple(
    weight: torch.Tensor,
    calibration_data: torch.Tensor,
    config: GPTQConfig = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPTQ 量化的简化接口

    将校准数据计算 Hessian 和量化步骤封装在一起。

    Args:
        weight: 权重矩阵 [out_features, in_features]
        calibration_data: 校准数据 [n_samples, in_features]
        config: GPTQ 配置（默认 INT4, group=128）

    Returns:
        (quantized_weight, scales, zero_points)
    """
    if config is None:
        config = GPTQConfig()

    # 计算 Hessian
    hessian = compute_hessian(calibration_data)

    # 执行 GPTQ 量化
    return gptq_quantize_layer(weight, hessian, config)


if __name__ == "__main__":
    print("=" * 60)
    print("GPTQ 简化实现演示")
    print("=" * 60)

    torch.manual_seed(42)

    # 1. 创建测试场景
    print("\n--- 设置 ---")
    in_features = 256
    out_features = 128
    n_samples = 64

    # 随机权重和校准数据
    weight = torch.randn(out_features, in_features) * 0.1
    calibration_data = torch.randn(n_samples, in_features)
    test_input = torch.randn(8, in_features)

    # FP32 参考输出
    y_ref = test_input @ weight.T
    print(f"权重形状: {weight.shape}")
    print(f"校准数据: {n_samples} 个样本")

    # 2. 简单 round-to-nearest 量化（基线）
    print("\n--- 基线: Round-to-Nearest INT4 量化 ---")
    qmax = 7
    scale_rtn = weight.abs().max() / qmax
    w_q_rtn = torch.round(weight / scale_rtn).clamp(-qmax, qmax)
    w_deq_rtn = w_q_rtn * scale_rtn
    y_rtn = test_input @ w_deq_rtn.T

    mse_rtn = ((y_ref - y_rtn) ** 2).mean().item()
    print(f"  输出 MSE: {mse_rtn:.6f}")

    # 3. GPTQ INT4 量化
    print("\n--- GPTQ INT4 量化 ---")
    config = GPTQConfig(n_bits=4, group_size=128, block_size=64, symmetric=True)
    w_gptq, scales_gptq, zp_gptq = gptq_quantize_simple(weight, calibration_data, config)

    y_gptq = test_input @ w_gptq.T
    mse_gptq = ((y_ref - y_gptq) ** 2).mean().item()
    print(f"  输出 MSE: {mse_gptq:.6f}")

    # 4. GPTQ INT8 量化
    print("\n--- GPTQ INT8 量化 ---")
    config_int8 = GPTQConfig(n_bits=8, group_size=128, block_size=64, symmetric=True)
    w_gptq8, scales_gptq8, zp_gptq8 = gptq_quantize_simple(
        weight, calibration_data, config_int8
    )

    y_gptq8 = test_input @ w_gptq8.T
    mse_gptq8 = ((y_ref - y_gptq8) ** 2).mean().item()
    print(f"  输出 MSE: {mse_gptq8:.6f}")

    # 5. 对比总结
    print("\n--- 量化方法对比 ---")
    print(f"{'方法':<20} {'输出 MSE':<15} {'相对于 RTN 改善'}")
    print("-" * 55)
    print(f"{'RTN INT4':<20} {mse_rtn:<15.6f} {'基准'}")
    print(f"{'GPTQ INT4':<20} {mse_gptq:<15.6f} {mse_rtn/max(mse_gptq, 1e-10):.2f}x")
    print(f"{'GPTQ INT8':<20} {mse_gptq8:<15.6f} {mse_rtn/max(mse_gptq8, 1e-10):.2f}x")

    # 6. 不同 group_size 的影响
    print("\n--- Group Size 对 GPTQ INT4 的影响 ---")
    for gs in [32, 64, 128, 256]:
        if in_features % gs != 0:
            continue
        cfg = GPTQConfig(n_bits=4, group_size=gs, block_size=64)
        w_q, _, _ = gptq_quantize_simple(weight, calibration_data, cfg)
        y_q = test_input @ w_q.T
        mse = ((y_ref - y_q) ** 2).mean().item()
        n_scale_params = out_features * (in_features // gs)
        print(f"  group_size={gs:<4d}: MSE={mse:.6f}, "
              f"额外参数={n_scale_params} (占比{n_scale_params/(out_features*in_features)*100:.1f}%)")
