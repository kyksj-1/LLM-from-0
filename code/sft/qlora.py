"""
QLoRA 实现（NF4 量化 + 双量化 + LoRA）

本模块实现了 QLoRA 的核心组件:
- NF4 (4-bit NormalFloat) 量化
- 双量化 (Double Quantization)
- 量化线性层 + LoRA 适配器

数学基础:
- NF4 分位点: q_i = Phi^{-1}(i / 2^k)，其中 Phi^{-1} 是标准正态逆 CDF
- 量化: 权重归一化后映射到最近的 NF4 分位点
- 反量化: W_dequant = NF4_lookup[indices] * scale_factors
- 双量化: 对 scale_factors 本身再量化一次，节省 0.25 bit/param

注意: 本实现为教学目的的纯 PyTorch 版本，速度较慢。
生产环境建议使用 bitsandbytes 库。

参考:
- Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .lora import LoRALinear


def compute_nf4_quantiles() -> torch.Tensor:
    """
    计算 NF4 (4-bit NormalFloat) 的 16 个量化分位点

    NF4 的核心思想: 对于正态分布的权重，使用正态分布的
    分位点作为量化级别，这样在概率密度高的区域（零附近）
    有更多的量化级别。

    数学:
        q_i = Phi^{-1}((i + 0.5) / 16), i = 0, ..., 15
        其中 Phi^{-1} 是标准正态分布的逆 CDF

    实际实现中，为了对称性，使用 [-1, 1] 范围内的
    等概率分位点，确保 0 是一个量化值。

    Returns:
        16 个 NF4 分位点值，排序后的 Tensor
    """
    # 标准正态分布的逆 CDF (percent point function)
    # 使用 torch 近似计算
    # 将 [0, 1] 分为 16 等份（概率空间等分）

    # 负半部分: 8 个分位点
    # 正半部分: 8 个分位点
    # 确保 0 是其中一个值

    num_bits = 4
    num_levels = 2 ** num_bits  # 16

    # 方法: 在概率空间等间距采样，然后求逆 CDF
    # 负半部分取 [-1, 0) 的 8 个等概率分位点
    # 正半部分取 (0, 1] 的 7 个等概率分位点 + 0 本身

    # 计算分位点
    offset = 0.9677083  # 经验值，保证对称性
    # 负区间: 8 个值
    neg_quantiles = []
    for i in range(8):
        p = (i + 1) / 17  # 概率值
        # 使用近似的逆正态 CDF
        q = _approx_normal_ppf(p)
        neg_quantiles.append(q)

    # 正区间: 7 个值 + 0
    pos_quantiles = [0.0]
    for i in range(7):
        p = (i + 9) / 17  # 从概率 9/17 开始
        q = _approx_normal_ppf(p)
        pos_quantiles.append(q)

    all_quantiles = neg_quantiles + pos_quantiles
    quantiles = torch.tensor(sorted(all_quantiles), dtype=torch.float32)

    # 归一化到 [-1, 1]
    quantiles = quantiles / quantiles.abs().max()

    return quantiles


def _approx_normal_ppf(p: float) -> float:
    """
    近似计算标准正态分布的逆 CDF (Percent Point Function)

    使用 Beasley-Springer-Moro 近似算法

    Args:
        p: 概率值 (0, 1)

    Returns:
        对应的标准正态分位点
    """
    # Rational approximation for the normal ppf
    # 适用于 0 < p < 1
    a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ]
    b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ]
    c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ]
    d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)


def quantize_nf4(
    weight: torch.Tensor, block_size: int = 64
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    将权重量化为 NF4 格式

    步骤:
    1. 将权重分成大小为 block_size 的块
    2. 每块计算绝对值最大值作为缩放因子
    3. 归一化到 [-1, 1]
    4. 映射到最近的 NF4 分位点

    Args:
        weight: 原始权重张量
        block_size: 量化块大小（每组权重数量）

    Returns:
        (quantized_indices, scale_factors, nf4_quantiles)
        - quantized_indices: uint8 张量，存储每个权重的量化索引 (0-15)
        - scale_factors: 每个块的缩放因子
        - nf4_quantiles: 16 个 NF4 分位点值
    """
    # 获取 NF4 分位点
    nf4_quantiles = compute_nf4_quantiles().to(weight.device)

    # 展平权重
    original_shape = weight.shape
    flat_weight = weight.reshape(-1)

    # 填充到 block_size 的倍数
    num_elements = flat_weight.numel()
    pad_size = (block_size - num_elements % block_size) % block_size
    if pad_size > 0:
        flat_weight = torch.cat([flat_weight, torch.zeros(pad_size, device=weight.device)])

    # 分块
    num_blocks = flat_weight.numel() // block_size
    blocks = flat_weight.reshape(num_blocks, block_size)

    # 每块的缩放因子: 绝对值最大值
    scale_factors = blocks.abs().max(dim=1).values
    scale_factors = scale_factors.clamp(min=1e-8)  # 避免除零

    # 归一化到 [-1, 1]
    normalized = blocks / scale_factors.unsqueeze(1)

    # 映射到最近的 NF4 分位点
    # 对每个值，找到最近的分位点索引
    # normalized: (num_blocks, block_size)
    # nf4_quantiles: (16,)
    distances = (normalized.unsqueeze(-1) - nf4_quantiles.unsqueeze(0).unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1)  # (num_blocks, block_size)

    # 展平索引并转换为 uint8
    quantized_indices = indices.reshape(-1)[:num_elements].to(torch.uint8)

    return quantized_indices, scale_factors, nf4_quantiles


def dequantize_nf4(
    quantized_indices: torch.Tensor,
    scale_factors: torch.Tensor,
    nf4_quantiles: torch.Tensor,
    original_shape: torch.Size,
    block_size: int = 64,
) -> torch.Tensor:
    """
    将 NF4 量化的权重反量化回 float

    反量化公式: W_dequant = nf4_quantiles[indices] * scale_factors

    Args:
        quantized_indices: 量化索引 (uint8)
        scale_factors: 块缩放因子
        nf4_quantiles: NF4 分位点值
        original_shape: 原始权重形状
        block_size: 量化块大小

    Returns:
        反量化后的权重张量
    """
    # 查表获取分位点值
    dequantized = nf4_quantiles[quantized_indices.long()]

    # 填充到 block_size 倍数
    num_elements = dequantized.numel()
    pad_size = (block_size - num_elements % block_size) % block_size
    if pad_size > 0:
        dequantized = torch.cat([dequantized, torch.zeros(pad_size, device=dequantized.device)])

    # 分块并乘以缩放因子
    num_blocks = dequantized.numel() // block_size
    blocks = dequantized.reshape(num_blocks, block_size)
    blocks = blocks * scale_factors[:num_blocks].unsqueeze(1)

    # 展平并恢复形状
    result = blocks.reshape(-1)[:math.prod(original_shape)]
    return result.reshape(original_shape)


def double_quantize(
    scale_factors: torch.Tensor, block_size_2: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    双量化: 对一级缩放因子再进行一次量化

    标准量化中，每 block_size 个权重有一个 FP32 缩放因子。
    双量化将这些 FP32 缩放因子量化为 FP8 + 二级缩放因子。

    显存节省:
    - 单量化: 32 / block_size bit/param 的额外开销
    - 双量化: 8 / block_size + 32 / (block_size * block_size_2) bit/param

    Args:
        scale_factors: 一级缩放因子 (FP32)
        block_size_2: 二级量化的块大小

    Returns:
        (quantized_scales, second_level_scales, zero_points)
    """
    num_scales = scale_factors.numel()

    # 填充到 block_size_2 的倍数
    pad_size = (block_size_2 - num_scales % block_size_2) % block_size_2
    if pad_size > 0:
        padded = torch.cat([scale_factors, torch.zeros(pad_size, device=scale_factors.device)])
    else:
        padded = scale_factors

    # 分块
    num_blocks = padded.numel() // block_size_2
    blocks = padded.reshape(num_blocks, block_size_2)

    # 计算二级缩放因子和零点
    block_max = blocks.max(dim=1).values
    block_min = blocks.min(dim=1).values
    second_level_scales = (block_max - block_min) / 255.0  # 映射到 8-bit
    second_level_scales = second_level_scales.clamp(min=1e-8)
    zero_points = block_min

    # 量化为 8-bit
    normalized = (blocks - zero_points.unsqueeze(1)) / second_level_scales.unsqueeze(1)
    quantized_scales = normalized.clamp(0, 255).round().to(torch.uint8)

    return quantized_scales.reshape(-1)[:num_scales], second_level_scales, zero_points


class QLoRALinear(nn.Module):
    """
    QLoRA 线性层: NF4 量化的基座权重 + BF16 的 LoRA 适配器

    结构:
    - 基座权重: 以 NF4 格式存储，前向传播时反量化为 BF16/FP16 计算
    - LoRA 适配器: 以全精度存储和更新
    - 输出: W_dequant @ x + (alpha/r) * B @ A @ x

    注意: 本实现为教学目的，速度较慢。
    生产环境请使用 bitsandbytes 的 Linear4bit。

    Args:
        original_linear: 原始 nn.Linear 层
        rank: LoRA 秩
        alpha: LoRA 缩放因子
        block_size: NF4 量化块大小
        use_double_quant: 是否使用双量化
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        block_size: int = 64,
        use_double_quant: bool = True,
    ):
        super().__init__()

        self.d_out, self.d_in = original_linear.weight.shape
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.block_size = block_size
        self.use_double_quant = use_double_quant

        # === NF4 量化原始权重 ===
        with torch.no_grad():
            weight = original_linear.weight.data.float()
            self.original_shape = weight.shape

            # 量化
            indices, scales, quantiles = quantize_nf4(weight, block_size)

            # 存储为 buffer（不参与梯度计算）
            self.register_buffer("quant_indices", indices)
            self.register_buffer("nf4_quantiles", quantiles)

            if use_double_quant:
                q_scales, level2_scales, zero_points = double_quantize(scales)
                self.register_buffer("quant_scales", q_scales)
                self.register_buffer("level2_scales", level2_scales)
                self.register_buffer("zero_points", zero_points)
            else:
                self.register_buffer("scale_factors", scales)

        # 偏置
        if original_linear.bias is not None:
            self.register_buffer("bias", original_linear.bias.data.clone())
        else:
            self.bias = None

        # === LoRA 参数（全精度可训练）===
        self.lora_A = nn.Parameter(torch.empty(rank, self.d_in))
        self.lora_B = nn.Parameter(torch.zeros(self.d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def _dequantize(self) -> torch.Tensor:
        """
        反量化 NF4 权重

        Returns:
            反量化后的权重张量 (d_out, d_in)
        """
        if self.use_double_quant:
            # 反量化缩放因子
            num_blocks = self.quant_scales.numel()
            block_size_2 = 256
            pad_size = (block_size_2 - num_blocks % block_size_2) % block_size_2
            if pad_size > 0:
                padded_q = torch.cat([
                    self.quant_scales.float(),
                    torch.zeros(pad_size, device=self.quant_scales.device)
                ])
            else:
                padded_q = self.quant_scales.float()

            n_blocks_2 = padded_q.numel() // block_size_2
            blocks_2 = padded_q.reshape(n_blocks_2, block_size_2)
            scale_factors = (
                blocks_2 * self.level2_scales[:n_blocks_2].unsqueeze(1)
                + self.zero_points[:n_blocks_2].unsqueeze(1)
            ).reshape(-1)[:num_blocks]
        else:
            scale_factors = self.scale_factors

        # 反量化权重
        return dequantize_nf4(
            self.quant_indices,
            scale_factors,
            self.nf4_quantiles,
            self.original_shape,
            self.block_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        output = dequant(W_nf4) @ x + (alpha/r) * B @ A @ x

        Args:
            x: 输入张量 (..., d_in)

        Returns:
            输出张量 (..., d_out)
        """
        # 反量化基座权重
        weight_dequant = self._dequantize().to(x.dtype)

        # 基座路径
        base_output = F.linear(x, weight_dequant, self.bias)

        # LoRA 路径
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return base_output + lora_output

    def memory_footprint(self) -> Dict:
        """
        计算显存占用

        Returns:
            各组件的显存占用字典（字节）
        """
        # NF4 权重: 4 bit / param
        quant_weight_bytes = self.quant_indices.numel() // 2  # 4 bit 每个

        # 缩放因子
        if self.use_double_quant:
            scale_bytes = (
                self.quant_scales.numel() * 1  # uint8
                + self.level2_scales.numel() * 4  # fp32
                + self.zero_points.numel() * 4  # fp32
            )
        else:
            scale_bytes = self.scale_factors.numel() * 4  # fp32

        # LoRA 参数 (fp32)
        lora_bytes = (self.lora_A.numel() + self.lora_B.numel()) * 4

        return {
            "quant_weight_bytes": quant_weight_bytes,
            "scale_bytes": scale_bytes,
            "lora_bytes": lora_bytes,
            "total_bytes": quant_weight_bytes + scale_bytes + lora_bytes,
            "bits_per_param": (quant_weight_bytes + scale_bytes) * 8 / (self.d_out * self.d_in),
        }


# 需要导入 F
import torch.nn.functional as F
from typing import Dict


if __name__ == "__main__":
    print("=" * 60)
    print("QLoRA 实现演示")
    print("=" * 60)

    torch.manual_seed(42)

    # === 1. NF4 分位点 ===
    print("\n--- 1. NF4 分位点 ---")
    quantiles = compute_nf4_quantiles()
    print(f"NF4 分位点数量: {len(quantiles)}")
    print(f"分位点值: {quantiles.tolist()}")
    print(f"范围: [{quantiles.min():.4f}, {quantiles.max():.4f}]")
    print(f"包含零: {0.0 in quantiles.tolist() or any(abs(q) < 0.01 for q in quantiles.tolist())}")

    # === 2. 量化与反量化测试 ===
    print("\n--- 2. 量化与反量化测试 ---")
    # 创建正态分布权重（模拟预训练模型）
    weight = torch.randn(256, 512) * 0.02  # 模型权重通常很小

    # 量化
    indices, scales, quant = quantize_nf4(weight, block_size=64)
    print(f"原始权重形状: {weight.shape}")
    print(f"量化索引形状: {indices.shape}, dtype: {indices.dtype}")
    print(f"缩放因子数量: {scales.shape[0]}")

    # 反量化
    weight_recovered = dequantize_nf4(indices, scales, quant, weight.shape, block_size=64)
    print(f"反量化权重形状: {weight_recovered.shape}")

    # 量化误差
    error = (weight - weight_recovered).abs()
    print(f"平均量化误差: {error.mean():.6f}")
    print(f"最大量化误差: {error.max():.6f}")
    print(f"相对误差 (RMSE/RMS): {error.pow(2).mean().sqrt() / weight.pow(2).mean().sqrt():.4f}")

    # === 3. 双量化测试 ===
    print("\n--- 3. 双量化测试 ---")
    q_scales, l2_scales, zps = double_quantize(scales)
    print(f"量化后缩放因子: dtype={q_scales.dtype}")
    print(f"二级缩放因子数量: {l2_scales.shape[0]}")

    # 显存分析
    original_scale_bytes = scales.numel() * 4  # FP32
    double_quant_bytes = q_scales.numel() * 1 + l2_scales.numel() * 4 + zps.numel() * 4
    print(f"单量化缩放因子: {original_scale_bytes} bytes")
    print(f"双量化总计: {double_quant_bytes} bytes")
    print(f"节省: {original_scale_bytes - double_quant_bytes} bytes")

    # === 4. QLoRA 线性层测试 ===
    print("\n--- 4. QLoRA 线性层测试 ---")
    original_linear = nn.Linear(512, 256)
    qlora_layer = QLoRALinear(
        original_linear, rank=8, alpha=16.0,
        block_size=64, use_double_quant=True
    )

    x = torch.randn(2, 32, 512)
    output = qlora_layer(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 显存分析
    memory = qlora_layer.memory_footprint()
    print(f"\n显存分析:")
    print(f"  量化权重: {memory['quant_weight_bytes']:,} bytes")
    print(f"  缩放因子: {memory['scale_bytes']:,} bytes")
    print(f"  LoRA 参数: {memory['lora_bytes']:,} bytes")
    print(f"  总计: {memory['total_bytes']:,} bytes")
    print(f"  等效位宽: {memory['bits_per_param']:.2f} bits/param")

    # 与 FP32 对比
    fp32_bytes = 512 * 256 * 4
    print(f"  FP32 原始: {fp32_bytes:,} bytes")
    print(f"  压缩比: {fp32_bytes / memory['total_bytes']:.1f}x")

    # === 5. 模型级显存估算 ===
    print("\n--- 5. 模型级显存估算 ---")
    print(f"{'模型':>15} {'参数量':>10} {'FP16':>10} {'QLoRA NF4':>12} {'节省':>8}")
    print("-" * 60)
    for name, params_b in [("Llama-7B", 7), ("Llama-13B", 13), ("Llama-70B", 70)]:
        fp16_gb = params_b * 2  # 2 bytes per param
        qlora_gb = params_b * 4.25 / 8  # 4.25 bits per param
        saving = (1 - qlora_gb / fp16_gb) * 100
        print(f"{name:>15} {params_b:>9}B {fp16_gb:>9.1f}GB {qlora_gb:>11.1f}GB {saving:>7.0f}%")
