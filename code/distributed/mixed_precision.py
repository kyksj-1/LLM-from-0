"""
混合精度训练封装

本模块实现混合精度训练的核心组件:
    - 手动混合精度训练循环
    - 动态损失缩放 (Dynamic Loss Scaling)
    - 数值格式对比分析

核心思想:
    前向和反向传播使用低精度 (FP16/BF16) 以加速计算，
    参数更新使用高精度 (FP32) 以保持精度。

关键公式:
    FP16 最小非零值: 2^(-24) ~ 5.96e-8
    FP16 最大值: 65504
    BF16 最大值: ~3.4e38 (与 FP32 相同)

作者: LLM Tutorial
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field
import math


@dataclass
class DynamicLossScaler:
    """
    动态损失缩放器

    FP16 训练中，小梯度可能下溢为 0。损失缩放通过放大损失值
    来放大梯度，避免下溢。动态版本自动调整缩放因子。

    工作流程:
        1. 将损失乘以缩放因子 s: L_scaled = L * s
        2. 反向传播得到缩放后的梯度: g_scaled = g * s
        3. 检查梯度是否包含 inf/nan
        4a. 如果正常: 反缩放梯度 g = g_scaled / s，执行优化器步骤
        4b. 如果异常: 缩小缩放因子，跳过本步更新

    Attributes:
        init_scale: 初始缩放因子
        growth_factor: 缩放因子增长倍数
        backoff_factor: 缩放因子缩小倍数
        growth_interval: 连续多少步正常后尝试增大缩放因子
    """
    init_scale: float = 2.0 ** 16
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000

    # 运行时状态
    cur_scale: float = field(init=False)
    consecutive_normal: int = field(init=False, default=0)
    total_steps: int = field(init=False, default=0)
    overflow_count: int = field(init=False, default=0)

    def __post_init__(self):
        self.cur_scale = self.init_scale

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失值"""
        return loss * self.cur_scale

    def unscale_grads(self, optimizer: torch.optim.Optimizer) -> bool:
        """
        反缩放梯度并检查溢出

        Args:
            optimizer: 优化器

        Returns:
            True 如果梯度正常，False 如果检测到 inf/nan
        """
        found_inf = False

        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    # 反缩放
                    param.grad.data /= self.cur_scale

                    # 检查 inf/nan
                    if torch.isinf(param.grad.data).any() or torch.isnan(param.grad.data).any():
                        found_inf = True

        return not found_inf

    def update(self, grads_normal: bool):
        """
        更新缩放因子

        Args:
            grads_normal: 梯度是否正常（无 inf/nan）
        """
        self.total_steps += 1

        if grads_normal:
            self.consecutive_normal += 1
            if self.consecutive_normal >= self.growth_interval:
                self.cur_scale *= self.growth_factor
                self.consecutive_normal = 0
        else:
            self.cur_scale *= self.backoff_factor
            self.consecutive_normal = 0
            self.overflow_count += 1

    def state_dict(self) -> Dict:
        """导出状态"""
        return {
            "cur_scale": self.cur_scale,
            "consecutive_normal": self.consecutive_normal,
            "total_steps": self.total_steps,
            "overflow_count": self.overflow_count,
        }


class ManualMixedPrecisionTrainer:
    """
    手动混合精度训练器

    演示混合精度训练的完整流程，不使用 torch.cuda.amp。
    适用于理解混合精度训练的内部机制。

    训练流程:
        1. 维护 FP32 主权重
        2. 每步将主权重转换为 FP16 用于前向/反向传播
        3. 使用动态损失缩放避免梯度下溢
        4. 在 FP32 精度下执行参数更新
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        use_loss_scaling: bool = True,
    ):
        """
        Args:
            model: 模型 (参数应为 FP32)
            lr: 学习率
            use_loss_scaling: 是否使用动态损失缩放
        """
        self.model = model

        # 保存 FP32 主权重的副本
        self.fp32_master_weights = [
            p.data.clone().float() for p in model.parameters()
        ]

        # 优化器在 FP32 主权重上操作
        self.optimizer = torch.optim.Adam(
            [{"params": self.fp32_master_weights}], lr=lr
        )

        # 将 FP32 主权重包裹为需要梯度的参数
        for w in self.fp32_master_weights:
            w.requires_grad = True

        self.use_loss_scaling = use_loss_scaling
        if use_loss_scaling:
            self.scaler = DynamicLossScaler()
        else:
            self.scaler = None

    def train_step(
        self, inputs: torch.Tensor, targets: torch.Tensor, criterion: nn.Module
    ) -> Tuple[float, bool]:
        """
        执行一步混合精度训练

        Args:
            inputs: 输入数据
            targets: 目标标签
            criterion: 损失函数

        Returns:
            (损失值, 是否执行了参数更新)
        """
        # 步骤 1: 将 FP32 主权重复制到模型的 FP16 参数
        self._copy_master_to_model()

        # 步骤 2: FP16 前向传播（模拟）
        # 注意: 在 CPU 上实际没有 FP16 加速，这里仅演示流程
        outputs = self.model(inputs)
        loss = criterion(outputs, targets)

        # 步骤 3: 损失缩放
        if self.scaler is not None:
            scaled_loss = self.scaler.scale(loss)
        else:
            scaled_loss = loss

        # 步骤 4: 反向传播
        self.model.zero_grad()
        scaled_loss.backward()

        # 步骤 5: 将模型梯度复制到 FP32 主权重
        self._copy_grads_to_master()

        # 步骤 6: 反缩放梯度并检查溢出
        if self.scaler is not None:
            grads_normal = self.scaler.unscale_grads(self.optimizer)
            self.scaler.update(grads_normal)
        else:
            grads_normal = True

        # 步骤 7: 如果梯度正常，执行 FP32 参数更新
        if grads_normal:
            self.optimizer.step()
            updated = True
        else:
            updated = False

        self.optimizer.zero_grad()

        return loss.item(), updated

    def _copy_master_to_model(self):
        """将 FP32 主权重复制到模型参数"""
        for model_param, master_weight in zip(
            self.model.parameters(), self.fp32_master_weights
        ):
            model_param.data.copy_(master_weight.data)

    def _copy_grads_to_master(self):
        """将模型参数的梯度复制到 FP32 主权重"""
        for model_param, master_weight in zip(
            self.model.parameters(), self.fp32_master_weights
        ):
            if model_param.grad is not None:
                if master_weight.grad is None:
                    master_weight.grad = model_param.grad.float()
                else:
                    master_weight.grad.data.copy_(model_param.grad.data)


def analyze_numeric_formats():
    """分析不同数值格式的特性"""
    print("=" * 60)
    print("数值格式分析")
    print("=" * 60)

    formats = {
        "FP32": {"bits": 32, "exponent": 8, "mantissa": 23},
        "FP16": {"bits": 16, "exponent": 5, "mantissa": 10},
        "BF16": {"bits": 16, "exponent": 8, "mantissa": 7},
        "FP8 (E4M3)": {"bits": 8, "exponent": 4, "mantissa": 3},
        "FP8 (E5M2)": {"bits": 8, "exponent": 5, "mantissa": 2},
    }

    print(f"\n{'格式':>12} {'位宽':>6} {'指数':>6} {'尾数':>6} {'最大值':>14} {'最小正值':>14}")
    print("-" * 65)

    for name, fmt in formats.items():
        e = fmt["exponent"]
        m = fmt["mantissa"]

        # 最大值 (近似)
        bias = 2 ** (e - 1) - 1
        max_exp = 2 ** e - 1 - bias - 1  # 排除特殊值
        max_val = (2 - 2 ** (-m)) * 2 ** max_exp

        # 最小正正规数
        min_exp = 1 - bias
        min_val = 2 ** min_exp

        print(
            f"{name:>12} {fmt['bits']:>6} {e:>6} {m:>6} "
            f"{max_val:>14.2e} {min_val:>14.2e}"
        )

    # FP16 vs BF16 精度对比
    print("\n--- FP16 vs BF16 精度对比 ---")
    test_values = [1.0, 0.1, 0.001, 1e-4, 1e-6, 100.0, 10000.0, 60000.0]

    print(f"\n{'原始值 (FP32)':>15} {'FP16':>15} {'BF16':>15} {'FP16 误差':>12} {'BF16 误差':>12}")
    print("-" * 72)

    for val in test_values:
        fp32 = torch.tensor(val, dtype=torch.float32)
        fp16 = fp32.half().float()  # FP32 -> FP16 -> FP32
        bf16 = fp32.bfloat16().float()  # FP32 -> BF16 -> FP32

        fp16_err = abs(fp32.item() - fp16.item())
        bf16_err = abs(fp32.item() - bf16.item())

        print(
            f"{val:>15.6f} "
            f"{fp16.item():>15.6f} "
            f"{bf16.item():>15.6f} "
            f"{fp16_err:>12.2e} "
            f"{bf16_err:>12.2e}"
        )

    # 溢出测试
    print("\n--- 溢出行为对比 ---")
    large_values = [65504.0, 65505.0, 100000.0]
    for val in large_values:
        fp32 = torch.tensor(val, dtype=torch.float32)
        fp16 = fp32.half()
        bf16 = fp32.bfloat16()
        print(f"  值={val}: FP16={'inf' if torch.isinf(fp16) else fp16.item():.1f}, "
              f"BF16={bf16.float().item():.1f}")


def demo_loss_scaling():
    """演示动态损失缩放"""
    print("\n" + "=" * 60)
    print("动态损失缩放演示")
    print("=" * 60)

    scaler = DynamicLossScaler(
        init_scale=2 ** 8,
        growth_interval=5,
    )

    print(f"\n初始缩放因子: {scaler.cur_scale}")
    print(f"增长间隔: {scaler.growth_interval}")

    # 模拟正常步骤和溢出步骤
    steps = [
        (True, "正常"),
        (True, "正常"),
        (True, "正常"),
        (True, "正常"),
        (True, "正常"),  # 达到增长间隔，缩放因子增大
        (True, "正常"),
        (False, "溢出"),  # 检测到溢出，缩放因子缩小
        (True, "正常"),
        (True, "正常"),
        (True, "正常"),
    ]

    print(f"\n{'步骤':>6} {'状态':>8} {'缩放因子':>14} {'连续正常':>10} {'溢出次数':>10}")
    print("-" * 52)

    for i, (normal, status) in enumerate(steps):
        scaler.update(normal)
        print(
            f"{i + 1:>6} {status:>8} {scaler.cur_scale:>14.1f} "
            f"{scaler.consecutive_normal:>10} {scaler.overflow_count:>10}"
        )


def demo_mixed_precision_training():
    """演示完整的混合精度训练流程"""
    print("\n" + "=" * 60)
    print("混合精度训练演示")
    print("=" * 60)

    # 创建模型和数据
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    )

    X = torch.randn(128, 32)
    y = torch.randint(0, 10, (128,))
    criterion = nn.CrossEntropyLoss()

    # 创建混合精度训练器
    trainer = ManualMixedPrecisionTrainer(model, lr=1e-3, use_loss_scaling=True)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"初始损失缩放因子: {trainer.scaler.cur_scale}")

    # 训练
    print(f"\n{'步骤':>6} {'损失':>10} {'已更新':>8} {'缩放因子':>14}")
    print("-" * 42)

    for step in range(20):
        # 随机采样一个 batch
        idx = torch.randint(0, len(X), (32,))
        loss, updated = trainer.train_step(X[idx], y[idx], criterion)

        if step % 4 == 0 or not updated:
            print(
                f"{step + 1:>6} {loss:>10.4f} {'Y' if updated else 'N':>8} "
                f"{trainer.scaler.cur_scale:>14.1f}"
            )

    print(f"\n训练完成!")
    print(f"  总溢出次数: {trainer.scaler.overflow_count}")
    print(f"  最终缩放因子: {trainer.scaler.cur_scale}")


if __name__ == "__main__":
    analyze_numeric_formats()
    demo_loss_scaling()
    demo_mixed_precision_training()
