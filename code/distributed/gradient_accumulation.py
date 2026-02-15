"""
梯度累积 (Gradient Accumulation)

本模块实现梯度累积训练，用于在显存有限时模拟大 batch size 训练。

核心思想:
    将一个大 batch 拆分为多个小 micro-batch，
    依次计算梯度并累积，最后执行一次参数更新。

数学等价性:
    梯度累积 K 步的梯度等价于使用 K 倍 batch size 的梯度:
    g_accumulated = (1/K) * sum_{k=1}^{K} grad(loss_k)

注意事项:
    1. 损失需要除以累积步数以保持梯度量级
    2. 梯度累积不节省显存（每步仍需存储激活值），但允许使用更小的 micro-batch
    3. BatchNorm 等需要 batch 统计的层可能受影响

作者: LLM Tutorial
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Tuple, List
import time


class GradientAccumulationTrainer:
    """
    梯度累积训练器

    实现带梯度累积的训练循环，支持:
    - 可配置的累积步数
    - 梯度裁剪
    - 学习率调度
    - 训练指标记录
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        accumulation_steps: int = 4,
        max_grad_norm: Optional[float] = 1.0,
    ):
        """
        Args:
            model: 要训练的模型
            optimizer: 优化器
            accumulation_steps: 梯度累积步数
            max_grad_norm: 梯度裁剪的最大范数 (None 表示不裁剪)
        """
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm

        # 训练统计
        self.global_step = 0
        self.micro_step = 0
        self.total_loss = 0.0
        self.losses = []

    def train_step(self, batch_x: torch.Tensor, batch_y: torch.Tensor) -> Tuple[float, bool]:
        """
        执行一步训练（可能只是累积梯度，不更新参数）

        Args:
            batch_x: 输入数据
            batch_y: 目标标签

        Returns:
            (损失值, 是否执行了参数更新)
        """
        # 前向传播
        output = self.model(batch_x)
        loss = F.cross_entropy(output, batch_y)

        # 关键: 损失除以累积步数，保证累积后的梯度量级与大 batch 一致
        scaled_loss = loss / self.accumulation_steps
        scaled_loss.backward()

        self.total_loss += loss.item()
        self.micro_step += 1

        # 判断是否达到累积步数
        if self.micro_step % self.accumulation_steps == 0:
            # 梯度裁剪
            if self.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
            else:
                grad_norm = self._compute_grad_norm()

            # 参数更新
            self.optimizer.step()
            self.optimizer.zero_grad()

            # 记录统计
            avg_loss = self.total_loss / self.accumulation_steps
            self.losses.append(avg_loss)
            self.total_loss = 0.0
            self.global_step += 1

            return avg_loss, True

        return loss.item(), False

    def _compute_grad_norm(self) -> float:
        """计算当前梯度的 L2 范数"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def get_effective_batch_size(self, micro_batch_size: int) -> int:
        """计算等效的全局 batch size"""
        return micro_batch_size * self.accumulation_steps


def verify_gradient_equivalence(
    model_class,
    init_args: tuple,
    large_batch_size: int = 128,
    micro_batch_size: int = 32,
    lr: float = 1e-3,
):
    """
    验证梯度累积与大 batch 训练的等价性

    创建两个相同的模型:
    - 模型 A: 直接使用 large_batch_size 训练
    - 模型 B: 使用 micro_batch_size 训练，累积 large_batch_size / micro_batch_size 步

    验证两者的梯度和参数更新是否一致。

    Args:
        model_class: 模型类
        init_args: 模型初始化参数
        large_batch_size: 大 batch size
        micro_batch_size: 小 micro-batch size
        lr: 学习率
    """
    print("=" * 60)
    print("梯度等价性验证")
    print("=" * 60)

    accumulation_steps = large_batch_size // micro_batch_size
    print(f"\n大 batch size: {large_batch_size}")
    print(f"Micro-batch size: {micro_batch_size}")
    print(f"累积步数: {accumulation_steps}")

    # 创建确定性数据
    torch.manual_seed(42)
    X = torch.randn(large_batch_size, init_args[0])
    y = torch.randint(0, init_args[-1], (large_batch_size,))

    # --- 模型 A: 大 batch ---
    torch.manual_seed(0)
    model_a = model_class(*init_args)
    optimizer_a = torch.optim.SGD(model_a.parameters(), lr=lr)

    optimizer_a.zero_grad()
    output_a = model_a(X)
    loss_a = F.cross_entropy(output_a, y)
    loss_a.backward()

    # 保存模型 A 的梯度
    grads_a = {name: p.grad.clone() for name, p in model_a.named_parameters() if p.grad is not None}

    optimizer_a.step()
    params_a = {name: p.data.clone() for name, p in model_a.named_parameters()}

    # --- 模型 B: 梯度累积 ---
    torch.manual_seed(0)
    model_b = model_class(*init_args)
    optimizer_b = torch.optim.SGD(model_b.parameters(), lr=lr)

    optimizer_b.zero_grad()
    for k in range(accumulation_steps):
        start = k * micro_batch_size
        end = start + micro_batch_size
        output_b = model_b(X[start:end])
        # 关键: 损失除以累积步数
        loss_b = F.cross_entropy(output_b, y[start:end]) / accumulation_steps
        loss_b.backward()

    # 保存模型 B 的梯度
    grads_b = {name: p.grad.clone() for name, p in model_b.named_parameters() if p.grad is not None}

    optimizer_b.step()
    params_b = {name: p.data.clone() for name, p in model_b.named_parameters()}

    # --- 对比 ---
    print(f"\n{'参数名':>30} {'梯度差异':>14} {'参数差异':>14}")
    print("-" * 62)

    max_grad_diff = 0.0
    max_param_diff = 0.0

    for name in grads_a:
        grad_diff = (grads_a[name] - grads_b[name]).abs().max().item()
        param_diff = (params_a[name] - params_b[name]).abs().max().item()
        max_grad_diff = max(max_grad_diff, grad_diff)
        max_param_diff = max(max_param_diff, param_diff)

        print(f"{name:>30} {grad_diff:>14.2e} {param_diff:>14.2e}")

    print(f"\n最大梯度差异: {max_grad_diff:.2e}")
    print(f"最大参数差异: {max_param_diff:.2e}")

    if max_grad_diff < 1e-5:
        print("\n验证通过: 梯度累积与大 batch 训练在数值上等价")
    else:
        print("\n注意: 存在数值差异，可能是浮点精度导致的")


def demo_gradient_accumulation():
    """演示梯度累积训练的完整流程"""
    print("\n" + "=" * 60)
    print("梯度累积训练演示")
    print("=" * 60)

    # 创建数据
    torch.manual_seed(42)
    num_samples = 512
    input_dim = 64
    num_classes = 10

    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)

    # 模型
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )

    # 配置
    micro_batch_size = 16
    accumulation_steps = 8
    effective_batch_size = micro_batch_size * accumulation_steps
    num_epochs = 3

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Micro-batch size: {micro_batch_size}")
    print(f"累积步数: {accumulation_steps}")
    print(f"等效 batch size: {effective_batch_size}")

    # 创建训练器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = GradientAccumulationTrainer(
        model, optimizer,
        accumulation_steps=accumulation_steps,
        max_grad_norm=1.0,
    )

    dataloader = DataLoader(dataset, batch_size=micro_batch_size, shuffle=True)

    # 训练
    print(f"\n{'Epoch':>6} {'Step':>6} {'损失':>10} {'准确率':>10}")
    print("-" * 36)

    for epoch in range(num_epochs):
        model.train()
        for batch_x, batch_y in dataloader:
            loss, updated = trainer.train_step(batch_x, batch_y)

            if updated:
                # 计算训练准确率
                with torch.no_grad():
                    pred = model(X).argmax(dim=-1)
                    acc = (pred == y).float().mean().item()

                if trainer.global_step % 2 == 0:
                    print(f"{epoch + 1:>6} {trainer.global_step:>6} {loss:>10.4f} {acc:>10.1%}")

    print(f"\n训练完成!")
    print(f"  全局步数: {trainer.global_step}")
    print(f"  Micro 步数: {trainer.micro_step}")
    print(f"  最终损失: {trainer.losses[-1]:.4f}")


class SimpleClassifier(nn.Module):
    """简单的分类模型，用于验证实验"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # 验证梯度等价性
    verify_gradient_equivalence(
        model_class=SimpleClassifier,
        init_args=(64, 128, 10),
        large_batch_size=128,
        micro_batch_size=32,
    )

    # 演示梯度累积训练
    demo_gradient_accumulation()
