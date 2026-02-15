"""
ZeRO (Zero Redundancy Optimizer) 简化实现

本模块模拟 ZeRO-1 和 ZeRO-2 的核心思想:
    - ZeRO-1: 优化器状态分片
    - ZeRO-2: + 梯度分片

ZeRO 的核心思想是消除分布式数据并行中的显存冗余:
    标准 DDP: 每个 GPU 存储完整的参数、梯度和优化器状态
    ZeRO-1: 优化器状态按 GPU 数量分片
    ZeRO-2: 优化器状态 + 梯度按 GPU 数量分片
    ZeRO-3: 优化器状态 + 梯度 + 参数按 GPU 数量分片

显存公式 (混合精度 + Adam):
    DDP:    16P
    ZeRO-1: 4P + 12P/n
    ZeRO-2: 2P + 14P/n
    ZeRO-3: 16P/n

注意: 本实现在单进程中模拟，实际使用请参考 DeepSpeed 或 PyTorch FSDP。

作者: LLM Tutorial
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import math
from dataclasses import dataclass


@dataclass
class MemoryEstimate:
    """
    显存估算结果

    Attributes:
        params_gb: 参数占用的显存 (GB)
        gradients_gb: 梯度占用的显存 (GB)
        optimizer_gb: 优化器状态占用的显存 (GB)
        total_gb: 总显存占用 (GB)
    """
    params_gb: float
    gradients_gb: float
    optimizer_gb: float
    total_gb: float

    def __repr__(self):
        return (
            f"MemoryEstimate(\n"
            f"  参数:     {self.params_gb:.2f} GB\n"
            f"  梯度:     {self.gradients_gb:.2f} GB\n"
            f"  优化器:   {self.optimizer_gb:.2f} GB\n"
            f"  总计:     {self.total_gb:.2f} GB\n"
            f")"
        )


def estimate_memory(
    param_count_billion: float,
    num_gpus: int,
    zero_stage: int = 0,
    dtype_bytes: int = 2,
) -> MemoryEstimate:
    """
    估算不同 ZeRO 阶段的显存占用

    假设使用混合精度训练 (FP16 参数 + FP32 主权重 + Adam 优化器)

    显存组成:
        - FP16 模型参数: 2P bytes
        - FP16 梯度: 2P bytes
        - FP32 主权重: 4P bytes
        - Adam m (一阶矩): 4P bytes
        - Adam v (二阶矩): 4P bytes

    Args:
        param_count_billion: 参数量（十亿）
        num_gpus: GPU 数量
        zero_stage: ZeRO 阶段 (0=DDP, 1, 2, 3)
        dtype_bytes: 训练精度字节数（FP16=2）

    Returns:
        显存估算结果
    """
    P = param_count_billion * 1e9  # 参数总数
    n = num_gpus

    if zero_stage == 0:
        # 标准 DDP: 全部冗余
        params_bytes = 2 * P + 4 * P  # FP16 + FP32 主权重
        grad_bytes = 2 * P  # FP16 梯度
        opt_bytes = 8 * P  # Adam m + v (FP32)

    elif zero_stage == 1:
        # ZeRO-1: 分片优化器状态
        params_bytes = 2 * P + 4 * P / n  # FP16 完整 + FP32 分片
        grad_bytes = 2 * P  # 梯度完整
        opt_bytes = 8 * P / n  # 优化器分片

    elif zero_stage == 2:
        # ZeRO-2: + 分片梯度
        params_bytes = 2 * P + 4 * P / n  # FP16 完整 + FP32 分片
        grad_bytes = 2 * P / n  # 梯度分片
        opt_bytes = 8 * P / n  # 优化器分片

    elif zero_stage == 3:
        # ZeRO-3: + 分片参数
        params_bytes = (2 * P + 4 * P) / n  # 全部分片
        grad_bytes = 2 * P / n
        opt_bytes = 8 * P / n

    else:
        raise ValueError(f"不支持的 ZeRO 阶段: {zero_stage}")

    return MemoryEstimate(
        params_gb=params_bytes / 1e9,
        gradients_gb=grad_bytes / 1e9,
        optimizer_gb=opt_bytes / 1e9,
        total_gb=(params_bytes + grad_bytes + opt_bytes) / 1e9,
    )


class SimulatedZeROStage1:
    """
    模拟 ZeRO Stage 1: 优化器状态分片

    核心思想:
        每个 GPU 只维护 1/n 参数对应的优化器状态（Adam 的 m 和 v）。
        梯度和参数仍在每个 GPU 上完整保存。

    训练流程:
        1. 前向/反向传播: 与标准 DDP 相同
        2. All-Reduce: 同步所有 GPU 的梯度
        3. 优化器更新: 每个 GPU 只更新自己负责的参数分片
        4. All-Gather: 将更新后的参数广播到所有 GPU
    """

    def __init__(self, model: nn.Module, num_gpus: int, lr: float = 1e-3):
        """
        Args:
            model: 要训练的模型
            num_gpus: 模拟的 GPU 数量
            lr: 学习率
        """
        self.num_gpus = num_gpus
        self.lr = lr
        self.model = model

        # 将模型参数展平为一个向量
        self.param_groups = list(model.parameters())
        total_params = sum(p.numel() for p in self.param_groups)

        # 分片参数索引
        chunk_size = (total_params + num_gpus - 1) // num_gpus
        self.chunk_sizes = []
        remaining = total_params
        for i in range(num_gpus):
            size = min(chunk_size, remaining)
            self.chunk_sizes.append(size)
            remaining -= size

        # 每个 GPU 只维护自己分片的优化器状态
        self.optimizer_states = []
        for i in range(num_gpus):
            state = {
                "m": torch.zeros(self.chunk_sizes[i]),  # 一阶矩
                "v": torch.zeros(self.chunk_sizes[i]),  # 二阶矩
                "step": 0,
            }
            self.optimizer_states.append(state)

        print(f"ZeRO-1 初始化:")
        print(f"  总参数量: {total_params:,}")
        print(f"  GPU 数量: {num_gpus}")
        print(f"  每 GPU 优化器状态大小: ~{chunk_size:,} 参数")

    def step(self):
        """
        执行一步优化器更新

        模拟流程:
        1. 所有 GPU 已有完整梯度（假设 All-Reduce 已完成）
        2. 每个 GPU 只更新自己负责的参数分片
        3. All-Gather 更新后的参数（模拟为直接写回）
        """
        # 将所有参数和梯度展平
        all_params = torch.cat([p.data.flatten() for p in self.param_groups])
        all_grads = torch.cat([
            p.grad.flatten() if p.grad is not None else torch.zeros_like(p.data.flatten())
            for p in self.param_groups
        ])

        # 每个 GPU 更新自己负责的分片
        offset = 0
        for gpu_id in range(self.num_gpus):
            size = self.chunk_sizes[gpu_id]
            if size == 0:
                continue

            state = self.optimizer_states[gpu_id]
            state["step"] += 1

            # 获取该 GPU 负责的参数和梯度分片
            param_slice = all_params[offset:offset + size]
            grad_slice = all_grads[offset:offset + size]

            # Adam 更新（简化版）
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            state["m"] = beta1 * state["m"] + (1 - beta1) * grad_slice
            state["v"] = beta2 * state["v"] + (1 - beta2) * grad_slice ** 2

            # 偏差校正
            m_hat = state["m"] / (1 - beta1 ** state["step"])
            v_hat = state["v"] / (1 - beta2 ** state["step"])

            # 参数更新
            update = self.lr * m_hat / (torch.sqrt(v_hat) + eps)
            all_params[offset:offset + size] -= update

            offset += size

        # 将更新后的参数写回模型（模拟 All-Gather）
        offset = 0
        for p in self.param_groups:
            numel = p.numel()
            p.data = all_params[offset:offset + numel].reshape(p.shape)
            offset += numel

    def memory_report(self) -> Dict[str, float]:
        """
        报告每个 GPU 的显存使用情况

        Returns:
            各组成部分的显存占用（参数数量）
        """
        total_params = sum(p.numel() for p in self.param_groups)
        avg_opt_params = sum(self.chunk_sizes) / self.num_gpus

        return {
            "完整参数 (FP16)": total_params,
            "完整梯度 (FP16)": total_params,
            "分片优化器 (FP32)": avg_opt_params,
            "分片主权重 (FP32)": avg_opt_params,
        }


class SimulatedZeROStage2(SimulatedZeROStage1):
    """
    模拟 ZeRO Stage 2: 优化器状态 + 梯度分片

    与 ZeRO-1 的区别:
        梯度不再使用 All-Reduce，而是使用 Reduce-Scatter，
        每个 GPU 只保留自己负责分片的梯度。

    训练流程:
        1. 前向传播: 与标准 DDP 相同
        2. 反向传播: 梯度使用 Reduce-Scatter（而非 All-Reduce）
        3. 优化器更新: 每个 GPU 只更新自己的分片
        4. All-Gather: 广播更新后的参数
    """

    def __init__(self, model: nn.Module, num_gpus: int, lr: float = 1e-3):
        super().__init__(model, num_gpus, lr)
        print("  (ZeRO-2: 梯度也分片)")

    def memory_report(self) -> Dict[str, float]:
        """ZeRO-2 的显存报告"""
        total_params = sum(p.numel() for p in self.param_groups)
        avg_opt_params = sum(self.chunk_sizes) / self.num_gpus

        return {
            "完整参数 (FP16)": total_params,
            "分片梯度 (FP16)": avg_opt_params,  # 关键区别
            "分片优化器 (FP32)": avg_opt_params,
            "分片主权重 (FP32)": avg_opt_params,
        }


def compare_memory_across_stages():
    """对比不同 ZeRO 阶段在各种模型规模和 GPU 配置下的显存占用"""
    print("=" * 80)
    print("ZeRO 各阶段显存对比")
    print("=" * 80)

    models = [
        ("7B", 7.0),
        ("13B", 13.0),
        ("70B", 70.0),
        ("175B", 175.0),
    ]

    gpu_configs = [1, 4, 8, 16, 32, 64, 128]

    for model_name, param_b in models:
        print(f"\n--- 模型: {model_name} ({param_b}B 参数) ---")
        print(f"{'GPU 数':>8} {'DDP':>10} {'ZeRO-1':>10} {'ZeRO-2':>10} {'ZeRO-3':>10} {'A100 80GB':>12}")
        print("-" * 65)

        for n_gpu in gpu_configs:
            ddp = estimate_memory(param_b, n_gpu, zero_stage=0)
            z1 = estimate_memory(param_b, n_gpu, zero_stage=1)
            z2 = estimate_memory(param_b, n_gpu, zero_stage=2)
            z3 = estimate_memory(param_b, n_gpu, zero_stage=3)

            # 检查是否能放入 A100 80GB
            fits = []
            for mem in [ddp, z1, z2, z3]:
                fits.append("OK" if mem.total_gb <= 80 else "X")

            print(
                f"{n_gpu:>8} "
                f"{ddp.total_gb:>8.1f}GB "
                f"{z1.total_gb:>8.1f}GB "
                f"{z2.total_gb:>8.1f}GB "
                f"{z3.total_gb:>8.1f}GB "
                f"  {'/'.join(fits)}"
            )


def demo_zero_training():
    """演示 ZeRO 优化器的训练过程"""
    print("\n" + "=" * 60)
    print("ZeRO 训练演示")
    print("=" * 60)

    # 创建简单模型
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数量: {total_params:,}")

    # ZeRO-1 训练
    num_gpus = 4
    zero1 = SimulatedZeROStage1(model, num_gpus=num_gpus)

    # 模拟训练
    X = torch.randn(32, 64)
    y = torch.randint(0, 10, (32,))
    criterion = nn.CrossEntropyLoss()

    print(f"\n训练 5 步:")
    for step in range(5):
        # 前向
        output = model(X)
        loss = criterion(output, y)

        # 反向
        model.zero_grad()
        loss.backward()

        # ZeRO-1 优化器步进
        zero1.step()

        print(f"  Step {step + 1}: loss = {loss.item():.4f}")

    # 显存报告
    print(f"\nZeRO-1 每 GPU 显存报告:")
    for key, val in zero1.memory_report().items():
        print(f"  {key}: {val:,.0f} 参数")

    print(f"\nZeRO-2 每 GPU 显存报告:")
    zero2 = SimulatedZeROStage2(model, num_gpus=num_gpus)
    for key, val in zero2.memory_report().items():
        print(f"  {key}: {val:,.0f} 参数")


def communication_analysis():
    """分析不同 ZeRO 阶段的通信量"""
    print("\n" + "=" * 60)
    print("ZeRO 通信量分析")
    print("=" * 60)

    param_b = 7.0  # 7B 参数
    P = param_b * 1e9
    dtype_size = 2  # FP16

    print(f"\n模型: {param_b}B 参数")
    print(f"数据类型: FP16 ({dtype_size} bytes)")

    print(f"\n{'方案':>10} {'通信量 (每步)':>18} {'通信操作':>20} {'相对 DDP':>12}")
    print("-" * 65)

    # DDP: 一次 All-Reduce
    ddp_comm = 2 * P * dtype_size  # Ring All-Reduce: ~2P
    print(f"{'DDP':>10} {ddp_comm / 1e9:>14.1f} GB {'All-Reduce':>20} {'1.0x':>12}")

    # ZeRO-1: 与 DDP 相同 (All-Reduce 梯度 + All-Gather 参数)
    z1_comm = 2 * P * dtype_size + P * dtype_size  # All-Reduce + All-Gather
    # 实际上 ZeRO-1 只在优化器步骤后需要 All-Gather 更新的参数分片
    # 梯度 All-Reduce 与 DDP 完全相同
    z1_comm = ddp_comm  # 简化: 与 DDP 基本相同
    print(f"{'ZeRO-1':>10} {z1_comm / 1e9:>14.1f} GB {'AR + AG':>20} {'1.0x':>12}")

    # ZeRO-2: Reduce-Scatter + All-Gather
    z2_comm = P * dtype_size + P * dtype_size  # RS + AG
    print(f"{'ZeRO-2':>10} {z2_comm / 1e9:>14.1f} GB {'RS + AG':>20} {z2_comm / ddp_comm:>11.1f}x")

    # ZeRO-3: 前向 AG + 反向 AG + RS + AG
    z3_comm = 3 * P * dtype_size  # 前向 AG + 反向 AG + RS (不含最终 AG)
    print(f"{'ZeRO-3':>10} {z3_comm / 1e9:>14.1f} GB {'2xAG + RS + AG':>20} {z3_comm / ddp_comm:>11.1f}x")

    print(f"\nAR = All-Reduce, AG = All-Gather, RS = Reduce-Scatter")


if __name__ == "__main__":
    compare_memory_across_stages()
    demo_zero_training()
    communication_analysis()
