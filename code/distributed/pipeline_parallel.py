"""
流水线并行 (Pipeline Parallelism) 简化实现

本模块实现 GPipe 和 1F1B 两种流水线调度策略:
    - GPipe: 所有 micro-batch 先前向再反向
    - 1F1B: 交替执行前向和反向以降低显存峰值

核心概念:
    - 流水线阶段 (Stage): 模型的连续几层
    - Micro-batch: 将一个 batch 拆分为多个小 batch
    - 气泡 (Bubble): GPU 空闲等待的时间
    - 气泡率: bubble_ratio = (p-1) / (m+p-1)

注意: 本实现在单进程中模拟多阶段的流水线并行。
    实际使用时应结合 torch.distributed 的 Send/Recv 操作。

作者: LLM Tutorial
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class PipelineConfig:
    """
    流水线并行配置

    Attributes:
        num_stages: 流水线阶段数（等于使用的 GPU 数量）
        num_micro_batches: micro-batch 数量
        layers_per_stage: 每个阶段包含的层数
        d_model: 模型维度
        d_ff: FFN 隐藏维度
    """
    num_stages: int = 4
    num_micro_batches: int = 8
    layers_per_stage: int = 2
    d_model: int = 256
    d_ff: int = 512


class SimpleTransformerLayer(nn.Module):
    """
    简化的 Transformer 层

    用于流水线并行的演示。包含:
    - LayerNorm + 线性变换 (模拟注意力)
    - LayerNorm + FFN
    """

    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: 模型维度
            d_ff: FFN 隐藏维度
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn_proxy = nn.Linear(d_model, d_model)  # 简化的注意力
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 (Pre-Norm 风格)"""
        x = x + self.attn_proxy(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class PipelineStage(nn.Module):
    """
    流水线的单个阶段

    包含若干个连续的 Transformer 层。
    在实际环境中，每个 Stage 运行在不同的 GPU 上。
    """

    def __init__(self, stage_id: int, layers_per_stage: int, d_model: int, d_ff: int):
        """
        Args:
            stage_id: 阶段编号
            layers_per_stage: 该阶段包含的层数
            d_model: 模型维度
            d_ff: FFN 隐藏维度
        """
        super().__init__()
        self.stage_id = stage_id
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(d_model, d_ff) for _ in range(layers_per_stage)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """依次通过该阶段的所有层"""
        for layer in self.layers:
            x = layer(x)
        return x


class GPipeSchedule:
    """
    GPipe 流水线调度

    调度策略:
        1. 所有 micro-batch 依次通过所有 stage 的前向传播
        2. 所有 micro-batch 以相反顺序通过所有 stage 的反向传播

    气泡率: (p-1) / (m+p-1)
        其中 p = 阶段数, m = micro-batch 数

    缺点:
        峰值显存 = m * 激活值 (需要保存所有 micro-batch 的激活)
    """

    def __init__(self, stages: List[PipelineStage], num_micro_batches: int):
        """
        Args:
            stages: 流水线阶段列表
            num_micro_batches: micro-batch 数量
        """
        self.stages = stages
        self.num_stages = len(stages)
        self.num_micro_batches = num_micro_batches

    def forward(self, micro_batches: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        GPipe 前向传播

        Args:
            micro_batches: micro-batch 列表

        Returns:
            每个 micro-batch 的最终输出
        """
        # 存储中间激活值（用于反向传播）
        # activations[stage][micro_batch] = 该 stage 的输出
        activations = [[None] * self.num_micro_batches for _ in range(self.num_stages + 1)]

        # 存储输入
        for mb_id, mb in enumerate(micro_batches):
            activations[0][mb_id] = mb

        # 记录调度顺序（用于可视化）
        schedule = []

        # 前向传播: 所有 micro-batch 依次通过每个 stage
        for mb_id in range(self.num_micro_batches):
            for stage_id in range(self.num_stages):
                # 计算时间步
                time_step = mb_id + stage_id
                schedule.append((time_step, stage_id, f"F{mb_id + 1}"))

                input_tensor = activations[stage_id][mb_id]
                output = self.stages[stage_id](input_tensor)
                activations[stage_id + 1][mb_id] = output

        # 收集最终输出
        outputs = [activations[self.num_stages][mb_id] for mb_id in range(self.num_micro_batches)]

        return outputs, schedule

    def compute_bubble_ratio(self) -> float:
        """计算气泡率"""
        p = self.num_stages
        m = self.num_micro_batches
        return (p - 1) / (m + p - 1)

    def get_schedule_visualization(self) -> str:
        """
        生成调度可视化字符串

        Returns:
            文本形式的调度时间线
        """
        p = self.num_stages
        m = self.num_micro_batches

        # 总时间步 = 前向 (m + p - 1) + 反向 (m + p - 1)
        total_steps = 2 * (m + p - 1)

        # 初始化调度表
        schedule_table = [[".."] * total_steps for _ in range(p)]

        # 前向传播调度
        for mb_id in range(m):
            for stage_id in range(p):
                time_step = mb_id + stage_id
                schedule_table[stage_id][time_step] = f"F{mb_id + 1}"

        # 反向传播调度 (GPipe: 所有前向完成后再反向)
        forward_end = m + p - 2  # 最后一个前向完成的时间步
        for mb_id in range(m - 1, -1, -1):
            for stage_id in range(p - 1, -1, -1):
                time_step = forward_end + 1 + (m - 1 - mb_id) + (p - 1 - stage_id)
                if time_step < total_steps:
                    schedule_table[stage_id][time_step] = f"B{mb_id + 1}"

        # 格式化输出
        lines = []
        header = "时间步  " + "  ".join(f"{t:>4}" for t in range(total_steps))
        lines.append(header)
        lines.append("-" * len(header))

        for stage_id in range(p):
            row = f"GPU {stage_id}:  " + "  ".join(
                f"{schedule_table[stage_id][t]:>4}" for t in range(total_steps)
            )
            lines.append(row)

        return "\n".join(lines)


class OneFOneBSchedule:
    """
    1F1B 流水线调度

    调度策略:
        1. 预热阶段: 逐步填充流水线（只有前向）
        2. 稳态阶段: 交替执行前向和反向
        3. 冷却阶段: 处理剩余的反向传播

    优势:
        峰值显存 = p * 激活值 (而非 GPipe 的 m * 激活值)
        气泡率与 GPipe 相同: (p-1) / (m+p-1)
    """

    def __init__(self, stages: List[PipelineStage], num_micro_batches: int):
        """
        Args:
            stages: 流水线阶段列表
            num_micro_batches: micro-batch 数量
        """
        self.stages = stages
        self.num_stages = len(stages)
        self.num_micro_batches = num_micro_batches

    def get_schedule_visualization(self) -> str:
        """
        生成 1F1B 调度可视化

        Returns:
            文本形式的 1F1B 调度时间线
        """
        p = self.num_stages
        m = self.num_micro_batches

        # 总时间步
        total_steps = 2 * (m + p - 1)

        # 初始化
        schedule_table = [[".."] * total_steps for _ in range(p)]

        # 为每个 GPU 生成 1F1B 调度
        for stage_id in range(p):
            t = stage_id  # 该 stage 开始工作的时间步

            # 预热: 执行 (p - stage_id) 个前向
            warmup_fwd = p - stage_id
            fwd_mb = 0
            bwd_mb = 0

            for _ in range(min(warmup_fwd, m)):
                if t < total_steps:
                    schedule_table[stage_id][t] = f"F{fwd_mb + 1}"
                    fwd_mb += 1
                    t += 1

            # 稳态: 1F1B 交替
            while bwd_mb < m and fwd_mb < m:
                if t < total_steps:
                    schedule_table[stage_id][t] = f"B{bwd_mb + 1}"
                    bwd_mb += 1
                    t += 1
                if fwd_mb < m and t < total_steps:
                    schedule_table[stage_id][t] = f"F{fwd_mb + 1}"
                    fwd_mb += 1
                    t += 1

            # 冷却: 剩余反向
            while bwd_mb < m:
                if t < total_steps:
                    schedule_table[stage_id][t] = f"B{bwd_mb + 1}"
                    bwd_mb += 1
                    t += 1

        # 格式化输出
        lines = []
        header = "时间步  " + "  ".join(f"{t:>4}" for t in range(min(total_steps, 20)))
        lines.append(header)
        lines.append("-" * len(header))

        for stage_id in range(p):
            row = f"GPU {stage_id}:  " + "  ".join(
                f"{schedule_table[stage_id][t]:>4}" for t in range(min(total_steps, 20))
            )
            lines.append(row)

        if total_steps > 20:
            lines.append(f"  ... (共 {total_steps} 个时间步)")

        return "\n".join(lines)

    def compute_peak_memory(self) -> int:
        """
        计算 1F1B 调度的峰值激活显存

        Returns:
            峰值显存中需要保存的 micro-batch 数量
        """
        # 1F1B 的峰值显存等于流水线阶段数
        return self.num_stages


def analyze_bubble_ratio():
    """分析不同配置下的气泡率"""
    print("=" * 60)
    print("气泡率分析")
    print("=" * 60)

    print("\n气泡率公式: bubble = (p-1) / (m+p-1)")
    print("其中 p = 阶段数 (GPU 数), m = micro-batch 数")

    print(f"\n{'p (阶段数)':>12} {'m (micro-batch)':>16} {'气泡率':>10} {'GPU 利用率':>12}")
    print("-" * 55)

    for p in [2, 4, 8, 16]:
        for m in [p, 2 * p, 4 * p, 8 * p, 16 * p]:
            bubble = (p - 1) / (m + p - 1)
            utilization = 1 - bubble
            print(f"{p:>12} {m:>16} {bubble:>10.1%} {utilization:>12.1%}")
        print()


def demo_gpipe():
    """演示 GPipe 调度"""
    print("=" * 60)
    print("GPipe 调度演示")
    print("=" * 60)

    config = PipelineConfig(
        num_stages=4,
        num_micro_batches=4,
        layers_per_stage=2,
        d_model=128,
        d_ff=256,
    )

    # 创建流水线阶段
    stages = [
        PipelineStage(i, config.layers_per_stage, config.d_model, config.d_ff)
        for i in range(config.num_stages)
    ]

    # 创建 micro-batches
    batch_size = 8
    seq_len = 16
    micro_batch_size = batch_size // config.num_micro_batches

    torch.manual_seed(42)
    micro_batches = [
        torch.randn(micro_batch_size, seq_len, config.d_model)
        for _ in range(config.num_micro_batches)
    ]

    # GPipe 调度
    gpipe = GPipeSchedule(stages, config.num_micro_batches)

    print(f"\n配置: {config.num_stages} 阶段, {config.num_micro_batches} micro-batches")
    print(f"气泡率: {gpipe.compute_bubble_ratio():.1%}")
    print(f"峰值激活显存: {config.num_micro_batches} micro-batches")

    # 执行前向传播
    outputs, schedule = gpipe.forward(micro_batches)

    print(f"\n输出数量: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Micro-batch {i + 1} 输出形状: {out.shape}")

    # 显示调度可视化
    print(f"\nGPipe 调度时间线:")
    print(gpipe.get_schedule_visualization())


def demo_1f1b():
    """演示 1F1B 调度"""
    print("\n" + "=" * 60)
    print("1F1B 调度演示")
    print("=" * 60)

    config = PipelineConfig(
        num_stages=4,
        num_micro_batches=8,
        layers_per_stage=2,
        d_model=128,
        d_ff=256,
    )

    stages = [
        PipelineStage(i, config.layers_per_stage, config.d_model, config.d_ff)
        for i in range(config.num_stages)
    ]

    schedule_1f1b = OneFOneBSchedule(stages, config.num_micro_batches)

    bubble_ratio = (config.num_stages - 1) / (config.num_micro_batches + config.num_stages - 1)

    print(f"\n配置: {config.num_stages} 阶段, {config.num_micro_batches} micro-batches")
    print(f"气泡率: {bubble_ratio:.1%}")
    print(f"GPipe 峰值显存: {config.num_micro_batches} micro-batches 的激活")
    print(f"1F1B 峰值显存: {schedule_1f1b.compute_peak_memory()} micro-batches 的激活")
    print(f"显存节省: {1 - schedule_1f1b.compute_peak_memory() / config.num_micro_batches:.1%}")

    print(f"\n1F1B 调度时间线:")
    print(schedule_1f1b.get_schedule_visualization())


def compare_schedules():
    """对比 GPipe 和 1F1B 的显存占用"""
    print("\n" + "=" * 60)
    print("GPipe vs 1F1B 显存对比")
    print("=" * 60)

    print(f"\n{'阶段数(p)':>10} {'micro-batch(m)':>16} {'GPipe 峰值':>12} {'1F1B 峰值':>12} {'节省':>8}")
    print("-" * 62)

    for p in [4, 8]:
        for m in [4, 8, 16, 32, 64]:
            gpipe_peak = m  # GPipe 需要保存所有 micro-batch 的激活
            one_f_one_b_peak = p  # 1F1B 只需保存 p 个 micro-batch 的激活
            saving = 1 - one_f_one_b_peak / gpipe_peak

            print(
                f"{p:>10} {m:>16} "
                f"{gpipe_peak:>10} mb {one_f_one_b_peak:>10} mb {saving:>8.1%}"
            )
        print()


if __name__ == "__main__":
    analyze_bubble_ratio()
    demo_gpipe()
    demo_1f1b()
    compare_schedules()
