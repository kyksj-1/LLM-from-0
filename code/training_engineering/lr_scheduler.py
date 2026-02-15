"""
学习率调度器实现：Cosine Decay / WSD / 多阶段调度

本模块实现了 LLM 预训练中常用的学习率调度策略。

核心调度策略:

1. Warmup + Cosine Decay（最经典）:
   - Warmup 阶段: lr 从 0 线性增长到 lr_max
   - Cosine 衰减: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
   - 用于 GPT-3, LLaMA 等

2. WSD (Warmup-Stable-Decay):
   - Warmup: lr 从 0 线性增长到 lr_max
   - Stable: lr 保持 lr_max 不变
   - Decay: lr 快速衰减到 lr_min
   - 优势: 可在任意时刻决定停止训练，进入 decay 阶段
   - 用于 MiniCPM, DeepSeek-V3

3. 多阶段调度:
   - 用于继续预训练、长上下文扩展等场景
   - 不同阶段使用不同的学习率和调度策略

参考:
- Loshchilov & Hutter (2017). SGDR: Stochastic Gradient Descent with Warm Restarts.
- Hu et al. (2024). MiniCPM: Unveiling the Potential of Small LMs with Scalable Training.
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report.
"""

import math
from typing import Optional, List, Dict
from abc import ABC, abstractmethod

import torch


class LRScheduler(ABC):
    """
    学习率调度器基类

    所有调度器需实现 get_lr(step) 方法，返回当前步数对应的学习率。
    """

    def __init__(self, optimizer: torch.optim.Optimizer, last_step: int = -1):
        self.optimizer = optimizer
        self.last_step = last_step
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    @abstractmethod
    def get_lr(self, step: int) -> float:
        """根据当前步数返回学习率"""
        pass

    def step(self, step: Optional[int] = None) -> float:
        """
        更新优化器的学习率

        Args:
            step: 当前步数（如不指定，自动递增）

        Returns:
            当前学习率
        """
        if step is None:
            self.last_step += 1
            step = self.last_step
        else:
            self.last_step = step

        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    def state_dict(self) -> Dict:
        """返回调度器状态（用于 checkpoint）"""
        return {"last_step": self.last_step}

    def load_state_dict(self, state_dict: Dict) -> None:
        """加载调度器状态"""
        self.last_step = state_dict["last_step"]


class CosineAnnealingScheduler(LRScheduler):
    """
    Warmup + Cosine Decay 调度器

    学习率变化:
    1. Warmup 阶段 (0 <= t < warmup_steps):
       lr(t) = lr_max * t / warmup_steps

    2. Cosine Decay 阶段 (warmup_steps <= t <= total_steps):
       lr(t) = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress))
       其中 progress = (t - warmup_steps) / (total_steps - warmup_steps)

    Warmup 的理论依据:
    - 训练初期，模型参数是随机初始化的
    - 梯度方向可能不可靠（指向噪声方向）
    - 用大学习率更新会导致参数跑到 loss landscape 的不好区域
    - Warmup 让模型先以小学习率"摸索"，找到好的方向后再加速

    Args:
        optimizer: PyTorch 优化器
        warmup_steps: 预热步数（通常为总步数的 0.1%-1%）
        total_steps: 总训练步数
        lr_max: 最大学习率
        lr_min: 最小学习率（通常为 lr_max 的 1/10 到 1/100）
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_max: float = 3e-4,
        lr_min: float = 0.0,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        super().__init__(optimizer)

    def get_lr(self, step: int) -> float:
        """
        计算第 step 步的学习率

        Cosine 衰减公式:
          eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T))

        当 t=0: cos(0) = 1 -> eta = eta_max
        当 t=T: cos(pi) = -1 -> eta = eta_min
        中间: 平滑过渡
        """
        if step < self.warmup_steps:
            # 线性预热: lr 从 0 增长到 lr_max
            return self.lr_max * step / max(self.warmup_steps, 1)
        elif step >= self.total_steps:
            return self.lr_min
        else:
            # Cosine 衰减
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                1.0 + math.cos(math.pi * progress)
            )


class WSDScheduler(LRScheduler):
    """
    WSD (Warmup-Stable-Decay) 调度器

    三阶段策略:
    1. Warmup: lr 从 0 线性增长到 lr_max
    2. Stable: lr 保持 lr_max
    3. Decay: lr 从 lr_max 快速衰减到 lr_min

    核心优势:
    - 传统 Cosine 调度需要预先确定总训练步数
    - WSD 的 Stable 阶段可以无限延续
    - 当决定停止训练时，切入 Decay 阶段即可
    - 这使得训练时间可以灵活调整

    与 Cosine 调度的训练曲线对比:
    - Cosine: 学习率持续下降，早期下降较慢，后期加速
    - WSD: 大部分时间保持高学习率，仅最后阶段快速衰减
    - 实验表明两者最终性能接近，但 WSD 更灵活

    Args:
        optimizer: PyTorch 优化器
        warmup_steps: 预热步数
        stable_steps: 稳定阶段步数
        decay_steps: 衰减阶段步数
        lr_max: 最大学习率
        lr_min: 最小学习率
        decay_type: 衰减类型 ("cosine", "linear", "exponential")
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        lr_max: float = 3e-4,
        lr_min: float = 0.0,
        decay_type: str = "cosine",
    ):
        self.warmup_steps = warmup_steps
        self.stable_steps = stable_steps
        self.decay_steps = decay_steps
        self.total_steps = warmup_steps + stable_steps + decay_steps
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.decay_type = decay_type
        super().__init__(optimizer)

    def get_lr(self, step: int) -> float:
        """
        计算第 step 步的学习率

        三阶段:
          [0, warmup_steps): 线性预热
          [warmup_steps, warmup_steps + stable_steps): 恒定高学习率
          [warmup_steps + stable_steps, total_steps]: 衰减
        """
        if step < self.warmup_steps:
            # 阶段 1: 线性预热
            return self.lr_max * step / max(self.warmup_steps, 1)
        elif step < self.warmup_steps + self.stable_steps:
            # 阶段 2: 稳定
            return self.lr_max
        else:
            # 阶段 3: 衰减
            decay_progress = (step - self.warmup_steps - self.stable_steps) / max(
                self.decay_steps, 1
            )
            decay_progress = min(decay_progress, 1.0)

            if self.decay_type == "cosine":
                # Cosine 衰减（最常用）
                decay_factor = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
            elif self.decay_type == "linear":
                # 线性衰减
                decay_factor = 1.0 - decay_progress
            elif self.decay_type == "exponential":
                # 指数衰减
                decay_factor = math.exp(-5.0 * decay_progress)
            else:
                raise ValueError(f"未知的衰减类型: {self.decay_type}")

            return self.lr_min + (self.lr_max - self.lr_min) * decay_factor


class MultiStageScheduler(LRScheduler):
    """
    多阶段学习率调度器

    用于多阶段预训练场景:
    - 标准预训练 -> 继续预训练
    - 预训练 -> 长上下文扩展
    - 预训练 -> 数据退火

    每个阶段可以独立配置:
    - 起始/结束学习率
    - warmup 步数
    - 衰减类型

    Args:
        optimizer: PyTorch 优化器
        stages: 阶段配置列表，每个阶段是一个字典:
            {
                "steps": int,       # 该阶段的总步数
                "lr_start": float,  # 起始学习率
                "lr_end": float,    # 结束学习率
                "warmup_steps": int, # 预热步数（默认 0）
                "decay_type": str,  # 衰减类型（默认 "cosine"）
            }
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        stages: List[Dict],
    ):
        self.stages = stages
        # 计算每个阶段的起止步数
        self.stage_boundaries = []
        cumulative = 0
        for stage in stages:
            start = cumulative
            end = cumulative + stage["steps"]
            self.stage_boundaries.append((start, end))
            cumulative = end
        self.total_steps = cumulative
        super().__init__(optimizer)

    def get_lr(self, step: int) -> float:
        """
        计算第 step 步的学习率

        根据 step 确定当前所在阶段，然后在该阶段内应用调度。
        """
        # 确定当前阶段
        for i, (start, end) in enumerate(self.stage_boundaries):
            if step < end:
                stage = self.stages[i]
                local_step = step - start
                stage_steps = stage["steps"]
                lr_start = stage["lr_start"]
                lr_end = stage["lr_end"]
                warmup_steps = stage.get("warmup_steps", 0)
                decay_type = stage.get("decay_type", "cosine")

                if local_step < warmup_steps:
                    # 阶段内预热
                    return lr_start * local_step / max(warmup_steps, 1)
                else:
                    # 阶段内衰减
                    progress = (local_step - warmup_steps) / max(
                        stage_steps - warmup_steps, 1
                    )
                    progress = min(progress, 1.0)
                    if decay_type == "cosine":
                        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                    elif decay_type == "linear":
                        factor = 1.0 - progress
                    else:
                        factor = 1.0 - progress
                    return lr_end + (lr_start - lr_end) * factor

        # 超过总步数，返回最后一个阶段的结束学习率
        return self.stages[-1]["lr_end"]


def visualize_schedules(total_steps: int = 10000, save_path: Optional[str] = None):
    """
    可视化不同学习率调度策略的对比

    Args:
        total_steps: 总训练步数
        save_path: 保存路径（可选）
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
    except ImportError:
        print("需要安装 matplotlib: pip install matplotlib")
        return

    # 创建 dummy 优化器
    dummy_param = torch.nn.Parameter(torch.zeros(1))
    warmup_steps = int(total_steps * 0.01)  # 1% warmup

    # 1. Cosine 调度
    opt1 = torch.optim.SGD([dummy_param], lr=3e-4)
    cosine = CosineAnnealingScheduler(
        opt1, warmup_steps=warmup_steps, total_steps=total_steps,
        lr_max=3e-4, lr_min=3e-5,
    )

    # 2. WSD 调度（Cosine 衰减）
    opt2 = torch.optim.SGD([dummy_param], lr=3e-4)
    wsd = WSDScheduler(
        opt2, warmup_steps=warmup_steps,
        stable_steps=int(total_steps * 0.7),
        decay_steps=int(total_steps * 0.29),
        lr_max=3e-4, lr_min=3e-5, decay_type="cosine",
    )

    # 3. 多阶段调度
    opt3 = torch.optim.SGD([dummy_param], lr=3e-4)
    multi = MultiStageScheduler(
        opt3, stages=[
            {"steps": int(total_steps * 0.6), "lr_start": 3e-4, "lr_end": 3e-5,
             "warmup_steps": warmup_steps, "decay_type": "cosine"},
            {"steps": int(total_steps * 0.3), "lr_start": 1e-4, "lr_end": 1e-5,
             "warmup_steps": int(total_steps * 0.01), "decay_type": "cosine"},
            {"steps": int(total_steps * 0.1), "lr_start": 1e-5, "lr_end": 1e-6,
             "warmup_steps": 0, "decay_type": "linear"},
        ]
    )

    # 计算学习率曲线
    steps = list(range(total_steps))
    cosine_lrs = [cosine.get_lr(s) for s in steps]
    wsd_lrs = [wsd.get_lr(s) for s in steps]
    multi_lrs = [multi.get_lr(s) for s in steps]

    # 绘图
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(steps, cosine_lrs, label="Cosine Decay", linewidth=2)
    ax.plot(steps, wsd_lrs, label="WSD (Warmup-Stable-Decay)", linewidth=2)
    ax.plot(steps, multi_lrs, label="Multi-Stage", linewidth=2, linestyle="--")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedules Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"图表已保存到: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    print("=" * 60)
    print("学习率调度器 演示")
    print("=" * 60)

    total_steps = 10000
    warmup_steps = 100

    # 创建 dummy 优化器
    dummy_param = torch.nn.Parameter(torch.zeros(1))

    # 1. Cosine Annealing
    print("\n--- Cosine Annealing ---")
    opt = torch.optim.SGD([dummy_param], lr=3e-4)
    scheduler = CosineAnnealingScheduler(
        opt, warmup_steps=warmup_steps, total_steps=total_steps,
        lr_max=3e-4, lr_min=3e-5,
    )
    key_steps = [0, 50, 100, 1000, 5000, 9000, 10000]
    for s in key_steps:
        lr = scheduler.get_lr(s)
        print(f"  step {s:>6d}: lr = {lr:.6f}")

    # 2. WSD 调度
    print("\n--- WSD (Warmup-Stable-Decay) ---")
    opt2 = torch.optim.SGD([dummy_param], lr=3e-4)
    wsd = WSDScheduler(
        opt2, warmup_steps=100, stable_steps=7000, decay_steps=2900,
        lr_max=3e-4, lr_min=3e-5, decay_type="cosine",
    )
    for s in key_steps:
        lr = wsd.get_lr(s)
        print(f"  step {s:>6d}: lr = {lr:.6f}")

    # 3. 多阶段调度（模拟: 预训练 -> 继续预训练 -> 退火）
    print("\n--- Multi-Stage (预训练 -> 继续预训练 -> 退火) ---")
    opt3 = torch.optim.SGD([dummy_param], lr=3e-4)
    multi = MultiStageScheduler(
        opt3, stages=[
            {"steps": 6000, "lr_start": 3e-4, "lr_end": 3e-5,
             "warmup_steps": 100, "decay_type": "cosine"},
            {"steps": 3000, "lr_start": 1e-4, "lr_end": 1e-5,
             "warmup_steps": 50, "decay_type": "cosine"},
            {"steps": 1000, "lr_start": 1e-5, "lr_end": 1e-6,
             "warmup_steps": 0, "decay_type": "linear"},
        ]
    )
    for s in [0, 50, 100, 3000, 6000, 6050, 8000, 9000, 9500, 10000]:
        lr = multi.get_lr(s)
        print(f"  step {s:>6d}: lr = {lr:.6f}")

    # 4. 可视化
    print("\n--- 生成对比图 ---")
    try:
        visualize_schedules(total_steps=10000, save_path=None)
    except Exception as e:
        print(f"可视化跳过（可能无图形界面）: {e}")
        print("可调用 visualize_schedules(save_path='lr_schedules.png') 保存图片")

    # 5. State Dict 测试
    print("\n--- 调度器状态保存/恢复 ---")
    opt4 = torch.optim.SGD([dummy_param], lr=3e-4)
    sched = CosineAnnealingScheduler(
        opt4, warmup_steps=100, total_steps=10000,
        lr_max=3e-4, lr_min=3e-5,
    )
    for _ in range(500):
        sched.step()
    state = sched.state_dict()
    print(f"保存状态: step = {state['last_step']}")

    # 模拟恢复
    opt5 = torch.optim.SGD([dummy_param], lr=3e-4)
    sched2 = CosineAnnealingScheduler(
        opt5, warmup_steps=100, total_steps=10000,
        lr_max=3e-4, lr_min=3e-5,
    )
    sched2.load_state_dict(state)
    lr_before = sched.get_lr(501)
    lr_after = sched2.get_lr(501)
    print(f"恢复后 step 501 的 lr: {lr_before:.6f} == {lr_after:.6f} -> {lr_before == lr_after}")
