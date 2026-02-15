"""
学习率调度器

知识依赖:
- 模块 8C（训练工程）: 学习率调度策略

参考实现:
- code/training_engineering/lr_scheduler.py

Cosine Schedule with Linear Warmup:

    阶段 1 - 线性预热 (0 → warmup_steps):
        lr(t) = peak_lr * (t / warmup_steps)

    阶段 2 - 余弦退火 (warmup_steps → max_steps):
        progress = (t - warmup_steps) / (max_steps - warmup_steps)
        lr(t) = min_lr + 0.5 * (peak_lr - min_lr) * (1 + cos(π * progress))

    其中 min_lr = peak_lr * min_lr_ratio

    为什么用 Cosine Schedule?
    - 比恒定学习率更好: 后期小学习率有助于精细收敛
    - 比 Step Decay 更平滑: 避免学习率骤降导致的训练不稳定
    - 是 Llama、GPT-3、PaLM 等大模型的标准选择
"""

import math
from typing import Optional


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    max_steps: int,
    min_lr_ratio: float = 0.1,
):
    """
    创建带线性预热的余弦学习率调度器

    Args:
        optimizer: PyTorch 优化器
        warmup_steps: 预热步数
        max_steps: 总训练步数
        min_lr_ratio: 最小学习率与峰值学习率的比值

    Returns:
        torch.optim.lr_scheduler.LambdaLR 实例

    实现提示:
        使用 LambdaLR，定义一个 lr_lambda 函数:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + cos(pi * progress))
    """
    raise NotImplementedError(
        "TODO: 实现 Cosine Schedule with Warmup\n"
        "参考: 模块 8C (训练工程) 的学习率调度章节\n"
        "参考实现: code/training_engineering/lr_scheduler.py\n"
        "提示: torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)"
    )


def get_constant_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
):
    """
    带预热的恒定学习率（SFT/DPO 可用）

    预热阶段后保持学习率不变，适用于训练步数较少的微调阶段。

    Args:
        optimizer: 优化器
        warmup_steps: 预热步数

    Returns:
        LambdaLR 调度器
    """
    raise NotImplementedError(
        "TODO: 实现恒定学习率 + 预热\n"
        "提示: warmup 后 lr_lambda 返回 1.0"
    )
