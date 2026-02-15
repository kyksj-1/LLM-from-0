"""
预训练训练器

知识依赖:
- 模块 8C（训练工程）: 训练循环、梯度累积、混合精度、checkpoint
- 模块 9（分布式训练）: DDP/FSDP 封装

参考实现:
- code/training_engineering/trainer.py
- code/training_engineering/checkpointing.py

预训练训练循环:

    for step in range(max_steps):
        # 梯度累积: 累积 gradient_accumulation_steps 个 micro-batch
        for micro_step in range(gradient_accumulation_steps):
            batch = next(dataloader)
            with autocast(bf16):
                loss = model(batch) / gradient_accumulation_steps
            scaler.scale(loss).backward()

        # 梯度裁剪
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_grad_norm)

        # 优化器更新
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()

        # 日志 & checkpoint
        if step % log_interval == 0: log(loss, lr, ...)
        if step % save_interval == 0: save_checkpoint(...)
        if step % eval_interval == 0: evaluate(...)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from pathlib import Path

from ..model.config import ModelConfig, TrainingConfig


class Trainer:
    """
    预训练训练器

    封装完整的预训练循环: 前向传播 → 反向传播 → 梯度累积 → 优化器更新。

    Args:
        model: 语言模型
        train_loader: 训练数据 DataLoader
        val_loader: 验证数据 DataLoader
        model_config: 模型配置
        train_config: 训练配置
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        model_config: ModelConfig,
        train_config: TrainingConfig,
    ):
        # TODO: 初始化以下组件
        #   - 优化器: AdamW（注意排除 norm 和 bias 的 weight decay）
        #   - 学习率调度器: cosine with warmup
        #   - 混合精度: torch.cuda.amp.GradScaler (如果使用 bf16/fp16)
        #   - 日志: wandb 或 tensorboard
        raise NotImplementedError(
            "TODO: 初始化训练器\n"
            "参考: 模块 8C (训练工程) 的训练器设计章节\n"
            "参考实现: code/training_engineering/trainer.py\n"
            "关键: AdamW 需要区分 decay 和 no_decay 参数组"
        )

    def training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        单步训练（含梯度累积）

        Args:
            batch: {"input_ids": [batch, seq_len], "labels": [batch, seq_len]}

        Returns:
            该步的平均损失值

        实现步骤:
            1. with autocast: loss = model(input_ids, labels)["loss"]
            2. loss = loss / gradient_accumulation_steps
            3. scaler.scale(loss).backward()
            4. (累积够步数后) 梯度裁剪 + optimizer.step() + scheduler.step()
        """
        raise NotImplementedError(
            "TODO: 实现单步训练\n"
            "参考: 模块 8C 的梯度累积章节\n"
            "参考: code/training_engineering/trainer.py"
        )

    def save_checkpoint(self, step: int):
        """
        保存训练检查点

        需要保存:
        - model.state_dict()
        - optimizer.state_dict()
        - scheduler.state_dict()
        - scaler.state_dict()
        - 当前步数 step
        - 配置信息

        Args:
            step: 当前训练步数
        """
        raise NotImplementedError(
            "TODO: 实现 checkpoint 保存\n"
            "参考: code/training_engineering/checkpointing.py"
        )

    def load_checkpoint(self, path: str) -> int:
        """
        加载检查点并恢复训练状态

        Args:
            path: checkpoint 文件路径

        Returns:
            恢复的训练步数
        """
        raise NotImplementedError(
            "TODO: 实现 checkpoint 加载\n"
            "参考: code/training_engineering/checkpointing.py"
        )

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估

        Returns:
            {"val_loss": float, "val_perplexity": float}
        """
        raise NotImplementedError(
            "TODO: 实现验证集评估\n"
            "参考: 模块 8B 的困惑度计算"
        )

    def train(self):
        """
        完整的训练循环

        实现步骤:
            1. 如果有 checkpoint，加载恢复
            2. for step in range(start_step, max_steps):
                a. batch = next(train_loader)
                b. loss = training_step(batch)
                c. 定期日志、评估、保存
            3. 保存最终 checkpoint
        """
        raise NotImplementedError(
            "TODO: 实现完整训练循环\n"
            "参考: code/training_engineering/trainer.py 的 train() 方法"
        )
