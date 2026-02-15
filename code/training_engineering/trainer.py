"""
预训练训练器：整合优化器、调度器、检查点、监控等组件

本模块实现一个完整的 LLM 预训练训练器，将以下组件整合为统一的训练循环:
- 优化器 (AdamW / 8-bit Adam)
- 学习率调度器 (Cosine / WSD / 多阶段)
- Checkpoint 管理 (保存/恢复/轮换)
- 训练监控 (损失/梯度/权重追踪)
- 稳定性诊断 (Loss spike 检测、NaN 处理)
- 梯度管理 (裁剪、累积)

训练流程:
  1. 初始化模型、优化器、调度器
  2. (可选) 从 checkpoint 恢复
  3. 训练循环:
     a. 前向传播
     b. 计算损失
     c. 反向传播
     d. (梯度累积: 若未累积够，回到 a)
     e. 梯度裁剪
     f. 优化器更新
     g. 学习率调度
     h. 监控与日志
     i. (按需) 保存 checkpoint
  4. 训练结束，保存最终 checkpoint

梯度累积的等价性证明:
  标准大 batch: g = (1/B) * sum_{i=1}^{B} grad(x_i)
  累积 K 步:    g = (1/K) * sum_{k=1}^{K} [(1/(B/K)) * sum_{i in batch_k} grad(x_i)]
             = (1/B) * sum_{i=1}^{B} grad(x_i)
  数学上完全等价（前提: loss 除以累积步数归一化）

参考:
- Karpathy, nanoGPT
- Megatron-LM Training Framework
"""

import os
import time
import math
from typing import Dict, Optional, Any, Callable, Iterator, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .optimizer import AdamW, separate_weight_decay_params
from .lr_scheduler import CosineAnnealingScheduler, WSDScheduler
from .checkpointing import CheckpointManager
from .monitoring import TrainingMonitor
from .stability_diagnosis import StabilityDiagnoser
from .utils import (
    set_seed, get_rng_states, compute_grad_norm,
    format_training_log, compute_tokens_per_second,
)


@dataclass
class TrainingConfig:
    """
    训练配置

    包含预训练的所有超参数设置。

    Attributes:
        # 基础参数
        max_steps: 最大训练步数
        batch_size: 微批大小（每个 GPU）
        seq_len: 序列长度
        grad_accum_steps: 梯度累积步数（有效 batch = batch_size * grad_accum_steps）

        # 优化器参数
        lr: 最大学习率
        weight_decay: 权重衰减
        beta1: Adam 一阶矩衰减率
        beta2: Adam 二阶矩衰减率
        eps: Adam epsilon
        grad_clip: 梯度裁剪阈值（0 表示不裁剪）

        # 学习率调度
        scheduler_type: 调度器类型 ("cosine", "wsd")
        warmup_steps: 预热步数
        lr_min: 最小学习率

        # WSD 调度器专用
        wsd_stable_steps: WSD 稳定阶段步数
        wsd_decay_steps: WSD 衰减阶段步数

        # Checkpoint
        save_interval: checkpoint 保存间隔
        save_dir: checkpoint 保存目录
        max_checkpoints: 最多保留的 checkpoint 数量

        # 监控
        log_interval: 日志打印间隔
        eval_interval: 评估间隔

        # 其他
        seed: 随机种子
        dtype: 训练精度 ("fp32", "fp16", "bf16")
    """
    # 基础参数
    max_steps: int = 10000
    batch_size: int = 8
    seq_len: int = 1024
    grad_accum_steps: int = 1

    # 优化器参数
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # 学习率调度
    scheduler_type: str = "cosine"
    warmup_steps: int = 100
    lr_min: float = 3e-5

    # WSD 调度器专用
    wsd_stable_steps: int = 7000
    wsd_decay_steps: int = 2900

    # Checkpoint
    save_interval: int = 1000
    save_dir: str = "checkpoints"
    max_checkpoints: int = 5

    # 监控
    log_interval: int = 10
    eval_interval: int = 500

    # 其他
    seed: int = 42
    dtype: str = "fp32"

    @property
    def effective_batch_size(self) -> int:
        """有效批大小 = 微批大小 * 梯度累积步数"""
        return self.batch_size * self.grad_accum_steps


class Trainer:
    """
    LLM 预训练训练器

    整合所有训练组件，提供完整的预训练流程。

    Args:
        model: 要训练的模型
        config: 训练配置
        train_dataloader: 训练数据加载器
        eval_dataloader: 验证数据加载器（可选）
        eval_fn: 自定义评估函数（可选）
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataloader: Optional[DataLoader] = None,
        eval_dataloader: Optional[DataLoader] = None,
        eval_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.eval_fn = eval_fn

        self.device = next(model.parameters()).device

        # 设置随机种子
        set_seed(config.seed)

        # 初始化优化器
        self.optimizer = self._create_optimizer()

        # 初始化学习率调度器
        self.scheduler = self._create_scheduler()

        # 初始化 Checkpoint 管理器
        self.ckpt_manager = CheckpointManager(
            save_dir=config.save_dir,
            max_keep=config.max_checkpoints,
            save_interval_steps=config.save_interval,
        )

        # 初始化监控器
        self.monitor = TrainingMonitor(
            model,
            log_interval=config.log_interval,
        )

        # 初始化诊断器
        self.diagnoser = StabilityDiagnoser(model)

        # 训练状态
        self.global_step = 0
        self.tokens_seen = 0
        self.best_eval_loss = float("inf")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器，自动分离 weight decay 参数组"""
        param_groups = separate_weight_decay_params(
            self.model,
            weight_decay=self.config.weight_decay,
        )
        return AdamW(
            [
                {"params": param_groups[0]["params"], "weight_decay": self.config.weight_decay},
                {"params": param_groups[1]["params"], "weight_decay": 0.0},
            ],
            lr=self.config.lr,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def _create_scheduler(self):
        """创建学习率调度器"""
        if self.config.scheduler_type == "cosine":
            return CosineAnnealingScheduler(
                self.optimizer,
                warmup_steps=self.config.warmup_steps,
                total_steps=self.config.max_steps,
                lr_max=self.config.lr,
                lr_min=self.config.lr_min,
            )
        elif self.config.scheduler_type == "wsd":
            return WSDScheduler(
                self.optimizer,
                warmup_steps=self.config.warmup_steps,
                stable_steps=self.config.wsd_stable_steps,
                decay_steps=self.config.wsd_decay_steps,
                lr_max=self.config.lr,
                lr_min=self.config.lr_min,
            )
        else:
            raise ValueError(f"未知的调度器类型: {self.config.scheduler_type}")

    def _get_batch_iterator(self) -> Iterator:
        """获取无限循环的批数据迭代器"""
        if self.train_dataloader is None:
            raise ValueError("未提供训练数据加载器")
        while True:
            for batch in self.train_dataloader:
                yield batch

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        执行单个训练步（含梯度累积）

        梯度累积等价性:
        - 实际执行 grad_accum_steps 次前向/反向传播
        - 每次损失除以 grad_accum_steps 进行归一化
        - 效果等价于使用 effective_batch_size 的单步更新

        Args:
            batch: 输入批次，需包含 "input_ids" 和 "labels"

        Returns:
            {"loss": 平均损失, "grad_norm": 梯度范数}
        """
        self.model.train()
        total_loss = 0.0

        # 将 batch 拆分为 micro-batches 用于梯度累积
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        micro_batch_size = input_ids.shape[0] // self.config.grad_accum_steps
        if micro_batch_size == 0:
            micro_batch_size = input_ids.shape[0]

        for k in range(self.config.grad_accum_steps):
            start = k * micro_batch_size
            end = min(start + micro_batch_size, input_ids.shape[0])
            if start >= end:
                break

            micro_input = input_ids[start:end]
            micro_labels = labels[start:end]

            # 前向传播
            logits = self.model(micro_input)

            # 计算损失（归一化: 除以累积步数）
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                micro_labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / self.config.grad_accum_steps

            # 反向传播
            loss.backward()
            total_loss += loss.item()

        # 梯度裁剪
        grad_norm = 0.0
        if self.config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.grad_clip
            ).item()
        else:
            grad_norm, _ = compute_grad_norm(self.model)

        # 优化器更新
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        # 学习率调度
        lr = self.scheduler.step(self.global_step)

        return {
            "loss": total_loss,
            "grad_norm": grad_norm,
            "lr": lr,
        }

    def evaluate(self) -> Dict[str, float]:
        """
        在验证集上评估模型

        Returns:
            {"eval_loss": 验证损失, "eval_ppl": 验证困惑度}
        """
        if self.eval_fn is not None:
            return self.eval_fn(self.model)

        if self.eval_dataloader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                n_tokens = (labels != -100).sum().item()
                total_loss += loss.item()
                total_tokens += n_tokens

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))  # 防止溢出

        self.model.train()
        return {"eval_loss": avg_loss, "eval_ppl": ppl}

    def train(self) -> Dict[str, Any]:
        """
        执行完整的训练循环

        Returns:
            训练摘要（最终损失、步数、总时间等）
        """
        print("=" * 60)
        print("开始预训练")
        print(f"  有效批大小: {self.config.effective_batch_size}")
        print(f"  最大步数: {self.config.max_steps}")
        print(f"  学习率: {self.config.lr}")
        print(f"  调度器: {self.config.scheduler_type}")
        print(f"  梯度累积: {self.config.grad_accum_steps} 步")
        print(f"  梯度裁剪: {self.config.grad_clip}")
        print("=" * 60)

        start_time = time.time()
        batch_iter = self._get_batch_iterator()

        for step in range(self.global_step, self.config.max_steps):
            self.global_step = step
            step_start = time.time()

            # 获取数据批次
            batch = next(batch_iter)

            # 训练步
            step_metrics = self.train_step(batch)
            step_time = time.time() - step_start

            # 更新 token 计数
            self.tokens_seen += self.config.effective_batch_size * self.config.seq_len

            # 计算吞吐量
            tokens_per_sec = compute_tokens_per_second(
                self.config.batch_size, self.config.seq_len,
                step_time, self.config.grad_accum_steps,
            )

            # 监控
            report = self.monitor.on_step_end(
                step=step,
                loss=step_metrics["loss"],
                lr=step_metrics["lr"],
                extra_metrics={"grad_norm": step_metrics["grad_norm"]},
            )

            # 告警处理
            for alert in report.get("alerts", []):
                print(f"\n[ALERT] step {step}: {alert['message']}")

            # 日志输出
            if step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                log = format_training_log(
                    step=step,
                    loss=step_metrics["loss"],
                    lr=step_metrics["lr"],
                    grad_norm=step_metrics["grad_norm"],
                    tokens_per_sec=tokens_per_sec,
                    elapsed=elapsed,
                )
                print(log)

            # 评估
            if step > 0 and step % self.config.eval_interval == 0:
                eval_metrics = self.evaluate()
                if eval_metrics:
                    print(f"  [EVAL] step {step}: {eval_metrics}")
                    if eval_metrics.get("eval_loss", float("inf")) < self.best_eval_loss:
                        self.best_eval_loss = eval_metrics["eval_loss"]

            # Checkpoint 保存
            if self.ckpt_manager.should_save(step):
                self.ckpt_manager.save(
                    step=step,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    rng_states=get_rng_states(),
                    extra={
                        "loss": step_metrics["loss"],
                        "tokens_seen": self.tokens_seen,
                    },
                )

        # 训练结束
        total_time = time.time() - start_time
        final_loss = step_metrics["loss"]

        # 保存最终 checkpoint
        self.ckpt_manager.save(
            step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            rng_states=get_rng_states(),
            extra={
                "loss": final_loss,
                "tokens_seen": self.tokens_seen,
                "training_complete": True,
            },
        )

        summary = {
            "final_loss": final_loss,
            "total_steps": self.global_step,
            "tokens_seen": self.tokens_seen,
            "total_time_seconds": total_time,
            "best_eval_loss": self.best_eval_loss,
        }
        print("\n" + "=" * 60)
        print("训练完成!")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        return summary

    def resume_from_checkpoint(self, checkpoint_path: Optional[str] = None) -> None:
        """
        从 checkpoint 恢复训练

        恢复内容:
        1. 模型权重
        2. 优化器状态
        3. 学习率调度器状态
        4. 随机数种子状态
        5. 训练步数

        Args:
            checkpoint_path: checkpoint 路径（默认加载最新的）
        """
        meta = self.ckpt_manager.load(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            checkpoint_path=checkpoint_path,
        )
        self.global_step = meta.get("step", 0) + 1
        self.tokens_seen = meta.get("tokens_seen", 0)
        print(f"从 checkpoint 恢复: step = {self.global_step}, tokens = {self.tokens_seen}")


if __name__ == "__main__":
    # 注意: 由于使用了相对导入，本文件需要作为模块运行:
    # python -m code.training_engineering.trainer
    # 从项目根目录执行
    from torch.utils.data import Dataset, DataLoader

    print("=" * 60)
    print("预训练训练器 演示")
    print("=" * 60)

    # 创建简单模型
    class SimpleTransformer(nn.Module):
        """用于演示的简化 Transformer"""
        def __init__(self, vocab_size: int = 1000, d_model: int = 128, n_layers: int = 2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=4, dim_feedforward=256,
                    batch_first=True, norm_first=True,
                )
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, input_ids):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            x = self.norm(x)
            return self.lm_head(x)

    # 创建简单数据集
    class RandomTextDataset(Dataset):
        """随机文本数据集（演示用）"""
        def __init__(self, vocab_size: int = 1000, seq_len: int = 64, size: int = 1000):
            self.data = torch.randint(0, vocab_size, (size, seq_len + 1))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            tokens = self.data[idx]
            return {
                "input_ids": tokens[:-1],
                "labels": tokens[1:],
            }

    # 设置
    model = SimpleTransformer(vocab_size=1000, d_model=128, n_layers=2)
    dataset = RandomTextDataset(vocab_size=1000, seq_len=64, size=200)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    config = TrainingConfig(
        max_steps=50,
        batch_size=8,
        seq_len=64,
        grad_accum_steps=2,
        lr=1e-3,
        warmup_steps=5,
        scheduler_type="cosine",
        save_interval=25,
        save_dir="demo_checkpoints",
        log_interval=10,
        eval_interval=25,
    )

    # 创建训练器并训练
    trainer = Trainer(
        model=model,
        config=config,
        train_dataloader=dataloader,
    )

    summary = trainer.train()

    # 清理演示文件
    import shutil
    if os.path.exists("demo_checkpoints"):
        shutil.rmtree("demo_checkpoints")
        print("\n已清理演示 checkpoint 目录")
