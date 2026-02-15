"""
SFT 训练器

本模块实现了监督微调的完整训练循环，包括：
- 梯度累积
- 学习率调度（Cosine + Warmup）
- 梯度裁剪
- 训练日志记录
- Prompt Masking 损失计算

数学基础:
- SFT 损失: L = -sum_t 1[t > m] * log P(s_t | s_{<t})
- 其中 m 是 prompt 长度，只有 response 部分参与损失计算
- 通过设置 labels[prompt_positions] = -100 实现

参考:
- Ouyang et al. (2022). Training language models to follow instructions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Any
import math
import time


class CosineWarmupScheduler:
    """
    Cosine 学习率调度器（带 Warmup）

    学习率变化:
    - Warmup 阶段 (0 ~ warmup_steps): 线性从 0 增加到 lr
    - Cosine 衰减阶段 (warmup_steps ~ total_steps):
      lr_t = lr_min + 0.5 * (lr - lr_min) * (1 + cos(pi * progress))

    Args:
        optimizer: PyTorch 优化器
        warmup_steps: Warmup 步数
        total_steps: 总训练步数
        lr_min_ratio: 最小学习率相对初始学习率的比例
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        lr_min_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.lr_min_ratio = lr_min_ratio

        # 记录初始学习率
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        self.current_step = 0

    def step(self) -> None:
        """更新学习率"""
        self.current_step += 1
        lr_scale = self._get_lr_scale()

        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = base_lr * lr_scale

    def _get_lr_scale(self) -> float:
        """
        计算当前步的学习率缩放因子

        Returns:
            学习率缩放因子 (0 ~ 1)
        """
        step = self.current_step

        if step < self.warmup_steps:
            # 线性 Warmup
            return step / max(self.warmup_steps, 1)
        else:
            # Cosine 衰减
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return self.lr_min_ratio + (1 - self.lr_min_ratio) * cosine_decay

    def get_lr(self) -> float:
        """返回当前学习率"""
        return self.optimizer.param_groups[0]["lr"]


class SFTTrainer:
    """
    监督微调训练器

    支持:
    - 梯度累积 (gradient accumulation)
    - 学习率调度 (cosine warmup)
    - 梯度裁剪 (gradient clipping)
    - 训练日志

    Args:
        model: 要微调的模型
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        max_grad_norm: 梯度裁剪的最大范数
        gradient_accumulation_steps: 梯度累积步数
        device: 训练设备
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[CosineWarmupScheduler] = None,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 1,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device

        # 训练状态
        self.global_step = 0
        self.total_loss = 0.0
        self.log_history: List[Dict[str, Any]] = []

    def compute_loss(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 SFT 损失

        核心: labels 中值为 -100 的位置不参与损失计算（Prompt Masking）

        Args:
            logits: 模型输出 (batch, seq_len, vocab_size)
            labels: 目标标签 (batch, seq_len)，prompt 部分为 -100

        Returns:
            标量损失值
        """
        # 移位: logits[:-1] 预测 labels[1:]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # 展平
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,  # 忽略 prompt 部分
        )

        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        单步训练

        Args:
            batch: 包含 input_ids, attention_mask, labels 的字典

        Returns:
            当前步的损失值
        """
        self.model.train()

        # 数据移到设备
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)

        # 前向传播
        outputs = self.model(input_ids, attention_mask=attention_mask)

        # 处理不同模型的输出格式
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs[0]

        # 计算损失
        loss = self.compute_loss(logits, labels)

        # 梯度累积: 损失除以累积步数
        loss_scaled = loss / self.gradient_accumulation_steps

        # 反向传播
        loss_scaled.backward()

        return loss.item()

    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int = 0,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            dataloader: 训练数据 DataLoader
            epoch: 当前 epoch 编号
            log_interval: 日志打印间隔（步数）

        Returns:
            包含 epoch 统计信息的字典
        """
        self.model.train()
        epoch_loss = 0.0
        num_steps = 0
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            # 训练一步
            loss = self.train_step(batch)
            epoch_loss += loss
            num_steps += 1

            # 梯度累积完成后执行优化步骤
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.max_grad_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.max_grad_norm,
                    )
                else:
                    grad_norm = 0.0

                # 优化器步骤
                self.optimizer.step()
                self.optimizer.zero_grad()

                # 学习率调度
                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

                # 日志记录
                if self.global_step % log_interval == 0:
                    avg_loss = epoch_loss / num_steps
                    lr = self.scheduler.get_lr() if self.scheduler else self.optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - start_time
                    steps_per_sec = num_steps / elapsed

                    log_entry = {
                        "epoch": epoch,
                        "step": self.global_step,
                        "loss": avg_loss,
                        "lr": lr,
                        "grad_norm": grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                        "steps_per_sec": steps_per_sec,
                    }
                    self.log_history.append(log_entry)

                    print(
                        f"Epoch {epoch} | Step {self.global_step} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | "
                        f"Grad Norm: {log_entry['grad_norm']:.4f} | "
                        f"Speed: {steps_per_sec:.1f} steps/s"
                    )

        avg_epoch_loss = epoch_loss / max(num_steps, 1)
        return {
            "epoch": epoch,
            "avg_loss": avg_epoch_loss,
            "num_steps": num_steps,
            "time": time.time() - start_time,
        }

    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int = 3,
        log_interval: int = 10,
    ) -> List[Dict[str, float]]:
        """
        完整训练循环

        Args:
            dataloader: 训练数据
            num_epochs: 训练轮数
            log_interval: 日志间隔

        Returns:
            每个 epoch 的统计信息列表
        """
        print(f"\n开始训练: {num_epochs} epochs")
        print(f"梯度累积步数: {self.gradient_accumulation_steps}")
        print(f"最大梯度范数: {self.max_grad_norm}")
        print(f"设备: {self.device}")
        print("-" * 60)

        epoch_stats = []
        for epoch in range(num_epochs):
            stats = self.train_epoch(dataloader, epoch, log_interval)
            epoch_stats.append(stats)
            print(
                f"\nEpoch {epoch} 完成 | "
                f"平均损失: {stats['avg_loss']:.4f} | "
                f"步数: {stats['num_steps']} | "
                f"耗时: {stats['time']:.1f}s"
            )
            print("-" * 60)

        return epoch_stats


def create_sft_optimizer(
    model: nn.Module,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    lora_only: bool = True,
) -> torch.optim.AdamW:
    """
    创建 SFT 优化器

    Args:
        model: 模型
        lr: 学习率
        weight_decay: 权重衰减
        lora_only: 是否只优化 LoRA 参数

    Returns:
        AdamW 优化器
    """
    if lora_only:
        # 只优化 LoRA 参数
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        # 全参数优化（区分是否 weight decay）
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )

    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


if __name__ == "__main__":
    print("=" * 60)
    print("SFT 训练器演示")
    print("=" * 60)

    torch.manual_seed(42)

    # === 创建简单模型 ===
    class TinyLM(nn.Module):
        """微型语言模型（用于演示）"""
        def __init__(self, vocab_size=1000, d_model=128, n_layers=2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=4, dim_feedforward=512,
                    batch_first=True, dropout=0.1
                )
                for _ in range(n_layers)
            ])
            self.lm_head = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids, attention_mask=None):
            x = self.embedding(input_ids)
            for layer in self.layers:
                x = layer(x)
            return self.lm_head(x)

    model = TinyLM(vocab_size=1000, d_model=128, n_layers=2)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # === 创建模拟训练数据 ===
    batch_size = 4
    seq_len = 64
    prompt_len = 20  # 前 20 个 token 是 prompt

    # 模拟一个 batch
    def make_fake_batch():
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100  # Prompt Masking
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # === 创建训练器 ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    total_steps = 20
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=5, total_steps=total_steps
    )

    trainer = SFTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=1.0,
        gradient_accumulation_steps=2,
        device="cpu",
    )

    # === 模拟训练 ===
    print("\n--- 模拟训练过程 ---")
    for step in range(10):
        batch = make_fake_batch()
        loss = trainer.train_step(batch)

        if (step + 1) % trainer.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            trainer.global_step += 1

            lr = scheduler.get_lr()
            print(f"Step {trainer.global_step}: loss={loss:.4f}, lr={lr:.2e}")

    # === 演示 Prompt Masking ===
    print("\n--- Prompt Masking 演示 ---")
    batch = make_fake_batch()
    print(f"input_ids 形状: {batch['input_ids'].shape}")
    print(f"labels 形状: {batch['labels'].shape}")
    print(f"labels 中 -100 的数量 (prompt): {(batch['labels'] == -100).sum().item()}")
    print(f"labels 中非 -100 的数量 (response): {(batch['labels'] != -100).sum().item()}")
    print(f"Prompt 比例: {(batch['labels'] == -100).float().mean() * 100:.1f}%")

    # === 学习率调度可视化数据 ===
    print("\n--- 学习率调度 ---")
    test_scheduler = CosineWarmupScheduler(
        torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1e-4),
        warmup_steps=10,
        total_steps=100,
    )
    print(f"{'步数':>6} {'学习率':>12}")
    for s in [0, 5, 10, 25, 50, 75, 100]:
        test_scheduler.current_step = s
        lr_scale = test_scheduler._get_lr_scale()
        print(f"{s:>6} {lr_scale * 1e-4:>12.2e}")
