"""
DPO 训练器实现

本模块实现了完整的 DPO 训练循环，包括：
- 数据加载与批处理
- 策略模型和参考模型的前向传播
- DPO 损失计算与反向传播
- 训练监控与日志记录

支持多种偏好优化方法（DPO/IPO/SimPO），通过参数切换。

参考:
- Rafailov et al. (2023). Direct Preference Optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional, List, Any
import copy
import logging
import time

# 导入损失函数
from dpo_loss import dpo_loss, ipo_loss, compute_log_probs
from simpo_loss import simpo_loss
from kto_loss import kto_loss

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


class DPOTrainer:
    """
    DPO 训练器

    支持以下偏好优化方法:
    - DPO: 标准直接偏好优化
    - IPO: 身份偏好优化（防过拟合）
    - SimPO: 简单偏好优化（无参考模型）

    训练流程:
    1. 加载策略模型和参考模型
    2. 对每个批次:
       a. 计算策略模型在偏好/非偏好回答上的 log P
       b. 计算参考模型在偏好/非偏好回答上的 log P（DPO/IPO）
       c. 计算损失并反向传播
       d. 更新策略模型参数
    3. 记录训练指标

    使用示例:
        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            optimizer=optimizer,
            loss_type="dpo",
            beta=0.1,
        )
        metrics = trainer.train_epoch(dataloader)
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        loss_type: str = "dpo",
        beta: float = 0.1,
        gamma: float = 0.5,
        label_smoothing: float = 0.0,
        max_grad_norm: float = 1.0,
        device: str = "cpu",
    ):
        """
        初始化 DPO 训练器

        Args:
            model: 策略模型（将被优化）
            ref_model: 参考模型（DPO/IPO 需要，SimPO 不需要）
            optimizer: 优化器，若未指定则使用 AdamW
            loss_type: 损失类型 "dpo" / "ipo" / "simpo"
            beta: KL 惩罚系数
            gamma: SimPO 的目标奖励差
            label_smoothing: DPO 的标签平滑系数
            max_grad_norm: 梯度裁剪阈值
            device: 训练设备
        """
        self.model = model.to(device)
        self.device = device
        self.loss_type = loss_type
        self.beta = beta
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.max_grad_norm = max_grad_norm

        # 参考模型（冻结参数）
        if ref_model is not None:
            self.ref_model = ref_model.to(device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        elif loss_type != "simpo":
            # DPO 和 IPO 需要参考模型，如果未提供则复制策略模型
            logger.info("未提供参考模型，将复制策略模型作为参考")
            self.ref_model = copy.deepcopy(model).to(device)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            self.ref_model = None

        # 优化器
        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=1e-6,
                weight_decay=0.01,
            )

        # 训练历史
        self.train_history: List[Dict[str, float]] = []

    def compute_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算模型在给定序列上的对数概率

        Args:
            model: 语言模型
            input_ids: 输入 token ID，形状 [batch_size, seq_len]
            attention_mask: 注意力掩码，形状 [batch_size, seq_len]
            labels: 目标标签，形状 [batch_size, seq_len]

        Returns:
            对数概率，形状 [batch_size]
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

        # 移位：logits[t] 对应 labels[t+1] 的预测
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        shift_mask = attention_mask[:, 1:]

        # 计算序列级的对数概率
        log_probs = compute_log_probs(shift_logits, shift_labels, shift_mask)
        return log_probs

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算偏好优化损失

        根据 loss_type 调用不同的损失函数。

        Args:
            batch: 包含以下键的字典:
                - chosen_input_ids: 偏好回答的 token ID
                - chosen_attention_mask: 偏好回答的掩码
                - chosen_labels: 偏好回答的标签
                - rejected_input_ids: 非偏好回答的 token ID
                - rejected_attention_mask: 非偏好回答的掩码
                - rejected_labels: 非偏好回答的标签

        Returns:
            (loss, metrics)
        """
        # 将数据移到设备上
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # 计算策略模型的对数概率
        policy_chosen_logps = self.compute_logps(
            self.model,
            batch["chosen_input_ids"],
            batch["chosen_attention_mask"],
            batch["chosen_labels"],
        )
        policy_rejected_logps = self.compute_logps(
            self.model,
            batch["rejected_input_ids"],
            batch["rejected_attention_mask"],
            batch["rejected_labels"],
        )

        if self.loss_type == "simpo":
            # SimPO 不需要参考模型
            chosen_lengths = batch["chosen_attention_mask"][:, 1:].sum(dim=-1)
            rejected_lengths = batch["rejected_attention_mask"][:, 1:].sum(dim=-1)

            loss, metrics = simpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                chosen_lengths,
                rejected_lengths,
                beta=self.beta,
                gamma=self.gamma,
            )
        else:
            # DPO 和 IPO 需要参考模型的对数概率
            with torch.no_grad():
                ref_chosen_logps = self.compute_logps(
                    self.ref_model,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                ref_rejected_logps = self.compute_logps(
                    self.ref_model,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )

            if self.loss_type == "dpo":
                loss, metrics = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.beta,
                    label_smoothing=self.label_smoothing,
                )
            elif self.loss_type == "ipo":
                loss, metrics = ipo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=self.beta,
                )
            else:
                raise ValueError(f"不支持的损失类型: {self.loss_type}")

        return loss, metrics

    def train_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        执行单步训练

        Args:
            batch: 训练批次数据

        Returns:
            训练指标字典
        """
        self.model.train()
        self.optimizer.zero_grad()

        # 前向传播
        loss, metrics = self.compute_loss(batch)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        if self.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )
            metrics["grad_norm"] = grad_norm.detach()

        # 参数更新
        self.optimizer.step()

        # 转换为 Python 浮点数
        return {k: v.item() if isinstance(v, torch.Tensor) else v
                for k, v in metrics.items()}

    def train_epoch(
        self,
        dataloader: DataLoader,
        log_interval: int = 10,
    ) -> Dict[str, float]:
        """
        训练一个 epoch

        Args:
            dataloader: 训练数据加载器
            log_interval: 日志打印间隔

        Returns:
            epoch 级别的平均指标
        """
        epoch_metrics: Dict[str, List[float]] = {}
        start_time = time.time()

        for step, batch in enumerate(dataloader):
            step_metrics = self.train_step(batch)

            # 累积指标
            for k, v in step_metrics.items():
                if k not in epoch_metrics:
                    epoch_metrics[k] = []
                epoch_metrics[k].append(v)

            # 打印日志
            if (step + 1) % log_interval == 0:
                loss = step_metrics.get("loss", 0)
                acc = step_metrics.get("reward_accuracy", 0)
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {step + 1} | Loss: {loss:.4f} | "
                    f"Reward Acc: {acc:.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

        # 计算 epoch 平均指标
        avg_metrics = {}
        for k, v_list in epoch_metrics.items():
            avg_metrics[k] = sum(v_list) / len(v_list)

        self.train_history.append(avg_metrics)
        return avg_metrics

    def get_implicit_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算隐式奖励

        隐式奖励 = beta * log(pi_theta(y|x) / pi_ref(y|x))

        Args:
            input_ids, attention_mask, labels: 输入数据

        Returns:
            隐式奖励，形状 [batch_size]
        """
        self.model.eval()
        with torch.no_grad():
            policy_logps = self.compute_logps(
                self.model, input_ids, attention_mask, labels
            )
            if self.ref_model is not None:
                ref_logps = self.compute_logps(
                    self.ref_model, input_ids, attention_mask, labels
                )
                return self.beta * (policy_logps - ref_logps)
            else:
                # SimPO: 直接返回长度归一化的 log prob
                lengths = attention_mask[:, 1:].sum(dim=-1)
                return self.beta * policy_logps / lengths


class SimpleLanguageModel(nn.Module):
    """
    简单的语言模型（用于演示）

    这是一个仅包含嵌入层和线性层的极简语言模型，
    用于验证 DPO 训练器的正确性。
    """

    def __init__(self, vocab_size: int = 100, hidden_size: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]（未使用，保持接口兼容）

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        embeddings = self.embedding(input_ids)
        logits = self.linear(embeddings)
        return logits


def create_dummy_batch(
    batch_size: int = 4,
    seq_len: int = 20,
    vocab_size: int = 100,
) -> Dict[str, torch.Tensor]:
    """
    创建用于演示的虚拟批次数据

    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        vocab_size: 词汇表大小

    Returns:
        包含偏好和非偏好回答的批次字典
    """
    batch = {
        "chosen_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "chosen_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "chosen_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "rejected_input_ids": torch.randint(0, vocab_size, (batch_size, seq_len)),
        "rejected_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "rejected_labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
    }
    return batch


if __name__ == "__main__":
    print("=" * 60)
    print("DPO 训练器演示")
    print("=" * 60)

    torch.manual_seed(42)
    vocab_size = 100
    hidden_size = 64

    # 创建模型
    model = SimpleLanguageModel(vocab_size, hidden_size)
    ref_model = copy.deepcopy(model)

    # 场景 1: DPO 训练
    print("\n--- 场景 1: DPO 训练 ---")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        loss_type="dpo",
        beta=0.1,
    )

    # 模拟几步训练
    for step in range(5):
        batch = create_dummy_batch(batch_size=4, seq_len=20, vocab_size=vocab_size)
        metrics = trainer.train_step(batch)
        print(f"  Step {step + 1}: loss={metrics['loss']:.4f}, "
              f"reward_acc={metrics.get('reward_accuracy', 0):.4f}")

    # 场景 2: IPO 训练
    print("\n--- 场景 2: IPO 训练 ---")
    model_ipo = SimpleLanguageModel(vocab_size, hidden_size)
    ref_model_ipo = copy.deepcopy(model_ipo)
    trainer_ipo = DPOTrainer(
        model=model_ipo,
        ref_model=ref_model_ipo,
        loss_type="ipo",
        beta=0.1,
    )

    for step in range(5):
        batch = create_dummy_batch(batch_size=4, seq_len=20, vocab_size=vocab_size)
        metrics = trainer_ipo.train_step(batch)
        print(f"  Step {step + 1}: loss={metrics['loss']:.4f}, "
              f"reward_acc={metrics.get('reward_accuracy', 0):.4f}")

    # 场景 3: SimPO 训练（无参考模型）
    print("\n--- 场景 3: SimPO 训练（无参考模型）---")
    model_simpo = SimpleLanguageModel(vocab_size, hidden_size)
    trainer_simpo = DPOTrainer(
        model=model_simpo,
        ref_model=None,
        loss_type="simpo",
        beta=2.0,
        gamma=0.5,
    )

    for step in range(5):
        batch = create_dummy_batch(batch_size=4, seq_len=20, vocab_size=vocab_size)
        metrics = trainer_simpo.train_step(batch)
        print(f"  Step {step + 1}: loss={metrics['loss']:.4f}, "
              f"reward_acc={metrics.get('reward_accuracy', 0):.4f}")

    print("\n训练器演示完成。")
    print("注意: 这里使用了极简模型和随机数据，仅用于验证训练器的正确性。")
    print("实际使用时，请替换为真实的预训练语言模型和偏好数据集。")
