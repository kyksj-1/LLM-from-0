"""
MLM 损失实现: Masked Language Modeling (BERT 风格)

本模块实现了 BERT 的 Masked Language Modeling 预训练目标,
包括 80/10/10 遮蔽策略和 MLM 损失计算。

核心公式:
- L_MLM = -sum_{i in M} log P(x_i | x_{\M})
- 遮蔽策略: 15% 的 token 被选中, 其中:
  - 80% 替换为 [MASK]
  - 10% 替换为随机 token
  - 10% 保持不变

参考:
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import random


class MLMMasker:
    """
    MLM 遮蔽器: 实现 BERT 的 80/10/10 遮蔽策略

    给定一个 token 序列, 随机选择 15% 的位置进行处理:
    - 80% 替换为 [MASK] token
    - 10% 替换为随机 token
    - 10% 保持原始 token 不变

    为什么是这样的比例?
    - 80% [MASK]: 直接的遮蔽信号, 模型必须从上下文预测
    - 10% 随机替换: 迫使模型不能简单地"忽略"非 [MASK] 位置
    - 10% 不变: 让模型学到"原始 token 也可能是答案"

    Args:
        mask_prob: 总遮蔽概率 (默认 15%)
        mask_token_id: [MASK] token 的 id
        vocab_size: 词汇表大小 (用于随机替换)
        special_token_ids: 不参与遮蔽的特殊 token id 集合
    """

    def __init__(
        self,
        mask_prob: float = 0.15,
        mask_token_id: int = 3,
        vocab_size: int = 100,
        special_token_ids: Optional[set] = None,
    ):
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self.special_token_ids = special_token_ids or set()

    def __call__(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对输入序列应用 MLM 遮蔽

        Args:
            input_ids: [batch, seq_len] 原始 token id

        Returns:
            masked_input: [batch, seq_len] 遮蔽后的输入
            labels: [batch, seq_len] 标签 (非遮蔽位置为 -100)
            mask_positions: [batch, seq_len] bool, True 表示被遮蔽
        """
        masked_input = input_ids.clone()
        labels = torch.full_like(input_ids, -100)

        # 确定哪些位置可以被遮蔽 (排除特殊 token)
        eligible = torch.ones_like(input_ids, dtype=torch.bool)
        for special_id in self.special_token_ids:
            eligible &= (input_ids != special_id)

        # 随机选择 15% 的可遮蔽位置
        prob_matrix = torch.full_like(input_ids, self.mask_prob, dtype=torch.float)
        prob_matrix[~eligible] = 0.0
        mask_positions = torch.bernoulli(prob_matrix).bool()

        # 被遮蔽的位置才有标签
        labels[mask_positions] = input_ids[mask_positions]

        # 80%: 替换为 [MASK]
        mask_80 = torch.bernoulli(
            torch.full(mask_positions.shape, 0.8, device=input_ids.device)
        ).bool() & mask_positions
        masked_input[mask_80] = self.mask_token_id

        # 10%: 替换为随机 token (在剩余 20% 中的一半)
        remaining = mask_positions & ~mask_80
        mask_10_random = torch.bernoulli(
            torch.full(remaining.shape, 0.5, device=input_ids.device)
        ).bool() & remaining
        random_tokens = torch.randint(
            0, self.vocab_size, mask_10_random.shape, device=input_ids.device
        )
        masked_input[mask_10_random] = random_tokens[mask_10_random]

        # 10%: 保持不变 (remaining & ~mask_10_random, 无需操作)

        return masked_input, labels, mask_positions


class MLMLoss(nn.Module):
    """
    MLM 损失: 只在被遮蔽的位置计算交叉熵

    L_MLM = -(1/|M|) * sum_{i in M} log P(x_i | x_{\M})

    与 NTP 损失的关键区别:
    - NTP: 条件依赖 x_{<t} (左侧上下文)
    - MLM: 条件依赖 x_{\M} (双向上下文, 除遮蔽位置外的所有 token)
    - NTP: 每个 token 都贡献损失 (100% 训练信号)
    - MLM: 仅被遮蔽的 token 贡献损失 (~15% 训练信号)

    Args:
        ignore_index: 忽略的标签索引 (默认 -100, 即非遮蔽位置)
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 MLM 损失

        Args:
            logits: [batch, seq_len, vocab_size] 模型对每个位置的预测
            labels: [batch, seq_len] 标签 (非遮蔽位置为 -100)

        Returns:
            (loss, details) 包含损失值和统计信息
        """
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # 只在标签不为 -100 的位置计算损失
        loss = F.cross_entropy(
            logits_flat, labels_flat,
            ignore_index=self.ignore_index,
        )

        # 统计信息
        with torch.no_grad():
            num_masked = (labels_flat != self.ignore_index).sum().item()
            total_tokens = labels_flat.numel()
            mask_ratio = num_masked / total_tokens if total_tokens > 0 else 0

            # 被遮蔽位置的准确率
            valid_mask = labels_flat != self.ignore_index
            if valid_mask.any():
                preds = logits_flat[valid_mask].argmax(dim=-1)
                correct = (preds == labels_flat[valid_mask]).float().mean().item()
            else:
                correct = 0.0

        details = {
            "loss": loss.item(),
            "num_masked": num_masked,
            "mask_ratio": mask_ratio,
            "accuracy": correct,
        }

        return loss, details


class BERTEncoder(nn.Module):
    """
    简化版 BERT 编码器 (用于 MLM 演示)

    与 NTP 模型的关键区别: 不使用因果掩码, 允许双向注意力。

    Args:
        vocab_size: 词汇表大小
        d_model: 隐藏维度
        n_heads: 注意力头数
        n_layers: Transformer 层数
        max_seq_len: 最大序列长度
        dropout: Dropout 比率
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播 (双向注意力, 无因果掩码)

        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        h = self.dropout(h)

        # 注意: 不使用因果掩码! 这是 MLM 与 NTP 的核心区别
        h = self.transformer(h)
        h = self.ln_f(h)

        logits = self.head(h)
        return logits


if __name__ == "__main__":
    print("=" * 60)
    print("MLM (Masked Language Modeling) 损失演示")
    print("=" * 60)

    # 参数设置
    batch_size = 4
    seq_len = 32
    vocab_size = 50
    mask_token_id = 3
    pad_token_id = 0

    torch.manual_seed(42)

    # ---- 1. MLM 遮蔽策略演示 ----
    print("\n--- MLM 遮蔽策略 ---")
    masker = MLMMasker(
        mask_prob=0.15,
        mask_token_id=mask_token_id,
        vocab_size=vocab_size,
        special_token_ids={pad_token_id},
    )

    # 创建示例输入
    input_ids = torch.randint(4, vocab_size, (batch_size, seq_len))
    # 末尾加一些 padding
    input_ids[:, -5:] = pad_token_id

    masked_input, labels, mask_positions = masker(input_ids)

    # 展示第一个样本
    print(f"原始输入 (前20): {input_ids[0, :20].tolist()}")
    print(f"遮蔽后  (前20): {masked_input[0, :20].tolist()}")
    print(f"标签    (前20): {labels[0, :20].tolist()}")
    print(f"遮蔽位置(前20): {mask_positions[0, :20].tolist()}")

    num_masked = mask_positions.sum().item()
    total_eligible = (input_ids != pad_token_id).sum().item()
    print(f"\n被遮蔽 token 数: {num_masked}/{total_eligible} "
          f"(比例: {num_masked/total_eligible:.2%})")

    # ---- 2. MLM 损失计算 ----
    print("\n--- MLM 损失计算 ---")
    model = BERTEncoder(vocab_size)
    loss_fn = MLMLoss()

    logits = model(masked_input)
    loss, details = loss_fn(logits, labels)

    print(f"MLM Loss: {details['loss']:.4f}")
    print(f"遮蔽 token 数: {details['num_masked']}")
    print(f"遮蔽比例: {details['mask_ratio']:.2%}")
    print(f"遮蔽位置准确率: {details['accuracy']:.2%} (随机初始化, 应接近 1/V)")
    print(f"理论随机准确率: {1/vocab_size:.2%}")

    # ---- 3. NTP vs MLM 训练效率对比 ----
    print("\n--- NTP vs MLM 训练效率对比 ---")
    ntp_signal = seq_len  # NTP: 每个 token 都产生梯度
    mlm_signal = int(seq_len * 0.15)  # MLM: 约 15% 的 token 产生梯度
    print(f"序列长度: {seq_len}")
    print(f"NTP 每步训练信号: {ntp_signal} tokens (100%)")
    print(f"MLM 每步训练信号: ~{mlm_signal} tokens (~15%)")
    print(f"NTP 效率约为 MLM 的 {ntp_signal/mlm_signal:.1f} 倍")

    # ---- 4. 80/10/10 策略验证 ----
    print("\n--- 80/10/10 策略验证 ---")
    # 统计大量样本的遮蔽策略分布
    n_trials = 1000
    count_mask = 0
    count_random = 0
    count_unchanged = 0

    large_input = torch.randint(4, vocab_size, (n_trials, seq_len))
    masked, _, positions = masker(large_input)

    for i in range(n_trials):
        for j in range(seq_len):
            if positions[i, j]:
                if masked[i, j] == mask_token_id:
                    count_mask += 1
                elif masked[i, j] == large_input[i, j]:
                    count_unchanged += 1
                else:
                    count_random += 1

    total_masked = count_mask + count_random + count_unchanged
    if total_masked > 0:
        print(f"[MASK] 替换: {count_mask/total_masked:.1%} (期望 80%)")
        print(f"随机替换: {count_random/total_masked:.1%} (期望 10%)")
        print(f"保持不变: {count_unchanged/total_masked:.1%} (期望 10%)")
