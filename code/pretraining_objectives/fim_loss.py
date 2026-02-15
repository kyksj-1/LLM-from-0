"""
FIM 损失实现: Fill-in-the-Middle (代码模型核心目标)

本模块实现了 Fill-in-the-Middle 预训练目标,
这是 StarCoder, CodeLlama, DeepSeek-Coder 等代码模型的关键训练策略。

核心思想:
- 将序列分为 prefix, middle, suffix 三部分
- 重排为 [PRE] prefix [SUF] suffix [MID] middle 格式 (PSM)
- 在 middle 部分应用标准 NTP 损失
- 与标准 NTP 完全兼容, 可以混合使用

两种格式:
- PSM (Prefix-Suffix-Middle): [PRE] prefix [SUF] suffix [MID] middle
- SPM (Suffix-Prefix-Middle): [SUF] suffix [PRE] prefix [MID] middle

参考:
- Bavarian et al. (2022). Efficient Training of Language Models to
  Fill in the Middle.
- Li et al. (2023). StarCoder: May the Source Be with You!
- Guo et al. (2024). DeepSeek-Coder: When the Large Language Model
  Meets Programming.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import random


# 特殊 token ID (在实际中由分词器提供)
DEFAULT_SPECIAL_TOKENS = {
    "PRE": 4,   # <PRE> / <fim_prefix>
    "SUF": 5,   # <SUF> / <fim_suffix>
    "MID": 6,   # <MID> / <fim_middle>
    "EOT": 2,   # <EOT> / 结束 token
}


class FIMTransformer:
    """
    FIM 数据变换器: 将标准序列转换为 FIM 格式

    将原始 token 序列 [t_1, t_2, ..., t_n] 变换为:
    PSM: [PRE] prefix [SUF] suffix [MID] middle [EOT]
    SPM: [SUF] suffix [PRE] prefix [MID] middle [EOT]

    FIM rate 控制变换概率:
    - fim_rate = 0.5: 50% 概率应用 FIM, 50% 保持标准 NTP
    - fim_rate = 0.9: 90% 概率应用 FIM

    Args:
        fim_rate: FIM 变换的概率
        psm_rate: 使用 PSM 格式的概率 (1-psm_rate 为 SPM)
        special_tokens: 特殊 token id 字典
    """

    def __init__(
        self,
        fim_rate: float = 0.5,
        psm_rate: float = 1.0,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        self.fim_rate = fim_rate
        self.psm_rate = psm_rate
        self.tokens = special_tokens or DEFAULT_SPECIAL_TOKENS

    def transform(
        self, token_ids: List[int]
    ) -> Tuple[List[int], List[bool]]:
        """
        对单个序列应用 FIM 变换

        Args:
            token_ids: 原始 token id 列表

        Returns:
            (transformed_ids, loss_mask)
            - transformed_ids: 变换后的 token id 列表
            - loss_mask: 哪些位置计算损失 (True = 计算)
        """
        # 按 FIM rate 决定是否变换
        if random.random() > self.fim_rate:
            # 不变换, 标准 NTP (所有位置都计算损失)
            return token_ids, [True] * len(token_ids)

        n = len(token_ids)
        if n < 3:
            # 序列太短, 无法分割
            return token_ids, [True] * len(token_ids)

        # 随机选择分割点
        # 确保 prefix, middle, suffix 都至少有可能为空
        split_start = random.randint(0, n)
        split_end = random.randint(split_start, n)

        prefix = token_ids[:split_start]
        middle = token_ids[split_start:split_end]
        suffix = token_ids[split_end:]

        # 选择格式: PSM 或 SPM
        if random.random() < self.psm_rate:
            # PSM: [PRE] prefix [SUF] suffix [MID] middle [EOT]
            transformed = (
                [self.tokens["PRE"]] + prefix
                + [self.tokens["SUF"]] + suffix
                + [self.tokens["MID"]] + middle
                + [self.tokens["EOT"]]
            )

            # 损失掩码: 只在 middle + EOT 部分计算
            loss_mask = (
                [False] * (1 + len(prefix))       # [PRE] + prefix
                + [False] * (1 + len(suffix))      # [SUF] + suffix
                + [False]                           # [MID]
                + [True] * len(middle)              # middle
                + [True]                            # [EOT]
            )
        else:
            # SPM: [SUF] suffix [PRE] prefix [MID] middle [EOT]
            transformed = (
                [self.tokens["SUF"]] + suffix
                + [self.tokens["PRE"]] + prefix
                + [self.tokens["MID"]] + middle
                + [self.tokens["EOT"]]
            )

            loss_mask = (
                [False] * (1 + len(suffix))        # [SUF] + suffix
                + [False] * (1 + len(prefix))       # [PRE] + prefix
                + [False]                            # [MID]
                + [True] * len(middle)               # middle
                + [True]                             # [EOT]
            )

        return transformed, loss_mask

    def transform_batch(
        self, batch: List[List[int]], max_len: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对一个 batch 的序列应用 FIM 变换

        Args:
            batch: token id 列表的列表
            max_len: 最大序列长度 (截断或填充)

        Returns:
            input_ids: [batch, max_len]
            targets: [batch, max_len] (NTP 标签, 非损失位置为 -100)
            loss_mask: [batch, max_len] bool
        """
        transformed_batch = []
        masks_batch = []

        for seq in batch:
            transformed, mask = self.transform(seq)
            transformed_batch.append(transformed)
            masks_batch.append(mask)

        # 确定最大长度
        if max_len is None:
            max_len = max(len(s) for s in transformed_batch)

        # 填充和截断
        pad_id = 0  # padding token id
        input_ids = []
        targets = []
        loss_masks = []

        for seq, mask in zip(transformed_batch, masks_batch):
            # 截断
            seq = seq[:max_len]
            mask = mask[:max_len]

            # 构建 NTP 标签: 每个位置的目标是下一个 token
            # 输入: seq[:-1], 目标: seq[1:]
            inp = seq[:-1] if len(seq) > 1 else seq
            tgt = seq[1:] if len(seq) > 1 else [-100]
            m = mask[1:] if len(mask) > 1 else [False]

            # 填充到 max_len - 1
            target_len = max_len - 1
            inp = inp + [pad_id] * (target_len - len(inp))
            tgt = tgt + [-100] * (target_len - len(tgt))
            m = m + [False] * (target_len - len(m))

            # 非损失位置的 target 设为 -100
            tgt = [t if m_flag else -100 for t, m_flag in zip(tgt, m)]

            input_ids.append(inp[:target_len])
            targets.append(tgt[:target_len])
            loss_masks.append(m[:target_len])

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(loss_masks, dtype=torch.bool),
        )


class FIMLoss(nn.Module):
    """
    FIM 损失: 在 middle 部分计算 NTP 损失

    本质上是标准 NTP 损失, 但只在被选中的位置计算。
    通过 loss_mask 或将非计算位置的 target 设为 -100 来实现。

    L_FIM = -(1/|M|) * sum_{t in middle} log P(x_t | x'_{<t})

    其中 x' 是 FIM 变换后的序列。

    Args:
        ignore_index: 忽略的标签索引
    """

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 FIM 损失

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len] (非计算位置已设为 -100)
            loss_mask: [batch, seq_len] 可选的显式损失掩码

        Returns:
            (loss, details)
        """
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # 如果提供了显式掩码, 将非掩码位置的 target 设为 -100
        if loss_mask is not None:
            mask_flat = loss_mask.view(-1)
            targets_flat = targets_flat.clone()
            targets_flat[~mask_flat] = self.ignore_index

        loss = F.cross_entropy(
            logits_flat, targets_flat,
            ignore_index=self.ignore_index,
        )

        # 统计
        with torch.no_grad():
            valid = targets_flat != self.ignore_index
            num_loss_tokens = valid.sum().item()
            total_tokens = targets_flat.numel()

            if valid.any():
                preds = logits_flat[valid].argmax(dim=-1)
                acc = (preds == targets_flat[valid]).float().mean().item()
            else:
                acc = 0.0

        details = {
            "loss": loss.item(),
            "num_loss_tokens": num_loss_tokens,
            "total_tokens": total_tokens,
            "loss_ratio": num_loss_tokens / total_tokens if total_tokens > 0 else 0,
            "accuracy": acc,
        }

        return loss, details


def demonstrate_fim_transform():
    """演示 FIM 变换过程"""
    print("=== FIM 变换过程演示 ===\n")

    # 模拟一段简单的 "代码"
    # 用数字代表 token id, 方便展示
    original = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    token_names = {
        10: "def", 11: " ", 12: "add", 13: "(", 14: "a",
        15: ",", 16: "b", 17: ")", 18: ":", 19: "return",
        4: "<PRE>", 5: "<SUF>", 6: "<MID>", 2: "<EOT>",
    }

    def show_tokens(ids):
        return " ".join(token_names.get(i, f"[{i}]") for i in ids)

    print(f"原始序列: {show_tokens(original)}")
    print(f"Token IDs: {original}\n")

    # 手动演示一个 FIM 变换
    # 假设 split_start=4, split_end=7
    prefix = original[:4]   # [def, " ", add, "("]
    middle = original[4:7]  # [a, ",", b]
    suffix = original[7:]   # [")", ":", return]

    print(f"Prefix: {show_tokens(prefix)}")
    print(f"Middle (被挖空): {show_tokens(middle)}")
    print(f"Suffix: {show_tokens(suffix)}")

    # PSM 格式
    psm = [4] + prefix + [5] + suffix + [6] + middle + [2]
    print(f"\nPSM 格式: {show_tokens(psm)}")
    print(f"Token IDs: {psm}")

    # SPM 格式
    spm = [5] + suffix + [4] + prefix + [6] + middle + [2]
    print(f"\nSPM 格式: {show_tokens(spm)}")
    print(f"Token IDs: {spm}")

    # 损失掩码
    psm_mask = [False]*(1+4) + [False]*(1+3) + [False] + [True]*3 + [True]
    print(f"\nPSM 损失掩码: {psm_mask}")
    print("(True = 计算损失, 即 middle + EOT 部分)")


if __name__ == "__main__":
    print("=" * 60)
    print("FIM (Fill-in-the-Middle) 损失演示")
    print("=" * 60)

    # ---- 1. FIM 变换过程演示 ----
    demonstrate_fim_transform()

    # ---- 2. FIM 数据变换器 ----
    print("\n" + "=" * 60)
    print("FIM 数据变换器")
    print("=" * 60)

    transformer = FIMTransformer(fim_rate=0.5, psm_rate=1.0)

    # 创建示例数据
    random.seed(42)
    torch.manual_seed(42)

    # 模拟 4 个序列
    batch = [
        list(range(10, 25)),   # 15 tokens
        list(range(20, 32)),   # 12 tokens
        list(range(30, 48)),   # 18 tokens
        list(range(40, 52)),   # 12 tokens
    ]

    print(f"\n原始 batch:")
    for i, seq in enumerate(batch):
        print(f"  序列 {i}: 长度={len(seq)}, 前5个={seq[:5]}")

    # 应用 FIM 变换
    input_ids, targets, loss_mask = transformer.transform_batch(batch, max_len=25)

    print(f"\n变换后:")
    print(f"  input_ids 形状: {input_ids.shape}")
    print(f"  targets 形状: {targets.shape}")
    print(f"  loss_mask 形状: {loss_mask.shape}")

    for i in range(len(batch)):
        num_loss = loss_mask[i].sum().item()
        total = loss_mask[i].numel()
        is_fim = any(
            t.item() in DEFAULT_SPECIAL_TOKENS.values()
            for t in input_ids[i]
        )
        print(f"  序列 {i}: FIM={is_fim}, 损失token数={num_loss}/{total}")

    # ---- 3. FIM 损失计算 ----
    print("\n" + "=" * 60)
    print("FIM 损失计算")
    print("=" * 60)

    vocab_size = 60
    model = nn.Sequential(
        nn.Embedding(vocab_size, 64),
        nn.Linear(64, vocab_size),
    )

    fim_loss_fn = FIMLoss()

    logits = model(input_ids)
    loss, details = fim_loss_fn(logits, targets, loss_mask)

    print(f"\nFIM Loss: {details['loss']:.4f}")
    print(f"损失 token 数: {details['num_loss_tokens']}")
    print(f"总 token 数: {details['total_tokens']}")
    print(f"损失比例: {details['loss_ratio']:.2%}")
    print(f"准确率: {details['accuracy']:.2%}")

    # ---- 4. FIM rate 对比实验 ----
    print("\n" + "=" * 60)
    print("不同 FIM rate 的损失比例")
    print("=" * 60)

    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        random.seed(0)
        ft = FIMTransformer(fim_rate=rate)
        total_loss_tokens = 0
        total_tokens = 0
        n_trials = 100

        for _ in range(n_trials):
            seq = list(range(10, 30))
            transformed, mask = ft.transform(seq)
            total_loss_tokens += sum(mask)
            total_tokens += len(mask)

        avg_ratio = total_loss_tokens / total_tokens if total_tokens > 0 else 0
        print(f"  FIM rate={rate:.2f} -> "
              f"平均损失 token 比例: {avg_ratio:.2%}")
