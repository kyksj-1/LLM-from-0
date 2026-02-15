"""
Perplexity / BPB 计算与对比工具

本模块实现了语言模型评估的两个核心指标:
- Perplexity (PPL): 困惑度, 衡量模型在每个位置的平均不确定性
- Bits-per-Byte (BPB): 每字节比特数, 跨分词器的公平度量

核心公式:
- PPL = exp(-(1/T) * sum_t log P(x_t | x_{<t}))
- BPB = (NTP_loss * T) / (total_bytes * ln(2))

关键知识点:
- PPL 的信息论含义: 每 token 的平均选择数 (2^H)
- PPL 在不同分词器下不可直接比较
- BPB 通过归一化到字节级别实现跨分词器比较

参考:
- Gao et al. (2020). The Pile: An 800GB Dataset of Diverse Text.
  (BPB 度量的推广使用)
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models.
  (Chinchilla, 使用 BPB 作为评估指标)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


def compute_ntp_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "none",
) -> torch.Tensor:
    """
    计算逐 token 的 NTP 损失

    Args:
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len]
        ignore_index: 忽略的标签索引
        reduction: "none" 返回逐 token 损失, "mean" 返回平均值

    Returns:
        reduction="none": [batch, seq_len] 逐 token 损失
        reduction="mean": 标量平均损失
    """
    vocab_size = logits.size(-1)
    logits_flat = logits.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # 逐 token 损失
    per_token_loss = F.cross_entropy(
        logits_flat, targets_flat,
        ignore_index=ignore_index,
        reduction="none",
    )

    if reduction == "none":
        return per_token_loss.view(logits.shape[0], logits.shape[1])
    elif reduction == "mean":
        # 只对有效 token 求平均
        valid = targets_flat != ignore_index
        if valid.any():
            return per_token_loss[valid].mean()
        return torch.tensor(0.0, device=logits.device)
    else:
        raise ValueError(f"不支持的 reduction: {reduction}")


def compute_perplexity(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    计算 Perplexity (困惑度)

    PPL = exp(average NTP loss)

    PPL 的含义:
    - PPL = 1: 完美预测, 模型毫无不确定性
    - PPL = V (词汇表大小): 完全随机猜测
    - PPL = 10: 模型在每个位置上平均在 10 个候选中"犹豫"

    Args:
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len]
        ignore_index: 忽略的标签索引

    Returns:
        包含 loss 和 ppl 的字典
    """
    avg_loss = compute_ntp_loss(
        logits, targets, ignore_index=ignore_index, reduction="mean"
    )

    ppl = torch.exp(avg_loss).item()

    # 统计有效 token 数
    valid_tokens = (targets.view(-1) != ignore_index).sum().item()

    return {
        "loss": avg_loss.item(),
        "ppl": ppl,
        "valid_tokens": valid_tokens,
    }


def compute_bpb(
    logits: torch.Tensor,
    targets: torch.Tensor,
    total_bytes: int,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    计算 Bits-per-Byte (BPB)

    BPB = (total_nll) / (total_bytes * ln(2))

    其中:
    - total_nll = sum of per-token NTP losses (自然对数)
    - total_bytes = 原始文本的字节数
    - ln(2) 用于将 nats 转换为 bits

    BPB 的含义:
    - 平均每个原始字节需要多少比特来编码
    - 与分词器无关, 因为归一化到字节级别
    - 越低表示模型对文本的压缩能力越强

    Args:
        logits: [batch, seq_len, vocab_size]
        targets: [batch, seq_len]
        total_bytes: 原始文本的总字节数
        ignore_index: 忽略的标签索引

    Returns:
        包含 loss, ppl, bpb 等的字典
    """
    per_token_loss = compute_ntp_loss(
        logits, targets, ignore_index=ignore_index, reduction="none"
    )

    # 有效 token 的总损失
    valid_mask = (targets != ignore_index)
    total_nll = per_token_loss[valid_mask].sum().item()
    num_tokens = valid_mask.sum().item()

    # 平均每 token 损失
    avg_loss = total_nll / num_tokens if num_tokens > 0 else 0

    # PPL
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    # BPB: 将总 NLL (nats) 转换为 bits, 再除以总字节数
    bpb = total_nll / (total_bytes * math.log(2)) if total_bytes > 0 else 0

    return {
        "loss": avg_loss,
        "ppl": ppl,
        "bpb": bpb,
        "total_nll_nats": total_nll,
        "total_nll_bits": total_nll / math.log(2),
        "num_tokens": num_tokens,
        "num_bytes": total_bytes,
        "chars_per_token": total_bytes / num_tokens if num_tokens > 0 else 0,
    }


def compare_tokenizers(
    text: str,
    tokenizer_configs: List[Dict],
    model_fn=None,
) -> Dict[str, Dict]:
    """
    对比不同分词器下的 PPL 和 BPB

    这个函数演示了为什么 PPL 在不同分词器下不可比,
    而 BPB 提供了更公平的比较。

    Args:
        text: 待评估的文本
        tokenizer_configs: 分词器配置列表, 每个包含:
            - "name": 分词器名称
            - "tokenize": 分词函数 str -> List[int]
            - "vocab_size": 词汇表大小
        model_fn: 可选, 将 token id 映射到 logits 的函数

    Returns:
        各分词器的评估结果
    """
    total_bytes = len(text.encode("utf-8"))
    results = {}

    for config in tokenizer_configs:
        name = config["name"]
        tokenize = config["tokenize"]
        vocab_size = config["vocab_size"]

        # 分词
        token_ids = tokenize(text)
        num_tokens = len(token_ids)

        # 如果没有提供模型, 使用随机 logits (仅演示用)
        if model_fn is not None:
            input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long)
            logits = model_fn(input_ids)
        else:
            # 模拟: 使用均匀分布的 logits (PPL 应约等于 vocab_size)
            logits = torch.zeros(1, num_tokens - 1, vocab_size)

        targets = torch.tensor([token_ids[1:]], dtype=torch.long)

        # 计算指标
        metrics = compute_bpb(logits, targets, total_bytes)
        metrics["num_tokens"] = num_tokens
        metrics["vocab_size"] = vocab_size
        metrics["tokens_per_byte"] = num_tokens / total_bytes

        results[name] = metrics

    return results


class PerplexityTracker:
    """
    PPL 追踪器: 在训练过程中持续追踪 PPL 和 BPB

    用法:
        tracker = PerplexityTracker()
        for batch in dataloader:
            logits = model(batch.input_ids)
            tracker.update(logits, batch.targets, batch.num_bytes)
        metrics = tracker.compute()

    Args:
        warmup_steps: 预热步数 (前几步不计入统计)
    """

    def __init__(self, warmup_steps: int = 0):
        self.warmup_steps = warmup_steps
        self.reset()

    def reset(self):
        """重置追踪器"""
        self.total_nll = 0.0
        self.total_tokens = 0
        self.total_bytes = 0
        self.step_count = 0
        self.losses = []

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        num_bytes: int = 0,
        ignore_index: int = -100,
    ):
        """
        更新一个 batch 的统计量

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]
            num_bytes: 这个 batch 原始文本的字节数
            ignore_index: 忽略的标签索引
        """
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            return

        with torch.no_grad():
            per_token_loss = compute_ntp_loss(
                logits, targets, ignore_index=ignore_index, reduction="none"
            )
            valid_mask = (targets != ignore_index)
            batch_nll = per_token_loss[valid_mask].sum().item()
            batch_tokens = valid_mask.sum().item()

        self.total_nll += batch_nll
        self.total_tokens += batch_tokens
        self.total_bytes += num_bytes

        if batch_tokens > 0:
            self.losses.append(batch_nll / batch_tokens)

    def compute(self) -> Dict[str, float]:
        """
        计算累计的 PPL 和 BPB

        Returns:
            包含 loss, ppl, bpb 的字典
        """
        if self.total_tokens == 0:
            return {"loss": 0, "ppl": 0, "bpb": 0}

        avg_loss = self.total_nll / self.total_tokens
        ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        result = {
            "loss": avg_loss,
            "ppl": ppl,
            "total_tokens": self.total_tokens,
            "num_batches": len(self.losses),
        }

        if self.total_bytes > 0:
            bpb = self.total_nll / (self.total_bytes * math.log(2))
            result["bpb"] = bpb
            result["total_bytes"] = self.total_bytes

        return result


if __name__ == "__main__":
    print("=" * 60)
    print("Perplexity / BPB 计算演示")
    print("=" * 60)

    torch.manual_seed(42)

    # ---- 1. 基本 PPL 计算 ----
    print("\n--- 基本 PPL 计算 ---")
    batch_size = 2
    seq_len = 20
    vocab_size = 50

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    result = compute_perplexity(logits, targets)
    print(f"NTP Loss: {result['loss']:.4f}")
    print(f"PPL: {result['ppl']:.2f}")
    print(f"有效 tokens: {result['valid_tokens']}")
    print(f"(随机初始化模型, PPL 应接近 {vocab_size})")

    # ---- 2. PPL 与 BPB 的关系 ----
    print("\n--- PPL 与 BPB ---")
    text = "The quick brown fox jumps over the lazy dog."
    total_bytes = len(text.encode("utf-8"))
    print(f"文本: '{text}'")
    print(f"字节数: {total_bytes}")

    # 模拟不同分词粒度
    print(f"\n模拟不同分词粒度的影响:")

    # 场景: 相同文本, 不同分词 token 数
    for num_tokens, name in [(10, "粗粒度(10 tokens)"),
                              (20, "中粒度(20 tokens)"),
                              (45, "细粒度(45 tokens)")]:
        # 假设模型的总信息量相同 (约 50 nats)
        total_info_nats = 50.0
        avg_loss = total_info_nats / num_tokens
        ppl = math.exp(avg_loss)
        bpb = total_info_nats / (total_bytes * math.log(2))

        print(f"  {name}: PPL={ppl:.2f}, BPB={bpb:.4f}")

    print("\n  观察: PPL 随分词粒度变化, 而 BPB 保持不变!")
    print("  这就是为什么 BPB 比 PPL 更适合跨分词器比较。")

    # ---- 3. 不同模型能力下的 BPB ----
    print("\n--- 不同能力水平的模型 ---")
    for model_name, total_info in [
        ("随机模型", total_bytes * math.log(vocab_size)),
        ("较差模型", total_bytes * 3.0),
        ("一般模型", total_bytes * 1.5),
        ("优秀模型", total_bytes * 0.8),
        ("顶级模型", total_bytes * 0.5),
    ]:
        bpb = total_info / (total_bytes * math.log(2))
        ppl_20 = math.exp(total_info / 20)
        print(f"  {model_name}: BPB={bpb:.4f} bits/byte, "
              f"PPL(20 tokens)={ppl_20:.2f}")

    # ---- 4. 分词器对比实验 ----
    print("\n--- 分词器对比实验 ---")

    text = "Hello world! This is a simple test of perplexity measurement."
    total_bytes = len(text.encode("utf-8"))

    # 模拟三种分词器
    tokenizer_configs = [
        {
            "name": "字符级 (每字符一个 token)",
            "tokenize": lambda t: list(range(len(t))),
            "vocab_size": 128,
        },
        {
            "name": "单词级 (约每词一个 token)",
            "tokenize": lambda t: list(range(len(t.split()))),
            "vocab_size": 50000,
        },
        {
            "name": "BPE (介于字符与单词之间)",
            "tokenize": lambda t: list(range(len(t) // 3)),
            "vocab_size": 32000,
        },
    ]

    for config in tokenizer_configs:
        token_ids = config["tokenize"](text)
        num_tokens = len(token_ids)
        tokens_per_byte = num_tokens / total_bytes
        print(f"\n  {config['name']}:")
        print(f"    Token 数: {num_tokens}, "
              f"字节数: {total_bytes}, "
              f"tokens/byte: {tokens_per_byte:.2f}")
        print(f"    词汇表: {config['vocab_size']}")

    # ---- 5. PPL 追踪器演示 ----
    print("\n--- PPL 追踪器演示 ---")
    tracker = PerplexityTracker(warmup_steps=2)

    for step in range(10):
        # 模拟训练过程中 loss 逐步降低
        scale = max(0.1, 1.0 - step * 0.08)
        logits = torch.randn(4, 16, vocab_size) / scale
        targets = torch.randint(0, vocab_size, (4, 16))

        tracker.update(logits, targets, num_bytes=4 * 16 * 3)

    metrics = tracker.compute()
    print(f"累计评估结果:")
    print(f"  平均 Loss: {metrics['loss']:.4f}")
    print(f"  PPL: {metrics['ppl']:.2f}")
    if "bpb" in metrics:
        print(f"  BPB: {metrics['bpb']:.4f}")
    print(f"  总 tokens: {metrics['total_tokens']}")
    print(f"  评估 batches: {metrics['num_batches']}")

    # ---- 6. 典型 BPB 参考值 ----
    print("\n--- 典型 BPB 参考值 (英语文本) ---")
    print("  gzip 压缩:          ~2.5-3.0 BPB")
    print("  小模型 (<1B):       ~1.0-1.5 BPB")
    print("  中型模型 (7B):      ~0.7-0.9 BPB")
    print("  大型模型 (70B+):    ~0.5-0.7 BPB")
    print("  Shannon 熵下限:     ~0.8-1.3 bits/char")
    print("  注: 比 gzip 低 = 比通用压缩算法更'理解'语言")
