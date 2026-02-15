"""
简单训练脚本

本模块实现了一个完整但简洁的语言模型训练循环, 用于在小数据集上训练 mini-GPT。
适用于教学目的, 展示 Decoder-Only 模型的训练全流程。

功能:
- 字符级/子词级文本数据加载
- 自回归训练 (Next Token Prediction)
- Cosine 学习率调度 + Warmup
- 梯度裁剪
- 训练过程监控 (loss 曲线)
- 自动生成文本样本

数学基础:
- 训练目标: L = -1/T * sum log P(x_t | x_{<t})
- 即交叉熵损失 (Cross-Entropy Loss)

参考:
- Karpathy, nanoGPT: https://github.com/karpathy/nanoGPT
"""

import math
import time
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import ModelConfig, mini_config
from model import DecoderOnlyModel
from generation import generate
from tokenizer_wrapper import SimpleCharTokenizer


# ============================================================
# 数据集
# ============================================================

class TextDataset(Dataset):
    """
    字符级文本数据集

    将长文本切分为固定长度的片段, 每个片段用于训练:
    输入 = tokens[i:i+seq_len]
    目标 = tokens[i+1:i+seq_len+1]

    Args:
        text: 原始文本
        tokenizer: 分词器
        seq_len: 序列长度
    """

    def __init__(self, text: str, tokenizer: SimpleCharTokenizer, seq_len: int = 256):
        self.seq_len = seq_len
        # 编码整个文本 (不加特殊 token)
        self.tokens = tokenizer.encode(text, add_bos=False, add_eos=False)
        self.n_samples = max(0, len(self.tokens) - seq_len)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (input, target) 对

        input  = tokens[idx : idx + seq_len]
        target = tokens[idx + 1 : idx + seq_len + 1]
        """
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


# ============================================================
# 学习率调度
# ============================================================

def get_cosine_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> float:
    """
    Cosine 学习率调度 (含 Warmup)

    阶段 1 (Warmup): 从 0 线性增长到 max_lr
    阶段 2 (Cosine Decay): 从 max_lr 余弦衰减到 min_lr

    lr(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * progress))

    Args:
        step: 当前训练步数
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_steps: Warmup 步数
        total_steps: 总训练步数

    Returns:
        当前步的学习率
    """
    # Warmup 阶段
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 训练结束后
    if step >= total_steps:
        return min_lr

    # Cosine 衰减阶段
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================
# 训练函数
# ============================================================

def train(
    model: DecoderOnlyModel,
    train_dataset: TextDataset,
    val_dataset: Optional[TextDataset] = None,
    epochs: int = 10,
    batch_size: int = 32,
    max_lr: float = 3e-4,
    min_lr: float = 1e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.1,
    grad_clip: float = 1.0,
    eval_interval: int = 100,
    generate_interval: int = 500,
    generate_prompt: str = "",
    device: str = "auto",
) -> dict:
    """
    完整的训练循环

    Args:
        model: 语言模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集 (可选)
        epochs: 训练轮数
        batch_size: 批大小
        max_lr: 最大学习率
        min_lr: 最小学习率
        warmup_ratio: Warmup 步数占总步数的比例
        weight_decay: 权重衰减 (AdamW)
        grad_clip: 梯度裁剪阈值
        eval_interval: 评估间隔 (步数)
        generate_interval: 生成样本间隔 (步数)
        generate_prompt: 生成时使用的 prompt
        device: 设备 ("auto" / "cpu" / "cuda")

    Returns:
        训练日志字典 (包含 train_losses, val_losses, learning_rates)
    """
    # --- 设备选择 ---
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"使用设备: {device}")
    print(f"模型参数量: {model.count_parameters():,}")

    # --- 数据加载 ---
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
    )

    # --- 优化器 ---
    # 分离需要和不需要权重衰减的参数
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=max_lr, betas=(0.9, 0.95), eps=1e-8)

    # --- 学习率调度 ---
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    print(f"总训练步数: {total_steps}, Warmup 步数: {warmup_steps}")

    # --- 训练日志 ---
    log = {
        "train_losses": [],
        "val_losses": [],
        "learning_rates": [],
        "steps": [],
    }

    # --- 训练循环 ---
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            # 更新学习率
            lr = get_cosine_lr(global_step, max_lr, min_lr, warmup_steps, total_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # 前向传播
            logits = model(x)  # [batch, seq_len, vocab_size]

            # 计算交叉熵损失
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # 参数更新
            optimizer.step()

            # 记录
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            log["train_losses"].append(loss.item())
            log["learning_rates"].append(lr)
            log["steps"].append(global_step)

            # 定期打印
            if global_step % eval_interval == 0:
                avg_loss = epoch_loss / n_batches
                ppl = math.exp(min(avg_loss, 20))  # 防止溢出
                msg = (f"Step {global_step:>6d} | "
                       f"Epoch {epoch+1}/{epochs} | "
                       f"Loss {loss.item():.4f} | "
                       f"PPL {ppl:.2f} | "
                       f"LR {lr:.2e}")

                # 验证集评估
                if val_dataset is not None:
                    val_loss = evaluate(model, val_dataset, batch_size, device)
                    val_ppl = math.exp(min(val_loss, 20))
                    log["val_losses"].append(val_loss)
                    msg += f" | Val Loss {val_loss:.4f} | Val PPL {val_ppl:.2f}"

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss

                print(msg)

    print(f"\n训练完成! 最佳验证 Loss: {best_val_loss:.4f}")
    return log


@torch.no_grad()
def evaluate(
    model: DecoderOnlyModel,
    dataset: TextDataset,
    batch_size: int = 32,
    device: str = "cpu",
    max_batches: int = 50,
) -> float:
    """
    在数据集上评估模型

    Args:
        model: 语言模型
        dataset: 评估数据集
        batch_size: 批大小
        device: 设备
        max_batches: 最大评估批次数 (避免评估过慢)

    Returns:
        平均交叉熵损失
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        if n_batches >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================
# 主程序: 在示例文本上训练
# ============================================================

if __name__ == "__main__":
    # --- 准备数据 ---
    # 使用一段示例文本 (实际训练需要更多数据)
    sample_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die, to sleep—
    No more—and by a sleep to say we end
    The heartache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wished. To die, to sleep;
    To sleep, perchance to dream. Ay, there's the rub,
    For in that sleep of death what dreams may come
    When we have shuffled off this mortal coil,
    Must give us pause. There's the respect
    That makes calamity of so long life.
    For who would bear the whips and scorns of time,
    The oppressor's wrong, the proud man's contumely,
    The pangs of despised love, the law's delay,
    The insolence of office and the spurns
    That patient merit of the unworthy takes,
    When he himself might his quietus make
    With a bare bodkin? Who would fardels bear,
    To grunt and sweat under a weary life,
    But that the dread of something after death,
    The undiscovered country from whose bourn
    No traveller returns, puzzles the will
    And makes us rather bear those ills we have
    Than fly to others that we know not of?
    Thus conscience does make cowards of us all,
    And thus the native hue of resolution
    Is sicklied o'er with the pale cast of thought,
    And enterprises of great pitch and moment
    With this regard their currents turn awry,
    And lose the name of action.
    """ * 20  # 重复多次以增加数据量

    print("=== Decoder-Only 模型训练示例 ===\n")

    # --- 构建分词器 ---
    tokenizer = SimpleCharTokenizer(sample_text)
    print(f"词汇表大小: {tokenizer.vocab_size}")
    print(f"文本长度: {len(sample_text)} 字符")
    print(f"Token 数量: {len(tokenizer.encode(sample_text, add_bos=False))}")

    # --- 构建模型配置 ---
    config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        d_model=128,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,  # GQA
        ffn_type="swiglu",
        norm_type="rmsnorm",
        dropout=0.1,
    )
    print(f"\n{config.summary()}")

    # --- 构建数据集 ---
    # 80% 训练, 20% 验证
    split = int(len(sample_text) * 0.8)
    train_dataset = TextDataset(sample_text[:split], tokenizer, seq_len=config.max_seq_len)
    val_dataset = TextDataset(sample_text[split:], tokenizer, seq_len=config.max_seq_len)
    print(f"\n训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")

    # --- 创建模型 ---
    model = DecoderOnlyModel(config)
    print(f"模型参数量: {model.count_parameters():,}")

    # --- 训练 ---
    log = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=5,
        batch_size=16,
        max_lr=1e-3,
        min_lr=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        grad_clip=1.0,
        eval_interval=50,
        device="auto",
    )

    # --- 生成示例 ---
    print("\n=== 生成示例 ===")
    model.eval()
    device = next(model.parameters()).device

    prompt_text = "To be"
    prompt_ids = tokenizer.encode(prompt_text, add_bos=True)
    prompt_tensor = torch.tensor([prompt_ids], device=device)

    output = generate(
        model, prompt_tensor, max_new_tokens=100,
        temperature=0.8, top_p=0.9,
    )
    generated_text = tokenizer.decode(output[0].tolist(), skip_special=True)
    print(f"Prompt: '{prompt_text}'")
    print(f"生成: '{generated_text}'")
