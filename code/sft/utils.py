"""
SFT 工具函数

本模块提供监督微调中常用的工具函数:
- 参数统计（总参数/可训练参数/冻结参数）
- 显存估算
- 训练日志工具
- 数据处理辅助函数

参考:
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import math


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    统计模型的参数量

    Args:
        model: PyTorch 模型

    Returns:
        包含各类参数统计的字典:
        - total: 总参数量
        - trainable: 可训练参数量
        - frozen: 冻结参数量
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
    }


def print_trainable_parameters(model: nn.Module) -> None:
    """
    打印模型的参数统计信息（类似 PEFT 库的 print_trainable_parameters）

    Args:
        model: PyTorch 模型
    """
    stats = count_parameters(model)
    ratio = stats["trainable"] / max(stats["total"], 1) * 100

    print(
        f"trainable params: {stats['trainable']:,} || "
        f"all params: {stats['total']:,} || "
        f"trainable%: {ratio:.4f}"
    )


def estimate_memory(
    model: nn.Module,
    batch_size: int = 1,
    seq_len: int = 2048,
    optimizer_type: str = "adam",
    mixed_precision: bool = True,
) -> Dict[str, float]:
    """
    估算模型训练的显存占用（GB）

    显存组成:
    1. 模型参数: 全精度权重 + 可训练参数的梯度
    2. 优化器状态: Adam 需要 2 倍可训练参数的状态
    3. 激活值: 与 batch_size * seq_len * d_model * num_layers 成正比
    4. 临时缓冲区

    Args:
        model: 模型
        batch_size: 批量大小
        seq_len: 序列长度
        optimizer_type: 优化器类型 ("adam" 或 "sgd")
        mixed_precision: 是否使用混合精度训练

    Returns:
        各组件显存占用（GB）
    """
    stats = count_parameters(model)
    bytes_per_param = 2 if mixed_precision else 4  # FP16 或 FP32

    # 1. 模型参数
    param_memory = stats["total"] * bytes_per_param / 1e9

    # 2. 梯度（只有可训练参数）
    grad_memory = stats["trainable"] * bytes_per_param / 1e9

    # 3. 优化器状态
    if optimizer_type.lower() == "adam":
        # Adam: momentum (fp32) + variance (fp32)
        optimizer_memory = stats["trainable"] * 4 * 2 / 1e9
    elif optimizer_type.lower() == "sgd":
        optimizer_memory = stats["trainable"] * 4 / 1e9  # momentum
    else:
        optimizer_memory = 0

    # 4. 激活值估算（粗略）
    # 每个 Transformer 层大约需要 2 * batch * seq * d_model 的激活
    # 这里简化估算
    d_model = _estimate_hidden_size(model)
    n_layers = _estimate_num_layers(model)
    activation_memory = (
        2 * batch_size * seq_len * d_model * n_layers * bytes_per_param / 1e9
    )

    total = param_memory + grad_memory + optimizer_memory + activation_memory

    return {
        "parameters_GB": param_memory,
        "gradients_GB": grad_memory,
        "optimizer_GB": optimizer_memory,
        "activations_GB": activation_memory,
        "total_GB": total,
    }


def _estimate_hidden_size(model: nn.Module) -> int:
    """
    估算模型的隐藏维度

    Args:
        model: 模型

    Returns:
        估算的隐藏维度大小
    """
    for name, param in model.named_parameters():
        if "weight" in name and len(param.shape) == 2:
            return max(param.shape)
    return 512  # 默认值


def _estimate_num_layers(model: nn.Module) -> int:
    """
    估算模型的层数

    Args:
        model: 模型

    Returns:
        估算的层数
    """
    # 尝试从常见的属性名推断
    for attr_name in ["layers", "blocks", "encoder", "decoder"]:
        if hasattr(model, attr_name):
            child = getattr(model, attr_name)
            if isinstance(child, nn.ModuleList):
                return len(child)
    return 12  # 默认值


def compute_lora_savings(
    d_model: int,
    rank: int,
    num_target_layers: int = 4,
    num_transformer_layers: int = 32,
) -> Dict[str, float]:
    """
    计算 LoRA 相对全参数微调的参数节省

    Args:
        d_model: 模型隐藏维度
        rank: LoRA 秩
        num_target_layers: 每个 Transformer 层中应用 LoRA 的线性层数量
                          (如 Q,K,V,O = 4)
        num_transformer_layers: Transformer 总层数

    Returns:
        参数统计字典
    """
    # 每个线性层的参数量
    full_params_per_layer = d_model * d_model
    lora_params_per_layer = 2 * d_model * rank  # A + B

    # 总计
    total_full = full_params_per_layer * num_target_layers * num_transformer_layers
    total_lora = lora_params_per_layer * num_target_layers * num_transformer_layers
    ratio = total_lora / total_full * 100

    return {
        "full_params": total_full,
        "lora_params": total_lora,
        "ratio_percent": ratio,
        "compression_ratio": total_full / total_lora,
    }


def create_prompt_mask(
    input_ids: torch.Tensor,
    prompt_lengths: List[int],
) -> torch.Tensor:
    """
    创建 Prompt Masking 的 labels

    将指令部分的 label 设为 -100，只保留回答部分。

    Args:
        input_ids: token ID 张量 (batch, seq_len)
        prompt_lengths: 每个样本的 prompt 长度列表

    Returns:
        labels 张量（prompt 部分为 -100）
    """
    labels = input_ids.clone()

    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100

    return labels


def compute_effective_batch_size(
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    num_gpus: int = 1,
) -> int:
    """
    计算有效批量大小

    effective_batch_size = micro_batch * grad_accum * num_gpus

    Args:
        micro_batch_size: 单次前向的批量大小
        gradient_accumulation_steps: 梯度累积步数
        num_gpus: GPU 数量

    Returns:
        有效批量大小
    """
    return micro_batch_size * gradient_accumulation_steps * num_gpus


def format_number(n: int) -> str:
    """
    格式化大数字

    Args:
        n: 数字

    Returns:
        格式化字符串 (如 "6.7B", "13.0B")
    """
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    else:
        return str(n)


if __name__ == "__main__":
    print("=" * 60)
    print("SFT 工具函数演示")
    print("=" * 60)

    # === 1. 参数统计 ===
    print("\n--- 1. 参数统计 ---")

    class DemoModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32000, 512)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(512, 8, 2048, batch_first=True)
                for _ in range(6)
            ])
            self.head = nn.Linear(512, 32000)

        def forward(self, x):
            x = self.embed(x)
            for layer in self.layers:
                x = layer(x)
            return self.head(x)

    model = DemoModel()
    stats = count_parameters(model)
    print(f"总参数: {format_number(stats['total'])}")
    print(f"可训练: {format_number(stats['trainable'])}")
    print(f"冻结: {format_number(stats['frozen'])}")
    print_trainable_parameters(model)

    # 冻结部分参数
    for param in model.embed.parameters():
        param.requires_grad = False
    print("\n冻结 Embedding 后:")
    print_trainable_parameters(model)

    # === 2. 显存估算 ===
    print("\n--- 2. 显存估算 ---")
    mem = estimate_memory(model, batch_size=4, seq_len=2048)
    print(f"参数显存: {mem['parameters_GB']:.2f} GB")
    print(f"梯度显存: {mem['gradients_GB']:.2f} GB")
    print(f"优化器显存: {mem['optimizer_GB']:.2f} GB")
    print(f"激活值显存: {mem['activations_GB']:.2f} GB")
    print(f"总计: {mem['total_GB']:.2f} GB")

    # === 3. LoRA 参数节省分析 ===
    print("\n--- 3. LoRA 参数节省分析 ---")
    print(f"{'模型':>15} {'d_model':>8} {'rank':>6} {'全参数':>12} {'LoRA参数':>12} {'比例':>8} {'压缩比':>8}")
    print("-" * 75)

    configs = [
        ("Llama-7B", 4096, 32, 4, 32),
        ("Llama-13B", 5120, 32, 4, 40),
        ("Llama-70B", 8192, 32, 4, 80),
        ("Llama-7B (r=8)", 4096, 8, 4, 32),
        ("Llama-7B (r=64)", 4096, 64, 4, 32),
    ]

    for name, d, r, n_target, n_layers in configs:
        result = compute_lora_savings(d, r, n_target, n_layers)
        print(
            f"{name:>15} {d:>8} {r:>6} "
            f"{format_number(result['full_params']):>12} "
            f"{format_number(result['lora_params']):>12} "
            f"{result['ratio_percent']:>7.2f}% "
            f"{result['compression_ratio']:>7.0f}x"
        )

    # === 4. Prompt Masking 演示 ===
    print("\n--- 4. Prompt Masking 演示 ---")
    batch_ids = torch.tensor([
        [101, 2023, 345, 789, 101, 456, 234, 0],
        [101, 567, 890, 101, 321, 654, 0, 0],
    ])
    prompt_lens = [4, 3]  # 第一条 prompt 4 个 token，第二条 3 个 token

    labels = create_prompt_mask(batch_ids, prompt_lens)
    print(f"input_ids:\n{batch_ids}")
    print(f"prompt 长度: {prompt_lens}")
    print(f"labels (prompt 部分为 -100):\n{labels}")

    # === 5. 有效批量大小 ===
    print("\n--- 5. 有效批量大小计算 ---")
    scenarios = [
        (4, 8, 1),   # 单 GPU
        (4, 8, 4),   # 4 GPU
        (2, 16, 8),  # 8 GPU
    ]
    for micro, accum, gpus in scenarios:
        eff = compute_effective_batch_size(micro, accum, gpus)
        print(f"micro={micro}, accum={accum}, gpus={gpus} -> 有效 batch={eff}")
