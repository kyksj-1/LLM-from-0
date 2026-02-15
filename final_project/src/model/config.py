"""
模型与训练配置

本文件是终极项目中唯一完整提供的代码文件。
它定义了模型和训练的所有超参数，是其他所有模块的基础依赖。

包含:
- ModelConfig: 模型架构配置（层数、维度、注意力头数等）
- TrainingConfig: 训练超参数配置（学习率、batch size、优化器等）
- config_300m(): Version A 预设配置（300M 参数，单卡）
- config_1b(): Version B 预设配置（1B 参数，多卡）
- from_yaml(): 从 YAML 文件加载配置

知识依赖: 模块 4（Decoder-Only 架构配置）、模块 8B（Scaling Laws 指导超参数选择）
参考实现: code/decoder_only/config.py
"""

import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """
    Decoder-Only LLM 的模型配置

    本配置采用 Llama 风格架构:
    - RMSNorm（而非 LayerNorm）
    - SwiGLU FFN（而非标准 FFN）
    - RoPE 旋转位置编码（而非学习的位置编码）
    - GQA 分组查询注意力（而非标准 MHA）
    - Pre-Norm（而非 Post-Norm）
    - 无偏置项

    这些选择基于 Llama 2/3、DeepSeek 等现代 LLM 的最佳实践。
    详细原理见模块 4（Decoder-Only 架构）的 README.md。

    Args:
        vocab_size: 词汇表大小
        max_seq_len: 最大序列长度
        d_model: 隐藏层维度（模型宽度）
        n_layers: Transformer Block 层数（模型深度）
        n_heads: Query 注意力头数
        n_kv_heads: Key/Value 注意力头数（GQA 的关键参数）
        d_ff: SwiGLU FFN 的中间维度
        norm_eps: RMSNorm 的 epsilon
        rope_base: RoPE 基底频率（影响位置编码的周期）
        dropout: Dropout 概率（预训练通常设为 0）
        tie_weights: 是否共享 Embedding 和 LM Head 权重
    """

    # --- 模型结构 ---
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 4       # GQA: 每 4 个 Q 头共享 1 组 KV
    d_ff: int = 2730           # SwiGLU: ≈ 8/3 * d_model
    norm_eps: float = 1e-6
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_weights: bool = True

    def __post_init__(self):
        """验证配置合法性"""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) 必须能被 n_kv_heads ({self.n_kv_heads}) 整除"
        )

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度: d_model / n_heads"""
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """每个 KV 头对应的 Q 头数（GQA 的重复因子）"""
        return self.n_heads // self.n_kv_heads

    def estimate_params(self) -> int:
        """
        估算模型总参数量

        计算公式（Llama 风格，无 bias）:
        - 每层注意力: d * (n_heads + 2 * n_kv_heads) * head_dim + n_heads * head_dim * d
        - 每层 FFN (SwiGLU): 3 * d * d_ff（W_gate, W_up, W_down）
        - 每层 Norm: 2 * d（attention norm + ffn norm）
        - Embedding: vocab_size * d
        - Final Norm: d
        - LM Head: 若不共享权重则 vocab_size * d

        Returns:
            估算参数量
        """
        d = self.d_model
        h = self.head_dim
        L = self.n_layers

        # 每层参数
        attn_per_layer = d * self.n_heads * h + 2 * d * self.n_kv_heads * h + self.n_heads * h * d
        ffn_per_layer = 3 * d * self.d_ff
        norm_per_layer = 2 * d

        # 全局参数
        embedding = self.vocab_size * d
        final_norm = d
        lm_head = 0 if self.tie_weights else self.vocab_size * d

        total = L * (attn_per_layer + ffn_per_layer + norm_per_layer) + embedding + final_norm + lm_head
        return total

    def summary(self) -> str:
        """生成配置摘要"""
        params = self.estimate_params()
        return (
            f"模型配置摘要:\n"
            f"  层数: {self.n_layers}, 隐藏维度: {self.d_model}\n"
            f"  注意力: {self.n_heads} Q头, {self.n_kv_heads} KV头 (GQA, head_dim={self.head_dim})\n"
            f"  FFN: SwiGLU (d_ff={self.d_ff})\n"
            f"  词汇表: {self.vocab_size}, 最大序列: {self.max_seq_len}\n"
            f"  估算参数量: {params:,} ({params / 1e6:.0f}M)"
        )


@dataclass
class TrainingConfig:
    """
    训练超参数配置

    包含预训练、SFT、DPO 各阶段的超参数。
    超参数选择依据见模块 8B（Scaling Laws）和模块 8C（训练工程）。

    Args:
        # --- 优化器 ---
        learning_rate: 峰值学习率
        min_lr_ratio: 最小学习率与峰值的比值
        weight_decay: AdamW 权重衰减
        beta1: Adam beta1
        beta2: Adam beta2
        max_grad_norm: 梯度裁剪阈值

        # --- 训练规模 ---
        batch_size: 每个 GPU 的 micro batch size
        gradient_accumulation_steps: 梯度累积步数
        max_steps: 最大训练步数
        warmup_steps: 学习率预热步数

        # --- 精度与效率 ---
        mixed_precision: 混合精度类型 ("bf16" / "fp16" / "none")
        gradient_checkpointing: 是否使用梯度检查点（省显存）

        # --- 日志与保存 ---
        log_interval: 日志打印间隔（步）
        save_interval: checkpoint 保存间隔（步）
        eval_interval: 评估间隔（步）

        # --- SFT 专用 ---
        sft_learning_rate: SFT 学习率（通常比预训练小）
        sft_max_steps: SFT 训练步数

        # --- DPO 专用 ---
        dpo_beta: DPO 的 β 参数（控制偏离参考模型的程度）
        dpo_learning_rate: DPO 学习率
    """

    # --- 优化器 ---
    learning_rate: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    max_grad_norm: float = 1.0

    # --- 训练规模 ---
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    max_steps: int = 50000
    warmup_steps: int = 2000

    # --- 精度与效率 ---
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True

    # --- 日志与保存 ---
    log_interval: int = 10
    save_interval: int = 1000
    eval_interval: int = 500
    output_dir: str = "./checkpoints"

    # --- SFT ---
    sft_learning_rate: float = 2e-5
    sft_max_steps: int = 3000

    # --- DPO ---
    dpo_beta: float = 0.1
    dpo_learning_rate: float = 5e-7

    @property
    def effective_batch_size(self) -> int:
        """等效全局 batch size = micro_batch × gradient_accumulation"""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def total_tokens_per_step(self) -> str:
        """提示：需要乘以 seq_len 和 GPU 数才是真正的 tokens/step"""
        return f"batch_size({self.batch_size}) × grad_accum({self.gradient_accumulation_steps}) × seq_len × n_gpus"


def config_300m() -> tuple:
    """
    Version A: 300M 参数模型配置（单卡 24GB GPU）

    架构选择理由:
    - 24 层 × 1024 维 ≈ 300M 参数，单张 24GB GPU 可训练
    - GQA (4 KV heads) 减少 KV Cache 显存占用
    - SwiGLU FFN 比标准 FFN 效果更好（相同参数量下）
    - 梯度检查点 + BF16 混合精度节省显存

    Returns:
        (ModelConfig, TrainingConfig)
    """
    model = ModelConfig(
        vocab_size=32000,
        max_seq_len=2048,
        d_model=1024,
        n_layers=24,
        n_heads=16,
        n_kv_heads=4,
        d_ff=2730,
        rope_base=10000.0,
        dropout=0.0,
        tie_weights=True,
    )
    training = TrainingConfig(
        learning_rate=3e-4,
        batch_size=8,
        gradient_accumulation_steps=4,
        max_steps=50000,
        warmup_steps=2000,
        mixed_precision="bf16",
        gradient_checkpointing=True,
    )
    return model, training


def config_1b() -> tuple:
    """
    Version B: 1B 参数模型配置（4-8 卡 GPU）

    与 300M 版本的主要区别:
    - 更大的模型: 32 层 × 2048 维
    - 更大的词汇表: 64K（支持更多语言和特殊 token）
    - 更长的上下文: 4096（需要更多 KV Cache 显存）
    - 需要分布式训练: FSDP 或 DeepSpeed ZeRO

    Returns:
        (ModelConfig, TrainingConfig)
    """
    model = ModelConfig(
        vocab_size=64000,
        max_seq_len=4096,
        d_model=2048,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,
        d_ff=5462,
        rope_base=10000.0,
        dropout=0.0,
        tie_weights=False,  # 1B 模型不共享权重
    )
    training = TrainingConfig(
        learning_rate=3e-4,
        batch_size=4,          # 每卡 batch 更小
        gradient_accumulation_steps=8,
        max_steps=100000,
        warmup_steps=4000,
        mixed_precision="bf16",
        gradient_checkpointing=True,
    )
    return model, training


def from_yaml(path: str) -> tuple:
    """
    从 YAML 配置文件加载模型和训练配置

    Args:
        path: YAML 文件路径（如 configs/model_300m.yaml）

    Returns:
        (ModelConfig, TrainingConfig)
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = ModelConfig(**cfg.get("model", {}))
    train_cfg = TrainingConfig(**cfg.get("training", {}))

    return model_cfg, train_cfg


if __name__ == "__main__":
    # 展示两个版本的配置
    print("=" * 60)
    print("终极项目模型配置")
    print("=" * 60)

    for name, config_fn in [("Version A (300M)", config_300m), ("Version B (1B)", config_1b)]:
        model_cfg, train_cfg = config_fn()
        print(f"\n--- {name} ---")
        print(model_cfg.summary())
        print(f"  训练: lr={train_cfg.learning_rate}, "
              f"effective_bs={train_cfg.effective_batch_size}, "
              f"steps={train_cfg.max_steps}")
        print(f"  tokens/step = {train_cfg.total_tokens_per_step}")
