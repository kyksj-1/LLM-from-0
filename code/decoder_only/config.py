"""
Decoder-Only 模型配置

本模块定义了灵活的模型配置类，支持 GPT / Llama / Gemma 三种风格的 Decoder-Only 架构。
通过 dataclass 设计，可以方便地切换归一化方式、激活函数、位置编码等组件。

预置配置:
- gpt2_small_config(): GPT-2 Small (124M) 配置
- llama2_7b_config(): Llama 2 7B 配置
- llama3_8b_config(): Llama 3 8B 配置
- gemma2_2b_config(): Gemma 2 2B 配置
- mini_config(): 用于测试的极小模型配置

参考:
- Radford et al. (2019). Language Models are Unsupervised Multitask Learners. (GPT-2)
- Touvron et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
- Team Gemma (2024). Gemma 2: Improving Open Language Models at a Practical Size.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Decoder-Only 模型的统一配置类

    通过调整参数可以模拟不同架构风格:
    - GPT-2 风格: LayerNorm + GELU FFN + 学习位置编码 + MHA + 有 bias
    - Llama 风格: RMSNorm + SwiGLU + RoPE + GQA + 无 bias
    - Gemma 风格: RMSNorm + GeGLU + RoPE + GQA + 嵌入缩放 + 大词汇表

    Args:
        vocab_size: 词汇表大小
        max_seq_len: 最大序列长度
        d_model: 隐藏维度 (模型宽度)
        n_layers: Transformer Block 层数
        n_heads: Query 注意力头数
        n_kv_heads: KV 注意力头数 (None 表示与 n_heads 相同, 即 MHA;
                    设为 1 即 MQA; 设为 1 < x < n_heads 即 GQA)
        d_ff: FFN 隐藏维度 (None 表示自动计算)
        ffn_type: FFN 类型 ("standard" / "swiglu" / "geglu")
        norm_type: 归一化类型 ("layernorm" / "rmsnorm")
        norm_eps: 归一化 epsilon
        rope_base: RoPE 基底频率
        use_rope: 是否使用旋转位置编码 (False 时使用学习的位置编码)
        dropout: Dropout 概率 (预训练通常为 0)
        tie_weights: 是否共享 Embedding 和 LM Head 的权重
        bias: 线性层是否使用偏置项
        scale_embeddings: 是否对嵌入乘以 sqrt(d_model) (Gemma 风格)
    """

    # --- 基本结构 ---
    vocab_size: int = 32000
    max_seq_len: int = 2048
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: Optional[int] = None

    # --- FFN ---
    d_ff: Optional[int] = None
    ffn_type: str = "swiglu"

    # --- 归一化 ---
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6

    # --- 位置编码 ---
    rope_base: float = 10000.0
    use_rope: bool = True

    # --- 训练相关 ---
    dropout: float = 0.0
    tie_weights: bool = True
    bias: bool = False

    # --- 嵌入缩放 ---
    scale_embeddings: bool = False

    def __post_init__(self):
        """自动计算默认值并验证配置合法性"""
        # KV 头数默认与 Q 头数相同 (MHA)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # 自动计算 FFN 隐藏维度
        if self.d_ff is None:
            if self.ffn_type == "standard":
                self.d_ff = 4 * self.d_model
            else:
                # SwiGLU / GeGLU: 为保持与标准 FFN 相同参数量
                # 标准 FFN: 2 * d * d_ff = 2 * d * 4d = 8d^2
                # SwiGLU:   3 * d * d_ff' (三个矩阵)
                # 令 3 * d * d_ff' = 8d^2 => d_ff' = 8d/3
                self.d_ff = int(8 * self.d_model / 3)
                # 向上对齐到 256 的倍数 (提高计算效率)
                self.d_ff = ((self.d_ff + 255) // 256) * 256

        # 验证配置合法性
        self._validate()

    def _validate(self):
        """验证配置参数的合法性"""
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) 必须能被 n_heads ({self.n_heads}) 整除"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) 必须能被 n_kv_heads ({self.n_kv_heads}) 整除"
        )
        assert self.ffn_type in ("standard", "swiglu", "geglu"), (
            f"不支持的 FFN 类型: {self.ffn_type}"
        )
        assert self.norm_type in ("layernorm", "rmsnorm"), (
            f"不支持的归一化类型: {self.norm_type}"
        )

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度"""
        return self.d_model // self.n_heads

    @property
    def n_kv_groups(self) -> int:
        """每个 KV 头对应的 Q 头数 (GQA 中的重复因子)"""
        return self.n_heads // self.n_kv_heads

    def estimate_parameters(self) -> int:
        """
        估算模型参数量

        Returns:
            估算的总参数量
        """
        d = self.d_model
        L = self.n_layers
        V = self.vocab_size

        # 注意力层: Q 投影 + KV 投影 + 输出投影
        n_kv = self.n_kv_heads
        d_h = self.head_dim
        attn_params = d * (self.n_heads * d_h) + 2 * d * (n_kv * d_h) + (self.n_heads * d_h) * d

        # FFN 层
        if self.ffn_type == "standard":
            ffn_params = 2 * d * self.d_ff  # W1 + W2
        else:
            ffn_params = 3 * d * self.d_ff  # W_gate + W_up + W_down

        # RMSNorm / LayerNorm (每层 2 个 + 最终 1 个)
        norm_params = d * (2 * L + 1)
        if self.norm_type == "layernorm":
            norm_params *= 2  # gamma + beta

        # Embedding + LM Head
        embed_params = V * d
        if not self.tie_weights:
            embed_params *= 2  # Embedding + LM Head 各一份

        # 位置编码
        pos_params = 0
        if not self.use_rope:
            pos_params = self.max_seq_len * d

        # 总参数量
        total = L * (attn_params + ffn_params) + norm_params + embed_params + pos_params
        return total

    def summary(self) -> str:
        """生成配置摘要字符串"""
        params = self.estimate_parameters()
        lines = [
            f"=== 模型配置摘要 ===",
            f"  架构风格: {'GPT' if not self.use_rope else 'Llama/Gemma'} 风格",
            f"  词汇表大小: {self.vocab_size:,}",
            f"  最大序列长度: {self.max_seq_len:,}",
            f"  隐藏维度: {self.d_model}",
            f"  层数: {self.n_layers}",
            f"  注意力头: {self.n_heads} Q头, {self.n_kv_heads} KV头 (head_dim={self.head_dim})",
            f"  FFN: {self.ffn_type} (d_ff={self.d_ff})",
            f"  归一化: {self.norm_type}",
            f"  位置编码: {'RoPE (base={})'.format(self.rope_base) if self.use_rope else '学习的位置编码'}",
            f"  Dropout: {self.dropout}",
            f"  权重共享: {self.tie_weights}",
            f"  偏置项: {self.bias}",
            f"  嵌入缩放: {self.scale_embeddings}",
            f"  估算参数量: {params:,} ({params / 1e6:.1f}M)",
        ]
        return "\n".join(lines)


# ========== 预置配置函数 ==========

def gpt2_small_config() -> ModelConfig:
    """GPT-2 Small (124M) 配置"""
    return ModelConfig(
        vocab_size=50257,
        max_seq_len=1024,
        d_model=768,
        n_layers=12,
        n_heads=12,
        n_kv_heads=12,
        d_ff=3072,
        ffn_type="standard",
        norm_type="layernorm",
        use_rope=False,
        dropout=0.1,
        tie_weights=True,
        bias=True,
    )


def llama2_7b_config() -> ModelConfig:
    """Llama 2 7B 配置"""
    return ModelConfig(
        vocab_size=32000,
        max_seq_len=4096,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=32,  # Llama 2 7B 使用 MHA
        d_ff=11008,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=10000.0,
        dropout=0.0,
        tie_weights=False,
        bias=False,
    )


def llama3_8b_config() -> ModelConfig:
    """Llama 3 8B 配置"""
    return ModelConfig(
        vocab_size=128256,
        max_seq_len=8192,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=8,  # GQA: 8 个 KV 头
        d_ff=14336,
        ffn_type="swiglu",
        norm_type="rmsnorm",
        rope_base=500000.0,  # 增大基底以支持更长上下文
        dropout=0.0,
        tie_weights=False,
        bias=False,
    )


def gemma2_2b_config() -> ModelConfig:
    """Gemma 2 2B 配置"""
    return ModelConfig(
        vocab_size=256000,
        max_seq_len=8192,
        d_model=2304,
        n_layers=26,
        n_heads=8,
        n_kv_heads=4,  # GQA
        d_ff=9216,
        ffn_type="geglu",
        norm_type="rmsnorm",
        dropout=0.0,
        tie_weights=True,
        bias=False,
        scale_embeddings=True,
    )


def mini_config() -> ModelConfig:
    """用于测试的极小模型配置 (约 4M 参数)"""
    return ModelConfig(
        vocab_size=1000,
        max_seq_len=256,
        d_model=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,  # 测试 GQA
        ffn_type="swiglu",
        norm_type="rmsnorm",
        dropout=0.0,
    )


if __name__ == "__main__":
    # 展示各预置配置的参数量
    configs = {
        "Mini (测试)": mini_config(),
        "GPT-2 Small": gpt2_small_config(),
        "Llama 2 7B": llama2_7b_config(),
        "Llama 3 8B": llama3_8b_config(),
        "Gemma 2 2B": gemma2_2b_config(),
    }

    for name, cfg in configs.items():
        params = cfg.estimate_parameters()
        print(f"{name:20s}: {params:>15,} 参数 ({params / 1e6:>8.1f}M)")

    print("\n" + "=" * 60)
    print("\nMini 配置详情:")
    print(mini_config().summary())
