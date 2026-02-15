"""
Decoder-Only 模型完整实现

本模块实现了一个完整的 Decoder-Only Transformer 语言模型，支持:
- MHA / GQA / MQA 三种注意力模式
- RMSNorm / LayerNorm 归一化
- SwiGLU / GeGLU / 标准 FFN
- RoPE / 学习位置编码
- 嵌入缩放 (Gemma 风格)
- 残差分支缩放初始化 (GPT-2 风格)

数学基础:
- 自回归分解: P(x_1,...,x_T) = prod P(x_t | x_{<t})
- 注意力: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
- GQA: 多个 Q 头共享一组 KV, 减少 KV Cache
- RoPE: 通过旋转矩阵编码相对位置

参考:
- Radford et al. (2019). Language Models are Unsupervised Multitask Learners. (GPT-2)
- Touvron et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
- Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
- Shazeer (2020). GLU Variants Improve Transformer.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


# ============================================================
# 归一化层
# ============================================================

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    去除均值中心化，仅做缩放。被 Llama、Gemma、DeepSeek 采用。

    数学: RMSNorm(x) = x / RMS(x) * gamma
    其中 RMS(x) = sqrt(mean(x^2) + eps)

    Args:
        dim: 归一化的维度
        eps: 防止除零的小量
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim]

        Returns:
            归一化后的张量, 形状与输入相同
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class LayerNorm(nn.Module):
    """
    标准 Layer Normalization

    数学: LN(x) = gamma * (x - mu) / sqrt(sigma^2 + eps) + beta

    Args:
        dim: 归一化的维度
        eps: 防止除零的小量
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


def build_norm(config: ModelConfig) -> nn.Module:
    """根据配置创建归一化层"""
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.d_model, config.norm_eps)
    elif config.norm_type == "layernorm":
        return LayerNorm(config.d_model, config.norm_eps)
    else:
        raise ValueError(f"不支持的归一化类型: {config.norm_type}")


# ============================================================
# 旋转位置编码 (RoPE)
# ============================================================

class RotaryPositionEncoding(nn.Module):
    """
    RoPE: 旋转位置编码

    核心思想: 通过旋转矩阵将绝对位置信息编码为相对位置信息。
    对于位置 m 的向量 x, 应用旋转:
        f(x, m) = R(m) x
    其中 R(m) 是块对角旋转矩阵。

    关键性质: <f(q,m), f(k,n)> = g(q, k, m-n)
    即内积只依赖于相对位置 m-n。

    Args:
        dim: 旋转编码的维度 (通常为 head_dim)
        max_seq_len: 预计算的最大序列长度
        base: 基底频率 (Llama 1/2: 10000, Llama 3: 500000)
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        # 频率向量: theta_i = 1 / (base^(2i/d)), i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 预计算 cos/sin 缓存
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        """预计算 cos/sin 缓存以加速推理"""
        t = torch.arange(max_seq_len, device=self.inv_freq.device).float()
        # 外积: [max_seq_len, dim/2]
        angles = torch.outer(t, self.inv_freq)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        对输入应用旋转位置编码

        Args:
            x: [batch, n_heads, seq_len, head_dim]
            offset: 位置偏移 (用于 KV Cache 增量推理)

        Returns:
            旋转后的张量, 形状与输入相同
        """
        seq_len = x.shape[2]

        # 取出当前位置范围的 cos/sin
        cos = self.cos_cached[offset:offset + seq_len]  # [seq_len, dim/2]
        sin = self.sin_cached[offset:offset + seq_len]

        # 扩展维度以匹配输入
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2]
        sin = sin.unsqueeze(0).unsqueeze(0)

        # 将 x 拆分为偶数和奇数位置
        x1 = x[..., ::2]   # 偶数位置
        x2 = x[..., 1::2]  # 奇数位置

        # 应用旋转: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos,
        ], dim=-1).flatten(-2)

        return rotated


# ============================================================
# 注意力层 (支持 MHA / GQA / MQA)
# ============================================================

class GroupedQueryAttention(nn.Module):
    """
    统一的注意力实现, 支持 MHA / GQA / MQA

    - n_kv_heads == n_heads: 标准 Multi-Head Attention (MHA)
    - n_kv_heads == 1: Multi-Query Attention (MQA)
    - 1 < n_kv_heads < n_heads: Grouped-Query Attention (GQA)

    数学:
    Q = xW_Q, K = xW_K, V = xW_V
    Attn(Q,K,V) = softmax(QK^T / sqrt(d_k)) V

    GQA 时, 每 n_rep 个 Q 头共享一组 KV 头:
    n_rep = n_heads / n_kv_heads

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = config.n_kv_groups  # 每个 KV 头对应的 Q 头数

        # Q/K/V/O 投影矩阵
        self.wq = nn.Linear(
            config.d_model, config.n_heads * self.head_dim, bias=config.bias
        )
        self.wk = nn.Linear(
            config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias
        )
        self.wv = nn.Linear(
            config.d_model, config.n_kv_heads * self.head_dim, bias=config.bias
        )
        self.wo = nn.Linear(
            config.n_heads * self.head_dim, config.d_model, bias=config.bias
        )

        self.attn_dropout = nn.Dropout(config.dropout)

        # RoPE (如果启用)
        if config.use_rope:
            self.rope = RotaryPositionEncoding(
                self.head_dim, config.max_seq_len, config.rope_base
            )
        else:
            self.rope = None

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch, seq_len, d_model]
            mask: 因果掩码 [seq_len, seq_len] 或 None (自动生成)

        Returns:
            [batch, seq_len, d_model]
        """
        B, S, _ = x.shape

        # --- 投影 ---
        q = self.wq(x)  # [B, S, n_heads * head_dim]
        k = self.wk(x)  # [B, S, n_kv_heads * head_dim]
        v = self.wv(x)  # [B, S, n_kv_heads * head_dim]

        # --- 重塑为多头形式 ---
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        # q: [B, n_heads, S, head_dim]
        # k, v: [B, n_kv_heads, S, head_dim]

        # --- 应用 RoPE ---
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # --- GQA: 复制 KV 头以匹配 Q 头数 ---
        if self.n_rep > 1:
            # [B, n_kv_heads, S, head_dim] -> [B, n_heads, S, head_dim]
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # --- 注意力计算 ---
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # scores: [B, n_heads, S, S]

        # 应用因果掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        else:
            # 自动生成因果掩码 (下三角)
            causal_mask = torch.triu(
                torch.ones(S, S, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # --- 加权求和 ---
        out = torch.matmul(attn_weights, v)  # [B, n_heads, S, head_dim]

        # --- 重塑并输出投影 ---
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.wo(out)


# ============================================================
# FFN 层 (标准 / SwiGLU / GeGLU)
# ============================================================

class StandardFFN(nn.Module):
    """
    标准 FFN: GELU(xW1)W2

    Args:
        d_model: 输入/输出维度
        d_ff: 中间层维度
        bias: 是否使用偏置
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN: (Swish(xW_gate) * xW_up) W_down

    被 Llama、DeepSeek 采用。Swish(x) = x * sigmoid(x) = SiLU(x)

    Args:
        d_model: 输入/输出维度
        d_ff: 中间层维度
        bias: 是否使用偏置
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU = Swish 激活 (PyTorch 内置)
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


class GeGLUFFN(nn.Module):
    """
    GeGLU FFN: (GELU(xW_gate) * xW_up) W_down

    被 Gemma 采用。与 SwiGLU 类似, 但使用 GELU 替代 Swish。

    Args:
        d_model: 输入/输出维度
        d_ff: 中间层维度
        bias: 是否使用偏置
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, d_ff: int, bias: bool = False, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        self.w_up = nn.Linear(d_model, d_ff, bias=bias)
        self.w_down = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.gelu(self.w_gate(x))
        up = self.w_up(x)
        return self.dropout(self.w_down(gate * up))


def build_ffn(config: ModelConfig) -> nn.Module:
    """根据配置创建 FFN 层"""
    if config.ffn_type == "standard":
        return StandardFFN(config.d_model, config.d_ff, config.bias, config.dropout)
    elif config.ffn_type == "swiglu":
        return SwiGLUFFN(config.d_model, config.d_ff, config.bias, config.dropout)
    elif config.ffn_type == "geglu":
        return GeGLUFFN(config.d_model, config.d_ff, config.bias, config.dropout)
    else:
        raise ValueError(f"不支持的 FFN 类型: {config.ffn_type}")


# ============================================================
# Transformer Block
# ============================================================

class DecoderBlock(nn.Module):
    """
    单个 Decoder Block (Pre-Norm 风格)

    结构:
    x -> RMSNorm -> Attention -> + x -> RMSNorm -> FFN -> + residual -> 输出

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = build_norm(config)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = build_norm(config)
        self.ffn = build_ffn(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: 因果掩码

        Returns:
            [batch, seq_len, d_model]
        """
        # Pre-Norm + Attention + 残差
        x = x + self.attn(self.attn_norm(x), mask)
        # Pre-Norm + FFN + 残差
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ============================================================
# 完整模型
# ============================================================

class DecoderOnlyModel(nn.Module):
    """
    完整的 Decoder-Only 语言模型

    支持 GPT / Llama / Gemma 风格配置:
    - GPT-2: LayerNorm + GELU + 学习PE + MHA + bias
    - Llama: RMSNorm + SwiGLU + RoPE + GQA + 无 bias
    - Gemma: RMSNorm + GeGLU + RoPE + GQA + 嵌入缩放

    Args:
        config: 模型配置 (ModelConfig 实例)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # --- Token Embedding ---
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # --- 位置编码 (仅在不使用 RoPE 时) ---
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        else:
            self.position_embedding = None

        self.embed_dropout = nn.Dropout(config.dropout)

        # --- Transformer Blocks ---
        self.layers = nn.ModuleList([
            DecoderBlock(config) for _ in range(config.n_layers)
        ])

        # --- 最终归一化 ---
        self.final_norm = build_norm(config)

        # --- LM Head ---
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # --- 权重共享 ---
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        # --- 初始化权重 ---
        self._init_weights()

    def _init_weights(self):
        """
        初始化权重

        策略:
        - 线性层: 正态分布 N(0, 0.02)
        - Embedding: 正态分布 N(0, 0.02)
        - 残差分支输出层: 缩放 1/sqrt(2*n_layers)
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # GPT-2 风格: 残差分支的输出投影缩放
        # 每层有两个残差分支 (Attention 和 FFN), 共 2*n_layers 个
        scale = 1.0 / math.sqrt(2 * self.config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight") or name.endswith("w2.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 * scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len] Token ID 序列
            mask: 注意力掩码 (可选, None 时自动使用因果掩码)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, S = input_ids.shape

        # Token Embedding
        x = self.token_embedding(input_ids)  # [B, S, d_model]

        # 嵌入缩放 (Gemma 风格)
        if self.config.scale_embeddings:
            x = x * math.sqrt(self.config.d_model)

        # 学习的位置编码 (非 RoPE 时)
        if self.position_embedding is not None:
            positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
            x = x + self.position_embedding(positions)

        x = self.embed_dropout(x)

        # 逐层通过 Transformer Blocks
        for layer in self.layers:
            x = layer(x, mask)

        # 最终归一化
        x = self.final_norm(x)

        # LM Head: 映射回词汇表维度
        logits = self.lm_head(x)  # [B, S, vocab_size]

        return logits

    def count_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops_per_token(self) -> int:
        """
        估算每个 token 的前向传播 FLOPs

        近似公式 (矩阵乘法 FLOPs = 2 * M * N * K):
        - 注意力投影: 2 * d * (n_h*d_h + 2*n_kv*d_h + n_h*d_h)
        - 注意力 QK/AV: 2 * S * n_h * d_h * 2
        - FFN: 2 * d * d_ff * {2 或 3}

        Returns:
            前向 FLOPs (每 token)
        """
        c = self.config
        d = c.d_model
        d_ff = c.d_ff
        L = c.n_layers
        V = c.vocab_size

        # 注意力 QKV + O 投影 (每层)
        attn_proj_flops = 2 * d * (c.n_heads + 2 * c.n_kv_heads + c.n_heads) * c.head_dim

        # FFN (每层)
        if c.ffn_type == "standard":
            ffn_flops = 2 * 2 * d * d_ff  # W1 + W2
        else:
            ffn_flops = 2 * 3 * d * d_ff  # W_gate + W_up + W_down

        # LM Head
        lm_head_flops = 2 * d * V

        # 总计 (不含注意力 QK^T 和 AV 的序列长度相关项)
        total_flops = L * (attn_proj_flops + ffn_flops) + lm_head_flops
        return total_flops

    def estimate_memory_gb(self, dtype_bytes: int = 2) -> float:
        """
        估算模型参数占用的显存 (GB)

        Args:
            dtype_bytes: 每个参数占用的字节数 (FP32=4, FP16/BF16=2)

        Returns:
            参数显存 (GB)
        """
        return self.count_parameters() * dtype_bytes / (1024 ** 3)


if __name__ == "__main__":
    from config import mini_config, gpt2_small_config

    # --- 测试 Mini 模型 ---
    print("=== Mini 模型测试 ===")
    config = mini_config()
    model = DecoderOnlyModel(config)

    print(f"参数量: {model.count_parameters():,}")
    print(f"估算参数量: {config.estimate_parameters():,}")
    print(f"模型显存 (FP16): {model.estimate_memory_gb(2):.3f} GB")
    print(f"每 token FLOPs: {model.estimate_flops_per_token():,}")

    # 前向传播测试
    input_ids = torch.randint(0, config.vocab_size, (2, 64))
    logits = model(input_ids)
    print(f"输入: {input_ids.shape}")
    print(f"输出: {logits.shape}")

    # 因果性验证
    input_ids_2 = input_ids.clone()
    input_ids_2[:, 32:] = 0
    logits_2 = model(input_ids_2)
    diff = (logits[:, :32] - logits_2[:, :32]).abs().max().item()
    print(f"因果性验证 (应接近 0): {diff:.8f}")

    # --- 测试 GPT-2 Small 配置 ---
    print("\n=== GPT-2 Small 配置测试 ===")
    gpt2_config = gpt2_small_config()
    gpt2_model = DecoderOnlyModel(gpt2_config)
    print(f"GPT-2 Small 参数量: {gpt2_model.count_parameters():,}")

    # 简短前向传播测试 (不需要太长序列)
    test_ids = torch.randint(0, gpt2_config.vocab_size, (1, 32))
    test_logits = gpt2_model(test_ids)
    print(f"输入: {test_ids.shape}")
    print(f"输出: {test_logits.shape}")
    print("GPT-2 Small 前向传播成功!")
