"""
Transformer Block

知识依赖:
- 模块 3（Transformer 核心架构）: 残差连接、归一化
- 模块 4（Decoder-Only 架构）: Pre-Norm 结构

参考实现:
- code/decoder_only/model.py 中的 TransformerBlock 类

核心结构 (Pre-Norm Decoder Block):

    x  ─────────────────────────┐
    │                           │
    RMSNorm                     │ (残差连接 1)
    │                           │
    GQA Attention               │
    │                           │
    + ←─────────────────────────┘
    │
    ─────────────────────────┐
    │                        │
    RMSNorm                  │ (残差连接 2)
    │                        │
    SwiGLU FFN               │
    │                        │
    + ←──────────────────────┘

Pre-Norm vs Post-Norm:
- Pre-Norm: 先归一化再计算，训练更稳定（Llama/GPT-3+ 采用）
- Post-Norm: 先计算再归一化（原版 Transformer）
- Pre-Norm 不需要 warmup 也能稳定训练，但最终性能可能略低
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .config import ModelConfig
from .attention import GQAAttention
from .feedforward import SwiGLUFeedForward


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    数学: RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

    与 LayerNorm 的区别: 不做均值中心化（去掉 mean 减法），
    计算更快，效果相当。被 Llama、Gemma、DeepSeek 采用。

    Args:
        dim: 归一化的维度大小
        eps: 防止除零的小量
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        # TODO: 初始化可学习的缩放参数 gamma (初始化为全 1)
        raise NotImplementedError(
            "TODO: 初始化 RMSNorm\n"
            "提示: self.weight = nn.Parameter(torch.ones(dim))\n"
            "参考: 模块 3 (Transformer) 的归一化章节"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., dim]

        Returns:
            归一化后的张量，形状与输入相同

        实现:
            rms = sqrt(mean(x^2, dim=-1) + eps)
            return x / rms * self.weight
        """
        raise NotImplementedError(
            "TODO: 实现 RMSNorm 前向传播\n"
            "参考: code/decoder_only/model.py 中的 RMSNorm"
        )


class TransformerBlock(nn.Module):
    """
    单个 Transformer Decoder Block (Pre-Norm 风格)

    组件:
    - attention_norm: RMSNorm (注意力前的归一化)
    - attention: GQA Attention
    - ffn_norm: RMSNorm (FFN 前的归一化)
    - ffn: SwiGLU FeedForward

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO: 初始化以下组件
        #   - self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        #   - self.attention = GQAAttention(config)
        #   - self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        #   - self.ffn = SwiGLUFeedForward(config)
        raise NotImplementedError(
            "TODO: 初始化 Transformer Block 的四个组件\n"
            "参考: 模块 4 (Decoder-Only) 的 Block 结构图"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Pre-Norm Decoder Block 前向传播

        Args:
            x: [batch, seq_len, d_model]
            mask: 因果掩码
            kv_cache: 推理时的 KV Cache

        Returns:
            (output, new_kv_cache)

        实现步骤 (Pre-Norm):
            1. residual = x
            2. x = attention_norm(x)
            3. x, kv_cache = attention(x, mask, kv_cache)
            4. x = residual + x               # 残差连接 1
            5. residual = x
            6. x = ffn_norm(x)
            7. x = ffn(x)
            8. x = residual + x               # 残差连接 2
        """
        raise NotImplementedError(
            "TODO: 实现 Pre-Norm Block 前向传播\n"
            "关键: 先 Norm 再计算，残差连接跨越 Norm+计算\n"
            "参考: code/decoder_only/model.py 中的 TransformerBlock"
        )
