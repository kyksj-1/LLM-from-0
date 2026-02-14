"""
code/transformer 包

提供 Transformer 核心组件的模块化实现。

组件:
- attention: Self-Attention, Multi-Head Attention
- normalization: LayerNorm, RMSNorm
- feedforward: FFN, SwiGLU
- block: TransformerBlock
- model: 完整的 Decoder-only Transformer
"""

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .normalization import LayerNorm, RMSNorm
from .feedforward import FeedForward, SwiGLU
from .block import TransformerBlock
from .model import Transformer

__all__ = [
    "MultiHeadAttention",
    "scaled_dot_product_attention",
    "LayerNorm",
    "RMSNorm",
    "FeedForward",
    "SwiGLU",
    "TransformerBlock",
    "Transformer",
]
