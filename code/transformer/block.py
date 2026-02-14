"""
Transformer Block 实现

组装注意力、归一化、前馈网络为完整的 Transformer Block。

Pre-Norm 结构 (现代 LLM 标准):
    h = x + Dropout(Attention(RMSNorm(x)))
    y = h + Dropout(FFN(RMSNorm(h)))

参考:
- Vaswani et al. (2017). Attention Is All You Need.
- Xiong et al. (2020). On Layer Normalization in the Transformer Architecture.
"""

import torch
import torch.nn as nn
from typing import Optional

from .attention import MultiHeadAttention
from .normalization import RMSNorm
from .feedforward import SwiGLU


class TransformerBlock(nn.Module):
    """
    完整的 Transformer Block (Pre-Norm 风格)

    组成:
    1. RMSNorm + Multi-Head Attention + 残差连接
    2. RMSNorm + SwiGLU FFN + 残差连接

    Args:
        d_model: 模型维度
        n_heads: 注意力头数量
        d_ff: FFN 中间维度 (默认 8/3 * d_model)
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        # 注意力子层
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.attention_norm = RMSNorm(d_model)

        # FFN 子层
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.ffn_norm = RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
            mask: 注意力掩码
            is_causal: 是否使用因果掩码

        Returns:
            [batch, seq_len, d_model]
        """
        # Pre-Norm + Attention + 残差
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x, x, x, mask, is_causal)
        x = self.dropout(x)
        x = residual + x

        # Pre-Norm + FFN + 残差
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = residual + x

        return x
