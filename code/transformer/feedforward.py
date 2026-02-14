"""
前馈网络实现：标准 FFN, SwiGLU

本模块实现了 Transformer Block 中的前馈网络层。

标准 FFN: FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
SwiGLU:   SwiGLU(x) = (Swish(xW_gate) * xW_up) W_down

SwiGLU 使用门控机制，被 Llama/Gemma/DeepSeek 等模型采用。

参考:
- Shazeer (2020). GLU Variants Improve Transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    标准前馈网络 (使用 GELU 激活)

    FFN(x) = Dropout(W_2(GELU(W_1(x))))

    中间维度通常为 4 * d_model。

    Args:
        d_model: 模型维度
        d_ff: 中间层维度 (默认 4 * d_model)
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_model]
        Returns:
            [..., d_model]
        """
        return self.dropout(self.w2(self.gelu(self.w1(x))))


class SwiGLU(nn.Module):
    """
    SwiGLU 前馈网络

    SwiGLU(x) = (Swish(x W_gate) * x W_up) W_down

    其中 Swish(x) = x * sigmoid(x)

    门控机制让网络选择性地传递信息。
    为保持与标准 FFN 相同的参数量，中间维度设为 2/3 * 4d = 8d/3。

    Args:
        d_model: 模型维度
        d_ff: 中间层维度 (默认 8/3 * d_model)
        dropout: Dropout 概率
    """

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or int(8 * d_model / 3)

        # 门控投影: 计算 Swish(x W_gate)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        # 上投影: 计算 x W_up
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        # 下投影: 投影回 d_model
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_model]
        Returns:
            [..., d_model]
        """
        # Swish(x W_gate) = sigmoid(x W_gate) * (x W_gate)
        gate = F.silu(self.w_gate(x))
        # 上投影
        up = self.w_up(x)
        # 门控 + 下投影
        return self.dropout(self.w_down(gate * up))
