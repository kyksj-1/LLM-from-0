"""
SwiGLU 前馈网络

知识依赖:
- 模块 3（Transformer 核心架构）: 标准 FFN 结构
- 模块 4（Decoder-Only 架构）: SwiGLU 变体的数学推导

参考实现:
- code/decoder_only/model.py 中的 FeedForward 类
- code/transformer/feedforward.py

核心数学:

标准 FFN:
    FFN(x) = W_2 · GELU(W_1 · x + b_1) + b_2
    参数量: 2 × d × d_ff

SwiGLU FFN (Shazeer, 2020):
    SwiGLU(x) = W_down · (Swish(W_gate · x) ⊙ (W_up · x))
    其中 Swish(x) = x · sigmoid(x)，⊙ 表示逐元素相乘。
    参数量: 3 × d × d_ff（三个矩阵）

    为保持与标准 FFN 参数量一致:
    d_ff_swiglu = round(8/3 × d_model)

    SwiGLU 通过门控机制让网络选择性地激活信息，
    在相同参数量下表现优于标准 FFN + GELU。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU 前馈网络

    三个权重矩阵:
    - W_gate: [d_model, d_ff] - 门控路径
    - W_up:   [d_model, d_ff] - 值路径
    - W_down: [d_ff, d_model] - 输出投影

    前向计算: output = W_down @ (swish(W_gate @ x) * W_up @ x)

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # TODO: 初始化三个线性层（均无 bias）
        #   - self.w_gate: nn.Linear(d_model, d_ff, bias=False)
        #   - self.w_up:   nn.Linear(d_model, d_ff, bias=False)
        #   - self.w_down: nn.Linear(d_ff, d_model, bias=False)
        raise NotImplementedError(
            "TODO: 初始化 SwiGLU 的三个线性层\n"
            "参考: 模块 4 (Decoder-Only) 的 SwiGLU 章节\n"
            "参考实现: code/decoder_only/model.py 中的 FeedForward"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        SwiGLU 前向传播

        Args:
            x: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, d_model]

        实现步骤:
            1. gate = swish(W_gate @ x)   # swish(x) = x * sigmoid(x)，即 F.silu(x)
            2. up = W_up @ x
            3. hidden = gate * up          # 逐元素相乘（门控）
            4. output = W_down @ hidden
        """
        raise NotImplementedError(
            "TODO: 实现 SwiGLU 前向传播\n"
            "提示: PyTorch 的 F.silu() 即 Swish 激活函数\n"
            "参考: code/decoder_only/model.py"
        )
