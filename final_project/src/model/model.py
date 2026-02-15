"""
完整的 Decoder-Only LLM

知识依赖:
- 模块 4（Decoder-Only 架构）: 完整模型组装、权重初始化、损失计算
- 模块 3（Transformer）: Embedding、LM Head

参考实现:
- code/decoder_only/model.py 中的 DecoderOnlyLM 类

模型结构:

    input_ids [batch, seq_len]
        │
        ▼
    Token Embedding  →  [batch, seq_len, d_model]
        │
        ▼
    TransformerBlock × n_layers  (每层含 GQA Attention + SwiGLU FFN)
        │
        ▼
    Final RMSNorm
        │
        ▼
    LM Head (线性投影到 vocab_size)  →  logits [batch, seq_len, vocab_size]

权重初始化:
    - Embedding: N(0, 0.02)
    - 线性层: N(0, 0.02)
    - 残差分支的输出投影: N(0, 0.02 / sqrt(2 * n_layers))
      (让深层网络的残差贡献逐渐减小，稳定训练)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from .config import ModelConfig
from .block import TransformerBlock, RMSNorm


class DecoderOnlyLM(nn.Module):
    """
    完整的 Decoder-Only 语言模型

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # TODO: 初始化以下组件
        #   - self.embed: nn.Embedding(vocab_size, d_model)
        #   - self.layers: nn.ModuleList of TransformerBlock
        #   - self.norm: RMSNorm (最终归一化)
        #   - self.lm_head: nn.Linear(d_model, vocab_size, bias=False)
        #   - 如果 tie_weights: self.lm_head.weight = self.embed.weight
        #   - 调用 self._init_weights() 初始化参数
        raise NotImplementedError(
            "TODO: 初始化完整 LLM\n"
            "参考: 模块 4 (Decoder-Only) 的模型组装章节\n"
            "参考实现: code/decoder_only/model.py 中的 DecoderOnlyLM"
        )

    def _init_weights(self, module: nn.Module):
        """
        权重初始化

        策略（参考 GPT-2 / Llama）:
        - Embedding: N(0, 0.02)
        - 线性层: N(0, 0.02)
        - 残差分支输出投影: N(0, 0.02 / sqrt(2 * n_layers))
        - RMSNorm gamma: 全 1（默认）

        Args:
            module: 要初始化的子模块
        """
        raise NotImplementedError(
            "TODO: 实现权重初始化\n"
            "参考: 模块 4 的权重初始化章节\n"
            "关键: 残差分支缩放 1/sqrt(2*n_layers)"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> dict:
        """
        模型前向传播

        Args:
            input_ids: token ID 序列 [batch, seq_len]
            labels: 目标 token ID（用于计算损失）[batch, seq_len]
                    通常 labels = input_ids 右移一位
            kv_caches: 每层的 KV Cache（推理时使用）

        Returns:
            dict 包含:
            - "logits": [batch, seq_len, vocab_size]
            - "loss": 标量（仅当 labels 不为 None 时）
            - "kv_caches": 更新后的 KV Cache 列表

        实现步骤:
            1. x = embed(input_ids)
            2. 创建因果掩码（上三角矩阵，对角线以上为 -inf）
            3. for layer in layers: x, kv = layer(x, mask, kv_cache)
            4. x = norm(x)
            5. logits = lm_head(x)
            6. 如果有 labels: loss = cross_entropy(logits, labels)
               注意: logits 形状需要 reshape 为 [batch*seq, vocab]
        """
        raise NotImplementedError(
            "TODO: 实现完整的前向传播和损失计算\n"
            "参考: code/decoder_only/model.py\n"
            "关键: 因果掩码的生成 + 交叉熵损失的计算"
        )

    def count_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
