"""
完整的 Decoder-only Transformer 模型

组装所有组件为完整的语言模型，包含文本生成功能。

架构: Token Embedding → N × TransformerBlock → RMSNorm → LM Head

参考:
- Vaswani et al. (2017). Attention Is All You Need.
- Radford et al. (2018). Improving Language Understanding by Generative Pre-Training. (GPT)
"""

import torch
import torch.nn as nn
from typing import Optional

from .block import TransformerBlock
from .normalization import RMSNorm


class Transformer(nn.Module):
    """
    完整的 Transformer 语言模型 (Decoder-only)

    特点:
    - Pre-Norm + RMSNorm
    - SwiGLU 激活
    - 因果注意力
    - 可选权重绑定 (Embedding 与 LM Head 共享)

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        n_heads: 注意力头数量
        n_layers: Transformer Block 层数
        d_ff: FFN 中间维度
        max_seq_len: 最大序列长度
        dropout: Dropout 概率
        tie_weights: 是否绑定 Embedding 和 LM Head 权重
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # N 层 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 最终归一化
        self.final_norm = RMSNorm(d_model)

        # 语言模型头: 将隐状态映射回词汇表
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定: Embedding 和 LM Head 共享权重
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len] 词元 ID
            mask: [batch, seq_len] 注意力掩码 (可选)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        # Token Embedding
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        # 通过所有 Transformer Block (因果注意力)
        for layer in self.layers:
            x = layer(x, mask=mask, is_causal=True)

        # 最终归一化
        x = self.final_norm(x)

        # 输出 logits
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ) -> torch.Tensor:
        """
        自回归文本生成

        Args:
            input_ids: [batch, seq_len] 输入词元 ID
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数 (越高越随机)
            top_k: Top-K 采样
            top_p: Nucleus 采样

        Returns:
            [batch, seq_len + max_new_tokens] 生成的序列
        """
        for _ in range(max_new_tokens):
            # 截断到最大序列长度
            idx_cond = input_ids[:, -self.max_seq_len:]

            # 前向传播，取最后一个位置的 logits
            logits = self.forward(idx_cond)[:, -1, :] / temperature

            # Top-K 采样: 只保留概率最高的 K 个 token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-P (Nucleus) 采样: 保留累积概率达到 P 的 token
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                mask = cumulative_probs > top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = False
                indices_to_remove = mask.scatter(1, sorted_indices, mask)
                logits[indices_to_remove] = float("-inf")

            # 采样下一个 token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters())

    def estimate_memory(self, batch_size: int = 1, seq_len: int = 512) -> dict:
        """
        估计显存使用 (GB)

        Returns:
            dict: 参数显存、激活显存、总显存
        """
        param_memory = self.count_parameters() * 4 / 1e9  # FP32
        activation_memory = (
            2 * batch_size * seq_len * self.d_model * self.n_layers * 2 / 1e9
        )
        return {
            "parameters_GB": round(param_memory, 4),
            "activations_GB": round(activation_memory, 4),
            "total_GB": round(param_memory + activation_memory, 4),
        }
