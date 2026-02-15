"""
文本生成策略

知识依赖:
- 模块 14（推理加速）: KV Cache 增量推理、采样策略
- 模块 4（Decoder-Only）: 自回归生成原理

参考实现:
- code/decoder_only/generation.py
- code/inference/kv_cache.py

自回归生成流程:
    1. Prefill 阶段: 一次性处理完整 prompt，初始化 KV Cache
    2. Decode 阶段: 逐 token 生成，每步只处理 1 个 token + 读取 KV Cache

采样策略:
- Greedy: 直接取 argmax（确定性，质量稳定但可能重复）
- Top-k: 只从概率最高的 k 个 token 中采样
- Top-p (Nucleus): 只从累积概率达到 p 的最小 token 集合中采样
- Temperature: 控制分布的"尖锐度"，T<1 更确定，T>1 更随机

    adjusted_logits = logits / temperature
    对 adjusted_logits 应用 top-k 或 top-p 过滤
    token = sample(softmax(adjusted_logits))
"""

import torch
import torch.nn.functional as F
from typing import Optional, List

from .config import ModelConfig


class TextGenerator:
    """
    文本生成器（封装自回归生成逻辑）

    使用 KV Cache 加速推理: Prefill 一次，Decode 逐 token。

    Args:
        model: 已加载的 DecoderOnlyLM 模型
        tokenizer: 分词器实例（需要有 encode/decode 方法）
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """
        从 prompt 生成文本

        Args:
            prompt: 输入文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数
            top_k: Top-k 采样的 k 值（0 表示不使用）
            top_p: Top-p (Nucleus) 采样的 p 值（1.0 表示不使用）
            do_sample: 是否采样（False 则使用 greedy decoding）

        Returns:
            生成的完整文本（prompt + generated）

        实现步骤:
            1. 用 tokenizer 将 prompt 编码为 input_ids
            2. Prefill: 一次前向传播处理完整 prompt，获得初始 KV Cache
            3. Decode 循环 (max_new_tokens 次):
               a. 用最后一个 token + KV Cache 做前向传播
               b. 取最后一个位置的 logits
               c. 应用 temperature 缩放
               d. 如果 do_sample: 应用 top_k/top_p 过滤后采样
                  否则: argmax
               e. 如果生成了 EOS token: break
               f. 更新 KV Cache
            4. 用 tokenizer 解码生成的 token 序列
        """
        raise NotImplementedError(
            "TODO: 实现自回归文本生成\n"
            "参考: 模块 14 (推理加速) 的 KV Cache 章节\n"
            "参考实现: code/decoder_only/generation.py"
        )

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        do_sample: bool,
    ) -> int:
        """
        从 logits 中采样下一个 token

        Args:
            logits: [vocab_size] 最后一个位置的 logits
            temperature: 温度
            top_k: top-k 值
            top_p: top-p 值
            do_sample: 是否采样

        Returns:
            采样得到的 token ID

        实现步骤:
            1. logits = logits / temperature
            2. 如果 top_k > 0: 将排名 top_k 之外的 logits 设为 -inf
            3. 如果 top_p < 1.0: 将累积概率超过 top_p 的 logits 设为 -inf
            4. probs = softmax(logits)
            5. 如果 do_sample: multinomial 采样
               否则: argmax
        """
        raise NotImplementedError(
            "TODO: 实现 Top-k/Top-p 采样\n"
            "参考: 模块 14 的采样策略章节\n"
            "参考实现: code/decoder_only/generation.py 中的采样函数"
        )
