"""
decoder_only - Decoder-Only 语言模型实现

本包实现了完整的 Decoder-Only Transformer 语言模型,
支持 GPT / Llama / Gemma 三种架构风格。

模块:
- config: 模型配置 (ModelConfig + 预置配置)
- model: 完整模型实现 (DecoderOnlyModel)
- generation: 文本生成策略 (Greedy / Top-K / Top-P / Beam Search)
- tokenizer_wrapper: 分词器封装
- train_simple: 简单训练脚本
- utils: 工具函数 (参数统计、FLOPs 估算、KV Cache 分析)
"""

from config import ModelConfig
from model import DecoderOnlyModel

__all__ = ["ModelConfig", "DecoderOnlyModel"]
