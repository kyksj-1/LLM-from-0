"""
推理引擎

知识依赖:
- 模块 14（推理加速）: KV Cache、连续批处理、Prefill/Decode 分离

参考实现:
- code/inference/kv_cache.py
- code/inference/paged_attention.py

推理优化策略:
1. KV Cache: 缓存历史 KV 避免重复计算（最基础、最重要）
2. 连续批处理 (Continuous Batching): 动态管理不同长度的请求
3. Flash Attention: 减少 HBM 访问（需要安装 flash-attn 库）
4. torch.compile: JIT 编译加速
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict


class InferenceEngine:
    """
    推理引擎

    封装模型推理的优化逻辑: KV Cache 管理、批处理、生成控制。

    Args:
        model: 已加载的语言模型
        tokenizer: 分词器
        device: 推理设备
    """

    def __init__(self, model: nn.Module, tokenizer, device: str = "cuda"):
        # TODO: 初始化推理引擎
        # 1. model.eval()
        # 2. 可选: model = torch.compile(model)  # PyTorch 2.0+ JIT 编译
        # 3. 初始化 KV Cache 管理器
        raise NotImplementedError(
            "TODO: 初始化推理引擎\n"
            "参考: 模块 14 (推理加速) 的推理引擎章节"
        )

    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        批量文本生成

        Args:
            prompts: 输入文本列表
            max_new_tokens: 每个 prompt 最多生成的 token 数
            temperature: 温度
            top_p: nucleus 采样阈值

        Returns:
            生成的文本列表

        实现步骤:
            1. 将所有 prompt 编码并 padding 到相同长度
            2. Prefill: 批量处理所有 prompt
            3. Decode: 逐 token 生成（使用 KV Cache）
            4. 根据各 prompt 的 EOS 位置决定是否停止
            5. 解码并返回结果
        """
        raise NotImplementedError(
            "TODO: 实现批量推理\n"
            "参考: code/inference/kv_cache.py\n"
            "提示: 注意 padding 和 attention_mask 的处理"
        )

    def estimate_memory(self, batch_size: int, seq_len: int) -> Dict[str, float]:
        """
        估算推理显存需求

        Returns:
            {"model_gb": float, "kv_cache_gb": float, "total_gb": float}
        """
        raise NotImplementedError(
            "TODO: 实现显存估算\n"
            "参考: 模块 14 的显存分析章节\n"
            "参考实现: code/inference/benchmark.py 中的 estimate_memory_usage()"
        )
