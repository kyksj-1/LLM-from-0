"""
SFT 指令微调训练器

知识依赖:
- 模块 10（SFT / 指令微调）: 对话模板、损失掩码、LoRA

参考实现:
- code/sft/sft_trainer.py
- code/sft/lora.py

SFT 的关键区别（相比预训练）:
1. 数据格式: (instruction, response) 对，而非纯文本
2. 损失掩码: 只对 response 部分计算损失
3. 学习率: 比预训练小 1-2 个数量级（避免灾难性遗忘）
4. 可选 LoRA: 参数高效微调，只训练低秩增量矩阵

LoRA 数学原理:
    W' = W + ΔW = W + BA
    其中 B ∈ R^{d×r}, A ∈ R^{r×d}, r << d
    参数量: 2 × d × r（远小于 d × d）
    训练时冻结 W，只训练 A, B

SFT 训练流程:
    1. 加载预训练 checkpoint
    2. (可选) 应用 LoRA
    3. 准备指令数据（应用对话模板 + 损失掩码）
    4. 以较小学习率训练
    5. (可选) 合并 LoRA 权重
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class SFTTrainer:
    """
    SFT 指令微调训练器

    继承预训练训练器的基本结构，增加:
    - 对话模板格式化
    - 损失掩码（只对 response 计算损失）
    - 可选的 LoRA 支持

    Args:
        model: 预训练模型（从 checkpoint 加载）
        train_data: SFT 训练数据
        tokenizer: 分词器
        train_config: 训练配置
        use_lora: 是否使用 LoRA
        lora_rank: LoRA 秩
    """

    def __init__(
        self,
        model: nn.Module,
        train_data,
        tokenizer,
        train_config,
        use_lora: bool = False,
        lora_rank: int = 16,
    ):
        # TODO: 初始化 SFT 训练器
        # 如果 use_lora:
        #   1. 冻结所有原始参数: model.requires_grad_(False)
        #   2. 对注意力层的 W_q, W_v 注入 LoRA 适配器
        #   3. 只有 LoRA 参数可训练
        # 初始化优化器（使用 sft_learning_rate）
        raise NotImplementedError(
            "TODO: 初始化 SFT 训练器\n"
            "参考: 模块 10 (SFT) 的 LoRA 章节\n"
            "参考实现: code/sft/sft_trainer.py, code/sft/lora.py"
        )

    def prepare_sft_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        准备 SFT 训练 batch

        将 (instruction, response) 格式化为对话模板，
        并生成损失掩码。

        Args:
            batch: {"instruction": List[str], "response": List[str]}

        Returns:
            {
                "input_ids": [batch, seq_len],
                "labels": [batch, seq_len],  # instruction 部分为 -100
                "attention_mask": [batch, seq_len],
            }

        实现步骤:
            1. 用对话模板拼接: <bos><|user|>\n{instruction}<|end_turn|>\n<|assistant|>\n
            2. 记录 response 开始位置
            3. 拼接 response: {response}<|end_turn|>
            4. 分词整个序列
            5. labels 中 response 之前的部分设为 -100
        """
        raise NotImplementedError(
            "TODO: 实现 SFT 数据格式化\n"
            "关键: 找到 response 的起始位置，设置损失掩码\n"
            "参考: code/sft/chat_template.py"
        )

    def sft_training_step(self, batch: Dict) -> float:
        """
        SFT 单步训练

        与预训练的区别: 使用 prepare_sft_batch 处理数据

        Returns:
            损失值
        """
        raise NotImplementedError(
            "TODO: 实现 SFT 训练步\n"
            "参考: code/sft/sft_trainer.py"
        )
