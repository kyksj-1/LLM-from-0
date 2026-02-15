"""
数据集类

知识依赖:
- 模块 7（数据工程）: 数据预处理、打包策略
- 模块 10（SFT）: 指令数据格式、对话模板
- 模块 12（DPO）: 偏好数据格式

参考实现:
- code/data_engineering/pipeline.py
- code/sft/dataset.py
- code/dpo/preference_dataset.py

三种数据集:

1. PretrainDataset: 预训练数据
   - 输入: tokenized 的长文本
   - 打包策略: 将多个文档拼接为 max_seq_len 长度的块
   - 输出: input_ids [seq_len], labels [seq_len] (labels = input_ids 右移一位)

2. SFTDataset: 指令微调数据
   - 输入: (instruction, response) 对
   - 对话模板: <|user|>instruction<|end_turn|><|assistant|>response<|end_turn|>
   - 损失掩码: 只对 response 部分计算损失（mask instruction tokens）

3. DPODataset: 偏好优化数据
   - 输入: (prompt, chosen_response, rejected_response) 三元组
   - 输出: chosen 和 rejected 的 token 序列
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Optional
from pathlib import Path


class PretrainDataset(Dataset):
    """
    预训练数据集

    将 tokenized 的文本打包成固定长度的序列。
    多个文档之间用 EOS token 分隔。

    Args:
        data_path: tokenized 数据文件路径（.bin 或 .npy）
        seq_len: 序列长度
    """

    def __init__(self, data_path: str, seq_len: int = 2048):
        # TODO: 加载预处理好的 token 数据（mmap 方式以支持大文件）
        # 提示:
        #   self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        #   self.seq_len = seq_len
        #   self.n_samples = len(self.data) // seq_len
        raise NotImplementedError(
            "TODO: 加载预训练数据\n"
            "参考: 模块 7 (数据工程) 的数据预处理章节\n"
            "提示: 使用 numpy.memmap 支持大文件的内存映射读取"
        )

    def __len__(self) -> int:
        raise NotImplementedError("TODO: 返回样本数量")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个训练样本

        Returns:
            {"input_ids": [seq_len], "labels": [seq_len]}
            labels = input_ids 右移一位（即 labels[i] = input_ids[i+1]）
            第一个位置的 label 被忽略（设为 -100）
        """
        raise NotImplementedError(
            "TODO: 实现预训练数据的取样逻辑\n"
            "参考: 模块 7 的数据打包策略"
        )


class SFTDataset(Dataset):
    """
    SFT 指令微调数据集

    将 (instruction, response) 对按照对话模板格式化，
    并生成损失掩码（只对 response 部分计算损失）。

    对话模板示例:
        <bos><|user|>\n{instruction}<|end_turn|>\n<|assistant|>\n{response}<|end_turn|>

    Args:
        data: 指令数据列表，每条格式 {"instruction": str, "response": str}
        tokenizer: 分词器实例
        max_len: 最大序列长度
    """

    def __init__(self, data: List[dict], tokenizer, max_len: int = 2048):
        # TODO: 保存数据和分词器引用，预处理对话模板
        raise NotImplementedError(
            "TODO: 初始化 SFT 数据集\n"
            "参考: 模块 10 (SFT) 的对话模板章节\n"
            "参考实现: code/sft/dataset.py"
        )

    def __len__(self) -> int:
        raise NotImplementedError("TODO: 返回样本数量")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个 SFT 训练样本

        Returns:
            {
                "input_ids": [seq_len],
                "labels": [seq_len],
                "attention_mask": [seq_len]
            }
            labels 中 instruction 部分设为 -100（不计算损失），
            只有 response 部分保留真实 token ID。
        """
        raise NotImplementedError(
            "TODO: 实现 SFT 数据的格式化和损失掩码\n"
            "关键: instruction 部分的 labels 设为 -100\n"
            "参考: code/sft/dataset.py"
        )


class DPODataset(Dataset):
    """
    DPO 偏好数据集

    每条数据包含: prompt + chosen response + rejected response

    Args:
        data: 偏好数据列表，每条格式:
              {"prompt": str, "chosen": str, "rejected": str}
        tokenizer: 分词器实例
        max_len: 最大序列长度
    """

    def __init__(self, data: List[dict], tokenizer, max_len: int = 2048):
        raise NotImplementedError(
            "TODO: 初始化 DPO 数据集\n"
            "参考: 模块 12 (DPO) 的偏好数据章节\n"
            "参考实现: code/dpo/preference_dataset.py"
        )

    def __len__(self) -> int:
        raise NotImplementedError("TODO: 返回样本数量")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个 DPO 训练样本

        Returns:
            {
                "chosen_input_ids": [seq_len],
                "chosen_labels": [seq_len],
                "rejected_input_ids": [seq_len],
                "rejected_labels": [seq_len],
            }
        """
        raise NotImplementedError(
            "TODO: 实现偏好数据的编码\n"
            "参考: code/dpo/preference_dataset.py"
        )
