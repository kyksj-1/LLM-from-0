"""
数据管线

知识依赖:
- 模块 7（数据工程）: 数据清洗、去重、质量过滤

参考实现:
- code/data_engineering/pipeline.py
- code/data_engineering/quality_filter.py

数据准备流程:
    1. 下载原始数据（推荐数据集见下方）
    2. 文本清洗: 去除 HTML 标签、重复行、乱码
    3. 质量过滤: 按语言、长度、困惑度过滤低质量文本
    4. 去重: MinHash + LSH 近似去重
    5. 分词: 用训练好的分词器将文本转为 token ID
    6. 打包: 将 token 序列拼接并切分为固定长度的块
    7. 保存为二进制格式（numpy memmap）

推荐数据集:
- 预训练:
  - TinyStories (Microsoft): 小规模故事数据，适合小模型
  - SlimPajama (CerebrasAI): RedPajama 的去重版本，627B tokens
  - 中文: WuDaoCorpora (智源), SkyPile (DeepSeek)
- SFT:
  - Alpaca (Stanford): 52K 指令数据
  - ShareGPT: 用户与 ChatGPT 的对话数据
  - MOSS-SFT: 中文指令数据
- DPO:
  - HH-RLHF (Anthropic): 人类偏好数据
  - UltraFeedback: 大规模偏好数据
"""

from pathlib import Path
from typing import List, Optional
import numpy as np


def download_and_prepare(
    dataset_name: str,
    output_dir: str,
    max_samples: Optional[int] = None,
):
    """
    下载并预处理训练数据

    Args:
        dataset_name: 数据集名称（如 "tinystories", "slimpajama"）
        output_dir: 输出目录
        max_samples: 最大样本数（用于调试）

    产出:
        {output_dir}/train.txt — 清洗后的训练文本
        {output_dir}/val.txt — 验证集文本

    实现提示:
        可以使用 HuggingFace datasets 库下载:
        from datasets import load_dataset
        ds = load_dataset("roneneldan/TinyStories")
    """
    raise NotImplementedError(
        "TODO: 实现数据下载和预处理\n"
        "参考: 模块 7 (数据工程) 的数据管线章节\n"
        "提示: 使用 HuggingFace datasets 库下载数据"
    )


def tokenize_and_save(
    input_path: str,
    output_path: str,
    tokenizer_path: str,
    seq_len: int = 2048,
):
    """
    将文本文件分词并保存为二进制格式

    Args:
        input_path: 输入文本文件路径
        output_path: 输出 .bin 文件路径
        tokenizer_path: 分词器模型路径
        seq_len: 序列长度（用于打包）

    产出:
        {output_path} — numpy memmap 格式的 token ID 序列

    实现步骤:
        1. 加载分词器
        2. 逐行读取文本并编码为 token ID
        3. 在文档之间插入 EOS token
        4. 拼接所有 token 并保存为 numpy 数组
    """
    raise NotImplementedError(
        "TODO: 实现分词和二进制保存\n"
        "参考: 模块 7 的数据序列化章节\n"
        "提示: 使用 np.memmap 保存大规模 token 数据"
    )


def create_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    seq_len: int,
    num_workers: int = 4,
):
    """
    创建训练和验证 DataLoader

    Args:
        train_path: 训练数据路径
        val_path: 验证数据路径
        batch_size: 批大小
        seq_len: 序列长度
        num_workers: 数据加载线程数

    Returns:
        (train_loader, val_loader)

    实现提示:
        from torch.utils.data import DataLoader
        from .dataset import PretrainDataset
    """
    raise NotImplementedError(
        "TODO: 创建 DataLoader\n"
        "提示: 使用 PretrainDataset + DataLoader\n"
        "注意: 预训练数据不需要 shuffle（已经随机打包）"
    )
