"""
指令微调数据集处理

本模块实现了指令微调数据的加载、格式化和 DataLoader 构建。
支持多种指令格式（Alpaca、ShareGPT），自动处理 Prompt Masking。

核心概念:
- 指令数据三元组: (instruction, input, output)
- Prompt Masking: 只在回答部分计算损失，指令部分 label 设为 -100
- 对话模板: 将原始数据转换为模型可接受的格式

参考:
- Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- Self-Instruct: Wang et al. (2023)
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class InstructSample:
    """
    单条指令微调样本

    Attributes:
        instruction: 任务描述/指令
        input_text: 可选的附加输入
        output_text: 期望的回答
        system_prompt: 可选的系统提示
    """
    instruction: str
    input_text: str = ""
    output_text: str = ""
    system_prompt: str = ""


class InstructDataset(Dataset):
    """
    指令微调数据集

    支持 Alpaca 格式的 JSON 数据:
    [
        {
            "instruction": "...",
            "input": "...",
            "output": "..."
        },
        ...
    ]

    自动处理 Prompt Masking：将指令部分的 label 设为 -100，
    使得损失函数只在回答部分计算。

    Args:
        data_path: JSON 数据文件路径
        tokenizer: 分词器（需支持 encode 方法）
        max_length: 最大序列长度
        template_fn: 将 InstructSample 转换为 (prompt, response) 字符串的函数
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        tokenizer: Any = None,
        max_length: int = 2048,
        template_fn: Optional[callable] = None,
        samples: Optional[List[InstructSample]] = None,
    ):
        self.max_length = max_length
        self.tokenizer = tokenizer

        # 默认模板函数
        self.template_fn = template_fn or self._default_template

        # 加载数据
        if samples is not None:
            self.samples = samples
        elif data_path is not None:
            self.samples = self._load_data(data_path)
        else:
            self.samples = []

    def _load_data(self, data_path: str) -> List[InstructSample]:
        """
        从 JSON 文件加载数据

        Args:
            data_path: JSON 文件路径

        Returns:
            InstructSample 列表
        """
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        samples = []
        for item in raw_data:
            sample = InstructSample(
                instruction=item.get("instruction", ""),
                input_text=item.get("input", ""),
                output_text=item.get("output", ""),
                system_prompt=item.get("system", ""),
            )
            samples.append(sample)

        return samples

    @staticmethod
    def _default_template(sample: InstructSample) -> Tuple[str, str]:
        """
        默认的 Alpaca 格式模板

        Args:
            sample: 指令样本

        Returns:
            (prompt, response) 字符串元组
        """
        if sample.input_text:
            prompt = (
                f"### Instruction:\n{sample.instruction}\n\n"
                f"### Input:\n{sample.input_text}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"### Instruction:\n{sample.instruction}\n\n"
                f"### Response:\n"
            )

        response = sample.output_text
        return prompt, response

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单条数据

        Returns:
            包含 input_ids, attention_mask, labels 的字典
            labels 中指令部分设为 -100（不计算损失）
        """
        sample = self.samples[idx]
        prompt, response = self.template_fn(sample)

        return self._encode_with_mask(prompt, response)

    def _encode_with_mask(
        self, prompt: str, response: str
    ) -> Dict[str, torch.Tensor]:
        """
        编码并生成带 Prompt Masking 的 labels

        Prompt Masking 的核心:
        - 完整序列 = prompt + response
        - labels 中 prompt 部分设为 -100
        - 只有 response 部分参与损失计算

        Args:
            prompt: 指令文本
            response: 回答文本

        Returns:
            {input_ids, attention_mask, labels}
        """
        if self.tokenizer is None:
            # 无分词器时返回原始文本（用于调试）
            return {
                "prompt": prompt,
                "response": response,
                "full_text": prompt + response,
            }

        # 分别编码 prompt 和 response
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        response_ids = self.tokenizer.encode(response, add_special_tokens=False)

        # 拼接
        input_ids = prompt_ids + response_ids

        # 截断到最大长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]

        # 构建 labels: prompt 部分设为 -100
        prompt_len = len(prompt_ids)
        labels = [-100] * prompt_len + response_ids[: self.max_length - prompt_len]

        # 确保 labels 和 input_ids 长度一致
        labels = labels[: len(input_ids)]

        # 构建 attention_mask
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(
    batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0
) -> Dict[str, torch.Tensor]:
    """
    DataLoader 的 collate 函数，处理变长序列的 padding

    Args:
        batch: 一个 batch 的样本列表
        pad_token_id: padding token 的 ID

    Returns:
        批量化的 {input_ids, attention_mask, labels}
    """
    # 找到 batch 内最大长度
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        # 右侧 padding
        input_ids_list.append(
            torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        attention_mask_list.append(
            torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
        )
        labels_list.append(
            torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
        )

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels": torch.stack(labels_list),
    }


def create_dataloader(
    dataset: InstructDataset,
    batch_size: int = 8,
    shuffle: bool = True,
    pad_token_id: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """
    创建 DataLoader

    Args:
        dataset: 指令微调数据集
        batch_size: 批量大小
        shuffle: 是否打乱
        pad_token_id: padding token ID
        num_workers: 数据加载工作线程数

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    # === 演示: 不使用分词器的基本功能 ===
    print("=" * 60)
    print("指令微调数据集演示")
    print("=" * 60)

    # 创建示例数据
    samples = [
        InstructSample(
            instruction="什么是机器学习？",
            output_text="机器学习是人工智能的一个分支，它让计算机能够从数据中学习规律，而无需显式编程。",
        ),
        InstructSample(
            instruction="将以下文本翻译成英文",
            input_text="今天天气很好",
            output_text="The weather is nice today.",
        ),
        InstructSample(
            instruction="写一个 Python 函数，计算两个数的最大公约数",
            output_text="def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        ),
    ]

    # 创建数据集（不使用分词器）
    dataset = InstructDataset(samples=samples)

    print(f"\n数据集大小: {len(dataset)}")
    print(f"\n--- 样本 0 ---")
    item = dataset[0]
    print(f"Prompt:\n{item['prompt']}")
    print(f"Response:\n{item['response']}")

    print(f"\n--- 样本 1 (带 input) ---")
    item = dataset[1]
    print(f"Prompt:\n{item['prompt']}")
    print(f"Response:\n{item['response']}")

    # === 演示: Prompt Masking 原理 ===
    print("\n" + "=" * 60)
    print("Prompt Masking 原理演示")
    print("=" * 60)

    # 模拟分词结果
    prompt_tokens = [101, 2023, 3456, 7890, 102]  # 5 个 token
    response_tokens = [2054, 1010, 1037, 102]      # 4 个 token

    full_input = prompt_tokens + response_tokens
    labels = [-100] * len(prompt_tokens) + response_tokens

    print(f"\n完整 input_ids:  {full_input}")
    print(f"labels:          {labels}")
    print(f"prompt 长度:     {len(prompt_tokens)}")
    print(f"response 长度:   {len(response_tokens)}")
    print(f"\n-100 表示不计算损失的位置（指令部分）")
    print(f"只有 response 部分 {response_tokens} 参与损失计算")
