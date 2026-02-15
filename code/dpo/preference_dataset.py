"""
偏好数据处理模块

本模块实现了偏好数据集的加载、预处理和批处理功能。
支持多种偏好数据格式：成对偏好（DPO/IPO/SimPO/ORPO）和二元反馈（KTO）。

偏好数据格式:
- 成对格式: (prompt, chosen_response, rejected_response)
- 二元格式: (prompt, response, is_desirable)

参考:
- Anthropic/hh-rlhf 数据集格式
- argilla/ultrafeedback-binarized-preferences 数据集格式
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


class PreferenceDataset(Dataset):
    """
    成对偏好数据集

    每个样本包含:
    - prompt: 用户的输入
    - chosen: 偏好的回答（更好的）
    - rejected: 非偏好的回答（更差的）

    数据会被分词后存储为 token ID，供模型直接使用。
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: Any = None,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        """
        初始化偏好数据集

        Args:
            data: 偏好数据列表，每个元素包含 "prompt", "chosen", "rejected"
            tokenizer: 分词器（若为 None，使用简单的字符级分词）
            max_length: 最大序列长度（prompt + response）
            max_prompt_length: prompt 的最大长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Returns:
            包含以下键的字典:
            - chosen_input_ids: 偏好回答的 token ID [seq_len]
            - chosen_attention_mask: 偏好回答的掩码 [seq_len]
            - chosen_labels: 偏好回答的标签 [seq_len]
            - rejected_input_ids: 非偏好回答的 token ID [seq_len]
            - rejected_attention_mask: 非偏好回答的掩码 [seq_len]
            - rejected_labels: 非偏好回答的标签 [seq_len]
        """
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # 分词
        chosen_encoded = self._encode(prompt, chosen)
        rejected_encoded = self._encode(prompt, rejected)

        return {
            "chosen_input_ids": chosen_encoded["input_ids"],
            "chosen_attention_mask": chosen_encoded["attention_mask"],
            "chosen_labels": chosen_encoded["labels"],
            "rejected_input_ids": rejected_encoded["input_ids"],
            "rejected_attention_mask": rejected_encoded["attention_mask"],
            "rejected_labels": rejected_encoded["labels"],
        }

    def _encode(
        self, prompt: str, response: str
    ) -> Dict[str, torch.Tensor]:
        """
        对 prompt + response 进行分词编码

        标签设置：
        - prompt 部分的标签设为 -100（不参与损失计算）
        - response 部分的标签为实际 token ID

        Args:
            prompt: 用户输入
            response: 模型回答

        Returns:
            编码后的张量字典
        """
        if self.tokenizer is not None:
            # 使用真实的分词器
            prompt_encoded = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_prompt_length,
            )
            response_encoded = self.tokenizer(
                response,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - len(prompt_encoded["input_ids"]),
            )

            input_ids = prompt_encoded["input_ids"] + response_encoded["input_ids"]
            attention_mask = [1] * len(input_ids)

            # 标签：prompt 部分为 -100，response 部分为实际 token
            labels = [-100] * len(prompt_encoded["input_ids"]) + response_encoded["input_ids"]
        else:
            # 简单的字符级编码（用于演示）
            prompt_ids = [ord(c) % 256 for c in prompt[:self.max_prompt_length]]
            response_ids = [ord(c) % 256 for c in response[:self.max_length - len(prompt_ids)]]

            input_ids = prompt_ids + response_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + response_ids

        # 填充到 max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class BinaryFeedbackDataset(Dataset):
    """
    二元反馈数据集（用于 KTO）

    每个样本包含:
    - prompt: 用户的输入
    - response: 模型的回答
    - is_desirable: 布尔值，True 表示好回答，False 表示坏回答

    与 PreferenceDataset 不同，这里不需要成对的 (chosen, rejected)。
    每个样本独立标注为"好"或"坏"。
    """

    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: Any = None,
        max_length: int = 512,
        max_prompt_length: int = 256,
    ):
        """
        初始化二元反馈数据集

        Args:
            data: 数据列表，每个元素包含 "prompt", "response", "is_desirable"
            tokenizer: 分词器
            max_length: 最大序列长度
            max_prompt_length: prompt 最大长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本

        Returns:
            包含 input_ids, attention_mask, labels, is_desirable
        """
        item = self.data[idx]
        prompt = item["prompt"]
        response = item["response"]
        is_desirable = item["is_desirable"]

        encoded = self._encode(prompt, response)
        encoded["is_desirable"] = torch.tensor(is_desirable, dtype=torch.bool)

        return encoded

    def _encode(
        self, prompt: str, response: str
    ) -> Dict[str, torch.Tensor]:
        """编码 prompt + response"""
        if self.tokenizer is not None:
            prompt_encoded = self.tokenizer(
                prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_prompt_length,
            )
            response_encoded = self.tokenizer(
                response,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_length - len(prompt_encoded["input_ids"]),
            )

            input_ids = prompt_encoded["input_ids"] + response_encoded["input_ids"]
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_encoded["input_ids"]) + response_encoded["input_ids"]
        else:
            prompt_ids = [ord(c) % 256 for c in prompt[:self.max_prompt_length]]
            response_ids = [ord(c) % 256 for c in response[:self.max_length - len(prompt_ids)]]

            input_ids = prompt_ids + response_ids
            attention_mask = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + response_ids

        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [0] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def create_sample_preference_data(
    num_samples: int = 100,
) -> List[Dict[str, str]]:
    """
    创建示例偏好数据（用于演示和测试）

    生成简单的问答对，模拟偏好数据的格式。

    Args:
        num_samples: 样本数量

    Returns:
        偏好数据列表
    """
    data = []
    prompts = [
        "What is machine learning?",
        "Explain deep learning.",
        "How does a neural network work?",
        "What is NLP?",
        "Describe transformer architecture.",
    ]

    chosen_templates = [
        "Machine learning is a subset of AI that enables systems to learn from data.",
        "Deep learning uses multi-layer neural networks to learn representations.",
        "A neural network consists of interconnected layers that process information.",
        "NLP is the field of AI focused on understanding human language.",
        "The transformer uses self-attention to process sequences in parallel.",
    ]

    rejected_templates = [
        "Machine learning is just computers.",
        "Deep learning is very complex.",
        "Neural networks are like brains.",
        "NLP means natural language processing.",
        "Transformer is a model.",
    ]

    for i in range(num_samples):
        idx = i % len(prompts)
        data.append({
            "prompt": prompts[idx],
            "chosen": chosen_templates[idx],
            "rejected": rejected_templates[idx],
        })

    return data


def create_sample_binary_data(
    num_samples: int = 100,
) -> List[Dict[str, Any]]:
    """
    创建示例二元反馈数据（用于 KTO 的演示和测试）

    Args:
        num_samples: 样本数量

    Returns:
        二元反馈数据列表
    """
    import random
    random.seed(42)

    data = []
    prompts = [
        "What is AI?",
        "Explain backpropagation.",
        "What is gradient descent?",
    ]

    good_responses = [
        "AI is the simulation of human intelligence by machines.",
        "Backpropagation computes gradients by applying the chain rule.",
        "Gradient descent iteratively updates parameters to minimize loss.",
    ]

    bad_responses = [
        "AI is robots.",
        "Backpropagation is just math.",
        "Gradient descent goes down.",
    ]

    for i in range(num_samples):
        idx = i % len(prompts)
        is_desirable = random.random() > 0.4  # 约 60% 好回答

        if is_desirable:
            response = good_responses[idx]
        else:
            response = bad_responses[idx]

        data.append({
            "prompt": prompts[idx],
            "response": response,
            "is_desirable": is_desirable,
        })

    return data


def preference_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    成对偏好数据的整理函数（用于 DataLoader）

    将一个批次的样本整理为统一的张量字典。

    Args:
        batch: 样本列表

    Returns:
        批次字典
    """
    result = {}
    for key in batch[0].keys():
        result[key] = torch.stack([item[key] for item in batch])
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("偏好数据处理模块演示")
    print("=" * 60)

    # 场景 1: 成对偏好数据集
    print("\n--- 场景 1: 成对偏好数据集（用于 DPO/IPO/SimPO）---")
    pref_data = create_sample_preference_data(num_samples=20)
    print(f"数据量: {len(pref_data)}")
    print(f"示例数据:")
    print(f"  Prompt: {pref_data[0]['prompt']}")
    print(f"  Chosen: {pref_data[0]['chosen']}")
    print(f"  Rejected: {pref_data[0]['rejected']}")

    dataset = PreferenceDataset(pref_data, max_length=128)
    print(f"\n数据集大小: {len(dataset)}")

    # 获取一个样本
    sample = dataset[0]
    print(f"样本键: {list(sample.keys())}")
    print(f"chosen_input_ids 形状: {sample['chosen_input_ids'].shape}")
    print(f"chosen_attention_mask 形状: {sample['chosen_attention_mask'].shape}")

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=preference_collate_fn,
    )

    batch = next(iter(dataloader))
    print(f"\n批次:")
    for key, tensor in batch.items():
        print(f"  {key}: {tensor.shape}")

    # 场景 2: 二元反馈数据集（用于 KTO）
    print("\n--- 场景 2: 二元反馈数据集（用于 KTO）---")
    binary_data = create_sample_binary_data(num_samples=20)
    print(f"数据量: {len(binary_data)}")

    num_good = sum(1 for d in binary_data if d["is_desirable"])
    num_bad = len(binary_data) - num_good
    print(f"好回答: {num_good}, 坏回答: {num_bad}")

    print(f"\n示例（好回答）:")
    good_sample = next(d for d in binary_data if d["is_desirable"])
    print(f"  Prompt: {good_sample['prompt']}")
    print(f"  Response: {good_sample['response']}")
    print(f"  Is Desirable: {good_sample['is_desirable']}")

    print(f"\n示例（坏回答）:")
    bad_sample = next(d for d in binary_data if not d["is_desirable"])
    print(f"  Prompt: {bad_sample['prompt']}")
    print(f"  Response: {bad_sample['response']}")
    print(f"  Is Desirable: {bad_sample['is_desirable']}")

    binary_dataset = BinaryFeedbackDataset(binary_data, max_length=128)
    sample = binary_dataset[0]
    print(f"\n样本键: {list(sample.keys())}")
    print(f"is_desirable: {sample['is_desirable'].item()}")

    # 场景 3: 数据统计
    print("\n--- 场景 3: 数据集统计 ---")
    lengths = []
    for i in range(len(dataset)):
        s = dataset[i]
        chosen_len = s["chosen_attention_mask"].sum().item()
        rejected_len = s["rejected_attention_mask"].sum().item()
        lengths.append((chosen_len, rejected_len))

    chosen_lengths = [l[0] for l in lengths]
    rejected_lengths = [l[1] for l in lengths]
    print(f"偏好回答平均长度: {sum(chosen_lengths)/len(chosen_lengths):.1f}")
    print(f"非偏好回答平均长度: {sum(rejected_lengths)/len(rejected_lengths):.1f}")

    print("\n演示完成。")
