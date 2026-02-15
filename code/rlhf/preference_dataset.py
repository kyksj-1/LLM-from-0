"""
偏好数据处理模块

实现 RLHF 训练所需的偏好数据集（Preference Dataset）处理逻辑。

偏好数据是 RLHF 的基石：
- 每条数据包含一个 prompt 和两个 response（chosen 和 rejected）
- 人类标注者（或 AI）选择更好的 response 作为 chosen
- 奖励模型通过学习这些偏好来评估 response 质量

支持的数据格式：
1. 标准三元组：(prompt, chosen, rejected)
2. 排序格式：(prompt, [response_1, response_2, ...], ranking)
3. Anthropic HH-RLHF 格式：多轮对话的偏好数据
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
import json
import random


class PreferenceDataset(Dataset):
    """
    偏好数据集

    处理 (prompt, chosen_response, rejected_response) 三元组数据。

    Bradley-Terry 模型假设:
        P(chosen > rejected | prompt) = sigma(r(prompt, chosen) - r(prompt, rejected))

    其中 r 是奖励模型需要学习的奖励函数。

    Attributes:
        data: 偏好数据列表
        max_length: 序列最大长度
        tokenizer: 分词器（如果有）
    """

    def __init__(
        self,
        data: List[Dict[str, str]],
        max_length: int = 512,
        tokenizer=None
    ):
        """
        Args:
            data: 偏好数据列表，每条包含 'prompt', 'chosen', 'rejected' 字段
            max_length: token 序列最大长度
            tokenizer: 分词器实例（如果为 None，使用简单字符编码）
        """
        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        返回一条偏好数据的编码结果。

        Returns:
            字典，包含:
            - chosen_input_ids: chosen response 的 token id
            - chosen_attention_mask: chosen 的注意力掩码
            - rejected_input_ids: rejected response 的 token id
            - rejected_attention_mask: rejected 的注意力掩码
        """
        item = self.data[idx]
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # 拼接 prompt 和 response
        chosen_text = prompt + chosen
        rejected_text = prompt + rejected

        if self.tokenizer is not None:
            # 使用真实分词器
            chosen_encoding = self.tokenizer(
                chosen_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            rejected_encoding = self.tokenizer(
                rejected_text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "chosen_input_ids": chosen_encoding["input_ids"].squeeze(0),
                "chosen_attention_mask": chosen_encoding["attention_mask"].squeeze(0),
                "rejected_input_ids": rejected_encoding["input_ids"].squeeze(0),
                "rejected_attention_mask": rejected_encoding["attention_mask"].squeeze(0),
            }
        else:
            # 简单字符编码（教学演示用）
            chosen_ids = self._simple_encode(chosen_text)
            rejected_ids = self._simple_encode(rejected_text)
            return {
                "chosen_input_ids": chosen_ids,
                "chosen_attention_mask": (chosen_ids != 0).long(),
                "rejected_input_ids": rejected_ids,
                "rejected_attention_mask": (rejected_ids != 0).long(),
            }

    def _simple_encode(self, text: str) -> torch.Tensor:
        """
        简单字符编码（教学演示用，不依赖外部分词器）。

        将字符转换为 ASCII 码，填充或截断到 max_length。
        """
        ids = [ord(c) % 256 for c in text[:self.max_length]]
        # 填充到 max_length
        padding_length = self.max_length - len(ids)
        ids = ids + [0] * padding_length
        return torch.tensor(ids, dtype=torch.long)


class RankingPreferenceDataset(Dataset):
    """
    排序偏好数据集

    处理包含多个 response 排名的数据，转换为成对比较。

    给定一个 prompt 和 K 个 response 的排名 r_1 > r_2 > ... > r_K，
    可以生成 C(K, 2) = K*(K-1)/2 个偏好对。

    这种做法可以充分利用排名信息，提高数据效率。
    Anthropic 的 HH-RLHF 数据集就包含了这种排名格式的数据。
    """

    def __init__(
        self,
        data: List[Dict],
        max_length: int = 512,
        tokenizer=None,
        max_pairs_per_prompt: int = 3
    ):
        """
        Args:
            data: 排序数据列表，每条包含 'prompt' 和按质量降序排列的 'responses'
            max_length: token 序列最大长度
            tokenizer: 分词器实例
            max_pairs_per_prompt: 每个 prompt 最多生成的偏好对数
        """
        self.max_length = max_length
        self.tokenizer = tokenizer
        # 将排名数据展开为偏好对
        self.pairs = self._expand_rankings(data, max_pairs_per_prompt)

    def _expand_rankings(
        self,
        data: List[Dict],
        max_pairs: int
    ) -> List[Dict[str, str]]:
        """
        将排名数据展开为偏好对。

        排名 [r1, r2, r3]（r1最好）展开为:
        (r1, r2), (r1, r3), (r2, r3)
        """
        pairs = []
        for item in data:
            prompt = item["prompt"]
            responses = item["responses"]  # 按质量降序排列
            item_pairs = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    item_pairs.append({
                        "prompt": prompt,
                        "chosen": responses[i],
                        "rejected": responses[j],
                    })
            # 限制每个 prompt 的偏好对数
            if len(item_pairs) > max_pairs:
                item_pairs = random.sample(item_pairs, max_pairs)
            pairs.extend(item_pairs)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """返回编码后的偏好对。"""
        pair = self.pairs[idx]
        chosen_text = pair["prompt"] + pair["chosen"]
        rejected_text = pair["prompt"] + pair["rejected"]

        # 简单字符编码
        chosen_ids = self._simple_encode(chosen_text)
        rejected_ids = self._simple_encode(rejected_text)

        return {
            "chosen_input_ids": chosen_ids,
            "chosen_attention_mask": (chosen_ids != 0).long(),
            "rejected_input_ids": rejected_ids,
            "rejected_attention_mask": (rejected_ids != 0).long(),
        }

    def _simple_encode(self, text: str) -> torch.Tensor:
        """简单字符编码。"""
        ids = [ord(c) % 256 for c in text[:self.max_length]]
        padding_length = self.max_length - len(ids)
        ids = ids + [0] * padding_length
        return torch.tensor(ids, dtype=torch.long)


def create_preference_collator(pad_token_id: int = 0):
    """
    创建偏好数据的 collate 函数。

    用于 DataLoader 的 collate_fn 参数，将多条偏好数据合并为 batch。

    Args:
        pad_token_id: 填充 token 的 id

    Returns:
        collate 函数
    """
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        result = {}
        for key in batch[0].keys():
            result[key] = torch.stack([item[key] for item in batch])
        return result
    return collate_fn


def generate_synthetic_preferences(
    num_samples: int = 100,
    prompt_templates: Optional[List[str]] = None,
    seed: int = 42
) -> List[Dict[str, str]]:
    """
    生成合成偏好数据（教学演示用）。

    在真实 RLHF 中，偏好数据来自人类标注：
    1. 给标注者一个 prompt
    2. 展示两个（或多个）模型生成的 response
    3. 标注者选择更好的 response

    此函数生成简单的合成数据用于代码测试。

    Args:
        num_samples: 生成的样本数
        prompt_templates: 提示模板列表
        seed: 随机种子

    Returns:
        偏好数据列表
    """
    random.seed(seed)

    if prompt_templates is None:
        prompt_templates = [
            "请解释什么是{}。",
            "如何学习{}？",
            "{}的主要优点是什么？",
            "请比较{}和其替代方案。",
            "{}在实际中有哪些应用？",
        ]

    topics = [
        "机器学习", "深度学习", "自然语言处理", "计算机视觉",
        "强化学习", "生成模型", "注意力机制", "Transformer",
        "卷积神经网络", "循环神经网络", "图神经网络", "知识图谱",
        "推荐系统", "语音识别", "目标检测", "语义分割",
    ]

    # 好的回答特征：详细、结构化、有具体例子
    good_patterns = [
        "这是一个很好的问题。{}主要包含以下几个方面：首先，{}；其次，{}；最后，{}。",
        "{}可以从多个角度理解。从理论层面看，{}。从实践层面看，{}。总结来说，{}。",
        "关于{}，需要注意以下要点：1) {}；2) {}；3) {}。",
    ]

    # 差的回答特征：简短、含糊、缺乏深度
    bad_patterns = [
        "{}就是那个东西。",
        "这个嘛，{}差不多就是这样的。",
        "{}没什么特别的。",
    ]

    data = []
    for i in range(num_samples):
        topic = random.choice(topics)
        prompt = random.choice(prompt_templates).format(topic)
        chosen = random.choice(good_patterns).format(
            topic,
            f"{topic}的基础概念很重要",
            f"{topic}有很多实际应用",
            f"深入学习{topic}需要理论和实践结合"
        )
        rejected = random.choice(bad_patterns).format(topic)

        data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        })

    return data


if __name__ == "__main__":
    print("=" * 60)
    print("偏好数据处理模块演示")
    print("=" * 60)

    # 1. 生成合成偏好数据
    print("\n--- 1. 生成合成偏好数据 ---")
    synthetic_data = generate_synthetic_preferences(num_samples=50)
    print(f"生成了 {len(synthetic_data)} 条偏好数据")
    print(f"\n示例数据:")
    sample = synthetic_data[0]
    print(f"  Prompt:   {sample['prompt'][:60]}...")
    print(f"  Chosen:   {sample['chosen'][:60]}...")
    print(f"  Rejected: {sample['rejected'][:60]}...")

    # 2. 创建 PreferenceDataset
    print("\n--- 2. PreferenceDataset ---")
    dataset = PreferenceDataset(synthetic_data, max_length=128)
    print(f"数据集大小: {len(dataset)}")
    item = dataset[0]
    print(f"返回的键: {list(item.keys())}")
    for key, value in item.items():
        print(f"  {key}: 形状={value.shape}, dtype={value.dtype}")

    # 3. 创建 DataLoader
    print("\n--- 3. DataLoader ---")
    collate_fn = create_preference_collator()
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, shuffle=True)
    batch = next(iter(loader))
    print(f"Batch 内容:")
    for key, value in batch.items():
        print(f"  {key}: 形状={value.shape}")

    # 4. 排序偏好数据集
    print("\n--- 4. RankingPreferenceDataset ---")
    ranking_data = [
        {
            "prompt": "什么是深度学习？",
            "responses": [
                "深度学习是机器学习的一个分支，使用多层神经网络来学习数据的层次化表示。",
                "深度学习就是用很深的网络来做机器学习。",
                "深度学习就是那个AI技术。",
            ]
        },
        {
            "prompt": "如何训练神经网络？",
            "responses": [
                "训练神经网络的标准流程包括：数据准备、模型定义、前向传播、损失计算、反向传播和参数更新。",
                "用梯度下降法来训练。",
                "训练就完事了。",
            ]
        }
    ]
    ranking_dataset = RankingPreferenceDataset(ranking_data, max_length=128)
    print(f"排名数据: {len(ranking_data)} 个 prompt")
    print(f"展开后的偏好对: {len(ranking_dataset)} 对")

    print("\n" + "=" * 60)
    print("偏好数据处理演示完成!")
    print("=" * 60)
