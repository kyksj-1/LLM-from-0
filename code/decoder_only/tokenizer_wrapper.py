"""
分词器封装模块

本模块提供了一个统一的分词器接口, 封装了不同来源的分词器:
- HuggingFace tokenizers (如 GPT-2, Llama)
- SentencePiece (如 Gemma)
- 简单的字符级分词器 (用于教学和测试)

功能:
- 统一的 encode / decode 接口
- 自动处理特殊 token (BOS, EOS, PAD)
- Batch 编码与 padding

参考:
- HuggingFace Tokenizers: https://huggingface.co/docs/tokenizers
"""

from typing import List, Optional, Union
import re


class SimpleCharTokenizer:
    """
    简单的字符级分词器 (用于教学和测试)

    将每个字符映射为一个 token ID。
    特殊 token:
    - 0: [PAD]
    - 1: [BOS]
    - 2: [EOS]
    - 3: [UNK]

    Args:
        text: 用于构建词汇表的文本
        max_vocab_size: 最大词汇表大小 (包括特殊 token)
    """

    # 特殊 token 定义
    PAD_TOKEN = "[PAD]"
    BOS_TOKEN = "[BOS]"
    EOS_TOKEN = "[EOS]"
    UNK_TOKEN = "[UNK]"

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, text: str = "", max_vocab_size: int = 1000):
        # 从文本中收集所有唯一字符
        chars = sorted(set(text))
        # 限制词汇表大小 (减去 4 个特殊 token)
        if len(chars) > max_vocab_size - 4:
            chars = chars[:max_vocab_size - 4]

        # 构建映射
        self.special_tokens = {
            self.PAD_TOKEN: self.PAD_ID,
            self.BOS_TOKEN: self.BOS_ID,
            self.EOS_TOKEN: self.EOS_ID,
            self.UNK_TOKEN: self.UNK_ID,
        }

        self.char_to_id = {char: i + 4 for i, char in enumerate(chars)}
        self.id_to_char = {i + 4: char for i, char in enumerate(chars)}

        # 加入特殊 token 的反向映射
        for token, token_id in self.special_tokens.items():
            self.id_to_char[token_id] = token

        self._vocab_size = len(chars) + 4

    @property
    def vocab_size(self) -> int:
        """词汇表大小"""
        return self._vocab_size

    @property
    def bos_token_id(self) -> int:
        return self.BOS_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_ID

    @property
    def pad_token_id(self) -> int:
        return self.PAD_ID

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        """
        将文本编码为 token ID 列表

        Args:
            text: 输入文本
            add_bos: 是否在开头添加 [BOS]
            add_eos: 是否在结尾添加 [EOS]

        Returns:
            token ID 列表
        """
        ids = []
        if add_bos:
            ids.append(self.BOS_ID)

        for char in text:
            ids.append(self.char_to_id.get(char, self.UNK_ID))

        if add_eos:
            ids.append(self.EOS_ID)

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        将 token ID 列表解码为文本

        Args:
            ids: token ID 列表
            skip_special: 是否跳过特殊 token

        Returns:
            解码后的文本
        """
        chars = []
        special_ids = set(self.special_tokens.values())

        for token_id in ids:
            if skip_special and token_id in special_ids:
                continue
            chars.append(self.id_to_char.get(token_id, self.UNK_TOKEN))

        return "".join(chars)

    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> dict:
        """
        批量编码文本

        Args:
            texts: 文本列表
            max_length: 最大长度 (None 表示使用批次内最大长度)
            padding: 是否进行 padding
            add_bos: 是否添加 [BOS]
            add_eos: 是否添加 [EOS]

        Returns:
            包含 input_ids 和 attention_mask 的字典
        """
        # 编码所有文本
        all_ids = [self.encode(text, add_bos, add_eos) for text in texts]

        # 确定最大长度
        if max_length is None:
            max_length = max(len(ids) for ids in all_ids)
        else:
            # 截断
            all_ids = [ids[:max_length] for ids in all_ids]

        # Padding
        attention_masks = []
        padded_ids = []
        for ids in all_ids:
            pad_len = max_length - len(ids)
            attention_mask = [1] * len(ids) + [0] * pad_len
            padded = ids + [self.PAD_ID] * pad_len
            padded_ids.append(padded)
            attention_masks.append(attention_mask)

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_masks,
        }


class TokenizerWrapper:
    """
    分词器统一封装

    提供统一接口, 可以封装不同来源的分词器。
    优先使用 HuggingFace transformers 的分词器 (如果可用),
    否则回退到简单的字符级分词器。

    Args:
        tokenizer_name_or_path: HuggingFace 分词器名称/路径, 或 None 使用字符分词器
        fallback_text: 字符分词器的训练文本 (仅在回退时使用)
    """

    def __init__(
        self,
        tokenizer_name_or_path: Optional[str] = None,
        fallback_text: str = "",
    ):
        self._hf_tokenizer = None
        self._char_tokenizer = None

        if tokenizer_name_or_path is not None:
            try:
                from transformers import AutoTokenizer
                self._hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
                print(f"已加载 HuggingFace 分词器: {tokenizer_name_or_path}")
            except (ImportError, OSError) as e:
                print(f"无法加载 HuggingFace 分词器: {e}")
                print("回退到字符级分词器")
                self._char_tokenizer = SimpleCharTokenizer(fallback_text)
        else:
            self._char_tokenizer = SimpleCharTokenizer(fallback_text)

    @property
    def vocab_size(self) -> int:
        """词汇表大小"""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.vocab_size
        return self._char_tokenizer.vocab_size

    @property
    def bos_token_id(self) -> Optional[int]:
        """BOS token ID"""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.bos_token_id
        return self._char_tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        """EOS token ID"""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.eos_token_id
        return self._char_tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        """PAD token ID"""
        if self._hf_tokenizer is not None:
            pad_id = self._hf_tokenizer.pad_token_id
            if pad_id is None:
                return self._hf_tokenizer.eos_token_id
            return pad_id
        return self._char_tokenizer.pad_token_id

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        将文本编码为 token ID 列表

        Args:
            text: 输入文本
            add_bos: 是否添加 BOS (HuggingFace 分词器可能自动处理)
            add_eos: 是否添加 EOS

        Returns:
            token ID 列表
        """
        if self._hf_tokenizer is not None:
            ids = self._hf_tokenizer.encode(text, add_special_tokens=add_bos)
            if add_eos and self.eos_token_id is not None:
                ids.append(self.eos_token_id)
            return ids
        return self._char_tokenizer.encode(text, add_bos, add_eos)

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        将 token ID 列表解码为文本

        Args:
            ids: token ID 列表
            skip_special: 是否跳过特殊 token

        Returns:
            解码后的文本
        """
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(ids, skip_special_tokens=skip_special)
        return self._char_tokenizer.decode(ids, skip_special)


if __name__ == "__main__":
    # --- 测试字符级分词器 ---
    print("=== 字符级分词器测试 ===")
    sample_text = "Hello, World! This is a test. 你好世界！"
    tokenizer = SimpleCharTokenizer(sample_text)

    print(f"词汇表大小: {tokenizer.vocab_size}")

    # 编码
    encoded = tokenizer.encode("Hello!", add_bos=True, add_eos=True)
    print(f"编码 'Hello!': {encoded}")

    # 解码
    decoded = tokenizer.decode(encoded, skip_special=True)
    print(f"解码: '{decoded}'")

    # 批量编码
    batch = tokenizer.batch_encode(
        ["Hello", "World!", "Hi"],
        padding=True,
    )
    print(f"批量编码 input_ids: {batch['input_ids']}")
    print(f"批量编码 attention_mask: {batch['attention_mask']}")

    # --- 测试 TokenizerWrapper ---
    print("\n=== TokenizerWrapper 测试 ===")
    # 使用字符分词器 (不需要 HuggingFace)
    wrapper = TokenizerWrapper(fallback_text=sample_text)
    print(f"词汇表大小: {wrapper.vocab_size}")
    print(f"BOS ID: {wrapper.bos_token_id}")
    print(f"EOS ID: {wrapper.eos_token_id}")

    encoded = wrapper.encode("Hello!")
    decoded = wrapper.decode(encoded)
    print(f"编码: {encoded}")
    print(f"解码: '{decoded}'")
