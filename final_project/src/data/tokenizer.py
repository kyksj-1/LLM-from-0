"""
BPE 分词器

知识依赖:
- 模块 1（分词与词汇表构建）: BPE 算法原理、SentencePiece 工具

参考实现:
- code/tokenization/bpe_tokenizer.py

分词器选择:
    本项目推荐使用 sentencepiece 训练 BPE 分词器。
    也可以使用 HuggingFace tokenizers 库。

    为什么自己训练而不是用现成的?
    - 控制词汇表大小和构成
    - 适配训练数据的语言分布
    - 理解分词器的工作原理

    特殊 token 设计:
    - <pad>: 填充 token (ID: 0)
    - <unk>: 未知 token (ID: 1)
    - <bos>: 序列开始 (ID: 2)
    - <eos>: 序列结束 (ID: 3)
    - 对话模板 token (SFT 阶段使用):
      <|system|>, <|user|>, <|assistant|>, <|end_turn|>
"""

from pathlib import Path
from typing import List, Optional


class Tokenizer:
    """
    BPE 分词器封装

    提供训练、编码、解码功能。
    底层可使用 sentencepiece 或 tokenizers 库。

    Args:
        model_path: 已训练的分词器模型路径（.model 文件）
    """

    # 特殊 token ID
    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    def __init__(self, model_path: Optional[str] = None):
        # TODO: 加载已训练的分词器模型
        # 提示: 使用 sentencepiece.SentencePieceProcessor()
        raise NotImplementedError(
            "TODO: 加载分词器模型\n"
            "参考: 模块 1 (分词) 的 SentencePiece 章节\n"
            "提示: import sentencepiece as spm; sp = spm.SentencePieceProcessor()"
        )

    @staticmethod
    def train(
        input_files: List[str],
        model_prefix: str,
        vocab_size: int = 32000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
    ):
        """
        训练 BPE 分词器

        Args:
            input_files: 训练数据文件列表（纯文本）
            model_prefix: 输出模型的前缀（会生成 .model 和 .vocab 文件）
            vocab_size: 词汇表大小
            character_coverage: 字符覆盖率（中文建议 0.9995）
            model_type: 分词算法类型 ("bpe" / "unigram")

        产出:
            {model_prefix}.model — 分词器模型
            {model_prefix}.vocab — 词汇表

        实现提示:
            使用 sentencepiece.SentencePieceTrainer.train() 方法
            关键参数:
            - input: 输入文件（逗号分隔）
            - model_prefix: 输出前缀
            - vocab_size: 词汇表大小
            - model_type: "bpe"
            - character_coverage: 0.9995（适合中文）
            - pad_id, unk_id, bos_id, eos_id: 特殊 token ID
            - user_defined_symbols: 自定义特殊 token
        """
        raise NotImplementedError(
            "TODO: 训练 BPE 分词器\n"
            "参考: 模块 1 的 SentencePiece 训练章节\n"
            "提示: sentencepiece.SentencePieceTrainer.train(...)"
        )

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]:
        """
        将文本编码为 token ID 序列

        Args:
            text: 输入文本
            add_bos: 是否在开头添加 BOS token
            add_eos: 是否在末尾添加 EOS token

        Returns:
            token ID 列表
        """
        raise NotImplementedError(
            "TODO: 实现文本编码\n"
            "提示: sp.encode(text, out_type=int)"
        )

    def decode(self, ids: List[int]) -> str:
        """
        将 token ID 序列解码为文本

        Args:
            ids: token ID 列表

        Returns:
            解码后的文本
        """
        raise NotImplementedError(
            "TODO: 实现 token 解码\n"
            "提示: sp.decode(ids)"
        )

    @property
    def vocab_size(self) -> int:
        """词汇表大小"""
        raise NotImplementedError("TODO: 返回词汇表大小")
