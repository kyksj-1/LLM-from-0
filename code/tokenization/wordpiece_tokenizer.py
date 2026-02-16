"""
WordPiece分词器完整实现
基于最大似然准则的子词分词算法

作者: LLM学习教程
模块: 模块1 - Tokenization

数学原理:
WordPiece 与 BPE 的核心区别在于合并策略:
- BPE: 选择频率最高的相邻对
- WordPiece: 选择使语料库似然值增加最大的相邻对

合并得分公式:
    score(a, b) = freq(ab) / (freq(a) * freq(b))

等价于互信息 (Pointwise Mutual Information):
    PMI(a, b) = log P(ab) / (P(a) * P(b))
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class WordPieceTokenizer:
    """
    从零实现的WordPiece分词器

    WordPiece 最早用于 Google 的日语/韩语分词系统,
    后被 BERT 采用, 成为 NLP 领域最广泛使用的分词算法之一。

    与 BPE 的关键区别:
    1. 合并策略: 似然值最大化 vs 频率最大化
    2. 编码标记: 使用 "##" 前缀标记非首字符子词
    3. 未知词处理: 使用 [UNK] 标记

    Args:
        vocab_size: 目标词汇表大小
        unk_token: 未知词标记, 默认 "[UNK]"
        prefix: 非首字符子词前缀, 默认 "##"
        special_tokens: 特殊标记列表
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        unk_token: str = "[UNK]",
        prefix: str = "##",
        special_tokens: Optional[List[str]] = None,
    ):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.prefix = prefix
        self.special_tokens = special_tokens or ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

        # 词汇表: token -> id
        self.vocab: Dict[str, int] = {}
        # 逆词汇表: id -> token
        self.id_to_token: Dict[int, str] = {}

    def _pre_tokenize(self, text: str) -> List[str]:
        """
        预分词: 将文本按空白和标点切分为词列表

        Args:
            text: 输入文本

        Returns:
            词列表
        """
        # 简单的按空白和标点分割
        text = text.lower().strip()
        words = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        return words

    def _word_to_chars(self, word: str) -> List[str]:
        """
        将单词拆分为字符序列, 非首字符添加 ## 前缀

        例: "hello" -> ["h", "##e", "##l", "##l", "##o"]

        Args:
            word: 输入单词

        Returns:
            带前缀标记的字符列表
        """
        if not word:
            return []
        chars = [word[0]]
        for c in word[1:]:
            chars.append(f"{self.prefix}{c}")
        return chars

    def _compute_pair_scores(
        self,
        word_freqs: Dict[str, int],
        word_splits: Dict[str, List[str]],
    ) -> Dict[Tuple[str, str], float]:
        """
        计算所有相邻对的 WordPiece 得分

        得分公式: score(a, b) = freq(ab) / (freq(a) * freq(b))

        这本质上是 Pointwise Mutual Information (PMI):
        - 高分: a 和 b 经常一起出现（相对于各自频率）
        - 低分: a 和 b 的共现纯属偶然

        Args:
            word_freqs: 词频字典 {word: count}
            word_splits: 当前分割 {word: [subword1, subword2, ...]}

        Returns:
            相邻对得分字典 {(a, b): score}
        """
        # 统计每个子词的频率
        token_freqs: Dict[str, int] = defaultdict(int)
        # 统计相邻对的频率
        pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)

        for word, freq in word_freqs.items():
            split = word_splits[word]
            if len(split) == 1:
                token_freqs[split[0]] += freq
                continue
            for i in range(len(split)):
                token_freqs[split[i]] += freq
                if i < len(split) - 1:
                    pair_freqs[(split[i], split[i + 1])] += freq

        # 计算得分: freq(ab) / (freq(a) * freq(b))
        scores: Dict[Tuple[str, str], float] = {}
        for pair, pair_freq in pair_freqs.items():
            a, b = pair
            fa = token_freqs.get(a, 1)
            fb = token_freqs.get(b, 1)
            scores[pair] = pair_freq / (fa * fb)

        return scores

    def _merge_pair(
        self,
        pair: Tuple[str, str],
        word_splits: Dict[str, List[str]],
    ) -> Dict[str, List[str]]:
        """
        在所有单词的分割中合并指定的相邻对

        Args:
            pair: 要合并的相邻对 (a, b)
            word_splits: 当前分割

        Returns:
            更新后的分割
        """
        a, b = pair
        for word in word_splits:
            split = word_splits[word]
            if len(split) <= 1:
                continue
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and split[i] == a and split[i + 1] == b:
                    # 合并 a + b
                    # 合并时需要处理 ## 前缀
                    if b.startswith(self.prefix):
                        merged = a + b[len(self.prefix):]
                    else:
                        merged = a + b
                    new_split.append(merged)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            word_splits[word] = new_split
        return word_splits

    def train(self, corpus: List[str], verbose: bool = False):
        """
        训练 WordPiece 分词器

        训练流程:
        1. 预分词, 统计词频
        2. 将每个词拆分为字符 (非首字符加 ## 前缀)
        3. 循环: 计算得分 → 合并最高分对 → 更新分割
        4. 直到达到目标词汇表大小

        Args:
            corpus: 训练语料 (文本列表)
            verbose: 是否打印训练过程
        """
        # 步骤1: 预分词并统计词频
        word_freqs: Dict[str, int] = defaultdict(int)
        for text in corpus:
            for word in self._pre_tokenize(text):
                word_freqs[word] += 1

        # 步骤2: 初始化——将每个词拆为字符
        word_splits: Dict[str, List[str]] = {}
        for word in word_freqs:
            word_splits[word] = self._word_to_chars(word)

        # 初始词汇表: 特殊标记 + 所有字符
        self.vocab = {}
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i

        # 收集所有初始字符
        all_chars = set()
        for split in word_splits.values():
            all_chars.update(split)
        for char in sorted(all_chars):
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        # 步骤3: 迭代合并
        num_merges = self.vocab_size - len(self.vocab)
        for step in range(num_merges):
            # 计算所有对的得分
            scores = self._compute_pair_scores(word_freqs, word_splits)
            if not scores:
                if verbose:
                    print(f"步骤 {step}: 没有更多可合并的对")
                break

            # 选择得分最高的对
            best_pair = max(scores, key=scores.get)
            best_score = scores[best_pair]

            # 合并
            word_splits = self._merge_pair(best_pair, word_splits)

            # 构造合并后的 token
            a, b = best_pair
            if b.startswith(self.prefix):
                new_token = a + b[len(self.prefix):]
            else:
                new_token = a + b

            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            if verbose and step % 100 == 0:
                print(f"步骤 {step}/{num_merges}: 合并 {best_pair} -> {new_token} (score={best_score:.6f})")

        # 构建逆映射
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"训练完成, 词汇表大小: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        编码: 将文本转换为 token ID 序列

        WordPiece 使用贪心最长匹配 (Greedy Longest-Match-First):
        从左到右扫描, 每次匹配词汇表中最长的子词。

        Args:
            text: 输入文本

        Returns:
            token ID 列表
        """
        tokens = []
        for word in self._pre_tokenize(text):
            word_tokens = self._tokenize_word(word)
            tokens.extend(word_tokens)
        return [self.vocab.get(t, self.vocab[self.unk_token]) for t in tokens]

    def _tokenize_word(self, word: str) -> List[str]:
        """
        对单个词进行 WordPiece 分词 (贪心最长匹配)

        算法:
        1. 从词的开头开始
        2. 找到词汇表中匹配的最长前缀
        3. 输出该前缀, 继续处理剩余部分 (加 ## 前缀)
        4. 如果找不到匹配, 输出 [UNK]

        Args:
            word: 输入单词

        Returns:
            子词 token 列表
        """
        if word in self.vocab:
            return [word]

        tokens = []
        start = 0
        while start < len(word):
            end = len(word)
            found = False
            while start < end:
                substr = word[start:end]
                if start > 0:
                    substr = f"{self.prefix}{substr}"
                if substr in self.vocab:
                    tokens.append(substr)
                    found = True
                    break
                end -= 1
            if not found:
                return [self.unk_token]
            start = end

        return tokens

    def decode(self, ids: List[int]) -> str:
        """
        解码: 将 token ID 序列还原为文本

        Args:
            ids: token ID 列表

        Returns:
            还原的文本
        """
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        # 合并 ## 前缀的子词
        text_parts = []
        for token in tokens:
            if token in self.special_tokens:
                continue
            if token.startswith(self.prefix):
                text_parts.append(token[len(self.prefix):])
            else:
                if text_parts:
                    text_parts.append(" ")
                text_parts.append(token)
        return "".join(text_parts)


if __name__ == "__main__":
    # 演示: 训练一个小型 WordPiece 分词器
    corpus = [
        "the cat sat on the mat",
        "the cat ate the mouse",
        "the dog sat on the log",
        "the dog ate the bone",
        "a cat is a small animal",
        "a dog is a loyal animal",
        "the small cat sat on the big mat",
        "the loyal dog ate a big bone",
    ]

    print("=" * 60)
    print("WordPiece 分词器训练演示")
    print("=" * 60)

    tokenizer = WordPieceTokenizer(vocab_size=100)
    tokenizer.train(corpus, verbose=True)

    print(f"\n词汇表大小: {len(tokenizer.vocab)}")
    print(f"词汇表示例: {list(tokenizer.vocab.keys())[:30]}")

    # 测试编码/解码
    test_texts = [
        "the cat sat on the mat",
        "a small dog",
        "unknown words here",
    ]

    print("\n--- 编码/解码测试 ---")
    for text in test_texts:
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids)
        tokens = [tokenizer.id_to_token.get(i, "[?]") for i in ids]
        print(f"原文:   {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs:    {ids}")
        print(f"解码:   {decoded}")
        print()
