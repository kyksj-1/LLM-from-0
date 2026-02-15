"""
Unigram Language Model 分词器实现
自顶向下的子词分词算法, 使用 Viterbi 解码

作者: LLM学习教程
模块: 模块1 - Tokenization

数学原理:
Unigram 模型假设每个子词独立出现, 文本的概率为:
    P(x1, x2, ..., xn) = ∏ P(xi)

最优分词通过 Viterbi 算法求解:
    x* = argmax_x ∑ log P(xi)

训练使用 EM 算法:
    E步: 用当前模型计算每种分词方式的期望
    M步: 更新子词概率, 删除低贡献子词

损失函数:
    L = -∑_{s ∈ D} log P(s | V)
    其中 P(s | V) = ∑_{x ∈ S(s)} ∏ P(xi)
"""

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class UnigramTokenizer:
    """
    从零实现的 Unigram Language Model 分词器

    与 BPE/WordPiece 的核心区别:
    - BPE/WordPiece: 自底向上 (从字符开始合并)
    - Unigram: 自顶向下 (从大词汇表开始剪枝)

    训练流程:
    1. 初始化一个较大的候选词汇表
    2. 用 EM 算法估计每个子词的概率
    3. 计算每个子词的损失贡献
    4. 删除损失贡献最小的子词
    5. 重复 2-4 直到达到目标大小

    推理使用 Viterbi 动态规划找到最优分词路径。

    Args:
        vocab_size: 目标词汇表大小
        initial_vocab_size: 初始候选词汇表大小
        shrinking_factor: 每轮剪枝保留比例
        unk_token: 未知词标记
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        initial_vocab_size: int = 0,
        shrinking_factor: float = 0.75,
        unk_token: str = "<unk>",
    ):
        self.vocab_size = vocab_size
        self.initial_vocab_size = initial_vocab_size or vocab_size * 4
        self.shrinking_factor = shrinking_factor
        self.unk_token = unk_token

        # 子词概率表: token -> log_prob
        self.token_probs: Dict[str, float] = {}
        # 词汇表: token -> id
        self.vocab: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

    def _build_initial_vocab(
        self, word_freqs: Dict[str, int]
    ) -> Dict[str, float]:
        """
        构建初始候选词汇表

        策略: 收集所有字符 + 高频子串

        Args:
            word_freqs: 词频字典

        Returns:
            初始子词概率 {subword: log_probability}
        """
        # 统计所有子串的频率
        substr_freqs: Dict[str, int] = defaultdict(int)

        for word, freq in word_freqs.items():
            # 添加所有子串 (长度 1 到 min(len(word), max_piece_len))
            max_len = min(len(word), 20)
            for start in range(len(word)):
                for end in range(start + 1, min(start + max_len, len(word)) + 1):
                    substr = word[start:end]
                    substr_freqs[substr] += freq

        # 按频率排序, 取前 initial_vocab_size 个
        sorted_substrs = sorted(substr_freqs.items(), key=lambda x: -x[1])

        # 确保所有单字符都在词汇表中
        all_chars = set()
        for word in word_freqs:
            all_chars.update(word)

        initial_vocab: Dict[str, float] = {}
        for c in sorted(all_chars):
            initial_vocab[c] = substr_freqs.get(c, 1)

        for substr, freq in sorted_substrs:
            if len(initial_vocab) >= self.initial_vocab_size:
                break
            if substr not in initial_vocab:
                initial_vocab[substr] = freq

        # 归一化为概率
        total = sum(initial_vocab.values())
        return {k: math.log(v / total) for k, v in initial_vocab.items()}

    def _viterbi(self, text: str) -> List[str]:
        """
        Viterbi 算法: 找到最优分词路径

        动态规划:
            dp[i] = 文本前 i 个字符的最大对数概率
            bp[i] = 最优路径的回溯指针 (最优分词的起始位置)

        转移方程:
            dp[j] = max_{i<j, text[i:j] ∈ V} (dp[i] + log P(text[i:j]))

        时间复杂度: O(n * max_piece_len)

        Args:
            text: 输入文本

        Returns:
            最优分词的子词列表
        """
        n = len(text)
        if n == 0:
            return []

        # dp[i]: 前 i 个字符的最大 log probability
        NEG_INF = float("-inf")
        dp = [NEG_INF] * (n + 1)
        dp[0] = 0.0

        # 回溯指针: bp[i] = 最优分割点
        bp = [0] * (n + 1)

        for j in range(1, n + 1):
            for i in range(max(0, j - 20), j):  # 限制最大子词长度
                piece = text[i:j]
                if piece in self.token_probs:
                    score = dp[i] + self.token_probs[piece]
                    if score > dp[j]:
                        dp[j] = score
                        bp[j] = i

        # 如果无法完成分词, 尝试逐字符
        if dp[n] == NEG_INF:
            return list(text)

        # 回溯
        tokens = []
        pos = n
        while pos > 0:
            start = bp[pos]
            tokens.append(text[start:pos])
            pos = start

        tokens.reverse()
        return tokens

    def _compute_loss(
        self, word_freqs: Dict[str, int]
    ) -> float:
        """
        计算当前模型在语料上的负对数似然损失

        L = -∑_{w ∈ D} freq(w) * log P(best_segmentation(w))

        Args:
            word_freqs: 词频字典

        Returns:
            损失值
        """
        loss = 0.0
        for word, freq in word_freqs.items():
            tokens = self._viterbi(word)
            word_log_prob = sum(
                self.token_probs.get(t, -100.0) for t in tokens
            )
            loss -= freq * word_log_prob
        return loss

    def _compute_token_scores(
        self, word_freqs: Dict[str, int]
    ) -> Dict[str, float]:
        """
        计算每个子词的"损失贡献": 删除该子词后损失增加多少

        损失贡献大 → 该子词很重要, 应保留
        损失贡献小 → 该子词可以删除

        Args:
            word_freqs: 词频字典

        Returns:
            每个子词的损失贡献 {token: loss_increase}
        """
        scores: Dict[str, float] = {}
        base_loss = self._compute_loss(word_freqs)

        for token in list(self.token_probs.keys()):
            # 单字符不能删除 (确保可分词性)
            if len(token) == 1:
                scores[token] = float("inf")
                continue

            # 暂时删除该 token
            old_prob = self.token_probs.pop(token)
            new_loss = self._compute_loss(word_freqs)
            scores[token] = new_loss - base_loss
            # 恢复
            self.token_probs[token] = old_prob

        return scores

    def train(self, corpus: List[str], verbose: bool = False):
        """
        训练 Unigram 分词器

        EM 式训练流程:
        1. 初始化大词汇表
        2. 循环:
           a) E步: Viterbi 分词, 统计子词使用频率
           b) M步: 更新概率, 计算损失贡献, 删除低贡献子词
        3. 直到词汇表缩小到目标大小

        Args:
            corpus: 训练语料
            verbose: 是否打印过程
        """
        # 预分词
        word_freqs: Dict[str, int] = defaultdict(int)
        for text in corpus:
            text = text.lower().strip()
            for word in text.split():
                word_freqs[word] += 1

        # 步骤1: 初始化词汇表
        self.token_probs = self._build_initial_vocab(word_freqs)

        if verbose:
            print(f"初始词汇表大小: {len(self.token_probs)}")

        # 步骤2: 迭代剪枝
        iteration = 0
        while len(self.token_probs) > self.vocab_size:
            iteration += 1

            # E步: 更新概率 (基于 Viterbi 分词的使用频率)
            token_counts: Dict[str, int] = defaultdict(int)
            for word, freq in word_freqs.items():
                tokens = self._viterbi(word)
                for t in tokens:
                    token_counts[t] += freq

            # M步: 更新概率
            total = sum(token_counts.values())
            if total > 0:
                for token in self.token_probs:
                    count = token_counts.get(token, 0)
                    self.token_probs[token] = math.log(
                        (count + 1) / (total + len(self.token_probs))
                    )  # Laplace 平滑

            # 计算每个 token 的损失贡献
            scores = self._compute_token_scores(word_freqs)

            # 按损失贡献排序, 保留贡献大的
            target_size = max(
                self.vocab_size,
                int(len(self.token_probs) * self.shrinking_factor),
            )

            sorted_tokens = sorted(scores.items(), key=lambda x: -x[1])
            keep_tokens = set()
            for token, _ in sorted_tokens[:target_size]:
                keep_tokens.add(token)

            # 确保单字符保留
            for token in list(self.token_probs.keys()):
                if len(token) == 1:
                    keep_tokens.add(token)

            # 剪枝
            old_size = len(self.token_probs)
            self.token_probs = {
                k: v for k, v in self.token_probs.items() if k in keep_tokens
            }

            if verbose:
                loss = self._compute_loss(word_freqs)
                print(
                    f"迭代 {iteration}: {old_size} -> {len(self.token_probs)} "
                    f"(loss={loss:.2f})"
                )

        # 构建词汇表映射
        self.vocab = {self.unk_token: 0}
        for i, token in enumerate(sorted(self.token_probs.keys()), start=1):
            self.vocab[token] = i
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        if verbose:
            print(f"训练完成, 最终词汇表大小: {len(self.vocab)}")

    def encode(self, text: str) -> List[int]:
        """
        编码: 将文本分词并转为 ID 序列

        使用 Viterbi 算法找到最大概率分词

        Args:
            text: 输入文本

        Returns:
            token ID 列表
        """
        text = text.lower().strip()
        ids = []
        for word in text.split():
            tokens = self._viterbi(word)
            for t in tokens:
                ids.append(self.vocab.get(t, self.vocab[self.unk_token]))
        return ids

    def tokenize(self, text: str) -> List[str]:
        """
        分词: 返回子词 token 列表 (不转 ID)

        Args:
            text: 输入文本

        Returns:
            子词列表
        """
        text = text.lower().strip()
        tokens = []
        for word in text.split():
            tokens.extend(self._viterbi(word))
        return tokens

    def decode(self, ids: List[int]) -> str:
        """
        解码: 将 ID 序列还原为文本

        Args:
            ids: token ID 列表

        Returns:
            还原的文本
        """
        tokens = [self.id_to_token.get(i, self.unk_token) for i in ids]
        # 简单拼接 (Unigram 不使用 ## 前缀, 依赖空格分隔)
        return " ".join(tokens)


if __name__ == "__main__":
    # 演示: 训练一个小型 Unigram 分词器
    corpus = [
        "the cat sat on the mat",
        "the cat ate the mouse and the cheese",
        "the dog sat on the log near the house",
        "the dog ate the bone in the garden",
        "a small cat is a nice animal",
        "a big dog is a loyal animal",
        "the small cat sat on the big mat",
        "the loyal dog ate a big bone",
        "cats and dogs are popular pets",
        "the mouse ran away from the cat",
    ]

    print("=" * 60)
    print("Unigram Language Model 分词器训练演示")
    print("=" * 60)

    tokenizer = UnigramTokenizer(vocab_size=50, initial_vocab_size=200)
    tokenizer.train(corpus, verbose=True)

    print(f"\n词汇表: {sorted(tokenizer.vocab.keys())[:30]}...")

    # 测试分词
    test_texts = [
        "the cat sat on the mat",
        "a small dog ate the bone",
        "the mouse ran away",
    ]

    print("\n--- 分词测试 ---")
    for text in test_texts:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        print(f"原文:   {text}")
        print(f"Tokens: {tokens}")
        print(f"IDs:    {ids}")
        print()

    # 展示 Viterbi 算法的工作过程
    print("--- Viterbi 分词示例 ---")
    word = "animals"
    tokens = tokenizer._viterbi(word)
    probs = [tokenizer.token_probs.get(t, -100) for t in tokens]
    print(f"单词: {word}")
    print(f"分词: {tokens}")
    print(f"各段 log P: {[f'{p:.3f}' for p in probs]}")
    print(f"总 log P: {sum(probs):.3f}")
