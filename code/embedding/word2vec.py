"""
Word2Vec 实现：Skip-Gram + Negative Sampling

从零实现词嵌入训练，包含：
- Skip-Gram 模型
- Negative Sampling 优化
- 训练循环
- 类比推理测试

作者: LLM学习教程
模块: 模块2 - Embedding
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from typing import List, Tuple, Dict
import random
import math


class SkipGramModel(nn.Module):
    """
    Skip-Gram 模型

    核心思想：给定中心词，预测上下文词

    目标函数（Negative Sampling）：
    L = log σ(u_o^T v_c) + Σ_k E[log σ(-u_k^T v_c)]

    其中：
    - v_c: 中心词嵌入
    - u_o: 正样本（上下文词）嵌入
    - u_k: 负样本嵌入
    - σ: sigmoid 函数
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        """
        Args:
            vocab_size: 词汇表大小
            embed_dim: 嵌入维度
        """
        super().__init__()
        # 中心词嵌入矩阵
        self.center_embedding = nn.Embedding(vocab_size, embed_dim)
        # 上下文词嵌入矩阵
        self.context_embedding = nn.Embedding(vocab_size, embed_dim)

        # 初始化
        nn.init.uniform_(self.center_embedding.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.context_embedding.weight)

    def forward(
        self,
        center: torch.Tensor,
        context: torch.Tensor,
        neg_samples: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 Negative Sampling 损失

        Args:
            center: 中心词ID [batch_size]
            context: 正样本上下文词ID [batch_size]
            neg_samples: 负样本ID [batch_size, n_neg]

        Returns:
            损失值 (标量)
        """
        # 获取嵌入
        center_v = self.center_embedding(center)       # [batch, dim]
        context_u = self.context_embedding(context)    # [batch, dim]
        neg_u = self.context_embedding(neg_samples)    # [batch, n_neg, dim]

        # 正样本得分: log σ(u_o^T v_c)
        pos_score = torch.sum(center_v * context_u, dim=-1)  # [batch]
        pos_loss = -torch.nn.functional.logsigmoid(pos_score).mean()

        # 负样本得分: Σ log σ(-u_k^T v_c)
        neg_score = torch.bmm(neg_u, center_v.unsqueeze(-1)).squeeze(-1)  # [batch, n_neg]
        neg_loss = -torch.nn.functional.logsigmoid(-neg_score).mean()

        return pos_loss + neg_loss

    def get_embedding(self) -> torch.Tensor:
        """返回最终的词嵌入（中心词嵌入）"""
        return self.center_embedding.weight.data


class Word2VecTrainer:
    """
    Word2Vec 训练器

    包含数据预处理、负采样、训练循环
    """

    def __init__(
        self,
        embed_dim: int = 100,
        window_size: int = 5,
        min_count: int = 5,
        n_neg: int = 5,
        subsample_threshold: float = 1e-3
    ):
        """
        Args:
            embed_dim: 嵌入维度
            window_size: 上下文窗口大小
            min_count: 最小词频
            n_neg: 负样本数量
            subsample_threshold: 高频词下采样阈值
        """
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.min_count = min_count
        self.n_neg = n_neg
        self.subsample_threshold = subsample_threshold

        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.word_freqs: List[float] = []
        self.neg_dist: List[float] = []  # 负采样分布

    def build_vocab(self, corpus: List[List[str]]) -> None:
        """
        构建词汇表

        Args:
            corpus: 分词后的文本列表
        """
        # 统计词频
        counter = Counter()
        for sentence in corpus:
            counter.update(sentence)

        # 过滤低频词
        vocab = {w: c for w, c in counter.items() if c >= self.min_count}

        # 构建映射
        for i, (word, count) in enumerate(sorted(vocab.items(), key=lambda x: -x[1])):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
            self.word_freqs.append(count)

        # 构建负采样分布: P(w)^0.75 / Z
        total = sum(f ** 0.75 for f in self.word_freqs)
        self.neg_dist = [(f ** 0.75) / total for f in self.word_freqs]

        print(f"词汇表大小: {len(self.word_to_id)}")

    def _generate_pairs(self, corpus: List[List[str]]) -> List[Tuple[int, int]]:
        """
        生成 (中心词, 上下文词) 对

        Args:
            corpus: 分词后的文本列表

        Returns:
            (center_id, context_id) 对列表
        """
        pairs = []
        for sentence in corpus:
            # 转换为ID
            ids = [self.word_to_id[w] for w in sentence if w in self.word_to_id]

            for i, center_id in enumerate(ids):
                # 动态窗口大小
                window = random.randint(1, self.window_size)

                for j in range(max(0, i - window), min(len(ids), i + window + 1)):
                    if i != j:
                        pairs.append((center_id, ids[j]))

        return pairs

    def _negative_sample(self, batch_size: int) -> torch.Tensor:
        """
        负采样

        按 P(w)^0.75 分布采样

        Returns:
            负样本ID [batch_size, n_neg]
        """
        neg_samples = torch.multinomial(
            torch.tensor(self.neg_dist),
            batch_size * self.n_neg,
            replacement=True
        ).view(batch_size, self.n_neg)
        return neg_samples

    def train(
        self,
        corpus: List[List[str]],
        epochs: int = 5,
        batch_size: int = 512,
        lr: float = 0.025,
        device: str = 'cpu'
    ) -> SkipGramModel:
        """
        训练 Word2Vec 模型

        Args:
            corpus: 分词后的文本列表
            epochs: 训练轮数
            batch_size: 批量大小
            lr: 学习率
            device: 计算设备

        Returns:
            训练好的模型
        """
        # 构建词汇表
        self.build_vocab(corpus)

        # 创建模型
        vocab_size = len(self.word_to_id)
        model = SkipGramModel(vocab_size, self.embed_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # 生成训练对
        pairs = self._generate_pairs(corpus)
        print(f"训练对数量: {len(pairs)}")

        # 训练循环
        for epoch in range(epochs):
            random.shuffle(pairs)
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                if len(batch) < 2:
                    continue

                centers = torch.tensor([p[0] for p in batch], device=device)
                contexts = torch.tensor([p[1] for p in batch], device=device)
                neg_samples = self._negative_sample(len(batch)).to(device)

                loss = model(centers, contexts, neg_samples)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        self.model = model
        return model

    def most_similar(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        查找最相似的词

        Args:
            word: 查询词
            top_k: 返回数量

        Returns:
            [(词, 余弦相似度)] 列表
        """
        if word not in self.word_to_id:
            return []

        embeddings = self.model.get_embedding()
        word_id = self.word_to_id[word]
        word_vec = embeddings[word_id]

        # 计算余弦相似度
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        normalized = embeddings / (norms + 1e-8)
        word_normalized = word_vec / (torch.norm(word_vec) + 1e-8)

        similarities = torch.mv(normalized, word_normalized)

        # 排除自身
        similarities[word_id] = -1.0

        # Top-K
        top_vals, top_ids = torch.topk(similarities, top_k)

        return [(self.id_to_word[idx.item()], val.item()) for idx, val in zip(top_ids, top_vals)]

    def analogy(self, word_a: str, word_b: str, word_c: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        类比推理: a - b + c = ?

        例如: king - man + woman = queen

        Args:
            word_a, word_b, word_c: 三个词
            top_k: 返回数量

        Returns:
            [(词, 相似度)] 列表
        """
        for w in [word_a, word_b, word_c]:
            if w not in self.word_to_id:
                print(f"词 '{w}' 不在词汇表中")
                return []

        embeddings = self.model.get_embedding()
        vec_a = embeddings[self.word_to_id[word_a]]
        vec_b = embeddings[self.word_to_id[word_b]]
        vec_c = embeddings[self.word_to_id[word_c]]

        # a - b + c
        target = vec_a - vec_b + vec_c

        # 计算余弦相似度
        norms = torch.norm(embeddings, dim=1, keepdim=True)
        normalized = embeddings / (norms + 1e-8)
        target_normalized = target / (torch.norm(target) + 1e-8)

        similarities = torch.mv(normalized, target_normalized)

        # 排除输入词
        exclude = {self.word_to_id[w] for w in [word_a, word_b, word_c]}
        for idx in exclude:
            similarities[idx] = -1.0

        top_vals, top_ids = torch.topk(similarities, top_k)

        return [(self.id_to_word[idx.item()], val.item()) for idx, val in zip(top_ids, top_vals)]


# ============ 使用示例 ============

if __name__ == "__main__":
    # 示例语料
    corpus = [
        "the king sits on the throne".split(),
        "the queen sits beside the king".split(),
        "a man and a woman walk together".split(),
        "the prince is the son of the king".split(),
        "the princess is the daughter of the queen".split(),
        "the boy and the girl play together".split(),
        "he is a strong man".split(),
        "she is a strong woman".split(),
    ]

    # 扩展语料（重复以增加训练数据）
    corpus = corpus * 100

    # 训练
    trainer = Word2VecTrainer(embed_dim=50, window_size=3, min_count=1, n_neg=5)
    model = trainer.train(corpus, epochs=10, batch_size=128, lr=0.01)

    # 测试相似词
    print("\n最相似的词:")
    for word in ["king", "man", "woman"]:
        similar = trainer.most_similar(word, top_k=5)
        if similar:
            print(f"  {word}: {similar}")

    # 测试类比推理
    print("\n类比推理:")
    result = trainer.analogy("king", "man", "woman")
    print(f"  king - man + woman = {result}")
