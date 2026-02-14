"""
Word2Vec 实现：Skip-Gram + CBOW + Negative Sampling

本模块实现了经典的 Word2Vec 词嵌入训练算法。

核心概念:
- Skip-Gram: 给定中心词预测上下文词
- CBOW: 给定上下文词预测中心词
- Negative Sampling: 将多分类问题转化为二分类，大幅降低计算复杂度

数学基础:
- Skip-Gram 目标: max Σ Σ log P(w_{t+j} | w_t)
- Negative Sampling: log σ(u_o^T v_c) + Σ E[log σ(-u_k^T v_c)]
- 噪声分布: P_n(w) ∝ f(w)^{3/4}

参考:
- Mikolov et al. (2013). Efficient Estimation of Word Representations in Vector Space.
- Mikolov et al. (2013). Distributed Representations of Words and Phrases.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
from typing import List, Tuple, Optional


class SkipGram(nn.Module):
    """
    Skip-Gram 模型 + Negative Sampling

    给定中心词，预测上下文词。使用负采样优化替代完整 Softmax。

    Args:
        vocab_size: 词汇表大小
        embed_dim: 嵌入维度
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        # 中心词嵌入矩阵 (v_c)
        self.center_embed = nn.Embedding(vocab_size, embed_dim)
        # 上下文词嵌入矩阵 (u_o)
        self.context_embed = nn.Embedding(vocab_size, embed_dim)

        # 初始化：均匀分布 [-0.5/dim, 0.5/dim]
        nn.init.uniform_(self.center_embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.context_embed.weight)

    def forward(
        self,
        center: torch.Tensor,
        context: torch.Tensor,
        neg_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 Negative Sampling 损失

        Args:
            center: 中心词索引 [batch_size]
            context: 正样本上下文词索引 [batch_size]
            neg_samples: 负样本索引 [batch_size, K]

        Returns:
            标量损失值
        """
        # 中心词嵌入: v_c
        center_v = self.center_embed(center)        # [batch, dim]
        # 正样本上下文嵌入: u_o
        context_u = self.context_embed(context)      # [batch, dim]
        # 正样本得分: sigmoid(u_o^T v_c)
        pos_score = torch.sigmoid((center_v * context_u).sum(dim=-1))

        # 负样本嵌入: u_k
        neg_u = self.context_embed(neg_samples)      # [batch, K, dim]
        # 负样本得分: sigmoid(-u_k^T v_c)
        neg_score = torch.sigmoid(-(center_v.unsqueeze(1) * neg_u).sum(dim=-1))

        # 损失: -log(正样本得分) - Σlog(负样本得分)
        loss = -torch.log(pos_score + 1e-8).mean() - torch.log(neg_score + 1e-8).mean()
        return loss

    def get_embeddings(self) -> torch.Tensor:
        """返回训练好的词嵌入矩阵"""
        return self.center_embed.weight.detach()


class CBOW(nn.Module):
    """
    CBOW (Continuous Bag of Words) 模型

    给定上下文词，预测中心词。

    Args:
        vocab_size: 词汇表大小
        embed_dim: 嵌入维度
    """

    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Embedding(vocab_size, embed_dim)

        nn.init.uniform_(self.embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.zeros_(self.output.weight)

    def forward(
        self,
        context_words: torch.Tensor,
        center: torch.Tensor,
        neg_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_words: 上下文词索引 [batch_size, 2*window]
            center: 中心词索引 [batch_size]
            neg_samples: 负样本索引 [batch_size, K]

        Returns:
            标量损失值
        """
        # 上下文词嵌入取平均: v̄ = mean(v_{c+j})
        ctx_embed = self.embed(context_words).mean(dim=1)  # [batch, dim]

        # 正样本得分
        center_u = self.output(center)  # [batch, dim]
        pos_score = torch.sigmoid((ctx_embed * center_u).sum(dim=-1))

        # 负样本得分
        neg_u = self.output(neg_samples)  # [batch, K, dim]
        neg_score = torch.sigmoid(-(ctx_embed.unsqueeze(1) * neg_u).sum(dim=-1))

        loss = -torch.log(pos_score + 1e-8).mean() - torch.log(neg_score + 1e-8).mean()
        return loss


class Word2VecDataset(Dataset):
    """
    Word2Vec 训练数据集

    从文本语料中提取 (中心词, 上下文词) 样本对。

    Args:
        corpus: 分词后的文本列表
        window_size: 上下文窗口大小
        min_count: 最小词频
        neg_samples: 每个正样本的负采样数量
    """

    def __init__(
        self,
        corpus: List[List[str]],
        window_size: int = 5,
        min_count: int = 5,
        neg_samples: int = 5,
    ):
        self.window_size = window_size
        self.neg_samples = neg_samples

        # 构建词汇表
        word_counts = Counter(w for sent in corpus for w in sent)
        self.vocab = {
            w: i for i, (w, c) in enumerate(word_counts.most_common())
            if c >= min_count
        }
        self.idx2word = {i: w for w, i in self.vocab.items()}
        self.vocab_size = len(self.vocab)

        # 计算噪声分布: P_n(w) ∝ f(w)^{3/4}
        counts = np.array([
            word_counts.get(self.idx2word[i], 0)
            for i in range(self.vocab_size)
        ], dtype=np.float64)
        self.noise_dist = counts ** 0.75
        self.noise_dist /= self.noise_dist.sum()

        # 生成训练样本: (center_idx, context_idx)
        self.samples = []
        for sent in corpus:
            indices = [self.vocab[w] for w in sent if w in self.vocab]
            for i, center_idx in enumerate(indices):
                # 动态窗口大小
                actual_window = np.random.randint(1, window_size + 1)
                start = max(0, i - actual_window)
                end = min(len(indices), i + actual_window + 1)
                for j in range(start, end):
                    if j != i:
                        self.samples.append((center_idx, indices[j]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        center, context = self.samples[idx]
        # 负采样
        neg = np.random.choice(
            self.vocab_size,
            size=self.neg_samples,
            replace=True,
            p=self.noise_dist,
        )
        return (
            torch.tensor(center, dtype=torch.long),
            torch.tensor(context, dtype=torch.long),
            torch.tensor(neg, dtype=torch.long),
        )


def train_word2vec(
    corpus: List[List[str]],
    embed_dim: int = 100,
    window_size: int = 5,
    min_count: int = 5,
    neg_samples: int = 5,
    epochs: int = 5,
    batch_size: int = 512,
    lr: float = 0.025,
    model_type: str = "skipgram",
    device: str = "cpu",
) -> Tuple[nn.Module, dict]:
    """
    训练 Word2Vec 模型

    Args:
        corpus: 分词后的语料
        embed_dim: 嵌入维度
        window_size: 上下文窗口大小
        min_count: 最小词频
        neg_samples: 负采样数量
        epochs: 训练轮数
        batch_size: 批大小
        lr: 学习率
        model_type: "skipgram" 或 "cbow"
        device: 计算设备

    Returns:
        (model, vocab_dict): 训练好的模型和词汇表映射
    """
    dataset = Word2VecDataset(corpus, window_size, min_count, neg_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model_type == "skipgram":
        model = SkipGram(dataset.vocab_size, embed_dim).to(device)
    else:
        model = CBOW(dataset.vocab_size, embed_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (center, context, neg) in enumerate(dataloader):
            center, context, neg = center.to(device), context.to(device), neg.to(device)
            loss = model(center, context, neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return model, dataset.vocab


def analogy(
    embeddings: torch.Tensor,
    vocab: dict,
    idx2word: dict,
    word_a: str,
    word_b: str,
    word_c: str,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    词类比推理: a - b + c ≈ ?

    例如: king - man + woman ≈ queen

    Args:
        embeddings: 词嵌入矩阵 [vocab_size, dim]
        vocab: 词 → 索引映射
        idx2word: 索引 → 词映射
        word_a, word_b, word_c: 类比词
        top_k: 返回前k个结果

    Returns:
        [(word, similarity_score), ...]
    """
    if word_a not in vocab or word_b not in vocab or word_c not in vocab:
        raise ValueError(f"词不在词汇表中")

    # 计算目标向量: vec(a) - vec(b) + vec(c)
    vec = (
        embeddings[vocab[word_a]]
        - embeddings[vocab[word_b]]
        + embeddings[vocab[word_c]]
    )

    # 归一化
    vec = vec / vec.norm()
    norms = embeddings / embeddings.norm(dim=-1, keepdim=True)

    # 余弦相似度
    similarities = torch.matmul(norms, vec)

    # 排除输入词
    exclude = {vocab[word_a], vocab[word_b], vocab[word_c]}
    results = []
    for idx in similarities.argsort(descending=True):
        idx = idx.item()
        if idx not in exclude:
            results.append((idx2word[idx], similarities[idx].item()))
        if len(results) >= top_k:
            break

    return results
