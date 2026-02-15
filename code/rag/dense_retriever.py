"""
稠密检索器（Dense Retriever）：基于 Embedding 的双塔检索

本模块实现了基于 Bi-Encoder 架构的稠密向量检索系统，包括：
1. 双塔编码器（Query Encoder + Document Encoder）
2. InfoNCE 对比学习训练
3. 向量索引构建与相似度搜索

核心思想：将 Query 和 Document 分别编码为稠密向量，通过内积或余弦相似度
衡量相关性。文档向量可以离线预计算并建立索引，在线推理时只需编码 Query
并计算相似度。

数学形式：
    sim(q, d) = E_q(q)^T * E_d(d)
    L_InfoNCE = -log(exp(sim(q, d+)/tau) / sum(exp(sim(q, dj)/tau)))
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiEncoder(nn.Module):
    """
    双塔编码器模型

    Query 和 Document 分别通过独立的（或共享的）Transformer 编码器
    映射到同一向量空间。

    参数:
        vocab_size: 词汇表大小
        d_model: 隐藏层维度
        n_heads: 注意力头数
        n_layers: Transformer 层数
        max_seq_len: 最大序列长度
        output_dim: 输出向量维度
        shared_encoder: 是否共享 Query 和 Doc 编码器
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 512,
        output_dim: int = 128,
        shared_encoder: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.shared_encoder = shared_encoder

        # 词嵌入和位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.query_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if shared_encoder:
            # 共享编码器：Query 和 Doc 使用同一个编码器
            self.doc_encoder = self.query_encoder
        else:
            # 独立编码器
            doc_encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            self.doc_encoder = nn.TransformerEncoder(
                doc_encoder_layer, num_layers=n_layers
            )

        # 投影层：将 [CLS] 向量映射到输出空间
        self.query_projector = nn.Linear(d_model, output_dim)
        self.doc_projector = nn.Linear(d_model, output_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _encode(
        self,
        input_ids: torch.Tensor,
        encoder: nn.TransformerEncoder,
        projector: nn.Linear,
    ) -> torch.Tensor:
        """
        通用编码函数

        参数:
            input_ids: 输入 token ID, shape: [batch_size, seq_len]
            encoder: Transformer 编码器
            projector: 输出投影层

        返回:
            向量表示, shape: [batch_size, output_dim]
        """
        batch_size, seq_len = input_ids.shape

        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        # Transformer 编码
        x = encoder(x)

        # 使用第一个 token（[CLS]）的表示作为序列向量
        cls_output = x[:, 0, :]

        # 投影到输出空间并 L2 归一化
        output = projector(cls_output)
        output = F.normalize(output, p=2, dim=-1)

        return output

    def encode_query(self, query_ids: torch.Tensor) -> torch.Tensor:
        """
        编码 Query

        参数:
            query_ids: Query 的 token ID, shape: [batch_size, seq_len]

        返回:
            Query 向量, shape: [batch_size, output_dim]
        """
        return self._encode(query_ids, self.query_encoder, self.query_projector)

    def encode_doc(self, doc_ids: torch.Tensor) -> torch.Tensor:
        """
        编码 Document

        参数:
            doc_ids: Document 的 token ID, shape: [batch_size, seq_len]

        返回:
            Document 向量, shape: [batch_size, output_dim]
        """
        return self._encode(doc_ids, self.doc_encoder, self.doc_projector)

    def forward(
        self, query_ids: torch.Tensor, doc_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：同时编码 Query 和 Document

        参数:
            query_ids: shape [batch_size, query_len]
            doc_ids: shape [batch_size, doc_len]

        返回:
            (query_vectors, doc_vectors), 均为 shape [batch_size, output_dim]
        """
        query_vectors = self.encode_query(query_ids)
        doc_vectors = self.encode_doc(doc_ids)
        return query_vectors, doc_vectors


class InfoNCELoss(nn.Module):
    """
    InfoNCE 对比学习损失函数

    给定一个 batch 中的 N 个 (query, positive_doc) 对，
    InfoNCE 将其视为 N 分类问题：第 i 个 query 需要从 N 个 doc 中
    识别出正确的 positive_doc。

    数学形式:
        L_i = -log(exp(sim(q_i, d_i+)/tau) / sum_j(exp(sim(q_i, dj)/tau)))
        L = (1/N) * sum_i(L_i)

    参数:
        temperature: 温度参数 tau，控制分布的锐度
            - tau -> 0: 分布趋于 one-hot，只关注最难的负样本
            - tau -> inf: 分布趋于均匀，所有负样本同等对待
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_vectors: torch.Tensor,
        doc_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 InfoNCE 损失

        参数:
            query_vectors: shape [batch_size, dim]，已 L2 归一化
            doc_vectors: shape [batch_size, dim]，已 L2 归一化

        返回:
            标量损失值
        """
        batch_size = query_vectors.shape[0]

        # 计算所有 query-doc 对之间的相似度矩阵
        # similarity[i][j] = query_i 和 doc_j 之间的余弦相似度
        similarity = torch.mm(query_vectors, doc_vectors.t()) / self.temperature
        # shape: [batch_size, batch_size]

        # 对角线上的是正样本对（query_i 与 doc_i）
        # 非对角线上的是负样本对（batch 内负样本）
        labels = torch.arange(batch_size, device=similarity.device)

        # InfoNCE = 交叉熵损失，目标是让 similarity[i][i] 最大
        loss = F.cross_entropy(similarity, labels)

        return loss


class DenseRetriever:
    """
    稠密向量检索器

    支持离线构建文档索引和在线查询检索。
    使用暴力搜索（brute-force），适合中小规模数据集。
    大规模场景请结合 HNSW 索引（见 hnsw_index.py）。

    参数:
        model: 训练好的 BiEncoder 模型
        device: 计算设备
    """

    def __init__(self, model: BiEncoder, device: str = "cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

        # 文档索引
        self.doc_vectors: Optional[torch.Tensor] = None  # [num_docs, dim]
        self.doc_ids: list = []

    @torch.no_grad()
    def build_index(
        self,
        doc_token_ids: list[torch.Tensor],
        doc_ids: list[str],
        batch_size: int = 64,
    ):
        """
        构建文档向量索引

        参数:
            doc_token_ids: 文档的 token ID 列表
            doc_ids: 文档 ID 列表
            batch_size: 编码批大小
        """
        self.doc_ids = doc_ids
        all_vectors = []

        # 分批编码文档
        for i in range(0, len(doc_token_ids), batch_size):
            batch = doc_token_ids[i : i + batch_size]
            # 填充到相同长度
            max_len = max(t.shape[0] for t in batch)
            padded = torch.zeros(len(batch), max_len, dtype=torch.long)
            for j, t in enumerate(batch):
                padded[j, : t.shape[0]] = t

            padded = padded.to(self.device)
            vectors = self.model.encode_doc(padded)
            all_vectors.append(vectors.cpu())

        self.doc_vectors = torch.cat(all_vectors, dim=0)
        print(f"索引构建完成: {self.doc_vectors.shape[0]} 篇文档, "
              f"向量维度 {self.doc_vectors.shape[1]}")

    @torch.no_grad()
    def search(
        self,
        query_token_ids: torch.Tensor,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """
        检索与查询最相关的文档

        参数:
            query_token_ids: 查询的 token ID, shape: [seq_len]
            top_k: 返回的文档数量

        返回:
            [(doc_id, score), ...] 按分数降序排列
        """
        if self.doc_vectors is None:
            raise ValueError("请先调用 build_index() 构建索引")

        # 编码查询
        query_ids = query_token_ids.unsqueeze(0).to(self.device)
        query_vector = self.model.encode_query(query_ids).cpu()

        # 计算余弦相似度（向量已 L2 归一化，内积即余弦相似度）
        similarities = torch.mm(query_vector, self.doc_vectors.t()).squeeze(0)

        # 获取 Top-K
        top_k = min(top_k, len(self.doc_ids))
        scores, indices = torch.topk(similarities, top_k)

        results = [
            (self.doc_ids[idx.item()], score.item())
            for idx, score in zip(indices, scores)
        ]
        return results


def train_bi_encoder(
    model: BiEncoder,
    train_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    epochs: int = 10,
    lr: float = 2e-5,
    temperature: float = 0.07,
    batch_size: int = 32,
    device: str = "cpu",
) -> list[float]:
    """
    训练双塔编码器

    参数:
        model: BiEncoder 模型
        train_pairs: [(query_ids, doc_ids), ...] 训练数据对
        epochs: 训练轮数
        lr: 学习率
        temperature: InfoNCE 温度参数
        batch_size: 批大小
        device: 计算设备

    返回:
        每个 epoch 的平均损失列表
    """
    model = model.to(device)
    model.train()

    criterion = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        # 简单地按顺序遍历数据（实际应用中应该 shuffle）
        for i in range(0, len(train_pairs), batch_size):
            batch_pairs = train_pairs[i : i + batch_size]
            if len(batch_pairs) < 2:
                continue  # InfoNCE 至少需要 2 个样本

            # 提取 query 和 doc
            queries = [p[0] for p in batch_pairs]
            docs = [p[1] for p in batch_pairs]

            # 填充到相同长度
            max_q_len = max(q.shape[0] for q in queries)
            max_d_len = max(d.shape[0] for d in docs)

            query_batch = torch.zeros(len(queries), max_q_len, dtype=torch.long)
            doc_batch = torch.zeros(len(docs), max_d_len, dtype=torch.long)

            for j, q in enumerate(queries):
                query_batch[j, : q.shape[0]] = q
            for j, d in enumerate(docs):
                doc_batch[j, : d.shape[0]] = d

            query_batch = query_batch.to(device)
            doc_batch = doc_batch.to(device)

            # 前向传播
            query_vectors, doc_vectors = model(query_batch, doc_batch)

            # 计算 InfoNCE 损失
            loss = criterion(query_vectors, doc_vectors)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return epoch_losses


if __name__ == "__main__":
    print("=" * 60)
    print("稠密检索器（Dense Retriever）演示")
    print("=" * 60)

    # 超参数
    vocab_size = 5000
    d_model = 128
    output_dim = 64
    n_heads = 4
    n_layers = 2
    max_seq_len = 64

    # 1. 创建模型
    print("\n--- 1. 创建 BiEncoder 模型 ---")
    model = BiEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
        output_dim=output_dim,
        shared_encoder=True,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 2. 生成模拟训练数据
    print("\n--- 2. 生成模拟训练数据 ---")
    num_pairs = 200
    train_pairs = []
    for _ in range(num_pairs):
        q_len = torch.randint(5, 20, (1,)).item()
        d_len = torch.randint(20, 50, (1,)).item()
        query = torch.randint(1, vocab_size, (q_len,))
        doc = torch.randint(1, vocab_size, (d_len,))
        train_pairs.append((query, doc))
    print(f"训练数据量: {len(train_pairs)} 对")

    # 3. 训练
    print("\n--- 3. 训练 BiEncoder ---")
    losses = train_bi_encoder(
        model=model,
        train_pairs=train_pairs,
        epochs=5,
        lr=1e-4,
        temperature=0.07,
        batch_size=32,
    )

    # 4. 构建索引并检索
    print("\n--- 4. 构建索引并检索 ---")
    # 生成模拟文档库
    num_docs = 100
    doc_token_ids = [torch.randint(1, vocab_size, (30,)) for _ in range(num_docs)]
    doc_ids = [f"doc_{i}" for i in range(num_docs)]

    retriever = DenseRetriever(model)
    retriever.build_index(doc_token_ids, doc_ids)

    # 查询
    query = torch.randint(1, vocab_size, (10,))
    results = retriever.search(query, top_k=5)
    print("\n检索结果 (Top-5):")
    for doc_id, score in results:
        print(f"  {doc_id}: score = {score:.4f}")

    # 5. InfoNCE 损失演示
    print("\n--- 5. InfoNCE 损失计算演示 ---")
    criterion = InfoNCELoss(temperature=0.07)

    # 模拟一组归一化的向量
    batch_size_demo = 8
    dim = 64
    q_vecs = F.normalize(torch.randn(batch_size_demo, dim), dim=-1)
    d_vecs = F.normalize(torch.randn(batch_size_demo, dim), dim=-1)

    # 让正样本对更相似（模拟训练后的效果）
    d_vecs_aligned = F.normalize(q_vecs + 0.1 * torch.randn_like(q_vecs), dim=-1)

    loss_random = criterion(q_vecs, d_vecs)
    loss_aligned = criterion(q_vecs, d_vecs_aligned)
    print(f"随机向量对的 InfoNCE 损失: {loss_random.item():.4f}")
    print(f"对齐向量对的 InfoNCE 损失: {loss_aligned.item():.4f}")
    print(f"(对齐后损失更低, 说明正样本对更容易被区分)")

    # 6. 温度参数影响
    print("\n--- 6. 温度参数对损失的影响 ---")
    for temp in [0.01, 0.05, 0.1, 0.5, 1.0]:
        crit = InfoNCELoss(temperature=temp)
        loss = crit(q_vecs, d_vecs)
        print(f"  tau = {temp:.2f}, Loss = {loss.item():.4f}")

    print("\n演示完成!")
