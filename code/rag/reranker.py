"""
Cross-Encoder 重排序器（Reranker）

本模块实现了基于 Cross-Encoder 的文档重排序系统，包括：
1. Cross-Encoder 模型架构（Query + Doc 联合编码）
2. 训练逻辑（二分类：相关/不相关）
3. 重排序推理（对候选文档精排）

Cross-Encoder vs Bi-Encoder:
- Bi-Encoder: Query 和 Doc 独立编码，速度快但交互浅
- Cross-Encoder: Query 和 Doc 拼接后联合编码，精度高但速度慢
- 实际系统: Bi-Encoder 召回 Top-100 → Cross-Encoder 精排为 Top-10

数学形式:
    score(q, d) = sigmoid(W * h_[CLS])
    其中 h_[CLS] 是 BERT 编码 "[CLS] q [SEP] d [SEP]" 后的 [CLS] 向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CrossEncoder(nn.Module):
    """
    Cross-Encoder 重排序模型

    将 Query 和 Document 拼接后输入 Transformer 编码器，
    使用 [CLS] 位置的输出向量经过线性层预测相关性分数。

    与 Bi-Encoder 不同，Cross-Encoder 允许 Query 和 Document
    在每一层 Attention 中进行深度交互，因此精度更高。

    参数:
        vocab_size: 词汇表大小
        d_model: 隐藏层维度
        n_heads: 注意力头数
        n_layers: Transformer 层数
        max_seq_len: 最大序列长度（Query + Doc 拼接后的总长度）
        dropout: Dropout 概率
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 词嵌入和位置嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        # 段嵌入: 0 表示 Query 部分, 1 表示 Document 部分
        self.segment_embedding = nn.Embedding(2, d_model)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 分类头: [CLS] 向量 -> 标量分数
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

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

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        输入格式: [CLS] query_tokens [SEP] doc_tokens [SEP]
        段ID格式: [0,    0, ..., 0,    0,   1, ..., 1,    1]

        参数:
            input_ids: 拼接后的 token ID, shape: [batch_size, seq_len]
            segment_ids: 段 ID, shape: [batch_size, seq_len]

        返回:
            相关性分数, shape: [batch_size]
        """
        batch_size, seq_len = input_ids.shape

        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # 嵌入层: token + position + segment
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.segment_embedding(segment_ids)
        )

        # Transformer 编码
        x = self.encoder(x)

        # 使用 [CLS] 位置的输出
        cls_output = x[:, 0, :]

        # 分类
        score = self.classifier(cls_output).squeeze(-1)

        return score


class Reranker:
    """
    文档重排序器

    使用训练好的 Cross-Encoder 模型对候选文档进行精细排序。

    典型使用场景:
    1. Bi-Encoder 从百万文档中召回 Top-100
    2. Reranker (Cross-Encoder) 将 Top-100 精排为 Top-10
    3. 将 Top-10 文档传入 LLM 生成最终回答

    参数:
        model: 训练好的 CrossEncoder 模型
        device: 计算设备
        max_length: 最大输入长度（Query + Doc 拼接后）
        cls_token_id: [CLS] token ID
        sep_token_id: [SEP] token ID
    """

    def __init__(
        self,
        model: CrossEncoder,
        device: str = "cpu",
        max_length: int = 512,
        cls_token_id: int = 1,
        sep_token_id: int = 2,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id

    def _prepare_input(
        self,
        query_ids: torch.Tensor,
        doc_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        准备 Cross-Encoder 的输入

        格式: [CLS] query [SEP] doc [SEP]

        参数:
            query_ids: Query 的 token IDs
            doc_ids: Document 的 token IDs

        返回:
            (input_ids, segment_ids) 拼接和截断后的输入
        """
        # 拼接: [CLS] + query + [SEP] + doc + [SEP]
        cls = torch.tensor([self.cls_token_id])
        sep = torch.tensor([self.sep_token_id])

        input_tokens = torch.cat([cls, query_ids, sep, doc_ids, sep])

        # 截断到最大长度
        if len(input_tokens) > self.max_length:
            input_tokens = input_tokens[: self.max_length]

        # 段 ID: Query 部分为 0, Doc 部分为 1
        query_length = len(query_ids) + 2  # [CLS] + query + [SEP]
        segment_ids = torch.zeros(len(input_tokens), dtype=torch.long)
        segment_ids[query_length:] = 1

        return input_tokens, segment_ids

    @torch.no_grad()
    def rerank(
        self,
        query_ids: torch.Tensor,
        doc_ids_list: list[torch.Tensor],
        doc_metadata: Optional[list[dict]] = None,
    ) -> list[tuple[int, float, Optional[dict]]]:
        """
        对候选文档进行重排序

        参数:
            query_ids: Query 的 token IDs, shape: [query_len]
            doc_ids_list: 候选文档的 token IDs 列表
            doc_metadata: 文档的元信息列表（可选）

        返回:
            [(doc_index, score, metadata), ...] 按分数降序排列
        """
        all_input_ids = []
        all_segment_ids = []

        # 准备所有 Query-Doc 对的输入
        for doc_ids in doc_ids_list:
            input_ids, segment_ids = self._prepare_input(query_ids, doc_ids)
            all_input_ids.append(input_ids)
            all_segment_ids.append(segment_ids)

        # 填充到相同长度
        max_len = max(len(ids) for ids in all_input_ids)
        batch_input = torch.zeros(len(all_input_ids), max_len, dtype=torch.long)
        batch_segment = torch.zeros(len(all_segment_ids), max_len, dtype=torch.long)

        for i, (input_ids, segment_ids) in enumerate(
            zip(all_input_ids, all_segment_ids)
        ):
            batch_input[i, : len(input_ids)] = input_ids
            batch_segment[i, : len(segment_ids)] = segment_ids

        # 推理
        batch_input = batch_input.to(self.device)
        batch_segment = batch_segment.to(self.device)
        scores = self.model(batch_input, batch_segment)

        # 排序
        results = []
        for i, score in enumerate(scores.cpu().tolist()):
            meta = doc_metadata[i] if doc_metadata else None
            results.append((i, score, meta))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


def train_cross_encoder(
    model: CrossEncoder,
    train_data: list[tuple[torch.Tensor, torch.Tensor, float]],
    epochs: int = 10,
    lr: float = 2e-5,
    batch_size: int = 32,
    device: str = "cpu",
    cls_token_id: int = 1,
    sep_token_id: int = 2,
) -> list[float]:
    """
    训练 Cross-Encoder 模型

    训练目标: 二分类（相关/不相关），使用 Binary Cross-Entropy 损失。

    参数:
        model: CrossEncoder 模型
        train_data: [(query_ids, doc_ids, label), ...] label 为 1 (相关) 或 0 (不相关)
        epochs: 训练轮数
        lr: 学习率
        batch_size: 批大小
        device: 计算设备
        cls_token_id: [CLS] token ID
        sep_token_id: [SEP] token ID

    返回:
        每个 epoch 的平均损失列表
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(train_data), batch_size):
            batch = train_data[i : i + batch_size]
            if not batch:
                continue

            # 准备输入
            all_input_ids = []
            all_segment_ids = []
            labels = []

            for query_ids, doc_ids, label in batch:
                cls = torch.tensor([cls_token_id])
                sep = torch.tensor([sep_token_id])
                input_ids = torch.cat([cls, query_ids, sep, doc_ids, sep])
                query_length = len(query_ids) + 2
                segment_ids = torch.zeros(len(input_ids), dtype=torch.long)
                segment_ids[query_length:] = 1

                all_input_ids.append(input_ids)
                all_segment_ids.append(segment_ids)
                labels.append(label)

            # 填充
            max_len = max(len(ids) for ids in all_input_ids)
            batch_input = torch.zeros(len(all_input_ids), max_len, dtype=torch.long)
            batch_segment = torch.zeros(len(all_segment_ids), max_len, dtype=torch.long)

            for j, (inp, seg) in enumerate(zip(all_input_ids, all_segment_ids)):
                batch_input[j, : len(inp)] = inp
                batch_segment[j, : len(seg)] = seg

            batch_input = batch_input.to(device)
            batch_segment = batch_segment.to(device)
            batch_labels = torch.tensor(labels, dtype=torch.float32, device=device)

            # 前向传播
            scores = model(batch_input, batch_segment)
            loss = criterion(scores, batch_labels)

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
    print("Cross-Encoder 重排序器演示")
    print("=" * 60)

    # 超参数
    vocab_size = 5000
    d_model = 128
    n_heads = 4
    n_layers = 2
    max_seq_len = 128

    # 1. 创建模型
    print("\n--- 1. 创建 Cross-Encoder 模型 ---")
    model = CrossEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 2. 生成模拟训练数据
    print("\n--- 2. 生成模拟训练数据 ---")
    train_data = []
    for _ in range(300):
        q_len = torch.randint(3, 15, (1,)).item()
        d_len = torch.randint(10, 40, (1,)).item()
        query = torch.randint(3, vocab_size, (q_len,))
        doc = torch.randint(3, vocab_size, (d_len,))
        # 随机标签: 1 (相关) 或 0 (不相关)
        label = float(torch.randint(0, 2, (1,)).item())
        train_data.append((query, doc, label))
    print(f"训练数据: {len(train_data)} 个样本")

    # 3. 训练
    print("\n--- 3. 训练 Cross-Encoder ---")
    losses = train_cross_encoder(
        model=model,
        train_data=train_data,
        epochs=5,
        lr=1e-4,
        batch_size=32,
    )

    # 4. 重排序演示
    print("\n--- 4. 重排序演示 ---")
    reranker = Reranker(model, max_length=max_seq_len)

    # 模拟查询和候选文档
    query = torch.randint(3, vocab_size, (8,))
    candidate_docs = [torch.randint(3, vocab_size, (25,)) for _ in range(10)]
    doc_metadata = [{"doc_id": f"doc_{i}", "source": "test"} for i in range(10)]

    print(f"查询长度: {len(query)} tokens")
    print(f"候选文档数: {len(candidate_docs)}")

    results = reranker.rerank(query, candidate_docs, doc_metadata)

    print("\n重排序结果:")
    for rank, (doc_idx, score, meta) in enumerate(results):
        print(f"  Rank {rank + 1}: Doc {meta['doc_id']}, score = {score:.4f}")

    # 5. Cross-Encoder vs Bi-Encoder 对比
    print("\n--- 5. Cross-Encoder 特点说明 ---")
    print("Cross-Encoder vs Bi-Encoder:")
    print("  Cross-Encoder: Query 和 Doc 在每层 Attention 中深度交互")
    print("  Bi-Encoder: Query 和 Doc 独立编码，仅最后计算相似度")
    print("")
    print("  Cross-Encoder 精度更高，但速度更慢:")
    print(f"  - 10 个候选文档需要 10 次完整前向传播")
    print(f"  - 而 Bi-Encoder 只需 1 次 Query 编码 + 内积计算")
    print("")
    print("  典型使用流程:")
    print("  1. Bi-Encoder 从百万文档中召回 Top-100")
    print("  2. Cross-Encoder 对 Top-100 精排为 Top-10")
    print("  3. Top-10 文档送入 LLM 生成回答")

    print("\n演示完成!")
