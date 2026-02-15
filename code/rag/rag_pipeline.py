"""
RAG Pipeline：完整的 Retrieve-Read-Generate 管道

本模块实现了一个完整的 RAG（Retrieval-Augmented Generation）系统，
将检索器、重排序器和生成器串联为统一的 Pipeline。

RAG 的数学形式:
    P(y|x) ≈ sum_{z in TopK(x)} P(y|x,z) * P(z|x)

其中:
    - P(z|x): 检索模型，给定查询 x，文档 z 的检索概率
    - P(y|x,z): 生成模型，给定查询 x 和参考文档 z，生成回答 y 的概率

Pipeline 流程:
    1. 索引阶段: 文档分块 → 编码为向量 → 存入向量索引
    2. 检索阶段: 查询编码 → 向量检索 Top-K → (可选) 重排序
    3. 生成阶段: 构建 Prompt (查询 + 检索文档) → LLM 生成回答

本实现支持:
    - 稠密检索 (Dense Retrieval)
    - 稀疏检索 (BM25)
    - 混合检索 (Hybrid Search + RRF 融合)
    - 重排序 (Cross-Encoder Reranking)
"""

import math
from typing import Optional, Protocol

from bm25_retriever import BM25Retriever


class TextSplitter:
    """
    文本分块器

    将长文档切分为固定大小的 chunk，支持重叠（overlap）以保留上下文连续性。

    参数:
        chunk_size: 每个 chunk 的最大字符数
        chunk_overlap: 相邻 chunk 之间的重叠字符数
        separator: 分割分隔符
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separator: str = "\n",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split(self, text: str) -> list[str]:
        """
        将文本切分为 chunk

        参数:
            text: 输入文本

        返回:
            chunk 列表
        """
        # 先按分隔符切分
        paragraphs = text.split(self.separator)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # 如果当前 chunk 加上新段落不超过限制，追加
            if len(current_chunk) + len(para) + 1 <= self.chunk_size:
                if current_chunk:
                    current_chunk += self.separator + para
                else:
                    current_chunk = para
            else:
                # 保存当前 chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # 如果单个段落超过 chunk_size，强制切分
                if len(para) > self.chunk_size:
                    for i in range(0, len(para), self.chunk_size - self.chunk_overlap):
                        chunk = para[i : i + self.chunk_size]
                        if chunk.strip():
                            chunks.append(chunk.strip())
                    current_chunk = ""
                else:
                    # 新 chunk 从重叠部分开始
                    if chunks and self.chunk_overlap > 0:
                        overlap = chunks[-1][-self.chunk_overlap :]
                        current_chunk = overlap + self.separator + para
                    else:
                        current_chunk = para

        # 添加最后一个 chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def split_documents(
        self,
        documents: list[str],
        doc_ids: Optional[list[str]] = None,
    ) -> tuple[list[str], list[str]]:
        """
        批量切分文档

        参数:
            documents: 文档列表
            doc_ids: 文档 ID 列表

        返回:
            (chunks, chunk_doc_ids) chunk 列表和对应的源文档 ID
        """
        if doc_ids is None:
            doc_ids = [str(i) for i in range(len(documents))]

        all_chunks = []
        all_chunk_doc_ids = []

        for doc, doc_id in zip(documents, doc_ids):
            chunks = self.split(doc)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_chunk_doc_ids.append(f"{doc_id}_chunk_{i}")

        return all_chunks, all_chunk_doc_ids


class RRFMerger:
    """
    倒数排名融合（Reciprocal Rank Fusion）

    将多个检索系统的排序结果融合为统一的排序。
    RRF 只使用排名信息，无需分数归一化。

    公式: RRF(d) = sum_r 1/(k + rank_r(d))

    参数:
        k: 平滑参数（通常 k=60）
    """

    def __init__(self, k: int = 60):
        self.k = k

    def merge(
        self,
        *ranked_lists: list[tuple[str, float]],
        weights: Optional[list[float]] = None,
    ) -> list[tuple[str, float]]:
        """
        融合多个排序列表

        参数:
            *ranked_lists: 多个排序列表，每个为 [(doc_id, score), ...]
            weights: 每个列表的权重（可选）

        返回:
            融合后的排序列表 [(doc_id, rrf_score), ...]
        """
        if weights is None:
            weights = [1.0] * len(ranked_lists)

        rrf_scores: dict[str, float] = {}

        for weight, ranked_list in zip(weights, ranked_lists):
            for rank, (doc_id, _) in enumerate(ranked_list):
                # rank 从 0 开始，RRF 公式中排名从 1 开始
                rrf_score = weight / (self.k + rank + 1)
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_score

        # 按 RRF 分数降序排列
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results


class SimpleGenerator:
    """
    简化版生成器（模拟 LLM 生成）

    在实际系统中，此组件调用 LLM API（如 OpenAI、Anthropic、DeepSeek）。
    本实现使用模板拼接模拟生成过程。
    """

    def __init__(self, model_name: str = "mock-llm"):
        self.model_name = model_name

    def generate(
        self,
        query: str,
        context_docs: list[str],
        max_tokens: int = 500,
    ) -> str:
        """
        基于检索到的文档生成回答

        参数:
            query: 用户查询
            context_docs: 检索到的文档列表
            max_tokens: 最大生成 token 数

        返回:
            生成的回答文本
        """
        # 构建 Prompt
        prompt = self._build_prompt(query, context_docs)

        # 模拟 LLM 生成（实际系统中调用 API）
        response = self._mock_generate(prompt, query, context_docs)

        return response

    def _build_prompt(self, query: str, context_docs: list[str]) -> str:
        """
        构建 RAG Prompt

        参数:
            query: 用户查询
            context_docs: 参考文档列表

        返回:
            完整的 Prompt 文本
        """
        context = "\n\n".join(
            [f"[文档 {i + 1}] {doc}" for i, doc in enumerate(context_docs)]
        )

        prompt = f"""请基于以下参考文档回答用户的问题。如果参考文档中没有足够信息，请明确说明。

参考文档:
{context}

用户问题: {query}

回答:"""

        return prompt

    def _mock_generate(
        self, prompt: str, query: str, context_docs: list[str]
    ) -> str:
        """模拟 LLM 生成（仅用于演示）"""
        if not context_docs:
            return f"未找到与问题 '{query}' 相关的参考文档。"

        # 简单地返回检索到的文档摘要
        doc_summaries = []
        for i, doc in enumerate(context_docs):
            summary = doc[:100] + "..." if len(doc) > 100 else doc
            doc_summaries.append(f"[参考{i + 1}] {summary}")

        return (
            f"基于 {len(context_docs)} 篇参考文档的回答:\n"
            + "\n".join(doc_summaries)
            + f"\n\n(注: 此为模拟生成，实际系统中由 LLM 生成自然语言回答)"
        )


class RAGPipeline:
    """
    完整的 RAG Pipeline

    整合文本分块、BM25 检索、RRF 融合和生成器，
    形成完整的 Retrieve-Read-Generate 管道。

    参数:
        chunk_size: 文本分块大小
        chunk_overlap: 分块重叠大小
        bm25_k1: BM25 的 k1 参数
        bm25_b: BM25 的 b 参数
        rrf_k: RRF 的平滑参数
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
    ):
        self.splitter = TextSplitter(chunk_size, chunk_overlap)
        self.bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)
        self.rrf = RRFMerger(k=rrf_k)
        self.generator = SimpleGenerator()

        # 文档存储
        self.chunks: list[str] = []
        self.chunk_ids: list[str] = []
        self.chunk_map: dict[str, str] = {}  # chunk_id -> chunk_text

        self._indexed = False

    def add_documents(
        self,
        documents: list[str],
        doc_ids: Optional[list[str]] = None,
    ):
        """
        添加文档到知识库

        参数:
            documents: 文档列表
            doc_ids: 文档 ID 列表
        """
        print(f"添加 {len(documents)} 篇文档...")

        # 分块
        chunks, chunk_ids = self.splitter.split_documents(documents, doc_ids)
        self.chunks.extend(chunks)
        self.chunk_ids.extend(chunk_ids)

        # 更新 chunk 映射
        for chunk_id, chunk in zip(chunk_ids, chunks):
            self.chunk_map[chunk_id] = chunk

        # 重建 BM25 索引
        self.bm25.build_index(self.chunks, self.chunk_ids)
        self._indexed = True

        print(f"文档分块完成: {len(self.chunks)} 个 chunk")

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "bm25",
    ) -> list[tuple[str, float]]:
        """
        检索相关文档

        参数:
            query: 用户查询
            top_k: 返回的文档数量
            method: 检索方法 ("bm25")

        返回:
            [(chunk_id, score), ...] 排序后的结果
        """
        if not self._indexed:
            raise ValueError("请先调用 add_documents() 添加文档")

        if method == "bm25":
            return self.bm25.search(query, top_k=top_k)
        else:
            raise ValueError(f"不支持的检索方法: {method}")

    def query(
        self,
        question: str,
        top_k: int = 5,
        method: str = "bm25",
        verbose: bool = False,
    ) -> str:
        """
        完整的 RAG 问答

        步骤:
        1. 检索相关文档
        2. 获取文档内容
        3. 生成回答

        参数:
            question: 用户问题
            top_k: 检索的文档数量
            method: 检索方法
            verbose: 是否打印中间结果

        返回:
            生成的回答
        """
        if verbose:
            print(f"\n问题: {question}")
            print(f"检索方法: {method}, Top-K: {top_k}")

        # 步骤 1: 检索
        results = self.retrieve(question, top_k=top_k, method=method)

        if verbose:
            print(f"\n检索结果:")
            for chunk_id, score in results:
                chunk_text = self.chunk_map.get(chunk_id, "")
                preview = chunk_text[:80] + "..." if len(chunk_text) > 80 else chunk_text
                print(f"  {chunk_id} (score={score:.4f}): {preview}")

        # 步骤 2: 获取文档内容
        context_docs = []
        for chunk_id, _ in results:
            if chunk_id in self.chunk_map:
                context_docs.append(self.chunk_map[chunk_id])

        # 步骤 3: 生成回答
        answer = self.generator.generate(question, context_docs)

        if verbose:
            print(f"\n生成的回答:")
            print(answer)

        return answer

    def get_stats(self) -> dict:
        """获取 Pipeline 统计信息"""
        return {
            "total_chunks": len(self.chunks),
            "avg_chunk_length": (
                sum(len(c) for c in self.chunks) / len(self.chunks)
                if self.chunks
                else 0
            ),
            "bm25_vocab_size": len(self.bm25.doc_freq) if self._indexed else 0,
        }


class HybridRAGPipeline(RAGPipeline):
    """
    混合检索 RAG Pipeline

    在 BM25 稀疏检索的基础上，支持（模拟的）稠密向量检索，
    并使用 RRF 算法融合两种检索结果。

    继承自 RAGPipeline，扩展了混合检索能力。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._dense_scores: dict[str, dict[str, float]] = {}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        method: str = "hybrid",
    ) -> list[tuple[str, float]]:
        """
        混合检索

        参数:
            query: 用户查询
            top_k: 返回的文档数量
            method: "bm25" | "dense" | "hybrid"

        返回:
            [(chunk_id, score), ...]
        """
        if not self._indexed:
            raise ValueError("请先调用 add_documents() 添加文档")

        if method == "bm25":
            return self.bm25.search(query, top_k=top_k)

        elif method == "dense":
            # 模拟稠密检索（实际应使用向量模型 + 向量索引）
            return self._mock_dense_search(query, top_k)

        elif method == "hybrid":
            # BM25 + 稠密检索 + RRF 融合
            bm25_results = self.bm25.search(query, top_k=top_k * 2)
            dense_results = self._mock_dense_search(query, top_k * 2)

            # RRF 融合
            merged = self.rrf.merge(bm25_results, dense_results)
            return merged[:top_k]

        else:
            raise ValueError(f"不支持的检索方法: {method}")

    def _mock_dense_search(
        self, query: str, top_k: int
    ) -> list[tuple[str, float]]:
        """
        模拟稠密向量检索

        在实际系统中，应使用 Embedding 模型 + 向量索引（如 HNSW）。
        此处使用简单的字符重叠度模拟语义相似度。
        """
        query_chars = set(query.lower())
        scores = []

        for chunk_id, chunk in zip(self.chunk_ids, self.chunks):
            chunk_chars = set(chunk.lower())
            # Jaccard 相似度作为模拟的语义相似度
            intersection = len(query_chars & chunk_chars)
            union = len(query_chars | chunk_chars)
            sim = intersection / union if union > 0 else 0.0
            scores.append((chunk_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Pipeline 完整演示")
    print("=" * 60)

    # 1. 准备文档
    print("\n--- 1. 准备知识库文档 ---")
    documents = [
        """Transformer 是由 Vaswani 等人在 2017 年提出的深度学习架构。
它的核心创新是 Self-Attention 机制，允许每个位置直接关注序列中的所有其他位置。
Transformer 完全摒弃了 RNN 的循环结构，通过并行计算大幅提升了训练效率。
Self-Attention 的计算公式为 Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V，
其中 Q、K、V 分别是查询、键和值矩阵。""",

        """GPT（Generative Pre-trained Transformer）是 OpenAI 开发的自回归语言模型系列。
GPT-1 首次证明了大规模预训练加微调的范式的有效性。
GPT-2 展示了 zero-shot 能力，GPT-3 通过 175B 参数实现了 few-shot learning。
GPT 系列使用 Transformer 的 Decoder 架构，通过因果掩码确保自回归生成。""",

        """HNSW（Hierarchical Navigable Small World）是目前最流行的近似最近邻搜索算法。
它结合了跳表的分层思想和小世界网络的导航性质。
HNSW 的搜索复杂度为 O(log N)，在百万级向量上可以在毫秒级完成搜索。
核心超参数包括 M（每层最大连接数）和 ef（搜索候选集大小）。""",

        """BM25 是最经典的基于关键词的检索算法，是 TF-IDF 的概率改进版。
BM25 引入了词频饱和机制和文档长度归一化，解决了 TF-IDF 的两个主要问题。
其核心参数包括 k1（词频饱和参数，通常 1.2-2.0）和 b（长度归一化参数，通常 0.75）。
在深度学习时代，BM25 仍然是稀疏检索的基准方法。""",

        """RAG（Retrieval-Augmented Generation）通过引入外部知识库来增强大语言模型。
RAG 的核心流程包括：索引构建、文档检索和答案生成三个阶段。
Advanced RAG 在此基础上引入了 Query 重写、HyDE 假设性文档嵌入和 Reranking 重排序等优化。
RAG 有效地缓解了 LLM 的幻觉问题和知识截止问题。""",

        """GraphRAG 是微软提出的基于知识图谱的 RAG 方法，专门用于回答全局性问题。
其工作流程包括：实体抽取、知识图谱构建、Leiden 社区发现算法、社区摘要生成。
与传统向量 RAG 相比，GraphRAG 能够进行跨文档的归纳推理。
但 GraphRAG 的索引构建成本较高，需要大量 LLM 调用进行实体抽取和摘要生成。""",
    ]
    doc_ids = [
        "transformer_intro",
        "gpt_series",
        "hnsw_algorithm",
        "bm25_algorithm",
        "rag_overview",
        "graph_rag",
    ]
    print(f"文档数量: {len(documents)}")

    # 2. 创建 RAG Pipeline
    print("\n--- 2. 创建 RAG Pipeline ---")
    rag = RAGPipeline(
        chunk_size=300,
        chunk_overlap=30,
        bm25_k1=1.5,
        bm25_b=0.75,
    )
    rag.add_documents(documents, doc_ids)

    stats = rag.get_stats()
    print(f"Pipeline 统计: {stats}")

    # 3. 问答演示
    print("\n--- 3. 问答演示 ---")
    questions = [
        "HNSW 算法的搜索复杂度是多少？",
        "RAG 如何解决幻觉问题？",
        "BM25 的核心参数有哪些？",
        "GraphRAG 与传统 RAG 有什么区别？",
    ]

    for q in questions:
        print("\n" + "=" * 40)
        answer = rag.query(q, top_k=3, verbose=True)

    # 4. 混合检索演示
    print("\n--- 4. 混合检索演示 ---")
    hybrid_rag = HybridRAGPipeline(
        chunk_size=300,
        chunk_overlap=30,
    )
    hybrid_rag.add_documents(documents, doc_ids)

    test_query = "transformer attention mechanism"
    print(f"\n查询: '{test_query}'")

    for method in ["bm25", "dense", "hybrid"]:
        results = hybrid_rag.retrieve(test_query, top_k=3, method=method)
        print(f"\n  {method.upper()} 结果:")
        for chunk_id, score in results:
            print(f"    {chunk_id}: {score:.4f}")

    # 5. 文本分块演示
    print("\n--- 5. 文本分块演示 ---")
    splitter = TextSplitter(chunk_size=150, chunk_overlap=20)
    long_text = documents[0]
    chunks = splitter.split(long_text)
    print(f"原文长度: {len(long_text)} 字符")
    print(f"分块数量: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i} ({len(chunk)} chars): {chunk[:60]}...")

    # 6. RRF 融合演示
    print("\n--- 6. RRF 融合演示 ---")
    rrf = RRFMerger(k=60)

    list1 = [("doc_A", 10.5), ("doc_B", 8.3), ("doc_C", 5.1)]
    list2 = [("doc_B", 0.95), ("doc_C", 0.88), ("doc_D", 0.72)]

    merged = rrf.merge(list1, list2)
    print("列表 1 (BM25):", [(d, f"{s:.2f}") for d, s in list1])
    print("列表 2 (向量):", [(d, f"{s:.2f}") for d, s in list2])
    print("RRF 融合结果:")
    for doc_id, score in merged:
        print(f"  {doc_id}: RRF score = {score:.6f}")

    print("\n演示完成!")
