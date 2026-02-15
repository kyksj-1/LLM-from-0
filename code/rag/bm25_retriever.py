"""
BM25 稀疏检索器：手写实现

本模块从零实现 BM25（Best Matching 25）算法，这是最经典的基于关键词的
稀疏检索算法，本质上是 TF-IDF 的概率改进版。

BM25 公式:
    BM25(Q, d) = sum_i IDF(t_i) * f(t_i,d) * (k1+1) / (f(t_i,d) + k1*(1-b+b*|d|/avgdl))

其中:
    - f(t_i, d): 词 t_i 在文档 d 中的词频
    - |d|: 文档长度（词数）
    - avgdl: 语料库平均文档长度
    - k1: TF 饱和参数（通常 1.2-2.0）
    - b: 长度归一化参数（通常 0.75）
    - IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

BM25 相对 TF-IDF 的改进:
    1. TF 饱和: 词频增大时权重趋于上界 k1+1，避免高频词主导
    2. 文档长度归一化: 长文档有轻微惩罚，避免长度偏差
"""

import math
from collections import Counter
from typing import Optional


class BM25Retriever:
    """
    BM25 检索器

    参数:
        k1: TF 饱和参数，控制词频的影响程度
            - k1=0: 不考虑词频，退化为二值模型
            - k1越大: 词频影响越大
            - 推荐: 1.2 到 2.0
        b: 文档长度归一化参数
            - b=0: 不做长度归一化
            - b=1: 完全按比例归一化
            - 推荐: 0.75
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        # 索引数据结构
        self.doc_ids: list[str] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0

        # 词频统计
        self.doc_term_freqs: list[Counter] = []  # 每篇文档的词频
        self.doc_freq: Counter = Counter()  # 每个词出现在多少篇文档中
        self.total_terms: int = 0

        self._indexed = False

    def _tokenize(self, text: str) -> list[str]:
        """
        简单分词器

        对于中文文本，实际应用中应使用 jieba 等分词工具。
        此处仅做空格分词演示。

        参数:
            text: 输入文本

        返回:
            分词后的 token 列表
        """
        # 转小写 + 按空格分词
        tokens = text.lower().strip().split()
        # 过滤空 token
        return [t for t in tokens if t]

    def build_index(self, documents: list[str], doc_ids: Optional[list[str]] = None):
        """
        构建 BM25 索引

        参数:
            documents: 文档列表（纯文本）
            doc_ids: 文档 ID 列表（可选，默认用序号）
        """
        self.num_docs = len(documents)
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(self.num_docs)]

        self.doc_term_freqs = []
        self.doc_lengths = []
        self.doc_freq = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            term_freq = Counter(tokens)

            self.doc_term_freqs.append(term_freq)
            self.doc_lengths.append(len(tokens))

            # 更新文档频率（每个词在多少篇文档中出现）
            for term in set(tokens):
                self.doc_freq[term] += 1

        self.avg_doc_length = sum(self.doc_lengths) / max(self.num_docs, 1)
        self._indexed = True

        print(f"BM25 索引构建完成:")
        print(f"  文档数: {self.num_docs}")
        print(f"  平均文档长度: {self.avg_doc_length:.1f} 词")
        print(f"  词汇表大小: {len(self.doc_freq)}")

    def _idf(self, term: str) -> float:
        """
        计算逆文档频率 (IDF)

        使用 Lucene 变体公式，确保 IDF >= 0:
            IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5))

        参数:
            term: 查询词

        返回:
            IDF 值
        """
        n_t = self.doc_freq.get(term, 0)
        return math.log(1.0 + (self.num_docs - n_t + 0.5) / (n_t + 0.5))

    def _bm25_term_score(self, term: str, doc_idx: int) -> float:
        """
        计算单个词对单个文档的 BM25 分数

        BM25(t, d) = IDF(t) * f(t,d) * (k1+1) / (f(t,d) + k1*(1-b+b*|d|/avgdl))

        参数:
            term: 查询词
            doc_idx: 文档索引

        返回:
            该词对该文档的 BM25 贡献分数
        """
        # 词频
        tf = self.doc_term_freqs[doc_idx].get(term, 0)
        if tf == 0:
            return 0.0

        # IDF
        idf = self._idf(term)

        # 文档长度归一化
        dl = self.doc_lengths[doc_idx]
        norm = 1.0 - self.b + self.b * (dl / self.avg_doc_length)

        # BM25 TF 分量
        tf_component = (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)

        return idf * tf_component

    def score(self, query: str, doc_idx: int) -> float:
        """
        计算查询与文档的 BM25 分数

        BM25(Q, d) = sum_t BM25(t, d)

        参数:
            query: 查询文本
            doc_idx: 文档索引

        返回:
            BM25 分数
        """
        if not self._indexed:
            raise ValueError("请先调用 build_index() 构建索引")

        query_terms = self._tokenize(query)
        total_score = 0.0

        for term in query_terms:
            total_score += self._bm25_term_score(term, doc_idx)

        return total_score

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        检索与查询最相关的文档

        参数:
            query: 查询文本
            top_k: 返回的文档数量

        返回:
            [(doc_id, score), ...] 按分数降序排列
        """
        if not self._indexed:
            raise ValueError("请先调用 build_index() 构建索引")

        # 计算查询与每篇文档的 BM25 分数
        scores = []
        for idx in range(self.num_docs):
            s = self.score(query, idx)
            if s > 0:
                scores.append((self.doc_ids[idx], s))

        # 按分数降序排列
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:top_k]

    def get_term_contributions(
        self, query: str, doc_idx: int
    ) -> list[tuple[str, float]]:
        """
        获取查询各词对文档分数的贡献（用于解释检索结果）

        参数:
            query: 查询文本
            doc_idx: 文档索引

        返回:
            [(term, score), ...] 按贡献降序排列
        """
        query_terms = self._tokenize(query)
        contributions = []

        for term in query_terms:
            score = self._bm25_term_score(term, doc_idx)
            if score > 0:
                contributions.append((term, score))

        contributions.sort(key=lambda x: x[1], reverse=True)
        return contributions


class TFIDFRetriever:
    """
    TF-IDF 检索器（用于与 BM25 对比）

    TF-IDF(t, d) = TF(t, d) * IDF(t)
    TF(t, d) = 词 t 在文档 d 中的频次
    IDF(t) = log(N / n(t))
    """

    def __init__(self):
        self.doc_ids: list[str] = []
        self.doc_term_freqs: list[Counter] = []
        self.doc_freq: Counter = Counter()
        self.num_docs: int = 0
        self._indexed = False

    def _tokenize(self, text: str) -> list[str]:
        """简单分词器"""
        return [t for t in text.lower().strip().split() if t]

    def build_index(self, documents: list[str], doc_ids: Optional[list[str]] = None):
        """构建 TF-IDF 索引"""
        self.num_docs = len(documents)
        self.doc_ids = doc_ids if doc_ids else [str(i) for i in range(self.num_docs)]

        self.doc_term_freqs = []
        self.doc_freq = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            term_freq = Counter(tokens)
            self.doc_term_freqs.append(term_freq)
            for term in set(tokens):
                self.doc_freq[term] += 1

        self._indexed = True

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """TF-IDF 检索"""
        if not self._indexed:
            raise ValueError("请先调用 build_index() 构建索引")

        query_terms = self._tokenize(query)
        scores = []

        for idx in range(self.num_docs):
            total_score = 0.0
            for term in query_terms:
                tf = self.doc_term_freqs[idx].get(term, 0)
                if tf > 0 and self.doc_freq.get(term, 0) > 0:
                    idf = math.log(self.num_docs / self.doc_freq[term])
                    total_score += tf * idf
            if total_score > 0:
                scores.append((self.doc_ids[idx], total_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


if __name__ == "__main__":
    print("=" * 60)
    print("BM25 检索器演示")
    print("=" * 60)

    # 1. 准备文档
    print("\n--- 1. 准备示例文档 ---")
    documents = [
        "transformer architecture uses self attention mechanism for sequence modeling",
        "BERT is a bidirectional transformer model for natural language understanding",
        "GPT uses autoregressive transformer decoder for text generation",
        "attention mechanism allows the model to focus on relevant parts of input",
        "recurrent neural networks suffer from vanishing gradient problem",
        "convolutional neural networks are widely used in computer vision tasks",
        "language models predict the next token based on previous context",
        "HNSW is an efficient algorithm for approximate nearest neighbor search",
        "BM25 is a probabilistic information retrieval algorithm based on TF-IDF",
        "vector databases store high dimensional embeddings for similarity search",
        "RAG combines retrieval and generation to reduce hallucination in LLMs",
        "knowledge graphs represent structured information as entities and relations",
    ]
    doc_ids = [f"doc_{i}" for i in range(len(documents))]

    # 2. 构建 BM25 索引
    print("\n--- 2. 构建 BM25 索引 ---")
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.build_index(documents, doc_ids)

    # 3. 检索
    print("\n--- 3. BM25 检索演示 ---")
    queries = [
        "transformer attention mechanism",
        "approximate nearest neighbor search algorithm",
        "retrieval augmented generation hallucination",
    ]

    for query in queries:
        print(f"\n查询: '{query}'")
        results = bm25.search(query, top_k=3)
        for doc_id, score in results:
            idx = doc_ids.index(doc_id)
            print(f"  {doc_id} (score={score:.4f}): {documents[idx][:60]}...")

        # 展示词项贡献
        if results:
            top_doc_idx = doc_ids.index(results[0][0])
            contributions = bm25.get_term_contributions(query, top_doc_idx)
            print(f"  词项贡献分解:")
            for term, contrib in contributions:
                print(f"    '{term}': {contrib:.4f}")

    # 4. BM25 vs TF-IDF 对比
    print("\n--- 4. BM25 vs TF-IDF 对比 ---")
    tfidf = TFIDFRetriever()
    tfidf.build_index(documents, doc_ids)

    test_query = "transformer attention mechanism"
    bm25_results = bm25.search(test_query, top_k=5)
    tfidf_results = tfidf.search(test_query, top_k=5)

    print(f"\n查询: '{test_query}'")
    print("\nBM25 排序:")
    for doc_id, score in bm25_results:
        print(f"  {doc_id}: {score:.4f}")

    print("\nTF-IDF 排序:")
    for doc_id, score in tfidf_results:
        print(f"  {doc_id}: {score:.4f}")

    # 5. 参数敏感性分析
    print("\n--- 5. BM25 参数敏感性分析 ---")
    test_query = "transformer attention"
    print(f"查询: '{test_query}'")

    print("\nk1 参数影响 (b=0.75):")
    for k1 in [0.0, 0.5, 1.0, 1.5, 2.0, 5.0]:
        bm25_temp = BM25Retriever(k1=k1, b=0.75)
        bm25_temp.build_index(documents, doc_ids)
        results = bm25_temp.search(test_query, top_k=1)
        if results:
            print(f"  k1={k1:.1f}: Top-1 = {results[0][0]} (score={results[0][1]:.4f})")

    print("\nb 参数影响 (k1=1.5):")
    for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
        bm25_temp = BM25Retriever(k1=1.5, b=b)
        bm25_temp.build_index(documents, doc_ids)
        results = bm25_temp.search(test_query, top_k=1)
        if results:
            print(f"  b={b:.2f}: Top-1 = {results[0][0]} (score={results[0][1]:.4f})")

    # 6. IDF 值分析
    print("\n--- 6. IDF 值分析 ---")
    print("常见词的 IDF 值:")
    test_terms = ["transformer", "model", "the", "attention", "HNSW", "BM25"]
    for term in test_terms:
        idf_val = bm25._idf(term.lower())
        df = bm25.doc_freq.get(term.lower(), 0)
        print(f"  '{term}': IDF={idf_val:.4f}, 出现在 {df}/{bm25.num_docs} 篇文档中")

    print("\n演示完成!")
