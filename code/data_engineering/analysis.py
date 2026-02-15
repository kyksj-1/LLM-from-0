"""
数据集统计分析工具

本模块提供了预训练数据集的统计分析功能，用于:
- 数据集基本统计（文档数、token数、词汇量等）
- 文本长度分布分析
- 词汇多样性分析
- n-gram 重复率分析
- 数据源分布分析
- 语言分布检测
- 数据质量报告生成

这些分析工具帮助数据工程师理解数据集特征，
评估清洗效果，指导数据混合决策。

参考:
- Dodge et al. (2021). Documenting Large Webtext Corpora.
- Penedo et al. (2024). The FineWeb Datasets.
"""

import math
import re
from typing import Dict, List, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class DatasetStatistics:
    """
    数据集统计结果

    Attributes:
        num_documents: 文档总数
        total_chars: 总字符数
        total_words: 总词数
        vocab_size: 词汇量（唯一词数）
        avg_doc_length_chars: 平均文档长度（字符）
        avg_doc_length_words: 平均文档长度（词）
        median_doc_length_words: 中位数文档长度（词）
        min_doc_length_words: 最短文档长度（词）
        max_doc_length_words: 最长文档长度（词）
        type_token_ratio: 词汇多样性指标
        source_distribution: 数据源分布
        length_histogram: 长度分布直方图数据
    """
    num_documents: int = 0
    total_chars: int = 0
    total_words: int = 0
    vocab_size: int = 0
    avg_doc_length_chars: float = 0.0
    avg_doc_length_words: float = 0.0
    median_doc_length_words: float = 0.0
    min_doc_length_words: int = 0
    max_doc_length_words: int = 0
    type_token_ratio: float = 0.0
    source_distribution: Dict[str, int] = field(default_factory=dict)
    length_histogram: Dict[str, int] = field(default_factory=dict)


class DatasetAnalyzer:
    """
    数据集分析器

    对文本数据集进行全面的统计分析。

    使用方法:
        analyzer = DatasetAnalyzer()
        stats = analyzer.analyze(documents)
        analyzer.print_report(stats)

    Args:
        estimate_tokens: 是否估计 token 数量
        token_ratio: 词到 token 的估计比率（英文约 1.3）
    """

    def __init__(
        self,
        estimate_tokens: bool = True,
        token_ratio: float = 1.3,
    ):
        self.estimate_tokens = estimate_tokens
        self.token_ratio = token_ratio

    def analyze(
        self,
        documents: List[str],
        sources: Optional[List[str]] = None,
    ) -> DatasetStatistics:
        """
        分析数据集

        Args:
            documents: 文档文本列表
            sources: 每个文档对应的数据源标签（可选）

        Returns:
            DatasetStatistics 统计结果
        """
        stats = DatasetStatistics()
        stats.num_documents = len(documents)

        if not documents:
            return stats

        # 基础统计
        word_counts = []
        all_words = Counter()

        for doc in documents:
            stats.total_chars += len(doc)
            words = doc.lower().split()
            word_count = len(words)
            word_counts.append(word_count)
            stats.total_words += word_count
            all_words.update(words)

        stats.vocab_size = len(all_words)

        # 长度统计
        word_counts.sort()
        stats.avg_doc_length_chars = stats.total_chars / stats.num_documents
        stats.avg_doc_length_words = stats.total_words / stats.num_documents
        stats.min_doc_length_words = word_counts[0]
        stats.max_doc_length_words = word_counts[-1]
        stats.median_doc_length_words = word_counts[len(word_counts) // 2]

        # 词汇多样性
        stats.type_token_ratio = (
            stats.vocab_size / stats.total_words
            if stats.total_words > 0 else 0
        )

        # 数据源分布
        if sources:
            stats.source_distribution = dict(Counter(sources))

        # 长度分布直方图
        stats.length_histogram = self._compute_length_histogram(word_counts)

        return stats

    def _compute_length_histogram(
        self, word_counts: List[int], num_bins: int = 10
    ) -> Dict[str, int]:
        """
        计算文档长度分布直方图

        将文档按词数分为若干区间，统计每个区间的文档数。
        """
        if not word_counts:
            return {}

        min_len = word_counts[0]
        max_len = word_counts[-1]

        if max_len == min_len:
            return {f"{min_len}-{max_len}": len(word_counts)}

        bin_size = max(1, (max_len - min_len) // num_bins)
        histogram = {}

        for wc in word_counts:
            bin_start = ((wc - min_len) // bin_size) * bin_size + min_len
            bin_end = bin_start + bin_size - 1
            bin_label = f"{bin_start}-{bin_end}"
            histogram[bin_label] = histogram.get(bin_label, 0) + 1

        return histogram

    def compute_ngram_repetition(
        self,
        documents: List[str],
        n: int = 5,
        sample_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        计算 n-gram 重复率

        在整个数据集层面分析 n-gram 的重复程度。

        Args:
            documents: 文档列表
            n: n-gram 阶数
            sample_size: 采样大小（如果数据集过大）

        Returns:
            包含重复率统计的字典
        """
        import random

        if sample_size and len(documents) > sample_size:
            docs = random.sample(documents, sample_size)
        else:
            docs = documents

        all_ngrams = Counter()
        total_ngrams = 0
        doc_ngram_counts = []

        for doc in docs:
            words = doc.lower().split()
            doc_ngrams = []
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i + n])
                doc_ngrams.append(ngram)
                all_ngrams[ngram] += 1
                total_ngrams += 1
            doc_ngram_counts.append(len(doc_ngrams))

        if total_ngrams == 0:
            return {
                'total_ngrams': 0,
                'unique_ngrams': 0,
                'repetition_rate': 0.0,
                'top_repeated_ratio': 0.0,
            }

        unique_ngrams = len(all_ngrams)
        repeated_ngrams = sum(
            count for count in all_ngrams.values() if count > 1
        )
        top_10_total = sum(
            count for _, count in all_ngrams.most_common(10)
        )

        return {
            'total_ngrams': total_ngrams,
            'unique_ngrams': unique_ngrams,
            'repetition_rate': 1.0 - unique_ngrams / total_ngrams,
            'top_repeated_ratio': top_10_total / total_ngrams,
            'most_common': [
                (' '.join(ngram), count)
                for ngram, count in all_ngrams.most_common(10)
            ],
        }

    def compute_vocabulary_analysis(
        self,
        documents: List[str],
        top_k: int = 20,
    ) -> Dict[str, object]:
        """
        词汇分析

        分析数据集的词汇分布特征。

        Args:
            documents: 文档列表
            top_k: 返回最高频的 K 个词

        Returns:
            词汇分析结果
        """
        word_freq = Counter()
        total_words = 0

        for doc in documents:
            words = doc.lower().split()
            word_freq.update(words)
            total_words += len(words)

        vocab_size = len(word_freq)

        # Zipf 定律检验: 第 r 名词频 ~ C / r
        top_words = word_freq.most_common(top_k)

        # Hapax legomena: 只出现一次的词
        hapax = sum(1 for count in word_freq.values() if count == 1)

        # 覆盖率: Top-K 词覆盖总词数的比例
        top_k_coverage = (
            sum(count for _, count in top_words) / total_words
            if total_words > 0 else 0
        )

        return {
            'vocab_size': vocab_size,
            'total_words': total_words,
            'hapax_legomena': hapax,
            'hapax_ratio': hapax / vocab_size if vocab_size > 0 else 0,
            'top_k_coverage': top_k_coverage,
            'top_words': [(word, count) for word, count in top_words],
        }

    def compute_compression_ratio(
        self,
        documents: List[str],
        sample_size: int = 100,
    ) -> float:
        """
        计算数据集的压缩比

        使用 zlib 压缩率作为信息密度的近似指标。
        压缩比越低，信息密度越高（冗余越少）。

        Args:
            documents: 文档列表
            sample_size: 采样大小

        Returns:
            平均压缩比 (compressed_size / original_size)
        """
        import zlib
        import random

        if len(documents) > sample_size:
            docs = random.sample(documents, sample_size)
        else:
            docs = documents

        ratios = []
        for doc in docs:
            original = doc.encode('utf-8')
            compressed = zlib.compress(original)
            if len(original) > 0:
                ratios.append(len(compressed) / len(original))

        return sum(ratios) / len(ratios) if ratios else 0.0

    def detect_potential_contamination(
        self,
        documents: List[str],
        benchmark_texts: List[str],
        n: int = 10,
        threshold: float = 0.7,
    ) -> List[Tuple[int, int, float]]:
        """
        检测潜在的数据污染

        检查训练文档与 benchmark 文本之间的 n-gram 重叠。

        Args:
            documents: 训练文档列表
            benchmark_texts: 评估数据集文本列表
            n: n-gram 阶数
            threshold: 重叠比例阈值

        Returns:
            [(doc_index, benchmark_index, overlap_ratio), ...] 列表
        """
        # 构建训练集的 n-gram 索引
        doc_ngram_sets = []
        for doc in documents:
            words = doc.lower().split()
            ngrams = set()
            for i in range(len(words) - n + 1):
                ngrams.add(tuple(words[i:i + n]))
            doc_ngram_sets.append(ngrams)

        contaminated = []
        for bench_idx, bench_text in enumerate(benchmark_texts):
            bench_words = bench_text.lower().split()
            bench_ngrams = []
            for i in range(len(bench_words) - n + 1):
                bench_ngrams.append(tuple(bench_words[i:i + n]))

            if not bench_ngrams:
                continue

            for doc_idx, doc_ngrams in enumerate(doc_ngram_sets):
                matched = sum(
                    1 for ng in bench_ngrams if ng in doc_ngrams
                )
                overlap = matched / len(bench_ngrams)
                if overlap >= threshold:
                    contaminated.append((doc_idx, bench_idx, overlap))

        return sorted(contaminated, key=lambda x: -x[2])

    def print_report(self, stats: DatasetStatistics) -> None:
        """
        打印数据集分析报告

        Args:
            stats: DatasetStatistics 统计结果
        """
        print("\n" + "=" * 60)
        print("数据集分析报告")
        print("=" * 60)

        print(f"\n--- 基本统计 ---")
        print(f"  文档总数:       {stats.num_documents:,}")
        print(f"  总字符数:       {stats.total_chars:,}")
        print(f"  总词数:         {stats.total_words:,}")
        print(f"  词汇量:         {stats.vocab_size:,}")

        if self.estimate_tokens:
            est_tokens = int(stats.total_words * self.token_ratio)
            print(f"  估计 token 数:  ~{est_tokens:,} "
                  f"(词数 x {self.token_ratio})")

        print(f"\n--- 文档长度分布 (词) ---")
        print(f"  最短:           {stats.min_doc_length_words:,}")
        print(f"  平均:           {stats.avg_doc_length_words:,.1f}")
        print(f"  中位数:         {stats.median_doc_length_words:,.1f}")
        print(f"  最长:           {stats.max_doc_length_words:,}")

        print(f"\n--- 词汇多样性 ---")
        print(f"  Type-Token Ratio: {stats.type_token_ratio:.4f}")

        if stats.source_distribution:
            print(f"\n--- 数据源分布 ---")
            total = sum(stats.source_distribution.values())
            for source, count in sorted(
                stats.source_distribution.items(),
                key=lambda x: -x[1],
            ):
                pct = count / total * 100
                bar = '#' * int(pct / 2)
                print(f"  {source:<15} {count:>6} ({pct:5.1f}%) {bar}")

        if stats.length_histogram:
            print(f"\n--- 长度分布直方图 ---")
            max_count = max(stats.length_histogram.values())
            for bin_label, count in stats.length_histogram.items():
                bar_len = int(count / max_count * 30) if max_count > 0 else 0
                bar = '#' * bar_len
                print(f"  {bin_label:<15} {count:>5} {bar}")


class QualityProfiler:
    """
    数据质量分析器

    对数据集进行质量维度的分析，生成质量画像。

    分析维度:
    - 文本流畅度（简化的 perplexity 估计）
    - 重复度
    - 格式质量
    - 信息密度

    Args:
        reference_texts: 参考高质量文本（用于对比）
    """

    def __init__(self, reference_texts: Optional[List[str]] = None):
        self.reference_texts = reference_texts

    def profile(
        self, documents: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        生成数据质量画像

        Args:
            documents: 文档列表

        Returns:
            各维度的统计摘要
        """
        profile = {}

        # 维度 1: 长度分布质量
        lengths = [len(doc.split()) for doc in documents]
        profile['length'] = {
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'std': self._std(lengths),
            'short_ratio': sum(1 for l in lengths if l < 50) / len(lengths) if lengths else 0,
            'long_ratio': sum(1 for l in lengths if l > 5000) / len(lengths) if lengths else 0,
        }

        # 维度 2: 词汇质量
        all_words = Counter()
        doc_ttrs = []
        for doc in documents:
            words = doc.lower().split()
            if words:
                all_words.update(words)
                doc_ttrs.append(len(set(words)) / len(words))
        profile['vocabulary'] = {
            'avg_ttr': sum(doc_ttrs) / len(doc_ttrs) if doc_ttrs else 0,
            'global_ttr': len(all_words) / sum(all_words.values()) if all_words else 0,
        }

        # 维度 3: 重复度
        dup_ratios = []
        for doc in documents:
            lines = [l.strip() for l in doc.split('\n') if l.strip()]
            if lines:
                unique = len(set(lines))
                dup_ratios.append(1.0 - unique / len(lines))
        profile['repetition'] = {
            'avg_line_dup_ratio': sum(dup_ratios) / len(dup_ratios) if dup_ratios else 0,
            'high_dup_docs': sum(1 for r in dup_ratios if r > 0.3) / len(dup_ratios) if dup_ratios else 0,
        }

        # 维度 4: 格式质量
        special_ratios = []
        for doc in documents:
            if doc:
                special = sum(1 for c in doc if not c.isalnum() and not c.isspace())
                special_ratios.append(special / len(doc))
        profile['format'] = {
            'avg_special_ratio': sum(special_ratios) / len(special_ratios) if special_ratios else 0,
            'noisy_docs': sum(1 for r in special_ratios if r > 0.3) / len(special_ratios) if special_ratios else 0,
        }

        return profile

    def _std(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def print_profile(
        self, profile: Dict[str, Dict[str, float]]
    ) -> None:
        """打印质量画像"""
        print("\n" + "=" * 50)
        print("数据质量画像")
        print("=" * 50)

        for dimension, metrics in profile.items():
            print(f"\n  [{dimension}]")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")


if __name__ == '__main__':
    print("=" * 60)
    print("数据集统计分析工具 演示")
    print("=" * 60)

    # 模拟数据集
    documents = [
        "Machine learning is a branch of artificial intelligence that "
        "focuses on building systems that learn from data. "
        "These systems improve performance over time.",

        "Natural language processing enables computers to understand "
        "and generate human language. It has applications in translation "
        "and sentiment analysis.",

        "Deep learning uses neural networks with multiple layers to "
        "extract hierarchical features. Convolutional networks are used "
        "for image recognition. Recurrent networks handle sequences.",

        "Reinforcement learning trains agents through trial and error. "
        "The agent learns to maximize cumulative reward by interacting "
        "with an environment.",

        "The transformer architecture introduced self-attention mechanisms "
        "that revolutionized NLP. BERT and GPT are based on transformers.",

        "Computer vision deals with image and video understanding. "
        "Object detection, segmentation, and recognition are key tasks.",

        "Transfer learning allows knowledge from one task to be applied "
        "to another. Pre-trained models like BERT and GPT demonstrate "
        "the power of transfer learning in NLP.",

        "Generative adversarial networks consist of a generator and "
        "discriminator trained together. GANs can produce realistic "
        "images and other synthetic data.",
    ]

    sources = ["web", "wiki", "books", "web", "wiki", "web", "books", "web"]

    # 1. 基础统计分析
    print("\n--- 基础统计分析 ---")
    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze(documents, sources)
    analyzer.print_report(stats)

    # 2. n-gram 重复分析
    print("\n--- N-gram 重复分析 ---")
    ngram_stats = analyzer.compute_ngram_repetition(documents, n=3)
    print(f"  总 3-gram 数: {ngram_stats['total_ngrams']}")
    print(f"  唯一 3-gram 数: {ngram_stats['unique_ngrams']}")
    print(f"  重复率: {ngram_stats['repetition_rate']:.4f}")
    print(f"  Top-10 占比: {ngram_stats['top_repeated_ratio']:.4f}")
    if ngram_stats.get('most_common'):
        print("  最常见 3-gram:")
        for ngram, count in ngram_stats['most_common'][:5]:
            print(f"    '{ngram}': {count}")

    # 3. 词汇分析
    print("\n--- 词汇分析 ---")
    vocab_stats = analyzer.compute_vocabulary_analysis(documents, top_k=10)
    print(f"  词汇量: {vocab_stats['vocab_size']}")
    print(f"  Hapax legomena: {vocab_stats['hapax_legomena']} "
          f"({vocab_stats['hapax_ratio']:.2%})")
    print(f"  Top-10 覆盖率: {vocab_stats['top_k_coverage']:.2%}")
    print("  最高频词:")
    for word, count in vocab_stats['top_words'][:5]:
        print(f"    '{word}': {count}")

    # 4. 压缩比
    print("\n--- 信息密度 ---")
    comp_ratio = analyzer.compute_compression_ratio(documents)
    print(f"  平均压缩比: {comp_ratio:.4f}")
    print(f"  (越低表示信息密度越高)")

    # 5. 数据污染检测
    print("\n--- 数据污染检测 ---")
    benchmark = [
        "Machine learning is a branch of artificial intelligence that "
        "focuses on building systems that learn from data."
    ]
    contaminated = analyzer.detect_potential_contamination(
        documents, benchmark, n=5, threshold=0.5
    )
    if contaminated:
        for doc_idx, bench_idx, overlap in contaminated:
            print(f"  文档 {doc_idx} 与 benchmark {bench_idx} "
                  f"重叠: {overlap:.2%}")
    else:
        print("  未检测到显著污染")

    # 6. 质量画像
    print("\n--- 数据质量画像 ---")
    profiler = QualityProfiler()
    profile = profiler.profile(documents)
    profiler.print_profile(profile)
