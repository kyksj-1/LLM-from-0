"""
端到端数据处理管线

本模块实现了一条完整的预训练数据处理管线，将原始数据
经过多个处理阶段转化为可用于语言模型训练的高质量数据。

管线阶段:
1. 文本提取: HTML -> 纯文本
2. Unicode 规范化: 统一编码
3. 规则过滤: 基于启发式规则的快速过滤
4. PII 移除: 个人隐私信息脱敏
5. 精确去重: SHA-256 哈希去重
6. 近似去重: MinHash + LSH
7. 质量评分: 综合质量打分
8. 数据混合: 多源数据按比例采样

设计原则:
- 模块化: 每个阶段独立可配置
- 可扩展: 支持自定义处理阶段
- 可追踪: 记录每个阶段的处理统计

参考:
- Penedo et al. (2024). The FineWeb Datasets.
- HuggingFace datatrove 框架设计理念
"""

import hashlib
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# 导入本项目的其他模块
from text_extraction import HTMLTextExtractor, TextCleaner, UnicodeNormalizer
from quality_filter import RuleBasedFilter, PIIFilter, FilterResult
from deduplication import exact_dedup, MinHashLSHDeduplicator


@dataclass
class Document:
    """
    文档数据结构

    在管线中流转的基本单元，携带文本内容和元数据。

    Attributes:
        doc_id: 唯一标识符
        text: 文档文本
        source: 数据来源
        metadata: 额外元数据（质量分数、过滤原因等）
    """
    doc_id: int
    text: str
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StageStats:
    """
    处理阶段统计

    Attributes:
        stage_name: 阶段名称
        input_count: 输入文档数
        output_count: 输出文档数
        filtered_count: 被过滤的文档数
        elapsed_seconds: 耗时（秒）
        filter_reasons: 过滤原因统计
    """
    stage_name: str
    input_count: int = 0
    output_count: int = 0
    filtered_count: int = 0
    elapsed_seconds: float = 0.0
    filter_reasons: Dict[str, int] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        """通过率"""
        return (
            self.output_count / self.input_count
            if self.input_count > 0 else 0.0
        )


class PipelineStage:
    """
    管线处理阶段的基类

    每个处理阶段需要实现 process 方法，
    接受一个文档列表，返回处理后的文档列表。

    Args:
        name: 阶段名称
    """

    def __init__(self, name: str):
        self.name = name

    def process(self, documents: List[Document]) -> List[Document]:
        """
        处理文档列表

        Args:
            documents: 输入文档列表

        Returns:
            处理后的文档列表
        """
        raise NotImplementedError


class TextExtractionStage(PipelineStage):
    """
    文本提取阶段

    从 HTML 文本中提取纯文本内容。
    如果文档已经是纯文本，则跳过提取，只做基本清洗。

    Args:
        min_paragraph_length: 最短段落长度
        min_output_length: 提取后的最短文本长度
    """

    def __init__(
        self,
        min_paragraph_length: int = 20,
        min_output_length: int = 100,
    ):
        super().__init__("文本提取")
        self._extractor = HTMLTextExtractor(
            min_paragraph_length=min_paragraph_length
        )
        self._cleaner = TextCleaner(remove_urls=True)
        self.min_output_length = min_output_length

    def process(self, documents: List[Document]) -> List[Document]:
        """处理文档: 提取文本并清洗"""
        result = []
        for doc in documents:
            text = doc.text

            # 检测是否为 HTML（简单启发式）
            if '<html' in text.lower() or '<body' in text.lower() or '<p>' in text.lower():
                extraction = self._extractor.extract(text)
                text = extraction.text
                doc.metadata['extraction_ratio'] = extraction.extraction_ratio
            else:
                text = self._cleaner.clean(text)

            if len(text) >= self.min_output_length:
                doc.text = text
                result.append(doc)

        return result


class UnicodeNormalizationStage(PipelineStage):
    """
    Unicode 规范化阶段

    统一文本编码，处理全角/半角、零宽字符等。
    """

    def __init__(self):
        super().__init__("Unicode 规范化")
        self._normalizer = UnicodeNormalizer()

    def process(self, documents: List[Document]) -> List[Document]:
        """处理文档: Unicode 规范化"""
        for doc in documents:
            doc.text = self._normalizer.normalize(doc.text)
        return documents


class RuleFilterStage(PipelineStage):
    """
    规则过滤阶段

    基于启发式规则快速过滤低质量文档。

    Args:
        min_length: 最短文本长度
        max_length: 最长文本长度
        max_special_ratio: 最大特殊字符比例
    """

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 100000,
        max_special_ratio: float = 0.3,
    ):
        super().__init__("规则过滤")
        self._filter = RuleBasedFilter(
            min_length=min_length,
            max_length=max_length,
            max_special_ratio=max_special_ratio,
        )

    def process(self, documents: List[Document]) -> List[Document]:
        """处理文档: 规则过滤"""
        result = []
        for doc in documents:
            filter_result = self._filter.filter(doc.text)
            doc.metadata['rule_filter_score'] = filter_result.score
            doc.metadata['rule_filter_passed'] = filter_result.passed
            if not filter_result.passed:
                doc.metadata['filter_reason'] = filter_result.reason
            if filter_result.passed:
                result.append(doc)
        return result


class PIIRemovalStage(PipelineStage):
    """
    PII 移除阶段

    检测并替换文本中的个人隐私信息。

    Args:
        mode: 'replace' 替换为占位符，'remove' 直接移除
    """

    def __init__(self, mode: str = 'replace'):
        super().__init__("PII 移除")
        self._filter = PIIFilter(mode=mode)

    def process(self, documents: List[Document]) -> List[Document]:
        """处理文档: PII 移除"""
        for doc in documents:
            cleaned_text, stats = self._filter.filter(doc.text)
            doc.text = cleaned_text
            if stats:
                doc.metadata['pii_removed'] = stats
        return documents


class ExactDeduplicationStage(PipelineStage):
    """
    精确去重阶段

    基于 SHA-256 哈希值移除完全相同的文档。
    """

    def __init__(self):
        super().__init__("精确去重")

    def process(self, documents: List[Document]) -> List[Document]:
        """处理文档: 精确去重"""
        doc_dict = {doc.doc_id: doc.text for doc in documents}
        kept_ids = exact_dedup(doc_dict)
        kept_set = set(kept_ids)
        return [doc for doc in documents if doc.doc_id in kept_set]


class ApproximateDeduplicationStage(PipelineStage):
    """
    近似去重阶段

    使用 MinHash + LSH 算法检测和移除近似重复的文档。

    Args:
        num_hashes: MinHash 签名长度
        num_bands: LSH 带数
        threshold: Jaccard 相似度阈值
        n_gram: n-gram 的 n 值
    """

    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        threshold: float = 0.8,
        n_gram: int = 5,
    ):
        super().__init__("近似去重")
        self.num_hashes = num_hashes
        self.num_bands = num_bands
        self.threshold = threshold
        self.n_gram = n_gram

    def process(self, documents: List[Document]) -> List[Document]:
        """处理文档: MinHash + LSH 近似去重"""
        if len(documents) < 2:
            return documents

        doc_dict = {doc.doc_id: doc.text for doc in documents}

        deduplicator = MinHashLSHDeduplicator(
            num_hashes=self.num_hashes,
            num_bands=self.num_bands,
            threshold=self.threshold,
            n_gram=self.n_gram,
        )
        kept_ids = deduplicator.deduplicate(doc_dict, verbose=False)
        kept_set = set(kept_ids)

        return [doc for doc in documents if doc.doc_id in kept_set]


class QualityScoringStage(PipelineStage):
    """
    质量评分阶段

    为每个文档计算综合质量分数，过滤低于阈值的文档。

    质量分数基于多个维度:
    - 文本长度（适中为佳）
    - 词汇多样性
    - 段落结构
    - 之前阶段的评分

    Args:
        min_score: 最低质量分数
    """

    def __init__(self, min_score: float = 0.3):
        super().__init__("质量评分")
        self.min_score = min_score

    def _compute_quality_score(self, doc: Document) -> float:
        """计算文档质量分数"""
        text = doc.text
        score = 0.5  # 基础分

        # 维度 1: 文本长度（适中为佳）
        length = len(text)
        if 200 <= length <= 50000:
            score += 0.15
        elif 100 <= length <= 100000:
            score += 0.05

        # 维度 2: 词汇多样性 (Type-Token Ratio)
        words = text.lower().split()
        if words:
            ttr = len(set(words)) / len(words)
            # TTR 0.3-0.7 为好（太高可能是列表，太低可能是重复）
            if 0.3 <= ttr <= 0.8:
                score += 0.15
            elif 0.2 <= ttr <= 0.9:
                score += 0.05

        # 维度 3: 段落结构
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) >= 2:
            score += 0.1

        # 维度 4: 之前阶段的评分（如果有）
        prev_score = doc.metadata.get('rule_filter_score', None)
        if prev_score is not None:
            score = 0.5 * score + 0.5 * prev_score

        return min(1.0, max(0.0, score))

    def process(self, documents: List[Document]) -> List[Document]:
        """处理文档: 质量评分与过滤"""
        result = []
        for doc in documents:
            score = self._compute_quality_score(doc)
            doc.metadata['quality_score'] = score
            if score >= self.min_score:
                result.append(doc)
        return result


class CustomStage(PipelineStage):
    """
    自定义处理阶段

    允许用户通过传入函数来定义自定义的处理逻辑。

    Args:
        name: 阶段名称
        process_fn: 处理函数，接受 List[Document] 返回 List[Document]
    """

    def __init__(
        self,
        name: str,
        process_fn: Callable[[List[Document]], List[Document]],
    ):
        super().__init__(name)
        self._process_fn = process_fn

    def process(self, documents: List[Document]) -> List[Document]:
        """处理文档: 调用自定义函数"""
        return self._process_fn(documents)


class DataPipeline:
    """
    数据处理管线

    将多个处理阶段串联，按顺序对文档执行处理。
    记录每个阶段的统计信息，支持断点续传。

    使用方法:
        pipeline = DataPipeline()
        pipeline.add_stage(TextExtractionStage())
        pipeline.add_stage(RuleFilterStage())
        pipeline.add_stage(ExactDeduplicationStage())
        results = pipeline.run(documents)

    Args:
        verbose: 是否打印处理进度
    """

    def __init__(self, verbose: bool = True):
        self.stages: List[PipelineStage] = []
        self.stats: List[StageStats] = []
        self.verbose = verbose

    def add_stage(self, stage: PipelineStage) -> 'DataPipeline':
        """
        添加处理阶段

        Args:
            stage: PipelineStage 实例

        Returns:
            self（支持链式调用）
        """
        self.stages.append(stage)
        return self

    def run(self, documents: List[Document]) -> List[Document]:
        """
        运行管线

        按顺序执行所有处理阶段，记录统计信息。

        Args:
            documents: 输入文档列表

        Returns:
            处理后的文档列表
        """
        self.stats = []

        if self.verbose:
            print("=" * 60)
            print(f"数据处理管线启动: {len(documents)} 个文档")
            print(f"处理阶段: {len(self.stages)} 个")
            print("=" * 60)

        current_docs = documents

        for i, stage in enumerate(self.stages):
            start_time = time.time()
            input_count = len(current_docs)

            if self.verbose:
                print(f"\n[{i+1}/{len(self.stages)}] {stage.name} "
                      f"(输入: {input_count} 个文档)")

            # 执行处理
            current_docs = stage.process(current_docs)

            elapsed = time.time() - start_time
            output_count = len(current_docs)
            filtered_count = input_count - output_count

            # 记录统计
            stage_stats = StageStats(
                stage_name=stage.name,
                input_count=input_count,
                output_count=output_count,
                filtered_count=filtered_count,
                elapsed_seconds=elapsed,
            )
            self.stats.append(stage_stats)

            if self.verbose:
                print(f"  -> 输出: {output_count} 个文档 "
                      f"(过滤 {filtered_count}, "
                      f"通过率 {stage_stats.pass_rate:.1%}, "
                      f"耗时 {elapsed:.2f}s)")

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"管线完成: {len(documents)} -> {len(current_docs)} 个文档")
            total_time = sum(s.elapsed_seconds for s in self.stats)
            print(f"总耗时: {total_time:.2f}s")
            print("=" * 60)

        return current_docs

    def get_stats_summary(self) -> Dict[str, Any]:
        """
        获取管线执行统计摘要

        Returns:
            包含各阶段统计的字典
        """
        return {
            'total_stages': len(self.stats),
            'total_time': sum(s.elapsed_seconds for s in self.stats),
            'stages': [
                {
                    'name': s.stage_name,
                    'input': s.input_count,
                    'output': s.output_count,
                    'filtered': s.filtered_count,
                    'pass_rate': f"{s.pass_rate:.1%}",
                    'time': f"{s.elapsed_seconds:.2f}s",
                }
                for s in self.stats
            ],
        }

    def print_stats(self) -> None:
        """打印管线统计报告"""
        print("\n管线执行报告")
        print("-" * 70)
        print(f"{'阶段':<20} {'输入':>8} {'输出':>8} "
              f"{'过滤':>8} {'通过率':>8} {'耗时':>8}")
        print("-" * 70)

        for s in self.stats:
            print(f"{s.stage_name:<20} {s.input_count:>8} "
                  f"{s.output_count:>8} {s.filtered_count:>8} "
                  f"{s.pass_rate:>7.1%} {s.elapsed_seconds:>7.2f}s")

        print("-" * 70)
        if self.stats:
            total_input = self.stats[0].input_count
            total_output = self.stats[-1].output_count
            total_time = sum(s.elapsed_seconds for s in self.stats)
            overall_rate = (
                total_output / total_input if total_input > 0 else 0
            )
            print(f"{'总计':<20} {total_input:>8} "
                  f"{total_output:>8} "
                  f"{total_input - total_output:>8} "
                  f"{overall_rate:>7.1%} {total_time:>7.2f}s")


def create_default_pipeline(
    min_length: int = 100,
    dedup_threshold: float = 0.8,
    quality_threshold: float = 0.3,
) -> DataPipeline:
    """
    创建默认的数据处理管线

    包含常用的处理阶段，适合大多数场景。

    Args:
        min_length: 最短文本长度
        dedup_threshold: 去重相似度阈值
        quality_threshold: 最低质量分数

    Returns:
        配置好的 DataPipeline
    """
    pipeline = DataPipeline(verbose=True)
    pipeline.add_stage(TextExtractionStage(min_output_length=min_length))
    pipeline.add_stage(UnicodeNormalizationStage())
    pipeline.add_stage(RuleFilterStage(min_length=min_length))
    pipeline.add_stage(PIIRemovalStage(mode='replace'))
    pipeline.add_stage(ExactDeduplicationStage())
    pipeline.add_stage(ApproximateDeduplicationStage(
        threshold=dedup_threshold
    ))
    pipeline.add_stage(QualityScoringStage(min_score=quality_threshold))

    return pipeline


if __name__ == '__main__':
    print("=" * 60)
    print("端到端数据处理管线 演示")
    print("=" * 60)

    # 构造模拟输入数据（混合 HTML 和纯文本）
    raw_documents = [
        Document(
            doc_id=0,
            text="""
            <html><head><title>AI Research</title></head><body>
            <p>Artificial intelligence has transformed many industries.
            Machine learning models can now process vast amounts of data
            and extract meaningful patterns that would be impossible
            for humans to detect manually.</p>
            <p>Deep learning, a subset of machine learning, uses neural
            networks with multiple layers to learn hierarchical
            representations of data.</p>
            <footer>Copyright 2024. All rights reserved.</footer>
            </body></html>
            """,
            source="web",
        ),
        Document(
            doc_id=1,
            text=(
                "Natural language processing is a field of computer science "
                "that deals with the interaction between computers and human "
                "language. It involves developing algorithms and models that "
                "enable computers to understand, interpret, and generate "
                "human language in a meaningful way. NLP has applications in "
                "machine translation, sentiment analysis, and chatbots."
            ),
            source="wiki",
        ),
        Document(
            doc_id=2,
            text=(
                "Natural language processing is a field of computer science "
                "that deals with the interaction between computers and human "
                "language. It involves developing algorithms and models that "
                "enable computers to understand, interpret, and generate "
                "human language in a meaningful way. NLP has applications in "
                "machine translation, sentiment analysis, and chatbots."
            ),
            source="web",  # 精确重复
        ),
        Document(
            doc_id=3,
            text="Short text",  # 太短
            source="web",
        ),
        Document(
            doc_id=4,
            text="!@#$%^&*()" * 50,  # 特殊字符过多
            source="web",
        ),
        Document(
            doc_id=5,
            text=(
                "Contact me at test@example.com or call 13800138000. "
                "The transformer architecture has fundamentally changed "
                "how we approach sequence modeling tasks. "
                "Attention mechanisms allow models to focus on relevant "
                "parts of the input, enabling better performance on "
                "long-range dependencies."
            ),
            source="wiki",
        ),
        Document(
            doc_id=6,
            text=(
                "Reinforcement learning from human feedback (RLHF) is "
                "a technique used to align language models with human "
                "preferences. It involves training a reward model based "
                "on human comparisons, then using this reward model to "
                "fine-tune the language model using reinforcement learning "
                "algorithms like PPO."
            ),
            source="books",
        ),
    ]

    # 创建并运行默认管线
    pipeline = create_default_pipeline(
        min_length=50,
        dedup_threshold=0.8,
        quality_threshold=0.3,
    )

    results = pipeline.run(raw_documents)

    # 打印统计报告
    pipeline.print_stats()

    # 显示最终结果
    print("\n--- 最终保留的文档 ---")
    for doc in results:
        preview = doc.text[:80].replace('\n', ' ')
        score = doc.metadata.get('quality_score', 'N/A')
        print(f"  ID={doc.doc_id}, source={doc.source}, "
              f"score={score}, text='{preview}...'")
