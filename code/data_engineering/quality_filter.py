"""
文本质量过滤器

本模块实现了多种预训练数据质量过滤方法，包括:
- 基于启发式规则的过滤（长度、字符比例、重复度等）
- 基于语言模型 perplexity 的质量评分
- 有害内容和 PII（个人隐私信息）过滤
- 综合质量评分

这些过滤器通常串联使用，形成一条多层次的质量过滤管线:
规则过滤（快速粗筛）-> 模型过滤（perplexity）-> 有害内容过滤（安全）

参考:
- Wenzek et al. (2020). CCNet: Extracting High Quality Monolingual Datasets.
- Raffel et al. (2020). Exploring the Limits of Transfer Learning (C4).
"""

import re
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class FilterResult:
    """
    过滤结果

    记录文档是否通过过滤以及各项评分指标。

    Attributes:
        passed: 是否通过过滤
        score: 综合质量分数 [0, 1]
        reason: 如果被过滤，给出原因
        metrics: 各项指标的详细值
    """
    passed: bool
    score: float = 0.0
    reason: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)


class RuleBasedFilter:
    """
    基于启发式规则的质量过滤器

    使用多条统计规则快速过滤明显低质量的文档。
    这是过滤管线中的第一层，计算代价最低。

    规则维度:
    - 文本长度（过短/过长）
    - 特殊字符比例
    - 数字字符比例
    - 平均行长度
    - 重复行比例
    - 重复 n-gram 比例
    - 停用词密度

    Args:
        min_length: 最短文本长度（字符数）
        max_length: 最长文本长度（字符数）
        max_special_ratio: 最大特殊字符比例
        max_digit_ratio: 最大数字字符比例
        min_avg_line_length: 最小平均行长度
        max_avg_line_length: 最大平均行长度
        max_duplicate_line_ratio: 最大重复行比例
        max_duplicate_ngram_ratio: 最大重复 n-gram 比例
        min_stopword_ratio: 最小停用词比例
    """

    # 英文停用词列表（精简版）
    ENGLISH_STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both', 'either',
        'neither', 'each', 'every', 'all', 'any', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'only', 'own', 'same', 'than',
        'too', 'very', 'just', 'because', 'if', 'when', 'where', 'how',
        'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
        'she', 'her', 'it', 'its', 'they', 'them', 'their',
    }

    # 中文停用词列表（精简版）
    CHINESE_STOPWORDS = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
        '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你',
        '会', '着', '没有', '看', '好', '自己', '这', '他', '她', '它',
        '们', '那', '些', '什么', '怎么', '如何', '可以', '因为', '所以',
    }

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 100000,
        max_special_ratio: float = 0.3,
        max_digit_ratio: float = 0.5,
        min_avg_line_length: float = 10.0,
        max_avg_line_length: float = 500.0,
        max_duplicate_line_ratio: float = 0.5,
        max_duplicate_ngram_ratio: float = 0.3,
        min_stopword_ratio: float = 0.0,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.max_special_ratio = max_special_ratio
        self.max_digit_ratio = max_digit_ratio
        self.min_avg_line_length = min_avg_line_length
        self.max_avg_line_length = max_avg_line_length
        self.max_duplicate_line_ratio = max_duplicate_line_ratio
        self.max_duplicate_ngram_ratio = max_duplicate_ngram_ratio
        self.min_stopword_ratio = min_stopword_ratio

    def _compute_special_char_ratio(self, text: str) -> float:
        """计算特殊字符比例（非字母数字且非空白）"""
        if not text:
            return 0.0
        special = sum(
            1 for c in text
            if not c.isalnum() and not c.isspace()
        )
        return special / len(text)

    def _compute_digit_ratio(self, text: str) -> float:
        """计算数字字符比例"""
        if not text:
            return 0.0
        digits = sum(1 for c in text if c.isdigit())
        return digits / len(text)

    def _compute_avg_line_length(self, text: str) -> float:
        """计算平均行长度"""
        lines = [l for l in text.split('\n') if l.strip()]
        if not lines:
            return 0.0
        return sum(len(l) for l in lines) / len(lines)

    def _compute_duplicate_line_ratio(self, text: str) -> float:
        """计算重复行比例"""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        if not lines:
            return 0.0
        unique_lines = set(lines)
        return 1.0 - len(unique_lines) / len(lines)

    def _compute_duplicate_ngram_ratio(
        self, text: str, n: int = 5
    ) -> float:
        """
        计算重复 n-gram 比例

        重复 n-gram 比例高通常意味着广告、SEO 或模板文本。
        """
        words = text.lower().split()
        if len(words) < n:
            return 0.0

        ngrams = [
            tuple(words[i:i + n]) for i in range(len(words) - n + 1)
        ]
        counter = Counter(ngrams)

        # 重复 n-gram 的数量（出现次数 > 1 的 n-gram 总出现次数）
        duplicate_count = sum(
            count for count in counter.values() if count > 1
        )
        return duplicate_count / len(ngrams) if ngrams else 0.0

    def _compute_stopword_ratio(self, text: str) -> float:
        """计算停用词比例"""
        words = text.lower().split()
        if not words:
            return 0.0
        all_stopwords = self.ENGLISH_STOPWORDS | self.CHINESE_STOPWORDS
        stopword_count = sum(1 for w in words if w in all_stopwords)
        return stopword_count / len(words)

    def filter(self, text: str) -> FilterResult:
        """
        对文本应用规则过滤

        Args:
            text: 输入文本

        Returns:
            FilterResult 包含是否通过及各项指标
        """
        metrics = {}

        # 规则 1: 长度检查
        text_length = len(text)
        metrics['length'] = text_length
        if text_length < self.min_length:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"文本太短: {text_length} < {self.min_length}",
                metrics=metrics,
            )
        if text_length > self.max_length:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"文本太长: {text_length} > {self.max_length}",
                metrics=metrics,
            )

        # 规则 2: 特殊字符比例
        special_ratio = self._compute_special_char_ratio(text)
        metrics['special_char_ratio'] = special_ratio
        if special_ratio > self.max_special_ratio:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"特殊字符过多: {special_ratio:.2%} > "
                       f"{self.max_special_ratio:.2%}",
                metrics=metrics,
            )

        # 规则 3: 数字比例
        digit_ratio = self._compute_digit_ratio(text)
        metrics['digit_ratio'] = digit_ratio
        if digit_ratio > self.max_digit_ratio:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"数字过多: {digit_ratio:.2%} > "
                       f"{self.max_digit_ratio:.2%}",
                metrics=metrics,
            )

        # 规则 4: 平均行长度
        avg_line_len = self._compute_avg_line_length(text)
        metrics['avg_line_length'] = avg_line_len
        if avg_line_len < self.min_avg_line_length:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"平均行太短: {avg_line_len:.1f} < "
                       f"{self.min_avg_line_length}",
                metrics=metrics,
            )
        if avg_line_len > self.max_avg_line_length:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"平均行太长: {avg_line_len:.1f} > "
                       f"{self.max_avg_line_length}",
                metrics=metrics,
            )

        # 规则 5: 重复行比例
        dup_line_ratio = self._compute_duplicate_line_ratio(text)
        metrics['duplicate_line_ratio'] = dup_line_ratio
        if dup_line_ratio > self.max_duplicate_line_ratio:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"重复行过多: {dup_line_ratio:.2%} > "
                       f"{self.max_duplicate_line_ratio:.2%}",
                metrics=metrics,
            )

        # 规则 6: 重复 n-gram 比例
        dup_ngram_ratio = self._compute_duplicate_ngram_ratio(text)
        metrics['duplicate_ngram_ratio'] = dup_ngram_ratio
        if dup_ngram_ratio > self.max_duplicate_ngram_ratio:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"重复 n-gram 过多: {dup_ngram_ratio:.2%} > "
                       f"{self.max_duplicate_ngram_ratio:.2%}",
                metrics=metrics,
            )

        # 规则 7: 停用词比例
        stopword_ratio = self._compute_stopword_ratio(text)
        metrics['stopword_ratio'] = stopword_ratio
        if stopword_ratio < self.min_stopword_ratio:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"停用词过少: {stopword_ratio:.2%} < "
                       f"{self.min_stopword_ratio:.2%}",
                metrics=metrics,
            )

        # 通过所有规则，计算综合分数
        score = self._compute_score(metrics)
        metrics['quality_score'] = score

        return FilterResult(
            passed=True, score=score, reason="通过", metrics=metrics
        )

    def _compute_score(self, metrics: Dict[str, float]) -> float:
        """
        基于各项指标计算综合质量分数

        分数越高表示质量越好，范围 [0, 1]。
        """
        score = 1.0

        # 特殊字符越少越好
        score -= metrics.get('special_char_ratio', 0) * 0.5

        # 数字比例适中为好
        digit_penalty = max(0, metrics.get('digit_ratio', 0) - 0.1) * 0.5
        score -= digit_penalty

        # 重复越少越好
        score -= metrics.get('duplicate_line_ratio', 0) * 0.3
        score -= metrics.get('duplicate_ngram_ratio', 0) * 0.3

        # 停用词比例适中为好（表示自然语言）
        sw_ratio = metrics.get('stopword_ratio', 0)
        if sw_ratio > 0.05:
            score += 0.1  # 有合理的停用词密度，加分

        return max(0.0, min(1.0, score))


class PIIFilter:
    """
    个人隐私信息（PII）过滤器

    检测并移除/替换文本中的个人隐私信息，包括:
    - 邮箱地址
    - 电话号码（中国大陆手机号、固定电话）
    - IP 地址
    - 身份证号
    - 银行卡号

    Args:
        mode: 'remove' 删除 PII，'replace' 替换为占位符
    """

    # PII 正则模式
    PATTERNS = {
        'email': (
            r'\b[\w.+-]+@[\w-]+\.[\w.-]+\b',
            '[EMAIL]'
        ),
        'phone_cn_mobile': (
            r'\b1[3-9]\d{9}\b',
            '[PHONE]'
        ),
        'phone_cn_landline': (
            r'\b0\d{2,3}-?\d{7,8}\b',
            '[PHONE]'
        ),
        'ip_address': (
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            '[IP]'
        ),
        'id_card_cn': (
            r'\b\d{17}[\dXx]\b',
            '[ID_CARD]'
        ),
        'bank_card': (
            r'\b\d{16,19}\b',
            '[BANK_CARD]'
        ),
    }

    def __init__(self, mode: str = 'replace'):
        assert mode in ('remove', 'replace')
        self.mode = mode
        # 编译正则表达式
        self._compiled = {
            name: (re.compile(pattern), placeholder)
            for name, (pattern, placeholder) in self.PATTERNS.items()
        }

    def detect(self, text: str) -> Dict[str, List[str]]:
        """
        检测文本中的 PII

        Args:
            text: 输入文本

        Returns:
            {pii_type: [匹配到的内容列表]}
        """
        found = {}
        for name, (regex, _) in self._compiled.items():
            matches = regex.findall(text)
            if matches:
                found[name] = matches
        return found

    def filter(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        过滤文本中的 PII

        Args:
            text: 输入文本

        Returns:
            (处理后的文本, {pii_type: 替换次数})
        """
        stats = {}
        for name, (regex, placeholder) in self._compiled.items():
            if self.mode == 'replace':
                new_text, count = regex.subn(placeholder, text)
            else:
                new_text, count = regex.subn('', text)

            if count > 0:
                stats[name] = count
                text = new_text

        return text, stats


class PerplexityFilter:
    """
    基于 Perplexity 的质量过滤器

    使用简单的字符级或词级统计模型近似计算 perplexity，
    用于评估文档的"自然语言程度"。

    注意: 工业级实现通常使用 KenLM 或预训练的 Transformer 模型。
    本实现使用简化的字符 n-gram 模型作为演示。

    PPL(x) = exp(-1/T * sum(log P(x_t | x_{<t})))

    - 低 PPL: 文本流畅、自然
    - 高 PPL: 文本混乱、可能是乱码
    - 极低 PPL: 文本过于简单或重复

    Args:
        min_ppl: 最小 perplexity（过低说明文本过于简单）
        max_ppl: 最大 perplexity（过高说明文本质量差）
        ngram_order: n-gram 阶数
    """

    def __init__(
        self,
        min_ppl: float = 10.0,
        max_ppl: float = 1000.0,
        ngram_order: int = 3,
    ):
        self.min_ppl = min_ppl
        self.max_ppl = max_ppl
        self.ngram_order = ngram_order
        self._ngram_counts: Optional[Counter] = None
        self._context_counts: Optional[Counter] = None
        self._vocab_size: int = 0

    def fit(self, texts: List[str]) -> None:
        """
        在参考文本上训练 n-gram 语言模型

        统计字符级 n-gram 频率，用于后续计算 perplexity。

        Args:
            texts: 参考文本列表（应为高质量文本）
        """
        self._ngram_counts = Counter()
        self._context_counts = Counter()
        vocab = set()

        for text in texts:
            chars = list(text.lower())
            vocab.update(chars)

            for i in range(len(chars) - self.ngram_order + 1):
                ngram = tuple(chars[i:i + self.ngram_order])
                context = ngram[:-1]
                self._ngram_counts[ngram] += 1
                self._context_counts[context] += 1

        self._vocab_size = len(vocab)

    def compute_perplexity(self, text: str) -> float:
        """
        计算文本的 perplexity

        PPL = exp(-1/N * sum(log P(c_i | c_{i-n+1}...c_{i-1})))

        使用拉普拉斯平滑（add-1 smoothing）处理未见 n-gram。

        Args:
            text: 输入文本

        Returns:
            perplexity 值
        """
        if self._ngram_counts is None:
            raise RuntimeError("请先调用 fit() 训练模型")

        chars = list(text.lower())
        n = self.ngram_order

        if len(chars) < n:
            return float('inf')

        total_log_prob = 0.0
        count = 0

        for i in range(len(chars) - n + 1):
            ngram = tuple(chars[i:i + n])
            context = ngram[:-1]

            # 拉普拉斯平滑
            ngram_count = self._ngram_counts.get(ngram, 0) + 1
            context_count = self._context_counts.get(context, 0) + self._vocab_size

            prob = ngram_count / context_count
            total_log_prob += math.log(prob)
            count += 1

        if count == 0:
            return float('inf')

        avg_log_prob = total_log_prob / count
        return math.exp(-avg_log_prob)

    def filter(self, text: str) -> FilterResult:
        """
        基于 perplexity 过滤文档

        Args:
            text: 输入文本

        Returns:
            FilterResult 包含是否通过及 perplexity 值
        """
        ppl = self.compute_perplexity(text)
        metrics = {'perplexity': ppl}

        if ppl < self.min_ppl:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"Perplexity 过低 ({ppl:.1f} < {self.min_ppl}): "
                       f"文本可能过于简单或重复",
                metrics=metrics,
            )

        if ppl > self.max_ppl:
            return FilterResult(
                passed=False, score=0.0,
                reason=f"Perplexity 过高 ({ppl:.1f} > {self.max_ppl}): "
                       f"文本可能是乱码或非自然语言",
                metrics=metrics,
            )

        # 计算分数: PPL 越接近理想范围中心越好
        log_ppl = math.log(ppl)
        log_min = math.log(self.min_ppl)
        log_max = math.log(self.max_ppl)
        log_mid = (log_min + log_max) / 2
        log_range = (log_max - log_min) / 2

        # 分数: 偏离中心越远越低
        deviation = abs(log_ppl - log_mid) / log_range
        score = max(0.0, 1.0 - deviation)

        return FilterResult(
            passed=True, score=score,
            reason="通过", metrics=metrics,
        )


class CompositeFilter:
    """
    组合过滤器

    将多个过滤器串联，文档需要通过所有过滤器才能被保留。

    Args:
        filters: 过滤器列表，按顺序执行
        weights: 各过滤器在综合评分中的权重
    """

    def __init__(
        self,
        filters: Optional[List] = None,
        weights: Optional[List[float]] = None,
    ):
        self.filters = filters or []
        self.weights = weights or [1.0] * len(self.filters)
        assert len(self.filters) == len(self.weights)

    def add_filter(self, f, weight: float = 1.0) -> None:
        """添加一个过滤器"""
        self.filters.append(f)
        self.weights.append(weight)

    def filter(self, text: str) -> FilterResult:
        """
        对文本应用所有过滤器

        Args:
            text: 输入文本

        Returns:
            综合的 FilterResult
        """
        all_metrics = {}
        total_score = 0.0
        total_weight = 0.0

        for filt, weight in zip(self.filters, self.weights):
            result = filt.filter(text)
            all_metrics.update(result.metrics)

            if not result.passed:
                return FilterResult(
                    passed=False, score=0.0,
                    reason=f"[{filt.__class__.__name__}] {result.reason}",
                    metrics=all_metrics,
                )

            total_score += result.score * weight
            total_weight += weight

        avg_score = total_score / total_weight if total_weight > 0 else 0.0

        return FilterResult(
            passed=True, score=avg_score,
            reason="通过所有过滤器", metrics=all_metrics,
        )


if __name__ == '__main__':
    print("=" * 60)
    print("文本质量过滤器 演示")
    print("=" * 60)

    # 测试文档
    test_docs = {
        "高质量文章": (
            "Machine learning is a branch of artificial intelligence that "
            "focuses on building systems that can learn from data. "
            "These systems improve their performance over time without being "
            "explicitly programmed. Deep learning, a subset of machine learning, "
            "uses neural networks with many layers to model complex patterns. "
            "Recent advances in transformer architectures have revolutionized "
            "natural language processing and computer vision."
        ),
        "低质量-太短": "Hello world",
        "低质量-特殊字符过多": "!@#$%^&*()_+" * 20 + " some text here",
        "低质量-重复内容": "Buy now! Best price! " * 30,
        "包含PII": (
            "Please contact John at john.doe@example.com or call "
            "13800138000 for more information. His ID is 110101199001011234."
        ),
        "极简单文本": "the the the the the the the " * 20,
    }

    # 1. 规则过滤
    print("\n--- 规则过滤 ---")
    rule_filter = RuleBasedFilter(min_length=50, max_special_ratio=0.3)
    for name, text in test_docs.items():
        result = rule_filter.filter(text)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {name}: score={result.score:.2f}, "
              f"reason={result.reason}")

    # 2. PII 过滤
    print("\n--- PII 检测与过滤 ---")
    pii_filter = PIIFilter(mode='replace')
    pii_text = test_docs["包含PII"]
    print(f"  原文: {pii_text}")
    detected = pii_filter.detect(pii_text)
    print(f"  检测到: {detected}")
    cleaned, stats = pii_filter.filter(pii_text)
    print(f"  清洗后: {cleaned}")
    print(f"  替换统计: {stats}")

    # 3. Perplexity 过滤
    print("\n--- Perplexity 过滤 ---")
    # 用高质量文本训练参考模型
    reference_texts = [
        "Natural language processing is a subfield of linguistics and "
        "computer science concerned with the interactions between computers "
        "and human language.",
        "Deep learning algorithms use multiple layers to progressively "
        "extract higher level features from raw input.",
        "The transformer architecture has become the foundation for most "
        "modern language models.",
    ]

    ppl_filter = PerplexityFilter(min_ppl=5.0, max_ppl=500.0)
    ppl_filter.fit(reference_texts)

    for name, text in test_docs.items():
        if len(text) < 20:
            continue
        result = ppl_filter.filter(text)
        ppl_val = result.metrics.get('perplexity', float('inf'))
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {name}: PPL={ppl_val:.1f}, "
              f"score={result.score:.2f}")

    # 4. 组合过滤
    print("\n--- 组合过滤 ---")
    composite = CompositeFilter(
        filters=[rule_filter, ppl_filter],
        weights=[0.6, 0.4],
    )
    for name, text in test_docs.items():
        result = composite.filter(text)
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {name}: score={result.score:.2f}, "
              f"reason={result.reason}")
