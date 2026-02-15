"""
推理评估工具

本模块实现了 LLM 推理能力评估的核心工具，包括：
- GSM8K / MATH 格式的答案解析
- 多种答案提取方式
- 准确率计算
- pass@k 指标计算

评估基准:
- GSM8K: 小学数学，精确匹配
- MATH: 高中数学竞赛，标准化答案对比
- HumanEval: 代码生成，pass@k
- MMLU: 多选题，选项匹配

参考:
- Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems.
- Hendrycks et al. (2021). Measuring Massive Multitask Language Understanding.
- Chen et al. (2021). Evaluating Large Language Models Trained on Code.
"""

import re
import math
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class EvalSample:
    """
    评估样本

    Attributes:
        question: 问题文本
        gold_answer: 标准答案
        model_response: 模型的完整输出（含推理过程）
        extracted_answer: 从模型输出中提取的答案
        is_correct: 是否正确
        difficulty: 难度级别（可选）
        category: 类别（可选）
    """
    question: str
    gold_answer: str
    model_response: str = ""
    extracted_answer: str = ""
    is_correct: bool = False
    difficulty: str = ""
    category: str = ""


@dataclass
class EvalResult:
    """
    评估结果

    Attributes:
        accuracy: 总体准确率
        n_total: 总样本数
        n_correct: 正确样本数
        accuracy_by_difficulty: 按难度分组的准确率
        accuracy_by_category: 按类别分组的准确率
        samples: 所有评估样本
    """
    accuracy: float
    n_total: int
    n_correct: int
    accuracy_by_difficulty: Dict[str, float] = field(default_factory=dict)
    accuracy_by_category: Dict[str, float] = field(default_factory=dict)
    samples: List[EvalSample] = field(default_factory=list)


class AnswerExtractor:
    """
    答案提取器

    支持从多种格式的模型输出中提取最终答案：
    - \\boxed{} 格式（MATH 数据集常用）
    - "答案是X" / "The answer is X"
    - "####X" 格式（GSM8K 常用）
    - 选项匹配 A/B/C/D（MMLU 常用）
    - 最后一个数字（兜底策略）
    """

    @staticmethod
    def extract_boxed(response: str) -> Optional[str]:
        """
        提取 \\boxed{} 中的内容

        常用于 MATH 数据集，答案以 \\boxed{42} 形式给出。

        Args:
            response: 模型输出

        Returns:
            提取的答案，若无匹配则返回 None
        """
        # 处理嵌套花括号的情况
        matches = re.findall(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', response)
        if matches:
            return matches[-1].strip()
        return None

    @staticmethod
    def extract_gsm8k(response: str) -> Optional[str]:
        """
        提取 GSM8K 格式的答案

        GSM8K 标准格式: "#### 42" 或推理后跟数字答案

        Args:
            response: 模型输出

        Returns:
            提取的答案
        """
        # 匹配 "#### 数字" 格式
        pattern = r'####\s*(-?\d+[\d,]*\.?\d*)'
        matches = re.findall(pattern, response)
        if matches:
            # 移除逗号（如 1,000 -> 1000）
            return matches[-1].replace(',', '').strip()

        return None

    @staticmethod
    def extract_chinese_answer(response: str) -> Optional[str]:
        """
        提取中文格式的答案

        匹配 "答案是X"、"答案为X"、"答案：X" 等格式

        Args:
            response: 模型输出

        Returns:
            提取的答案
        """
        patterns = [
            r'答案[是为：:]\s*(.+?)(?:[\n。]|$)',
            r'所以[，,]?\s*(.+?)(?:[\n。]|$)',
            r'因此[，,]?\s*(.+?)(?:[\n。]|$)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1].strip()
        return None

    @staticmethod
    def extract_english_answer(response: str) -> Optional[str]:
        """
        提取英文格式的答案

        匹配 "The answer is X"、"Therefore, X" 等格式

        Args:
            response: 模型输出

        Returns:
            提取的答案
        """
        patterns = [
            r'[Tt]he (?:final )?answer is[:\s]*(.+?)[\n.]',
            r'[Tt]herefore[,\s]+(?:the answer is[:\s]*)?(.+?)[\n.]',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1].strip()
        return None

    @staticmethod
    def extract_mcq_option(response: str) -> Optional[str]:
        """
        提取多选题选项

        匹配 A/B/C/D/E 选项（MMLU 格式）

        Args:
            response: 模型输出

        Returns:
            提取的选项字母
        """
        # 匹配明确的选项标记
        patterns = [
            r'(?:答案|answer|选)[是为:\s]*\(?([A-E])\)?',
            r'\b([A-E])\b(?:\s*[\.\)])',  # "A." 或 "A)"
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                return matches[-1].upper()

        # 回退：文本中最后出现的单独字母
        matches = re.findall(r'\b([A-E])\b', response)
        if matches:
            return matches[-1].upper()

        return None

    @staticmethod
    def extract_last_number(response: str) -> Optional[str]:
        """
        提取最后一个数字

        兜底策略：当其他方法都无法提取时使用。

        Args:
            response: 模型输出

        Returns:
            最后一个数字字符串
        """
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]
        return None

    def extract(self, response: str, format_hint: str = "auto") -> str:
        """
        综合答案提取

        按优先级尝试多种提取方法。

        Args:
            response: 模型输出
            format_hint: 格式提示
                - "auto": 自动检测
                - "gsm8k": GSM8K 格式优先
                - "math": \\boxed 格式优先
                - "mcq": 多选题格式
                - "number": 直接提取数字

        Returns:
            提取到的答案字符串
        """
        if format_hint == "gsm8k":
            extractors = [
                self.extract_gsm8k,
                self.extract_chinese_answer,
                self.extract_english_answer,
                self.extract_boxed,
                self.extract_last_number,
            ]
        elif format_hint == "math":
            extractors = [
                self.extract_boxed,
                self.extract_chinese_answer,
                self.extract_english_answer,
                self.extract_last_number,
            ]
        elif format_hint == "mcq":
            extractors = [
                self.extract_mcq_option,
            ]
        elif format_hint == "number":
            extractors = [
                self.extract_last_number,
            ]
        else:  # auto
            extractors = [
                self.extract_boxed,
                self.extract_gsm8k,
                self.extract_chinese_answer,
                self.extract_english_answer,
                self.extract_mcq_option,
                self.extract_last_number,
            ]

        for extractor in extractors:
            result = extractor(response)
            if result is not None:
                return result

        # 最终兜底：返回最后一行
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else ""


class AnswerComparator:
    """
    答案比较器

    处理答案的标准化和比较，支持：
    - 数值比较（考虑精度）
    - 字符串精确匹配
    - 数学表达式等价性（简化版）
    """

    @staticmethod
    def normalize_number(s: str) -> Optional[float]:
        """
        将字符串标准化为数值

        处理逗号、百分号、分数等格式。

        Args:
            s: 答案字符串

        Returns:
            标准化后的数值，无法转换则返回 None
        """
        s = s.strip()

        # 移除逗号
        s = s.replace(',', '')

        # 处理百分号
        if s.endswith('%'):
            try:
                return float(s[:-1]) / 100
            except ValueError:
                pass

        # 处理分数
        frac_match = re.match(r'^(-?\d+)/(\d+)$', s)
        if frac_match:
            num, den = int(frac_match.group(1)), int(frac_match.group(2))
            if den != 0:
                return num / den

        # 直接转换
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def normalize_string(s: str) -> str:
        """
        标准化字符串用于比较

        Args:
            s: 原始字符串

        Returns:
            标准化后的字符串
        """
        s = s.strip().lower()
        # 移除多余空白
        s = re.sub(r'\s+', ' ', s)
        # 移除末尾标点
        s = s.rstrip('。.!！')
        return s

    def compare(
        self,
        prediction: str,
        gold: str,
        tolerance: float = 1e-4,
    ) -> bool:
        """
        比较预测答案与标准答案

        Args:
            prediction: 预测答案
            gold: 标准答案
            tolerance: 数值比较的容差

        Returns:
            是否匹配
        """
        # 尝试数值比较
        pred_num = self.normalize_number(prediction)
        gold_num = self.normalize_number(gold)

        if pred_num is not None and gold_num is not None:
            return abs(pred_num - gold_num) < tolerance

        # 字符串比较
        return self.normalize_string(prediction) == self.normalize_string(gold)


class ReasoningEvaluator:
    """
    推理评估器

    对模型在推理数据集上的表现进行评估。

    支持的评估格式:
    - GSM8K: 精确匹配数值
    - MATH: \\boxed{} 答案精确匹配
    - MMLU: 多选题选项匹配
    - HumanEval: pass@k（代码执行）
    """

    def __init__(self, format_hint: str = "auto"):
        """
        初始化评估器

        Args:
            format_hint: 数据集格式提示
        """
        self.extractor = AnswerExtractor()
        self.comparator = AnswerComparator()
        self.format_hint = format_hint

    def evaluate_single(
        self,
        question: str,
        gold_answer: str,
        model_response: str,
        difficulty: str = "",
        category: str = "",
    ) -> EvalSample:
        """
        评估单个样本

        Args:
            question: 问题
            gold_answer: 标准答案
            model_response: 模型输出
            difficulty: 难度级别
            category: 类别

        Returns:
            评估样本
        """
        # 提取答案
        extracted = self.extractor.extract(model_response, self.format_hint)

        # 比较答案
        is_correct = self.comparator.compare(extracted, gold_answer)

        return EvalSample(
            question=question,
            gold_answer=gold_answer,
            model_response=model_response,
            extracted_answer=extracted,
            is_correct=is_correct,
            difficulty=difficulty,
            category=category,
        )

    def evaluate_batch(
        self,
        samples: List[Dict],
    ) -> EvalResult:
        """
        批量评估

        Args:
            samples: 样本列表，每个样本是一个字典，包含:
                - question: 问题
                - gold_answer: 标准答案
                - model_response: 模型输出
                - difficulty: 难度（可选）
                - category: 类别（可选）

        Returns:
            评估结果
        """
        eval_samples = []
        for s in samples:
            result = self.evaluate_single(
                question=s["question"],
                gold_answer=s["gold_answer"],
                model_response=s["model_response"],
                difficulty=s.get("difficulty", ""),
                category=s.get("category", ""),
            )
            eval_samples.append(result)

        # 总体准确率
        n_correct = sum(1 for s in eval_samples if s.is_correct)
        accuracy = n_correct / len(eval_samples) if eval_samples else 0.0

        # 按难度分组
        acc_by_diff = self._group_accuracy(eval_samples, "difficulty")

        # 按类别分组
        acc_by_cat = self._group_accuracy(eval_samples, "category")

        return EvalResult(
            accuracy=accuracy,
            n_total=len(eval_samples),
            n_correct=n_correct,
            accuracy_by_difficulty=acc_by_diff,
            accuracy_by_category=acc_by_cat,
            samples=eval_samples,
        )

    def _group_accuracy(
        self,
        samples: List[EvalSample],
        group_field: str,
    ) -> Dict[str, float]:
        """
        按字段分组计算准确率

        Args:
            samples: 评估样本列表
            group_field: 分组字段名

        Returns:
            {组名: 准确率}
        """
        groups: Dict[str, List[bool]] = {}
        for s in samples:
            key = getattr(s, group_field, "")
            if key:
                if key not in groups:
                    groups[key] = []
                groups[key].append(s.is_correct)

        return {
            k: sum(v) / len(v)
            for k, v in groups.items()
        }


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    计算 pass@k 指标

    pass@k = E[1 - C(n-c, k) / C(n, k)]

    其中:
    - n: 总采样数
    - c: 通过测试的数量
    - k: 每道题提交的代码数

    使用数值稳定的计算方式避免大数阶乘溢出。

    Args:
        n: 总采样数
        c: 通过测试的数量
        k: 提交次数

    Returns:
        pass@k 值
    """
    if n - c < k:
        return 1.0

    # 使用对数避免溢出: log(C(n-c, k) / C(n, k))
    # = log(C(n-c,k)) - log(C(n,k))
    # = Σ log(n-c-i) - Σ log(n-i) for i in 0..k-1
    log_ratio = 0.0
    for i in range(k):
        log_ratio += math.log(n - c - i) - math.log(n - i)

    return 1.0 - math.exp(log_ratio)


def demonstrate_answer_extraction():
    """演示答案提取功能"""
    print("=" * 70)
    print("答案提取演示")
    print("=" * 70)

    extractor = AnswerExtractor()

    test_cases = [
        # (模型输出, 格式提示, 预期答案)
        ("经过计算：15 - 5 = 10, 10 + 8 = 18, 18 / 2 = 9。答案是9。", "auto", "9"),
        ("The answer is 42.", "auto", "42"),
        ("Therefore, $\\boxed{\\frac{3}{4}}$ is the answer.", "math", "\\frac{3}{4}"),
        ("#### 256", "gsm8k", "256"),
        ("综合以上分析，选择B", "mcq", "B"),
        ("Step 1: 5*6=30. Step 2: 30+12=42.", "number", "42"),
        ("所以答案为：-3.14", "auto", "-3.14"),
    ]

    for response, fmt, expected in test_cases:
        result = extractor.extract(response, format_hint=fmt)
        status = "OK" if result == expected else f"MISMATCH (expected: {expected})"
        print(f"\n  输入: {response[:60]}...")
        print(f"  格式: {fmt}")
        print(f"  提取: {result} [{status}]")


def demonstrate_evaluation():
    """演示评估流程"""
    print("\n" + "=" * 70)
    print("推理评估演示")
    print("=" * 70)

    evaluator = ReasoningEvaluator(format_hint="auto")

    # 模拟 GSM8K 格式的评估数据
    samples = [
        {
            "question": "Roger有5个网球。他又买了2罐，每罐3个。现在有多少个？",
            "gold_answer": "11",
            "model_response": "Roger有5个球，又买了2*3=6个，总共5+6=11个。答案是11。",
            "difficulty": "easy",
            "category": "arithmetic",
        },
        {
            "question": "食堂有23个苹果，用了20个，又买了6个。现在有多少？",
            "gold_answer": "9",
            "model_response": "23-20=3, 3+6=9。答案是9。",
            "difficulty": "easy",
            "category": "arithmetic",
        },
        {
            "question": "一个班30人，男生比女生多6人。男生多少人？",
            "gold_answer": "18",
            "model_response": "设女生x人，x+(x+6)=30, x=12, 男生=12+6=18。答案是18。",
            "difficulty": "medium",
            "category": "algebra",
        },
        {
            "question": "一个长方体长5cm宽3cm高4cm，表面积是多少？",
            "gold_answer": "94",
            "model_response": "表面积=2*(5*3+5*4+3*4)=2*(15+20+12)=2*47=94。答案是94。",
            "difficulty": "medium",
            "category": "geometry",
        },
        {
            "question": "3的100次方除以7的余数是多少？",
            "gold_answer": "4",
            "model_response": "3^1=3, 3^2=9, 3^3=27, 3^4=81, 3^5=243, 3^6=729。3^6 mod 7 = 1。100=6*16+4, 所以3^100 mod 7 = 3^4 mod 7 = 81 mod 7 = 4。答案是5。",  # 推理正确但答案提取错误
            "difficulty": "hard",
            "category": "number_theory",
        },
    ]

    result = evaluator.evaluate_batch(samples)

    print(f"\n总体准确率: {result.accuracy:.1%} ({result.n_correct}/{result.n_total})")

    if result.accuracy_by_difficulty:
        print(f"\n按难度:")
        for diff, acc in result.accuracy_by_difficulty.items():
            print(f"  {diff}: {acc:.1%}")

    if result.accuracy_by_category:
        print(f"\n按类别:")
        for cat, acc in result.accuracy_by_category.items():
            print(f"  {cat}: {acc:.1%}")

    # 显示每个样本的详情
    print(f"\n详细结果:")
    for i, s in enumerate(result.samples):
        status = "CORRECT" if s.is_correct else "WRONG"
        print(f"  [{status}] Q: {s.question[:40]}... | "
              f"Gold: {s.gold_answer} | Pred: {s.extracted_answer}")


def demonstrate_pass_at_k():
    """演示 pass@k 计算"""
    print("\n" + "=" * 70)
    print("pass@k 指标计算演示")
    print("=" * 70)

    print("\npass@k = E[1 - C(n-c, k) / C(n, k)]")
    print("n: 总采样数, c: 通过数, k: 提交次数\n")

    # 不同场景
    scenarios = [
        (10, 3, 1),  # 10次采样，3次通过，提交1次
        (10, 3, 5),
        (10, 3, 10),
        (20, 5, 1),
        (20, 5, 5),
        (100, 10, 1),
        (100, 10, 10),
    ]

    print(f"{'n':>5} | {'c':>3} | {'k':>3} | {'pass@k':>8}")
    print("-" * 30)

    for n, c, k in scenarios:
        pak = compute_pass_at_k(n, c, k)
        print(f"{n:>5} | {c:>3} | {k:>3} | {pak:>7.1%}")


def demonstrate_answer_comparison():
    """演示答案比较功能"""
    print("\n" + "=" * 70)
    print("答案比较演示")
    print("=" * 70)

    comparator = AnswerComparator()

    test_cases = [
        # (预测, 标准, 预期结果)
        ("42", "42", True),
        ("42.0", "42", True),
        ("42.00001", "42", True),  # 精度范围内
        ("1,000", "1000", True),  # 逗号分隔
        ("3/4", "0.75", True),  # 分数
        ("50%", "0.5", True),  # 百分号
        ("是", "是", True),
        ("yes", "YES", True),  # 大小写不敏感
        ("42", "43", False),
        ("apple", "orange", False),
    ]

    for pred, gold, expected in test_cases:
        result = comparator.compare(pred, gold)
        status = "OK" if result == expected else "FAIL"
        print(f"  [{status}] '{pred}' vs '{gold}' → {result}")


if __name__ == "__main__":
    # 答案提取演示
    demonstrate_answer_extraction()

    # 评估流程演示
    demonstrate_evaluation()

    # pass@k 计算
    demonstrate_pass_at_k()

    # 答案比较
    demonstrate_answer_comparison()
