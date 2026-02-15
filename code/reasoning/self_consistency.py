"""
Self-Consistency 多次采样 + 多数投票

本模块实现了 Self-Consistency 推理策略：对同一问题多次采样不同的推理路径，
通过多数投票选择最一致的答案。

核心公式:
- 采样: (r_i, a_i) ~ P_θ(· | q), i = 1, ..., N
- 投票: â = argmax_a Σ_i 1[a_i = a]

参考:
- Wang et al. (2023). Self-Consistency Improves Chain of Thought Reasoning.
"""

import random
import collections
import math
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass


@dataclass
class SamplingPath:
    """
    单条采样路径

    Attributes:
        reasoning: 推理过程文本
        answer: 从推理中提取的最终答案
        log_prob: 该路径的对数概率（如果可用）
        temperature: 采样温度
    """
    reasoning: str
    answer: str
    log_prob: float = 0.0
    temperature: float = 0.7


@dataclass
class ConsistencyResult:
    """
    Self-Consistency 的结果

    Attributes:
        best_answer: 多数投票选出的最佳答案
        vote_distribution: 各答案的投票分布
        confidence: 置信度（最高票占总票比例）
        n_samples: 采样总数
        n_unique_answers: 不同答案的数量
        paths: 所有采样路径
    """
    best_answer: str
    vote_distribution: Dict[str, int]
    confidence: float
    n_samples: int
    n_unique_answers: int
    paths: List[SamplingPath]


def temperature_scaled_sampling(logits: List[float], temperature: float = 1.0) -> int:
    """
    温度缩放采样

    P(x_i) = exp(z_i / T) / Σ_j exp(z_j / T)

    温度 T 控制采样多样性:
    - T → 0: 退化为贪心选择（argmax）
    - T = 1: 标准采样
    - T → ∞: 均匀随机采样

    Args:
        logits: 原始 logit 值列表
        temperature: 采样温度

    Returns:
        采样得到的索引
    """
    if temperature <= 0:
        # 温度为0时退化为贪心
        return max(range(len(logits)), key=lambda i: logits[i])

    # 温度缩放
    scaled = [z / temperature for z in logits]

    # 数值稳定的 softmax
    max_val = max(scaled)
    exp_vals = [math.exp(z - max_val) for z in scaled]
    total = sum(exp_vals)
    probs = [e / total for e in exp_vals]

    # 按概率采样
    r = random.random()
    cumsum = 0.0
    for i, p in enumerate(probs):
        cumsum += p
        if r <= cumsum:
            return i
    return len(probs) - 1


class SelfConsistency:
    """
    Self-Consistency 推理引擎

    通过多次采样不同的推理路径，利用多数投票选出最一致的答案。

    核心流程:
    1. 使用温度 > 0 的采样生成 N 条推理路径
    2. 从每条路径中提取最终答案
    3. 对所有答案进行多数投票
    4. 返回票数最多的答案
    """

    def __init__(
        self,
        generator: Optional[Callable] = None,
        answer_extractor: Optional[Callable] = None,
        default_n_samples: int = 10,
        default_temperature: float = 0.7,
    ):
        """
        初始化 Self-Consistency 引擎

        Args:
            generator: 文本生成函数 (prompt, temperature) -> str
            answer_extractor: 答案提取函数 str -> str
            default_n_samples: 默认采样次数
            default_temperature: 默认采样温度
        """
        self.generator = generator or self._default_generator
        self.answer_extractor = answer_extractor or self._default_extractor
        self.default_n_samples = default_n_samples
        self.default_temperature = default_temperature

    def _default_generator(self, prompt: str, temperature: float) -> str:
        """
        默认生成器（模拟）

        实际使用时应替换为真实的 LLM API 调用。
        这里模拟不同温度下的采样行为。

        Args:
            prompt: 输入 prompt
            temperature: 采样温度

        Returns:
            模拟的推理输出
        """
        return f"[模拟推理路径 | temperature={temperature}]"

    def _default_extractor(self, response: str) -> str:
        """
        默认答案提取器

        尝试从响应文本中提取数值答案。

        Args:
            response: 模型输出文本

        Returns:
            提取的答案字符串
        """
        import re
        # 尝试多种提取模式
        patterns = [
            r'答案[是为：:]\s*(-?\d+\.?\d*)',
            r'\\boxed\{([^}]+)\}',
            r'[Tt]he answer is\s*(-?\d+\.?\d*)',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, response)
            if matches:
                return matches[-1].strip()

        # 回退：最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]

        return response.strip()

    def sample_paths(
        self,
        prompt: str,
        n_samples: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[SamplingPath]:
        """
        采样多条推理路径

        Args:
            prompt: 输入 prompt
            n_samples: 采样次数
            temperature: 采样温度

        Returns:
            采样路径列表
        """
        n = n_samples or self.default_n_samples
        temp = temperature or self.default_temperature

        paths = []
        for i in range(n):
            # 生成推理
            reasoning = self.generator(prompt, temp)
            # 提取答案
            answer = self.answer_extractor(reasoning)
            # 记录路径
            paths.append(SamplingPath(
                reasoning=reasoning,
                answer=answer,
                temperature=temp,
            ))

        return paths

    def majority_vote(self, paths: List[SamplingPath]) -> ConsistencyResult:
        """
        对采样路径进行多数投票

        â = argmax_a Σ_i 1[a_i = a]

        Args:
            paths: 采样路径列表

        Returns:
            投票结果
        """
        # 统计各答案的票数
        counter = collections.Counter(p.answer for p in paths)

        # 找到票数最多的答案
        best_answer, best_count = counter.most_common(1)[0]

        # 计算置信度
        confidence = best_count / len(paths)

        return ConsistencyResult(
            best_answer=best_answer,
            vote_distribution=dict(counter),
            confidence=confidence,
            n_samples=len(paths),
            n_unique_answers=len(counter),
            paths=paths,
        )

    def weighted_vote(self, paths: List[SamplingPath]) -> ConsistencyResult:
        """
        加权多数投票

        â = argmax_a Σ_i P(r_i, a_i | q) · 1[a_i = a]

        使用路径的对数概率作为权重。

        Args:
            paths: 采样路径列表（需要有 log_prob 信息）

        Returns:
            加权投票结果
        """
        weighted_counts: Dict[str, float] = {}

        for path in paths:
            weight = math.exp(path.log_prob) if path.log_prob != 0 else 1.0
            if path.answer not in weighted_counts:
                weighted_counts[path.answer] = 0.0
            weighted_counts[path.answer] += weight

        # 找到加权票数最多的答案
        best_answer = max(weighted_counts, key=weighted_counts.get)
        total_weight = sum(weighted_counts.values())
        confidence = weighted_counts[best_answer] / total_weight if total_weight > 0 else 0.0

        # 转换为整数投票分布（用于显示）
        int_distribution = {
            a: sum(1 for p in paths if p.answer == a)
            for a in weighted_counts
        }

        return ConsistencyResult(
            best_answer=best_answer,
            vote_distribution=int_distribution,
            confidence=confidence,
            n_samples=len(paths),
            n_unique_answers=len(weighted_counts),
            paths=paths,
        )

    def run(
        self,
        prompt: str,
        n_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        weighted: bool = False,
    ) -> ConsistencyResult:
        """
        执行完整的 Self-Consistency 流程

        Args:
            prompt: 输入 prompt
            n_samples: 采样次数
            temperature: 采样温度
            weighted: 是否使用加权投票

        Returns:
            一致性投票结果
        """
        paths = self.sample_paths(prompt, n_samples, temperature)

        if weighted:
            return self.weighted_vote(paths)
        else:
            return self.majority_vote(paths)


def analyze_consistency_scaling(
    results_by_n: Dict[int, List[bool]],
) -> Dict[int, float]:
    """
    分析 Self-Consistency 随采样次数的 Scaling 行为

    Args:
        results_by_n: {采样次数: [是否正确的列表]}

    Returns:
        {采样次数: 准确率}
    """
    accuracy_by_n = {}
    for n, correctness_list in results_by_n.items():
        accuracy = sum(correctness_list) / len(correctness_list)
        accuracy_by_n[n] = accuracy

    return accuracy_by_n


def simulate_self_consistency_experiment():
    """
    模拟 Self-Consistency 实验

    用模拟数据演示:
    1. 不同采样次数下的投票行为
    2. 置信度分析
    3. 准确率随 N 的变化
    """
    print("=" * 70)
    print("Self-Consistency 模拟实验")
    print("=" * 70)

    # 模拟一道数学题的多次采样
    # 正确答案是 42，但模型有一定概率给出错误答案
    random.seed(42)

    # 模拟生成器：70%概率给出正确答案42，30%概率给出其他答案
    correct_answer = "42"
    wrong_answers = ["38", "44", "40", "36"]

    def mock_generator(prompt: str, temperature: float) -> str:
        """模拟 LLM 生成"""
        if random.random() < 0.7:  # 70% 正确率
            return f"计算过程...答案是{correct_answer}。"
        else:
            wrong = random.choice(wrong_answers)
            return f"计算过程...答案是{wrong}。"

    # 创建 Self-Consistency 引擎
    sc = SelfConsistency(
        generator=mock_generator,
        default_n_samples=10,
        default_temperature=0.7,
    )

    # 实验1：单次 Self-Consistency 运行
    print("\n--- 实验1: 单次运行 (N=10) ---")
    result = sc.run("测试问题", n_samples=10)
    print(f"最佳答案: {result.best_answer}")
    print(f"投票分布: {result.vote_distribution}")
    print(f"置信度: {result.confidence:.2%}")
    print(f"不同答案数: {result.n_unique_answers}")

    # 实验2：不同 N 值的 Scaling 分析
    print("\n--- 实验2: Scaling 分析 ---")
    n_values = [1, 3, 5, 10, 20, 40]
    n_trials = 100  # 每个 N 值重复实验次数

    print(f"{'N':>5} | {'准确率':>8} | {'平均置信度':>10}")
    print("-" * 35)

    for n in n_values:
        correct_count = 0
        total_confidence = 0.0

        for _ in range(n_trials):
            result = sc.run("测试问题", n_samples=n)
            if result.best_answer == correct_answer:
                correct_count += 1
            total_confidence += result.confidence

        accuracy = correct_count / n_trials
        avg_confidence = total_confidence / n_trials
        print(f"{n:>5} | {accuracy:>7.1%} | {avg_confidence:>9.1%}")

    # 实验3：温度对多样性的影响
    print("\n--- 实验3: 温度对多样性的影响 ---")
    temperatures = [0.1, 0.3, 0.5, 0.7, 1.0]

    for temp in temperatures:
        result = sc.run("测试问题", n_samples=20, temperature=temp)
        print(f"T={temp:.1f} | 不同答案数: {result.n_unique_answers} | "
              f"置信度: {result.confidence:.2%} | 最佳: {result.best_answer}")


def demonstrate_temperature_effect():
    """演示温度对采样分布的影响"""
    print("\n" + "=" * 70)
    print("温度缩放采样演示")
    print("=" * 70)

    # 模拟3个候选答案的 logits
    logits = [2.0, 1.5, 0.5]  # 答案A最可能，C最不可能
    labels = ["答案A", "答案B", "答案C"]

    temperatures = [0.1, 0.5, 1.0, 2.0]
    n_samples = 1000

    print(f"\nLogits: {logits}")
    print(f"采样 {n_samples} 次:\n")

    for temp in temperatures:
        counts = [0] * len(logits)
        for _ in range(n_samples):
            idx = temperature_scaled_sampling(logits, temp)
            counts[idx] += 1

        probs = [c / n_samples for c in counts]
        dist_str = " | ".join(f"{labels[i]}: {probs[i]:.1%}" for i in range(len(labels)))
        print(f"  T={temp:.1f}: {dist_str}")


if __name__ == "__main__":
    # 模拟 Self-Consistency 实验
    simulate_self_consistency_experiment()

    # 温度效果演示
    demonstrate_temperature_effect()
