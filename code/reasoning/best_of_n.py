"""
Best-of-N 采样 + Verifier 排序

本模块实现了 Best-of-N 采样策略：生成 N 个候选回答，
使用 Verifier 模型对每个回答评分，选择分数最高的回答。

核心公式:
- â = argmax_{(r_i, a_i)} V(q, r_i, a_i), i = 1, ..., N
- 至少一个正确的概率: P = 1 - (1-p)^N

参考:
- Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems.
- Lightman et al. (2023). Let's Verify Step by Step.
"""

import math
import random
from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class Candidate:
    """
    候选回答

    Attributes:
        reasoning: 推理过程
        answer: 最终答案
        verifier_score: Verifier 给出的分数 (0-1)
        generation_index: 生成序号
    """
    reasoning: str
    answer: str
    verifier_score: float = 0.0
    generation_index: int = 0


@dataclass
class BestOfNResult:
    """
    Best-of-N 的结果

    Attributes:
        selected: 被选中的候选（分数最高）
        all_candidates: 所有候选列表
        n_total: 总候选数
        score_distribution: 分数分布统计
    """
    selected: Candidate
    all_candidates: List[Candidate]
    n_total: int
    score_distribution: Dict[str, float]


class BestOfN:
    """
    Best-of-N 采样引擎

    生成 N 个候选回答，用 Verifier 评分后选择最优。

    核心流程:
    1. 对问题采样 N 个回答
    2. 用 Verifier 对每个回答打分
    3. 选择分数最高的回答
    """

    def __init__(
        self,
        generator: Optional[Callable] = None,
        verifier: Optional[Callable] = None,
        default_n: int = 10,
        default_temperature: float = 0.7,
    ):
        """
        初始化 Best-of-N 引擎

        Args:
            generator: 回答生成函数 (question, temperature) -> (reasoning, answer)
            verifier: 验证评分函数 (question, reasoning, answer) -> float
            default_n: 默认采样数量
            default_temperature: 默认采样温度
        """
        self.generator = generator or self._default_generator
        self.verifier = verifier or self._default_verifier
        self.default_n = default_n
        self.default_temperature = default_temperature

    def _default_generator(self, question: str, temperature: float) -> Tuple[str, str]:
        """
        默认生成器（模拟）

        Args:
            question: 问题文本
            temperature: 采样温度

        Returns:
            (推理过程, 最终答案)
        """
        return (
            f"[模拟推理 | T={temperature}]",
            f"[模拟答案]"
        )

    def _default_verifier(self, question: str, reasoning: str, answer: str) -> float:
        """
        默认验证器（模拟）

        Args:
            question: 问题
            reasoning: 推理过程
            answer: 答案

        Returns:
            验证分数 (0-1)
        """
        return random.uniform(0.2, 0.95)

    def generate_candidates(
        self,
        question: str,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[Candidate]:
        """
        生成 N 个候选回答并评分

        Args:
            question: 问题文本
            n: 采样数量
            temperature: 采样温度

        Returns:
            评分后的候选列表
        """
        n = n or self.default_n
        temperature = temperature or self.default_temperature

        candidates = []
        for i in range(n):
            # 生成回答
            reasoning, answer = self.generator(question, temperature)

            # Verifier 评分
            score = self.verifier(question, reasoning, answer)

            candidates.append(Candidate(
                reasoning=reasoning,
                answer=answer,
                verifier_score=score,
                generation_index=i,
            ))

        return candidates

    def select_best(self, candidates: List[Candidate]) -> Candidate:
        """
        选择分数最高的候选

        Args:
            candidates: 候选列表

        Returns:
            分数最高的候选
        """
        return max(candidates, key=lambda c: c.verifier_score)

    def run(
        self,
        question: str,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> BestOfNResult:
        """
        执行完整的 Best-of-N 流程

        Args:
            question: 问题文本
            n: 采样数量
            temperature: 采样温度

        Returns:
            Best-of-N 结果
        """
        candidates = self.generate_candidates(question, n, temperature)
        selected = self.select_best(candidates)

        # 计算分数分布统计
        scores = [c.verifier_score for c in candidates]
        score_dist = {
            "min": min(scores),
            "max": max(scores),
            "mean": sum(scores) / len(scores),
            "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5,
        }

        return BestOfNResult(
            selected=selected,
            all_candidates=candidates,
            n_total=len(candidates),
            score_distribution=score_dist,
        )


def compute_success_probability(single_correct_prob: float, n: int) -> float:
    """
    计算 Best-of-N 的成功概率

    P(至少一个正确) = 1 - (1 - p)^N

    Args:
        single_correct_prob: 单次采样的正确概率
        n: 采样次数

    Returns:
        至少有一个正确答案的概率
    """
    return 1.0 - (1.0 - single_correct_prob) ** n


def compute_optimal_n(
    single_correct_prob: float,
    target_success_prob: float = 0.95,
) -> int:
    """
    计算达到目标成功率需要的最小 N

    1 - (1-p)^N >= target
    N >= log(1-target) / log(1-p)

    Args:
        single_correct_prob: 单次正确概率
        target_success_prob: 目标成功概率

    Returns:
        需要的最小采样次数 N
    """
    if single_correct_prob >= 1.0:
        return 1
    if single_correct_prob <= 0.0:
        return float('inf')

    n = math.log(1 - target_success_prob) / math.log(1 - single_correct_prob)
    return math.ceil(n)


def compare_selection_strategies(candidates: List[Candidate], correct_answer: str) -> Dict:
    """
    比较不同的候选选择策略

    Args:
        candidates: 候选列表
        correct_answer: 正确答案

    Returns:
        各策略的表现
    """
    results = {}

    # 策略1: Random（随机选择）
    random_choice = random.choice(candidates)
    results["random"] = {
        "answer": random_choice.answer,
        "correct": random_choice.answer == correct_answer,
        "score": random_choice.verifier_score,
    }

    # 策略2: Best-of-N（Verifier 选择最高分）
    best = max(candidates, key=lambda c: c.verifier_score)
    results["best_of_n"] = {
        "answer": best.answer,
        "correct": best.answer == correct_answer,
        "score": best.verifier_score,
    }

    # 策略3: Majority Vote（多数投票）
    from collections import Counter
    vote_counter = Counter(c.answer for c in candidates)
    majority_answer = vote_counter.most_common(1)[0][0]
    results["majority_vote"] = {
        "answer": majority_answer,
        "correct": majority_answer == correct_answer,
        "votes": vote_counter.most_common(3),
    }

    # 策略4: Weighted Vote（Verifier 加权投票）
    weighted_scores = {}
    for c in candidates:
        if c.answer not in weighted_scores:
            weighted_scores[c.answer] = 0.0
        weighted_scores[c.answer] += c.verifier_score
    weighted_best = max(weighted_scores, key=weighted_scores.get)
    results["weighted_vote"] = {
        "answer": weighted_best,
        "correct": weighted_best == correct_answer,
        "weighted_score": weighted_scores[weighted_best],
    }

    return results


def simulate_best_of_n_experiment():
    """
    模拟 Best-of-N 实验

    展示:
    1. N 值与成功概率的关系
    2. 不同选择策略的比较
    3. Verifier 质量对结果的影响
    """
    print("=" * 70)
    print("Best-of-N 采样实验")
    print("=" * 70)

    random.seed(42)

    # 实验1: 成功概率随 N 的变化
    print("\n--- 实验1: 成功概率 vs 采样次数 N ---")
    single_probs = [0.1, 0.2, 0.3, 0.5, 0.7]

    print(f"{'p':>5} | ", end="")
    n_values = [1, 3, 5, 10, 20, 50]
    for n in n_values:
        print(f"N={n:>2} | ", end="")
    print()
    print("-" * 55)

    for p in single_probs:
        print(f"{p:>5.1f} | ", end="")
        for n in n_values:
            success_prob = compute_success_probability(p, n)
            print(f"{success_prob:>5.1%} | ", end="")
        print()

    # 实验2: 达到目标成功率需要的最小 N
    print(f"\n--- 实验2: 达到 95% 成功率需要的最小 N ---")
    for p in single_probs:
        optimal_n = compute_optimal_n(p, 0.95)
        print(f"  单次正确率 p={p:.1f} → 需要 N={optimal_n}")

    # 实验3: 模拟完整的 Best-of-N 流程
    print(f"\n--- 实验3: 模拟 Best-of-N 完整流程 ---")

    correct_answer = "42"
    wrong_answers = ["38", "44", "40", "36", "50"]

    def mock_generator(question: str, temperature: float) -> Tuple[str, str]:
        """模拟生成器: 60% 概率给出正确答案"""
        if random.random() < 0.6:
            return (f"计算过程...结果是{correct_answer}", correct_answer)
        else:
            wrong = random.choice(wrong_answers)
            return (f"计算过程...结果是{wrong}", wrong)

    def mock_verifier(question: str, reasoning: str, answer: str) -> float:
        """模拟 Verifier: 正确答案倾向于获得更高分"""
        base_score = random.uniform(0.3, 0.7)
        if answer == correct_answer:
            base_score += random.uniform(0.1, 0.3)  # 正确答案加分
        return min(base_score, 1.0)

    bon = BestOfN(
        generator=mock_generator,
        verifier=mock_verifier,
        default_n=10,
    )

    # 运行多次实验
    n_trials = 100
    n_values_exp = [1, 3, 5, 10, 20]

    print(f"\n{'N':>5} | {'Best-of-N 正确率':>15} | {'平均最高分':>10}")
    print("-" * 40)

    for n in n_values_exp:
        correct_count = 0
        total_best_score = 0.0

        for _ in range(n_trials):
            result = bon.run("测试问题", n=n)
            if result.selected.answer == correct_answer:
                correct_count += 1
            total_best_score += result.selected.verifier_score

        accuracy = correct_count / n_trials
        avg_best_score = total_best_score / n_trials
        print(f"{n:>5} | {accuracy:>14.1%} | {avg_best_score:>10.3f}")

    # 实验4: 选择策略比较
    print(f"\n--- 实验4: 选择策略比较 (N=20, 100次实验) ---")

    strategy_correct = {"random": 0, "best_of_n": 0, "majority_vote": 0, "weighted_vote": 0}

    for _ in range(n_trials):
        candidates = bon.generate_candidates("测试问题", n=20)
        comparison = compare_selection_strategies(candidates, correct_answer)
        for strategy, info in comparison.items():
            if info.get("correct", False):
                strategy_correct[strategy] += 1

    print(f"{'策略':>15} | {'正确率':>8}")
    print("-" * 30)
    for strategy, count in strategy_correct.items():
        print(f"{strategy:>15} | {count/n_trials:>7.1%}")


def demonstrate_verifier_quality_impact():
    """
    演示 Verifier 质量对 Best-of-N 效果的影响
    """
    print("\n" + "=" * 70)
    print("Verifier 质量对 Best-of-N 效果的影响")
    print("=" * 70)

    random.seed(42)
    correct_answer = "42"

    def mock_generator(question: str, temperature: float) -> Tuple[str, str]:
        if random.random() < 0.5:
            return ("正确推理", correct_answer)
        else:
            return ("错误推理", random.choice(["38", "44", "40"]))

    # 定义不同质量的 Verifier
    def perfect_verifier(q, r, a):
        """完美 Verifier: 正确答案 score=1, 错误答案 score=0"""
        return 1.0 if a == correct_answer else 0.0

    def good_verifier(q, r, a):
        """好的 Verifier: 正确答案倾向于高分"""
        if a == correct_answer:
            return random.uniform(0.6, 1.0)
        else:
            return random.uniform(0.1, 0.5)

    def mediocre_verifier(q, r, a):
        """一般的 Verifier: 略有区分能力"""
        if a == correct_answer:
            return random.uniform(0.4, 0.8)
        else:
            return random.uniform(0.3, 0.7)

    def random_verifier(q, r, a):
        """随机 Verifier: 完全无区分能力"""
        return random.random()

    verifiers = {
        "完美": perfect_verifier,
        "好": good_verifier,
        "一般": mediocre_verifier,
        "随机": random_verifier,
    }

    n_trials = 200
    N = 10

    print(f"\nN={N}, 单次正确率=50%, {n_trials}次实验")
    print(f"\n{'Verifier 质量':>15} | {'Best-of-N 正确率':>15}")
    print("-" * 35)

    for name, verifier_fn in verifiers.items():
        bon = BestOfN(generator=mock_generator, verifier=verifier_fn, default_n=N)
        correct_count = sum(
            1 for _ in range(n_trials)
            if bon.run("测试").selected.answer == correct_answer
        )
        print(f"{name:>15} | {correct_count/n_trials:>14.1%}")


if __name__ == "__main__":
    # Best-of-N 主实验
    simulate_best_of_n_experiment()

    # Verifier 质量影响
    demonstrate_verifier_quality_impact()
