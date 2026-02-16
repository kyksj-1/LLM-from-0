"""
Self-Play 数据生成循环: 模型自我对弈提升数学解题能力

本模块实现了一个完整的 Self-Play 循环框架:
1. 模型对数学问题采样多条解答路径
2. 用验证器检查答案正确性
3. 选择最优路径作为训练数据
4. 用筛选后的数据微调模型（或输出供外部训练）
5. 迭代上述过程

核心思想:
- 对每个问题采样 K 条回答，至少一条正确的概率为 1 - (1-p)^K
- 用验证器从正确回答中选出最优（最短、最清晰）的作为训练目标
- 每轮迭代后模型能力提升，pass rate p 增大，飞轮效应启动

依赖:
- openai: OpenAI 兼容 API 客户端
- sympy: 数学表达式验证 (可选)

安装:
    pip install openai sympy

参考:
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL.
- Chen et al. (2024). Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models.
"""

import json
import re
import logging
import random
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 数学答案验证器
# ============================================================

class MathVerifier:
    """
    数学答案验证器

    支持多种答案格式:
    - \\boxed{answer} 格式
    - 纯数字
    - 数学表达式（通过 sympy 简化后比较）
    """

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """
        从模型输出中提取最终答案

        优先级:
        1. \\boxed{...} 格式
        2. "答案是..." / "答案为..." 后的内容
        3. "The answer is..." 后的内容
        4. 最后一行的数字
        """
        # 尝试 \\boxed{} 格式
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        matches = re.findall(boxed_pattern, text)
        if matches:
            return matches[-1].strip()

        # 尝试中文"答案"格式
        cn_pattern = r'答案[是为：:]\s*(.+?)(?:\n|$|。)'
        matches = re.findall(cn_pattern, text)
        if matches:
            return matches[-1].strip()

        # 尝试英文 "answer" 格式
        en_pattern = r'[Tt]he (?:final )?answer is[:\s]*(.+?)(?:\n|$|\.)'
        matches = re.findall(en_pattern, text)
        if matches:
            return matches[-1].strip()

        # 回退：提取最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', text)
        if numbers:
            return numbers[-1]

        return None

    @staticmethod
    def normalize_answer(answer: str) -> str:
        """标准化答案格式"""
        # 去除空格和多余符号
        answer = answer.strip().strip("$").strip()
        # 去除千位分隔符
        answer = answer.replace(",", "")
        # 标准化分数
        answer = answer.replace("\\frac", "frac")
        return answer

    @staticmethod
    def math_equal(pred: str, target: str, tolerance: float = 1e-6) -> bool:
        """
        比较两个数学答案是否相等

        支持:
        - 精确数值比较
        - 浮点数近似比较
        - sympy 符号比较 (若可用)
        """
        pred = MathVerifier.normalize_answer(pred)
        target = MathVerifier.normalize_answer(target)

        # 字符串精确匹配
        if pred == target:
            return True

        # 数值比较
        try:
            pred_val = float(pred)
            target_val = float(target)
            if abs(pred_val - target_val) < tolerance:
                return True
            # 相对误差
            if target_val != 0 and abs((pred_val - target_val) / target_val) < tolerance:
                return True
        except (ValueError, OverflowError):
            pass

        # sympy 符号比较
        try:
            import sympy
            pred_expr = sympy.sympify(pred)
            target_expr = sympy.sympify(target)
            if sympy.simplify(pred_expr - target_expr) == 0:
                return True
        except Exception:
            pass

        return False

    def verify(self, solution: str, ground_truth: str) -> bool:
        """
        验证一条解答是否正确

        Args:
            solution: 模型的完整解答文本
            ground_truth: 正确答案

        Returns:
            True 如果答案正确
        """
        predicted = self.extract_answer(solution)
        if predicted is None:
            return False
        return self.math_equal(predicted, ground_truth)


# ============================================================
# 解答路径评估与选择
# ============================================================

@dataclass
class SolutionCandidate:
    """一条解答候选"""
    solution: str
    is_correct: bool
    answer: Optional[str]
    length: int  # token 近似长度（字符数）
    quality_score: float = 0.0


class SolutionSelector:
    """
    从多条正确解答中选择最优的

    选择标准:
    1. 正确性（必须条件）
    2. 简洁性（更短的推理更优）
    3. 结构清晰度（有步骤标记的更优）
    """

    @staticmethod
    def score_solution(solution: str) -> float:
        """
        对解答质量打分

        高分特征:
        - 有明确的步骤标记（Step 1, 步骤一, 等）
        - 有中间计算过程
        - 长度适中（不过长也不过短）
        - 有明确的最终答案标记
        """
        score = 0.0

        # 有步骤标记
        step_patterns = [r'[Ss]tep \d', r'步骤\s*\d', r'第\s*\d\s*步', r'\d[.)]\s']
        for pattern in step_patterns:
            if re.search(pattern, solution):
                score += 2.0
                break

        # 有最终答案标记
        if re.search(r'\\boxed\{', solution) or re.search(r'答案[是为]', solution):
            score += 1.0

        # 长度惩罚：过短或过长都不好
        length = len(solution)
        if 100 < length < 2000:
            score += 1.0  # 适中长度
        elif length > 5000:
            score -= 1.0  # 过于冗长

        # 有数学公式
        if re.search(r'[=+\-*/]', solution):
            score += 0.5

        return score

    def select_best(self, candidates: list[SolutionCandidate]) -> Optional[SolutionCandidate]:
        """从正确候选中选择最优解答"""
        correct = [c for c in candidates if c.is_correct]
        if not correct:
            return None

        # 按质量分数排序，分数相同时选更短的
        correct.sort(key=lambda c: (-c.quality_score, c.length))
        return correct[0]


# ============================================================
# Self-Play 引擎
# ============================================================

@dataclass
class SelfPlayResult:
    """一轮 Self-Play 的结果"""
    round_num: int
    total_problems: int
    solved_problems: int
    pass_at_1: float
    pass_at_k: float
    training_samples: int


class SelfPlayEngine:
    """
    Self-Play 数学推理数据生成引擎

    核心循环:
    1. 对题库中的每个问题采样 K 条解答
    2. 用验证器过滤正确解答
    3. 选择最优解答作为训练数据
    4. 输出训练数据（供外部训练脚本使用）
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        model: str = "deepseek-chat",
        num_samples: int = 8,
        temperature: float = 0.7,
        real_data_ratio: float = 0.3,
    ):
        """
        Args:
            api_base: OpenAI 兼容 API 地址
            api_key: API 密钥
            model: 模型名称
            num_samples: 每题采样次数 K
            temperature: 采样温度
            real_data_ratio: 真实数据混入比例（防止分布坍塌）
        """
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.num_samples = num_samples
        self.temperature = temperature
        self.real_data_ratio = real_data_ratio
        self.verifier = MathVerifier()
        self.selector = SolutionSelector()

    def _solve(self, problem: str) -> str:
        """让模型解答一道数学题"""
        prompt = f"""请解答以下数学问题，给出详细的推理过程。
在最后用 \\boxed{{答案}} 的格式给出最终答案。

问题: {problem}

解答:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()

    def _sample_solutions(self, problem: str, ground_truth: str) -> list[SolutionCandidate]:
        """对一道题采样 K 条解答并评估"""
        candidates = []

        for i in range(self.num_samples):
            try:
                solution = self._solve(problem)
                answer = self.verifier.extract_answer(solution)
                is_correct = self.verifier.verify(solution, ground_truth)
                quality = SolutionSelector.score_solution(solution)

                candidates.append(SolutionCandidate(
                    solution=solution,
                    is_correct=is_correct,
                    answer=answer,
                    length=len(solution),
                    quality_score=quality,
                ))
            except Exception as e:
                logger.warning("采样失败 (问题: %s..., 第 %d 次): %s", problem[:30], i + 1, e)

        return candidates

    def run_one_round(
        self,
        problems: list[dict],
        round_num: int = 0,
    ) -> tuple[list[dict], SelfPlayResult]:
        """
        执行一轮 Self-Play

        Args:
            problems: 问题列表，每项包含 "problem" 和 "answer" 字段
            round_num: 当前轮次编号

        Returns:
            (训练数据列表, 本轮结果统计)
        """
        logger.info("=== Self-Play 第 %d 轮开始 (共 %d 题) ===", round_num, len(problems))

        training_data = []
        solved_count = 0
        pass_at_1_correct = 0

        for idx, item in enumerate(problems):
            problem = item["problem"]
            ground_truth = item["answer"]

            # 采样解答
            candidates = self._sample_solutions(problem, ground_truth)

            if not candidates:
                continue

            # 统计 pass@1
            if candidates[0].is_correct:
                pass_at_1_correct += 1

            # 统计 pass@K
            correct_candidates = [c for c in candidates if c.is_correct]
            if correct_candidates:
                solved_count += 1

                # 选择最优解答
                best = self.selector.select_best(candidates)
                if best:
                    training_data.append({
                        "problem": problem,
                        "solution": best.solution,
                        "answer": best.answer,
                        "round": round_num,
                        "num_correct": len(correct_candidates),
                        "num_sampled": len(candidates),
                        "quality_score": best.quality_score,
                    })

            if (idx + 1) % 5 == 0:
                logger.info(
                    "进度: %d / %d, 已解决: %d, 训练样本: %d",
                    idx + 1, len(problems), solved_count, len(training_data),
                )

        # 计算统计指标
        total = len(problems)
        result = SelfPlayResult(
            round_num=round_num,
            total_problems=total,
            solved_problems=solved_count,
            pass_at_1=pass_at_1_correct / max(total, 1),
            pass_at_k=solved_count / max(total, 1),
            training_samples=len(training_data),
        )

        logger.info(
            "第 %d 轮完成: pass@1=%.2f%%, pass@%d=%.2f%%, 训练样本=%d",
            round_num, result.pass_at_1 * 100,
            self.num_samples, result.pass_at_k * 100,
            result.training_samples,
        )
        return training_data, result

    def run(
        self,
        problems: list[dict],
        num_rounds: int = 3,
        output_dir: str = "self_play_output",
    ) -> list[SelfPlayResult]:
        """
        运行完整的 Self-Play 循环

        注意: 模型的实际微调需要在外部完成（如使用 transformers Trainer）。
        本方法只负责生成每轮的训练数据。

        Args:
            problems: 完整题库
            num_rounds: 迭代轮数
            output_dir: 输出目录

        Returns:
            每轮的结果统计
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_results = []
        all_training_data = []

        for round_num in range(num_rounds):
            # 每轮随机采样子集（模拟课程学习）
            round_problems = random.sample(problems, min(len(problems), 50))

            # 执行一轮 Self-Play
            training_data, result = self.run_one_round(round_problems, round_num)

            # 混入真实数据（防止分布坍塌）
            if self.real_data_ratio > 0 and all_training_data:
                num_real = int(len(training_data) * self.real_data_ratio / (1 - self.real_data_ratio))
                real_samples = random.sample(
                    all_training_data,
                    min(num_real, len(all_training_data)),
                )
                training_data.extend(real_samples)
                logger.info("混入 %d 条历史真实数据", len(real_samples))

            # 保存本轮数据
            round_file = output_path / f"round_{round_num}_training_data.json"
            round_file.write_text(json.dumps(training_data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("第 %d 轮训练数据已保存到 %s", round_num, round_file)

            all_results.append(result)
            all_training_data.extend(training_data)

        # 保存汇总结果
        summary_file = output_path / "summary.json"
        summary_file.write_text(
            json.dumps([asdict(r) for r in all_results], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 打印进度曲线
        print("\n=== Self-Play 进度汇总 ===")
        print(f"{'轮次':>4} | {'pass@1':>8} | {'pass@K':>8} | {'训练样本':>8}")
        print("-" * 40)
        for r in all_results:
            print(f"{r.round_num:>4} | {r.pass_at_1:>7.2%} | {r.pass_at_k:>7.2%} | {r.training_samples:>8}")

        return all_results


# ============================================================
# 示例数学题库
# ============================================================

SAMPLE_MATH_PROBLEMS = [
    {"problem": "计算 (3 + 5) × 2 - 4 的值。", "answer": "12"},
    {"problem": "一个长方形的长是 12cm，宽是 8cm，求它的面积和周长。", "answer": "96"},
    {"problem": "如果 2x + 5 = 17，求 x 的值。", "answer": "6"},
    {"problem": "一个三角形三条边长分别为 3, 4, 5，判断这是什么三角形并求面积。", "answer": "6"},
    {"problem": "从 1 到 100 的所有整数之和是多少？", "answer": "5050"},
    {"problem": "一个圆的半径是 7cm，求它的面积（取 π = 3.14）。", "answer": "153.86"},
    {"problem": "甲乙两人从相距 100km 的两地相向而行，甲速 30km/h，乙速 20km/h，几小时后相遇？", "answer": "2"},
    {"problem": "计算组合数 C(10, 3) 的值。", "answer": "120"},
    {"problem": "求方程 x^2 - 5x + 6 = 0 的两个根。", "answer": "2"},
    {"problem": "一个等差数列首项 a1=3，公差 d=2，求前 20 项的和。", "answer": "440"},
    {"problem": "计算 log_2(32) 的值。", "answer": "5"},
    {"problem": "一个骰子投两次，两次之和为 7 的概率是多少？", "answer": "1/6"},
    {"problem": "如果 f(x) = x^3 - 3x + 1，求 f'(2) 的值。", "answer": "9"},
    {"problem": "一个等比数列首项 1，公比 2，求前 10 项之和。", "answer": "1023"},
    {"problem": "计算积分 ∫(0 到 1) x^2 dx 的值。", "answer": "1/3"},
]


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Self-Play 数学推理数据生成")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                        help="OpenAI 兼容 API 地址")
    parser.add_argument("--api-key", type=str, default="not-needed", help="API 密钥")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="模型名称")
    parser.add_argument("--num-samples", type=int, default=8, help="每题采样次数 K")
    parser.add_argument("--num-rounds", type=int, default=3, help="Self-Play 轮数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--output-dir", type=str, default="self_play_output", help="输出目录")
    parser.add_argument("--problem-file", type=str, default=None,
                        help="题库 JSON 文件（每项含 problem 和 answer 字段），不指定则使用示例题库")
    args = parser.parse_args()

    # 加载题库
    if args.problem_file:
        problems = json.loads(Path(args.problem_file).read_text(encoding="utf-8"))
    else:
        problems = SAMPLE_MATH_PROBLEMS
        logger.info("使用内置示例题库 (%d 题)", len(problems))

    # 运行 Self-Play
    engine = SelfPlayEngine(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        num_samples=args.num_samples,
        temperature=args.temperature,
    )

    results = engine.run(
        problems=problems,
        num_rounds=args.num_rounds,
        output_dir=args.output_dir,
    )
