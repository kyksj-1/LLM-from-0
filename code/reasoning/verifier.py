"""
简化版 Verifier 模型 (ORM/PRM)

本模块实现了两种 Verifier 模型的简化版本：
- ORM (Outcome Reward Model): 只评估最终答案
- PRM (Process Reward Model): 评估每个推理步骤

核心公式:
- ORM: V_ORM(q, a) = σ(f_φ(q, a))
- PRM: V_PRM(q, r₁, ..., r_K) = Π_k P(step r_k correct | q, r₁, ..., r_{k-1})

参考:
- Cobbe et al. (2021). Training Verifiers to Solve Math Word Problems.
- Lightman et al. (2023). Let's Verify Step by Step.
"""

import re
import math
import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class VerificationResult:
    """
    验证结果

    Attributes:
        overall_score: 总体分数 (0-1)
        step_scores: 每个步骤的分数列表
        reasoning_steps: 推理步骤文本列表
        error_step: 第一个错误步骤的索引（-1 表示无错误）
        method: 使用的验证方法 ("orm" 或 "prm")
    """
    overall_score: float
    step_scores: List[float]
    reasoning_steps: List[str]
    error_step: int = -1
    method: str = "orm"


class OutcomeRewardModel(nn.Module):
    """
    结果奖励模型 (ORM)

    只评估最终答案的正确性，不关心中间推理过程。

    V_ORM(q, a) = σ(MLP(concat(q_emb, a_emb)))

    简化实现：使用简单的 MLP 对问题和答案的嵌入进行评分。
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        """
        初始化 ORM

        Args:
            input_dim: 输入嵌入维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()

        # 问题编码器
        self.question_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 答案编码器
        self.answer_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 评分头：将问题和答案的表示合并后输出分数
        self.scoring_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # 输出 0-1 之间的分数
        )

    def forward(
        self,
        question_emb: torch.Tensor,
        answer_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            question_emb: 问题嵌入 [batch, input_dim]
            answer_emb: 答案嵌入 [batch, input_dim]

        Returns:
            验证分数 [batch, 1]，范围 (0, 1)
        """
        # 编码
        q_repr = self.question_encoder(question_emb)
        a_repr = self.answer_encoder(answer_emb)

        # 拼接并评分
        combined = torch.cat([q_repr, a_repr], dim=-1)
        score = self.scoring_head(combined)

        return score

    def compute_loss(
        self,
        question_emb: torch.Tensor,
        answer_emb: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 ORM 训练损失

        使用二元交叉熵损失:
        L = -[y·log(V) + (1-y)·log(1-V)]

        Args:
            question_emb: 问题嵌入 [batch, input_dim]
            answer_emb: 答案嵌入 [batch, input_dim]
            labels: 标签 [batch, 1]，1=正确 0=错误

        Returns:
            损失值
        """
        scores = self.forward(question_emb, answer_emb)
        loss = nn.functional.binary_cross_entropy(scores, labels)
        return loss


class ProcessRewardModel(nn.Module):
    """
    过程奖励模型 (PRM)

    评估每个推理步骤的正确性，而非只看最终答案。

    V_PRM(q, r₁, ..., r_K) = Π_k f_φ(q, r₁, ..., r_k)

    简化实现：使用 Transformer 编码上下文，
    对每个推理步骤位置输出正确性分数。
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        """
        初始化 PRM

        Args:
            input_dim: 输入嵌入维度
            hidden_dim: 隐藏层维度
            n_heads: 注意力头数
            n_layers: Transformer 层数
        """
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 使用 Transformer 编码推理上下文
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # 步骤评分头：对每个步骤输出正确性分数
        self.step_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        step_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        对每个推理步骤输出正确性分数。

        Args:
            step_embeddings: 步骤嵌入 [batch, n_steps, input_dim]
            mask: 注意力掩码 [batch, n_steps]

        Returns:
            每步的正确性分数 [batch, n_steps, 1]
        """
        # 投影到隐藏维度
        x = self.input_proj(step_embeddings)

        # Transformer 编码（使用因果掩码，每步只能看到前面的步骤）
        seq_len = x.shape[1]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        x = self.transformer(x, mask=causal_mask)

        # 对每步评分
        step_scores = self.step_scorer(x)

        return step_scores

    def compute_overall_score(
        self,
        step_embeddings: torch.Tensor,
        aggregation: str = "product",
    ) -> torch.Tensor:
        """
        计算整体验证分数

        Args:
            step_embeddings: 步骤嵌入 [batch, n_steps, input_dim]
            aggregation: 聚合方式
                - "product": Π_k score_k（乘积，严格）
                - "min": min_k score_k（最小值，更严格）
                - "mean": mean_k score_k（平均，宽松）

        Returns:
            整体分数 [batch]
        """
        step_scores = self.forward(step_embeddings).squeeze(-1)

        if aggregation == "product":
            # 乘积聚合：任何一步出错都会显著降低总分
            overall = torch.prod(step_scores, dim=-1)
        elif aggregation == "min":
            # 最小值聚合：总分 = 最弱步骤的分数
            overall = torch.min(step_scores, dim=-1).values
        elif aggregation == "mean":
            # 平均聚合：较为宽松
            overall = torch.mean(step_scores, dim=-1)
        else:
            raise ValueError(f"不支持的聚合方式: {aggregation}")

        return overall

    def compute_loss(
        self,
        step_embeddings: torch.Tensor,
        step_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算 PRM 训练损失

        对每个步骤使用二元交叉熵损失。

        Args:
            step_embeddings: 步骤嵌入 [batch, n_steps, input_dim]
            step_labels: 每步标签 [batch, n_steps]，1=正确 0=错误
            mask: 有效步骤掩码 [batch, n_steps]

        Returns:
            损失值
        """
        step_scores = self.forward(step_embeddings).squeeze(-1)

        # 二元交叉熵损失
        loss = nn.functional.binary_cross_entropy(step_scores, step_labels, reduction="none")

        # 如果有掩码，只计算有效步骤的损失
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()

        return loss


class RuleBasedVerifier:
    """
    基于规则的 Verifier

    不依赖神经网络，使用启发式规则对推理过程进行验证。
    适合作为简单基线或在没有训练数据时使用。

    评估维度:
    1. 格式规范性 (0-0.3)
    2. 推理一致性 (0-0.4)
    3. 答案合理性 (0-0.3)
    """

    def __init__(self):
        """初始化规则验证器"""
        # 步骤分隔模式
        self.step_patterns = [
            r"(?:步骤|Step|第)\s*\d+",  # "步骤1" 或 "Step 1"
            r"\d+[\.\)]\s*",  # "1. " 或 "1) "
            r"(?:首先|然后|接着|最后|因此)",  # 中文连接词
        ]

    def extract_steps(self, reasoning: str) -> List[str]:
        """
        从推理文本中提取步骤

        Args:
            reasoning: 推理过程文本

        Returns:
            步骤列表
        """
        # 按换行分割
        lines = [line.strip() for line in reasoning.split('\n') if line.strip()]

        if len(lines) <= 1:
            # 尝试按句号分割
            lines = [s.strip() for s in reasoning.split('。') if s.strip()]

        return lines

    def extract_numbers(self, text: str) -> List[float]:
        """
        从文本中提取数值

        Args:
            text: 输入文本

        Returns:
            数值列表
        """
        numbers = re.findall(r'-?\d+\.?\d*', text)
        return [float(n) for n in numbers]

    def check_format(self, reasoning: str) -> float:
        """
        检查格式规范性 (0-0.3)

        Args:
            reasoning: 推理过程

        Returns:
            格式分数
        """
        score = 0.0

        # 是否包含清晰的步骤标记
        steps = self.extract_steps(reasoning)
        if len(steps) >= 2:
            score += 0.1  # 有多个步骤

        # 是否包含最终答案标记
        answer_markers = ["答案", "结果", "所以", "因此", "answer", "result"]
        if any(m in reasoning.lower() for m in answer_markers):
            score += 0.1

        # 格式整齐（没有异常字符、混乱文本）
        if len(reasoning) > 10 and not re.search(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\+\-\*\/\=\.\,\;\:\!\?\(\)\[\]\{\}]', reasoning[:200]):
            score += 0.1

        return score

    def check_consistency(self, reasoning: str) -> float:
        """
        检查推理一致性 (0-0.4)

        检查步骤之间的数值是否一致。

        Args:
            reasoning: 推理过程

        Returns:
            一致性分数
        """
        steps = self.extract_steps(reasoning)
        if len(steps) < 2:
            return 0.2  # 无法检查一致性，给基础分

        score = 0.0
        step_score = 0.4 / len(steps)

        for i, step in enumerate(steps):
            # 检查每步中的等式是否正确
            # 匹配简单的算术等式: a + b = c, a - b = c, a * b = c, a / b = c
            equations = re.findall(r'(\d+\.?\d*)\s*([+\-*/×÷])\s*(\d+\.?\d*)\s*[=＝]\s*(\d+\.?\d*)', step)

            step_valid = True
            for eq in equations:
                a, op, b, result = float(eq[0]), eq[1], float(eq[2]), float(eq[3])
                expected = None
                if op in ['+', '＋']:
                    expected = a + b
                elif op in ['-', '－']:
                    expected = a - b
                elif op in ['*', '×']:
                    expected = a * b
                elif op in ['/', '÷']:
                    expected = a / b if b != 0 else None

                if expected is not None and abs(expected - result) > 0.01:
                    step_valid = False
                    break

            if step_valid:
                score += step_score

        return score

    def check_answer(self, reasoning: str, expected_answer: Optional[str] = None) -> float:
        """
        检查答案合理性 (0-0.3)

        Args:
            reasoning: 推理过程
            expected_answer: 预期答案（如果有）

        Returns:
            答案分数
        """
        score = 0.0

        # 提取答案
        numbers = self.extract_numbers(reasoning)
        if numbers:
            score += 0.1  # 至少包含数值

            # 最后一个数字通常是答案
            answer = numbers[-1]

            # 检查是否在合理范围内（不是极端值）
            if -1e6 < answer < 1e6:
                score += 0.1

        # 如果提供了预期答案，检查是否匹配
        if expected_answer is not None:
            try:
                expected_num = float(expected_answer)
                if numbers and abs(numbers[-1] - expected_num) < 0.01:
                    score += 0.1
            except ValueError:
                # 非数值答案，尝试字符串匹配
                if expected_answer in reasoning:
                    score += 0.1

        return score

    def verify(
        self,
        question: str,
        reasoning: str,
        expected_answer: Optional[str] = None,
    ) -> VerificationResult:
        """
        完整验证

        Args:
            question: 问题
            reasoning: 推理过程
            expected_answer: 预期答案

        Returns:
            验证结果
        """
        # 各维度评分
        format_score = self.check_format(reasoning)
        consistency_score = self.check_consistency(reasoning)
        answer_score = self.check_answer(reasoning, expected_answer)

        # 总分
        overall = format_score + consistency_score + answer_score

        # 步骤级评分
        steps = self.extract_steps(reasoning)
        step_scores = []
        error_step = -1

        for i, step in enumerate(steps):
            # 简单的步骤评分：检查是否包含有效内容
            step_s = 0.5  # 基础分
            if self.extract_numbers(step):
                step_s += 0.3
            equations = re.findall(r'\d+\s*[+\-*/×÷]\s*\d+\s*[=＝]\s*\d+', step)
            if equations:
                step_s += 0.2

            step_scores.append(min(step_s, 1.0))

            # 记录第一个低分步骤
            if step_s < 0.5 and error_step == -1:
                error_step = i

        return VerificationResult(
            overall_score=overall,
            step_scores=step_scores,
            reasoning_steps=steps,
            error_step=error_step,
            method="rule_based",
        )


def demonstrate_orm():
    """演示 ORM 的使用"""
    print("=" * 70)
    print("Outcome Reward Model (ORM) 演示")
    print("=" * 70)

    # 创建 ORM 模型
    input_dim = 64
    orm = OutcomeRewardModel(input_dim=input_dim, hidden_dim=32)

    # 模拟输入
    batch_size = 4
    question_emb = torch.randn(batch_size, input_dim)
    answer_emb = torch.randn(batch_size, input_dim)
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]])

    # 前向传播
    scores = orm(question_emb, answer_emb)
    print(f"\n输入维度: question={question_emb.shape}, answer={answer_emb.shape}")
    print(f"ORM 评分: {scores.squeeze().tolist()}")
    print(f"真实标签: {labels.squeeze().tolist()}")

    # 计算损失
    loss = orm.compute_loss(question_emb, answer_emb, labels)
    print(f"BCE 损失: {loss.item():.4f}")

    # 参数量
    n_params = sum(p.numel() for p in orm.parameters())
    print(f"ORM 参数量: {n_params:,}")


def demonstrate_prm():
    """演示 PRM 的使用"""
    print("\n" + "=" * 70)
    print("Process Reward Model (PRM) 演示")
    print("=" * 70)

    # 创建 PRM 模型
    input_dim = 64
    prm = ProcessRewardModel(input_dim=input_dim, hidden_dim=64, n_heads=4, n_layers=2)

    # 模拟输入：2个样本，每个有5个推理步骤
    batch_size = 2
    n_steps = 5
    step_embs = torch.randn(batch_size, n_steps, input_dim)

    # 步骤标签：1=正确 0=错误
    # 样本1: 步骤 1-4 正确，步骤 5 错误
    # 样本2: 全部正确
    step_labels = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
    ])

    # 步骤级评分
    step_scores = prm(step_embs)
    print(f"\n输入维度: step_embeddings={step_embs.shape}")
    print(f"步骤评分 (样本1): {step_scores[0].squeeze().tolist()}")
    print(f"步骤评分 (样本2): {step_scores[1].squeeze().tolist()}")
    print(f"步骤标签 (样本1): {step_labels[0].tolist()}")
    print(f"步骤标签 (样本2): {step_labels[1].tolist()}")

    # 不同聚合方式
    for agg in ["product", "min", "mean"]:
        overall = prm.compute_overall_score(step_embs, aggregation=agg)
        print(f"\n聚合方式 '{agg}': {overall.tolist()}")

    # 计算损失
    loss = prm.compute_loss(step_embs, step_labels)
    print(f"\n步骤级 BCE 损失: {loss.item():.4f}")

    # 参数量
    n_params = sum(p.numel() for p in prm.parameters())
    print(f"PRM 参数量: {n_params:,}")


def demonstrate_rule_based_verifier():
    """演示基于规则的 Verifier"""
    print("\n" + "=" * 70)
    print("基于规则的 Verifier 演示")
    print("=" * 70)

    verifier = RuleBasedVerifier()

    # 测试用例1：正确的推理
    print("\n--- 测试1: 正确的推理 ---")
    reasoning_good = (
        "步骤1: 小明有15颗糖果\n"
        "步骤2: 给了小红5颗，剩余 15 - 5 = 10 颗\n"
        "步骤3: 从小华得到8颗，共有 10 + 8 = 18 颗\n"
        "步骤4: 给妈妈一半，18 / 2 = 9 颗\n"
        "因此答案是9"
    )
    result = verifier.verify("小明有多少糖果？", reasoning_good, expected_answer="9")
    print(f"总分: {result.overall_score:.3f}")
    print(f"步骤分: {[f'{s:.2f}' for s in result.step_scores]}")
    print(f"错误步骤: {result.error_step}")

    # 测试用例2：包含计算错误的推理
    print("\n--- 测试2: 包含计算错误的推理 ---")
    reasoning_bad = (
        "步骤1: 小明有15颗糖果\n"
        "步骤2: 给了小红5颗，剩余 15 - 5 = 11 颗\n"  # 错误！
        "步骤3: 从小华得到8颗，共有 11 + 8 = 19 颗\n"
        "答案是19"
    )
    result = verifier.verify("小明有多少糖果？", reasoning_bad, expected_answer="9")
    print(f"总分: {result.overall_score:.3f}")
    print(f"步骤分: {[f'{s:.2f}' for s in result.step_scores]}")
    print(f"错误步骤: {result.error_step}")

    # 测试用例3：格式混乱的推理
    print("\n--- 测试3: 简短推理 ---")
    reasoning_short = "15-5+8=18, 18/2=9"
    result = verifier.verify("小明有多少糖果？", reasoning_short, expected_answer="9")
    print(f"总分: {result.overall_score:.3f}")
    print(f"步骤分: {[f'{s:.2f}' for s in result.step_scores]}")

    # ORM vs PRM 对比示例
    print("\n--- ORM vs PRM 直觉对比 ---")
    print("推理: step1(正确) -> step2(错误) -> step3(正确) -> 答案(碰巧正确)")
    print("ORM: 只看最终答案正确 → 高分")
    print("PRM: 发现 step2 错误 → 总分被拉低")
    print("结论: PRM 更严格，能发现中间错误")


if __name__ == "__main__":
    # ORM 演示
    demonstrate_orm()

    # PRM 演示
    demonstrate_prm()

    # 基于规则的 Verifier 演示
    demonstrate_rule_based_verifier()
