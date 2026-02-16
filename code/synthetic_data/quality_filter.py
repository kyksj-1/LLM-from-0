"""
数据质量过滤器: 基于 LLM 打分 + 规则 + 困惑度的合成数据过滤管线

本模块实现了多层数据质量过滤体系:
1. 规则过滤: 长度、重复率、格式检查
2. LLM-as-Judge 评分: 用 LLM 对数据质量打分
3. 困惑度过滤: 剔除 PPL 过高或过低的异常数据

依赖:
- openai: OpenAI 兼容 API 客户端
- transformers + torch: 计算困惑度 (可选)

安装:
    pip install openai transformers torch

参考:
- Gunasekar et al. (2023). Textbooks Are All You Need. (Phi-1 的数据过滤)
"""

import json
import re
import math
import logging
from collections import Counter
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 数据结构
# ============================================================

@dataclass
class QualityScore:
    """质量评分结果"""
    instruction: str
    response: str
    rule_passed: bool
    rule_details: dict
    llm_score: Optional[float] = None
    llm_reasoning: Optional[str] = None
    perplexity: Optional[float] = None
    final_passed: bool = False


# ============================================================
# 规则过滤器
# ============================================================

class RuleFilter:
    """
    基于规则的快速过滤器

    过滤条件:
    - 长度: 回答不能过短或过长
    - n-gram 重复率: 检测生成退化
    - 格式: 检测 LLM 角色泄露和元表述
    - 语言一致性: 检测编码错误
    """

    def __init__(
        self,
        min_response_length: int = 50,
        max_response_length: int = 10000,
        max_ngram_repeat_rate: float = 0.3,
        ngram_size: int = 10,
    ):
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.max_ngram_repeat_rate = max_ngram_repeat_rate
        self.ngram_size = ngram_size

    def compute_ngram_repeat_rate(self, text: str, n: int) -> float:
        """
        计算 n-gram 重复率

        RepRate(T, n) = 1 - |unique n-grams| / |total n-grams|

        Args:
            text: 输入文本
            n: n-gram 大小

        Returns:
            重复率 [0, 1]，越高表示越多重复
        """
        words = text.split()
        if len(words) < n:
            return 0.0

        ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
        total = len(ngrams)
        unique = len(set(ngrams))

        if total == 0:
            return 0.0
        return 1.0 - unique / total

    def check_meta_expressions(self, text: str) -> list[str]:
        """检测 LLM 角色泄露和元表述"""
        issues = []
        patterns = [
            (r"作为(一个)?AI(语言)?模型", "AI角色泄露"),
            (r"As an AI( language model)?", "AI角色泄露(英文)"),
            (r"I('m| am) (just )?an? AI", "AI角色泄露(英文)"),
            (r"我(是|只是)(一个)?语言模型", "语言模型自述"),
            (r"I (don't|cannot|can't) (actually )?have", "能力否认"),
            (r"我(没有|无法)(真正|实际)", "能力否认(中文)"),
        ]
        for pattern, label in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(label)
        return issues

    def check_special_chars(self, text: str) -> float:
        """计算特殊字符/乱码比例"""
        if not text:
            return 0.0
        # 控制字符（排除换行和制表符）
        special_count = sum(1 for c in text if ord(c) < 32 and c not in "\n\t\r")
        return special_count / len(text)

    def filter(self, instruction: str, response: str) -> tuple[bool, dict]:
        """
        执行规则过滤

        Returns:
            (是否通过, 详细信息字典)
        """
        details = {
            "response_length": len(response),
            "repeat_rate": 0.0,
            "meta_issues": [],
            "special_char_rate": 0.0,
            "reasons": [],
        }

        passed = True

        # 长度检查
        if len(response) < self.min_response_length:
            details["reasons"].append(f"回答过短: {len(response)} < {self.min_response_length}")
            passed = False
        if len(response) > self.max_response_length:
            details["reasons"].append(f"回答过长: {len(response)} > {self.max_response_length}")
            passed = False

        # n-gram 重复率
        repeat_rate = self.compute_ngram_repeat_rate(response, self.ngram_size)
        details["repeat_rate"] = round(repeat_rate, 4)
        if repeat_rate > self.max_ngram_repeat_rate:
            details["reasons"].append(f"重复率过高: {repeat_rate:.4f} > {self.max_ngram_repeat_rate}")
            passed = False

        # 元表述检测
        meta_issues = self.check_meta_expressions(response)
        details["meta_issues"] = meta_issues
        if meta_issues:
            details["reasons"].append(f"元表述: {', '.join(meta_issues)}")
            passed = False

        # 特殊字符检测
        special_rate = self.check_special_chars(response)
        details["special_char_rate"] = round(special_rate, 6)
        if special_rate > 0.01:
            details["reasons"].append(f"特殊字符过多: {special_rate:.4%}")
            passed = False

        return passed, details


# ============================================================
# LLM-as-Judge 评分器
# ============================================================

LLM_JUDGE_PROMPT = """请评估以下指令-回答对的质量。

评分标准（每项 1-5 分）：
1. 准确性：信息是否正确？
2. 完整性：是否充分回答了问题？
3. 清晰度：表达是否清楚、有条理？
4. 有帮助：对提问者是否有实际帮助？

请严格按照以下 JSON 格式输出（不要输出其他内容）：
{{"accuracy": <1-5>, "completeness": <1-5>, "clarity": <1-5>, "helpfulness": <1-5>, "overall": <1-5>, "reasoning": "<简要说明>"}}

---
指令: {instruction}

回答: {response}

评分（JSON）:"""


class LLMJudge:
    """LLM-as-Judge: 使用 LLM 对数据质量进行评分"""

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        model: str = "deepseek-chat",
        threshold: float = 3.0,
    ):
        """
        Args:
            api_base: OpenAI 兼容 API 地址
            api_key: API 密钥
            model: 评分模型名称
            threshold: 最低通过分数（overall >= threshold 才通过）
        """
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.threshold = threshold

    def score(self, instruction: str, response: str) -> tuple[float, str]:
        """
        对指令-回答对进行评分

        Returns:
            (综合分数, 评分理由)
        """
        prompt = LLM_JUDGE_PROMPT.format(instruction=instruction, response=response)

        try:
            result = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 低温度以提高评分一致性
                max_tokens=256,
            )
            content = result.choices[0].message.content.strip()

            # 解析 JSON 响应
            # 尝试提取 JSON 部分（LLM 可能在 JSON 前后加文字）
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                scores = json.loads(json_match.group())
                overall = float(scores.get("overall", 0))
                reasoning = scores.get("reasoning", "")
                return overall, reasoning

            logger.warning("无法解析 LLM 评分响应: %s", content[:200])
            return 0.0, "解析失败"

        except Exception as e:
            logger.error("LLM 评分调用失败: %s", e)
            return 0.0, f"API 错误: {e}"


# ============================================================
# 困惑度计算器
# ============================================================

class PerplexityCalculator:
    """
    基于预训练语言模型的困惑度计算器

    困惑度 (PPL) 反映模型对文本的"意外程度":
    - PPL 过低: 文本过于模板化/重复
    - PPL 过高: 文本不通顺/噪声
    - 中间区间: 正常的、有信息量的文本

    PPL(y) = exp(-1/|y| * sum(log P(y_t | y_<t)))
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        ppl_low: float = 5.0,
        ppl_high: float = 500.0,
        device: str = "cpu",
    ):
        """
        Args:
            model_name: HuggingFace 模型名称
            ppl_low: PPL 下界（低于此值视为异常）
            ppl_high: PPL 上界（高于此值视为异常）
            device: 计算设备
        """
        self.model_name = model_name
        self.ppl_low = ppl_low
        self.ppl_high = ppl_high
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            logger.info("加载困惑度计算模型: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            self._model.eval()

    def compute(self, text: str, max_length: int = 512) -> float:
        """
        计算文本的困惑度

        Args:
            text: 输入文本
            max_length: 最大 token 数

        Returns:
            困惑度值
        """
        import torch

        self._load_model()

        encodings = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss.item()

        return math.exp(loss)

    def is_in_range(self, ppl: float) -> bool:
        """检查困惑度是否在合理范围内"""
        return self.ppl_low <= ppl <= self.ppl_high


# ============================================================
# 综合过滤管线
# ============================================================

class QualityFilterPipeline:
    """
    综合数据质量过滤管线

    三层过滤:
    1. 规则过滤（快速、无 API 调用）
    2. LLM-as-Judge 评分（精确、需 API 调用）
    3. 困惑度过滤（可选、需本地模型）

    使用策略: 先用规则过滤去除明显差数据，再用 LLM 评分精选
    """

    def __init__(
        self,
        rule_filter: Optional[RuleFilter] = None,
        llm_judge: Optional[LLMJudge] = None,
        ppl_calculator: Optional[PerplexityCalculator] = None,
    ):
        self.rule_filter = rule_filter or RuleFilter()
        self.llm_judge = llm_judge
        self.ppl_calculator = ppl_calculator

    def process_one(self, instruction: str, response: str) -> QualityScore:
        """对一条数据执行完整的质量评估"""
        result = QualityScore(instruction=instruction, response=response, rule_passed=False, rule_details={})

        # 第一层: 规则过滤
        rule_passed, rule_details = self.rule_filter.filter(instruction, response)
        result.rule_passed = rule_passed
        result.rule_details = rule_details

        if not rule_passed:
            result.final_passed = False
            return result

        # 第二层: LLM 评分
        if self.llm_judge:
            score, reasoning = self.llm_judge.score(instruction, response)
            result.llm_score = score
            result.llm_reasoning = reasoning

            if score < self.llm_judge.threshold:
                result.final_passed = False
                return result

        # 第三层: 困惑度过滤
        if self.ppl_calculator:
            ppl = self.ppl_calculator.compute(response)
            result.perplexity = ppl

            if not self.ppl_calculator.is_in_range(ppl):
                result.final_passed = False
                return result

        result.final_passed = True
        return result

    def process_batch(
        self,
        data: list[dict],
        instruction_key: str = "instruction",
        response_key: str = "response",
    ) -> tuple[list[dict], list[QualityScore]]:
        """
        批量过滤数据

        Args:
            data: 数据列表，每项是包含 instruction 和 response 的字典
            instruction_key: 指令字段名
            response_key: 回答字段名

        Returns:
            (通过的数据列表, 所有评分结果列表)
        """
        passed_data = []
        all_scores = []

        for i, item in enumerate(data):
            instruction = item.get(instruction_key, "")
            response = item.get(response_key, "")

            score = self.process_one(instruction, response)
            all_scores.append(score)

            if score.final_passed:
                passed_data.append(item)

            if (i + 1) % 10 == 0:
                logger.info("进度: %d / %d, 通过: %d", i + 1, len(data), len(passed_data))

        logger.info(
            "过滤完成: 总计 %d 条, 通过 %d 条 (%.1f%%)",
            len(data), len(passed_data), 100 * len(passed_data) / max(len(data), 1),
        )
        return passed_data, all_scores


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="合成数据质量过滤器")
    parser.add_argument("input_file", type=str, help="输入 JSON 文件（包含 instruction 和 response 字段）")
    parser.add_argument("--output", type=str, default="filtered_output.json", help="输出文件路径")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1", help="LLM API 地址")
    parser.add_argument("--api-key", type=str, default="not-needed", help="API 密钥")
    parser.add_argument("--model", type=str, default="deepseek-chat", help="评分模型名称")
    parser.add_argument("--threshold", type=float, default=3.0, help="LLM 最低通过分数")
    parser.add_argument("--no-llm", action="store_true", help="跳过 LLM 评分（只用规则过滤）")
    parser.add_argument("--use-ppl", action="store_true", help="启用困惑度过滤（需本地模型）")
    parser.add_argument("--ppl-model", type=str, default="gpt2", help="困惑度计算模型")
    args = parser.parse_args()

    # 加载数据
    data = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    logger.info("加载了 %d 条数据", len(data))

    # 构建过滤管线
    rule_filter = RuleFilter()

    llm_judge = None
    if not args.no_llm:
        llm_judge = LLMJudge(
            api_base=args.api_base,
            api_key=args.api_key,
            model=args.model,
            threshold=args.threshold,
        )

    ppl_calc = None
    if args.use_ppl:
        ppl_calc = PerplexityCalculator(model_name=args.ppl_model)

    pipeline = QualityFilterPipeline(
        rule_filter=rule_filter,
        llm_judge=llm_judge,
        ppl_calculator=ppl_calc,
    )

    # 执行过滤
    passed, scores = pipeline.process_batch(data)

    # 保存结果
    Path(args.output).write_text(json.dumps(passed, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("过滤后数据已保存到 %s", args.output)

    # 统计信息
    rule_passed = sum(1 for s in scores if s.rule_passed)
    llm_scored = [s for s in scores if s.llm_score is not None]
    avg_llm_score = sum(s.llm_score for s in llm_scored) / max(len(llm_scored), 1)

    print("\n=== 过滤统计 ===")
    print(f"输入数据: {len(data)} 条")
    print(f"规则过滤通过: {rule_passed} 条 ({100 * rule_passed / max(len(data), 1):.1f}%)")
    if llm_scored:
        print(f"LLM 平均分: {avg_llm_score:.2f}")
    print(f"最终通过: {len(passed)} 条 ({100 * len(passed) / max(len(data), 1):.1f}%)")

    # 规则过滤失败原因统计
    all_reasons = []
    for s in scores:
        all_reasons.extend(s.rule_details.get("reasons", []))
    reason_counts = Counter(r.split(":")[0] for r in all_reasons)
    if reason_counts:
        print("\n规则过滤失败原因分布:")
        for reason, count in reason_counts.most_common():
            print(f"  {reason}: {count}")
