"""
评估框架

知识依赖:
- 模块 8B（Scaling Laws）: 困惑度计算
- 模块 13（CoT 与推理）: 推理能力评估

参考实现:
- code/pretraining_objectives/perplexity.py
- code/reasoning/evaluation.py

评估维度:
1. 困惑度 (Perplexity): 衡量语言建模能力的基础指标
   PPL = exp(-1/N * Σ log P(x_i | x_{<i}))
   越低越好，表示模型对数据的预测能力越强

2. 下游任务评估:
   - 阅读理解: 给定段落回答问题
   - 知识问答: 常识推理
   - 代码生成: HumanEval 风格
   - 数学推理: GSM8K 风格

3. 安全性评估:
   - 有害内容生成率
   - 拒绝回答有害请求的能力
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class Evaluator:
    """
    模型评估器

    Args:
        model: 语言模型
        tokenizer: 分词器
        device: 设备
    """

    def __init__(self, model: nn.Module, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def evaluate_perplexity(self, eval_data) -> float:
        """
        计算困惑度

        Args:
            eval_data: 评估数据 DataLoader

        Returns:
            困惑度值

        实现步骤:
            1. total_loss = 0, total_tokens = 0
            2. for batch in eval_data:
                   loss = model(input_ids, labels)["loss"]
                   total_loss += loss.item() * n_tokens
                   total_tokens += n_tokens
            3. avg_loss = total_loss / total_tokens
            4. perplexity = exp(avg_loss)
        """
        raise NotImplementedError(
            "TODO: 计算困惑度\n"
            "参考: 模块 8B 的困惑度章节\n"
            "参考实现: code/pretraining_objectives/perplexity.py"
        )

    @torch.no_grad()
    def evaluate_few_shot(
        self,
        task_name: str,
        examples: List[Dict],
        n_shot: int = 5,
    ) -> Dict[str, float]:
        """
        Few-shot 评估

        构造 prompt: 将 n_shot 个示例拼接为上下文，
        然后让模型预测最后一个问题的答案。

        Args:
            task_name: 任务名（如 "hellaswag", "arc_easy"）
            examples: 评估样本列表
            n_shot: few-shot 示例数

        Returns:
            {"accuracy": float, "n_samples": int}
        """
        raise NotImplementedError(
            "TODO: 实现 few-shot 评估\n"
            "参考: 模块 13 的评估章节\n"
            "提示: lm-eval-harness 库提供了标准评估框架"
        )

    def run_full_evaluation(self, eval_data, tasks: Optional[List[str]] = None) -> Dict:
        """
        运行完整评估

        Returns:
            {
                "perplexity": float,
                "tasks": {task_name: accuracy, ...}
            }
        """
        raise NotImplementedError(
            "TODO: 运行完整评估套件\n"
            "提示: 生产环境推荐使用 lm-eval-harness"
        )
