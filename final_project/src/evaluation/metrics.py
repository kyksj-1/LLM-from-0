"""
评估指标

知识依赖:
- 模块 8B（Scaling Laws）: 困惑度
- 模块 13（CoT 与推理）: 推理任务评估

参考实现:
- code/pretraining_objectives/perplexity.py
- code/reasoning/evaluation.py

常用评估指标:

1. 困惑度 (Perplexity):
   PPL = exp(-1/N * Σ log P(x_i | x_{<i}))
   适用: 预训练模型的基础质量

2. 准确率 (Accuracy):
   acc = correct / total
   适用: 分类任务、选择题

3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation):
   ROUGE-L: 基于最长公共子序列的 F1 分数
   适用: 文本生成质量、摘要

4. BLEU:
   基于 n-gram 精确率
   适用: 翻译质量

5. Pass@k:
   在 k 次采样中至少有一次正确的概率
   适用: 代码生成（HumanEval）
"""

import math
from typing import List, Optional


def perplexity(log_probs: List[float], n_tokens: int) -> float:
    """
    计算困惑度

    Args:
        log_probs: 每个 token 的 log 概率列表
        n_tokens: token 总数

    Returns:
        困惑度

    公式: PPL = exp(-sum(log_probs) / n_tokens)
    """
    raise NotImplementedError(
        "TODO: 实现困惑度计算\n"
        "参考: 模块 8B 的困惑度公式"
    )


def accuracy(predictions: List, targets: List) -> float:
    """
    计算准确率

    Args:
        predictions: 预测列表
        targets: 真实标签列表

    Returns:
        准确率 (0~1)
    """
    raise NotImplementedError("TODO: 实现准确率计算")


def rouge_l(prediction: str, reference: str) -> float:
    """
    计算 ROUGE-L F1 分数

    基于最长公共子序列 (LCS):
    precision = len(LCS) / len(prediction)
    recall = len(LCS) / len(reference)
    F1 = 2 * precision * recall / (precision + recall)

    Args:
        prediction: 模型生成的文本
        reference: 参考文本

    Returns:
        ROUGE-L F1 分数
    """
    raise NotImplementedError(
        "TODO: 实现 ROUGE-L\n"
        "提示: 先实现 LCS (动态规划)，再计算 F1"
    )


def pass_at_k(n_samples: int, n_correct: int, k: int) -> float:
    """
    计算 Pass@k 指标（代码生成评估）

    数学: pass@k = 1 - C(n-c, k) / C(n, k)
    其中 n = n_samples, c = n_correct

    Args:
        n_samples: 总采样次数
        n_correct: 正确次数
        k: k 值

    Returns:
        pass@k 概率
    """
    raise NotImplementedError(
        "TODO: 实现 Pass@k\n"
        "参考: Chen et al., 'Evaluating Large Language Models Trained on Code' (2021)"
    )
