"""
DPO 偏好优化训练器

知识依赖:
- 模块 12（DPO 及变体）: DPO 损失函数推导、β 参数、参考模型

参考实现:
- code/dpo/dpo_loss.py
- code/dpo/dpo_trainer.py

DPO 损失函数推导:

    RLHF 目标: max E[r(x,y)] - β * KL(π || π_ref)

    DPO 关键洞察: 最优策略可以用奖励函数表示
    r(x,y) = β * log(π(y|x) / π_ref(y|x)) + β * log Z(x)

    代入 Bradley-Terry 偏好模型:
    P(y_w > y_l | x) = σ(r(x,y_w) - r(x,y_l))

    消去 Z(x) 后得到 DPO 损失:
    L_DPO = -E[log σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

    其中:
    - π: 当前策略（正在训练的模型）
    - π_ref: 参考策略（冻结的 SFT 模型）
    - y_w: chosen response（人类偏好的）
    - y_l: rejected response（人类不偏好的）
    - β: 控制偏离参考模型的程度（典型值 0.1）
    - σ: sigmoid 函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class DPOTrainer:
    """
    DPO 偏好优化训练器

    需要两个模型:
    - policy_model: 正在训练的模型（从 SFT checkpoint 初始化）
    - ref_model: 参考模型（冻结的 SFT 模型副本）

    Args:
        policy_model: 策略模型
        ref_model: 参考模型（冻结，不更新）
        train_data: DPO 训练数据
        tokenizer: 分词器
        train_config: 训练配置
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        train_data,
        tokenizer,
        train_config,
    ):
        # TODO: 初始化 DPO 训练器
        # 1. 冻结 ref_model: ref_model.eval(); ref_model.requires_grad_(False)
        # 2. 初始化优化器（使用 dpo_learning_rate）
        # 3. 保存 beta 参数
        raise NotImplementedError(
            "TODO: 初始化 DPO 训练器\n"
            "参考: 模块 12 (DPO) 的训练流程章节\n"
            "参考实现: code/dpo/dpo_trainer.py"
        )

    def compute_log_probs(
        self, model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算序列的 log 概率

        Args:
            model: 语言模型
            input_ids: [batch, seq_len]
            labels: [batch, seq_len]（-100 位置不计入）

        Returns:
            每个样本的总 log 概率 [batch]

        实现步骤:
            1. logits = model(input_ids)["logits"]
            2. log_probs = log_softmax(logits, dim=-1)
            3. 在 labels 有效的位置（!= -100）收集 log_probs
            4. 对每个样本求和
        """
        raise NotImplementedError(
            "TODO: 计算序列 log 概率\n"
            "参考: code/dpo/dpo_loss.py 中的 compute_log_probs()"
        )

    def compute_dpo_loss(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算 DPO 损失

        Args:
            batch: {
                "chosen_input_ids": [batch, seq_len],
                "chosen_labels": [batch, seq_len],
                "rejected_input_ids": [batch, seq_len],
                "rejected_labels": [batch, seq_len],
            }

        Returns:
            (loss, metrics): 损失张量和统计指标

        实现步骤:
            1. 用 policy_model 计算 chosen 和 rejected 的 log_probs
            2. 用 ref_model 计算 chosen 和 rejected 的 log_probs
            3. log_ratio_w = policy_chosen - ref_chosen
            4. log_ratio_l = policy_rejected - ref_rejected
            5. loss = -log(sigmoid(beta * (log_ratio_w - log_ratio_l)))
            6. 取 batch 平均
        """
        raise NotImplementedError(
            "TODO: 实现 DPO 损失计算\n"
            "参考: 模块 12 的 DPO 五步推导\n"
            "参考实现: code/dpo/dpo_loss.py"
        )

    def dpo_training_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """DPO 单步训练"""
        raise NotImplementedError(
            "TODO: 实现 DPO 训练步\n"
            "参考: code/dpo/dpo_trainer.py"
        )
