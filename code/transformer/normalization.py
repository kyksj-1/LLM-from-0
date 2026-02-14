"""
归一化层实现：LayerNorm, RMSNorm

本模块实现了 Transformer 中使用的归一化层。

LayerNorm: y = γ * (x - μ) / (σ + ε) + β  (原始 Transformer)
RMSNorm:   y = γ * x / RMS(x)              (Llama/Gemma/DeepSeek)

RMSNorm 省去了均值中心化步骤，计算更快，效果相当。

参考:
- Ba et al. (2016). Layer Normalization.
- Zhang & Sennrich (2019). Root Mean Square Layer Normalization.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization

    对每个样本沿特征维度归一化:
        μ = mean(x), σ² = var(x)
        x̂ = (x - μ) / sqrt(σ² + ε)
        y = γ * x̂ + β

    与 BatchNorm 的区别: LayerNorm 不依赖 batch 统计量,
    适合变长序列和小 batch 训练。

    Args:
        d_model: 特征维度
        eps: 数值稳定性的小常数
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_model]
        Returns:
            归一化后的张量, 形状不变
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization

    去掉均值中心化，只做缩放:
        RMS(x) = sqrt(mean(x²))
        x̂ = x / (RMS(x) + ε)
        y = γ * x̂

    优势:
    1. 计算更快（省去均值计算）
    2. 效果相当甚至更好
    3. 被 Llama, Gemma, DeepSeek 等主流模型采用

    Args:
        d_model: 特征维度
        eps: 数值稳定性的小常数
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d_model]
        Returns:
            归一化后的张量, 形状不变
        """
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)
