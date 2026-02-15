"""
训练工程工具函数

提供训练过程中常用的辅助功能：
- 参数量统计与显存估算
- 随机种子管理
- 梯度范数计算
- 权重初始化工具
- 训练日志格式化

参考:
- GPT-2: Radford et al. (2019)
- Llama: Touvron et al. (2023)
"""

import os
import random
import math
import time
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import numpy as np


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数量

    Args:
        model: PyTorch 模型
        trainable_only: 是否只统计可训练参数

    Returns:
        参数总量（标量）
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def estimate_memory(
    num_params: int,
    optimizer_type: str = "adamw",
    precision: str = "fp32",
    batch_size: int = 1,
    seq_len: int = 2048,
    d_model: int = 768,
    n_layers: int = 12,
) -> Dict[str, float]:
    """
    估算训练显存需求（单位: GB）

    显存组成:
    1. 模型权重: P * bytes_per_param
    2. 梯度: P * bytes_per_grad
    3. 优化器状态: AdamW 需要 2P（一阶矩 m + 二阶矩 v）
    4. 激活值: 取决于 batch_size, seq_len, d_model, n_layers

    Args:
        num_params: 模型参数量
        optimizer_type: 优化器类型 ("adamw", "sgd", "adafactor", "8bit_adam")
        precision: 训练精度 ("fp32", "fp16", "bf16", "mixed")
        batch_size: 批大小
        seq_len: 序列长度
        d_model: 隐藏维度
        n_layers: 层数

    Returns:
        各部分显存估算（GB）
    """
    bytes_per_element = {"fp32": 4, "fp16": 2, "bf16": 2, "mixed": 2}
    param_bytes = bytes_per_element.get(precision, 4)

    # 模型权重
    model_memory = num_params * param_bytes

    # 梯度（与权重同精度）
    gradient_memory = num_params * param_bytes

    # 优化器状态（通常保持 FP32）
    optimizer_states = {
        "sgd": 0,          # SGD 无额外状态（不计动量）
        "sgd_momentum": 1, # SGD + Momentum: 1 倍参数（动量缓冲）
        "adamw": 2,        # AdamW: 2 倍参数（m_t + v_t）
        "8bit_adam": 0.5,  # 8-bit Adam: 量化状态，约 0.5 倍
        "adafactor": 0.5,  # Adafactor: 分解存储，约 0.5 倍
    }
    opt_multiplier = optimizer_states.get(optimizer_type, 2)
    # 优化器状态通常以 FP32 存储
    optimizer_memory = num_params * 4 * opt_multiplier

    # 激活值估算（粗略）
    # 每层: 大约 2 * batch_size * seq_len * d_model（输入 + 输出）
    # 注意力矩阵: batch_size * n_heads * seq_len^2
    activation_memory = (
        n_layers * batch_size * seq_len * d_model * 2 * param_bytes
    )

    total = model_memory + gradient_memory + optimizer_memory + activation_memory

    return {
        "model_weights_GB": model_memory / 1e9,
        "gradients_GB": gradient_memory / 1e9,
        "optimizer_states_GB": optimizer_memory / 1e9,
        "activations_GB": activation_memory / 1e9,
        "total_GB": total / 1e9,
    }


def set_seed(seed: int = 42) -> None:
    """
    设置全局随机种子，确保可复现性

    设置范围:
    1. Python random 模块
    2. NumPy
    3. PyTorch CPU
    4. PyTorch CUDA（如可用）

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # 确保 CUDA 卷积确定性（可能略微降低性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_states() -> Dict[str, object]:
    """
    获取当前所有随机数生成器的状态

    用于 checkpoint 保存，确保恢复训练后随机行为一致。

    Returns:
        包含所有 RNG 状态的字典
    """
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: Dict[str, object]) -> None:
    """
    恢复随机数生成器状态

    Args:
        states: 由 get_rng_states() 返回的状态字典
    """
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.random.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in states:
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def compute_grad_norm(
    model: nn.Module, norm_type: float = 2.0
) -> Tuple[float, Dict[str, float]]:
    """
    计算模型的全局梯度范数和各层梯度范数

    梯度范数是训练健康度的核心指标：
    - 范数突然增大 -> 可能的训练不稳定
    - 范数趋近于零 -> 梯度消失
    - 各层范数差异过大 -> 梯度流不均匀

    Args:
        model: PyTorch 模型
        norm_type: 范数类型（2.0 为 L2 范数）

    Returns:
        (全局梯度范数, {层名: 该层梯度范数})
    """
    per_layer_norms = {}
    total_norm_sq = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type).item()
            per_layer_norms[name] = param_norm
            total_norm_sq += param_norm ** norm_type

    global_norm = total_norm_sq ** (1.0 / norm_type)
    return global_norm, per_layer_norms


def init_weights_gpt2(module: nn.Module, n_layers: int, d_model: int) -> None:
    """
    GPT-2 风格的权重初始化

    规则:
    1. 线性层: N(0, 0.02)
    2. 嵌入层: N(0, 0.02)
    3. 残差投影: N(0, 0.02 / sqrt(2 * n_layers))
       - 因子 1/sqrt(2*n_layers) 防止残差路径方差随深度增长
       - 每层有 2 个残差连接（Attention + FFN）

    Args:
        module: 要初始化的模块
        n_layers: Transformer 层数
        d_model: 隐藏维度
    """
    std = 0.02
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)


def init_residual_scaling(module: nn.Module, n_layers: int) -> None:
    """
    残差分支缩放初始化

    对残差路径的最后一个线性层（输出投影）应用缩放因子 1/sqrt(2*N)，
    其中 N 是层数。这防止残差路径的方差随深度线性增长。

    推导:
    - 每层残差: x_{l+1} = x_l + f(x_l)
    - 若 Var(f(x_l)) = sigma^2，则 N 层后: Var(x_N) = Var(x_0) + N * sigma^2
    - 缩放后: Var(x_N) = Var(x_0) + N * sigma^2 / (2N) = Var(x_0) + sigma^2/2

    Args:
        module: 残差分支的输出投影层
        n_layers: Transformer 层数
    """
    if isinstance(module, nn.Linear):
        scale = 1.0 / math.sqrt(2 * n_layers)
        with torch.no_grad():
            module.weight.mul_(scale)


def format_training_log(
    step: int,
    loss: float,
    lr: float,
    grad_norm: float,
    tokens_per_sec: float,
    elapsed: float,
    extra: Optional[Dict] = None,
) -> str:
    """
    格式化训练日志行

    Args:
        step: 当前训练步数
        loss: 当前损失
        lr: 当前学习率
        grad_norm: 梯度范数
        tokens_per_sec: 每秒处理 token 数
        elapsed: 已用时间（秒）
        extra: 额外信息

    Returns:
        格式化的日志字符串
    """
    parts = [
        f"step={step:>6d}",
        f"loss={loss:.4f}",
        f"lr={lr:.2e}",
        f"grad_norm={grad_norm:.4f}",
        f"tok/s={tokens_per_sec:.0f}",
        f"elapsed={elapsed:.1f}s",
    ]
    if extra:
        for k, v in extra.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
    return " | ".join(parts)


def compute_tokens_per_second(
    batch_size: int, seq_len: int, step_time: float, grad_accum_steps: int = 1
) -> float:
    """
    计算每秒处理的 token 数

    Args:
        batch_size: 微批大小
        seq_len: 序列长度
        step_time: 单步训练时间（秒）
        grad_accum_steps: 梯度累积步数

    Returns:
        每秒 token 吞吐量
    """
    tokens_per_step = batch_size * seq_len * grad_accum_steps
    return tokens_per_step / max(step_time, 1e-8)


if __name__ == "__main__":
    print("=" * 60)
    print("训练工程工具函数 演示")
    print("=" * 60)

    # 1. 显存估算
    print("\n--- 显存估算 ---")
    # 模拟 GPT-2 Small (124M 参数)
    mem = estimate_memory(
        num_params=124_000_000,
        optimizer_type="adamw",
        precision="mixed",
        batch_size=8,
        seq_len=1024,
        d_model=768,
        n_layers=12,
    )
    print("GPT-2 Small (124M), batch=8, seq=1024:")
    for k, v in mem.items():
        print(f"  {k}: {v:.2f} GB")

    # 模拟 7B 模型
    mem_7b = estimate_memory(
        num_params=7_000_000_000,
        optimizer_type="adamw",
        precision="mixed",
        batch_size=1,
        seq_len=2048,
        d_model=4096,
        n_layers=32,
    )
    print("\n7B 模型, batch=1, seq=2048:")
    for k, v in mem_7b.items():
        print(f"  {k}: {v:.2f} GB")

    # 2. 随机种子管理
    print("\n--- 随机种子管理 ---")
    set_seed(42)
    t1 = torch.randn(3)
    states = get_rng_states()
    t2 = torch.randn(3)
    # 恢复状态后应得到相同的 t2
    set_rng_states(states)
    t3 = torch.randn(3)
    print(f"t2: {t2}")
    print(f"t3 (恢复 RNG 后): {t3}")
    print(f"t2 == t3: {torch.allclose(t2, t3)}")

    # 3. 梯度范数计算
    print("\n--- 梯度范数计算 ---")
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    x = torch.randn(4, 64)
    y = model(x).sum()
    y.backward()
    global_norm, layer_norms = compute_grad_norm(model)
    print(f"全局梯度范数: {global_norm:.4f}")
    for name, norm in layer_norms.items():
        print(f"  {name}: {norm:.4f}")

    # 4. 日志格式化
    print("\n--- 训练日志格式化 ---")
    log = format_training_log(
        step=1000, loss=3.245, lr=3e-4,
        grad_norm=1.23, tokens_per_sec=50000, elapsed=120.5,
        extra={"ppl": 25.67}
    )
    print(log)

    # 5. 参数量统计
    print("\n--- 参数量统计 ---")
    print(f"demo 模型参数量: {count_parameters(model):,}")
