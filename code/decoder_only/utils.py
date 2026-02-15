"""
工具函数

本模块提供 Decoder-Only 模型开发中常用的工具函数:
- 参数量统计与分析
- FLOPs 估算
- KV Cache 显存估算
- 模型结构可视化
- 因果掩码生成
- 权重初始化工具
- 模型保存/加载

参考:
- Kaplan et al. (2020). Scaling Laws for Neural Language Models.
  - C ≈ 6PD (训练总 FLOPs)
  - P ≈ 12Ld^2 (参数量估算)
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from config import ModelConfig


# ============================================================
# 参数量分析
# ============================================================

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    统计模型参数量

    Args:
        model: PyTorch 模型
        trainable_only: 是否只统计可训练参数

    Returns:
        参数总量
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """
    分模块统计参数量

    Args:
        model: PyTorch 模型

    Returns:
        {模块名: 参数量} 的字典
    """
    breakdown = {}
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        breakdown[name] = params
    return breakdown


def format_parameters(n: int) -> str:
    """
    格式化参数量显示

    Args:
        n: 参数数量

    Returns:
        格式化字符串, 如 "124.4M" 或 "6.7B"
    """
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


# ============================================================
# FLOPs 估算
# ============================================================

def estimate_flops(config: ModelConfig, seq_len: int) -> Dict[str, int]:
    """
    估算每个 token 的前向传播 FLOPs (矩阵乘法为主)

    矩阵乘法 [M, K] x [K, N] 的 FLOPs = 2 * M * K * N

    Args:
        config: 模型配置
        seq_len: 序列长度 (影响注意力的 FLOPs)

    Returns:
        各部分 FLOPs 的字典
    """
    d = config.d_model
    L = config.n_layers
    V = config.vocab_size
    S = seq_len
    n_h = config.n_heads
    n_kv = config.n_kv_heads
    d_h = config.head_dim
    d_ff = config.d_ff

    # --- 每层的 FLOPs ---

    # QKV 投影 (每个 token)
    # Q: [d] x [d, n_h*d_h] = 2 * d * n_h * d_h
    # K: [d] x [d, n_kv*d_h] = 2 * d * n_kv * d_h
    # V: [d] x [d, n_kv*d_h] = 2 * d * n_kv * d_h
    qkv_flops = 2 * d * (n_h + 2 * n_kv) * d_h

    # 注意力计算 (QK^T + softmax + AV), 每个 token 平均
    # QK^T: 每个头 2*S*d_h, n_h 个头
    # AV: 每个头 2*S*d_h, n_h 个头
    attn_flops = 2 * 2 * n_h * S * d_h

    # 输出投影
    # O: [n_h*d_h] x [n_h*d_h, d] = 2 * n_h * d_h * d
    out_proj_flops = 2 * n_h * d_h * d

    # FFN
    if config.ffn_type == "standard":
        # W1: 2*d*d_ff, W2: 2*d_ff*d
        ffn_flops = 2 * 2 * d * d_ff
    else:
        # SwiGLU/GeGLU: W_gate: 2*d*d_ff, W_up: 2*d*d_ff, W_down: 2*d_ff*d
        ffn_flops = 2 * 3 * d * d_ff

    # 每层总 FLOPs
    layer_flops = qkv_flops + attn_flops + out_proj_flops + ffn_flops

    # 全局 FLOPs
    embedding_flops = 0  # Embedding 查表不算 FLOPs
    lm_head_flops = 2 * d * V

    total_forward = L * layer_flops + lm_head_flops

    return {
        "qkv_projection": L * qkv_flops,
        "attention_compute": L * attn_flops,
        "output_projection": L * out_proj_flops,
        "ffn": L * ffn_flops,
        "lm_head": lm_head_flops,
        "total_forward": total_forward,
        "total_backward": 2 * total_forward,  # 反向约为前向的 2 倍
        "total_per_token": 3 * total_forward,  # 前向 + 反向 ≈ 3x 前向
    }


def estimate_training_flops(
    config: ModelConfig,
    n_tokens: int,
) -> float:
    """
    估算训练总 FLOPs

    使用 Kaplan 近似: C ≈ 6PD
    其中 P 为参数量, D 为训练 token 数

    Args:
        config: 模型配置
        n_tokens: 训练 token 总数

    Returns:
        训练总 FLOPs
    """
    P = config.estimate_parameters()
    return 6 * P * n_tokens


# ============================================================
# KV Cache 显存估算
# ============================================================

def estimate_kv_cache_memory(
    config: ModelConfig,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,
) -> Dict[str, float]:
    """
    估算 KV Cache 的显存占用

    KV Cache 大小 = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch_size * dtype_bytes

    Args:
        config: 模型配置
        seq_len: 序列长度
        batch_size: 批大小
        dtype_bytes: 每个元素的字节数 (FP32=4, FP16/BF16=2)

    Returns:
        KV Cache 显存信息字典 (单位: GB)
    """
    # 每层每 token 的 KV Cache 元素数
    kv_per_layer_per_token = 2 * config.n_kv_heads * config.head_dim

    # 总元素数
    total_elements = (
        kv_per_layer_per_token
        * config.n_layers
        * seq_len
        * batch_size
    )

    # 转换为 GB
    total_bytes = total_elements * dtype_bytes
    total_gb = total_bytes / (1024 ** 3)

    return {
        "kv_per_layer_per_token": kv_per_layer_per_token,
        "total_elements": total_elements,
        "total_bytes": total_bytes,
        "total_gb": total_gb,
    }


# ============================================================
# 因果掩码
# ============================================================

def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建因果注意力掩码 (下三角矩阵)

    mask[i][j] = 1 if j <= i else 0

    Args:
        seq_len: 序列长度
        device: 设备

    Returns:
        [seq_len, seq_len] 的布尔掩码
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
    return mask


def create_padding_mask(
    input_ids: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    创建 padding 掩码

    对于 padding 位置, mask = 0; 否则 mask = 1

    Args:
        input_ids: [batch, seq_len]
        pad_token_id: padding token 的 ID

    Returns:
        [batch, 1, 1, seq_len] 的掩码 (可广播到注意力形状)
    """
    mask = (input_ids != pad_token_id).unsqueeze(1).unsqueeze(2)
    return mask


# ============================================================
# 模型保存/加载
# ============================================================

def save_model(
    model: nn.Module,
    config: ModelConfig,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
):
    """
    保存模型检查点

    Args:
        model: 模型
        config: 模型配置
        path: 保存路径
        optimizer: 优化器 (可选)
        step: 当前训练步数
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
        "step": step,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)
    print(f"模型已保存到: {path}")


def load_model(
    path: str,
    device: str = "cpu",
) -> Tuple[nn.Module, ModelConfig, dict]:
    """
    加载模型检查点

    Args:
        path: 检查点路径
        device: 加载到的设备

    Returns:
        (model, config, checkpoint_dict)
    """
    from model import DecoderOnlyModel

    checkpoint = torch.load(path, map_location=device)

    # 重建配置
    config = ModelConfig(**checkpoint["config"])

    # 重建模型
    model = DecoderOnlyModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"模型已从 {path} 加载, 步数: {checkpoint.get('step', 0)}")
    return model, config, checkpoint


# ============================================================
# 模型信息打印
# ============================================================

def print_model_summary(model: nn.Module, config: ModelConfig):
    """
    打印模型摘要信息

    Args:
        model: 模型实例
        config: 模型配置
    """
    print("=" * 60)
    print("模型摘要")
    print("=" * 60)

    # 配置信息
    print(config.summary())

    # 参数量分解
    print("\n--- 参数量分解 ---")
    breakdown = parameter_breakdown(model)
    total = sum(breakdown.values())
    for name, count in breakdown.items():
        pct = count / total * 100
        print(f"  {name:25s}: {format_parameters(count):>10s} ({pct:5.1f}%)")
    print(f"  {'总计':25s}: {format_parameters(total):>10s}")

    # FLOPs 估算
    print("\n--- FLOPs 估算 (seq_len=512) ---")
    flops = estimate_flops(config, seq_len=512)
    for name, value in flops.items():
        if value > 1e9:
            print(f"  {name:25s}: {value / 1e9:.2f} GFLOPs")
        elif value > 1e6:
            print(f"  {name:25s}: {value / 1e6:.2f} MFLOPs")

    # KV Cache 估算
    print("\n--- KV Cache 估算 (seq_len=2048, batch=1, FP16) ---")
    kv_info = estimate_kv_cache_memory(config, seq_len=2048, batch_size=1)
    print(f"  每层每 token: {kv_info['kv_per_layer_per_token']} 元素")
    print(f"  总大小: {kv_info['total_gb']:.4f} GB")

    print("=" * 60)


if __name__ == "__main__":
    from config import mini_config, llama2_7b_config, llama3_8b_config
    from model import DecoderOnlyModel

    # --- 小模型测试 ---
    print("=== Mini 模型 ===")
    config = mini_config()
    model = DecoderOnlyModel(config)
    print_model_summary(model, config)

    # --- Llama 2 7B 配置分析 (只分析配置, 不创建模型) ---
    print("\n\n=== Llama 2 7B 配置分析 ===")
    llama_config = llama2_7b_config()
    print(llama_config.summary())

    # FLOPs 和 KV Cache
    flops = estimate_flops(llama_config, seq_len=4096)
    print(f"\n前向 FLOPs (每 token): {flops['total_forward'] / 1e9:.2f} GFLOPs")
    print(f"训练总 FLOPs (2T tokens): {estimate_training_flops(llama_config, 2e12):.2e}")

    kv = estimate_kv_cache_memory(llama_config, seq_len=4096, batch_size=1)
    print(f"KV Cache (seq=4096, batch=1, FP16): {kv['total_gb']:.4f} GB")

    # --- 对比 MHA vs GQA 的 KV Cache ---
    print("\n\n=== MHA vs GQA KV Cache 对比 ===")
    configs = {
        "Llama 2 7B (MHA, 32 KV头)": llama2_7b_config(),
        "Llama 3 8B (GQA, 8 KV头)": llama3_8b_config(),
    }
    for name, cfg in configs.items():
        kv = estimate_kv_cache_memory(cfg, seq_len=8192, batch_size=1)
        print(f"{name}: {kv['total_gb']:.4f} GB")

    # --- 因果掩码测试 ---
    print("\n\n=== 因果掩码 ===")
    mask = create_causal_mask(5)
    print(mask.int())
