"""
注意力模式可视化工具

本模块提供注意力权重矩阵的可视化功能，帮助直观理解
不同注意力机制的行为模式。

功能:
1. 单头注意力热图
2. 多头注意力对比图
3. 注意力熵分析
4. KV Cache 大小对比柱状图

使用方法:
    python visualize_attention.py

依赖: matplotlib (可选，用于生成图表)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Dict, Tuple


def compute_attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    计算注意力权重矩阵（不加权 Value，仅返回权重）

    Args:
        query: [batch, n_heads, seq_q, d_k]
        key: [batch, n_heads, seq_k, d_k]
        is_causal: 是否使用因果掩码

    Returns:
        attention_weights: [batch, n_heads, seq_q, seq_k]
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if is_causal:
        seq_q, seq_k = scores.shape[-2], scores.shape[-1]
        causal_mask = torch.triu(
            torch.ones(seq_q, seq_k, device=scores.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    weights = weights.nan_to_num(0.0)

    return weights


def compute_attention_entropy(weights: torch.Tensor) -> torch.Tensor:
    """
    计算注意力分布的熵

    熵越高表示注意力越分散（均匀关注），熵越低表示注意力越集中。

    H = -sum(alpha * log(alpha))

    Args:
        weights: [batch, n_heads, seq_q, seq_k] 注意力权重

    Returns:
        entropy: [batch, n_heads, seq_q] 每个位置的注意力熵
    """
    # 避免 log(0)
    eps = 1e-10
    log_weights = torch.log(weights + eps)
    entropy = -torch.sum(weights * log_weights, dim=-1)
    return entropy


def generate_attention_patterns(
    seq_len: int = 16,
    d_model: int = 64,
    n_heads: int = 4,
) -> Dict[str, torch.Tensor]:
    """
    使用随机输入生成注意力模式，用于可视化演示

    Args:
        seq_len: 序列长度
        d_model: 模型维度
        n_heads: 头数

    Returns:
        包含注意力权重和熵的字典
    """
    head_dim = d_model // n_heads

    # 生成随机 Q, K
    torch.manual_seed(42)  # 固定种子以保证可重复
    q = torch.randn(1, n_heads, seq_len, head_dim)
    k = torch.randn(1, n_heads, seq_len, head_dim)

    # 计算注意力权重
    weights = compute_attention_weights(q, k, is_causal=True)

    # 计算熵
    entropy = compute_attention_entropy(weights)

    return {
        "weights": weights,  # [1, n_heads, seq_q, seq_k]
        "entropy": entropy,  # [1, n_heads, seq_q]
    }


def print_attention_matrix(
    weights: torch.Tensor,
    head_idx: int = 0,
    batch_idx: int = 0,
    max_display: int = 10,
):
    """
    在终端以文本形式打印注意力矩阵

    Args:
        weights: [batch, n_heads, seq_q, seq_k]
        head_idx: 要显示的头索引
        batch_idx: 要显示的批次索引
        max_display: 最大显示的序列长度
    """
    w = weights[batch_idx, head_idx].detach().cpu()
    seq_len = min(w.shape[0], max_display)
    w = w[:seq_len, :seq_len]

    print(f"\n注意力矩阵 (Head {head_idx}, 前 {seq_len} 个位置):")
    print("      " + "".join(f" t{j:<4}" for j in range(seq_len)))
    print("      " + "-" * (seq_len * 6))

    for i in range(seq_len):
        row = f"t{i:<3} |"
        for j in range(seq_len):
            val = w[i, j].item()
            if val > 0.3:
                # 高注意力用 # 表示
                row += f" {'##':>4}"
            elif val > 0.1:
                row += f" {'++':>4}"
            elif val > 0.01:
                row += f" {'..':>4}"
            else:
                row += f" {'  ':>4}"
        print(row)

    print("\n图例: ## = 高注意力(>0.3), ++ = 中等(>0.1), .. = 低(>0.01), 空 = 极低")


def print_entropy_analysis(
    entropy: torch.Tensor,
    batch_idx: int = 0,
):
    """
    在终端打印注意力熵分析

    Args:
        entropy: [batch, n_heads, seq_q]
        batch_idx: 批次索引
    """
    n_heads = entropy.shape[1]
    seq_len = entropy.shape[2]

    print(f"\n注意力熵分析 (每个头的平均熵):")
    print("-" * 40)

    for h in range(n_heads):
        avg_entropy = entropy[batch_idx, h].mean().item()
        max_entropy = math.log(seq_len)  # 均匀分布时的最大熵
        normalized = avg_entropy / max_entropy if max_entropy > 0 else 0

        # 用条形图表示归一化熵
        bar_len = int(normalized * 20)
        bar = "|" * bar_len + "." * (20 - bar_len)

        print(f"  Head {h}: 平均熵={avg_entropy:.3f} "
              f"(归一化: {normalized:.2%}) [{bar}]")

    # 分类
    print(f"\n头的功能推测:")
    for h in range(n_heads):
        avg_entropy = entropy[batch_idx, h].mean().item()
        max_entropy = math.log(seq_len)
        normalized = avg_entropy / max_entropy if max_entropy > 0 else 0

        if normalized < 0.3:
            category = "集中注意力 (可能是局部/前一个 token 头)"
        elif normalized < 0.6:
            category = "中等分散 (可能是模式匹配头)"
        else:
            category = "广泛关注 (可能是全局聚合头)"

        print(f"  Head {h}: {category}")


def compare_kv_cache_sizes():
    """
    对比不同注意力变体的 KV Cache 大小并以文本柱状图展示
    """
    # 配置: 类似 Llama 3 70B
    configs = {
        "MHA (H=64)": 2 * 64 * 128,    # 2 * n_heads * d_head
        "GQA-8":      2 * 8 * 128,      # 2 * 8 * d_head
        "GQA-4":      2 * 4 * 128,      # 2 * 4 * d_head
        "MQA (G=1)":  2 * 1 * 128,      # 2 * 1 * d_head
        "MLA":        512 + 64,          # d_c + d_r
    }

    max_val = max(configs.values())
    bar_width = 40

    print("\nKV Cache 大小对比 (每层每 token, 元素数):")
    print("=" * 65)

    for name, elements in configs.items():
        bar_len = int(elements / max_val * bar_width)
        bar = "#" * bar_len
        print(f"{name:<15} {bar:<{bar_width}} {elements:>8}")

    print(f"\n基准 (MHA) 下的压缩比:")
    mha_size = configs["MHA (H=64)"]
    for name, elements in configs.items():
        ratio = mha_size / elements
        print(f"  {name}: {ratio:.1f}x 压缩")


def try_matplotlib_visualization():
    """
    尝试使用 matplotlib 生成注意力可视化图表

    如果 matplotlib 不可用，跳过图形化展示。
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        import matplotlib.pyplot as plt
        import numpy as np

        print("\n\nmatplotlib 可用，生成可视化图表...")

        # 生成注意力模式
        data = generate_attention_patterns(seq_len=16, d_model=64, n_heads=4)
        weights = data["weights"][0].detach().numpy()  # [n_heads, seq_q, seq_k]

        # 绘制 4 个头的注意力热图
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        fig.suptitle("Multi-Head Attention Patterns (Causal)", fontsize=14)

        for h in range(4):
            im = axes[h].imshow(weights[h], cmap='Blues', vmin=0, vmax=0.5)
            axes[h].set_title(f'Head {h}')
            axes[h].set_xlabel('Key Position')
            axes[h].set_ylabel('Query Position')

        plt.colorbar(im, ax=axes, shrink=0.8, label='Attention Weight')
        plt.tight_layout()
        plt.savefig('attention_patterns.png', dpi=150, bbox_inches='tight')
        print("  保存: attention_patterns.png")

        # KV Cache 对比柱状图
        fig, ax = plt.subplots(figsize=(8, 5))

        names = ["MHA", "GQA-8", "GQA-4", "MQA", "MLA"]
        sizes = [2*64*128, 2*8*128, 2*4*128, 2*128, 512+64]
        colors = ['#1f77b4', '#2ca02c', '#98df8a', '#ff7f0e', '#d62728']

        bars = ax.bar(names, sizes, color=colors)
        ax.set_ylabel('KV Cache Elements per Layer per Token')
        ax.set_title('KV Cache Size Comparison')

        # 在柱子上标注数值
        for bar, size in zip(bars, sizes):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                    f'{size}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('kv_cache_comparison.png', dpi=150, bbox_inches='tight')
        print("  保存: kv_cache_comparison.png")

    except ImportError:
        print("\nmatplotlib 未安装，跳过图形化可视化。")
        print("安装方法: pip install matplotlib")


if __name__ == "__main__":
    print("=" * 60)
    print("注意力模式可视化演示")
    print("=" * 60)

    # 生成注意力模式
    data = generate_attention_patterns(seq_len=16, d_model=64, n_heads=4)

    # 文本形式的注意力矩阵
    for h in range(4):
        print_attention_matrix(data["weights"], head_idx=h, max_display=8)

    # 熵分析
    print_entropy_analysis(data["entropy"])

    # KV Cache 大小对比
    compare_kv_cache_sizes()

    # 尝试 matplotlib 可视化
    try_matplotlib_visualization()
