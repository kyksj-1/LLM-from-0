"""
嵌入空间可视化工具

提供词嵌入和位置编码的可视化功能，包括:
- 位置编码热力图
- 位置相似度矩阵
- 词嵌入 t-SNE/PCA 投影
- 注意力衰减曲线

使用说明:
    所有可视化函数返回 matplotlib Figure 对象，可调用 fig.savefig() 保存。
    中文显示需要系统安装中文字体（如 SimHei）。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

# 设置中文字体（避免中文乱码）
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_positional_encoding(
    pe: torch.Tensor,
    title: str = "Positional Encoding",
    figsize: tuple = (12, 4),
) -> plt.Figure:
    """
    可视化位置编码矩阵

    Args:
        pe: 位置编码矩阵 [seq_len, d_model]
        title: 图标题
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    if pe.dim() > 2:
        pe = pe.squeeze()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(pe.numpy().T, aspect="auto", cmap="RdBu", origin="lower")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Position")
    ax.set_ylabel("Dimension")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_position_similarity(
    pe: torch.Tensor,
    title: str = "Position Similarity Matrix",
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """
    可视化位置编码的余弦相似度矩阵

    展示不同位置之间的相似度，有助于理解位置编码的距离特性。

    Args:
        pe: 位置编码矩阵 [seq_len, d_model]
        title: 图标题
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    if pe.dim() > 2:
        pe = pe.squeeze()

    # 归一化后计算余弦相似度
    pe_norm = pe / pe.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(pe_norm, pe_norm.T).numpy()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(similarity, cmap="viridis")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Position")
    ax.set_ylabel("Position")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_rope_decay(
    dim: int = 64,
    max_distance: int = 512,
    base: float = 10000.0,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    可视化 RoPE 的注意力衰减特性

    展示不同频率维度的注意力分数随距离的衰减情况。

    Args:
        dim: 头维度
        max_distance: 最大距离
        base: RoPE 频率基数
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    distances = torch.arange(max_distance).float()

    # 计算不同维度的频率
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))

    fig, ax = plt.subplots(figsize=figsize)

    # 选取几个代表性维度
    dims_to_plot = [0, dim // 8, dim // 4, dim // 2 - 1]
    for i in dims_to_plot:
        theta = inv_freq[i]
        # 注意力分数中的位置相关项: cos((m-n) * theta)
        decay = torch.cos(distances * theta)
        ax.plot(
            distances.numpy(),
            decay.numpy(),
            label=f"dim {2*i} (freq={theta:.4f})",
        )

    ax.set_xlabel("Relative Distance |m - n|")
    ax.set_ylabel("cos((m-n) * theta)")
    ax.set_title("RoPE Attention Decay by Dimension")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_alibi_bias(
    n_heads: int = 8,
    seq_len: int = 64,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    可视化 ALiBi 偏置

    展示不同头的偏置模式，每个头有不同的距离衰减斜率。

    Args:
        n_heads: 注意力头数量
        seq_len: 序列长度
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    slopes = 1.0 / (2.0 ** (torch.arange(1, n_heads + 1) * 8.0 / n_heads))
    distances = torch.arange(seq_len).float()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # 左图: 不同头的偏置衰减
    for h in range(min(n_heads, 8)):
        bias = -slopes[h] * distances
        ax1.plot(distances.numpy(), bias.numpy(), label=f"Head {h} (m={slopes[h]:.4f})")

    ax1.set_xlabel("Relative Distance")
    ax1.set_ylabel("Bias")
    ax1.set_title("ALiBi Bias per Head")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 右图: 第一个头的偏置矩阵
    rows = torch.arange(seq_len).unsqueeze(1)
    cols = torch.arange(seq_len).unsqueeze(0)
    bias_matrix = -slopes[0] * (rows - cols).abs().float()

    im = ax2.imshow(bias_matrix.numpy(), cmap="Blues_r")
    fig.colorbar(im, ax=ax2)
    ax2.set_xlabel("Key Position")
    ax2.set_ylabel("Query Position")
    ax2.set_title(f"ALiBi Bias Matrix (Head 0, m={slopes[0]:.4f})")

    fig.tight_layout()
    return fig


def plot_embedding_tsne(
    embeddings: torch.Tensor,
    words: List[str],
    labels: Optional[List[str]] = None,
    perplexity: float = 30.0,
    figsize: tuple = (10, 8),
) -> plt.Figure:
    """
    使用 t-SNE 可视化词嵌入

    Args:
        embeddings: 词嵌入矩阵 [n_words, dim]
        words: 词列表
        labels: 类别标签（可选，用于着色）
        perplexity: t-SNE 困惑度参数
        figsize: 图大小

    Returns:
        matplotlib Figure
    """
    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    coords = tsne.fit_transform(embeddings.numpy())

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        for i, (x, y) in enumerate(coords):
            ax.scatter(x, y, color=color_map[labels[i]], s=50)
            ax.annotate(words[i], (x, y), fontsize=8, alpha=0.8)
        # 图例
        for label, color in color_map.items():
            ax.scatter([], [], color=color, label=label)
        ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1], s=50)
        for i, (x, y) in enumerate(coords):
            ax.annotate(words[i], (x, y), fontsize=8, alpha=0.8)

    ax.set_title("Word Embeddings (t-SNE)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_encoding_comparison(
    sinusoidal_pe: torch.Tensor,
    rope_cos: torch.Tensor,
    alibi_bias: torch.Tensor,
    figsize: tuple = (15, 4),
) -> plt.Figure:
    """
    并排对比三种位置编码方案

    Args:
        sinusoidal_pe: 正弦编码 [seq_len, d_model]
        rope_cos: RoPE cos 缓存 [seq_len, dim]
        alibi_bias: ALiBi 偏置 [n_heads, seq_len, seq_len]

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Sinusoidal
    im0 = axes[0].imshow(sinusoidal_pe.numpy().T, aspect="auto", cmap="RdBu")
    fig.colorbar(im0, ax=axes[0])
    axes[0].set_title("Sinusoidal PE")
    axes[0].set_xlabel("Position")
    axes[0].set_ylabel("Dimension")

    # RoPE cos
    im1 = axes[1].imshow(rope_cos.numpy().T, aspect="auto", cmap="RdBu")
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title("RoPE (cos)")
    axes[1].set_xlabel("Position")
    axes[1].set_ylabel("Dimension")

    # ALiBi (第一个头)
    im2 = axes[2].imshow(alibi_bias[0].numpy(), cmap="Blues_r")
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title("ALiBi Bias (Head 0)")
    axes[2].set_xlabel("Key Position")
    axes[2].set_ylabel("Query Position")

    fig.suptitle("Position Encoding Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig
