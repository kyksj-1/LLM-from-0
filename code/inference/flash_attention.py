"""
Flash Attention 概念实现

本模块实现了 Flash Attention 的核心算法，包括：
- 标准注意力（用于对照）
- 分块注意力（Tiling）
- 在线 Softmax（Online Softmax）
- IO 复杂度分析

Flash Attention 的核心思想:
    不在 HBM（高带宽显存）中实例化完整的 N×N 注意力矩阵，
    而是在 SRAM（片上共享内存）中分块计算注意力。通过在线 Softmax
    算法，在只看到部分 K 块的情况下也能正确计算 softmax 归一化。

IO 复杂度:
    标准 Attention: Theta(Nd + N^2) HBM 访问
    Flash Attention: Theta(N^2 * d^2 / M) HBM 访问
    其中 M 是 SRAM 大小

参考论文:
    Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention
    with IO-Awareness", NeurIPS 2022
"""

import torch
import torch.nn.functional as F
import math
import time
from typing import Optional, Tuple


def standard_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    标准注意力计算（用于对照验证）

    完整实例化 N×N 的注意力矩阵，IO 复杂度为 O(Nd + N^2)。

    Args:
        Q: Query [batch, n_heads, seq_len, head_dim]
        K: Key   [batch, n_heads, seq_len, head_dim]
        V: Value [batch, n_heads, seq_len, head_dim]
        causal: 是否使用因果掩码

    Returns:
        注意力输出 [batch, n_heads, seq_len, head_dim]
    """
    d_k = Q.shape[-1]
    scale = 1.0 / math.sqrt(d_k)

    # 计算注意力分数矩阵 S = Q @ K^T / sqrt(d_k)
    # 形状: [batch, n_heads, seq_len, seq_len]
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    # 因果掩码
    if causal:
        seq_len = Q.shape[2]
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))

    # Softmax
    attn_weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(attn_weights, V)

    return output


def online_softmax_update(
    m_prev: torch.Tensor,
    l_prev: torch.Tensor,
    o_prev: torch.Tensor,
    s_block: torch.Tensor,
    v_block: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    在线 Softmax 更新

    处理新的一块 (s_block, v_block) 后，更新运行统计量 m, l, o。
    这是 Flash Attention 的核心数学技巧。

    算法:
        m_new = max(m_prev, rowmax(s_block))
        l_new = exp(m_prev - m_new) * l_prev + rowsum(exp(s_block - m_new))
        o_new = diag(exp(m_prev - m_new)) * o_prev + exp(s_block - m_new) @ v_block

    Args:
        m_prev: 之前的行最大值 [block_r]
        l_prev: 之前的指数和 [block_r]
        o_prev: 之前的累积输出 [block_r, d]
        s_block: 当前块的注意力分数 [block_r, block_c]
        v_block: 当前块的 V [block_c, d]

    Returns:
        (m_new, l_new, o_new): 更新后的统计量
    """
    # 当前块的行最大值
    m_block = s_block.max(dim=-1).values  # [block_r]

    # 更新全局最大值
    m_new = torch.maximum(m_prev, m_block)  # [block_r]

    # 修正因子：旧统计量需要乘以 exp(m_old - m_new)
    alpha = torch.exp(m_prev - m_new)  # [block_r]

    # 当前块的 exp(s - m_new)
    p_block = torch.exp(s_block - m_new.unsqueeze(-1))  # [block_r, block_c]

    # 更新指数和
    l_new = alpha * l_prev + p_block.sum(dim=-1)  # [block_r]

    # 更新累积输出
    o_new = (alpha.unsqueeze(-1) * o_prev
             + torch.matmul(p_block, v_block))  # [block_r, d]

    return m_new, l_new, o_new


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_size_r: int = 64,
    block_size_c: int = 64,
    causal: bool = False,
) -> torch.Tensor:
    """
    Flash Attention 概念实现

    使用分块计算和在线 Softmax 避免实例化完整的 N×N 注意力矩阵。
    此实现在 Python 层面模拟 Flash Attention 的逻辑，用于理解算法原理。
    实际的 Flash Attention 通过 Triton/CUDA 在 GPU SRAM 中运行。

    Args:
        Q: Query [batch, n_heads, seq_len, head_dim]
        K: Key   [batch, n_heads, seq_len, head_dim]
        V: Value [batch, n_heads, seq_len, head_dim]
        block_size_r: Q 的分块大小（行方向）
        block_size_c: KV 的分块大小（列方向）
        causal: 是否使用因果掩码

    Returns:
        注意力输出 [batch, n_heads, seq_len, head_dim]
    """
    batch, n_heads, seq_len, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    # 输出张量
    O = torch.zeros_like(Q)

    # 分块数
    n_blocks_r = math.ceil(seq_len / block_size_r)
    n_blocks_c = math.ceil(seq_len / block_size_c)

    for b in range(batch):
        for h in range(n_heads):
            # 对每个 Q 块
            for i in range(n_blocks_r):
                # Q 块的行范围
                r_start = i * block_size_r
                r_end = min(r_start + block_size_r, seq_len)
                r_len = r_end - r_start

                # 提取 Q 块
                q_block = Q[b, h, r_start:r_end, :]  # [r_len, d]

                # 初始化在线 softmax 统计量
                m_i = torch.full((r_len,), float("-inf"),
                                device=Q.device, dtype=Q.dtype)
                l_i = torch.zeros(r_len, device=Q.device, dtype=Q.dtype)
                o_i = torch.zeros(r_len, d, device=Q.device, dtype=Q.dtype)

                # 遍历所有 KV 块
                for j in range(n_blocks_c):
                    c_start = j * block_size_c
                    c_end = min(c_start + block_size_c, seq_len)

                    # 因果掩码：跳过不需要的 KV 块
                    if causal and c_start > r_end - 1:
                        break  # 后面的 KV 块都在因果掩码之外

                    # 提取 K, V 块
                    k_block = K[b, h, c_start:c_end, :]  # [c_len, d]
                    v_block = V[b, h, c_start:c_end, :]  # [c_len, d]

                    # 计算分块注意力分数
                    s_block = torch.matmul(q_block, k_block.T) * scale  # [r_len, c_len]

                    # 因果掩码（块内）
                    if causal:
                        c_len = c_end - c_start
                        for ri in range(r_len):
                            for ci in range(c_len):
                                if (c_start + ci) > (r_start + ri):
                                    s_block[ri, ci] = float("-inf")

                    # 在线 Softmax 更新
                    m_i, l_i, o_i = online_softmax_update(
                        m_i, l_i, o_i, s_block, v_block
                    )

                # 最终归一化
                O[b, h, r_start:r_end, :] = o_i / l_i.unsqueeze(-1)

    return O


def analyze_io_complexity(
    seq_len: int,
    d: int,
    sram_size_elements: int = 100000,
) -> dict:
    """
    分析标准 Attention 和 Flash Attention 的 IO 复杂度

    Args:
        seq_len: 序列长度 N
        d: head dimension
        sram_size_elements: SRAM 大小（可存储的元素数）

    Returns:
        包含 IO 复杂度分析的字典
    """
    N = seq_len
    M = sram_size_elements

    # 标准 Attention 的 HBM 访问量
    # 读 Q,K,V: 3*N*d, 写 S: N^2, 读 S: N^2, 写 P: N^2, 读 P,V: N^2+N*d, 写 O: N*d
    standard_io = 3 * N * d + 3 * N * N + N * d + N * d  # 简化

    # Flash Attention 的 HBM 访问量
    # 近似: Theta(N^2 * d^2 / M)
    flash_io = N * N * d * d / M  # 近似

    # 额外显存
    standard_mem = N * N  # S 和 P 矩阵
    flash_mem = N  # 只需要 m 和 l 向量

    # 块大小估算
    # 约束: B_r * d + 2 * B_c * d <= M
    block_c = min(M // (3 * d), N)
    block_r = min(M // (3 * d), N)

    return {
        "seq_len": N,
        "head_dim": d,
        "sram_size": M,
        "standard_io": standard_io,
        "flash_io": int(flash_io),
        "io_reduction": standard_io / max(flash_io, 1),
        "standard_extra_mem": standard_mem,
        "flash_extra_mem": flash_mem,
        "mem_reduction": standard_mem / max(flash_mem, 1),
        "estimated_block_size": block_r,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Flash Attention 概念实现演示")
    print("=" * 60)

    torch.manual_seed(42)

    # 1. 正确性验证：对比标准 Attention 和 Flash Attention
    print("\n--- 正确性验证 ---")
    batch, n_heads, seq_len, d = 2, 4, 128, 64

    Q = torch.randn(batch, n_heads, seq_len, d)
    K = torch.randn(batch, n_heads, seq_len, d)
    V = torch.randn(batch, n_heads, seq_len, d)

    # 标准 Attention
    out_standard = standard_attention(Q, K, V, causal=False)

    # Flash Attention（不同块大小）
    for bs in [32, 64, 128]:
        out_flash = flash_attention(Q, K, V, block_size_r=bs, block_size_c=bs, causal=False)
        max_diff = (out_standard - out_flash).abs().max().item()
        print(f"  Block size={bs}: max |diff| = {max_diff:.2e} "
              f"({'通过' if max_diff < 1e-5 else '失败'})")

    # 因果掩码版本
    print("\n--- 因果掩码版本验证 ---")
    out_std_causal = standard_attention(Q, K, V, causal=True)
    out_flash_causal = flash_attention(Q, K, V, block_size_r=32, block_size_c=32, causal=True)
    max_diff_causal = (out_std_causal - out_flash_causal).abs().max().item()
    print(f"  Causal mask: max |diff| = {max_diff_causal:.2e} "
          f"({'通过' if max_diff_causal < 1e-5 else '失败'})")

    # 2. 在线 Softmax 验证
    print("\n--- 在线 Softmax 独立验证 ---")
    x = torch.randn(16, 128)  # 16 行, 128 列

    # 标准 softmax
    std_softmax = F.softmax(x, dim=-1)

    # 在线 softmax（分成 4 块，每块 32 列）
    block_size = 32
    n_blocks = 128 // block_size

    m = torch.full((16,), float("-inf"))
    l = torch.zeros(16)
    # 验证只需要 softmax 值，不需要 V 乘法
    # 用单位矩阵模拟 V 来验证 softmax 值
    o = torch.zeros(16, 128)

    for j in range(n_blocks):
        s = j * block_size
        e = s + block_size
        x_block = x[:, s:e]

        # 使用单位矩阵作为 V
        v_identity = torch.eye(block_size).unsqueeze(0).expand(16, -1, -1)

        m_block = x_block.max(dim=-1).values
        m_new = torch.maximum(m, m_block)
        alpha = torch.exp(m - m_new)
        p_block = torch.exp(x_block - m_new.unsqueeze(-1))

        l = alpha * l + p_block.sum(dim=-1)

        # 累积 softmax 值
        o[:, s:e] = p_block

        # 修正之前的值
        for prev_j in range(j):
            ps = prev_j * block_size
            pe = ps + block_size
            o[:, ps:pe] *= alpha.unsqueeze(-1)

        m = m_new

    # 最终归一化
    online_softmax = o / l.unsqueeze(-1)
    softmax_diff = (std_softmax - online_softmax).abs().max().item()
    print(f"  标准 vs 在线 Softmax: max |diff| = {softmax_diff:.2e}")

    # 3. IO 复杂度分析
    print("\n--- IO 复杂度分析 ---")
    print(f"{'序列长度':<10} {'标准 IO':<15} {'Flash IO':<15} {'IO 减少':<10} {'显存减少'}")
    print("-" * 65)

    for N in [512, 1024, 4096, 16384, 65536]:
        analysis = analyze_io_complexity(N, d=128, sram_size_elements=100000)
        print(f"{N:<10} {analysis['standard_io']:<15,} {analysis['flash_io']:<15,} "
              f"{analysis['io_reduction']:<10.1f}x {analysis['mem_reduction']:.0f}x")

    # 4. 性能对比（小规模，CPU 上）
    print("\n--- 性能对比（CPU, 教学用）---")
    for seq_len in [64, 128, 256]:
        Q = torch.randn(1, 2, seq_len, 32)
        K = torch.randn(1, 2, seq_len, 32)
        V = torch.randn(1, 2, seq_len, 32)

        # 标准 Attention
        start = time.perf_counter()
        for _ in range(5):
            _ = standard_attention(Q, K, V)
        t_std = (time.perf_counter() - start) / 5

        # Flash Attention
        start = time.perf_counter()
        for _ in range(5):
            _ = flash_attention(Q, K, V, block_size_r=32, block_size_c=32)
        t_flash = (time.perf_counter() - start) / 5

        print(f"  seq_len={seq_len}: 标准={t_std*1000:.1f}ms, "
              f"Flash={t_flash*1000:.1f}ms")
        # 注意：Python 实现的 Flash Attention 在 CPU 上比标准实现慢
        # 因为多层 Python 循环的开销远超 IO 节省
        # 真正的加速需要 Triton/CUDA 实现
    print("  (注: Python 实现仅用于理解算法，实际加速需要 Triton/CUDA 内核)")
