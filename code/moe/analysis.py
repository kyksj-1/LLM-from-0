"""
MoE 专家利用率分析工具

本模块提供了分析 MoE 模型中专家利用率、路由分布、
负载均衡等指标的工具函数和可视化方法。

功能:
- 专家利用率统计（频率、变异系数）
- 路由分布可视化（热图、直方图）
- 负载均衡质量评估
- Dense vs MoE 参数效率对比

参考:
- Fedus et al. (2022). Switch Transformers.
- DeepSeek-AI (2024). DeepSeekMoE.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
import math


def analyze_routing_distribution(
    logits: torch.Tensor,
    top_k: int = 2,
    num_experts: Optional[int] = None,
) -> Dict[str, object]:
    """
    分析路由分布的各项指标

    Args:
        logits: [batch, seq_len, num_experts] 路由 logits
        top_k: Top-K 值
        num_experts: 专家数量（从 logits 推断）

    Returns:
        字典包含:
        - expert_frequency: [N] 各专家被选中的频率
        - expert_prob: [N] 各专家的平均路由概率
        - utilization: [N] 各专家利用率
        - cv: 变异系数（负载均衡度量）
        - entropy: 路由决策的熵（越高越分散）
        - top1_accuracy: Top-1 专家的平均选择集中度
    """
    N = num_experts or logits.shape[-1]
    flat_logits = logits.view(-1, N)  # [T, N]
    T = flat_logits.shape[0]

    # Top-K 选择
    _, top_k_indices = flat_logits.topk(top_k, dim=-1)  # [T, K]

    # 各专家被选中的频率
    counts = torch.zeros(N, device=logits.device)
    for k_idx in range(top_k):
        for i in range(N):
            counts[i] += (top_k_indices[:, k_idx] == i).sum().float()
    frequency = counts / (T * top_k)

    # 各专家的平均路由概率
    probs = F.softmax(flat_logits, dim=-1)  # [T, N]
    avg_prob = probs.mean(dim=0)  # [N]

    # 利用率（相对于理想均匀分布）
    ideal = 1.0 / N
    utilization = frequency / ideal

    # 变异系数 (CV)
    cv = utilization.std() / (utilization.mean() + 1e-8)

    # 路由决策的熵
    route_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    # 最大理论熵（均匀分布）
    max_entropy = math.log(N)

    # Top-1 专家的选择集中度
    top1_counts = torch.zeros(N, device=logits.device)
    top1_indices = flat_logits.argmax(dim=-1)
    for i in range(N):
        top1_counts[i] = (top1_indices == i).sum().float()
    top1_concentration = top1_counts.max() / T

    return {
        "expert_frequency": frequency.detach().cpu(),
        "expert_prob": avg_prob.detach().cpu(),
        "utilization": utilization.detach().cpu(),
        "cv": cv.item(),
        "entropy": route_entropy.item(),
        "max_entropy": max_entropy,
        "normalized_entropy": route_entropy.item() / max_entropy,
        "top1_concentration": top1_concentration.item(),
    }


def compute_flops_comparison(
    d_model: int,
    d_ff_dense: int,
    num_experts: int,
    top_k: int,
    d_ff_expert: Optional[int] = None,
    seq_len: int = 1,
) -> Dict[str, float]:
    """
    计算 Dense FFN 和 MoE FFN 的 FLOPs 对比

    Args:
        d_model: 模型维度
        d_ff_dense: Dense FFN 隐藏维度
        num_experts: MoE 专家数量
        top_k: MoE Top-K 值
        d_ff_expert: MoE 每个专家的隐藏维度
        seq_len: 序列长度

    Returns:
        各项 FLOPs 和参数量对比
    """
    d_ff_exp = d_ff_expert or d_ff_dense

    # Dense FFN 参数和 FLOPs
    dense_params = 2 * d_model * d_ff_dense
    dense_flops_per_token = 2 * dense_params  # 乘加各算一次

    # MoE 参数和 FLOPs
    expert_params = 2 * d_model * d_ff_exp
    moe_total_params = num_experts * expert_params + num_experts * d_model  # 加路由器
    moe_active_params = top_k * expert_params + num_experts * d_model
    moe_flops_per_token = top_k * 2 * expert_params + 2 * num_experts * d_model

    return {
        "dense": {
            "params": dense_params,
            "flops_per_token": dense_flops_per_token,
        },
        "moe": {
            "total_params": moe_total_params,
            "active_params": moe_active_params,
            "flops_per_token": moe_flops_per_token,
            "param_efficiency_ratio": moe_total_params / moe_active_params,
        },
        "comparison": {
            "moe_total_vs_dense_params": moe_total_params / dense_params,
            "moe_active_vs_dense_params": moe_active_params / dense_params,
            "moe_vs_dense_flops": moe_flops_per_token / dense_flops_per_token,
        },
    }


def analyze_expert_specialization(
    logits_list: List[torch.Tensor],
    token_types: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """
    分析专家是否存在领域专业化

    通过统计不同 token 被路由到不同专家的模式，
    判断专家是否形成了有意义的分工。

    Args:
        logits_list: 多层路由 logits 的列表
        token_types: [T] 可选的 token 类型标签（用于分析专家-领域对应关系）

    Returns:
        各层的专家专业化分析结果
    """
    results = {}

    for layer_idx, logits in enumerate(logits_list):
        flat_logits = logits.view(-1, logits.shape[-1])
        N = flat_logits.shape[-1]

        # 计算各 token 的 Top-1 专家
        top1 = flat_logits.argmax(dim=-1)  # [T]

        # 专家选择的一致性：同一个专家在不同 token 上的分数方差
        probs = F.softmax(flat_logits, dim=-1)
        expert_score_std = probs.std(dim=0)  # [N]

        # 专家间的相关性：不同专家的路由分数是否相关
        # 高相关性 → 专家分工不明确
        if flat_logits.shape[0] > 1:
            corr_matrix = torch.corrcoef(probs.t())  # [N, N]
            # 去除对角线后的平均相关性
            mask = ~torch.eye(N, dtype=torch.bool, device=corr_matrix.device)
            avg_correlation = corr_matrix[mask].mean().item()
        else:
            avg_correlation = 0.0

        results[f"layer_{layer_idx}"] = {
            "expert_score_std": expert_score_std.detach().cpu(),
            "avg_inter_expert_correlation": avg_correlation,
            "top1_distribution": [
                (top1 == i).sum().item() for i in range(N)
            ],
        }

    return results


def print_analysis_report(
    routing_stats: Dict,
    flops_comparison: Optional[Dict] = None,
) -> None:
    """
    打印分析报告

    Args:
        routing_stats: analyze_routing_distribution 的返回值
        flops_comparison: compute_flops_comparison 的返回值
    """
    print("=" * 60)
    print("MoE 专家利用率分析报告")
    print("=" * 60)

    # 路由分布
    freq = routing_stats["expert_frequency"]
    util = routing_stats["utilization"]
    N = len(freq)

    print(f"\n专家数量: {N}")
    print(f"路由熵: {routing_stats['entropy']:.4f} / {routing_stats['max_entropy']:.4f} "
          f"(归一化: {routing_stats['normalized_entropy']:.4f})")
    print(f"变异系数 (CV): {routing_stats['cv']:.4f} "
          f"({'好' if routing_stats['cv'] < 0.3 else '差'})")
    print(f"Top-1 最大集中度: {routing_stats['top1_concentration']:.4f}")

    print(f"\n各专家利用率:")
    for i in range(N):
        bar_len = int(util[i].item() * 20)
        bar = "#" * bar_len + "-" * (20 - bar_len)
        print(f"  Expert {i:2d}: [{bar}] {util[i].item():.2f}x "
              f"(freq={freq[i].item():.4f})")

    if flops_comparison:
        print(f"\n" + "=" * 60)
        print("FLOPs 与参数量对比")
        print("=" * 60)
        dense = flops_comparison["dense"]
        moe = flops_comparison["moe"]
        comp = flops_comparison["comparison"]

        print(f"\nDense FFN:")
        print(f"  参数量: {dense['params']:,}")
        print(f"  FLOPs/token: {dense['flops_per_token']:,}")

        print(f"\nMoE FFN:")
        print(f"  总参数量: {moe['total_params']:,}")
        print(f"  激活参数量: {moe['active_params']:,}")
        print(f"  FLOPs/token: {moe['flops_per_token']:,}")
        print(f"  参数效率比: {moe['param_efficiency_ratio']:.1f}x")

        print(f"\n对比:")
        print(f"  MoE 总参数 / Dense: {comp['moe_total_vs_dense_params']:.1f}x")
        print(f"  MoE 激活参数 / Dense: {comp['moe_active_vs_dense_params']:.1f}x")
        print(f"  MoE FLOPs / Dense FLOPs: {comp['moe_vs_dense_flops']:.2f}x")


if __name__ == "__main__":
    # 模拟路由数据
    batch_size = 8
    seq_len = 64
    num_experts = 8
    top_k = 2

    print("=" * 60)
    print("场景 1: 近似均匀的路由分布")
    print("=" * 60)

    # 均匀路由
    uniform_logits = torch.randn(batch_size, seq_len, num_experts) * 0.1
    stats_uniform = analyze_routing_distribution(uniform_logits, top_k)
    flops = compute_flops_comparison(
        d_model=512, d_ff_dense=2048,
        num_experts=8, top_k=2, d_ff_expert=2048
    )
    print_analysis_report(stats_uniform, flops)

    print("\n\n")
    print("=" * 60)
    print("场景 2: 不均匀的路由分布（路由坍塌模拟）")
    print("=" * 60)

    # 不均匀路由：专家 0 和 1 被严重偏好
    skewed_logits = torch.randn(batch_size, seq_len, num_experts)
    skewed_logits[:, :, 0] += 5.0  # 专家 0 高偏好
    skewed_logits[:, :, 1] += 3.0  # 专家 1 次偏好
    stats_skewed = analyze_routing_distribution(skewed_logits, top_k)
    print_analysis_report(stats_skewed)

    print("\n\n")
    print("=" * 60)
    print("场景 3: DeepSeek 风格的参数效率分析")
    print("=" * 60)

    flops_ds = compute_flops_comparison(
        d_model=4096,
        d_ff_dense=11008,      # 标准 Llama-7B FFN
        num_experts=64,
        top_k=6,
        d_ff_expert=1408,      # 细粒度专家
    )
    print_analysis_report(
        analyze_routing_distribution(
            torch.randn(4, 128, 64), top_k=6
        ),
        flops_ds,
    )
