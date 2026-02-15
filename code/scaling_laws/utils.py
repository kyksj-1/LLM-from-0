"""
Scaling Laws 工具函数

本模块提供了 Scaling Laws 相关的通用工具函数:
- 数字格式化
- 单位换算
- 数据加载与保存
- 常用的数学工具

参考:
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os


# ============================================================
# 数字格式化工具
# ============================================================

def format_params(n: float) -> str:
    """
    格式化参数量为人类可读字符串

    Args:
        n: 参数数量

    Returns:
        格式化字符串, 如 "7.0B", "1.5T", "300M"

    Examples:
        >>> format_params(7e9)
        '7.0B'
        >>> format_params(1.5e12)
        '1.5T'
    """
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    elif n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    else:
        return f"{n:.0f}"


def format_tokens(n: float) -> str:
    """
    格式化 token 数量

    Args:
        n: token 数量

    Returns:
        格式化字符串

    Examples:
        >>> format_tokens(15e12)
        '15.0T'
        >>> format_tokens(300e9)
        '300.0B'
    """
    return format_params(n)  # 复用参数格式化


def format_flops(flops: float) -> str:
    """
    格式化 FLOPs 为可读字符串

    Args:
        flops: FLOPs 数量

    Returns:
        格式化字符串, 如 "6.30 ZFLOPs"

    Examples:
        >>> format_flops(6.3e24)
        '6.30 YFLOPs'
    """
    prefixes = [
        (1e24, 'YFLOPs'),  # Yotta
        (1e21, 'ZFLOPs'),  # Zetta
        (1e18, 'EFLOPs'),  # Exa
        (1e15, 'PFLOPs'),  # Peta
        (1e12, 'TFLOPs'),  # Tera
        (1e9, 'GFLOPs'),   # Giga
        (1e6, 'MFLOPs'),   # Mega
    ]
    for threshold, suffix in prefixes:
        if flops >= threshold:
            return f"{flops/threshold:.2f} {suffix}"
    return f"{flops:.2e} FLOPs"


def format_cost(usd: float) -> str:
    """
    格式化美元成本

    Args:
        usd: 美元金额

    Returns:
        格式化字符串

    Examples:
        >>> format_cost(8800000)
        '$8.80M'
    """
    if usd >= 1e9:
        return f"${usd/1e9:.2f}B"
    elif usd >= 1e6:
        return f"${usd/1e6:.2f}M"
    elif usd >= 1e3:
        return f"${usd/1e3:.2f}K"
    else:
        return f"${usd:.2f}"


def format_time(hours: float) -> str:
    """
    格式化时间

    Args:
        hours: 小时数

    Returns:
        格式化字符串

    Examples:
        >>> format_time(720)
        '30.0 days'
    """
    if hours >= 8760:
        return f"{hours/8760:.1f} years"
    elif hours >= 24:
        return f"{hours/24:.1f} days"
    else:
        return f"{hours:.1f} hours"


# ============================================================
# 单位换算工具
# ============================================================

def tokens_to_bytes(tokens: float, bytes_per_token: float = 4.0) -> float:
    """
    将 token 数量转换为字节数

    Args:
        tokens: token 数量
        bytes_per_token: 每个 token 对应的平均字节数 (英文约 4, 中文约 2)

    Returns:
        字节数
    """
    return tokens * bytes_per_token


def bytes_to_human(n_bytes: float) -> str:
    """
    将字节数转换为人类可读字符串

    Args:
        n_bytes: 字节数

    Returns:
        格式化字符串, 如 "1.5 TB"
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} EB"


def flops_to_gpu_hours(
    flops: float,
    gpu_tflops: float = 990,
    mfu: float = 0.40,
) -> float:
    """
    将 FLOPs 转换为 GPU-hours

    Args:
        flops: 总 FLOPs
        gpu_tflops: GPU 峰值算力 (TFLOPS)
        mfu: Model FLOPs Utilization

    Returns:
        GPU-hours
    """
    effective_flops_per_sec = gpu_tflops * 1e12 * mfu
    return flops / (effective_flops_per_sec * 3600)


def gpu_hours_to_cost(
    gpu_hours: float,
    price_per_hour: float = 3.0,
) -> float:
    """
    将 GPU-hours 转换为美元成本

    Args:
        gpu_hours: GPU 小时数
        price_per_hour: 每 GPU 小时价格 (美元)

    Returns:
        美元成本
    """
    return gpu_hours * price_per_hour


# ============================================================
# 数学工具
# ============================================================

def compute_6nd(N: float, D: float) -> float:
    """
    计算 C ≈ 6ND

    Args:
        N: 参数量
        D: 训练 token 数

    Returns:
        估算的 FLOPs
    """
    return 6 * N * D


def chinchilla_optimal_ratio(alpha: float = 0.34, beta: float = 0.28) -> float:
    """
    计算 Chinchilla 最优的 tokens/param 比值

    Args:
        alpha: 参数量幂律指数
        beta: 数据量幂律指数

    Returns:
        最优 D/N 比值
    """
    # 在 Chinchilla 最优下, D_opt / N_opt 是一个常数
    # 根据论文, 这个值约为 20
    return 20.0


def overtraining_factor(N: float, D: float, optimal_ratio: float = 20.0) -> float:
    """
    计算过度训练因子

    Args:
        N: 参数量
        D: 实际训练 token 数
        optimal_ratio: Chinchilla 最优 D/N 比值

    Returns:
        过度训练因子 (>1 表示过度训练)
    """
    D_opt = optimal_ratio * N
    return D / D_opt


# ============================================================
# 数据 I/O
# ============================================================

def save_scaling_data(
    filepath: str,
    N: np.ndarray,
    D: np.ndarray,
    L: np.ndarray,
    metadata: Optional[Dict] = None,
) -> None:
    """
    保存 Scaling Laws 实验数据

    Args:
        filepath: 保存路径 (JSON 格式)
        N: 参数量数组
        D: 数据量数组
        L: 损失数组
        metadata: 附加元数据
    """
    data = {
        'N': N.tolist(),
        'D': D.tolist(),
        'L': L.tolist(),
        'metadata': metadata or {},
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[saved] {filepath}")


def load_scaling_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    加载 Scaling Laws 实验数据

    Args:
        filepath: 数据文件路径 (JSON 格式)

    Returns:
        (N, D, L, metadata)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return (
        np.array(data['N']),
        np.array(data['D']),
        np.array(data['L']),
        data.get('metadata', {}),
    )


# ============================================================
# 知名模型数据库
# ============================================================

KNOWN_MODELS = {
    # 模型名: (参数量, 训练 tokens, 组织)
    'GPT-3 175B': (175e9, 300e9, 'OpenAI'),
    'Gopher 280B': (280e9, 300e9, 'DeepMind'),
    'Chinchilla 70B': (70e9, 1.4e12, 'DeepMind'),
    'PaLM 540B': (540e9, 780e9, 'Google'),
    'PaLM 2': (340e9, 3.6e12, 'Google'),  # 估计值
    'Llama 1 65B': (65e9, 1.4e12, 'Meta'),
    'Llama 2 70B': (70e9, 2e12, 'Meta'),
    'Llama 3 8B': (8e9, 15e12, 'Meta'),
    'Llama 3 70B': (70e9, 15e12, 'Meta'),
    'Llama 3 405B': (405e9, 15e12, 'Meta'),
    'Gemma 2 2B': (2e9, 2e12, 'Google'),
    'Gemma 2 9B': (9e9, 8e12, 'Google'),
    'Gemma 2 27B': (27e9, 13e12, 'Google'),
    'DeepSeek-V2 236B': (236e9, 8.1e12, 'DeepSeek'),  # MoE 总参数
    'DeepSeek-V3 671B': (671e9, 14.8e12, 'DeepSeek'),  # MoE 总参数
    'Mistral 7B': (7e9, 8e12, 'Mistral'),
    'Mixtral 8x7B': (47e9, 8e12, 'Mistral'),  # MoE 总参数
}


def get_model_info(name: str) -> Optional[Dict]:
    """
    获取已知模型的信息

    Args:
        name: 模型名称

    Returns:
        模型信息字典, 如果未找到返回 None
    """
    if name in KNOWN_MODELS:
        N, D, org = KNOWN_MODELS[name]
        return {
            'name': name,
            'N': N,
            'D': D,
            'org': org,
            'C_approx': compute_6nd(N, D),
            'tokens_per_param': D / N,
            'overtraining_factor': overtraining_factor(N, D),
        }
    return None


def list_models(org: Optional[str] = None) -> List[Dict]:
    """
    列出所有已知模型

    Args:
        org: 按组织过滤 (如 'Google', 'Meta')

    Returns:
        模型信息列表
    """
    results = []
    for name in KNOWN_MODELS:
        info = get_model_info(name)
        if org is None or info['org'] == org:
            results.append(info)
    return results


if __name__ == '__main__':
    print("=" * 70)
    print("Scaling Laws 工具函数 - 演示")
    print("=" * 70)

    # === 1. 格式化工具 ===
    print("\n--- 1. 数字格式化 ---")
    print(f"  7B 参数: {format_params(7e9)}")
    print(f"  15T tokens: {format_tokens(15e12)}")
    print(f"  6.3e24 FLOPs: {format_flops(6.3e24)}")
    print(f"  $8.8M: {format_cost(8.8e6)}")
    print(f"  720 hours: {format_time(720)}")

    # === 2. 单位换算 ===
    print("\n--- 2. 单位换算 ---")
    tokens = 15e12
    print(f"  {format_tokens(tokens)} tokens ≈ {bytes_to_human(tokens_to_bytes(tokens))}")

    flops = compute_6nd(70e9, 15e12)
    gpu_h = flops_to_gpu_hours(flops, gpu_tflops=990, mfu=0.40)
    cost = gpu_hours_to_cost(gpu_h, price_per_hour=3.0)
    print(f"  70B x 15T tokens = {format_flops(flops)}")
    print(f"  = {gpu_h:,.0f} H100 GPU-hours")
    print(f"  = {format_cost(cost)}")

    # === 3. 已知模型数据库 ===
    print("\n--- 3. 已知模型 Scaling 配置 ---")
    print(f"{'Model':<25s} | {'N':>8s} | {'D':>8s} | {'C(6ND)':>12s} | "
          f"{'D/N':>6s} | {'OT factor':>10s}")
    print("-" * 80)
    for name in sorted(KNOWN_MODELS.keys()):
        info = get_model_info(name)
        print(f"{info['name']:<25s} | {format_params(info['N']):>8s} | "
              f"{format_tokens(info['D']):>8s} | {format_flops(info['C_approx']):>12s} | "
              f"{info['tokens_per_param']:>6.0f} | {info['overtraining_factor']:>10.1f}x")

    # === 4. 按组织过滤 ===
    print("\n--- 4. Meta 的模型 ---")
    meta_models = list_models(org='Meta')
    for m in meta_models:
        print(f"  {m['name']}: N={format_params(m['N'])}, D={format_tokens(m['D'])}, "
              f"OT={m['overtraining_factor']:.1f}x")

    # === 5. 数据保存/加载演示 ===
    print("\n--- 5. 数据保存/加载 ---")
    N_demo = np.array([1e7, 1e8, 1e9, 1e10])
    D_demo = 20 * N_demo
    L_demo = np.array([3.5, 2.8, 2.3, 2.0])

    save_scaling_data(
        'demo_scaling_data.json',
        N_demo, D_demo, L_demo,
        metadata={'source': 'synthetic', 'model': 'Chinchilla'}
    )

    N_loaded, D_loaded, L_loaded, meta = load_scaling_data('demo_scaling_data.json')
    print(f"  加载了 {len(N_loaded)} 个数据点, metadata: {meta}")

    # 清理演示文件
    os.remove('demo_scaling_data.json')
    print("  已清理演示文件")
