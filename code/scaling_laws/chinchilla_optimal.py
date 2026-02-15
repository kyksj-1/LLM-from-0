"""
Chinchilla 最优配置计算器

本模块实现了基于 Chinchilla Scaling Laws 的计算最优配置工具。

核心功能:
- 给定计算预算, 计算最优的参数量 N 和数据量 D
- 给定目标模型大小, 计算最优训练数据量
- 过度训练分析: 评估偏离 Chinchilla 最优的损失增量
- Kaplan vs Chinchilla 对比分析

数学基础:
- Chinchilla 损失模型: L(N, D) = A / N^alpha + B / D^beta + E
- Lagrange 乘子法: min L(N,D) s.t. C = 6ND
- 最优分配: N_opt ~ C^(beta/(alpha+beta)), D_opt ~ C^(alpha/(alpha+beta))

参考:
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple


@dataclass
class ScalingConfig:
    """
    Scaling Laws 参数配置

    提供了 Kaplan (2020) 和 Chinchilla (2022) 两组参数。
    """
    A: float
    B: float
    alpha: float
    beta: float
    E: float
    name: str

    @staticmethod
    def chinchilla() -> 'ScalingConfig':
        """Chinchilla (2022) 参数"""
        return ScalingConfig(
            A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69,
            name="Chinchilla (2022)"
        )

    @staticmethod
    def kaplan() -> 'ScalingConfig':
        """Kaplan (2020) 近似参数 (基于论文中的幂律指数换算)"""
        return ScalingConfig(
            A=6.0, B=7.0, alpha=0.076, beta=0.095, E=1.50,
            name="Kaplan (2020)"
        )


class ChinchillaCalculator:
    """
    Chinchilla 最优配置计算器

    给定计算预算或模型规模, 计算 Chinchilla 最优的资源分配。

    使用方法:
        calc = ChinchillaCalculator()
        result = calc.optimal_allocation(C=1e23)
        calc.compare_with_overtraining(N=8e9, D=15e12)
    """

    def __init__(self, config: Optional[ScalingConfig] = None):
        """
        初始化计算器

        Args:
            config: Scaling Laws 参数配置, 默认使用 Chinchilla 参数
        """
        if config is None:
            config = ScalingConfig.chinchilla()
        self.config = config
        self._a_exp = config.beta / (config.alpha + config.beta)
        self._b_exp = config.alpha / (config.alpha + config.beta)

    def loss(self, N: float, D: float) -> float:
        """
        计算 Chinchilla 损失

        L(N, D) = A / N^alpha + B / D^beta + E

        Args:
            N: 参数量
            D: 训练数据量 (tokens)

        Returns:
            预测损失
        """
        c = self.config
        return c.A / N ** c.alpha + c.B / D ** c.beta + c.E

    def optimal_allocation(self, C: float) -> Dict[str, float]:
        """
        给定计算预算, 计算 Chinchilla 最优分配

        通过 Lagrange 乘子法的解析解:
        N_opt = G * (C/6)^(beta/(alpha+beta))
        D_opt = (C/6)^(alpha/(alpha+beta)) / G

        其中 G = (alpha*A / (beta*B))^(1/(alpha+beta))

        Args:
            C: 计算预算 (FLOPs)

        Returns:
            包含 N_opt, D_opt, predicted_loss 等的字典
        """
        c = self.config

        # 计算比例常数 G
        G = (c.alpha * c.A / (c.beta * c.B)) ** (1 / (c.alpha + c.beta))

        # 最优分配
        base = C / 6.0
        N_opt = G * base ** self._a_exp
        D_opt = base ** self._b_exp / G

        # 验证: C = 6 * N_opt * D_opt (应该近似等于 C)
        C_check = 6 * N_opt * D_opt

        result = {
            'N_opt': N_opt,
            'D_opt': D_opt,
            'predicted_loss': self.loss(N_opt, D_opt),
            'tokens_per_param': D_opt / N_opt,
            'C_budget': C,
            'C_check': C_check,
            'config_name': c.name,
        }

        return result

    def optimal_for_model_size(self, N: float) -> Dict[str, float]:
        """
        给定目标模型大小, 计算 Chinchilla 最优训练数据量

        根据 Chinchilla 最优比例: D_opt ≈ 20 * N

        更精确的计算:
        D_opt = N / G^2  (其中 G 是 Lagrange 乘子法推导的常数)

        Args:
            N: 目标参数量

        Returns:
            包含 D_opt, C_required 等的字典
        """
        c = self.config
        G = (c.alpha * c.A / (c.beta * c.B)) ** (1 / (c.alpha + c.beta))

        # 从 N_opt = G * (C/6)^a 反推 C
        # (C/6)^a = N/G => C/6 = (N/G)^(1/a) => C = 6*(N/G)^(1/a)
        C = 6 * (N / G) ** (1 / self._a_exp)
        D_opt = C / (6 * N)

        return {
            'N': N,
            'D_opt': D_opt,
            'C_required': C,
            'predicted_loss': self.loss(N, D_opt),
            'tokens_per_param': D_opt / N,
        }

    def overtraining_analysis(
        self,
        N: float,
        D: float,
    ) -> Dict[str, float]:
        """
        过度训练分析

        比较实际配置 (N, D) 与 Chinchilla 最优配置的差异。

        Args:
            N: 实际参数量
            D: 实际训练数据量 (tokens)

        Returns:
            包含 overtraining_factor, loss_increase 等的字典
        """
        # 计算同等 FLOPs 下的 Chinchilla 最优
        C = 6 * N * D
        optimal = self.optimal_allocation(C)

        # 计算实际损失
        actual_loss = self.loss(N, D)
        optimal_loss = optimal['predicted_loss']
        loss_increase = actual_loss - optimal_loss

        # 计算过度训练因子
        D_chinchilla = optimal['D_opt']
        overtraining_factor = D / D_chinchilla if D > D_chinchilla else 0

        # 如果用 N 的 Chinchilla 最优数据量来算
        model_optimal = self.optimal_for_model_size(N)
        D_for_N = model_optimal['D_opt']
        overtraining_vs_model = D / D_for_N

        return {
            'N': N,
            'D': D,
            'C': C,
            'actual_loss': actual_loss,
            'optimal_loss': optimal_loss,
            'loss_increase': loss_increase,
            'loss_increase_pct': loss_increase / optimal_loss * 100,
            'N_opt_for_same_C': optimal['N_opt'],
            'D_opt_for_same_C': optimal['D_opt'],
            'overtraining_factor_vs_iso_flops': overtraining_factor,
            'overtraining_factor_vs_model_size': overtraining_vs_model,
        }

    def compare_kaplan_chinchilla(
        self, C_values: Optional[List[float]] = None
    ) -> List[Dict[str, float]]:
        """
        对比 Kaplan 和 Chinchilla 在不同计算预算下的最优分配

        Args:
            C_values: 计算预算列表 (FLOPs), 默认从 1e18 到 1e26

        Returns:
            对比结果列表
        """
        if C_values is None:
            C_values = [10 ** i for i in range(18, 27)]

        calc_kaplan = ChinchillaCalculator(ScalingConfig.kaplan())
        calc_chinchilla = ChinchillaCalculator(ScalingConfig.chinchilla())

        results = []
        for C in C_values:
            kaplan_opt = calc_kaplan.optimal_allocation(C)
            chinchilla_opt = calc_chinchilla.optimal_allocation(C)

            results.append({
                'C': C,
                'kaplan_N': kaplan_opt['N_opt'],
                'kaplan_D': kaplan_opt['D_opt'],
                'chinchilla_N': chinchilla_opt['N_opt'],
                'chinchilla_D': chinchilla_opt['D_opt'],
                'N_ratio': kaplan_opt['N_opt'] / chinchilla_opt['N_opt'],
                'D_ratio': kaplan_opt['D_opt'] / chinchilla_opt['D_opt'],
            })

        return results

    def sweep_overtraining(
        self,
        N: float,
        factors: Optional[List[float]] = None,
    ) -> List[Dict[str, float]]:
        """
        扫描不同过度训练因子下的损失

        Args:
            N: 固定的模型参数量
            factors: 过度训练因子列表 (1.0 = Chinchilla 最优)

        Returns:
            每个因子对应的损失和成本信息
        """
        if factors is None:
            factors = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

        model_opt = self.optimal_for_model_size(N)
        D_opt = model_opt['D_opt']

        results = []
        for r in factors:
            D = D_opt * r
            loss = self.loss(N, D)
            C = 6 * N * D

            results.append({
                'factor': r,
                'D': D,
                'loss': loss,
                'C': C,
                'loss_vs_optimal': loss - model_opt['predicted_loss'],
            })

        return results


def format_number(n: float) -> str:
    """
    格式化大数字为可读字符串

    Args:
        n: 数字

    Returns:
        格式化后的字符串, 如 "7.0B", "1.5T"
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
        return f"{n:.1f}"


if __name__ == '__main__':
    print("=" * 70)
    print("Chinchilla 最优配置计算器 - 演示")
    print("=" * 70)

    calc = ChinchillaCalculator()

    # === 1. 给定计算预算, 计算最优分配 ===
    print("\n--- 1. 给定计算预算的最优分配 ---")
    budgets = [1e20, 1e21, 1e22, 1e23, 1e24, 1e25]
    print(f"{'FLOPs':>12s} | {'N_opt':>10s} | {'D_opt':>10s} | {'Loss':>8s} | {'D/N':>6s}")
    print("-" * 60)
    for C in budgets:
        r = calc.optimal_allocation(C)
        print(f"{C:>12.2e} | {format_number(r['N_opt']):>10s} | "
              f"{format_number(r['D_opt']):>10s} | {r['predicted_loss']:>8.4f} | "
              f"{r['tokens_per_param']:>6.1f}")

    # === 2. 给定模型大小, 计算最优训练量 ===
    print("\n--- 2. 给定模型大小的 Chinchilla 最优训练量 ---")
    model_sizes = [1e8, 1e9, 7e9, 13e9, 70e9, 175e9, 540e9]
    print(f"{'Model':>10s} | {'D_opt':>10s} | {'C_required':>12s} | {'Loss':>8s} | {'D/N':>6s}")
    print("-" * 60)
    for N in model_sizes:
        r = calc.optimal_for_model_size(N)
        print(f"{format_number(N):>10s} | {format_number(r['D_opt']):>10s} | "
              f"{r['C_required']:>12.2e} | {r['predicted_loss']:>8.4f} | "
              f"{r['tokens_per_param']:>6.1f}")

    # === 3. 过度训练分析 ===
    print("\n--- 3. 过度训练分析: Llama 3 系列 ---")
    models = [
        ("Llama 3 8B", 8e9, 15e12),
        ("Llama 3 70B", 70e9, 15e12),
        ("Llama 3 405B", 405e9, 15e12),
    ]
    for name, N, D in models:
        r = calc.overtraining_analysis(N, D)
        print(f"\n  {name} (N={format_number(N)}, D={format_number(D)}):")
        print(f"    实际损失: {r['actual_loss']:.4f}")
        print(f"    同 FLOPs 下 Chinchilla 最优损失: {r['optimal_loss']:.4f}")
        print(f"    损失增加: +{r['loss_increase']:.4f} ({r['loss_increase_pct']:.1f}%)")
        print(f"    过度训练因子 (vs 模型最优): {r['overtraining_factor_vs_model_size']:.1f}x")
        print(f"    同 FLOPs 最优 N: {format_number(r['N_opt_for_same_C'])}")

    # === 4. Kaplan vs Chinchilla 对比 ===
    print("\n--- 4. Kaplan vs Chinchilla 最优分配对比 ---")
    comparisons = calc.compare_kaplan_chinchilla()
    print(f"{'FLOPs':>12s} | {'Kaplan N':>10s} | {'Chinch N':>10s} | "
          f"{'Kaplan D':>10s} | {'Chinch D':>10s} | {'N ratio':>8s}")
    print("-" * 75)
    for r in comparisons:
        print(f"{r['C']:>12.2e} | {format_number(r['kaplan_N']):>10s} | "
              f"{format_number(r['chinchilla_N']):>10s} | "
              f"{format_number(r['kaplan_D']):>10s} | "
              f"{format_number(r['chinchilla_D']):>10s} | "
              f"{r['N_ratio']:>8.2f}")

    # === 5. 过度训练因子扫描 ===
    print("\n--- 5. 8B 模型过度训练因子扫描 ---")
    sweep = calc.sweep_overtraining(N=8e9)
    print(f"{'Factor':>8s} | {'D':>10s} | {'Loss':>8s} | {'Loss增加':>10s} | {'FLOPs':>12s}")
    print("-" * 60)
    for r in sweep:
        print(f"{r['factor']:>8.1f}x | {format_number(r['D']):>10s} | "
              f"{r['loss']:>8.4f} | {r['loss_vs_optimal']:>+10.4f} | {r['C']:>12.2e}")
