"""
Scaling Laws 可视化工具

本模块提供了 Scaling Laws 相关的可视化功能:
- 幂律拟合曲线
- Iso-FLOPs 曲线
- Kaplan vs Chinchilla 对比图
- 过度训练分析图
- 涌现能力演示图

使用方法:
    from visualization import ScalingVisualizer
    viz = ScalingVisualizer()
    viz.plot_power_law_fit(N, L, params)
    viz.plot_emergence_demo()

依赖:
- matplotlib
- numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from typing import Optional, Tuple, List, Dict

# 中文字体设置 (优先使用系统已安装的中文字体)
try:
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                               'Arial Unicode MS', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    pass


class ScalingVisualizer:
    """
    Scaling Laws 可视化工具

    提供一系列预定义的图表模板, 方便快速可视化 Scaling Laws 的实验结果。
    """

    def __init__(self, style: str = 'default', figsize: Tuple[int, int] = (10, 6)):
        """
        初始化可视化工具

        Args:
            style: matplotlib 样式
            figsize: 默认图表大小
        """
        self.figsize = figsize
        if style != 'default':
            plt.style.use(style)

    def plot_power_law_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fit_params: Optional[Tuple[float, float, float]] = None,
        xlabel: str = 'N (params)',
        ylabel: str = 'Loss',
        title: str = 'Scaling Law Fit',
        save_path: Optional[str] = None,
        extrapolate_factor: float = 10.0,
    ) -> plt.Figure:
        """
        绘制幂律拟合图

        在双对数坐标系中绘制实验数据点和拟合曲线。

        Args:
            x: 自变量 (如参数量 N)
            y: 因变量 (如损失 L)
            fit_params: 拟合参数 (a, alpha, l_inf), 如果为 None 则只画数据点
            xlabel: X 轴标签
            ylabel: Y 轴标签
            title: 图标题
            save_path: 保存路径 (如果为 None 则不保存)
            extrapolate_factor: 外推倍数

        Returns:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)

        # 实验数据点
        ax.scatter(x, y, color='#1f77b4', s=80, zorder=5,
                   edgecolors='white', linewidth=1.5, label='experiment data')

        # 拟合曲线
        if fit_params is not None:
            a, alpha, l_inf = fit_params
            x_fit = np.logspace(
                np.log10(x.min() / 2),
                np.log10(x.max() * extrapolate_factor),
                200
            )
            y_fit = a * np.power(x_fit, -alpha) + l_inf

            # 拟合区域内的曲线 (实线)
            mask_interp = (x_fit >= x.min()) & (x_fit <= x.max())
            mask_extrap = x_fit > x.max()

            ax.plot(x_fit[mask_interp], y_fit[mask_interp], 'r-', linewidth=2,
                    label=f'fit: L = {a:.2f} * x^(-{alpha:.3f}) + {l_inf:.2f}')

            # 外推区域 (虚线)
            if np.any(mask_extrap):
                ax.plot(x_fit[mask_extrap], y_fit[mask_extrap], 'r--',
                        linewidth=1.5, alpha=0.7, label='extrapolation')

            # 不可约损失线
            ax.axhline(y=l_inf, color='gray', linestyle=':', alpha=0.5,
                       label=f'L_inf = {l_inf:.2f}')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[saved] {save_path}")

        return fig

    def plot_iso_flops(
        self,
        A: float = 406.4,
        B: float = 410.7,
        alpha: float = 0.34,
        beta: float = 0.28,
        E: float = 1.69,
        C_values: Optional[List[float]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制 Iso-FLOPs 曲线 (固定 FLOPs, 变 N/D 分配)

        这是 Chinchilla 方法 2 的可视化: 对每个固定的计算预算,
        展示不同 N 值下的损失, 找到最优分配点。

        Args:
            A, B, alpha, beta, E: Scaling Law 参数
            C_values: 计算预算列表
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        if C_values is None:
            C_values = [1e19, 1e20, 1e21, 1e22, 1e23]

        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(C_values)))

        for i, C in enumerate(C_values):
            # 对每个 FLOPs 预算, 扫描不同的 N
            N_range = np.logspace(
                np.log10(C / (6 * 1e13)),   # 最小 N (D 极大)
                np.log10(C / (6 * 1e6)),     # 最大 N (D 极小)
                200
            )
            # 过滤合理范围
            N_range = N_range[(N_range >= 1e6) & (N_range <= 1e12)]
            D_range = C / (6 * N_range)
            D_range = np.maximum(D_range, 1)  # 确保 D > 0

            # 计算损失
            losses = A / np.power(N_range, alpha) + B / np.power(D_range, beta) + E

            # 找到最优点
            opt_idx = np.argmin(losses)

            ax.plot(N_range, losses, color=colors[i], linewidth=1.5,
                    label=f'C = {C:.0e}')
            ax.scatter([N_range[opt_idx]], [losses[opt_idx]],
                       color=colors[i], s=100, zorder=5, edgecolors='black',
                       linewidth=1.5, marker='*')

        ax.set_xscale('log')
        ax.set_xlabel('N (params)', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Iso-FLOPs Curves (Chinchilla Method 2)', fontsize=14)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[saved] {save_path}")

        return fig

    def plot_kaplan_vs_chinchilla(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制 Kaplan vs Chinchilla 最优分配对比图

        展示两种 Scaling Laws 在不同计算预算下的最优参数量和数据量分配差异。

        Args:
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        C_range = np.logspace(18, 26, 100)

        # Kaplan: N ~ C^0.73, D ~ C^0.27
        N_kaplan = 1e-3 * C_range ** 0.73
        D_kaplan = C_range / (6 * N_kaplan)

        # Chinchilla: N ~ C^0.50, D ~ C^0.50
        N_chinchilla = 0.3 * C_range ** 0.50
        D_chinchilla = C_range / (6 * N_chinchilla)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图: N vs C
        axes[0].plot(C_range, N_kaplan, 'b-', linewidth=2,
                     label='Kaplan: N ~ C^0.73')
        axes[0].plot(C_range, N_chinchilla, 'r-', linewidth=2,
                     label='Chinchilla: N ~ C^0.50')
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel('Compute C (FLOPs)', fontsize=12)
        axes[0].set_ylabel('Optimal N (params)', fontsize=12)
        axes[0].set_title('Optimal N vs Compute', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 右图: D vs C
        axes[1].plot(C_range, D_kaplan, 'b-', linewidth=2,
                     label='Kaplan: D ~ C^0.27')
        axes[1].plot(C_range, D_chinchilla, 'r-', linewidth=2,
                     label='Chinchilla: D ~ C^0.50')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Compute C (FLOPs)', fontsize=12)
        axes[1].set_ylabel('Optimal D (tokens)', fontsize=12)
        axes[1].set_title('Optimal D vs Compute', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.suptitle('Kaplan (2020) vs Chinchilla (2022): Optimal Allocation',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[saved] {save_path}")

        return fig

    def plot_overtraining_analysis(
        self,
        N: float = 8e9,
        A: float = 406.4,
        B: float = 410.7,
        alpha: float = 0.34,
        beta: float = 0.28,
        E: float = 1.69,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制过度训练分析图

        展示固定模型大小下, 损失随过度训练因子的变化。

        Args:
            N: 固定的模型参数量
            A, B, alpha, beta, E: Scaling Law 参数
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        # Chinchilla 最优数据量
        G = (alpha * A / (beta * B)) ** (1 / (alpha + beta))
        a_exp = beta / (alpha + beta)
        C_opt = 6 * (N / G) ** (1 / a_exp)
        D_opt = C_opt / (6 * N)

        # 扫描过度训练因子
        factors = np.logspace(-0.5, 2.5, 200)  # 0.3x 到 300x
        D_values = D_opt * factors
        losses = A / N ** alpha + B / np.power(D_values, beta) + E
        loss_opt = A / N ** alpha + B / D_opt ** beta + E

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图: 损失 vs 过度训练因子
        axes[0].plot(factors, losses, 'b-', linewidth=2)
        axes[0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7,
                        label='Chinchilla optimal')
        axes[0].axhline(y=loss_opt, color='red', linestyle=':', alpha=0.5)

        # 标注典型模型
        markers = {
            'Llama 3 8B': 15e12 / D_opt,
        }
        for name, factor in markers.items():
            if 0.3 <= factor <= 300:
                loss_at = A / N ** alpha + B / (D_opt * factor) ** beta + E
                axes[0].scatter([factor], [loss_at], s=100, zorder=5,
                                edgecolors='black', linewidth=1.5)
                axes[0].annotate(name, (factor, loss_at),
                                 textcoords="offset points", xytext=(10, 10),
                                 fontsize=9)

        axes[0].set_xscale('log')
        axes[0].set_xlabel('Over-training factor (D / D_opt)', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'Over-training Analysis (N = {N:.0e})', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # 右图: 边际收益 (损失下降率)
        marginal = -np.diff(losses) / np.diff(factors)
        factors_mid = (factors[:-1] + factors[1:]) / 2

        axes[1].plot(factors_mid, marginal, 'g-', linewidth=2)
        axes[1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7,
                        label='Chinchilla optimal')
        axes[1].set_xscale('log')
        axes[1].set_yscale('log')
        axes[1].set_xlabel('Over-training factor', fontsize=12)
        axes[1].set_ylabel('Marginal benefit (-dL/dr)', fontsize=12)
        axes[1].set_title('Diminishing Returns of Over-training', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[saved] {save_path}")

        return fig

    def plot_emergence_demo(
        self,
        k_values: Optional[List[int]] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制涌现能力演示图

        展示当底层 per-token 能力平滑增长时, 不同步数任务的
        精确匹配准确率如何呈现"涌现式"跳变。

        这是 Schaeffer et al. (2023) 核心论点的可视化。

        Args:
            k_values: 任务步数列表 (每个任务需要 k 个 token 全部正确)
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        if k_values is None:
            k_values = [1, 5, 10, 20, 50]

        # 模拟 per-token 准确率随模型规模的平滑增长
        N_range = np.logspace(7, 11, 200)  # 10M 到 100B
        # sigmoid 形状的平滑增长
        p = 0.5 + 0.5 * np.tanh(0.5 * (np.log10(N_range) - 9.0))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = plt.cm.tab10(np.linspace(0, 0.8, len(k_values)))

        # 左图: per-token 准确率 (平滑)
        axes[0].plot(N_range, p, 'k-', linewidth=2.5, label='per-token accuracy p(N)')
        axes[0].set_xscale('log')
        axes[0].set_xlabel('N (model params)', fontsize=12)
        axes[0].set_ylabel('Per-token accuracy p(N)', fontsize=12)
        axes[0].set_title('Underlying ability: smooth growth', fontsize=13)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1.05])

        # 右图: 精确匹配准确率 (涌现)
        for i, k in enumerate(k_values):
            em_acc = p ** k
            axes[1].plot(N_range, em_acc, color=colors[i], linewidth=2,
                         label=f'k={k} steps (EM = p^{k})')

        axes[1].set_xscale('log')
        axes[1].set_xlabel('N (model params)', fontsize=12)
        axes[1].set_ylabel('Exact Match Accuracy', fontsize=12)
        axes[1].set_title('Measured ability: apparent "emergence"', fontsize=13)
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([0, 1.05])

        plt.suptitle('Emergence as a metric artifact (Schaeffer et al., 2023)',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[saved] {save_path}")

        return fig

    def plot_scaling_comparison_table(
        self,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        绘制知名模型的 Scaling 配置对比图

        展示不同模型的参数量、训练 tokens、tokens/param 比值。

        Args:
            save_path: 保存路径

        Returns:
            matplotlib Figure 对象
        """
        # 知名模型数据
        models = {
            'GPT-3': (175e9, 300e9, 'OpenAI'),
            'Gopher': (280e9, 300e9, 'DeepMind'),
            'Chinchilla': (70e9, 1.4e12, 'DeepMind'),
            'PaLM': (540e9, 780e9, 'Google'),
            'Llama 1 65B': (65e9, 1.4e12, 'Meta'),
            'Llama 2 70B': (70e9, 2e12, 'Meta'),
            'Llama 3 8B': (8e9, 15e12, 'Meta'),
            'Llama 3 70B': (70e9, 15e12, 'Meta'),
            'Llama 3 405B': (405e9, 15e12, 'Meta'),
            'Gemma 2 9B': (9e9, 8e12, 'Google'),
        }

        names = list(models.keys())
        Ns = np.array([models[n][0] for n in names])
        Ds = np.array([models[n][1] for n in names])
        orgs = [models[n][2] for n in names]
        ratios = Ds / Ns

        # 颜色编码: 按组织
        org_colors = {
            'OpenAI': '#1f77b4',
            'DeepMind': '#ff7f0e',
            'Google': '#2ca02c',
            'Meta': '#d62728',
        }
        colors = [org_colors.get(o, 'gray') for o in orgs]

        fig, ax = plt.subplots(1, 1, figsize=(10, 7))

        scatter = ax.scatter(Ns, Ds, c=colors, s=150, zorder=5,
                             edgecolors='black', linewidth=1)

        # 标注模型名称
        for i, name in enumerate(names):
            ax.annotate(name, (Ns[i], Ds[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, fontweight='bold')

        # Chinchilla 最优线: D = 20 * N
        N_line = np.logspace(np.log10(1e9), np.log10(1e12), 100)
        ax.plot(N_line, 20 * N_line, 'k--', linewidth=1.5, alpha=0.5,
                label='Chinchilla optimal (D = 20N)')

        # 图例 (按组织)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c, edgecolor='black', label=o)
            for o, c in org_colors.items()
        ]
        legend_elements.append(
            plt.Line2D([0], [0], color='black', linestyle='--',
                       label='D = 20N (Chinchilla)')
        )
        ax.legend(handles=legend_elements, fontsize=9, loc='upper left')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('N (parameters)', fontsize=12)
        ax.set_ylabel('D (training tokens)', fontsize=12)
        ax.set_title('LLM Scaling Configurations: N vs D', fontsize=14)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[saved] {save_path}")

        return fig


if __name__ == '__main__':
    print("=" * 70)
    print("Scaling Laws 可视化工具 - 演示")
    print("=" * 70)

    viz = ScalingVisualizer()

    # === 1. 幂律拟合图 ===
    print("\n--- 1. 幂律拟合图 ---")
    N_data = np.array([1e7, 5e7, 1e8, 5e8, 1e9, 5e9])
    L_data = np.array([3.8, 3.2, 2.95, 2.6, 2.45, 2.2])
    fig1 = viz.plot_power_law_fit(
        N_data, L_data,
        fit_params=(8.0, 0.08, 1.8),
        save_path='scaling_power_law_fit.png'
    )

    # === 2. Iso-FLOPs 曲线 ===
    print("\n--- 2. Iso-FLOPs 曲线 ---")
    fig2 = viz.plot_iso_flops(save_path='scaling_iso_flops.png')

    # === 3. Kaplan vs Chinchilla 对比 ===
    print("\n--- 3. Kaplan vs Chinchilla 对比图 ---")
    fig3 = viz.plot_kaplan_vs_chinchilla(save_path='scaling_kaplan_vs_chinchilla.png')

    # === 4. 过度训练分析 ===
    print("\n--- 4. 过度训练分析图 ---")
    fig4 = viz.plot_overtraining_analysis(save_path='scaling_overtraining.png')

    # === 5. 涌现能力演示 ===
    print("\n--- 5. 涌现能力演示图 ---")
    fig5 = viz.plot_emergence_demo(save_path='scaling_emergence.png')

    # === 6. 知名模型 Scaling 配置 ===
    print("\n--- 6. 知名模型 Scaling 配置对比 ---")
    fig6 = viz.plot_scaling_comparison_table(save_path='scaling_model_comparison.png')

    print("\n所有图表已生成。")
    plt.show()
