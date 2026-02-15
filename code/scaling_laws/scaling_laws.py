"""
Scaling Laws 拟合与预测工具

本模块实现了 LLM Scaling Laws 的核心拟合和预测功能。

数学基础:
- 幂律模型: L(x) = A * x^(-alpha) + L_inf
- 联合模型 (Chinchilla): L(N, D) = A / N^alpha + B / D^beta + E
- 最优分配: N_opt ~ C^(beta/(alpha+beta)), D_opt ~ C^(alpha/(alpha+beta))

参考:
- Kaplan et al. (2020). Scaling Laws for Neural Language Models.
- Hoffmann et al. (2022). Training Compute-Optimal Large Language Models. (Chinchilla)
"""

import numpy as np
from scipy.optimize import curve_fit, minimize
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field


@dataclass
class ScalingLawParams:
    """
    Scaling Law 拟合参数的数据类

    Attributes:
        A: 参数量项系数
        B: 数据量项系数
        alpha: 参数量幂律指数
        beta: 数据量幂律指数
        E: 不可约损失 (irreducible loss)
    """
    A: float = 406.4
    B: float = 410.7
    alpha: float = 0.34
    beta: float = 0.28
    E: float = 1.69


@dataclass
class ExperimentData:
    """
    Scaling Laws 实验数据

    Attributes:
        N: 参数量数组
        D: 数据量数组 (tokens)
        L: 对应的最终损失数组
    """
    N: np.ndarray
    D: np.ndarray
    L: np.ndarray

    def __post_init__(self):
        """验证数据一致性"""
        assert len(self.N) == len(self.D) == len(self.L), \
            f"数据长度不一致: N={len(self.N)}, D={len(self.D)}, L={len(self.L)}"
        assert np.all(self.N > 0), "参数量必须为正"
        assert np.all(self.D > 0), "数据量必须为正"
        assert np.all(self.L > 0), "损失必须为正"


def single_variable_power_law(x: np.ndarray, a: float, alpha: float,
                               l_inf: float) -> np.ndarray:
    """
    单变量幂律模型

    L(x) = a * x^(-alpha) + L_inf

    Args:
        x: 自变量 (参数量 N 或 数据量 D 或 计算量 C)
        a: 系数
        alpha: 幂律指数
        l_inf: 不可约损失

    Returns:
        预测损失
    """
    return a * np.power(x, -alpha) + l_inf


def chinchilla_loss(N: np.ndarray, D: np.ndarray, A: float, B: float,
                    alpha: float, beta: float, E: float) -> np.ndarray:
    """
    Chinchilla 联合损失模型

    L(N, D) = A / N^alpha + B / D^beta + E

    Args:
        N: 参数量
        D: 数据量 (tokens)
        A: 参数量项系数
        B: 数据量项系数
        alpha: 参数量幂律指数
        beta: 数据量幂律指数
        E: 不可约损失

    Returns:
        预测损失
    """
    return A / np.power(N, alpha) + B / np.power(D, beta) + E


def _chinchilla_loss_flat(ND: np.ndarray, A: float, B: float,
                          alpha: float, beta: float, E: float) -> np.ndarray:
    """
    Chinchilla 损失模型的平坦化版本 (用于 curve_fit)

    curve_fit 要求自变量为单个数组, 因此将 N 和 D 拼接为一个数组:
    ND = [N_1, ..., N_m, D_1, ..., D_m]

    Args:
        ND: 拼接的 N 和 D 数组, 长度为 2m
        A, B, alpha, beta, E: 模型参数

    Returns:
        预测损失
    """
    m = len(ND) // 2
    N = ND[:m]
    D = ND[m:]
    return chinchilla_loss(N, D, A, B, alpha, beta, E)


class ScalingLawFitter:
    """
    Scaling Laws 拟合器

    支持两种模式:
    1. 单变量拟合: L(x) = a * x^(-alpha) + L_inf
    2. 联合拟合 (Chinchilla): L(N, D) = A / N^alpha + B / D^beta + E
    """

    def __init__(self):
        self.params: Optional[ScalingLawParams] = None
        self._single_var_params: Optional[Tuple[float, float, float]] = None
        self._fit_mode: Optional[str] = None

    def fit_single_variable(
        self,
        x: np.ndarray,
        losses: np.ndarray,
        p0: Optional[Tuple[float, float, float]] = None,
        bounds: Optional[Tuple] = None,
    ) -> Tuple[float, float, float]:
        """
        拟合单变量幂律模型

        L(x) = a * x^(-alpha) + L_inf

        Args:
            x: 自变量数组 (如参数量 N)
            losses: 对应的损失数组
            p0: 初始参数猜测 (a, alpha, l_inf)
            bounds: 参数约束, 格式为 ([lower], [upper])

        Returns:
            拟合参数 (a, alpha, l_inf)
        """
        if p0 is None:
            p0 = (10.0, 0.1, 1.5)
        if bounds is None:
            bounds = ([0, 0, 0], [np.inf, 2.0, np.inf])

        popt, pcov = curve_fit(
            single_variable_power_law,
            x, losses,
            p0=p0, bounds=bounds,
            maxfev=10000
        )

        self._single_var_params = tuple(popt)
        self._fit_mode = 'single'

        # 计算拟合质量
        y_pred = single_variable_power_law(x, *popt)
        residuals = losses - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        print(f"[拟合结果] a={popt[0]:.4f}, alpha={popt[1]:.4f}, "
              f"L_inf={popt[2]:.4f}")
        print(f"[拟合质量] R^2 = {r_squared:.6f}")

        return tuple(popt)

    def fit_chinchilla(
        self,
        data: ExperimentData,
        p0: Optional[Tuple[float, ...]] = None,
        bounds: Optional[Tuple] = None,
    ) -> ScalingLawParams:
        """
        拟合 Chinchilla 联合损失模型

        L(N, D) = A / N^alpha + B / D^beta + E

        使用非线性最小二乘法 (Levenberg-Marquardt 算法)。

        Args:
            data: 实验数据
            p0: 初始参数猜测 (A, B, alpha, beta, E)
            bounds: 参数约束

        Returns:
            拟合得到的 ScalingLawParams
        """
        if p0 is None:
            p0 = (400.0, 400.0, 0.3, 0.3, 1.5)
        if bounds is None:
            bounds = ([0, 0, 0, 0, 0], [1e6, 1e6, 2.0, 2.0, 10.0])

        # 将 N 和 D 拼接为一个数组 (curve_fit 要求)
        ND = np.concatenate([data.N, data.D])

        popt, pcov = curve_fit(
            _chinchilla_loss_flat,
            ND, data.L,
            p0=p0, bounds=bounds,
            maxfev=20000
        )

        self.params = ScalingLawParams(
            A=popt[0], B=popt[1],
            alpha=popt[2], beta=popt[3],
            E=popt[4]
        )
        self._fit_mode = 'chinchilla'

        # 计算拟合质量
        y_pred = chinchilla_loss(data.N, data.D, *popt)
        residuals = data.L - y_pred
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((data.L - np.mean(data.L)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        print(f"[Chinchilla 拟合结果]")
        print(f"  A={popt[0]:.4f}, B={popt[1]:.4f}")
        print(f"  alpha={popt[2]:.4f}, beta={popt[3]:.4f}")
        print(f"  E={popt[4]:.4f}")
        print(f"[拟合质量] R^2 = {r_squared:.6f}")

        return self.params

    def predict(self, N: float, D: float) -> float:
        """
        使用拟合好的模型预测损失

        Args:
            N: 参数量
            D: 数据量 (tokens)

        Returns:
            预测的损失值
        """
        if self._fit_mode == 'chinchilla' and self.params is not None:
            return chinchilla_loss(
                np.array([N]), np.array([D]),
                self.params.A, self.params.B,
                self.params.alpha, self.params.beta,
                self.params.E
            )[0]
        elif self._fit_mode == 'single' and self._single_var_params is not None:
            return single_variable_power_law(
                np.array([N]), *self._single_var_params
            )[0]
        else:
            raise ValueError("模型未拟合。请先调用 fit_single_variable 或 fit_chinchilla。")

    def predict_with_confidence(
        self,
        data: ExperimentData,
        N_target: float,
        D_target: float,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
    ) -> Dict[str, float]:
        """
        使用 Bootstrap 方法预测损失并估计置信区间

        通过有放回采样原始数据, 重复拟合多次, 得到预测值的分布。

        Args:
            data: 原始实验数据
            N_target: 目标参数量
            D_target: 目标数据量
            n_bootstrap: Bootstrap 重采样次数
            confidence: 置信水平 (默认 95%)

        Returns:
            包含 mean, std, lower, upper 的字典
        """
        n_samples = len(data.N)
        predictions = []

        for _ in range(n_bootstrap):
            # 有放回采样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_data = ExperimentData(
                N=data.N[indices],
                D=data.D[indices],
                L=data.L[indices]
            )

            try:
                # 静默拟合 (不打印)
                ND = np.concatenate([boot_data.N, boot_data.D])
                popt, _ = curve_fit(
                    _chinchilla_loss_flat,
                    ND, boot_data.L,
                    p0=(400.0, 400.0, 0.3, 0.3, 1.5),
                    bounds=([0, 0, 0, 0, 0], [1e6, 1e6, 2.0, 2.0, 10.0]),
                    maxfev=20000
                )
                pred = chinchilla_loss(
                    np.array([N_target]), np.array([D_target]), *popt
                )[0]
                predictions.append(pred)
            except RuntimeError:
                # 某些 Bootstrap 样本可能导致拟合失败, 跳过
                continue

        predictions = np.array(predictions)
        alpha_half = (1 - confidence) / 2

        result = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'lower': np.percentile(predictions, alpha_half * 100),
            'upper': np.percentile(predictions, (1 - alpha_half) * 100),
            'n_successful': len(predictions),
        }

        print(f"[Bootstrap 预测] N={N_target:.2e}, D={D_target:.2e}")
        print(f"  均值: {result['mean']:.4f}")
        print(f"  标准差: {result['std']:.4f}")
        print(f"  {confidence*100:.0f}% CI: [{result['lower']:.4f}, {result['upper']:.4f}]")
        print(f"  成功拟合次数: {result['n_successful']}/{n_bootstrap}")

        return result


def generate_synthetic_scaling_data(
    N_range: Tuple[float, float] = (1e7, 1e10),
    n_models: int = 8,
    tokens_per_param: float = 20.0,
    params: Optional[ScalingLawParams] = None,
    noise_std: float = 0.02,
    seed: int = 42,
) -> ExperimentData:
    """
    生成合成的 Scaling Laws 实验数据

    用于教学和测试目的。基于 Chinchilla 损失模型生成, 并添加噪声。

    Args:
        N_range: 参数量范围 (最小, 最大)
        n_models: 模型数量
        tokens_per_param: 每个参数对应的训练 token 数
        params: Scaling Law 参数 (默认使用 Chinchilla 参数)
        noise_std: 高斯噪声标准差
        seed: 随机种子

    Returns:
        合成的实验数据
    """
    rng = np.random.RandomState(seed)

    if params is None:
        params = ScalingLawParams()

    # 在对数空间均匀采样参数量
    N = np.logspace(np.log10(N_range[0]), np.log10(N_range[1]), n_models)
    D = tokens_per_param * N

    # 计算理论损失并添加噪声
    L_true = chinchilla_loss(N, D, params.A, params.B,
                             params.alpha, params.beta, params.E)
    L_noisy = L_true + rng.normal(0, noise_std, size=len(L_true))

    # 确保损失为正
    L_noisy = np.maximum(L_noisy, params.E * 0.9)

    return ExperimentData(N=N, D=D, L=L_noisy)


if __name__ == '__main__':
    print("=" * 70)
    print("Scaling Laws 拟合与预测工具 - 演示")
    print("=" * 70)

    # === 1. 生成合成数据 ===
    print("\n--- 1. 生成合成实验数据 ---")
    data = generate_synthetic_scaling_data(
        N_range=(1e7, 3e9),
        n_models=10,
        tokens_per_param=20.0,
        noise_std=0.01,
    )
    print(f"生成了 {len(data.N)} 个模型的数据:")
    for i in range(len(data.N)):
        print(f"  N={data.N[i]:.2e}, D={data.D[i]:.2e}, L={data.L[i]:.4f}")

    # === 2. 单变量拟合 ===
    print("\n--- 2. 单变量拟合: L(N) ---")
    fitter = ScalingLawFitter()
    a, alpha, l_inf = fitter.fit_single_variable(data.N, data.L)

    # === 3. Chinchilla 联合拟合 ===
    print("\n--- 3. Chinchilla 联合拟合: L(N, D) ---")
    params = fitter.fit_chinchilla(data)

    # === 4. 预测大模型性能 ===
    print("\n--- 4. 预测大模型性能 ---")
    targets = [
        (7e9, 140e9, "7B (Chinchilla最优)"),
        (7e9, 1e12, "7B (过度训练)"),
        (70e9, 1.4e12, "70B (Chinchilla最优)"),
    ]
    for N_t, D_t, desc in targets:
        pred = fitter.predict(N_t, D_t)
        print(f"  {desc}: L = {pred:.4f}")

    # === 5. Bootstrap 置信区间 ===
    print("\n--- 5. Bootstrap 预测置信区间 ---")
    result = fitter.predict_with_confidence(
        data,
        N_target=70e9,
        D_target=1.4e12,
        n_bootstrap=200,
        confidence=0.95,
    )
