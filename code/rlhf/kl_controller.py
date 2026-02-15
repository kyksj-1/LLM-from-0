"""
KL 散度控制器模块

实现 RLHF 训练中的 KL 散度控制策略。

核心思想:
    在 RLHF 中，KL 惩罚约束策略不偏离参考模型太远:
        R_total = R_RM(x, y) - beta * KL(pi_theta || pi_ref)

    beta 的选择至关重要:
    - beta 太小: 策略偏离参考模型太远，可能导致 reward hacking
    - beta 太大: 策略过于保守，无法充分利用奖励信号

    自适应 KL 控制器会根据实际 KL 散度动态调整 beta:
    - 如果 KL > target: 增大 beta（加强约束）
    - 如果 KL < target: 减小 beta（放松约束）

本模块实现了三种 KL 控制策略:
1. 固定 beta (FixedKLController)
2. 自适应 beta (AdaptiveKLController) — 最常用
3. 基于 PID 的 beta 控制 (PIDKLController) — 更精细的控制
"""

import torch
from typing import Optional
import math


class FixedKLController:
    """
    固定 KL 系数控制器。

    最简单的策略：beta 在整个训练过程中保持不变。

    优点: 简单、可预测
    缺点: 无法适应训练动态
        - 训练初期 KL 可能很小（beta 浪费了约束能力）
        - 训练后期 KL 可能很大（beta 约束不够）

    适用场景: 快速实验、超参数搜索的基线
    """

    def __init__(self, kl_coef: float = 0.1):
        """
        Args:
            kl_coef: 固定的 KL 惩罚系数 beta
        """
        self.value = kl_coef

    def update(self, kl_value: float) -> None:
        """固定控制器不更新 beta。"""
        pass

    def get_coef(self) -> float:
        """返回当前 KL 系数。"""
        return self.value


class AdaptiveKLController:
    """
    自适应 KL 系数控制器。

    这是 InstructGPT / TRL 库使用的标准方法。

    算法:
        给定目标 KL 值 kl_target，在每次更新后:
        1. 计算实际 KL 与目标的比值: ratio = kl_actual / kl_target
        2. 根据比值调整 beta:
           - 如果 ratio > 1 (KL 过大): beta *= (1 + horizon * (ratio - 1))
           - 如果 ratio < 1 (KL 过小): beta /= (1 + horizon * (1 - ratio))

        horizon 控制调整速度:
        - horizon 大: 调整缓慢，更稳定
        - horizon 小: 调整迅速，更灵敏

    数学直觉:
        这本质上是一个比例控制器（P-controller）:
        log(beta_new) = log(beta_old) + K * (kl_actual - kl_target)

        其中 K 是控制增益。

    参考: Ziegler et al. "Fine-Tuning Language Models from Human Preferences" (2019)
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        kl_target: float = 6.0,
        horizon: float = 10000.0
    ):
        """
        Args:
            init_kl_coef: 初始 KL 系数
            kl_target: 目标 KL 散度值
                - 对于 7B 模型，典型值为 6.0
                - 对于更大的模型，可能需要更小的值
            horizon: 调整速度控制参数
                - 值越大，调整越缓慢
                - 典型值: 10000
        """
        self.value = init_kl_coef
        self.kl_target = kl_target
        self.horizon = horizon
        self._history = []  # 记录历史，用于分析

    def update(self, kl_value: float) -> None:
        """
        根据当前 KL 值更新系数。

        更新规则:
            proportional_error = (kl_value - kl_target) / kl_target
            multiplier = 1 + horizon_factor * proportional_error
            beta_new = beta_old * multiplier

        Args:
            kl_value: 当前 batch 的平均 KL 散度
        """
        # 计算比例误差
        proportional_error = (kl_value - self.kl_target) / self.kl_target

        # 计算乘法因子
        # 当 kl > target 时，proportional_error > 0，beta 增大
        # 当 kl < target 时，proportional_error < 0，beta 减小
        multiplier = 1.0 + (1.0 / self.horizon) * proportional_error

        # 更新 beta，确保非负
        self.value = max(self.value * multiplier, 1e-6)

        # 记录历史
        self._history.append({
            "kl": kl_value,
            "beta": self.value,
            "target": self.kl_target,
        })

    def get_coef(self) -> float:
        """返回当前 KL 系数。"""
        return self.value

    def get_history(self) -> list:
        """返回调整历史。"""
        return self._history


class PIDKLController:
    """
    基于 PID 控制的 KL 系数控制器。

    PID (Proportional-Integral-Derivative) 控制器比简单的比例控制更精细:
    - P (比例): 根据当前误差调整
    - I (积分): 根据累积误差调整，消除稳态误差
    - D (微分): 根据误差变化率调整，减少超调

    控制方程:
        u(t) = K_p * e(t) + K_i * integral(e) + K_d * de/dt

    其中:
        e(t) = kl_actual - kl_target  (误差)
        u(t) 用于调整 log(beta)

    这种控制器在 KL 目标变化或训练动态复杂时更稳定。
    """

    def __init__(
        self,
        init_kl_coef: float = 0.1,
        kl_target: float = 6.0,
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.05,
        integral_limit: float = 10.0
    ):
        """
        Args:
            init_kl_coef: 初始 KL 系数
            kl_target: 目标 KL 值
            kp: 比例增益
            ki: 积分增益
            kd: 微分增益
            integral_limit: 积分项的上限（防止积分饱和）
        """
        self.value = init_kl_coef
        self.kl_target = kl_target
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit

        # PID 状态
        self._integral = 0.0
        self._prev_error = 0.0
        self._history = []

    def update(self, kl_value: float) -> None:
        """
        使用 PID 控制更新 KL 系数。

        Args:
            kl_value: 当前 KL 散度
        """
        # 计算误差
        error = kl_value - self.kl_target

        # 比例项
        p_term = self.kp * error

        # 积分项（带限幅）
        self._integral += error
        self._integral = max(
            -self.integral_limit,
            min(self.integral_limit, self._integral)
        )
        i_term = self.ki * self._integral

        # 微分项
        d_term = self.kd * (error - self._prev_error)
        self._prev_error = error

        # PID 输出
        pid_output = p_term + i_term + d_term

        # 在 log 空间更新 beta
        log_beta = math.log(max(self.value, 1e-10))
        log_beta += pid_output
        self.value = max(math.exp(log_beta), 1e-6)

        self._history.append({
            "kl": kl_value,
            "beta": self.value,
            "target": self.kl_target,
            "p_term": p_term,
            "i_term": i_term,
            "d_term": d_term,
        })

    def get_coef(self) -> float:
        """返回当前 KL 系数。"""
        return self.value

    def get_history(self) -> list:
        """返回调整历史。"""
        return self._history

    def reset(self) -> None:
        """重置 PID 状态。"""
        self._integral = 0.0
        self._prev_error = 0.0


def simulate_kl_dynamics(
    controller,
    num_steps: int = 100,
    initial_kl: float = 2.0,
    noise_std: float = 0.5,
    drift_rate: float = 0.05,
    seed: int = 42
) -> dict:
    """
    模拟 KL 散度的动态变化过程，测试控制器的响应。

    模拟逻辑:
        - KL 初始值较低，随训练逐渐增大（模拟策略偏移）
        - 控制器增大 beta 来抑制 KL 增长
        - 添加随机噪声模拟训练中的波动

    这个模拟帮助理解:
    1. 不同控制器的响应速度
    2. 稳态误差（KL 是否收敛到目标）
    3. 超调现象（KL 是否在目标附近振荡）

    Args:
        controller: KL 控制器实例
        num_steps: 模拟步数
        initial_kl: 初始 KL 值
        noise_std: 噪声标准差
        drift_rate: KL 自然增长率（模拟策略偏移）
        seed: 随机种子

    Returns:
        模拟结果字典
    """
    torch.manual_seed(seed)

    kl_values = []
    beta_values = []
    current_kl = initial_kl

    for step in range(num_steps):
        # 记录当前状态
        kl_values.append(current_kl)
        beta_values.append(controller.get_coef())

        # 更新控制器
        controller.update(current_kl)

        # 模拟 KL 动态
        # KL 受两个因素影响:
        # 1. 自然漂移（策略偏移导致 KL 增大）
        # 2. beta 的约束（beta 越大，KL 增长越慢）
        beta = controller.get_coef()
        # beta 对 KL 的抑制效果
        suppression = -beta * 0.5 * max(current_kl - controller.kl_target, 0)
        # KL 更新
        current_kl = max(
            0.1,
            current_kl + drift_rate + suppression +
            torch.randn(1).item() * noise_std
        )

    return {
        "kl_values": kl_values,
        "beta_values": beta_values,
        "target": controller.kl_target,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("KL 散度控制器演示")
    print("=" * 60)

    # 1. 固定 KL 控制器
    print("\n--- 1. 固定 KL 控制器 ---")
    fixed = FixedKLController(kl_coef=0.1)
    print(f"初始 beta = {fixed.get_coef()}")
    fixed.update(kl_value=8.0)
    print(f"更新后 beta = {fixed.get_coef()} (不变)")

    # 2. 自适应 KL 控制器
    print("\n--- 2. 自适应 KL 控制器 ---")
    adaptive = AdaptiveKLController(
        init_kl_coef=0.1,
        kl_target=6.0,
        horizon=10000.0
    )
    print(f"初始 beta = {adaptive.get_coef():.6f}, 目标 KL = {adaptive.kl_target}")

    # 模拟几步
    kl_sequence = [2.0, 4.0, 6.0, 8.0, 10.0, 7.0, 5.0, 6.0]
    print(f"\n{'Step':>5} {'KL':>8} {'Beta':>10} {'Direction':>12}")
    print("-" * 40)
    for i, kl in enumerate(kl_sequence):
        old_beta = adaptive.get_coef()
        adaptive.update(kl)
        direction = "increase" if adaptive.get_coef() > old_beta else "decrease"
        print(f"{i:>5} {kl:>8.2f} {adaptive.get_coef():>10.6f} {direction:>12}")

    print("\n解读: 当 KL > target=6.0 时 beta 增大，当 KL < target 时 beta 减小")

    # 3. PID KL 控制器
    print("\n--- 3. PID KL 控制器 ---")
    pid = PIDKLController(
        init_kl_coef=0.1,
        kl_target=6.0,
        kp=0.1,
        ki=0.01,
        kd=0.05
    )
    print(f"初始 beta = {pid.get_coef():.6f}")
    print(f"PID 参数: Kp={pid.kp}, Ki={pid.ki}, Kd={pid.kd}")

    for i, kl in enumerate(kl_sequence):
        pid.update(kl)
    history = pid.get_history()
    print(f"\n最终 beta = {pid.get_coef():.6f}")
    print(f"最后一步的 PID 分量: P={history[-1]['p_term']:.4f}, "
          f"I={history[-1]['i_term']:.4f}, D={history[-1]['d_term']:.4f}")

    # 4. 控制器对比模拟
    print("\n--- 4. 控制器响应对比 ---")
    controllers = {
        "Fixed": FixedKLController(kl_coef=0.1),
        "Adaptive": AdaptiveKLController(init_kl_coef=0.1, kl_target=6.0, horizon=5000.0),
        "PID": PIDKLController(init_kl_coef=0.1, kl_target=6.0, kp=0.1, ki=0.01, kd=0.05),
    }

    print(f"\n{'Controller':<12} {'Final KL':>10} {'Final Beta':>12} {'Avg KL':>10}")
    print("-" * 48)
    for name, ctrl in controllers.items():
        result = simulate_kl_dynamics(ctrl, num_steps=50, initial_kl=2.0)
        avg_kl = sum(result["kl_values"]) / len(result["kl_values"])
        print(f"{name:<12} {result['kl_values'][-1]:>10.4f} "
              f"{result['beta_values'][-1]:>12.6f} {avg_kl:>10.4f}")

    print(f"\n目标 KL = 6.0")
    print("自适应和 PID 控制器应该能更好地将 KL 维持在目标附近")

    print("\n" + "=" * 60)
    print("KL 控制器演示完成!")
    print("=" * 60)
