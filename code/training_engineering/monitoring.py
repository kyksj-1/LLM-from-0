"""
训练监控工具：损失/梯度/权重追踪与可视化

大规模 LLM 训练中，监控是确保训练健康的关键。
本模块提供训练过程中的实时监控工具。

核心监控指标:
1. 损失曲线: 训练/验证 loss 的变化趋势
2. 梯度范数: 各层梯度范数的时间变化
3. 权重范数: 参数的绝对大小是否持续增长
4. 参数更新比率: |delta_W| / |W|，衡量每步更新的相对幅度
5. 学习率: 调度器的实际学习率

异常检测:
- Loss spike: 损失突然增大
- 梯度爆炸: 梯度范数突然增大
- NaN/Inf: 训练出现数值异常
- 权重退化: 权重范数持续不变（死神经元）

工具生态:
- WandB (Weights & Biases): 云端实验跟踪
- TensorBoard: 本地可视化
- 本模块提供独立的监控实现，也可与上述工具集成

参考:
- Weights & Biases Documentation
- TensorBoard Documentation
"""

import time
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn


class MetricsTracker:
    """
    训练指标跟踪器

    记录训练过程中的各种指标，支持:
    - 实时记录
    - 滑动平均
    - 历史查询
    - 导出到 JSON / CSV

    Args:
        window_size: 滑动平均窗口大小
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self._windows: Dict[str, List[float]] = defaultdict(list)

    def log(self, step: int, metrics: Dict[str, float]) -> None:
        """
        记录一组指标

        Args:
            step: 当前步数
            metrics: 指标字典 {名称: 值}
        """
        for name, value in metrics.items():
            if math.isfinite(value):
                self._history[name].append((step, value))
                self._windows[name].append(value)
                # 维护窗口大小
                if len(self._windows[name]) > self.window_size:
                    self._windows[name].pop(0)

    def get_smoothed(self, name: str) -> float:
        """
        获取指标的滑动平均值

        Args:
            name: 指标名称

        Returns:
            滑动平均值
        """
        window = self._windows.get(name, [])
        if not window:
            return 0.0
        return sum(window) / len(window)

    def get_latest(self, name: str) -> Optional[float]:
        """获取指标的最新值"""
        history = self._history.get(name, [])
        if history:
            return history[-1][1]
        return None

    def get_history(self, name: str) -> List[Tuple[int, float]]:
        """获取指标的完整历史"""
        return self._history.get(name, [])

    def detect_spike(
        self, name: str, threshold: float = 3.0
    ) -> bool:
        """
        检测指标是否出现突增（spike）

        判断条件: 当前值 > 滑动平均 * threshold

        Args:
            name: 指标名称
            threshold: 倍数阈值

        Returns:
            是否检测到 spike
        """
        current = self.get_latest(name)
        smoothed = self.get_smoothed(name)
        if current is None or smoothed <= 0:
            return False
        return current > smoothed * threshold

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        生成所有指标的摘要

        Returns:
            {指标名: {latest, smoothed, min, max, count}}
        """
        result = {}
        for name, history in self._history.items():
            values = [v for _, v in history]
            result[name] = {
                "latest": values[-1] if values else 0.0,
                "smoothed": self.get_smoothed(name),
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
                "count": len(values),
            }
        return result

    def export_json(self, path: str) -> None:
        """导出历史到 JSON 文件"""
        data = {}
        for name, history in self._history.items():
            data[name] = [{"step": s, "value": v} for s, v in history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class TrainingMonitor:
    """
    训练过程监控器

    整合各种监控指标，提供统一的接口。

    功能:
    1. 模型权重监控（权重范数、更新比率）
    2. 梯度监控（梯度范数、分层分析）
    3. 损失监控（spike 检测、趋势分析）
    4. 数值稳定性检查（NaN/Inf 检测）

    Args:
        model: 要监控的模型
        log_interval: 每隔多少步记录一次详细指标
        alert_grad_norm: 梯度范数超过此值时告警
        alert_loss_spike: 损失超过滑动平均的此倍数时告警
    """

    def __init__(
        self,
        model: nn.Module,
        log_interval: int = 100,
        alert_grad_norm: float = 10.0,
        alert_loss_spike: float = 5.0,
    ):
        self.model = model
        self.log_interval = log_interval
        self.alert_grad_norm = alert_grad_norm
        self.alert_loss_spike = alert_loss_spike

        self.tracker = MetricsTracker(window_size=100)
        self._prev_params: Dict[str, torch.Tensor] = {}
        self._alerts: List[Dict] = []

        # 保存初始权重快照（用于计算更新比率）
        self._snapshot_params()

    def _snapshot_params(self):
        """保存当前参数的快照"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._prev_params[name] = param.data.clone()

    def on_step_end(
        self,
        step: int,
        loss: float,
        lr: float,
        extra_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        每个训练步结束后调用

        收集并记录:
        1. 损失
        2. 学习率
        3. 梯度范数（如果是详细日志步）
        4. 权重统计（如果是详细日志步）

        Args:
            step: 当前步数
            loss: 当前损失
            lr: 当前学习率
            extra_metrics: 额外指标

        Returns:
            本步的监控报告
        """
        report = {"step": step, "loss": loss, "lr": lr, "alerts": []}

        # 基础指标
        metrics = {"loss": loss, "lr": lr}

        # 检查 NaN/Inf
        if not math.isfinite(loss):
            alert = {
                "type": "NaN_LOSS",
                "step": step,
                "message": f"损失出现非有限值: {loss}",
            }
            self._alerts.append(alert)
            report["alerts"].append(alert)

        # Loss spike 检测
        if self.tracker.detect_spike("loss", self.alert_loss_spike):
            smoothed = self.tracker.get_smoothed("loss")
            alert = {
                "type": "LOSS_SPIKE",
                "step": step,
                "message": f"Loss spike 检测到: {loss:.4f} > {smoothed:.4f} * {self.alert_loss_spike}",
            }
            self._alerts.append(alert)
            report["alerts"].append(alert)

        # 详细监控（每 log_interval 步）
        if step % self.log_interval == 0:
            # 梯度范数
            grad_stats = self._compute_gradient_stats()
            metrics.update(grad_stats)
            report["gradient_stats"] = grad_stats

            if grad_stats.get("grad_norm_global", 0) > self.alert_grad_norm:
                alert = {
                    "type": "GRAD_EXPLOSION",
                    "step": step,
                    "message": f"梯度范数过大: {grad_stats['grad_norm_global']:.4f}",
                }
                self._alerts.append(alert)
                report["alerts"].append(alert)

            # 权重统计
            weight_stats = self._compute_weight_stats()
            metrics.update(weight_stats)
            report["weight_stats"] = weight_stats

            # 参数更新比率
            update_ratios = self._compute_update_ratios()
            metrics.update(update_ratios)
            report["update_ratios"] = update_ratios

            # 更新参数快照
            self._snapshot_params()

        if extra_metrics:
            metrics.update(extra_metrics)

        self.tracker.log(step, metrics)
        return report

    def _compute_gradient_stats(self) -> Dict[str, float]:
        """
        计算梯度统计信息

        Returns:
            {
                grad_norm_global: 全局梯度 L2 范数,
                grad_norm_max_layer: 最大层梯度范数,
                grad_norm_min_layer: 最小层梯度范数,
                grad_nan_count: 含 NaN 梯度的参数数量,
            }
        """
        total_norm_sq = 0.0
        max_norm = 0.0
        min_norm = float("inf")
        nan_count = 0

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()

                if not math.isfinite(grad_norm):
                    nan_count += 1
                    continue

                total_norm_sq += grad_norm ** 2
                max_norm = max(max_norm, grad_norm)
                min_norm = min(min_norm, grad_norm)

        global_norm = math.sqrt(total_norm_sq)
        if min_norm == float("inf"):
            min_norm = 0.0

        return {
            "grad_norm_global": global_norm,
            "grad_norm_max_layer": max_norm,
            "grad_norm_min_layer": min_norm,
            "grad_nan_count": float(nan_count),
        }

    def _compute_weight_stats(self) -> Dict[str, float]:
        """
        计算权重统计信息

        Returns:
            {
                weight_norm_mean: 所有参数范数的均值,
                weight_norm_max: 最大参数范数,
                weight_abs_max: 权重绝对值最大值,
            }
        """
        norms = []
        abs_max = 0.0

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                norm = param.data.norm(2).item()
                norms.append(norm)
                abs_max = max(abs_max, param.data.abs().max().item())

        return {
            "weight_norm_mean": sum(norms) / max(len(norms), 1),
            "weight_norm_max": max(norms) if norms else 0.0,
            "weight_abs_max": abs_max,
        }

    def _compute_update_ratios(self) -> Dict[str, float]:
        """
        计算参数更新比率: |delta_W| / |W|

        这是衡量每步更新幅度的关键指标:
        - 比率太大 (> 0.01): 更新过于激进，可能不稳定
        - 比率太小 (< 1e-6): 学习停滞
        - 典型健康范围: 1e-4 ~ 1e-3

        Returns:
            {update_ratio_mean: 平均更新比率}
        """
        ratios = []

        for name, param in self.model.named_parameters():
            if name in self._prev_params and param.requires_grad:
                delta = (param.data - self._prev_params[name]).norm(2).item()
                weight_norm = param.data.norm(2).item()
                if weight_norm > 1e-12:
                    ratios.append(delta / weight_norm)

        mean_ratio = sum(ratios) / max(len(ratios), 1)
        return {"update_ratio_mean": mean_ratio}

    def get_alerts(self, last_n: Optional[int] = None) -> List[Dict]:
        """获取告警记录"""
        if last_n is not None:
            return self._alerts[-last_n:]
        return self._alerts

    def get_summary(self) -> Dict:
        """获取监控摘要"""
        return {
            "metrics_summary": self.tracker.summary(),
            "total_alerts": len(self._alerts),
            "alert_types": dict(
                defaultdict(
                    int,
                    {a["type"]: sum(1 for x in self._alerts if x["type"] == a["type"])
                     for a in self._alerts}
                )
            ),
        }


class WandBLogger:
    """
    WandB 日志记录器的轻量级封装

    使用方式:
    1. pip install wandb
    2. wandb login
    3. 在训练脚本中初始化 WandBLogger

    注意: 这是一个教学封装，展示与 WandB 集成的方式。
    """

    def __init__(
        self,
        project: str = "llm-pretraining",
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._wandb = None

        if enabled:
            try:
                import wandb
                self._wandb = wandb
                wandb.init(project=project, name=name, config=config)
            except ImportError:
                print("WandB 未安装。请运行: pip install wandb")
                self.enabled = False
            except Exception as e:
                print(f"WandB 初始化失败: {e}")
                self.enabled = False

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """记录指标"""
        if self.enabled and self._wandb:
            self._wandb.log(metrics, step=step)

    def finish(self) -> None:
        """结束记录"""
        if self.enabled and self._wandb:
            self._wandb.finish()


class TensorBoardLogger:
    """
    TensorBoard 日志记录器的轻量级封装

    使用方式:
    1. pip install tensorboard
    2. 在训练脚本中初始化 TensorBoardLogger
    3. 训练结束后运行: tensorboard --logdir=runs/
    """

    def __init__(
        self,
        log_dir: str = "runs",
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._writer = None

        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("TensorBoard 未安装。请运行: pip install tensorboard")
                self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量指标"""
        if self.enabled and self._writer:
            self._writer.add_scalar(tag, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int) -> None:
        """记录多个标量指标"""
        for tag, value in metrics.items():
            self.log_scalar(tag, value, step)

    def close(self) -> None:
        """关闭写入器"""
        if self._writer:
            self._writer.close()


if __name__ == "__main__":
    print("=" * 60)
    print("训练监控工具 演示")
    print("=" * 60)

    # 创建模型
    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # 初始化监控器
    monitor = TrainingMonitor(
        model, log_interval=5, alert_grad_norm=10.0, alert_loss_spike=5.0
    )

    # 模拟训练
    print("\n--- 模拟训练循环 ---")
    for step in range(1, 31):
        x = torch.randn(8, 64)
        target = torch.randint(0, 10, (8,))

        output = model(x)
        loss = criterion(output, target)

        # 模拟偶尔的 loss spike
        if step == 15:
            loss = loss * 10  # 人为制造 spike

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 监控
        report = monitor.on_step_end(
            step=step,
            loss=loss.item(),
            lr=optimizer.param_groups[0]["lr"],
        )

        if report["alerts"]:
            for alert in report["alerts"]:
                print(f"  [ALERT] step {step}: {alert['message']}")

        if step % 10 == 0:
            print(f"  step {step}: loss={loss.item():.4f}")

    # 监控摘要
    print("\n--- 监控摘要 ---")
    summary = monitor.get_summary()
    metrics_summary = summary["metrics_summary"]
    if "loss" in metrics_summary:
        ls = metrics_summary["loss"]
        print(f"  Loss: latest={ls['latest']:.4f}, min={ls['min']:.4f}, "
              f"max={ls['max']:.4f}, smoothed={ls['smoothed']:.4f}")
    print(f"  总告警数: {summary['total_alerts']}")

    # MetricsTracker 单独演示
    print("\n--- MetricsTracker 演示 ---")
    tracker = MetricsTracker(window_size=5)
    for i in range(10):
        tracker.log(i, {"test_metric": i * 1.5 + 0.1})
    print(f"  最新值: {tracker.get_latest('test_metric')}")
    print(f"  滑动平均: {tracker.get_smoothed('test_metric'):.2f}")
    print(f"  是否 spike: {tracker.detect_spike('test_metric', threshold=2.0)}")
