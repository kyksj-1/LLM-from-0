"""
训练稳定性诊断工具

大规模 LLM 训练中最常见的问题是训练不稳定：
- Loss spike（损失突增）
- 训练发散（loss 持续增大或出现 NaN）
- 梯度爆炸/消失
- 数值溢出

本模块提供系统化的诊断工具，帮助定位问题根因并给出修复建议。

诊断流程（对应 README.md 中的 Mermaid 决策树）:
  Step 1: 检查梯度范数 -> 是否爆炸？
  Step 2: 检查 logit 范围 -> 是否 NaN/Inf？
  Step 3: 检查数据 -> 是否有异常样本？
  Step 4: 回退学习率 -> 是否超出稳定范围？

Loss Spike 根因分类:
1. 数据异常: 异常样本、数据损坏、编码错误
2. 数值精度: FP16 溢出 (>65504)、梯度爆炸
3. 学习率过大: 超出当前 loss landscape 的稳定区间
4. 批次方差: 极端 mini-batch 的偶然影响

参考:
- Chowdhery et al. (2022). PaLM: Scaling Language Modeling with Pathways.
- Zhang et al. (2022). OPT: Open Pre-trained Transformer Language Models.
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
import torch.nn as nn


class DiagnosisLevel(Enum):
    """诊断级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DiagnosisReport:
    """
    诊断报告

    Attributes:
        level: 诊断级别
        category: 问题类别
        message: 诊断描述
        suggestion: 修复建议
        details: 详细数据
    """
    level: DiagnosisLevel
    category: str
    message: str
    suggestion: str
    details: Optional[Dict] = None


class StabilityDiagnoser:
    """
    训练稳定性诊断器

    提供系统化的训练问题诊断:
    1. 梯度健康检查
    2. 权重健康检查
    3. 损失健康检查
    4. 数值精度检查
    5. 综合诊断

    Args:
        model: 要诊断的模型
        grad_norm_threshold: 梯度范数告警阈值
        weight_norm_threshold: 权重范数告警阈值
        loss_spike_threshold: loss spike 检测倍数
        fp16_max: FP16 最大值（65504）
    """

    def __init__(
        self,
        model: nn.Module,
        grad_norm_threshold: float = 10.0,
        weight_norm_threshold: float = 1000.0,
        loss_spike_threshold: float = 5.0,
        fp16_max: float = 65504.0,
    ):
        self.model = model
        self.grad_norm_threshold = grad_norm_threshold
        self.weight_norm_threshold = weight_norm_threshold
        self.loss_spike_threshold = loss_spike_threshold
        self.fp16_max = fp16_max

        self._loss_history: List[float] = []

    def diagnose_gradients(self) -> List[DiagnosisReport]:
        """
        诊断梯度健康状况

        检查项目:
        1. 梯度中是否有 NaN/Inf
        2. 全局梯度范数是否过大（梯度爆炸）
        3. 梯度范数是否过小（梯度消失）
        4. 各层梯度范数的分布是否均匀

        Returns:
            诊断报告列表
        """
        reports = []
        layer_norms = {}
        nan_layers = []
        inf_layers = []
        zero_grad_layers = []
        total_norm_sq = 0.0

        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data

            # 检查 NaN
            if torch.isnan(grad).any():
                nan_layers.append(name)
                continue

            # 检查 Inf
            if torch.isinf(grad).any():
                inf_layers.append(name)
                continue

            norm = grad.norm(2).item()
            layer_norms[name] = norm
            total_norm_sq += norm ** 2

            # 检查零梯度（梯度消失）
            if norm < 1e-10:
                zero_grad_layers.append(name)

        # NaN 梯度
        if nan_layers:
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.CRITICAL,
                category="NaN_GRADIENT",
                message=f"发现 {len(nan_layers)} 层梯度包含 NaN",
                suggestion=(
                    "1. 检查损失函数是否接收到异常输入（如 log(0)）\n"
                    "2. 检查是否使用了 FP16 且发生了数值溢出\n"
                    "3. 尝试降低学习率或使用 BF16 精度\n"
                    "4. 检查数据预处理是否有异常值"
                ),
                details={"nan_layers": nan_layers},
            ))

        # Inf 梯度
        if inf_layers:
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.CRITICAL,
                category="INF_GRADIENT",
                message=f"发现 {len(inf_layers)} 层梯度包含 Inf",
                suggestion=(
                    "1. 如果使用 FP16，检查是否超出 65504 的表示范围\n"
                    "2. 考虑切换到 BF16（动态范围更大）\n"
                    "3. 启用梯度裁剪（gradient clipping）\n"
                    "4. 降低学习率"
                ),
                details={"inf_layers": inf_layers},
            ))

        # 全局梯度范数
        global_norm = math.sqrt(total_norm_sq) if total_norm_sq > 0 else 0.0
        if global_norm > self.grad_norm_threshold:
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.WARNING,
                category="GRAD_EXPLOSION",
                message=f"全局梯度范数过大: {global_norm:.4f} (阈值: {self.grad_norm_threshold})",
                suggestion=(
                    "1. 启用梯度裁剪: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n"
                    "2. 降低学习率\n"
                    "3. 检查是否有异常数据批次\n"
                    "4. 检查模型初始化是否合理"
                ),
                details={"global_norm": global_norm, "layer_norms": layer_norms},
            ))

        # 梯度消失
        if zero_grad_layers and len(zero_grad_layers) > len(layer_norms) * 0.1:
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.WARNING,
                category="GRAD_VANISHING",
                message=f"发现 {len(zero_grad_layers)} 层梯度接近零",
                suggestion=(
                    "1. 检查是否使用了 Pre-Norm（比 Post-Norm 更稳定）\n"
                    "2. 检查残差连接是否正常\n"
                    "3. 考虑使用更大的学习率\n"
                    "4. 检查初始化方法是否适配激活函数"
                ),
                details={"zero_grad_layers": zero_grad_layers},
            ))

        # 层间梯度不均匀
        if layer_norms:
            norms = list(layer_norms.values())
            if len(norms) > 2:
                ratio = max(norms) / max(min(norms), 1e-12)
                if ratio > 1000:
                    reports.append(DiagnosisReport(
                        level=DiagnosisLevel.WARNING,
                        category="GRAD_IMBALANCE",
                        message=f"层间梯度范数差异过大: max/min = {ratio:.1f}",
                        suggestion=(
                            "1. 检查模型架构是否有残差连接\n"
                            "2. 考虑使用残差分支缩放 (1/sqrt(2N))\n"
                            "3. 检查归一化层是否工作正常"
                        ),
                        details={"max_layer_norm": max(norms), "min_layer_norm": min(norms)},
                    ))

        if not reports:
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.INFO,
                category="GRAD_HEALTHY",
                message=f"梯度健康: 全局范数 = {global_norm:.4f}",
                suggestion="无需操作",
            ))

        return reports

    def diagnose_weights(self) -> List[DiagnosisReport]:
        """
        诊断权重健康状况

        检查项目:
        1. 权重中是否有 NaN/Inf
        2. 权重范数是否异常大
        3. 权重是否含有极端值（可能超出 FP16 范围）
        """
        reports = []
        nan_params = []
        large_params = []

        for name, param in self.model.named_parameters():
            data = param.data

            if torch.isnan(data).any():
                nan_params.append(name)
                continue

            if torch.isinf(data).any():
                nan_params.append(name)
                continue

            abs_max = data.abs().max().item()
            norm = data.norm(2).item()

            if norm > self.weight_norm_threshold:
                large_params.append((name, norm, abs_max))

            # 检查 FP16 溢出风险
            if abs_max > self.fp16_max * 0.9:
                reports.append(DiagnosisReport(
                    level=DiagnosisLevel.WARNING,
                    category="FP16_OVERFLOW_RISK",
                    message=f"参数 {name} 的最大绝对值 {abs_max:.1f} 接近 FP16 上限 {self.fp16_max}",
                    suggestion=(
                        "1. 考虑使用 BF16 替代 FP16（动态范围更大）\n"
                        "2. 增大 weight decay 限制权重增长\n"
                        "3. 检查学习率是否过大"
                    ),
                ))

        if nan_params:
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.CRITICAL,
                category="NaN_WEIGHTS",
                message=f"发现 {len(nan_params)} 个参数包含 NaN/Inf",
                suggestion="训练已损坏，需要从最近的 checkpoint 恢复",
                details={"nan_params": nan_params},
            ))

        if large_params:
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.WARNING,
                category="LARGE_WEIGHTS",
                message=f"发现 {len(large_params)} 个参数范数超过阈值",
                suggestion="增大 weight decay 或降低学习率",
                details={"large_params": [(n, f"{norm:.1f}") for n, norm, _ in large_params]},
            ))

        if not reports:
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.INFO,
                category="WEIGHTS_HEALTHY",
                message="权重健康",
                suggestion="无需操作",
            ))

        return reports

    def diagnose_loss(self, current_loss: float) -> List[DiagnosisReport]:
        """
        诊断损失健康状况

        Args:
            current_loss: 当前损失值

        Returns:
            诊断报告列表
        """
        reports = []

        # 检查 NaN/Inf
        if not math.isfinite(current_loss):
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.CRITICAL,
                category="NaN_LOSS",
                message=f"损失出现非有限值: {current_loss}",
                suggestion=(
                    "训练已发散。排查步骤:\n"
                    "1. 检查梯度是否有 NaN（调用 diagnose_gradients()）\n"
                    "2. 检查权重是否有 NaN（调用 diagnose_weights()）\n"
                    "3. 从最近的健康 checkpoint 恢复\n"
                    "4. 降低学习率后重新训练"
                ),
            ))
            return reports

        self._loss_history.append(current_loss)

        # Loss spike 检测
        if len(self._loss_history) > 10:
            recent_avg = sum(self._loss_history[-11:-1]) / 10
            if current_loss > recent_avg * self.loss_spike_threshold:
                reports.append(DiagnosisReport(
                    level=DiagnosisLevel.WARNING,
                    category="LOSS_SPIKE",
                    message=(
                        f"检测到 Loss Spike: 当前 {current_loss:.4f}, "
                        f"近期平均 {recent_avg:.4f} "
                        f"(倍数: {current_loss/recent_avg:.1f}x)"
                    ),
                    suggestion=(
                        "PaLM 的处理经验:\n"
                        "1. 从 spike 前 100 步的 checkpoint 重启\n"
                        "2. 跳过导致 spike 的数据批次\n"
                        "3. 如果频繁出现，降低学习率"
                    ),
                ))

        # 损失平台期检测
        if len(self._loss_history) > 100:
            recent_50 = self._loss_history[-50:]
            early_50 = self._loss_history[-100:-50]
            avg_recent = sum(recent_50) / len(recent_50)
            avg_early = sum(early_50) / len(early_50)
            relative_change = abs(avg_recent - avg_early) / max(abs(avg_early), 1e-8)

            if relative_change < 0.001:
                reports.append(DiagnosisReport(
                    level=DiagnosisLevel.INFO,
                    category="LOSS_PLATEAU",
                    message=f"损失进入平台期: 近50步平均 {avg_recent:.4f}, 前50步平均 {avg_early:.4f}",
                    suggestion=(
                        "1. 可能需要更多训练步数\n"
                        "2. 检查学习率是否已衰减到过小\n"
                        "3. 考虑调整数据混合比例"
                    ),
                ))

        if not reports:
            reports.append(DiagnosisReport(
                level=DiagnosisLevel.INFO,
                category="LOSS_HEALTHY",
                message=f"损失正常: {current_loss:.4f}",
                suggestion="无需操作",
            ))

        return reports

    def full_diagnosis(self, current_loss: float) -> List[DiagnosisReport]:
        """
        执行完整诊断

        按照诊断流程依次检查:
        1. 梯度健康
        2. 权重健康
        3. 损失健康

        Args:
            current_loss: 当前损失

        Returns:
            所有诊断报告
        """
        all_reports = []
        all_reports.extend(self.diagnose_gradients())
        all_reports.extend(self.diagnose_weights())
        all_reports.extend(self.diagnose_loss(current_loss))
        return all_reports


def inject_anomaly(
    model: nn.Module,
    anomaly_type: str = "grad_explosion",
    layer_name: Optional[str] = None,
    magnitude: float = 1000.0,
) -> None:
    """
    向模型注入异常，用于测试诊断工具

    可注入的异常类型:
    - "grad_explosion": 放大某层梯度
    - "nan_grad": 将某层梯度设为 NaN
    - "large_weight": 放大某层权重
    - "nan_weight": 将某层权重设为 NaN

    Args:
        model: 模型
        anomaly_type: 异常类型
        layer_name: 目标层名称（默认选择第一个线性层）
        magnitude: 异常幅度
    """
    target_param = None
    for name, param in model.named_parameters():
        if layer_name and name != layer_name:
            continue
        if param.requires_grad and "weight" in name:
            target_param = param
            break

    if target_param is None:
        print("未找到目标参数")
        return

    with torch.no_grad():
        if anomaly_type == "grad_explosion":
            if target_param.grad is not None:
                target_param.grad.mul_(magnitude)
            else:
                target_param.grad = torch.randn_like(target_param) * magnitude
        elif anomaly_type == "nan_grad":
            if target_param.grad is None:
                target_param.grad = torch.zeros_like(target_param)
            target_param.grad[0, 0] = float("nan")
        elif anomaly_type == "large_weight":
            target_param.mul_(magnitude)
        elif anomaly_type == "nan_weight":
            target_param.data[0, 0] = float("nan")
        else:
            raise ValueError(f"未知的异常类型: {anomaly_type}")


def format_diagnosis_report(reports: List[DiagnosisReport]) -> str:
    """
    格式化诊断报告为可读字符串

    Args:
        reports: 诊断报告列表

    Returns:
        格式化的报告文本
    """
    lines = ["=" * 60, "训练稳定性诊断报告", "=" * 60]

    # 按严重程度排序
    severity_order = {
        DiagnosisLevel.CRITICAL: 0,
        DiagnosisLevel.WARNING: 1,
        DiagnosisLevel.INFO: 2,
    }
    sorted_reports = sorted(reports, key=lambda r: severity_order.get(r.level, 3))

    for report in sorted_reports:
        level_str = f"[{report.level.value}]"
        lines.append(f"\n{level_str} {report.category}")
        lines.append(f"  {report.message}")
        if report.suggestion != "无需操作":
            lines.append(f"  建议:")
            for line in report.suggestion.split("\n"):
                lines.append(f"    {line}")
        if report.details:
            lines.append(f"  详细信息: {report.details}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("训练稳定性诊断工具 演示")
    print("=" * 60)

    # 创建模型
    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    diagnoser = StabilityDiagnoser(
        model,
        grad_norm_threshold=10.0,
        weight_norm_threshold=1000.0,
    )

    # 1. 正常情况诊断
    print("\n--- 正常训练诊断 ---")
    x = torch.randn(4, 64)
    y = model(x).sum()
    y.backward()

    reports = diagnoser.full_diagnosis(current_loss=3.14)
    print(format_diagnosis_report(reports))

    # 2. 梯度爆炸诊断
    print("\n--- 注入梯度爆炸后诊断 ---")
    model.zero_grad()
    y = model(x).sum()
    y.backward()
    inject_anomaly(model, anomaly_type="grad_explosion", magnitude=1000.0)

    reports = diagnoser.diagnose_gradients()
    print(format_diagnosis_report(reports))

    # 3. NaN 权重诊断
    print("\n--- 注入 NaN 权重后诊断 ---")
    inject_anomaly(model, anomaly_type="nan_weight")
    reports = diagnoser.diagnose_weights()
    print(format_diagnosis_report(reports))

    # 4. Loss spike 诊断
    print("\n--- Loss Spike 诊断 ---")
    diagnoser2 = StabilityDiagnoser(nn.Linear(1, 1))
    # 模拟正常损失序列
    for i in range(20):
        diagnoser2.diagnose_loss(3.0 - i * 0.01)
    # 注入 spike
    reports = diagnoser2.diagnose_loss(15.0)
    print(format_diagnosis_report(reports))
