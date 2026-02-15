"""
NTP 损失实现: Next Token Prediction + Label Smoothing + Z-loss

本模块实现了自回归语言模型的核心损失函数,
包括标准交叉熵、Label Smoothing 正则化和 Z-loss 稳定化技巧。

核心公式:
- NTP 损失: L_NTP = -(1/T) * sum_t log P(x_t | x_{<t})
- Label Smoothing: L_LS = (1-alpha) * L_CE + alpha * L_uniform
- Z-loss: L_z = c * (log sum exp(z))^2

参考:
- Vaswani et al. (2017). Attention Is All You Need. (Label Smoothing)
- Chowdhery et al. (2022). PaLM. (Z-loss)
- Szegedy et al. (2016). Rethinking the Inception Architecture. (Label Smoothing 原始论文)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class NTPLoss(nn.Module):
    """
    标准 Next Token Prediction 损失

    这是自回归语言模型最基本的损失函数:
    对每个位置, 用交叉熵衡量模型预测与真实 token 的差距。

    L = -(1/T) * sum_{t=1}^{T} log P(x_t | x_{<t})

    Args:
        ignore_index: 忽略的标签索引 (如 padding token)
        reduction: 损失聚合方式 ("mean" 或 "sum" 或 "none")
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算 NTP 损失

        Args:
            logits: [batch, seq_len, vocab_size] 模型输出的 logits
            targets: [batch, seq_len] 目标 token id

        Returns:
            标量损失值
        """
        # 展平为 [B*T, V] 和 [B*T]
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        loss = F.cross_entropy(
            logits_flat, targets_flat,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        return loss


class LabelSmoothingLoss(nn.Module):
    """
    带 Label Smoothing 的交叉熵损失

    Label Smoothing 将 one-hot 标签替换为软标签:
    - 正确类别: 1 - alpha
    - 其他类别: alpha / (V - 1)

    等价形式: L_LS = (1 - alpha) * L_CE + alpha * L_uniform

    这个等价形式的推导:

    L_LS = -sum_w y'_w * log P(w)
         = -(1-alpha) * log P(w*) - sum_{w != w*} (alpha/(V-1)) * log P(w)
         = (1-alpha) * L_CE + alpha * (-1/V * sum_w log P(w))  (近似)

    严格来说, 均匀分布部分应该是 -(1/(V-1)) * sum_{w!=w*} log P(w),
    但 PyTorch 的实现中直接对所有 V 个类计算均匀交叉熵。

    Args:
        vocab_size: 词汇表大小
        smoothing: 平滑系数 alpha (0.0 = 标准 CE, 1.0 = 均匀分布)
        ignore_index: 忽略的标签索引
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        计算带 Label Smoothing 的损失

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]

        Returns:
            标量损失值
        """
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # 使用 PyTorch 内置的 label_smoothing 参数
        loss = F.cross_entropy(
            logits_flat, targets_flat,
            label_smoothing=self.smoothing,
            ignore_index=self.ignore_index,
        )
        return loss

    def forward_manual(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        手动实现 Label Smoothing (用于教学理解)

        展示 Label Smoothing 的完整计算过程,
        而非直接调用 PyTorch 的 label_smoothing 参数。

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]

        Returns:
            (loss, details) 其中 details 包含 CE 和 uniform 部分
        """
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)  # [N, V]
        targets_flat = targets.view(-1)            # [N]

        # 创建有效位置掩码
        valid_mask = (targets_flat != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=logits.device), {}

        # 只在有效位置计算
        valid_logits = logits_flat[valid_mask]      # [M, V]
        valid_targets = targets_flat[valid_mask]     # [M]

        # 计算 log softmax
        log_probs = F.log_softmax(valid_logits, dim=-1)  # [M, V]

        # 1. 标准交叉熵部分: -log P(w*)
        ce_loss = F.nll_loss(log_probs, valid_targets, reduction="mean")

        # 2. 均匀分布部分: -(1/V) * sum_w log P(w)
        uniform_loss = -log_probs.mean()

        # 3. 组合
        total_loss = self.confidence * ce_loss + self.smoothing * uniform_loss

        details = {
            "ce_loss": ce_loss.item(),
            "uniform_loss": uniform_loss.item(),
            "total_loss": total_loss.item(),
        }

        return total_loss, details


class ZLoss(nn.Module):
    """
    Z-loss: Logit 归一化惩罚 (PaLM 使用)

    L_z = c * (log sum_w exp(z_w))^2

    作用机制:
    - log Z = log sum exp(z_w) 是对数配分函数
    - 当 logits 增长时, log Z 增大, L_z 增大
    - 梯度 dL_z/dz_i = 2c * logZ * softmax(z_i)
    - 形成自稳定反馈: logits 越大 -> 惩罚越大 -> 被拉回

    工程意义:
    - 防止混合精度训练中的数值溢出
    - 减少 loss spike 的频率
    - PaLM 使用 c = 1e-4

    Args:
        coefficient: Z-loss 系数 c
    """

    def __init__(self, coefficient: float = 1e-4):
        super().__init__()
        self.coefficient = coefficient

    def forward(
        self,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """
        计算 Z-loss

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len] 可选, 用于排除 padding 位置
            ignore_index: 忽略的标签索引

        Returns:
            标量 Z-loss 值
        """
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)  # [N, V]

        # log Z = logsumexp(logits)
        log_z = torch.logsumexp(logits_flat, dim=-1)  # [N]

        # 排除 padding 位置
        if targets is not None:
            targets_flat = targets.view(-1)
            valid_mask = (targets_flat != ignore_index)
            log_z = log_z[valid_mask]

        # Z-loss = c * mean(log_z^2)
        z_loss = self.coefficient * (log_z ** 2).mean()

        return z_loss


class CombinedPretrainingLoss(nn.Module):
    """
    组合预训练损失: NTP + Label Smoothing + Z-loss

    L_total = L_CE(label_smoothing) + L_z

    这是工业界常用的预训练损失配置:
    - PaLM: NTP + Z-loss (无 Label Smoothing)
    - 原始 Transformer: NTP + Label Smoothing (无 Z-loss)
    - 完整配置: NTP + Label Smoothing + Z-loss

    Args:
        vocab_size: 词汇表大小
        label_smoothing: Label Smoothing 系数 (0.0 = 不使用)
        z_loss_coeff: Z-loss 系数 (0.0 = 不使用)
        ignore_index: 忽略的标签索引
    """

    def __init__(
        self,
        vocab_size: int,
        label_smoothing: float = 0.0,
        z_loss_coeff: float = 0.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index

        # 主损失 (Label Smoothing 通过 F.cross_entropy 的参数实现)
        self.z_loss = ZLoss(coefficient=z_loss_coeff) if z_loss_coeff > 0 else None

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算组合损失

        Args:
            logits: [batch, seq_len, vocab_size]
            targets: [batch, seq_len]

        Returns:
            (total_loss, loss_dict) 其中 loss_dict 包含各项损失的数值
        """
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)

        # 1. 主 NTP 损失 (带可选的 Label Smoothing)
        ce_loss = F.cross_entropy(
            logits_flat, targets_flat,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
        )

        total_loss = ce_loss
        loss_dict = {"ce_loss": ce_loss.item()}

        # 2. Z-loss
        if self.z_loss is not None:
            z_loss_val = self.z_loss(logits, targets, self.ignore_index)
            total_loss = total_loss + z_loss_val
            loss_dict["z_loss"] = z_loss_val.item()

        loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict


if __name__ == "__main__":
    print("=" * 60)
    print("NTP 损失函数演示")
    print("=" * 60)

    # 模拟数据
    batch_size = 4
    seq_len = 16
    vocab_size = 100

    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    # 模拟一些 padding 位置
    targets[:, -3:] = -100

    # ---- 1. 标准 NTP 损失 ----
    print("\n--- 标准 NTP 损失 ---")
    ntp = NTPLoss()
    loss = ntp(logits, targets)
    print(f"NTP Loss: {loss.item():.4f}")
    print(f"PPL (exp(loss)): {torch.exp(loss).item():.2f}")

    # ---- 2. Label Smoothing 损失 ----
    print("\n--- Label Smoothing 损失 ---")
    for alpha in [0.0, 0.05, 0.1, 0.2]:
        ls = LabelSmoothingLoss(vocab_size, smoothing=alpha)
        loss = ls(logits, targets)
        print(f"  alpha={alpha:.2f} -> Loss: {loss.item():.4f}")

    # ---- 3. Label Smoothing 手动实现 (教学版) ----
    print("\n--- Label Smoothing 手动实现 ---")
    ls_manual = LabelSmoothingLoss(vocab_size, smoothing=0.1)
    loss, details = ls_manual.forward_manual(logits, targets)
    print(f"  CE 部分: {details['ce_loss']:.4f}")
    print(f"  Uniform 部分: {details['uniform_loss']:.4f}")
    print(f"  总损失: {details['total_loss']:.4f}")

    # ---- 4. Z-loss ----
    print("\n--- Z-loss ---")
    for c in [0.0, 1e-5, 1e-4, 1e-3]:
        if c == 0:
            print(f"  c={c:.0e} -> Z-loss: 0.0000 (未启用)")
            continue
        z = ZLoss(coefficient=c)
        z_val = z(logits, targets)
        print(f"  c={c:.0e} -> Z-loss: {z_val.item():.6f}")

    # ---- 5. 组合损失 ----
    print("\n--- 组合损失 ---")
    configs = [
        ("标准 CE", 0.0, 0.0),
        ("CE + Label Smoothing", 0.1, 0.0),
        ("CE + Z-loss (PaLM 风格)", 0.0, 1e-4),
        ("CE + LS + Z-loss (完整)", 0.1, 1e-4),
    ]

    for name, ls_val, z_val in configs:
        combined = CombinedPretrainingLoss(
            vocab_size, label_smoothing=ls_val, z_loss_coeff=z_val
        )
        total, details = combined(logits, targets)
        detail_str = " | ".join(f"{k}: {v:.4f}" for k, v in details.items())
        print(f"  [{name}] {detail_str}")

    # ---- 6. Z-loss 的梯度分析 ----
    print("\n--- Z-loss 梯度分析 ---")
    print("验证: logits 越大 -> Z-loss 梯度越大 (自稳定机制)")
    for scale in [1.0, 5.0, 10.0, 50.0]:
        scaled_logits = logits * scale
        scaled_logits.requires_grad_(True)
        z = ZLoss(coefficient=1e-4)
        z_val = z(scaled_logits, targets)
        z_val.backward()
        grad_norm = scaled_logits.grad.norm().item()
        print(f"  Logit 缩放 x{scale:.0f} -> Z-loss: {z_val.item():.4f}, 梯度范数: {grad_norm:.4f}")
