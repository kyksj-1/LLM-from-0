"""
优化器实现：AdamW 从零实现 + 8-bit Adam 简化版

本模块从数学原理出发，逐步实现 AdamW 优化器，并提供一个简化版的
8-bit Adam 实现用于教学演示。

数学基础:

Adam 优化器的更新规则:
  m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t          (一阶矩，动量)
  v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2        (二阶矩，自适应学习率)
  m_hat_t = m_t / (1 - beta_1^t)                         (偏差修正)
  v_hat_t = v_t / (1 - beta_2^t)                         (偏差修正)
  theta_t = theta_{t-1} - lr * m_hat_t / (sqrt(v_hat_t) + eps)

AdamW 的核心区别:
  Weight Decay 与 L2 正则化在 Adam 下不等价。
  L2: 将惩罚加入梯度 -> g_t' = g_t + lambda * theta -> 惩罚被自适应学习率缩放
  WD: 直接缩减权重 -> theta_t = (1 - lr * lambda) * theta_{t-1} - lr * update
  AdamW 实现的是解耦的 Weight Decay。

参考:
- Loshchilov & Hutter (2019). Decoupled Weight Decay Regularization.
- Kingma & Ba (2015). Adam: A Method for Stochastic Optimization.
- Dettmers et al. (2022). 8-bit Optimizers via Block-wise Quantization.
"""

import math
from typing import Dict, List, Optional, Tuple, Iterator

import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """
    AdamW 优化器的从零实现

    与标准 Adam 的关键区别:
    - Weight Decay 直接作用于权重，而非加入梯度
    - 这使得正则化效果与自适应学习率解耦

    显存开销分析（FP32 精度下）:
    - 模型参数: P * 4 字节
    - 一阶矩 m_t: P * 4 字节
    - 二阶矩 v_t: P * 4 字节
    - 梯度: P * 4 字节
    - 合计: 16P 字节（即 16 * 参数量 字节）
    - 例: 7B 模型 -> 约 112 GB

    Args:
        params: 模型参数
        lr: 学习率（默认 1e-3）
        betas: 一阶矩和二阶矩的衰减率（默认 (0.9, 0.95)）
            - beta_1=0.9: 通用默认，控制梯度的指数移动平均
            - beta_2=0.95: GPT-3/LLaMA 常用值，比原始 Adam 的 0.999 更小
              原因: 大规模训练中数据分布变化大，更小的 beta_2 使优化器
              对近期梯度方差更敏感，提高稳定性
        eps: 防止除零的小常数（默认 1e-8）
            - 某些训练中使用 1e-6 可提高稳定性
        weight_decay: 权重衰减系数（默认 0.1）
            - 通常对嵌入层和 LayerNorm 不施加 weight_decay
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
    ):
        if lr < 0.0:
            raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"无效的 beta_1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"无效的 beta_2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"无效的 epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"无效的 weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一步参数更新

        更新过程:
        1. 计算一阶矩 m_t (梯度的指数移动平均 -> 动量)
        2. 计算二阶矩 v_t (梯度平方的指数移动平均 -> 自适应学习率)
        3. 偏差修正: 消除零初始化带来的偏差
        4. 参数更新: theta -= lr * m_hat / (sqrt(v_hat) + eps)
        5. 权重衰减: theta *= (1 - lr * weight_decay)  [解耦!]

        Returns:
            损失值（如果提供了 closure）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                # 获取或初始化优化器状态
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # 一阶矩: 梯度的指数移动平均 (动量)
                    state["exp_avg"] = torch.zeros_like(p)
                    # 二阶矩: 梯度平方的指数移动平均
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m = state["exp_avg"]       # 一阶矩
                v = state["exp_avg_sq"]    # 二阶矩
                state["step"] += 1
                t = state["step"]

                # 步骤 1: 更新一阶矩（动量）
                # m_t = beta_1 * m_{t-1} + (1 - beta_1) * g_t
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                # 步骤 2: 更新二阶矩（自适应学习率的基础）
                # v_t = beta_2 * v_{t-1} + (1 - beta_2) * g_t^2
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 步骤 3: 偏差修正
                # 因为 m_0 = 0, v_0 = 0，初始阶段矩估计偏小
                # 修正: m_hat = m_t / (1 - beta_1^t)
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t

                # 等价于: step_size = lr / bias_correction1
                # 然后: p -= step_size * m / (sqrt(v / bias_correction2) + eps)
                step_size = lr / bias_correction1
                bias_correction2_sqrt = math.sqrt(bias_correction2)

                # 步骤 4: 计算更新量并应用
                # denom = sqrt(v_hat) + eps = sqrt(v / (1 - beta_2^t)) + eps
                denom = (v.sqrt() / bias_correction2_sqrt).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

                # 步骤 5: 解耦的 Weight Decay
                # 关键: 这里直接缩减权重，而非加入梯度
                # theta_t = theta_t * (1 - lr * lambda)
                if weight_decay > 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

        return loss


class Adam8bit(Optimizer):
    """
    8-bit Adam 简化实现（教学版本）

    核心思想: 将优化器状态（m_t, v_t）量化为 8-bit 整数存储，
    更新时反量化为 FP32 计算，然后重新量化。

    量化策略（Block-wise Quantization）:
    1. 将状态张量分成固定大小的块（block_size）
    2. 每个块独立计算缩放因子: scale = max(|block|) / 127
    3. 量化: q = round(value / scale)
    4. 存储: 8-bit 整数 + 每个块的 FP32 缩放因子

    显存节省:
    - 标准 AdamW: 每参数 8 字节（m: 4B + v: 4B）
    - 8-bit Adam: 每参数约 2 字节 + 少量开销
    - 节省约 75% 优化器显存

    注意: 这是教学简化版，实际生产中请使用 bitsandbytes 库。

    参考:
    - Dettmers et al. (2022). 8-bit Optimizers via Block-wise Quantization.
    """

    def __init__(
        self,
        params: Iterator[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        block_size: int = 2048,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, block_size=block_size
        )
        super().__init__(params, defaults)

    @staticmethod
    def _quantize_block(tensor: torch.Tensor, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        分块量化: FP32 -> INT8

        量化公式:
          scale_i = max(|block_i|) / 127
          q_i = round(block_i / scale_i)   （取值范围 [-127, 127]）

        Args:
            tensor: 要量化的 FP32 张量
            block_size: 块大小

        Returns:
            (量化后的 INT8 张量, 每个块的缩放因子)
        """
        flat = tensor.flatten()
        n = flat.numel()
        # 对齐到块大小
        n_padded = ((n + block_size - 1) // block_size) * block_size
        padded = torch.zeros(n_padded, device=flat.device, dtype=flat.dtype)
        padded[:n] = flat

        # 分块
        blocks = padded.view(-1, block_size)
        # 每个块的缩放因子
        scales = blocks.abs().max(dim=1).values / 127.0
        scales = scales.clamp(min=1e-12)  # 防止除零

        # 量化
        quantized = (blocks / scales.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)

        return quantized, scales

    @staticmethod
    def _dequantize_block(
        quantized: torch.Tensor,
        scales: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        分块反量化: INT8 -> FP32

        反量化公式:
          value = q * scale

        Args:
            quantized: INT8 张量
            scales: 每个块的缩放因子
            original_shape: 原始张量形状

        Returns:
            反量化后的 FP32 张量
        """
        # 反量化
        dequantized = quantized.float() * scales.unsqueeze(1)
        # 恢复形状
        n = 1
        for s in original_shape:
            n *= s
        return dequantized.flatten()[:n].view(original_shape)

    @torch.no_grad()
    def step(self, closure=None):
        """
        8-bit Adam 的更新步骤

        流程:
        1. 反量化 m_t, v_t 为 FP32
        2. 用标准 Adam 公式更新 m_t, v_t
        3. 计算参数更新
        4. 重新量化 m_t, v_t 为 INT8

        Returns:
            损失值（如果提供了 closure）
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            block_size = group["block_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    # 初始化为零（量化形式）
                    zero_tensor = torch.zeros_like(p)
                    m_q, m_s = self._quantize_block(zero_tensor, block_size)
                    v_q, v_s = self._quantize_block(zero_tensor, block_size)
                    state["exp_avg_q"] = m_q
                    state["exp_avg_scales"] = m_s
                    state["exp_avg_sq_q"] = v_q
                    state["exp_avg_sq_scales"] = v_s

                state["step"] += 1
                t = state["step"]

                # 步骤 1: 反量化
                m = self._dequantize_block(
                    state["exp_avg_q"], state["exp_avg_scales"], p.shape
                )
                v = self._dequantize_block(
                    state["exp_avg_sq_q"], state["exp_avg_sq_scales"], p.shape
                )

                # 步骤 2: 标准 Adam 更新
                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # 步骤 3: 偏差修正 + 参数更新
                bias_correction1 = 1.0 - beta1 ** t
                bias_correction2 = 1.0 - beta2 ** t
                step_size = lr / bias_correction1
                denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                p.addcdiv_(m, denom, value=-step_size)

                # Weight Decay
                if weight_decay > 0.0:
                    p.add_(p, alpha=-lr * weight_decay)

                # 步骤 4: 重新量化
                m_q, m_s = self._quantize_block(m, block_size)
                v_q, v_s = self._quantize_block(v, block_size)
                state["exp_avg_q"] = m_q
                state["exp_avg_scales"] = m_s
                state["exp_avg_sq_q"] = v_q
                state["exp_avg_sq_scales"] = v_s

        return loss


def separate_weight_decay_params(
    model: torch.nn.Module,
    weight_decay: float = 0.1,
    no_decay_keywords: Optional[List[str]] = None,
) -> List[Dict]:
    """
    分离需要/不需要 weight decay 的参数组

    通常不对以下参数施加 weight decay:
    - 偏置项（bias）: 参数量很少，正则化收益低
    - LayerNorm / RMSNorm 的参数: 归一化层参数不应被正则化
    - 嵌入层（某些方案中）

    Args:
        model: PyTorch 模型
        weight_decay: 权重衰减系数
        no_decay_keywords: 不施加 weight decay 的参数名关键词

    Returns:
        两个参数组: [{有 WD 的参数}, {无 WD 的参数}]
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "layernorm", "rmsnorm", "ln_", "norm"]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 检查参数名是否包含不需要 WD 的关键词
        name_lower = name.lower()
        if any(kw in name_lower for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    # 统计信息
    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)
    print(f"参数分组: {n_decay:,} 个参数施加 weight_decay={weight_decay}, "
          f"{n_no_decay:,} 个参数不施加 weight_decay")

    return param_groups


if __name__ == "__main__":
    import torch.nn as nn

    print("=" * 60)
    print("优化器实现 演示")
    print("=" * 60)

    # 创建一个简单模型
    model = nn.Sequential(
        nn.Linear(64, 256),
        nn.LayerNorm(256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # 1. 参数分组
    print("\n--- 参数分组 ---")
    param_groups = separate_weight_decay_params(model, weight_decay=0.1)

    # 2. 自定义 AdamW 优化器
    print("\n--- AdamW 优化器 ---")
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )

    # 模拟训练
    x = torch.randn(8, 64)
    target = torch.randint(0, 10, (8,))
    criterion = nn.CrossEntropyLoss()

    print("模拟 5 步训练:")
    for step in range(5):
        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"  step {step+1}: loss = {loss.item():.4f}")

    # 3. 8-bit Adam 演示
    print("\n--- 8-bit Adam 优化器 ---")
    model2 = nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )
    optimizer_8bit = Adam8bit(
        model2.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        block_size=256,
    )

    print("模拟 5 步训练:")
    for step in range(5):
        output = model2(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer_8bit.step()
        optimizer_8bit.zero_grad()
        print(f"  step {step+1}: loss = {loss.item():.4f}")

    # 4. 显存对比
    print("\n--- 优化器显存对比 ---")
    n_params = sum(p.numel() for p in model.parameters())
    adamw_bytes = n_params * 4 * 2  # m + v, 各 FP32
    adam8bit_bytes = n_params * 1 * 2 + (n_params // 256) * 4 * 2  # INT8 + scales
    print(f"模型参数量: {n_params:,}")
    print(f"AdamW 优化器状态: {adamw_bytes / 1024:.1f} KB")
    print(f"8-bit Adam 状态:  {adam8bit_bytes / 1024:.1f} KB (约 {adam8bit_bytes/adamw_bytes*100:.0f}%)")
