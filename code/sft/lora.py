"""
LoRA (Low-Rank Adaptation) 从零实现

本模块实现了 LoRA 的核心组件，包括：
- LoRALinear: 带 LoRA 适配器的线性层
- apply_lora_to_model: 将 LoRA 应用到模型的指定层
- 数学等价性验证

数学基础:
- 核心公式: h = W_0 x + (alpha/r) * B A x
- 其中 W_0 是冻结的预训练权重，B ∈ R^(d_out × r), A ∈ R^(r × d_in)
- 初始化: A ~ N(0, sigma^2), B = 0，保证训练开始时 Delta W = 0
- 参数节省: 2dr / d^2 = 2r/d (当 d_out = d_in = d)

参考:
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional, Dict, Set


class LoRALinear(nn.Module):
    """
    带 LoRA 适配器的线性层

    将原始线性变换 h = W_0 x 扩展为:
    h = W_0 x + (alpha / r) * B A x

    其中:
    - W_0: 冻结的预训练权重 (d_out × d_in)
    - A: 下投影矩阵 (r × d_in)，Kaiming 初始化
    - B: 上投影矩阵 (d_out × r)，零初始化
    - alpha: 缩放因子
    - r: LoRA 秩

    Args:
        original_linear: 原始的 nn.Linear 层
        rank: LoRA 秩 (r)
        alpha: 缩放因子 (alpha)
        dropout: LoRA Dropout 概率
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.original = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        d_out, d_in = original_linear.weight.shape

        # 冻结原始权重
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        # LoRA 矩阵
        # A: 下投影 (r × d_in)
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        # B: 上投影 (d_out × r)
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))

        # 初始化 A: Kaiming 均匀分布
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 已经初始化为零

        # 可选的 Dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 记录维度信息
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: h = W_0 x + (alpha/r) * B A x

        Args:
            x: 输入张量 (..., d_in)

        Returns:
            输出张量 (..., d_out)
        """
        # 原始线性变换（冻结权重）
        original_output = self.original(x)

        # LoRA 路径
        # x: (..., d_in) -> @ A^T -> (..., r) -> @ B^T -> (..., d_out)
        lora_input = self.lora_dropout(x)
        lora_output = lora_input @ self.lora_A.T @ self.lora_B.T
        lora_output = lora_output * self.scaling

        return original_output + lora_output

    def merge_weights(self) -> None:
        """
        将 LoRA 权重合并到原始权重中（就地操作）

        合并后: W_merged = W_0 + (alpha/r) * B @ A
        合并后可删除 LoRA 参数以节省显存
        """
        with torch.no_grad():
            delta_w = self.scaling * (self.lora_B @ self.lora_A)
            self.original.weight.data += delta_w

    def get_merged_linear(self) -> nn.Linear:
        """
        返回合并后的新 nn.Linear 层（不修改当前层）

        Returns:
            合并了 LoRA 权重的新 nn.Linear 层
        """
        merged = nn.Linear(
            self.d_in, self.d_out,
            bias=self.original.bias is not None,
        )

        with torch.no_grad():
            merged.weight.data = (
                self.original.weight.data + self.scaling * (self.lora_B @ self.lora_A)
            )
            if self.original.bias is not None:
                merged.bias.data = self.original.bias.data.clone()

        return merged

    def lora_parameters(self) -> List[nn.Parameter]:
        """返回 LoRA 可训练参数列表"""
        return [self.lora_A, self.lora_B]

    def num_lora_parameters(self) -> int:
        """返回 LoRA 参数总数"""
        return self.lora_A.numel() + self.lora_B.numel()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.d_in}, out_features={self.d_out}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"lora_params={self.num_lora_parameters():,}"
        )


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Set[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> nn.Module:
    """
    将 LoRA 应用到模型的指定模块

    遍历模型的所有 nn.Linear 层，如果层名匹配 target_modules，
    则用 LoRALinear 替换。

    Args:
        model: 原始模型
        target_modules: 需要添加 LoRA 的模块名集合
                       例如 {"q_proj", "v_proj", "k_proj", "o_proj"}
        rank: LoRA 秩
        alpha: 缩放因子
        dropout: LoRA Dropout

    Returns:
        添加了 LoRA 的模型（就地修改）
    """
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 遍历并替换目标模块
    replaced_count = 0
    for name, module in model.named_modules():
        # 检查是否是需要替换的模块
        # name 形如 "layer.0.self_attn.q_proj"
        module_name = name.split(".")[-1] if "." in name else name

        if module_name in target_modules and isinstance(module, nn.Linear):
            # 获取父模块
            parent_name = ".".join(name.split(".")[:-1])
            parent = model
            if parent_name:
                for attr in parent_name.split("."):
                    parent = getattr(parent, attr)

            # 替换为 LoRALinear
            lora_layer = LoRALinear(
                module, rank=rank, alpha=alpha, dropout=dropout
            )
            setattr(parent, module_name, lora_layer)
            replaced_count += 1

    print(f"已将 {replaced_count} 个线性层替换为 LoRALinear (rank={rank}, alpha={alpha})")

    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """
    获取模型中所有 LoRA 可训练参数

    Args:
        model: 添加了 LoRA 的模型

    Returns:
        LoRA 参数列表
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            lora_params.extend(module.lora_parameters())
    return lora_params


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    获取 LoRA 参数的 state_dict（用于保存 LoRA 权重）

    Args:
        model: 添加了 LoRA 的模型

    Returns:
        只包含 LoRA 参数的字典
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()
    return lora_state


if __name__ == "__main__":
    print("=" * 60)
    print("LoRA 从零实现演示")
    print("=" * 60)

    torch.manual_seed(42)

    # === 1. 基本功能测试 ===
    print("\n--- 1. 基本功能测试 ---")

    # 创建原始线性层
    linear = nn.Linear(512, 256, bias=True)
    print(f"原始层参数量: {sum(p.numel() for p in linear.parameters()):,}")

    # 添加 LoRA
    lora_linear = LoRALinear(linear, rank=8, alpha=16.0)
    print(f"LoRA 参数量: {lora_linear.num_lora_parameters():,}")
    print(f"参数比例: {lora_linear.num_lora_parameters() / (512 * 256) * 100:.2f}%")

    # 测试前向传播
    x = torch.randn(4, 32, 512)
    output = lora_linear(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # === 2. 初始化验证: 初始 LoRA 输出应为零 ===
    print("\n--- 2. 初始化验证 ---")

    linear2 = nn.Linear(256, 128)
    lora_linear2 = LoRALinear(linear2, rank=4, alpha=8.0)

    x2 = torch.randn(2, 256)
    original_out = linear2(x2)
    lora_out = lora_linear2(x2)

    diff = (original_out - lora_out).abs().max().item()
    print(f"初始状态下 LoRA 输出与原始输出的最大差异: {diff:.2e}")
    print(f"验证通过: {'是' if diff < 1e-6 else '否'}（应为零，因为 B=0）")

    # === 3. 合并等价性验证 ===
    print("\n--- 3. 合并等价性验证 ---")

    # 模拟训练（随机修改 LoRA 参数）
    with torch.no_grad():
        lora_linear.lora_A.normal_(0, 0.01)
        lora_linear.lora_B.normal_(0, 0.01)

    x3 = torch.randn(4, 32, 512)

    # LoRA 分离计算
    output_separate = lora_linear(x3)

    # 合并后计算
    merged_linear = lora_linear.get_merged_linear()
    output_merged = merged_linear(x3)

    max_diff = (output_separate - output_merged).abs().max().item()
    print(f"分离计算 vs 合并计算的最大误差: {max_diff:.2e}")
    print(f"数学等价性验证: {'通过' if max_diff < 1e-5 else '失败'}")

    # === 4. 应用到简单模型 ===
    print("\n--- 4. 应用到简单模型 ---")

    class SimpleTransformerLayer(nn.Module):
        """简单的 Transformer 层（用于演示 LoRA 应用）"""
        def __init__(self, d_model: int = 256):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model)
            self.k_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.o_proj = nn.Linear(d_model, d_model)
            self.ffn_up = nn.Linear(d_model, d_model * 4)
            self.ffn_down = nn.Linear(d_model * 4, d_model)

        def forward(self, x):
            # 简化的注意力计算（仅用于演示）
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            attn_out = self.o_proj(v)  # 简化
            ffn_out = self.ffn_down(torch.relu(self.ffn_up(attn_out)))
            return ffn_out + x

    model = SimpleTransformerLayer(d_model=256)

    # 统计原始参数
    total_before = sum(p.numel() for p in model.parameters())
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"原始模型 - 总参数: {total_before:,}, 可训练: {trainable_before:,}")

    # 应用 LoRA (只对 Q, K, V, O 投影)
    apply_lora_to_model(
        model,
        target_modules={"q_proj", "k_proj", "v_proj", "o_proj"},
        rank=8,
        alpha=16.0,
    )

    total_after = sum(p.numel() for p in model.parameters())
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = get_lora_parameters(model)
    lora_param_count = sum(p.numel() for p in lora_params)

    print(f"LoRA 模型 - 总参数: {total_after:,}, 可训练: {trainable_after:,}")
    print(f"LoRA 参数: {lora_param_count:,}")
    print(f"可训练参数比例: {trainable_after / total_after * 100:.2f}%")

    # 测试前向传播
    x4 = torch.randn(2, 16, 256)
    out4 = model(x4)
    print(f"模型前向传播: 输入 {x4.shape} -> 输出 {out4.shape}")

    # === 5. 参数节省分析 ===
    print("\n--- 5. 参数节省分析 ---")
    print(f"{'模型':>12} {'d':>6} {'rank':>6} {'全参数':>12} {'LoRA参数':>12} {'比例':>8}")
    print("-" * 60)
    for d, model_name in [(4096, "Llama-7B"), (5120, "Llama-13B"), (8192, "Llama-70B")]:
        for r in [8, 16, 64]:
            full_params = d * d
            lora_params_count = 2 * d * r
            ratio = lora_params_count / full_params * 100
            print(f"{model_name:>12} {d:>6} {r:>6} {full_params:>12,} {lora_params_count:>12,} {ratio:>7.2f}%")
