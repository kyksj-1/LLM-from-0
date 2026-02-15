"""
模型 FLOPs 与训练成本估算工具

本模块实现了 Transformer 模型的计算量 (FLOPs) 估算和训练成本计算。

数学基础:
- 前向传播: ~2P FLOPs/token (P = 参数量)
- 反向传播: ~4P FLOPs/token
- 总计: C ≈ 6PD (P = 参数量, D = 训练 tokens)
- GPU-hours = C / (GPU_FLOPS * 3600 * MFU)
- 成本($) = GPU-hours * 单价($/GPU-hour)

参考:
- Kaplan et al. (2020). Scaling Laws for Neural Language Models.
- PaLM Technical Report. (FLOPs 估算方法)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, List


@dataclass
class GPUSpec:
    """
    GPU 规格

    Attributes:
        name: GPU 型号名称
        bf16_tflops: BF16 峰值算力 (TFLOPS)
        memory_gb: 显存大小 (GB)
        price_per_hour: 单 GPU 小时价格 (美元)
        typical_mfu: 典型的 MFU (Model FLOPs Utilization)
    """
    name: str
    bf16_tflops: float
    memory_gb: float
    price_per_hour: float
    typical_mfu: float


# 常见 GPU 规格库
GPU_SPECS = {
    'A100_40GB': GPUSpec('A100 40GB', 312, 40, 1.5, 0.45),
    'A100_80GB': GPUSpec('A100 80GB', 312, 80, 2.0, 0.45),
    'H100_80GB': GPUSpec('H100 80GB', 990, 80, 3.0, 0.40),
    'H200_141GB': GPUSpec('H200 141GB', 990, 141, 4.0, 0.45),
    'B200_192GB': GPUSpec('B200 192GB', 2250, 192, 6.0, 0.40),
}


@dataclass
class TransformerConfig:
    """
    Transformer 模型配置

    Attributes:
        n_layers: 层数
        d_model: 隐藏维度
        n_heads: 注意力头数
        d_ff: FFN 中间维度 (如果为 None, 默认为 4 * d_model)
        vocab_size: 词汇表大小
        seq_length: 训练序列长度
        is_moe: 是否为 MoE 模型
        n_experts: MoE 专家数量 (仅当 is_moe=True)
        top_k: MoE 激活专家数 (仅当 is_moe=True)
        use_glu: 是否使用 GLU 变体的 FFN (如 SwiGLU)
    """
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: Optional[int] = None
    vocab_size: int = 32000
    seq_length: int = 4096
    is_moe: bool = False
    n_experts: int = 1
    top_k: int = 1
    use_glu: bool = True

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

    @staticmethod
    def llama3_8b() -> 'TransformerConfig':
        """Llama 3 8B 配置"""
        return TransformerConfig(
            n_layers=32, d_model=4096, n_heads=32,
            d_ff=14336, vocab_size=128256, seq_length=8192,
            use_glu=True,
        )

    @staticmethod
    def llama3_70b() -> 'TransformerConfig':
        """Llama 3 70B 配置"""
        return TransformerConfig(
            n_layers=80, d_model=8192, n_heads=64,
            d_ff=28672, vocab_size=128256, seq_length=8192,
            use_glu=True,
        )

    @staticmethod
    def deepseek_v3() -> 'TransformerConfig':
        """DeepSeek-V3 近似配置 (MoE)"""
        return TransformerConfig(
            n_layers=61, d_model=7168, n_heads=128,
            d_ff=18432, vocab_size=129280, seq_length=4096,
            is_moe=True, n_experts=256, top_k=8,
            use_glu=True,
        )


class FLOPsCalculator:
    """
    Transformer 模型 FLOPs 计算器

    提供详细的逐层 FLOPs 分析和训练成本估算。
    """

    def __init__(self, config: TransformerConfig):
        """
        初始化计算器

        Args:
            config: Transformer 模型配置
        """
        self.config = config

    def count_parameters(self) -> Dict[str, int]:
        """
        计算模型各部分的参数量

        Returns:
            各部分参数量的字典
        """
        c = self.config
        d = c.d_model
        d_ff = c.d_ff
        V = c.vocab_size

        # 注意力参数: QKV 投影 + 输出投影
        # Q: d -> d, K: d -> d_kv, V: d -> d_kv, O: d -> d
        attn_per_layer = 4 * d * d  # 简化: 假设 d_k * n_heads = d

        # FFN 参数
        if c.use_glu:
            # SwiGLU: gate(d -> d_ff) + up(d -> d_ff) + down(d_ff -> d)
            ffn_per_layer = 3 * d * d_ff
        else:
            # 标准 FFN: up(d -> d_ff) + down(d_ff -> d)
            ffn_per_layer = 2 * d * d_ff

        # MoE 的 FFN 参数
        if c.is_moe:
            # 路由器参数
            router_per_layer = d * c.n_experts
            # 每个专家有自己的 FFN
            ffn_total = ffn_per_layer * c.n_experts + router_per_layer
            # 激活参数 (每个 token 实际使用的)
            ffn_active = ffn_per_layer * c.top_k + router_per_layer
        else:
            ffn_total = ffn_per_layer
            ffn_active = ffn_per_layer

        # LayerNorm: 2d per norm, 2 norms per layer
        norm_per_layer = 4 * d

        # 嵌入层
        embedding = V * d

        # 汇总
        total_per_layer = attn_per_layer + ffn_total + norm_per_layer
        active_per_layer = attn_per_layer + ffn_active + norm_per_layer

        total = c.n_layers * total_per_layer + embedding
        # LM head 通常与嵌入共享权重, 不额外计算
        active = c.n_layers * active_per_layer + embedding

        return {
            'attention_per_layer': attn_per_layer,
            'ffn_per_layer_total': ffn_total // c.n_layers if c.is_moe else ffn_per_layer,
            'ffn_per_layer_active': ffn_active // c.n_layers if c.is_moe else ffn_per_layer,
            'norm_per_layer': norm_per_layer,
            'embedding': embedding,
            'total_per_layer': total_per_layer,
            'active_per_layer': active_per_layer,
            'total': total,
            'active': active,
            'n_layers': c.n_layers,
        }

    def flops_per_token_forward(self) -> Dict[str, float]:
        """
        计算前向传播每个 token 的 FLOPs (详细分解)

        Returns:
            各部分 FLOPs 的字典
        """
        c = self.config
        d = c.d_model
        d_ff = c.d_ff
        n = c.seq_length
        V = c.vocab_size

        # 注意力: QKV 投影 + 注意力计算 + 输出投影
        # QKV: 3 * (2 * d * d) = 6d^2 per layer (乘法 + 加法)
        qkv_proj = 6 * d * d
        # 注意力分数: Q*K^T = 2*n*d_k*n_heads = 2*n*d
        attn_score = 2 * n * d
        # Attention * V: 2*n*d
        attn_output = 2 * n * d
        # 输出投影: 2*d*d
        out_proj = 2 * d * d

        attn_per_layer = qkv_proj + attn_score + attn_output + out_proj

        # FFN
        if c.use_glu:
            # SwiGLU: gate(2*d*d_ff) + up(2*d*d_ff) + element_wise(d_ff) + down(2*d_ff*d)
            ffn_per_layer = 2 * d * d_ff + 2 * d * d_ff + d_ff + 2 * d_ff * d
        else:
            # 标准: up(2*d*d_ff) + down(2*d_ff*d)
            ffn_per_layer = 2 * d * d_ff + 2 * d_ff * d

        # MoE: 只有 top_k 个专家被激活
        if c.is_moe:
            ffn_active = ffn_per_layer * c.top_k
            # 路由器: 2*d*n_experts
            router_flops = 2 * d * c.n_experts
            ffn_active += router_flops
        else:
            ffn_active = ffn_per_layer

        # 嵌入 + LM head: 2*V*d (嵌入查找 + 输出投影)
        embedding_flops = 2 * V * d

        # 每层总 FLOPs
        per_layer = attn_per_layer + ffn_active

        # 全模型 FLOPs/token
        total = c.n_layers * per_layer + embedding_flops

        return {
            'qkv_projection': qkv_proj,
            'attention_score': attn_score,
            'attention_output': attn_output,
            'output_projection': out_proj,
            'attention_total_per_layer': attn_per_layer,
            'ffn_per_layer': ffn_active,
            'per_layer_total': per_layer,
            'embedding': embedding_flops,
            'total_forward': total,
            'approx_6x_check': total / self.count_parameters()['active'],
        }

    def estimate_training_flops(
        self, n_tokens: float, include_backward: bool = True
    ) -> float:
        """
        估算训练总 FLOPs

        Args:
            n_tokens: 训练 token 总数
            include_backward: 是否包含反向传播 (默认 True)

        Returns:
            总 FLOPs
        """
        forward_per_token = self.flops_per_token_forward()['total_forward']
        multiplier = 3.0 if include_backward else 1.0
        # 前向 + 反向 ≈ 3 * 前向 (反向约为前向的 2 倍)
        return forward_per_token * multiplier * n_tokens

    def estimate_training_cost(
        self,
        n_tokens: float,
        gpu: Optional[GPUSpec] = None,
        n_gpus: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        估算训练成本

        Args:
            n_tokens: 训练 token 总数
            gpu: GPU 规格 (默认 H100 80GB)
            n_gpus: GPU 数量 (如果指定, 计算训练时间)

        Returns:
            成本估算结果
        """
        if gpu is None:
            gpu = GPU_SPECS['H100_80GB']

        total_flops = self.estimate_training_flops(n_tokens)

        # 有效算力 = 峰值 * MFU
        effective_tflops = gpu.bf16_tflops * gpu.typical_mfu
        effective_flops_per_sec = effective_tflops * 1e12

        # GPU-hours
        gpu_hours = total_flops / (effective_flops_per_sec * 3600)

        # 美元成本
        cost_usd = gpu_hours * gpu.price_per_hour

        result = {
            'total_flops': total_flops,
            'gpu_name': gpu.name,
            'gpu_peak_tflops': gpu.bf16_tflops,
            'mfu': gpu.typical_mfu,
            'effective_tflops': effective_tflops,
            'gpu_hours': gpu_hours,
            'cost_usd': cost_usd,
        }

        if n_gpus is not None:
            # 计算墙钟时间 (假设线性扩展, 实际会有通信开销)
            wall_hours = gpu_hours / n_gpus
            wall_days = wall_hours / 24
            result['n_gpus'] = n_gpus
            result['wall_hours'] = wall_hours
            result['wall_days'] = wall_days

        return result

    def c6pd_approximation(self, n_tokens: float) -> Dict[str, float]:
        """
        使用 C ≈ 6PD 近似公式估算 FLOPs

        Args:
            n_tokens: 训练 token 总数

        Returns:
            近似估算结果和与详细估算的对比
        """
        params = self.count_parameters()
        P_active = params['active']
        P_total = params['total']

        c_active = 6 * P_active * n_tokens
        c_total = 6 * P_total * n_tokens
        c_detailed = self.estimate_training_flops(n_tokens)

        return {
            'P_active': P_active,
            'P_total': P_total,
            'C_6PD_active': c_active,
            'C_6PD_total': c_total,
            'C_detailed': c_detailed,
            'ratio_active': c_detailed / c_active if c_active > 0 else float('inf'),
            'ratio_total': c_detailed / c_total if c_total > 0 else float('inf'),
        }


def format_flops(flops: float) -> str:
    """
    格式化 FLOPs 为可读字符串

    Args:
        flops: FLOPs 数量

    Returns:
        格式化后的字符串
    """
    if flops >= 1e24:
        return f"{flops/1e24:.2f} YFLOPs"
    elif flops >= 1e21:
        return f"{flops/1e21:.2f} ZFLOPs"
    elif flops >= 1e18:
        return f"{flops/1e18:.2f} EFLOPs"
    elif flops >= 1e15:
        return f"{flops/1e15:.2f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    else:
        return f"{flops:.2e} FLOPs"


def format_params(n: float) -> str:
    """格式化参数量"""
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


if __name__ == '__main__':
    print("=" * 70)
    print("模型 FLOPs 与训练成本估算工具 - 演示")
    print("=" * 70)

    # === 1. Llama 3 8B 分析 ===
    print("\n--- 1. Llama 3 8B 参数量与 FLOPs 分析 ---")
    config_8b = TransformerConfig.llama3_8b()
    calc_8b = FLOPsCalculator(config_8b)

    params_8b = calc_8b.count_parameters()
    print(f"总参数量: {format_params(params_8b['total'])}")
    print(f"  注意力/层: {format_params(params_8b['attention_per_layer'])}")
    print(f"  FFN/层: {format_params(params_8b['ffn_per_layer_total'])}")
    print(f"  嵌入层: {format_params(params_8b['embedding'])}")

    flops_8b = calc_8b.flops_per_token_forward()
    print(f"\n前向 FLOPs/token: {flops_8b['total_forward']:.2e}")
    print(f"  注意力/层: {flops_8b['attention_total_per_layer']:.2e}")
    print(f"  FFN/层: {flops_8b['ffn_per_layer']:.2e}")
    print(f"  FLOPs/param 比值: {flops_8b['approx_6x_check']:.2f}x (理论 ≈ 2x)")

    # === 2. 训练成本估算 ===
    print("\n--- 2. Llama 3 8B 训练成本估算 (15T tokens) ---")
    cost_8b = calc_8b.estimate_training_cost(
        n_tokens=15e12,
        gpu=GPU_SPECS['H100_80GB'],
        n_gpus=2048,
    )
    print(f"总 FLOPs: {format_flops(cost_8b['total_flops'])}")
    print(f"GPU: {cost_8b['gpu_name']} (峰值 {cost_8b['gpu_peak_tflops']} TFLOPS, "
          f"MFU {cost_8b['mfu']:.0%})")
    print(f"GPU-hours: {cost_8b['gpu_hours']:.0f}")
    print(f"估算成本: ${cost_8b['cost_usd']:,.0f}")
    if 'wall_days' in cost_8b:
        print(f"使用 {cost_8b['n_gpus']} GPU, 墙钟时间: {cost_8b['wall_days']:.1f} 天")

    # === 3. C ≈ 6PD 近似检验 ===
    print("\n--- 3. C ≈ 6PD 近似精度检验 ---")
    approx_8b = calc_8b.c6pd_approximation(15e12)
    print(f"P(active) = {format_params(approx_8b['P_active'])}")
    print(f"C (6PD, active) = {format_flops(approx_8b['C_6PD_active'])}")
    print(f"C (detailed)    = {format_flops(approx_8b['C_detailed'])}")
    print(f"详细/6PD 比值: {approx_8b['ratio_active']:.3f}")

    # === 4. Llama 3 70B 分析 ===
    print("\n--- 4. Llama 3 70B 训练成本估算 ---")
    config_70b = TransformerConfig.llama3_70b()
    calc_70b = FLOPsCalculator(config_70b)

    params_70b = calc_70b.count_parameters()
    print(f"总参数量: {format_params(params_70b['total'])}")

    cost_70b = calc_70b.estimate_training_cost(
        n_tokens=15e12,
        gpu=GPU_SPECS['H100_80GB'],
        n_gpus=16384,
    )
    print(f"总 FLOPs: {format_flops(cost_70b['total_flops'])}")
    print(f"GPU-hours: {cost_70b['gpu_hours']:.0f}")
    print(f"估算成本: ${cost_70b['cost_usd']:,.0f}")
    if 'wall_days' in cost_70b:
        print(f"使用 {cost_70b['n_gpus']} GPU, 墙钟时间: {cost_70b['wall_days']:.1f} 天")

    # === 5. DeepSeek-V3 (MoE) 分析 ===
    print("\n--- 5. DeepSeek-V3 (MoE) 参数量分析 ---")
    config_v3 = TransformerConfig.deepseek_v3()
    calc_v3 = FLOPsCalculator(config_v3)

    params_v3 = calc_v3.count_parameters()
    print(f"总参数量: {format_params(params_v3['total'])}")
    print(f"激活参数量: {format_params(params_v3['active'])}")
    print(f"激活比例: {params_v3['active']/params_v3['total']:.1%}")

    # === 6. GPU 对比 ===
    print("\n--- 6. 不同 GPU 训练 70B 模型的成本对比 ---")
    print(f"{'GPU':>15s} | {'峰值(TFLOPS)':>12s} | {'MFU':>5s} | "
          f"{'GPU-hours':>12s} | {'成本($)':>12s}")
    print("-" * 70)
    for gpu_name, gpu_spec in GPU_SPECS.items():
        cost = calc_70b.estimate_training_cost(15e12, gpu=gpu_spec)
        print(f"{gpu_spec.name:>15s} | {gpu_spec.bf16_tflops:>12.0f} | "
              f"{gpu_spec.typical_mfu:>5.0%} | "
              f"{cost['gpu_hours']:>12,.0f} | ${cost['cost_usd']:>11,.0f}")
