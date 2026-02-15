"""
分布式训练工具函数

本模块提供分布式训练中常用的工具函数:
    - 显存估算与分析
    - 训练时间估算
    - 并行策略推荐
    - 3D 并行配置生成
    - 通信量计算

这些工具函数帮助用户在开始训练前评估资源需求，
选择最合适的并行策略。

作者: LLM Tutorial
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class ModelSpec:
    """
    模型规格

    Attributes:
        name: 模型名称
        num_params_billion: 参数量（十亿）
        num_layers: Transformer 层数
        hidden_dim: 隐藏维度
        num_heads: 注意力头数
        vocab_size: 词汇表大小
        max_seq_len: 最大序列长度
    """
    name: str
    num_params_billion: float
    num_layers: int
    hidden_dim: int
    num_heads: int
    vocab_size: int = 32000
    max_seq_len: int = 4096


@dataclass
class GPUSpec:
    """
    GPU 规格

    Attributes:
        name: GPU 型号
        memory_gb: 显存大小 (GB)
        bf16_tflops: BF16 算力 (TFLOPS)
        fp8_tflops: FP8 算力 (TFLOPS)
        nvlink_bandwidth_gbps: NVLink 带宽 (GB/s)
        pcie_bandwidth_gbps: PCIe 带宽 (GB/s)
    """
    name: str
    memory_gb: float
    bf16_tflops: float
    fp8_tflops: float = 0.0
    nvlink_bandwidth_gbps: float = 600.0
    pcie_bandwidth_gbps: float = 64.0


# 常见 GPU 规格
GPU_SPECS = {
    "A100_40GB": GPUSpec("A100 40GB", 40, 312, 0, 600, 64),
    "A100_80GB": GPUSpec("A100 80GB", 80, 312, 0, 600, 64),
    "H100_SXM": GPUSpec("H100 SXM", 80, 990, 1979, 900, 128),
    "H200": GPUSpec("H200", 141, 990, 1979, 900, 128),
    "V100_32GB": GPUSpec("V100 32GB", 32, 125, 0, 300, 32),
}

# 常见模型规格
MODEL_SPECS = {
    "llama2_7b": ModelSpec("Llama 2 7B", 6.7, 32, 4096, 32),
    "llama2_13b": ModelSpec("Llama 2 13B", 13.0, 40, 5120, 40),
    "llama2_70b": ModelSpec("Llama 2 70B", 68.9, 80, 8192, 64, 32000, 4096),
    "llama3_8b": ModelSpec("Llama 3 8B", 8.0, 32, 4096, 32, 128000, 8192),
    "llama3_70b": ModelSpec("Llama 3 70B", 70.6, 80, 8192, 64, 128000, 8192),
    "gpt3_175b": ModelSpec("GPT-3 175B", 175.0, 96, 12288, 96, 50257, 2048),
    "deepseek_v3": ModelSpec("DeepSeek-V3 671B", 671.0, 61, 7168, 128, 129280, 128000),
}


def estimate_training_memory(
    model: ModelSpec,
    batch_size: int = 1,
    seq_len: Optional[int] = None,
    precision: str = "bf16",
    zero_stage: int = 0,
    num_gpus: int = 1,
    activation_checkpointing: bool = False,
) -> Dict[str, float]:
    """
    估算训练所需的显存

    Args:
        model: 模型规格
        batch_size: 每 GPU 的 batch size
        seq_len: 序列长度 (None 使用模型默认值)
        precision: 训练精度 ("fp32", "fp16", "bf16", "fp8")
        zero_stage: ZeRO 阶段 (0=DDP, 1, 2, 3)
        num_gpus: GPU 数量
        activation_checkpointing: 是否使用激活检查点

    Returns:
        各部分的显存占用 (GB)
    """
    P = model.num_params_billion * 1e9
    L = model.num_layers
    d = model.hidden_dim
    S = seq_len or model.max_seq_len
    B = batch_size
    n = num_gpus

    # 参数类型大小
    if precision == "fp32":
        param_bytes = 4
    elif precision in ("fp16", "bf16"):
        param_bytes = 2
    elif precision == "fp8":
        param_bytes = 1
    else:
        param_bytes = 2

    # 混合精度训练的显存组成
    fp16_params = 2 * P  # FP16 模型参数
    fp32_master = 4 * P  # FP32 主权重
    fp16_grads = 2 * P   # FP16 梯度
    adam_states = 8 * P   # Adam m + v (FP32)

    # ZeRO 分片
    if zero_stage == 0:
        params_mem = fp16_params + fp32_master
        grad_mem = fp16_grads
        opt_mem = adam_states
    elif zero_stage == 1:
        params_mem = fp16_params + fp32_master / n
        grad_mem = fp16_grads
        opt_mem = adam_states / n
    elif zero_stage == 2:
        params_mem = fp16_params + fp32_master / n
        grad_mem = fp16_grads / n
        opt_mem = adam_states / n
    elif zero_stage == 3:
        params_mem = (fp16_params + fp32_master) / n
        grad_mem = fp16_grads / n
        opt_mem = adam_states / n
    else:
        raise ValueError(f"不支持的 ZeRO 阶段: {zero_stage}")

    # 激活值显存
    # 粗略估算: 每层约 2 * B * S * d * 10 (包括注意力矩阵等)
    bytes_per_element = 2  # FP16
    activation_per_layer = B * S * d * 34 * bytes_per_element  # Megatron-LM 的经验值

    if activation_checkpointing:
        # 只保存 sqrt(L) 个检查点 + 当前段的 sqrt(L) 层激活
        sqrt_L = int(math.sqrt(L))
        act_mem = activation_per_layer * (sqrt_L + sqrt_L)
    else:
        act_mem = activation_per_layer * L

    # 转换为 GB
    result = {
        "参数 (GB)": params_mem / 1e9,
        "梯度 (GB)": grad_mem / 1e9,
        "优化器 (GB)": opt_mem / 1e9,
        "激活值 (GB)": act_mem / 1e9,
        "总计 (GB)": (params_mem + grad_mem + opt_mem + act_mem) / 1e9,
    }

    return result


def estimate_training_time(
    model: ModelSpec,
    num_tokens_billion: float,
    num_gpus: int,
    gpu: GPUSpec,
    mfu: float = 0.4,
) -> Dict[str, float]:
    """
    估算训练时间

    使用公式: T = 6PD / (n * GPU_FLOPS * MFU)

    Args:
        model: 模型规格
        num_tokens_billion: 训练 token 总数（十亿）
        num_gpus: GPU 数量
        gpu: GPU 规格
        mfu: Model FLOPs Utilization (典型值 0.3-0.55)

    Returns:
        训练时间估算
    """
    P = model.num_params_billion * 1e9
    D = num_tokens_billion * 1e9

    # 总 FLOPs
    total_flops = 6 * P * D

    # 集群算力
    cluster_flops = num_gpus * gpu.bf16_tflops * 1e12 * mfu

    # 训练时间 (秒)
    time_seconds = total_flops / cluster_flops

    return {
        "总 FLOPs (EFLOPS)": total_flops / 1e18,
        "集群算力 (PFLOPS)": cluster_flops / 1e15,
        "训练时间 (秒)": time_seconds,
        "训练时间 (小时)": time_seconds / 3600,
        "训练时间 (天)": time_seconds / 86400,
        "GPU 时 (千小时)": num_gpus * time_seconds / 3600 / 1000,
    }


def recommend_parallel_config(
    model: ModelSpec,
    gpu: GPUSpec,
    num_gpus: int,
    batch_size_per_gpu: int = 1,
) -> Dict[str, int]:
    """
    推荐 3D 并行配置

    基于模型大小、GPU 规格和可用 GPU 数量，
    推荐 TP/PP/DP 的组合。

    原则:
        1. TP 优先使用节点内 NVLink (通常最多 8)
        2. PP 控制在合理范围以避免高气泡率
        3. DP = 总 GPU / (TP * PP)

    Args:
        model: 模型规格
        gpu: GPU 规格
        num_gpus: 总 GPU 数量
        batch_size_per_gpu: 每 GPU 的 batch size

    Returns:
        推荐的并行配置
    """
    P_billion = model.num_params_billion

    # 估算每个 GPU 需要的最小 TP 度
    # 模型参数 (FP16) 需要 2P bytes，加上优化器约 16P
    min_mem_per_gpu = 16 * P_billion  # GB (粗略)

    # 确定 TP
    tp = 1
    if P_billion > 13:
        tp = 2
    if P_billion > 30:
        tp = 4
    if P_billion > 65:
        tp = 8

    # TP 不能超过节点内 GPU 数（假设 8 GPU/节点）
    gpus_per_node = 8
    tp = min(tp, gpus_per_node)

    # 确定 PP
    # 使用 ZeRO-1 后每 GPU 的参数显存
    zero1_mem = (4 * P_billion + 12 * P_billion / (num_gpus / tp))
    pp = 1
    if zero1_mem / tp > gpu.memory_gb * 0.7:  # 留 30% 给激活值
        pp = max(2, int(math.ceil(zero1_mem / tp / (gpu.memory_gb * 0.7))))

    # PP 不能太大（气泡率约束）
    max_pp = min(16, num_gpus // tp)
    pp = min(pp, max_pp)

    # DP
    dp = num_gpus // (tp * pp)
    if dp < 1:
        dp = 1
        pp = num_gpus // tp

    # ZeRO 建议
    zero_stage = 0
    single_gpu_mem = estimate_training_memory(
        model, batch_size_per_gpu, zero_stage=0, num_gpus=num_gpus
    )["总计 (GB)"]

    if single_gpu_mem / tp > gpu.memory_gb:
        zero_stage = 1
    if single_gpu_mem / tp > gpu.memory_gb * 1.5:
        zero_stage = 2
    if P_billion > 100:
        zero_stage = 3

    return {
        "模型": model.name,
        "GPU": gpu.name,
        "总 GPU 数": num_gpus,
        "TP (张量并行)": tp,
        "PP (流水线并行)": pp,
        "DP (数据并行)": dp,
        "ZeRO 阶段": zero_stage,
        "每 DP 组有效 batch": batch_size_per_gpu * dp,
        "验证: TP*PP*DP": tp * pp * dp,
    }


def compute_communication_volume(
    model: ModelSpec,
    batch_size: int,
    seq_len: Optional[int] = None,
    tp: int = 1,
    pp: int = 1,
    dp: int = 1,
) -> Dict[str, float]:
    """
    计算各种并行策略的通信量

    Args:
        model: 模型规格
        batch_size: 每 GPU 的 batch size
        seq_len: 序列长度
        tp: 张量并行度
        pp: 流水线并行度
        dp: 数据并行度

    Returns:
        各类通信的数据量 (GB)
    """
    P = model.num_params_billion * 1e9
    d = model.hidden_dim
    L = model.num_layers
    S = seq_len or model.max_seq_len
    B = batch_size
    dtype_size = 2  # FP16

    comm = {}

    # DP 通信: All-Reduce 梯度
    if dp > 1:
        dp_comm = 2 * (dp - 1) / dp * P * dtype_size  # Ring All-Reduce
        comm["DP All-Reduce (每步)"] = dp_comm / 1e9

    # TP 通信: 每层 2 次 All-Reduce (前向 + 反向共 4 次)
    if tp > 1:
        per_allreduce = B * S * d * dtype_size
        tp_comm = 4 * L * 2 * (tp - 1) / tp * per_allreduce  # 前向2+反向2, 每次一个All-Reduce
        comm["TP All-Reduce (每步)"] = tp_comm / 1e9

    # PP 通信: 点对点传输激活值
    if pp > 1:
        # 每个 micro-batch 在每个 stage 边界传输 B * S * d
        per_activation = B * S * d * dtype_size
        # 前向 + 反向各一次
        pp_comm = 2 * (pp - 1) * per_activation
        comm["PP 点对点 (每步)"] = pp_comm / 1e9

    # 总通信量
    comm["总通信量 (每步)"] = sum(comm.values())

    return comm


def print_model_analysis(model_name: str, gpu_name: str = "A100_80GB", num_gpus: int = 8):
    """
    打印模型的完整分析报告

    Args:
        model_name: 模型名称 (MODEL_SPECS 中的键)
        gpu_name: GPU 型号 (GPU_SPECS 中的键)
        num_gpus: GPU 数量
    """
    model = MODEL_SPECS[model_name]
    gpu = GPU_SPECS[gpu_name]

    print("=" * 70)
    print(f"模型分析报告: {model.name}")
    print("=" * 70)

    # 基本信息
    print(f"\n--- 模型规格 ---")
    print(f"  参数量: {model.num_params_billion:.1f}B")
    print(f"  层数: {model.num_layers}")
    print(f"  隐藏维度: {model.hidden_dim}")
    print(f"  注意力头数: {model.num_heads}")
    print(f"  词汇表大小: {model.vocab_size:,}")
    print(f"  最大序列长度: {model.max_seq_len:,}")

    # 显存分析
    print(f"\n--- 显存分析 (单 GPU, {gpu.name}) ---")
    for stage in [0, 1, 2, 3]:
        mem = estimate_training_memory(model, batch_size=1, zero_stage=stage, num_gpus=num_gpus)
        fits = "OK" if mem["总计 (GB)"] <= gpu.memory_gb else "OOM"
        print(f"  ZeRO-{stage}: {mem['总计 (GB)']:>8.1f} GB [{fits}]")

    # 3D 并行推荐
    print(f"\n--- 推荐并行配置 ({num_gpus} x {gpu.name}) ---")
    config = recommend_parallel_config(model, gpu, num_gpus)
    for key, val in config.items():
        print(f"  {key}: {val}")

    # 训练时间估算
    print(f"\n--- 训练时间估算 ---")
    for tokens_b in [100, 500, 1000, 2000]:
        time_est = estimate_training_time(model, tokens_b, num_gpus, gpu)
        print(f"  {tokens_b}B tokens: {time_est['训练时间 (天)']:.1f} 天 "
              f"({time_est['GPU 时 (千小时)']:.1f}K GPU-hours)")


def compare_models():
    """对比不同模型的资源需求"""
    print("=" * 90)
    print("模型资源需求对比")
    print("=" * 90)

    gpu = GPU_SPECS["A100_80GB"]

    print(f"\n{'模型':>20} {'参数量':>10} {'DDP 显存':>12} {'ZeRO-3 (8GPU)':>15} "
          f"{'推荐 TP':>8} {'推荐 PP':>8}")
    print("-" * 80)

    for name, model in MODEL_SPECS.items():
        ddp_mem = estimate_training_memory(model, zero_stage=0, num_gpus=1)
        z3_mem = estimate_training_memory(model, zero_stage=3, num_gpus=8)
        config = recommend_parallel_config(model, gpu, 64)

        print(
            f"{model.name:>20} {model.num_params_billion:>8.1f}B "
            f"{ddp_mem['总计 (GB)']:>10.1f}GB "
            f"{z3_mem['总计 (GB)']:>13.1f}GB "
            f"{config['TP (张量并行)']:>8} "
            f"{config['PP (流水线并行)']:>8}"
        )


if __name__ == "__main__":
    # 对比所有模型
    compare_models()

    # 详细分析 Llama 2 70B
    print()
    print_model_analysis("llama2_70b", "A100_80GB", num_gpus=64)

    # 详细分析 Llama 3 8B
    print()
    print_model_analysis("llama3_8b", "A100_80GB", num_gpus=8)
