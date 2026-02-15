"""
分布式训练封装

知识依赖:
- 模块 9（分布式训练）: DDP、FSDP、DeepSpeed ZeRO

参考实现:
- code/distributed/data_parallel.py
- code/distributed/mixed_precision.py

分布式策略选择:

| 策略 | 适用场景 | 显存效率 | 通信开销 |
|------|---------|---------|---------|
| DDP | 模型能放进单卡 | 低 | 低 |
| FSDP | 模型放不进单卡 | 高 | 中 |
| DeepSpeed ZeRO-2 | 通用 | 高 | 中 |

Version A (300M): DDP 即可（单卡放得下）
Version B (1B): 推荐 FSDP 或 ZeRO-2

FSDP 核心思想:
    将模型参数、梯度、优化器状态分片到各 GPU。
    前向/反向传播时按需聚合参数（all-gather），
    计算完立即释放（reduce-scatter）。
    显存占用 ≈ 总量 / n_gpus。

初始化流程:
    1. torch.distributed.init_process_group("nccl")
    2. 设置 CUDA device
    3. 用 FSDP 或 DDP 包装模型
    4. 创建分布式 DataLoader (DistributedSampler)
"""

import torch
import torch.nn as nn
from typing import Optional


def setup_distributed(backend: str = "nccl"):
    """
    初始化分布式训练环境

    Args:
        backend: 通信后端 ("nccl" for GPU, "gloo" for CPU)

    实现步骤:
        1. torch.distributed.init_process_group(backend)
        2. local_rank = int(os.environ["LOCAL_RANK"])
        3. torch.cuda.set_device(local_rank)
        4. return local_rank, world_size

    启动方式:
        torchrun --nproc_per_node=4 scripts/pretrain.py
    """
    raise NotImplementedError(
        "TODO: 初始化分布式环境\n"
        "参考: 模块 9 (分布式训练) 的初始化章节\n"
        "参考实现: code/distributed/data_parallel.py"
    )


def cleanup_distributed():
    """清理分布式环境"""
    raise NotImplementedError(
        "TODO: torch.distributed.destroy_process_group()"
    )


def wrap_model_ddp(model: nn.Module, device_id: int) -> nn.Module:
    """
    用 DDP 包装模型（Version A 使用）

    Args:
        model: 原始模型
        device_id: 当前 GPU ID

    Returns:
        DDP 包装后的模型

    实现:
        model = model.to(device_id)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device_id])
    """
    raise NotImplementedError(
        "TODO: DDP 模型包装\n"
        "参考: 模块 9 的数据并行章节"
    )


def wrap_model_fsdp(model: nn.Module, mixed_precision: str = "bf16") -> nn.Module:
    """
    用 FSDP 包装模型（Version B 使用）

    Args:
        model: 原始模型
        mixed_precision: 混合精度类型

    Returns:
        FSDP 包装后的模型

    实现提示:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision

        混合精度策略:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )

        wrapping policy: 按 TransformerBlock 粒度分片
    """
    raise NotImplementedError(
        "TODO: FSDP 模型包装\n"
        "参考: 模块 9 的 FSDP 章节\n"
        "参考实现: code/distributed/data_parallel.py"
    )
