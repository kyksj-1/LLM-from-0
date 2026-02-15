# 从 Version A (300M) 到 Version B (1B) 扩展指南

> 本文档帮助你在成功训练 300M 模型后，将其扩展到 1B 参数规模。重点讲解需要改变什么、为什么要改变、以及如何在多 GPU 上高效训练。

---

## 目录

- [1. Version A 与 Version B 对比](#1-version-a-与-version-b-对比)
- [2. 硬件需求分析](#2-硬件需求分析)
- [3. 分布式训练策略](#3-分布式训练策略)
- [4. FSDP 实战指南](#4-fsdp-实战指南)
- [5. 超参数调整](#5-超参数调整)
- [6. 多机训练](#6-多机训练)
- [7. 扩展检查清单](#7-扩展检查清单)

---

## 1. Version A 与 Version B 对比

### 1.1 架构参数

| 参数 | Version A (300M) | Version B (1B) | 变化 |
|------|-----------------|----------------|------|
| n_layers | 24 | 32 | +33% |
| d_model | 1024 | 2048 | ×2 |
| n_heads | 16 | 32 | ×2 |
| n_kv_heads | 4 | 8 | ×2 |
| d_ff | 2730 | 5460 | ×2 |
| head_dim | 64 | 64 | 不变 |
| vocab_size | 32,000 | 64,000 | ×2 |
| max_seq_len | 2048 | 4096 | ×2 |
| **总参数量** | **~300M** | **~1B** | **~3.3×** |

### 1.2 训练配置

| 配置 | Version A | Version B | 说明 |
|------|-----------|-----------|------|
| GPU | 1× 24GB | 4-8× 24GB | 需要分布式 |
| 训练数据 | 3-6B tokens | 20-50B tokens | Chinchilla: ~20× 参数量 |
| Batch size (tokens) | 65K/step | 262K/step | 更大的 batch 更稳定 |
| 训练步数 | ~92K | ~76K-190K | 取决于数据量 |
| 训练时间 | 数小时-1天 | 数天 | 取决于 GPU 数量 |
| 分词器 | 32K vocab | 64K vocab | 需要重新训练 |

### 1.3 显存分析

**单卡显存估算 (BF16)**:

```
                        Version A       Version B
模型参数 (BF16):        600 MB          2.0 GB
优化器状态 (FP32):      2.4 GB          8.0 GB
梯度 (BF16):            600 MB          2.0 GB
激活值 (估算):          ~4 GB           ~16 GB
───────────────────────────────────────────────
总计:                   ~7.6 GB         ~28 GB ← 超过 24GB!
```

**结论**: Version B 无法在单卡 24GB 上训练，必须使用分布式策略。

---

## 2. 硬件需求分析

### 2.1 最低配置

| 方案 | GPU | 显存/卡 | 预计训练时间 | 成本估算 |
|------|-----|---------|------------|---------|
| 方案 A | 4× RTX 3090/4090 | 24 GB | 3-5 天 | 自有硬件 |
| 方案 B | 4× A100 40GB | 40 GB | 1-2 天 | ~$8/hr 云服务 |
| 方案 C | 8× A100 80GB | 80 GB | 12-24 小时 | ~$16/hr 云服务 |

### 2.2 GPU 互联要求

GPU 之间的通信带宽对分布式训练至关重要：

| 互联方式 | 带宽 | 延迟 | 适用场景 |
|---------|------|------|---------|
| NVLink | 600 GB/s | 低 | 同节点内 GPU |
| PCIe Gen4 | 32 GB/s | 中 | 消费级显卡 |
| 以太网 (100GbE) | 12.5 GB/s | 高 | 跨节点 |
| InfiniBand | 200-400 GB/s | 低 | 数据中心 |

**消费级 GPU（PCIe）**: 通信较慢，应选择通信量小的策略 (FSDP FULL_SHARD)。
**数据中心 GPU（NVLink）**: 通信快，可以更灵活选择策略。

---

## 3. 分布式训练策略

### 3.1 策略对比

| 策略 | 原理 | 显存节省 | 通信量 | 适用场景 |
|------|------|---------|-------|---------|
| DDP | 每卡完整模型，梯度同步 | 无 | 梯度 AllReduce | 模型能放入单卡 |
| FSDP | 参数/梯度/优化器分片 | 最大 | 参数 AllGather + 梯度 ReduceScatter | **推荐用于 1B** |
| DeepSpeed ZeRO-2 | 优化器+梯度分片 | 中等 | 梯度 AllReduce | 中等规模 |
| DeepSpeed ZeRO-3 | 全部分片 | 最大 | 类似 FSDP | 大规模训练 |

**对于 1B 模型，推荐使用 PyTorch FSDP**（原生支持，不需要额外依赖）。

### 3.2 为什么选 FSDP？

```
DDP (不分片):
  GPU 0: [完整模型] + [完整优化器] + [完整梯度]     → 超 24GB
  GPU 1: [完整模型] + [完整优化器] + [完整梯度]
  GPU 2: [完整模型] + [完整优化器] + [完整梯度]
  GPU 3: [完整模型] + [完整优化器] + [完整梯度]

FSDP (全分片):
  GPU 0: [1/4 模型] + [1/4 优化器] + [1/4 梯度]    → ~7 GB
  GPU 1: [1/4 模型] + [1/4 优化器] + [1/4 梯度]
  GPU 2: [1/4 模型] + [1/4 优化器] + [1/4 梯度]
  GPU 3: [1/4 模型] + [1/4 优化器] + [1/4 梯度]
  (需要时通过 AllGather 临时收集完整参数)
```

---

## 4. FSDP 实战指南

> **对应模块**: [模块 9 - 分布式训练](../../09_distributed/README.md)

### 4.1 环境初始化

```python
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def setup_distributed():
    """初始化分布式环境"""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

启动方式:
```bash
# 单机 4 卡
torchrun --nproc_per_node=4 scripts/pretrain.py --config configs/model_1b.yaml

# 多机（2 机 × 4 卡 = 8 卡）
torchrun --nnodes=2 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=master:29500 \
    scripts/pretrain.py --config configs/model_1b.yaml
```

### 4.2 FSDP 包装策略

FSDP 需要指定在哪个粒度进行分片。对于 Transformer 模型，通常按 TransformerBlock 分片：

```python
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial

# 按 TransformerBlock 粒度分片
wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={TransformerBlock},  # 你的 Block 类
)

model = FSDP(
    model,
    auto_wrap_policy=wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 完全分片
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    ),
    device_id=local_rank,
)
```

### 4.3 Sharding Strategy 选择

| 策略 | 显存节省 | 通信量 | 何时用 |
|------|---------|-------|--------|
| `FULL_SHARD` | 最大 | 最大 | 显存紧张（24GB 卡） |
| `SHARD_GRAD_OP` | 中等 | 中等 | 显存充裕（40GB+） |
| `NO_SHARD` | 无（= DDP） | 最小 | 模型能放入单卡 |

**建议**: 4×24GB 用 `FULL_SHARD`；4×A100-80GB 可以用 `SHARD_GRAD_OP`（通信更少更快）。

### 4.4 Checkpoint 保存与加载

FSDP 的 checkpoint 需要特殊处理，因为参数是分片的：

```python
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

# 保存: 收集完整 state_dict 到 rank 0
with FSDP.state_dict_type(
    model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
):
    state_dict = model.state_dict()
    if dist.get_rank() == 0:
        torch.save(state_dict, "checkpoint.pt")

# 加载: rank 0 加载后广播
if dist.get_rank() == 0:
    state_dict = torch.load("checkpoint.pt")
else:
    state_dict = None
# FSDP 会自动处理分发
model.load_state_dict(state_dict)
```

### 4.5 数据并行注意事项

分布式训练中，每个 GPU 处理不同的数据：

```python
from torch.utils.data.distributed import DistributedSampler

sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42,  # 保证可复现
)

dataloader = DataLoader(
    dataset,
    batch_size=micro_batch_size,  # 每卡的 batch size
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
)

# 每个 epoch 开始时更新 sampler
sampler.set_epoch(epoch)
```

---

## 5. 超参数调整

### 5.1 学习率缩放

从 Version A 扩展到 Version B 时，如果 batch size 增大了，学习率也需要相应调整。

**线性缩放规则**:
$$\text{lr}_{\text{new}} = \text{lr}_{\text{base}} \times \frac{\text{batch\_size}_{\text{new}}}{\text{batch\_size}_{\text{base}}}$$

但实践中，线性缩放在 batch size 变化很大时不够精确。更常用的是**平方根缩放**：

$$\text{lr}_{\text{new}} = \text{lr}_{\text{base}} \times \sqrt{\frac{\text{batch\_size}_{\text{new}}}{\text{batch\_size}_{\text{base}}}}$$

**示例**:
```
Version A: batch_size = 65K tokens, lr = 3e-4
Version B: batch_size = 262K tokens (4× 更大)

线性缩放: lr = 3e-4 × 4 = 1.2e-3  (可能太大，不推荐)
平方根缩放: lr = 3e-4 × 2 = 6e-4  (更安全的选择)
保守策略: lr = 3e-4 (不变，最稳妥)
```

**建议**: 1B 模型直接使用 `lr = 3e-4`。如果训练不稳定，降低到 `2e-4`。

### 5.2 Warmup 步数

模型越大，warmup 应该越长：

- Version A: 2000 步
- Version B: 2000-4000 步

### 5.3 训练 Token 数

根据 Chinchilla Scaling Law，最优训练 token 数约为参数量的 20 倍：

| 模型 | 参数量 | Chinchilla 最优 tokens | 实际推荐 |
|------|--------|----------------------|---------|
| Version A | 300M | 6B | 3-6B |
| Version B | 1B | 20B | 20-50B |

如果数据不够，可以对数据重复使用（训练多个 epoch），但效果会递减。

### 5.4 其他调整

| 超参数 | Version A | Version B | 说明 |
|--------|-----------|-----------|------|
| micro_batch_size | 8 | 4 (per GPU) | 每卡显存限制 |
| grad_accumulation | 4 | 4 | 保持 |
| seq_len | 2048 | 4096 | 更长上下文 |
| gradient_checkpointing | 可选 | **必须** | 1B 模型激活值太大 |

---

## 6. 多机训练

### 6.1 网络配置

如果需要跨机器训练（如 2 台 4 卡服务器）：

```bash
# 机器 1 (master)
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/pretrain.py

# 机器 2
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
torchrun --nnodes=2 --node_rank=1 --nproc_per_node=4 \
    --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    scripts/pretrain.py
```

### 6.2 常见网络问题

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 防火墙阻断 | 连接超时 | 开放 29500 端口 |
| DNS 解析失败 | hostname 错误 | 使用 IP 地址 |
| 网卡选错 | NCCL timeout | 设置 `NCCL_SOCKET_IFNAME=eth0` |
| 带宽不足 | 训练极慢 | 至少 10GbE，推荐 100GbE+ |

### 6.3 性能优化

跨机通信比机内慢得多，以下优化可以减少通信影响：

1. **增大梯度累积步数**: 减少通信频率
2. **使用梯度压缩**: 以精度换带宽
3. **通信-计算重叠**: FSDP 默认开启 prefetch
4. **检查点异步保存**: 避免所有 GPU 同时写磁盘

---

## 7. 扩展检查清单

### 7.1 开始前

- [ ] Version A 训练已完成，loss 降到 3-4
- [ ] SFT 和 DPO 流程在 Version A 上验证通过
- [ ] 4+ 张 24GB GPU 可用（或云服务账户就绪）
- [ ] 准备了 20B+ tokens 的训练数据
- [ ] 训练了 64K 词汇表的分词器（或准备复用 32K）

### 7.2 代码修改

- [ ] 实现 `src/training/distributed.py`
- [ ] 修改 `scripts/pretrain.py` 添加分布式支持
- [ ] 更新 `Trainer` 以支持 FSDP
- [ ] 使用 `configs/model_1b.yaml` 配置
- [ ] Checkpoint 保存/加载适配 FSDP

### 7.3 训练验证

- [ ] 单卡测试: `torchrun --nproc_per_node=1` 确认代码无误
- [ ] 多卡测试: `torchrun --nproc_per_node=4` 验证分布式通信
- [ ] 短训练测试: 跑 100 步，确认 loss 正常下降
- [ ] 显存检查: `nvidia-smi` 确认每卡显存 < 22GB
- [ ] 吞吐量基准: 记录 tokens/sec，与预期对比

### 7.4 预期指标

| 阶段 | 指标 | Version A | Version B |
|------|------|-----------|-----------|
| 初始 loss | cross-entropy | ~10.4 | ~11.1 (ln(64000)) |
| 最终 loss | cross-entropy | 3.0-4.0 | 2.5-3.5 |
| 困惑度 | PPL | 20-50 | 12-30 |
| HellaSwag | accuracy | 30-40% | 45-55% |
| 训练吞吐 | tokens/sec/GPU | ~20K | ~15K |

**Version B 的 loss 应该比 Version A 更低**——如果不是，说明训练配置可能有问题。
