# 常见问题排查指南

> 训练 LLM 过程中会遇到各种问题。本文档按**症状分类**，提供诊断方法和解决方案。

---

## 目录

- [1. 显存不足 (OOM)](#1-显存不足-oom)
- [2. Loss 异常](#2-loss-异常)
- [3. 训练速度问题](#3-训练速度问题)
- [4. 数值不稳定](#4-数值不稳定)
- [5. SFT 阶段问题](#5-sft-阶段问题)
- [6. DPO 阶段问题](#6-dpo-阶段问题)
- [7. 生成质量问题](#7-生成质量问题)
- [8. 环境与依赖问题](#8-环境与依赖问题)
- [9. 分布式训练常见问题（Version B）](#9-分布式训练常见问题version-b)

---

## 1. 显存不足 (OOM)

### 症状
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
Tried to allocate X GiB (GPU 0; 24.00 GiB total capacity; Y GiB already allocated)
```

### 诊断

先确认显存使用分布：

```python
# 在训练开始前和 forward 后分别打印
print(f"已分配: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"已缓存: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
print(f"峰值: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
```

显存占用大致分布:
```
模型参数 (BF16):   ~600 MB (300M × 2 bytes)
优化器状态 (FP32): ~2.4 GB (300M × 4 bytes × 2 for Adam)
梯度 (BF16):       ~600 MB
激活值:            可变，取决于 batch_size × seq_len
```

### 解决方案（按推荐顺序）

| 方案 | 节省量 | 代价 | 推荐度 |
|------|--------|------|--------|
| 1. 开启梯度检查点 | ~50-70% 激活显存 | 增加约 30% 计算时间 | ★★★ 强烈推荐 |
| 2. 减小 micro_batch_size | 线性减少 | 增大梯度累积步数保持有效 batch size | ★★★ |
| 3. 使用 BF16 混合精度 | ~50% 模型+梯度 | 几乎无 | ★★★ |
| 4. 减小 seq_len | 显著减少激活值 | 模型学到的上下文长度受限 | ★★ |
| 5. 使用 FSDP/DeepSpeed | 在多卡间分片 | 需要多 GPU | ★★ |
| 6. CPU offload | 部分放 CPU | 大幅降低速度 | ★ 最后手段 |

**梯度检查点** 启用方法：
```python
# 在模型的 __init__ 中
from torch.utils.checkpoint import checkpoint

# 在 forward 中
for block in self.blocks:
    x = checkpoint(block, x, ...)  # 不保存中间激活，反向时重算
```

---

## 2. Loss 异常

### 2.1 Loss 不下降（保持在初始值附近）

**可能原因**:

| 原因 | 诊断方法 | 解决方案 |
|------|---------|---------|
| 学习率太小 | 打印 lr 值确认 | 增大学习率或检查 scheduler |
| 梯度为 0 | 打印梯度范数 | 检查模型 forward 是否正确接入 loss |
| 数据加载问题 | 打印 batch 内容 | 确认数据不全是 pad token |
| 因果掩码方向错误 | 可视化注意力矩阵 | 上三角应为 -inf，不是下三角 |
| Labels 没对齐 | 检查 labels shift | `labels[t]` 应对应 `logits[t-1]` |

**因果掩码检查**:
```python
# 正确的因果掩码（上三角为 -inf）
mask = torch.triu(torch.ones(T, T), diagonal=1) * float('-inf')
# 含义: position i 只能看到 position 0..i
```

### 2.2 Loss 突然飙升 / 爆炸

**可能原因**:
- 学习率太大 → 降低 peak lr 或延长 warmup
- 梯度爆炸 → 启用梯度裁剪 `clip_grad_norm_(params, max_norm=1.0)`
- 数据中有异常样本 → 跳过 loss 异常大的 batch
- 训练后期学习率没衰减 → 检查 cosine scheduler 实现

**快速诊断**:
```python
# 在每步训练后打印
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"Step {step}: loss={loss:.4f}, grad_norm={grad_norm:.4f}, lr={scheduler.get_lr():.2e}")

# 如果 grad_norm 频繁 > 10，说明训练不稳定
```

### 2.3 Loss 变成 NaN

**NaN 通常意味着数值溢出**。排查顺序:

1. **检查学习率**: 是否远大于推荐值？
2. **检查数据**: 输入 token ID 是否在 [0, vocab_size) 范围内？
3. **检查 softmax 输入**: logits 是否存在 ±inf？
4. **检查除法**: RMSNorm 的 eps 是否足够大？(推荐 1e-6)
5. **检查混合精度**: FP16 的范围比 BF16 小，更容易溢出

**紧急修复**: 从最近的有效 checkpoint 恢复，降低学习率重新训练。

### 2.4 Loss 下降后停滞在较高值

**可能原因**:
- 数据质量低 → 检查数据清洗
- 模型容量不够 → 对于复杂数据可能需要更大模型
- 学习率衰减太快 → 调整 cosine schedule 的最小学习率比例
- 数据已遍历多次 → 检查 epoch 数，单次遍历足够

---

## 3. 训练速度问题

### 3.1 GPU 利用率低 (<50%)

**症状**: `nvidia-smi` 显示 GPU Util 在 0-50% 之间波动

**原因与方案**:

| 原因 | 诊断 | 解决方案 |
|------|------|---------|
| 数据加载瓶颈 | GPU 间歇性 0% | `num_workers=4+`, `pin_memory=True` |
| 文本长度不均 | 短序列浪费计算 | 序列打包 (packing)，将多个短文档拼接到 max_len |
| CPU 预处理慢 | 离线预处理 | 数据提前 tokenize 并保存为二进制 |

**序列打包** (Sequence Packing):
```
不打包: [tok tok tok PAD PAD PAD]  ← 50% 计算浪费
         [tok tok tok tok tok PAD]
打包:   [tok tok tok SEP tok tok]  ← 100% 利用
         [tok tok tok tok tok tok]
```

### 3.2 单步训练时间过长

**优化顺序** (从最有效到最不重要):

1. **使用 BF16 混合精度**: 通常提速 2-3×
2. **Flash Attention**: `pip install flash-attn`，提速 2-4× 并减少显存
3. **torch.compile**: `model = torch.compile(model)`，PyTorch 2.0+ 自动优化
4. **增大 batch size**: GPU 更善于处理大 batch
5. **数据加载并行**: `DataLoader(num_workers=4, pin_memory=True, prefetch_factor=2)`

---

## 4. 数值不稳定

### 4.1 FP16 vs BF16

| 特性 | FP16 | BF16 |
|------|------|------|
| 范围 | ±65504 | ±3.4×10³⁸ |
| 精度 | ~3.3 位有效数字 | ~2.4 位有效数字 |
| 稳定性 | 需要 loss scaling | 不需要 |
| 推荐度 | 不推荐用于预训练 | **推荐** |

如果 GPU 不支持 BF16（A100 之前的部分型号），使用 FP16 时需要启用 GradScaler:
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(dtype=torch.float16):
    loss = model(batch)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4.2 权重初始化问题

**症状**: 训练刚开始就出现 NaN 或 loss 极大

**排查**: 检查模型参数初始化后的统计量：
```python
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean():.6f}, std={param.std():.6f}")
    # 期望: mean ≈ 0, std ≈ 0.02
```

**常见错误**:
- 忘记初始化 → 权重是随机的（std 很大）
- 残差分支没有缩放 → 深层模型容易梯度爆炸
- 没有将 embedding 和 lm_head 共享权重 (weight tying) → 如果设计中需要共享

---

## 5. SFT 阶段问题

### 5.1 SFT 后模型退化

**症状**: SFT 后模型在预训练任务上的 perplexity 大幅上升，或生成质量变差

**原因**: 灾难性遗忘 (Catastrophic Forgetting)

**解决方案**:
- 降低学习率 (2e-5 → 1e-5 或更低)
- 减少训练 epoch (3 → 1-2)
- 使用 LoRA 代替全量微调（冻结大部分参数）
- 混入少量预训练数据（如 5%）

### 5.2 SFT 后模型不遵循指令格式

**症状**: 模型回答时不在 `<|assistant|>` 标记后生成，或者无法正确停止

**排查要点**:
1. 检查对话模板是否正确应用到训练数据
2. 检查特殊 token 是否被正确 tokenize（而非被拆分为子 token）
3. 检查 loss mask 是否只计算 assistant 部分

```python
# 验证特殊 token
for token_str, token_id in special_tokens.items():
    encoded = tokenizer.encode(token_str, add_special_tokens=False)
    assert encoded == [token_id], f"{token_str} 编码错误: {encoded} != [{token_id}]"
```

### 5.3 SFT 损失不下降

**排查**:
- 确认 labels 不全是 -100（如果全是 -100，loss 会是 0 或 NaN）
- 确认 labels 和 input_ids 正确错开了 1 位
- 检查学习率是否生效（打印 `optimizer.param_groups[0]['lr']`）

---

## 6. DPO 阶段问题

### 6.1 DPO Loss 不收敛

**可能原因**:

| 原因 | 诊断 | 解决方案 |
|------|------|---------|
| β 太大 | reward margin 几乎为 0 | 减小 β (0.5 → 0.1) |
| β 太小 | loss 快速降到 0 (过拟合) | 增大 β |
| 参考模型不对 | 检查 ref_model 是否冻结 | `ref_model.requires_grad_(False)` |
| 数据质量差 | chosen/rejected 差异小 | 使用质量更好的偏好数据 |
| 学习率太大 | loss 震荡或发散 | 降低到 5e-7 或更低 |

### 6.2 DPO 后模型变得"绕圈子"

**症状**: 模型回答变得冗长啰嗦，不断重复或不愿直接回答

**原因**: DPO 过度优化，模型学会了"安全"的回避策略

**解决方案**:
- 减少 DPO 训练步数
- 增大 β（使模型不偏离 SFT 太远）
- 检查偏好数据中 chosen 回答是否过于保守

### 6.3 Log-probability 计算错误

这是 DPO 实现中最常见的 bug。关键点：

```python
# 正确: 计算整个序列的 log probability
def compute_log_prob(model, input_ids, labels):
    logits = model(input_ids).logits  # [B, T, V]
    # 注意 shift: logits[t] 预测 labels[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    # 只取对应 label 的 log prob
    per_token_log_prob = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    # 排除 padding
    mask = (shift_labels != -100).float()
    return (per_token_log_prob * mask).sum(-1)  # [B], 每个样本一个值
```

**常见错误**:
- 忘记 shift（logits 和 labels 没有错开 1 位）
- 没有排除 padding token
- 用了 softmax + log 而非 log_softmax（数值不稳定）

---

## 7. 生成质量问题

### 7.1 生成重复

**症状**: 模型反复输出相同的词或句子

**解决方案**:
- 增加 temperature (0.7-1.0)
- 启用 repetition penalty (1.1-1.3)
- 使用 Top-p sampling (p=0.9-0.95)
- 检查 KV Cache 实现是否正确（不正确的 cache 会导致注意力失效）

### 7.2 生成不停止

**症状**: 模型一直生成，不输出 EOS token

**原因**:
- 训练数据中 EOS 标记不正确
- 生成时 max_length 设置过大
- SFT 对话模板中缺少 `<eos>` 标记

**快速修复**: 设置合理的 `max_new_tokens`（如 512），并在生成后截断。

### 7.3 生成乱码

**排查顺序**:
1. 分词器是否正确加载？（编码-解码测试）
2. 模型权重是否正确加载？（检查 key 映射）
3. 模型是否在 eval 模式？（`model.eval()`）
4. 推理精度是否与训练一致？

---

## 8. 环境与依赖问题

### 8.1 CUDA 版本不匹配

**症状**: `RuntimeError: CUDA error: no kernel image is available for execution on the device`

**解决方案**:
```bash
# 检查当前 CUDA 版本
nvcc --version
python -c "import torch; print(torch.version.cuda)"

# 确保 PyTorch CUDA 版本与系统一致
pip install torch --index-url https://download.pytorch.org/whl/cu118  # 或 cu121
```

### 8.2 Flash Attention 安装失败

```bash
# Flash Attention 对 CUDA 版本和 GPU 架构有要求
# 支持: Ampere (A100, RTX 30xx), Ada (RTX 40xx), Hopper (H100)
# 不支持: Turing (RTX 20xx), Volta (V100)

pip install flash-attn --no-build-isolation

# 如果安装失败，可以使用 PyTorch 自带的高效注意力
# torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+)
```

### 8.3 多 GPU 通信问题

**症状**: NCCL timeout 或 hang

**排查**:
```bash
# 检查 GPU 互联拓扑
nvidia-smi topo -m

# 设置 NCCL 环境变量（调试用）
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 指定网络接口
export NCCL_P2P_DISABLE=1       # 禁用 P2P（如果 NVLink 有问题）
```

---

## 快速排错流程图

```
问题发生
  ↓
loss 是 NaN? ─── 是 → 检查学习率、数据范围、混合精度设置
  ↓ 否
loss 不下降? ─── 是 → 检查梯度（是否为0）、因果掩码、labels 对齐
  ↓ 否
loss 爆炸? ──── 是 → 降低学习率、启用梯度裁剪
  ↓ 否
OOM? ────────── 是 → 梯度检查点 → 减小 batch → 混合精度
  ↓ 否
训练太慢? ──── 是 → 混合精度 → Flash Attention → 增大 batch
  ↓ 否
生成质量差? ── 是 → 检查 checkpoint 加载 → 调整采样参数
  ↓ 否
分布式问题? ── 是 → 参考第 9 节
```

---

## 9. 分布式训练常见问题（Version B）

> 当你从 Version A（单卡）扩展到 Version B（多卡）时，会遇到一系列分布式训练特有的问题。本节针对使用 PyTorch FSDP/DDP 进行多卡训练的场景。

### 9.1 NCCL All-Reduce 超时

| | 内容 |
|---|------|
| **症状** | 训练 hang 住不动，最终报错 `NCCL timeout` 或 `RuntimeError: NCCL communicator was aborted`。日志可能显示某些 rank 已完成当前操作而其他 rank 仍在等待。 |
| **诊断方法** | 1. 设置 `export NCCL_DEBUG=INFO` 查看详细通信日志；2. 检查 `nvidia-smi topo -m` 确认 GPU 互联拓扑（NVLink vs PCIe）；3. 用 `torch.distributed.barrier()` 在关键位置插入同步点，定位哪个操作导致挂起；4. 检查是否某个 rank 在 forward/backward 路径上走了不同的分支（导致集合通信不匹配）。 |
| **解决方案** | 1. 增大超时时间: `torch.distributed.init_process_group(timeout=timedelta(minutes=30))`；2. 确保所有 rank 执行完全相同的计算图（禁止条件分支导致部分 rank 跳过通信操作）；3. 如果是网络问题，尝试 `export NCCL_SOCKET_IFNAME=eth0` 指定正确的网络接口；4. PCIe 互联的机器上可能需要 `export NCCL_P2P_DISABLE=1`。 |

### 9.2 FSDP 与梯度检查点的兼容性问题

| | 内容 |
|---|------|
| **症状** | 使用 FSDP + `torch.utils.checkpoint.checkpoint()` 时出现 `RuntimeError: Expected to mark a variable ready only once` 或显存没有按预期减少。 |
| **诊断方法** | 1. 确认 FSDP wrapping 策略是否正确——每个 `TransformerBlock` 应该是一个独立的 FSDP unit；2. 检查 `checkpoint()` 函数的 `use_reentrant` 参数设置；3. 对比开启和关闭 checkpoint 时的峰值显存，确认 checkpoint 是否生效。 |
| **解决方案** | 1. 在 PyTorch 2.0+ 中，推荐使用 `use_reentrant=False`（新版非重入式实现，兼容性更好）；2. FSDP wrap 的粒度要与 checkpoint 的粒度对齐——对每个被 FSDP wrap 的模块单独做 checkpoint；3. 确保 `checkpoint()` 包裹的函数没有 `torch.no_grad()` 上下文。 |

### 9.3 多卡训练时 Loss 与单卡不一致

| | 内容 |
|---|------|
| **症状** | 相同配置（总 batch size、学习率、数据）下，多卡训练的 loss 曲线与单卡显著不同，最终收敛结果也有差异。 |
| **诊断方法** | 1. 检查 loss 是否在 All-Reduce 前正确除以了 `world_size`（DDP 默认会做 gradient averaging，但梯度累积时需要手动处理）；2. 确认所有 rank 是否看到了不同的数据（而非重复数据）；3. 检查有效 batch size 的计算是否正确: `effective_bs = micro_bs × grad_accum × n_gpus`。 |
| **解决方案** | 1. 使用 `DistributedSampler` 确保数据在 rank 间无重叠；2. 如果使用梯度累积，loss 的 scaling 公式为 `loss = raw_loss / (grad_accum_steps)`（DDP 已经处理了跨卡平均）；3. 确保学习率的 warmup 和 decay 是按全局步数计算的，而非每个 rank 独立计算；4. 对比验证: 设 n_gpus=1 + grad_accum=4 与 n_gpus=4 + grad_accum=1 的 loss 应该接近。 |

### 9.4 Checkpoint 保存时的 OOM

| | 内容 |
|---|------|
| **症状** | 训练正常进行，但在保存 checkpoint 时某张卡 OOM 崩溃。这在使用 FSDP 时尤其常见。 |
| **诊断方法** | 1. FSDP 的 `full_state_dict` 模式会将所有分片参数临时收集到 rank 0，导致 rank 0 的显存需求翻倍；2. 用 `torch.cuda.memory_summary()` 在保存前后对比显存状态；3. 检查是否同时保存了模型和优化器状态（优化器状态通常是模型参数的 2-3 倍）。 |
| **解决方案** | 1. 使用 FSDP 的 `ShardedStateDictType.SHARDED_STATE_DICT`，每个 rank 只保存自己的分片，避免全量收集：<br>`with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT): torch.save(...)`；2. 如果必须保存完整 state_dict，在保存前调用 `torch.cuda.empty_cache()` 释放缓存；3. 分步保存: 先保存模型参数、释放显存，再保存优化器状态；4. 考虑使用 CPU offload 保存: `with FSDP.state_dict_type(..., offload_to_cpu=True):`。 |

### 9.5 数据并行下的随机种子管理

| | 内容 |
|---|------|
| **症状** | 多次训练结果完全不可复现；或者所有 rank 的数据增强/dropout 行为完全相同（丧失了随机性的多样性）。 |
| **诊断方法** | 1. 检查每个 rank 的随机种子是否被显式设置；2. 打印各 rank 的 `torch.initial_seed()` 和 `numpy.random.get_state()` 确认是否相同；3. 验证 `DistributedSampler` 的 `seed` 参数是否一致，`epoch` 参数是否在每个 epoch 更新。 |
| **解决方案** | 1. **全局一致性**: 模型初始化使用相同种子（所有 rank 种子相同），确保初始参数一致；2. **数据多样性**: 数据加载使用 `rank + base_seed` 作为种子，确保各卡看到不同数据顺序；3. **Dropout 多样性**: 通常不需要特殊处理（DDP/FSDP 下各卡的 dropout mask 天然不同）；4. **完整可复现性**: 保存 checkpoint 时同时保存所有 rank 的 RNG 状态 `torch.cuda.get_rng_state()`，恢复时逐卡加载。 |

```python
# 推荐的多卡随机种子设置
def set_seed(base_seed: int, rank: int):
    """设置随机种子，确保可复现性与多样性的平衡。"""
    # 模型初始化: 所有 rank 使用相同种子
    torch.manual_seed(base_seed)
    # 数据加载: 各 rank 使用不同种子
    torch.manual_seed(base_seed + rank)
    np.random.seed(base_seed + rank)
    random.seed(base_seed + rank)
    # CUDA 随机数
    torch.cuda.manual_seed_all(base_seed + rank)
```
