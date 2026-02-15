# 训练全流程指南

> 本文档详细介绍从数据准备到模型部署的完整训练流程。每个阶段都会说明**为什么要这样做**、**具体怎么做**、以及**如何验证做对了**。

---

## 目录

- [阶段 1: 数据准备](#阶段-1-数据准备)
- [阶段 2: 分词器训练](#阶段-2-分词器训练)
- [阶段 3: 模型构建与验证](#阶段-3-模型构建与验证)
- [阶段 4: 预训练](#阶段-4-预训练)
- [阶段 5: SFT 指令微调](#阶段-5-sft-指令微调)
- [阶段 6: DPO 偏好对齐](#阶段-6-dpo-偏好对齐)
- [阶段 7: 评估与部署](#阶段-7-评估与部署)
- [阶段 8: 反思与迭代](#阶段-8-反思与迭代)
- [附录: 超参数速查表](#附录-超参数速查表)

---

## 阶段 1: 数据准备

> **对应模块**: [模块 7 - 数据工程](../../07_data_engineering/README.md)

### 1.1 数据集选择

训练数据的质量和规模直接决定模型能力的上限。以下是推荐的数据集，按规模从小到大排列：

**英文数据**:

| 数据集 | 规模 | 特点 | 适用场景 |
|--------|------|------|----------|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | ~2M 条短故事, ~500M tokens | 语法简单、故事完整 | 快速验证流程 (Version A) |
| [OpenWebText2](https://openwebtext2.readthedocs.io/) | ~17B tokens | Reddit 高质量链接 | 中等规模预训练 |
| [SlimPajama](https://huggingface.co/datasets/cerebras/SlimPajama-627B) | 627B tokens | 去重后的多源数据 | 大规模预训练 (Version B) |
| [RedPajama-v2](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2) | 30T tokens | 最大开源数据集之一 | 生产级训练 |

**中文数据**:

| 数据集 | 规模 | 特点 | 适用场景 |
|--------|------|------|----------|
| [SkyPile](https://huggingface.co/datasets/Skywork/SkyPile-150B) | 150B tokens | 高质量中文网页 | 中文模型训练 |
| [WuDaoCorpora](https://data.baai.ac.cn/details/WuDaoCorporaText) | 200GB 文本 | 多领域中文 | 通用中文训练 |
| [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX) | 多语言 | mC4 + OSCAR 清洗 | 多语言模型 |

**建议路径**:
- **第一次训练**: 用 TinyStories，数据小、质量高，几小时可跑通全流程
- **正式训练 (Version A)**: SlimPajama 采样 3-6B tokens
- **大规模训练 (Version B)**: SlimPajama 采样 20-50B tokens

### 1.2 数据下载

```bash
# 方式 1: HuggingFace Datasets（推荐）
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('roneneldan/TinyStories', split='train')
ds.to_json('data/tinystories_train.jsonl')
"

# 方式 2: 直接下载
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

# 方式 3: 使用 git lfs（适合大数据集）
git lfs install
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B
```

### 1.3 数据清洗

数据清洗是最容易被低估但影响巨大的环节。以下是推荐的清洗流程：

```
原始数据
  ↓
1. 编码统一 (UTF-8)
  ↓
2. HTML/XML 标签移除
  ↓
3. 特殊字符处理（保留合理的标点和符号）
  ↓
4. 语言过滤（如果只训练中文/英文模型）
  ↓
5. 长度过滤（去除过短 <50 字符 和过长 >100K 字符的文档）
  ↓
6. 质量过滤（困惑度过滤、重复率检查）
  ↓
7. 去重（精确去重 + 近似去重）
  ↓
清洗后数据
```

**关键决策**:

- **去重方法**: MinHash + LSH 是最常用的近似去重方法，推荐使用 [datasketch](https://github.com/ekzhu/datasketch) 库
- **质量过滤**: 可用一个小型语言模型计算困惑度，过滤掉困惑度极高（乱码）或极低（重复模板）的文本
- **数据混合比例**: 如果混合多种数据源，需要仔细调配比例。通常代码 10-15%、学术 10-15%、网页 50-60%、书籍 10-15%

### 1.4 数据格式

清洗后的数据保存为**纯文本**（供分词器训练）和 **token 化后的二进制文件**（供预训练使用）：

```
data/
├── raw/                    # 原始下载数据
├── cleaned/                # 清洗后的文本
│   ├── train.txt           # 训练集（纯文本，用于分词器训练）
│   └── val.txt             # 验证集
└── tokenized/              # token 化后的二进制数据
    ├── train.bin           # np.memmap 格式，dtype=uint16
    └── val.bin
```

### 1.5 验证方法

- [ ] 检查清洗后数据的总 token 数是否达到目标（Version A: 3-6B, Version B: 20-50B）
- [ ] 随机抽查 10-20 条样本，确认内容质量
- [ ] 检查数据中是否残留 HTML 标签、重复内容
- [ ] 验证 train/val 分割没有数据泄漏

---

## 阶段 2: 分词器训练

> **对应模块**: [模块 1 - 分词](../../01_tokenization/README.md)

### 2.1 为什么要自己训练分词器？

使用预训练好的分词器（如 GPT-2 的 tokenizer）也可以，但自己训练有以下优势：
- **词汇表适配**: 你的训练数据可能包含特定领域的术语
- **效率优化**: 针对数据分布优化，减少平均 token 数
- **中文支持**: 通用分词器对中文的覆盖可能不够好

### 2.2 训练命令

```bash
# 使用 sentencepiece 训练 BPE 分词器
python scripts/train_tokenizer.py \
    --input data/cleaned/train.txt \
    --output tokenizer/llm \
    --vocab_size 32000 \
    --character_coverage 0.9995 \
    --model_type bpe
```

### 2.3 关键参数说明

| 参数 | Version A | Version B | 说明 |
|------|-----------|-----------|------|
| `vocab_size` | 32,000 | 64,000 | 更大的词汇表减少 token 数，但增加 embedding 参数 |
| `character_coverage` | 0.9995 | 0.9995 | 覆盖 99.95% 的字符，适合包含中文的数据 |
| `model_type` | bpe | bpe | Byte-Pair Encoding，最常用的子词分词算法 |
| `byte_fallback` | True | True | 处理未见过的字符，使用 UTF-8 字节作为回退 |

### 2.4 特殊 Token 设计

```python
special_tokens = {
    "<pad>": 0,     # 填充 token
    "<unk>": 1,     # 未知 token（有 byte_fallback 时几乎不会用到）
    "<bos>": 2,     # 序列开始
    "<eos>": 3,     # 序列结束

    # SFT 阶段使用的对话 token（提前预留）
    "<|system|>": 4,
    "<|user|>": 5,
    "<|assistant|>": 6,
    "<|end_turn|>": 7,
}
```

**为什么要提前预留对话 token？** 因为分词器训练后词汇表就固定了。如果 SFT 阶段才添加新 token，需要调整 embedding 层大小，可能影响预训练学到的知识。

### 2.5 数据 Token 化

分词器训练好后，将训练数据转换为 token 序列并保存为二进制文件：

```python
# 伪代码
import numpy as np

tokenizer = Tokenizer.load("tokenizer/llm.model")
tokens = tokenizer.encode(text)

# 保存为 np.memmap（可随机访问，不需要全部加载到内存）
arr = np.memmap("data/tokenized/train.bin", dtype=np.uint16, mode='w+', shape=(len(tokens),))
arr[:] = tokens
arr.flush()
```

**为什么用 uint16？** 词汇表大小 32K < 65535 (uint16 最大值)，用 uint16 比 int32 节省一半存储空间。如果词汇表 > 65535，需要用 uint32。

### 2.6 验证方法

```python
# 1. 编码-解码一致性
text = "Hello, world! 你好世界"
assert tokenizer.decode(tokenizer.encode(text)) == text

# 2. 特殊 token
assert tokenizer.encode("<bos>", add_special_tokens=False) == [2]

# 3. 平均 token 数（好的分词器应该让文本更"压缩"）
# 英文: 约 1 token/4-5 字符
# 中文: 约 1-2 token/字
sample = open("data/cleaned/val.txt").read()[:10000]
tokens = tokenizer.encode(sample)
print(f"压缩率: {len(sample) / len(tokens):.1f} 字符/token")
```

---

## 阶段 3: 模型构建与验证

> **对应模块**: [模块 3 - Transformer](../../03_transformer/README.md), [模块 4 - Decoder-Only](../../04_decoder_only/README.md), [模块 5 - 注意力变体](../../05_attention_variants/README.md)

### 3.1 推荐实现顺序

模型各组件之间有依赖关系，建议按以下顺序实现：

```
1. RMSNorm          ← 最简单，独立组件
   ↓
2. SwiGLU FFN       ← 依赖 config，独立组件
   ↓
3. RoPE             ← 需要理解旋转矩阵
   ↓
4. GQA Attention    ← 依赖 RoPE，最复杂
   ↓
5. TransformerBlock ← 组装 Attention + FFN + RMSNorm
   ↓
6. DecoderOnlyLM   ← 组装 Embedding + N×Block + LM Head
```

### 3.2 各组件实现要点

#### RMSNorm

RMSNorm 比 LayerNorm 更简单——没有减均值步骤：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

实现时注意 `eps` 的默认值通常为 `1e-6`（比 LayerNorm 的 `1e-5` 更小）。

#### SwiGLU FFN

SwiGLU 使用三个权重矩阵而非两个：

$$\text{SwiGLU}(x) = (\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}) W_{\text{down}}$$

其中 Swish(x) = x · σ(x)。FFN 隐藏维度约为 `8/3 × d_model`，而非传统的 `4 × d_model`，这是为了在相同参数量下获得更好的性能。

#### RoPE (旋转位置编码)

RoPE 将位置信息编码为旋转操作，核心是对 Q、K 向量的每对相邻维度施加旋转：

$$\begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix} \leftarrow \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

其中 $\theta_i = 10000^{-2i/d}$, $m$ 是位置索引。

**实现技巧**: 不要用矩阵乘法实现旋转，而是用分组交换 + 逐元素乘法，更高效：
```python
# 将 [q0, q1, q2, q3, ...] 变为 [-q1, q0, -q3, q2, ...]
x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x)
output = x * cos + x_rotated * sin
```

#### GQA (分组查询注意力)

GQA 是 MHA 和 MQA 的折中。关键在于 KV 头的共享：

- Version A: 16 个 Q 头, 4 个 KV 头 → 每 4 个 Q 头共享 1 组 KV
- Version B: 32 个 Q 头, 8 个 KV 头 → 每 4 个 Q 头共享 1 组 KV

**广播方式**:
```python
# 方法 1: repeat_interleave（显式复制）
k = k.repeat_interleave(n_heads // n_kv_heads, dim=1)

# 方法 2: expand + reshape（不额外占内存，推荐）
k = k.unsqueeze(2).expand(-1, -1, n_heads // n_kv_heads, -1, -1).reshape(B, n_heads, T, head_dim)
```

### 3.3 权重初始化

权重初始化对训练稳定性至关重要。推荐的初始化方案：

| 层 | 初始化方法 | 说明 |
|----|-----------|------|
| Embedding | N(0, 0.02) | 标准做法 |
| Linear (常规) | N(0, 0.02) | Xavier 的简化版 |
| Linear (残差分支输出) | N(0, 0.02 / √(2·n_layers)) | 避免残差累加导致值爆炸 |
| RMSNorm γ | 全 1 | 初始时等于恒等变换 |
| Bias | 全 0 | 如果使用 bias 的话 |

### 3.4 验证清单

```python
from src.model.config import config_300m
from src.model.model import DecoderOnlyLM
import torch

# 1. 参数量验证
config, _ = config_300m()
model = DecoderOnlyLM(config)
n_params = model.count_parameters()
print(f"参数量: {n_params:,}")
assert 280_000_000 < n_params < 320_000_000, f"参数量不对: {n_params}"

# 2. 前向传播
x = torch.randint(0, config.vocab_size, (2, 128))
output = model(x, labels=x)
assert output['logits'].shape == (2, 128, config.vocab_size)

# 3. 初始 loss（应接近 ln(vocab_size)）
expected_loss = torch.log(torch.tensor(float(config.vocab_size)))
actual_loss = output['loss']
print(f"初始 loss: {actual_loss:.4f}, 期望 ≈ {expected_loss:.4f}")
assert abs(actual_loss - expected_loss) < 1.0, "初始 loss 偏差过大"

# 4. 梯度流检查
output['loss'].backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm == 0:
            print(f"WARNING: {name} 梯度为 0!")
        if torch.isnan(param.grad).any():
            print(f"WARNING: {name} 梯度包含 NaN!")
```

---

## 阶段 4: 预训练

> **对应模块**: [模块 8A - 预训练目标](../../08a_pretraining_objectives/README.md), [模块 8C - 训练工程](../../08c_training_engineering/README.md), [模块 9 - 分布式训练](../../09_distributed/README.md)

### 4.1 核心目标: 下一 Token 预测 (NTP)

预训练的目标非常简单——给定前 t 个 token，预测第 t+1 个 token：

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_1, ..., x_{t-1})$$

这就是交叉熵损失，PyTorch 中用 `F.cross_entropy(logits, targets)` 实现。

### 4.2 超参数选择

以下是 Version A (300M) 的推荐超参数：

| 超参数 | 值 | 选择理由 |
|--------|---|---------|
| 学习率 (peak) | 3e-4 | 300M 模型的常用值；参考 Chinchilla |
| 学习率调度 | Cosine decay | 平滑衰减到 peak 的 10% |
| 预热步数 | 2000 | 约占总步数的 1-2% |
| Batch size | 32 × 2048 = 65K tokens/step | micro_batch=8, grad_accum=4 |
| 总训练 tokens | 3-6B | Chinchilla 最优: ~20×参数量 |
| 权重衰减 | 0.1 | 只应用于 2D 参数 (矩阵)，不应用于 bias 和 norm |
| 梯度裁剪 | 1.0 | 全局梯度范数裁剪 |
| 混合精度 | BF16 | 比 FP16 更稳定（不需要 loss scaling） |
| 梯度检查点 | 开启 | 用计算换显存 |
| Adam β | (0.9, 0.95) | β2 比默认的 0.999 小，更稳定 |
| Adam ε | 1e-8 | 标准值 |

### 4.3 关于 Batch Size 的计算

```
有效 batch size (tokens/step)
  = micro_batch_size × seq_len × gradient_accumulation_steps × n_gpus
  = 8 × 2048 × 4 × 1
  = 65,536 tokens/step

总训练步数
  = total_tokens / tokens_per_step
  = 6,000,000,000 / 65,536
  ≈ 91,553 步
```

如果 24GB GPU 无法放下 micro_batch=8，可以减小到 4 并增大 gradient_accumulation 到 8，效果等价。

### 4.4 训练循环核心逻辑

```
for step in range(total_steps):
    optimizer.zero_grad()

    # 梯度累积
    for micro_step in range(gradient_accumulation_steps):
        batch = next(data_iter)
        with autocast(dtype=bfloat16):
            loss = model(batch) / gradient_accumulation_steps
        loss.backward()  # 梯度在多次 backward 中累积

    # 梯度裁剪
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

    # 更新参数
    optimizer.step()
    lr_scheduler.step()

    # 日志记录
    if step % log_interval == 0:
        log(loss, grad_norm, lr, throughput)

    # 保存 checkpoint
    if step % save_interval == 0:
        save_checkpoint(model, optimizer, step)
```

### 4.5 训练过程监控

训练过程中应关注以下指标：

| 指标 | 健康范围 | 异常信号 |
|------|---------|---------|
| Loss | 稳定下降 | 突然上升、NaN |
| 梯度范数 | 0.1-10 | >100 (爆炸)、≈0 (消失) |
| 学习率 | 按 schedule 变化 | 不变（scheduler 没启动） |
| 吞吐量 | 稳定 | 突然下降（数据加载瓶颈） |
| GPU 显存 | <90% 峰值 | >95%（即将 OOM） |
| GPU 利用率 | >80% | <50%（IO 瓶颈） |

### 4.6 预期 Loss 曲线

```
Loss
10.4 ┤ ╮         ← 初始 loss ≈ ln(32000) ≈ 10.37
     │  ╲
 8.0 ┤   ╲       ← 前 1000 步：快速下降
     │    ╲
 6.0 ┤     ╲
     │      ╰─╮  ← 5000-10000 步：下降放缓
 4.0 ┤        ╰──────────╮  ← 中期：缓慢下降
     │                   ╰────── ← 后期：趋于平稳
 3.0 ┤                            最终 loss ≈ 3.0-4.0
     └────────────────────────────── Steps
     0    10K   20K   50K   90K
```

**如果 loss 不按这个趋势下降**，请参考 [troubleshooting.md](troubleshooting.md)。

### 4.7 Checkpoint 策略

```python
# 推荐保存内容
checkpoint = {
    'step': current_step,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'lr_scheduler_state_dict': scheduler.state_dict(),
    'config': config,
    'loss': current_loss,
    'rng_state': torch.random.get_rng_state(),  # 可复现性
}
torch.save(checkpoint, f"checkpoints/300m/step_{step}.pt")
```

**保存频率**: 每 5000-10000 步保存一次。保留最新 3 个 + 性能最好的 1 个。

### 4.8 训练稳定性与 Loss Spike 处理

在大规模预训练中，**Loss Spike**（loss 突然飙升）是一个常见但棘手的问题。虽然 300M 规模的学生项目不太容易遇到严重的 spike，但理解这个问题以及工业界的应对策略，对你未来参与更大规模训练非常有帮助。

#### 什么是 Loss Spike？

Loss Spike 是指训练过程中 loss 突然飙升数倍甚至数十倍，然后可能自行恢复，也可能导致训练彻底崩溃（loss 变为 NaN）。

```
Loss
 6.0 ┤
     │                    ╭╮  ← Loss Spike!
 5.0 ┤                    ││
     │                    ││
 4.0 ┤ ╲                  │╰──╮
     │  ╲                 │   ╰──── 可能恢复
 3.5 ┤   ╰─────────── ╌╌╌╌╯         也可能不恢复 → NaN
     └──────────────────────────── Steps
```

#### Google PaLM 的经验

Google 在训练 540B 参数的 PaLM 模型时遇到了约 20 次 Loss Spike。他们的经验总结：

1. **恢复策略**: 从 spike 发生前约 100 步的 checkpoint 恢复训练，并**跳过**导致 spike 的数据 batch（约 200-500 个 batch）
2. **保存频率**: 将 checkpoint 保存频率从每 10000 步提高到每 1000 步，以确保有足够近的恢复点
3. **根因分析**: 大多数 spike 的根因不明确，可能与某些异常数据样本、随机种子状态或数值精度的边界情况有关
4. **关键教训**: spike 的发生频率与模型规模正相关——模型越大，训练越脆弱

#### DeepSeek 的 FP8 训练稳定性策略

DeepSeek V3 在使用 FP8 混合精度训练时面临额外的数值稳定性挑战：

1. **动态 Loss Scaling**: FP8 的有效范围非常窄（E4M3 格式仅 ±448），需要根据梯度的实际范围动态调整 scaling factor
2. **Per-tensor vs Per-channel 量化**:
   - Per-channel 量化精度更高，但硬件实现复杂、计算开销大
   - DeepSeek 选择了 per-tensor 量化 + 细粒度分组策略（将 tensor 分为多个小块分别量化），在精度与效率之间取得平衡
3. **混合精度策略**: 关键操作（如 softmax、layer norm、loss 计算）始终保持 FP32 精度，只对矩阵乘法使用 FP8

#### 学生项目建议

在 300M 规模下，Loss Spike 的发生概率较低，但你仍然应该：

1. **实现 checkpoint 恢复机制**: 确保训练可以从任意 checkpoint 完全恢复（包括优化器状态、学习率、RNG 状态）
2. **记录每步的 loss 和梯度范数**: 这是事后分析问题的基础
3. **设置梯度裁剪**: `clip_grad_norm_(model.parameters(), max_norm=1.0)` 是预防梯度爆炸的第一道防线
4. **保留异常日志**: 如果某步的 loss 突然增加超过 2 倍，自动记录该步的数据样本信息，便于后续分析

```python
# 简单的 spike 检测与日志记录
if step > 0 and current_loss > 2 * moving_avg_loss:
    logger.warning(f"[Step {step}] Loss spike detected: {current_loss:.4f} "
                   f"(moving avg: {moving_avg_loss:.4f})")
    # 记录异常 batch 的信息
    logger.warning(f"  Batch indices: {batch_indices}")
    logger.warning(f"  Grad norm: {grad_norm:.4f}")
```

---

## 阶段 5: SFT 指令微调

> **对应模块**: [模块 10 - SFT](../../10_sft/README.md)

### 5.1 为什么需要 SFT？

预训练模型只会"续写"——给它任何文本，它都会尝试接着写下去。SFT 教会模型遵循指令、以对话形式回答问题。

```
预训练模型: "今天天气怎么样？" → "今天天气怎么样？昨天下了一场大雨..."  (续写)
SFT 模型:   "今天天气怎么样？" → "作为 AI，我无法获取实时天气。建议..."  (回答)
```

### 5.2 推荐 SFT 数据集

| 数据集 | 规模 | 语言 | 特点 |
|--------|------|------|------|
| [Alpaca-52K](https://github.com/tatsu-lab/stanford_alpaca) | 52K 条 | 英文 | GPT-4 生成，质量较高 |
| [BELLE](https://github.com/LianjiaTech/BELLE) | 200K+ | 中文 | 中文指令数据 |
| [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) | 70K 对话 | 多语言 | 真实多轮对话 |
| [OpenAssistant](https://huggingface.co/datasets/OpenAssistant/oasst1) | 88K 条 | 多语言 | 人工标注，含排名 |

**建议**: 先用 Alpaca-52K 快速验证，再用更大数据集训练。

### 5.3 对话模板

SFT 数据需要统一的对话格式。以下是一个常见模板：

```
<bos><|system|>你是一个有帮助的 AI 助手。<|end_turn|>
<|user|>什么是机器学习？<|end_turn|>
<|assistant|>机器学习是人工智能的一个分支...<|end_turn|><eos>
```

**关键**: 只对 **assistant 的回复部分** 计算损失。system 和 user 的部分设置 `labels = -100`（PyTorch 的 ignore_index）。

### 5.4 SFT 超参数

| 超参数 | 推荐值 | 说明 |
|--------|-------|------|
| 学习率 | 2e-5 | 比预训练小 10-15 倍 |
| Epochs | 2-3 | 过多会过拟合 |
| Batch size | 128 | 可通过梯度累积达到 |
| 序列长度 | 2048 | 与预训练一致 |
| 权重衰减 | 0.0 | SFT 通常不用权重衰减 |

### 5.5 LoRA: 高效微调

如果 GPU 显存不够全量微调，可以使用 LoRA (Low-Rank Adaptation)：

```
原始层: y = Wx
LoRA:  y = Wx + BAx    (B ∈ R^{d×r}, A ∈ R^{r×d}, r << d)

参数量对比 (300M 模型):
  全量微调: ~300M 可训练参数
  LoRA (r=16): ~2-4M 可训练参数（节省 99%）
```

LoRA 通常注入到 Q、K、V、O 投影矩阵中。

### 5.6 验证方法

SFT 后用几个标准问题测试：

```python
prompts = [
    "解释什么是深度学习。",
    "写一首关于春天的诗。",
    "请用 Python 实现快速排序。",
    "将以下英文翻译成中文: Machine learning is a subset of AI.",
]

for prompt in prompts:
    response = model.generate(prompt, max_length=256)
    print(f"Q: {prompt}")
    print(f"A: {response}\n")
```

好的 SFT 模型应该: 遵循指令格式 → 回答内容相关 → 语言流畅 → 适时停止。

---

## 阶段 6: DPO 偏好对齐

> **对应模块**: [模块 12 - DPO](../../12_dpo/README.md)

### 6.1 为什么需要对齐？

SFT 模型能遵循指令，但可能：
- 生成不安全或有害的内容
- 编造不存在的事实 (hallucination)
- 回答质量不稳定

DPO (Direct Preference Optimization) 通过人类偏好数据直接优化模型，使其倾向于生成人类更喜欢的回答。

### 6.2 偏好数据格式

```json
{
    "prompt": "解释量子计算的基本原理",
    "chosen": "量子计算利用量子力学的原理，如叠加态和纠缠...(详细、准确的回答)",
    "rejected": "量子计算就是用量子来计算...(笼统、不准确的回答)"
}
```

**推荐数据集**:
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf): 人类标注的有帮助性和无害性偏好
- [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback): 大规模多维度偏好

### 6.3 DPO 损失函数

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]$$

其中:
- $\pi_\theta$: 当前训练模型
- $\pi_{\text{ref}}$: 参考模型 (SFT 后的冻结副本)
- $y_w$: chosen 回答, $y_l$: rejected 回答
- $\beta$: 控制偏离参考模型的程度（通常 0.1-0.5）

### 6.4 DPO 超参数

| 超参数 | 推荐值 | 说明 |
|--------|-------|------|
| β | 0.1 | 越大越保守（越接近 SFT） |
| 学习率 | 5e-7 | 非常小，避免灾难性遗忘 |
| Epochs | 1-3 | DPO 对过拟合非常敏感 |
| Batch size | 32-64 | 较小的 batch 即可 |

### 6.5 DPO 的实现关键点

1. **参考模型**: 必须是 SFT 模型的冻结副本，训练过程中不更新
2. **log-prob 计算**: 需要计算模型对整个回答序列的 log 概率，不是单个 token
3. **数值稳定性**: 使用 `log_softmax` 而非 `softmax` + `log`

### 6.6 验证方法

- DPO loss 应稳定下降（但不能降到 0，否则过拟合）
- chosen 回答的 reward 应高于 rejected
- 用同一问题对比 SFT 前后和 DPO 前后的回答

---

## 阶段 7: 评估与部署

> **对应模块**: [模块 14 - 推理加速](../../14_inference/README.md), [模块 8B - Scaling Laws](../../08b_scaling_laws/README.md), [模块 13 - 推理与评估](../../13_reasoning/README.md)

### 7.1 评估指标

| 指标 | 含义 | 计算方式 | 参考值 (300M) |
|------|------|---------|--------------|
| 困惑度 (PPL) | 模型对文本的"困惑程度" | $e^{\mathcal{L}}$ | 20-50 |
| HellaSwag | 常识推理 | 4 选 1 准确率 | 30-40% |
| ARC-Easy | 科学常识 | 多选准确率 | 35-45% |
| MMLU | 综合知识 | 多选准确率 | 25-30% |

**注意**: 300M 模型在 benchmark 上的分数不会很高——这是正常的，不要气馁。学习价值在于理解评估方法本身。

### 7.2 使用 lm-eval-harness

```bash
pip install lm-eval

# 评估示例
lm_eval --model hf \
    --model_args pretrained=./exported_model \
    --tasks hellaswag,arc_easy \
    --batch_size 8 \
    --output_path results/
```

### 7.3 文本生成

实现文本生成时的核心概念：

- **Greedy Decoding**: 每步选概率最高的 token，生成确定但单调
- **Top-k Sampling**: 从概率最高的 k 个 token 中采样
- **Top-p (Nucleus) Sampling**: 从累积概率 ≤ p 的 token 中采样
- **Temperature**: 控制概率分布的"尖锐度"，T<1 更确定，T>1 更随机

```
Temperature 效果:
  T=0.1: "机器学习是人工智能的一个分支。"（几乎确定性）
  T=0.7: "机器学习是一种让计算机从数据中学习的方法。"（平衡）
  T=1.5: "机器学习是探索数据海洋的魔法指南针！"（更有创意但可能胡说）
```

### 7.4 KV Cache

推理时使用 KV Cache 避免重复计算已处理过的 token：

```
不用 Cache: 每生成一个新 token，重新计算整个序列的注意力
  第 1 步: 计算 [t1] → output1
  第 2 步: 计算 [t1, t2] → output2
  第 3 步: 计算 [t1, t2, t3] → output3
  复杂度: O(n^2) per step, O(n^3) total

使用 Cache: 缓存已计算的 K, V，只计算新 token
  第 1 步: 计算 [t1] → output1, 缓存 K1, V1
  第 2 步: 计算 [t2], 用缓存的 K1V1 → output2, 缓存 K2, V2
  第 3 步: 计算 [t3], 用缓存的 K1V1+K2V2 → output3
  复杂度: O(n) per step, O(n^2) total
```

### 7.5 量化（可选）

量化可以减小模型体积、加速推理：

| 方法 | 精度 | 模型大小 | 质量损失 |
|------|------|---------|---------|
| FP32 | 32 bit | 1.2 GB (300M) | 基准 |
| BF16 | 16 bit | 600 MB | 几乎无 |
| INT8 | 8 bit | 300 MB | 轻微 |
| INT4 | 4 bit | 150 MB | 明显（但通常可接受） |

---

## 阶段 8: 反思与迭代

> 训练完成不是终点，而是改进的起点。这个阶段帮助你从"跑通了"走向"做得好"。

### 8.1 分析模型的失败案例

训练结束后，不要只看整体指标（loss、PPL、benchmark 分数），更重要的是**定性分析模型在哪些方面做得不好**：

```python
# 按类别分析评估结果
categories = {
    "常识推理": ["为什么天空是蓝色的？", "水的沸点是多少？"],
    "数学能力": ["计算 123 + 456", "解方程 2x + 3 = 7"],
    "代码生成": ["用 Python 写一个冒泡排序", "解释什么是递归"],
    "多轮对话": ["（多轮上下文跟踪测试）"],
    "安全性":   ["（拒绝有害请求的测试）"],
}

for category, prompts in categories.items():
    results = evaluate(model, prompts)
    print(f"[{category}] 通过率: {results['pass_rate']:.1%}")
    # 记录失败案例用于分析
    for failure in results['failures']:
        print(f"  FAIL: {failure['prompt']} -> {failure['response'][:100]}")
```

### 8.2 迭代改进方向

根据失败分析，确定改进优先级（通常按以下顺序排查）：

1. **数据质量**: 最常见的瓶颈。检查训练数据中是否缺乏某类知识、是否存在噪声数据、数据配比是否合理
2. **训练超参数**: 学习率、batch size、训练步数是否合适？是否训练不足或过拟合？
3. **模型架构**: 在 300M 规模下，架构差异的影响相对有限，但可以尝试调整层数/宽度比例

### 8.3 撰写技术报告

强烈建议你为整个训练过程撰写一份技术报告（即使只是给自己看），内容包括：

- **设计决策记录**: 为什么选择 GQA 而非 MHA？为什么 FFN 维度选 2730？每个决策背后的理由
- **实验结果汇总**: loss 曲线、各阶段（预训练/SFT/DPO）前后的对比示例、benchmark 分数
- **踩坑与教训**: 遇到了哪些 bug？花了多长时间排查？最终如何解决？
- **资源消耗统计**: 训练总耗时、GPU 小时数、数据处理时间
- **未来改进方向**: 如果再做一次，你会改变什么？

> 这种记录习惯对于科研和工程工作都非常宝贵。工业界的技术报告（如 PaLM、LLaMA、DeepSeek 的论文）本质上就是更详细版本的实验记录。

---

## 附录: 超参数速查表

### 预训练 (Version A, 300M)

```yaml
# 模型
n_layers: 24
d_model: 1024
n_heads: 16
n_kv_heads: 4
d_ff: 2730
vocab_size: 32000
max_seq_len: 2048

# 优化
optimizer: AdamW
lr: 3e-4
lr_schedule: cosine
warmup_steps: 2000
weight_decay: 0.1
grad_clip: 1.0
adam_beta1: 0.9
adam_beta2: 0.95

# 训练
micro_batch_size: 8
gradient_accumulation: 4
precision: bf16
gradient_checkpointing: true
total_tokens: 6_000_000_000
```

### SFT

```yaml
lr: 2e-5
epochs: 2-3
batch_size: 128
weight_decay: 0.0
# LoRA (可选)
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_targets: [q_proj, k_proj, v_proj, o_proj]
```

### DPO

```yaml
lr: 5e-7
beta: 0.1
epochs: 1-3
batch_size: 32
```

### 预训练 (Version B, 1B)

```yaml
# 模型
n_layers: 32
d_model: 2048
n_heads: 32
n_kv_heads: 8
d_ff: 5460
vocab_size: 64000
max_seq_len: 4096

# 优化（与 Version A 类似，调整 batch size）
lr: 3e-4
micro_batch_size: 4          # 每卡更小
gradient_accumulation: 4
n_gpus: 4                    # 多卡
# 有效 batch size = 4 × 4096 × 4 × 4 = 262,144 tokens/step
total_tokens: 20_000_000_000  # Chinchilla 最优 ≈ 20×1B
```
