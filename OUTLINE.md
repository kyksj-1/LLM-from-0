# LLM 从零到一完整教程 — 总体规划大纲

> **目标读者**: 具备反向传播、神经网络、微积分、线性代数、概率统计基础，同时渴望提高编程和工程能力的大学生
> **技术主线**: Google · DeepSeek · Anthropic 三条线贯穿
> **文档结构**: 每个模块 `README.md`（核心理论+代码+项目） + `advanced.md`（前沿工业实践）
> **代码规范**: PyTorch 为主，按逻辑职责拆分为独立 `.py` 文件，放置于 `code/{模块名}/`
> **项目设计**: 每模块 3-4 个项目，难度梯度递进（入门→进阶→挑战），开放式（只提供思路/流程/伪代码/关键片段）

---

## 目录

- [模块 0: 序章 — LLM 全景概览](#模块-0-序章--llm-全景概览)
- [模块 1: Tokenization — 分词与词汇表构建](#模块-1-tokenization--分词与词汇表构建)
- [模块 2: Embedding — 词嵌入与位置编码](#模块-2-embedding--词嵌入与位置编码)
- [模块 3: Transformer — 自注意力与架构设计](#模块-3-transformer--自注意力与架构设计)
- [模块 4: Decoder-Only 架构 — GPT/Llama/Gemma](#模块-4-decoder-only-架构--gptllamagemma)
- [模块 5: 注意力机制进阶 — MHA/MQA/GQA/MLA](#模块-5-注意力机制进阶--mhamqagqamla)
- [模块 6: MoE — 混合专家模型](#模块-6-moe--混合专家模型)
- [模块 7: 数据工程 — 预训练数据管线](#模块-7-数据工程--预训练数据管线)
- [模块 8A: 预训练（上）— 目标函数与理论基础](#模块-8a-预训练上--目标函数与理论基础)
- [模块 8B: 预训练（中）— Scaling Laws 与计算最优](#模块-8b-预训练中--scaling-laws-与计算最优)
- [模块 8C: 预训练（下）— 训练工程与实战](#模块-8c-预训练下--训练工程与实战)
- [模块 9: 分布式训练 — 并行策略与工程优化](#模块-9-分布式训练--并行策略与工程优化)
- [模块 10: SFT — 监督微调与参数高效微调](#模块-10-sft--监督微调与参数高效微调)
- [模块 11: RLHF — 基于人类反馈的强化学习](#模块-11-rlhf--基于人类反馈的强化学习)
- [模块 12: DPO 及变体 — 直接偏好优化](#模块-12-dpo-及变体--直接偏好优化)
- [模块 13: CoT 与推理 — 思维链与测试时计算](#模块-13-cot-与推理--思维链与测试时计算)
- [模块 14: 推理加速 — KV Cache/量化/系统优化](#模块-14-推理加速--kv-cache量化系统优化)
- [模块 15: RAG 与知识增强 — 检索、向量库与 GraphRAG](#模块-15-rag-与-知识增强--检索向量库与-graphrag)
- [模块 16: 前沿专题 — 可解释性/安全/多模态](#模块-16-前沿专题--可解释性安全多模态)
- [终极项目: 从零训练一个完整 LLM](#终极项目-从零训练一个完整-llm)

---

## 模块 0: 序章 — LLM 全景概览

### README.md

#### 1. 语言模型简史
- n-gram → 神经语言模型（Bengio 2003）→ RNN/LSTM → Attention → Transformer
- 里程碑时间线（Mermaid 时间轴图）
- 从"预测下一个词"到"涌现智能"的范式转变

#### 2. 三大架构范式
- Encoder-only（BERT）、Encoder-Decoder（T5）、Decoder-only（GPT）
- 各架构的适用场景与优劣对比（Mermaid 对比表）
- 为什么 Decoder-only 成为主流？数学与工程视角的双重解释

#### 3. 三条技术主线
- **Google**: Transformer → BERT → T5 → PaLM → Gemini/Gemma
  - 开创性贡献：Attention Is All You Need、Scaling Laws的早期探索
- **DeepSeek**: DeepSeek-V1 → V2(MLA+MoE) → V3(FP8+DualPipe) → R1(推理)
  - 核心创新：以极致工程效率追赶前沿
- **Anthropic**: Claude 1 → 2 → 3 → 3.5 → 4
  - 核心理念：安全优先（Constitutional AI）、可解释性（Mechanistic Interpretability）
  - 关键论文：RLHF原始论文（Christiano et al. 2017, 多位Anthropic创始成员参与）、Constitutional AI、Scaling Monosemanticity

#### 4. 学习路径总览
- 模块依赖关系图（Mermaid 有向图）
- 各模块预计学习时间与GPU需求
- 先修知识自检清单（含自测题）

### advanced.md

#### 1. LLM 行业格局深度分析
- 开源 vs 闭源的博弈
- Scaling Laws 的经济学视角
- 各公司技术路线的哲学差异

#### 2. Anthropic 的安全优先路线
- 从 OpenAI 分裂的历史背景
- "Race to the top" 理念
- 安全研究如何反哺能力提升

#### 3. 预训练范式的演进方向
- 从 Scaling Laws 到 Inference-time Compute Scaling
- 合成数据的兴起与风险
- 多模态统一架构趋势

---

## 模块 1: Tokenization — 分词与词汇表构建

### README.md

#### 1. 为什么需要分词
- 文本 → 数字的桥梁
- 分词粒度的权衡：字符级 vs 词级 vs 子词级（Mermaid 对比图）
- 词汇表大小 vs 序列长度的 trade-off 数学分析

#### 2. BPE 算法（Byte Pair Encoding）
- 算法流程（Mermaid 流程图）
- 完整数学推导：频率统计 → 贪心合并 → 编码长度最小化
- 时间复杂度分析：O(N × V) 其中 N=语料大小, V=目标词汇量
- 手动推演示例（step-by-step）

#### 3. WordPiece 算法
- 与 BPE 的核心区别：似然值最大化 vs 频率最大化
- 数学形式：选择使 log P(corpus) 增加最大的合并对
- $\Delta \log P = \log \frac{P(xy)}{P(x)P(y)}$ — 互信息的直观解释
- Google BERT 中的应用

#### 4. Unigram Language Model
- 自顶向下 vs BPE 的自底向上（Mermaid 对比流程图）
- EM 算法求解分词概率
- Viterbi 解码：最优分词路径的动态规划
- 损失函数推导：$L = -\sum_{x \in D} \log P(x | V)$

#### 5. SentencePiece 与 Tiktoken
- SentencePiece：语言无关的统一框架
- Tiktoken：OpenAI 的高效实现，cl100k_base 编码方案
- Byte-level BPE：GPT-2 的创新——彻底消除 OOV

#### 6. 现代 LLM 分词器对比
- GPT-4 / Gemma / Claude / DeepSeek 的分词策略对比表
- 多语言分词的挑战与解决方案
- 词汇表大小选择的工程经验

### advanced.md

#### 1. Google 的分词演进
- WordPiece（BERT）→ SentencePiece（T5）→ Gemma 的 256K 词汇表
- 多语言分词的 Google 实践

#### 2. DeepSeek 的分词策略
- DeepSeek 的中英混合分词优化
- 大词汇表（100K+）的工程权衡

#### 3. Anthropic 视角
- Claude 的分词策略（基于公开信息推断，标注推测部分）
- Tokenization 对安全性的影响：prompt injection 与分词边界
- 分词粒度对模型可解释性的影响

#### 4. 前沿话题
- Token-free models 的探索（ByT5, MegaByte）
- 动态词汇表与自适应分词
- 分词对下游任务性能的量化影响

### 代码目录 `code/tokenization/`

```
code/tokenization/
├── bpe_tokenizer.py          # [已完成] BPE + ByteBPE 完整实现
├── wordpiece_tokenizer.py    # WordPiece 算法实现
├── unigram_tokenizer.py      # Unigram LM + Viterbi 解码实现
├── tokenizer_comparison.py   # 各分词器的对比分析工具
└── utils.py                  # 预分词、文本清洗等工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 手动实现 BPE 并在小语料上训练 | ⭐ 入门 | 完整代码 + 详细注释 | 理解 BPE 合并过程 |
| 2 | 对比 BPE/WordPiece/Unigram 的分词效果 | ⭐⭐ 进阶 | 思路 + 评估指标设计 + 关键代码片段 | 理解三种算法的差异 |
| 3 | 训练一个中英混合分词器（10K词汇量） | ⭐⭐⭐ 挑战 | 思路 + 数据处理伪代码 + 评估方法 | 实战多语言分词 |
| 4 | 分析 GPT-2 tokenizer 的字节级编码 | ⭐⭐ 进阶 | 研究思路 + 分析框架 | 理解工业级分词器 |

---

## 模块 2: Embedding — 词嵌入与位置编码

### README.md

#### 1. 从 One-Hot 到稠密表示
- One-Hot 的维度灾难与语义缺失
- 分布式假说："You shall know a word by the company it keeps"
- 嵌入空间的几何直觉（Mermaid 示意图）

#### 2. Word2Vec
- **Skip-Gram**: 给定中心词预测上下文
  - 目标函数推导：$J = -\frac{1}{T}\sum_{t=1}^{T}\sum_{-c \le j \le c, j \ne 0} \log P(w_{t+j}|w_t)$
  - Softmax 瓶颈与计算复杂度分析
- **CBOW**: 给定上下文预测中心词
- **Negative Sampling**: 近似优化
  - 数学推导：从 Softmax 到二分类
  - 噪声分布选择：$P_n(w) \propto f(w)^{3/4}$ 的理论依据
- **层次 Softmax**: Huffman 树加速

#### 3. GloVe
- 共现矩阵与统计信息
- 目标函数：$J = \sum_{i,j} f(X_{ij})(w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij})^2$
- 加权函数 $f(x)$ 的设计动机
- Word2Vec 与 GloVe 的数学等价性分析

#### 4. 位置编码（Positional Encoding）
- **为什么需要位置信息**: Transformer 的排列不变性证明
- **正弦余弦编码**:
  - 公式：$PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$
  - 关键性质证明：相对位置可通过线性变换表示
  - 外推能力分析
- **RoPE（旋转位置编码）**:
  - 从绝对位置到相对位置的动机
  - 完整推导：二维旋转 → 高维推广
  - 核心定理证明：$\langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m-n)$
  - 远距离衰减特性的数学解释
- **ALiBi（Attention with Linear Biases）**:
  - 无需可学习参数的位置编码
  - 线性偏置的数学形式与外推优势
- **YaRN**: RoPE 的长度外推扩展
  - NTK-aware 插值的数学原理
  - 动态 NTK 缩放

#### 5. 可学习位置编码
- 绝对可学习 PE（GPT-2 风格）
- 优势与局限性对比

### advanced.md

#### 1. Google 的位置编码演进
- 正弦编码（原始Transformer）→ 相对位置编码（T5）→ RoPE（PaLM/Gemma）
- T5 的 relative position bias 设计
- Gemma 2 中 RoPE 的具体配置

#### 2. DeepSeek 的位置编码实践
- DeepSeek-V2 中 RoPE 的使用细节
- 与 MLA（Multi-head Latent Attention）的协同设计
- 长上下文扩展策略

#### 3. Anthropic 视角
- Claude 的上下文窗口演进（8K → 100K → 200K）
- 长上下文能力的位置编码支撑（基于公开信息分析）
- 嵌入空间与可解释性：Superposition Hypothesis

#### 4. 前沿话题
- 上下文窗口的理论极限
- 无限上下文的探索方向（Infini-Attention, Ring Attention）
- 嵌入空间的几何结构研究

### 代码目录 `code/embedding/`

```
code/embedding/
├── word2vec.py               # Skip-Gram + CBOW + Negative Sampling 实现
├── glove.py                  # GloVe 简化实现
├── positional_encoding.py    # 正弦编码 + RoPE + ALiBi 统一实现
├── rope.py                   # RoPE 的详细实现与可视化
├── visualize.py              # 嵌入空间可视化工具
└── utils.py                  # 工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 在小语料上训练 Word2Vec 并可视化 | ⭐ 入门 | 完整代码 + 可视化模板 | 理解词向量的语义空间 |
| 2 | 实现并对比正弦PE/RoPE/ALiBi | ⭐⭐ 进阶 | 关键代码 + 实验设计思路 | 理解位置编码的差异 |
| 3 | 验证 RoPE 的远距离衰减特性 | ⭐⭐ 进阶 | 实验设计 + 数学推导提示 | 深入理解 RoPE 数学 |
| 4 | 实现 YaRN 长度外推扩展 | ⭐⭐⭐ 挑战 | 论文关键公式 + 伪代码 + 参考实现指引 | 理解上下文外推 |

---

## 模块 3: Transformer — 自注意力与架构设计

### README.md

#### 1. 自注意力机制（Self-Attention）
- 直觉：每个位置"关注"所有其他位置
- 数学推导：
  - Query-Key-Value 分解的动机
  - $\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
  - **缩放因子证明**: 当 $d_k$ 较大时，$q \cdot k$ 的方差为 $d_k$，除以 $\sqrt{d_k}$ 使方差归一
  - 注意力权重的概率解释
- 因果掩码（Causal Mask）的必要性与实现（Mermaid 示意图）
- 计算复杂度分析：$O(n^2 d)$

#### 2. 多头注意力（Multi-Head Attention）
- 多头的动机：在不同子空间捕获不同模式
- 数学形式：$\text{MultiHead}(Q,K,V) = \text{Concat}(head_1,...,head_h)W^O$
- 参数量分析与计算量分析
- 头数选择的经验法则

#### 3. 归一化层
- **LayerNorm**:
  - 公式：$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma + \epsilon} + \beta$
  - 为什么不用 BatchNorm？序列长度变化与自回归的限制
- **RMSNorm**:
  - 公式：$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma$
  - 去除均值中心化的理论依据
  - 计算效率对比
- **Pre-Norm vs Post-Norm**:
  - 训练稳定性分析（Mermaid 梯度流图）
  - 为什么现代模型几乎都用 Pre-Norm

#### 4. 前馈网络（FFN）
- 标准 FFN：$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2$
- 激活函数演进：ReLU → GELU → SwiGLU
- **SwiGLU** 的数学形式与优势：
  - $\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xV) W_2$
  - 门控机制的直觉
  - 参数量调整：隐藏层维度 $\frac{2}{3} \times 4d$ 保持参数量不变

#### 5. 完整 Transformer Block
- 残差连接的梯度分析
- 完整数据流（Mermaid 详细架构图）
- Decoder-only Transformer 的完整实现

#### 6. Google Transformer 架构演进
- 原始 Transformer（2017）→ 改进（Pre-Norm, RMSNorm, SwiGLU）
- PaLM 的架构选择
- Gemma/Gemma 2 的具体配置

### advanced.md

#### 1. Google 的 Transformer 研究
- Transformer 架构搜索（Evolved Transformer）
- PaLM 的并行 Attention+FFN 设计
- Gemma 2 的 sliding window + global attention 交替

#### 2. DeepSeek 的架构创新
- DeepSeek-V2 的 Pre-Norm + 残差连接变体
- 架构选择背后的工程考量

#### 3. Anthropic 视角
- Transformer 可解释性：注意力头的功能分类
  - Induction Heads 的发现（Olsson et al. 2022）
  - 注意力头如何实现算法（如拷贝、排序）
- Superposition 假说：神经元如何编码超过维度数的特征
- 对 Transformer 行为的 Circuits 分析

#### 4. 前沿话题
- 亚二次注意力的理论探索（Linear Attention, State Space Models）
- Mamba 与 Transformer 的对比
- Transformer 的理论表达能力分析

### 代码目录 `code/transformer/`

```
code/transformer/
├── attention.py              # Self-Attention, Multi-Head Attention, Causal Mask
├── normalization.py          # LayerNorm, RMSNorm
├── feedforward.py            # FFN, SwiGLU
├── block.py                  # TransformerBlock（组装以上组件）
├── model.py                  # 完整 Decoder-only Transformer + generate()
└── utils.py                  # 工具函数（参数初始化、掩码生成等）
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 手动计算一个注意力矩阵 | ⭐ 入门 | 完整代码 + step-by-step 演示 | 直观理解注意力 |
| 2 | 实现完整 Transformer Block | ⭐⭐ 进阶 | 组件代码 + 组装指引 | 掌握架构搭建 |
| 3 | 对比 Pre-Norm/Post-Norm 的训练稳定性 | ⭐⭐ 进阶 | 实验设计 + 监控指标 + 关键代码 | 理解归一化的影响 |
| 4 | 在 Shakespeare 数据集上训练字符级 GPT | ⭐⭐⭐ 挑战 | 数据处理思路 + 训练框架伪代码 | 端到端训练体验 |

---

## 模块 4: Decoder-Only 架构 — GPT/Llama/Gemma

### README.md

#### 1. 自回归语言模型
- 联合概率的链式分解：$P(x_1,...,x_T) = \prod_{t=1}^T P(x_t | x_{<t})$
- Teacher Forcing 训练策略
- 自回归生成的数学框架

#### 2. GPT 系列架构
- GPT-1: 预训练 + 微调范式的开创
- GPT-2: Zero-shot 能力的涌现
- GPT-3: Few-shot Learning 与 In-Context Learning
- 架构细节对比表

#### 3. Llama 系列架构
- Llama 1/2/3 的架构演进
- 关键设计选择：RoPE + RMSNorm + SwiGLU + GQA
- 与 GPT 系列的差异对比（Mermaid 对比图）
- 各版本的具体超参数配置

#### 4. Gemma 系列架构（Google）
- Gemma 1/2 的设计哲学
- Multi-Query Attention 与 Grouped-Query Attention
- 与 Llama 的架构差异
- 开源策略与训练数据规模

#### 5. 模型配置与超参数
- 层数、头数、隐藏维度的关系
- 参数量估算公式：$P \approx 12 l d^2$（推导过程）
- FLOPs 估算：$C \approx 6PD$（Kaplan et al.）
- 不同规模模型的配置参考

#### 6. 完整实现
- 从零搭建一个 mini-GPT
- 配置类设计（dataclass）
- 模型初始化策略
- 文本生成：Greedy / Top-k / Top-p / Temperature

### advanced.md

#### 1. Google: PaLM → Gemini → Gemma
- PaLM 的 540B 参数与 Pathways 系统
- Gemini 的多模态统一架构
- Gemma 的开源设计权衡

#### 2. DeepSeek 的架构选择
- DeepSeek-V1 的基础架构
- 从标准 MHA 到 MLA 的演进动机
- 超参数搜索的工程方法论

#### 3. Anthropic 视角
- Claude 的架构演进（基于公开信息）
- 安全性约束如何影响架构设计
- Constitutional AI 如何在架构层面实现

#### 4. 前沿话题
- 深窄 vs 宽浅网络的 Scaling 行为
- 架构搜索自动化
- Sub-quadratic 模型（Mamba, RWKV, Griffin）的工业化进展

### 代码目录 `code/decoder_only/`

```
code/decoder_only/
├── config.py                 # 模型配置（GPT/Llama/Gemma 风格）
├── model.py                  # 完整 Decoder-only 模型
├── generation.py             # 文本生成策略（Greedy/Top-k/Top-p/Beam Search）
├── tokenizer_wrapper.py      # 分词器封装
├── train_simple.py           # 简单训练脚本
└── utils.py                  # 工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 搭建一个可配置的 mini-GPT | ⭐ 入门 | 完整代码框架 | 理解 Decoder-only 结构 |
| 2 | 对比 Greedy/Top-k/Top-p 生成效果 | ⭐⭐ 进阶 | 生成函数代码 + 评估思路 | 理解采样策略 |
| 3 | 复现 GPT-2 Small (124M) 的架构 | ⭐⭐⭐ 挑战 | 架构参数表 + 权重加载指引 | 理解工业级模型配置 |
| 4 | 分析 Llama 与 GPT 的参数效率差异 | ⭐⭐ 进阶 | 分析框架 + FLOPs 计算公式 | 理解架构设计权衡 |

---

## 模块 5: 注意力机制进阶 — MHA/MQA/GQA/MLA

### README.md

#### 1. 标准多头注意力（MHA）回顾
- 参数量：$4d^2$（$W_Q, W_K, W_V, W_O$）
- KV Cache 的显存占用：$2 \times n_{layers} \times n_{heads} \times d_{head} \times seq\_len$
- 推理瓶颈分析：为什么 KV Cache 成为主要开销

#### 2. Multi-Query Attention (MQA)
- 核心思想：所有 Query 头共享一组 KV
- 数学形式与参数量对比
- 显存节省分析：KV Cache 缩减为 $\frac{1}{h}$
- 质量-效率 trade-off 的实验结论

#### 3. Grouped-Query Attention (GQA)
- MQA 和 MHA 的折中方案（Mermaid 对比图）
- 分组策略：$G$ 组 KV 头对应 $H$ 个 Query 头
- Llama 2 中的具体配置
- 从 MHA checkpoint 转换为 GQA 的方法

#### 4. Multi-head Latent Attention (MLA) — DeepSeek
- 核心创新：低秩压缩 KV（Mermaid 架构图）
- 数学推导：
  - KV 联合压缩：$c_{KV} = W_{DKV} x$，其中 $d_c \ll d_{head} \times n_{heads}$
  - 解压缩：$K = W_{UK} c_{KV}$, $V = W_{UV} c_{KV}$
  - 与 RoPE 的兼容性处理
- KV Cache 压缩比分析
- 训练稳定性保证

#### 5. 统一对比分析
- MHA / MQA / GQA / MLA 的完整对比表
  - 参数量、KV Cache 大小、推理速度、模型质量
- 各机制适用场景指南
- 历史演进的技术逻辑（Mermaid 演进图）

#### 6. 高效训练数据策略
- Sequence Packing (序列打包/拼接):
  - 动机：消除 Padding 带来的计算浪费（从物理角度看，即消除无效功）
  - 算法实现：Best-fit Bin Packing 算法变体
  - Attention Mask 的处理：Block-diagonal Masking（对角分块掩码）的数学形式
  - 对收敛性的影响分析

### advanced.md

#### 1. Google 的注意力机制演进
- MHA（原始Transformer）→ MQA（PaLM）→ GQA（Gemma 2）
- 各阶段的工程权衡

#### 2. DeepSeek MLA 深度分析
- MLA 的完整数学推导
- 与 LoRA 思想的联系
- 低秩近似的理论基础（Johnson-Lindenstrauss 引理）
- 工程实现细节与优化

#### 3. Anthropic 视角
- 注意力机制的可解释性研究
- 注意力模式的功能分类（Anthropic 的 Circuits 研究）
- 不同注意力变体对可解释性的影响

#### 4. 前沿话题
- Ring Attention：超长序列的分布式注意力
- Differential Attention
- 注意力的信息论分析

### 代码目录 `code/attention_variants/`

```
code/attention_variants/
├── mha.py                    # 标准 Multi-Head Attention
├── mqa.py                    # Multi-Query Attention
├── gqa.py                    # Grouped-Query Attention
├── mla.py                    # Multi-head Latent Attention (DeepSeek)
├── kv_cache.py               # KV Cache 实现与管理
├── benchmark.py              # 性能基准测试工具
└── visualize_attention.py    # 注意力模式可视化
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 实现并对比 MHA/MQA/GQA | ⭐⭐ 进阶 | 关键代码 + 基准测试框架 | 理解三种注意力的差异 |
| 2 | 实现 MLA 的低秩压缩 | ⭐⭐⭐ 挑战 | 数学推导 + 核心代码片段 | 深入理解 MLA |
| 3 | 基准测试 KV Cache 的显存与速度 | ⭐⭐ 进阶 | 测试框架 + 指标定义 | 量化各方案的工程差异 |
| 4 | 将 MHA 模型转换为 GQA（模拟 uptraining） | ⭐⭐⭐ 挑战 | 转换策略伪代码 + 参考论文 | 理解注意力转换工程 |

---

## 模块 6: MoE — 混合专家模型

### README.md

#### 1. MoE 的核心思想
- 稀疏激活：不是所有参数都参与每次计算（Mermaid 示意图）
- 条件计算的数学框架
- MoE 的历史：从 Jacobs(1991) 到现代 Transformer MoE

#### 2. 路由机制（Gating / Router）
- Top-K 路由：$G(x) = \text{TopK}(\text{softmax}(W_g x))$
- 负载均衡问题：
  - 为什么会出现"赢者通吃"？
  - 辅助损失（Auxiliary Loss）：$L_{aux} = \alpha \sum_i f_i P_i$
  - 负载均衡损失的完整推导
- Expert Choice 路由（Google）

#### 3. DeepSeekMoE 架构
- 细粒度专家（Fine-grained Experts）
  - 将标准专家拆分为更多小专家的动机
  - 数学分析：专家数量 vs 专家容量的 trade-off
- 共享专家（Shared Experts）
  - 处理通用知识的专家不参与路由
  - 架构设计（Mermaid 图）
- 完整前向传播公式

#### 4. 训练挑战
- 路由坍塌（Router Collapse）
- 专家利用率不均
- 训练不稳定性与解决方案
- Token dropping 策略

#### 5. MoE 的参数与计算效率分析
- 总参数量 vs 激活参数量
- FLOPs 对比：Dense vs MoE（相同激活参数量）
- MoE 模型的 Scaling Laws

### advanced.md

#### 1. Google 的 MoE 研究
- Switch Transformer（Top-1 路由）
- GShard 的分布式 MoE
- Expert Choice Routing
- Gemini 中的 MoE 应用（推测+公开信息）

#### 2. DeepSeek MoE 深度分析
- DeepSeek-V2: 160 专家，每次激活 6
- DeepSeek-V3: 256 专家，每次激活 8
- 辅助损失 free 策略的创新
- 专家并行的工程实现

#### 3. Anthropic 视角
- MoE 模型的可解释性挑战
- 专家是否学到了语义上有意义的分工？
- 稀疏模型的安全性考量

#### 4. 前沿话题
- Soft MoE（可微路由）
- MoE + MLA 的协同设计
- 专家蒸馏：MoE → Dense
- MoE 推理的负载均衡问题

### 代码目录 `code/moe/`

```
code/moe/
├── router.py                 # Top-K Router, Expert Choice Router
├── expert.py                 # 标准专家, 细粒度专家, 共享专家
├── moe_layer.py              # MoE 层完整实现
├── auxiliary_loss.py         # 负载均衡辅助损失
├── moe_transformer.py        # MoE Transformer 完整模型
└── analysis.py               # 专家利用率分析工具
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 实现 Top-2 路由的 MoE 层 | ⭐⭐ 进阶 | 完整代码 + 可视化 | 理解 MoE 基本机制 |
| 2 | 实现负载均衡辅助损失 | ⭐⭐ 进阶 | 数学推导 + 关键代码 | 理解路由优化 |
| 3 | 实现 DeepSeekMoE 的细粒度+共享专家 | ⭐⭐⭐ 挑战 | 架构设计伪代码 + 论文关键公式 | 理解前沿 MoE 设计 |
| 4 | 对比 Dense vs MoE 在相同 FLOPs 下的性能 | ⭐⭐⭐ 挑战 | 实验设计 + 评估框架 | 量化 MoE 的效率优势 |

---

## 模块 7: 数据工程 — 预训练数据管线

### README.md

#### 1. 数据是 LLM 的燃料
- 数据质量 vs 数据数量的关系
- Chinchilla Scaling Laws 对数据量的启示
- 现代 LLM 训练数据规模概览（Mermaid 时间轴）

#### 2. 数据收集
- Common Crawl 与 Web 数据的获取
- 书籍、论文、代码等高质量数据源
- 数据许可与法律考量

#### 3. 数据清洗
- 去重：
  - 精确去重（哈希）
  - 近似去重（MinHash + LSH）— 完整算法推导
  - Jaccard 相似度的无偏估计证明
- 质量过滤：
  - 启发式规则（长度、语言检测、特殊字符比例）
  - 基于模型的质量评分（perplexity 过滤）
  - 有害内容过滤
- 文本提取与清洗
  - HTML → 纯文本的处理流程
  - PDF/图片中的文本提取

#### 4. 数据混合（Data Mixing）
- 不同数据源的配比策略
- 领域权重的设置原则
- 动态数据混合 vs 静态配比
- 课程学习（Curriculum Learning）

#### 5. 数据处理管线
- 端到端流程（Mermaid 流程图）
- 分布式数据处理框架
- 数据格式：JSON Lines, Parquet, Arrow

### advanced.md

#### 1. Google 的数据工程
- C4 数据集的构建方法论
- PaLM 的数据混合策略
- Gemma 的数据处理细节

#### 2. DeepSeek 的数据策略
- DeepSeek 的中英文数据配比
- 代码数据的处理方法
- 数据质量评估的迭代优化

#### 3. Anthropic 视角
- HH-RLHF 数据集的设计理念
- 安全数据筛选的方法论
- 合成数据在安全训练中的应用

#### 4. 前沿话题
- 合成数据的规模化生产（Textbooks Are All You Need）
- 数据质量的量化评估方法
- 数据飞轮：模型生成 → 数据筛选 → 再训练
- 数据污染检测

### 代码目录 `code/data_engineering/`

```
code/data_engineering/
├── deduplication.py          # MinHash + LSH 近似去重
├── quality_filter.py         # 质量过滤器（规则 + 模型）
├── text_extraction.py        # HTML/文本清洗
├── data_mixer.py             # 数据混合与采样
├── pipeline.py               # 端到端数据处理管线
└── analysis.py               # 数据集统计分析工具
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 实现 MinHash 近似去重 | ⭐⭐ 进阶 | 算法推导 + 完整代码 | 理解去重算法 |
| 2 | 构建一个小型数据清洗管线 | ⭐⭐ 进阶 | 管线框架 + 各阶段思路 | 理解数据工程流程 |
| 3 | 分析数据混合比例对模型质量的影响 | ⭐⭐⭐ 挑战 | 实验设计 + 评估指标 | 理解数据配比的重要性 |
| 4 | 实现基于 perplexity 的质量过滤 | ⭐⭐ 进阶 | 思路 + 参考论文 + 关键代码 | 理解模型驱动的数据筛选 |

---

## 模块 8A: 预训练（上）— 目标函数与理论基础

### README.md

#### 1. 语言模型的训练目标：Next Token Prediction
- 交叉熵损失：$L = -\frac{1}{T}\sum_{t=1}^T \log P(x_t | x_{<t})$
- 与 perplexity 的关系：$\text{PPL} = \exp(L)$
- Teacher Forcing 与自回归训练的数学框架
- Softmax 瓶颈：当词汇表巨大时的计算挑战

#### 2. 为什么 NTP 足以产生智能？
- 信息论视角：预测下一个词 ≈ 压缩数据
- 压缩即智能（Compression = Intelligence）：
  - Solomonoff 归纳推理的直觉
  - Kolmogorov 复杂性与最短描述长度（MDL）
  - Hutter Prize 的启示：压缩能力与智能的正相关
- Shannon 熵与语言的可预测性
- 从"统计鹦鹉"到"涌现理解"的哲学讨论

#### 3. 其他预训练目标
- **MLM（Masked Language Modeling）— BERT 系列**:
  - 随机遮蔽 15% token 的策略（80/10/10 规则）
  - 双向上下文的优势与自回归不兼容的劣势
  - 数学形式：$L_{MLM} = -\sum_{i \in M} \log P(x_i | x_{\backslash M})$
- **Prefix LM — T5/PaLM 系列**:
  - 前缀部分双向注意力 + 后缀部分因果注意力
  - 统一 NLU 与 NLG 的数学框架
- **UL2（Unifying Language Learning Paradigms）**:
  - 混合去噪器（Mixture-of-Denoisers）
  - R-Denoiser / S-Denoiser / X-Denoiser 的设计
  - Mode Switching Token 的创新
- **Fill-in-the-Middle（FIM）**:
  - 代码模型的关键目标（StarCoder, CodeLlama, DeepSeek-Coder）
  - PSM（Prefix-Suffix-Middle）vs SPM（Suffix-Prefix-Middle）格式
  - FIM rate 的选择（通常 50%-90%）
  - 数学形式与 NTP 的兼容性
- **去噪目标（Denoising Objectives）**:
  - Span Corruption（T5 风格）
  - 噪声类型：删除、替换、插入、排列
- **各目标的统一对比表（Mermaid 表格）**:
  - 上下文方向、适用任务、训练效率、与 NTP 的兼容性

#### 4. Perplexity 的理论与实践
- PPL 的信息论含义：每 token 的平均选择数（$2^H$）
- PPL 的局限性：不同分词器下不可直接比较
- **Bits-per-byte（BPB）**: 更公平的跨分词器度量
  - 计算方法：$\text{BPB} = \frac{L \times T}{\text{total\_bytes}} / \ln 2$
- Perplexity 与下游任务性能的关系：近似线性？

#### 5. 损失函数设计细节
- **Label Smoothing**:
  - 公式：$L_{LS} = (1-\alpha) L_{CE} + \alpha L_{uniform}$
  - 正则化效果与校准性改善
- **Z-loss（PaLM 使用）**:
  - Logit 归一化惩罚：$L_z = c \cdot \log^2 Z$
  - 防止 logit 漂移的工程意义
- **Auxiliary Losses 在预训练中的角色**:
  - MoE 负载均衡损失（回顾模块 6，衔接）
  - 路由器 z-loss

#### 6. 三条技术线的预训练目标选择
- **Google**: BERT(MLM) → T5(Prefix LM + Span Corruption) → PaLM(NTP) → Gemma(NTP)
  - 为什么最终回归 NTP？工程简洁性与 Scaling 表现
- **DeepSeek**: NTP 为主 + FIM 用于代码能力 + 辅助损失 free
- **Anthropic**: 预训练目标与安全性的关系，安全数据比例的考量

### advanced.md

#### 1. Google 的预训练目标研究
- 从 BERT 到 UL2 的探索历程
- PaLM 的 Z-loss 实践经验
- Gemini 的多模态预训练目标

#### 2. DeepSeek 的预训练目标策略
- FIM 在 DeepSeek-Coder 中的应用
- 代码预训练与通用预训练的目标设计差异
- DeepSeek-V3 的预训练目标配置

#### 3. Anthropic 视角
- 预训练阶段的安全性考量
- 目标函数设计如何影响模型的"价值观"
- Anthropic 对预训练 vs 后训练的安全投入分配

#### 4. 前沿话题
- 多模态预训练目标（图文交织训练）
- 预训练目标的自动搜索
- Token 级别的自适应损失加权
- 预训练目标与模型涌现能力的关系

### 代码目录 `code/pretraining_objectives/`

```
code/pretraining_objectives/
├── ntp_loss.py               # NTP 损失 + Label Smoothing + Z-loss
├── mlm_loss.py               # MLM 损失实现
├── fim_loss.py               # Fill-in-the-Middle 实现
├── perplexity.py             # PPL / BPB 计算与对比工具
└── utils.py                  # 工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 实现 NTP + Label Smoothing + Z-loss 训练循环 | ⭐ 入门 | 完整代码 + 详细注释 | 理解预训练损失函数 |
| 2 | 对比 NTP/MLM/Prefix LM 在小语料上的表现 | ⭐⭐ 进阶 | 实验设计 + 关键代码 + 评估方法 | 理解不同目标的特性 |
| 3 | 实现 Fill-in-the-Middle 并验证代码补全能力 | ⭐⭐ 进阶 | 数据格式说明 + 核心代码片段 | 理解 FIM 在代码模型中的作用 |
| 4 | 分析 PPL 与 BPB 在不同分词器下的差异 | ⭐⭐⭐ 挑战 | 分析框架 + 数学推导提示 | 理解评估指标的深层含义 |

---

## 模块 8B: 预训练（中）— Scaling Laws 与计算最优

### README.md

#### 1. Scaling Laws 基础：幂律关系
- 经验发现：LLM 性能遵循可预测的幂律
- 幂律在物理学中的普遍性（类比直觉）
- 三个核心变量：参数量 N、数据量 D、计算量 C

#### 2. Kaplan et al.（2020）: OpenAI Scaling Laws
- 损失与 N 的关系：$L(N) = (N_c/N)^{\alpha_N}$，$\alpha_N \approx 0.076$
- 损失与 D 的关系：$L(D) = (D_c/D)^{\alpha_D}$，$\alpha_D \approx 0.095$
- 损失与 C 的关系：$L(C) = (C_c/C)^{\alpha_C}$
- 最优分配的 Lagrange 乘子法推导：
  - 固定 C 下，$N_{opt} \propto C^{0.73}$, $D_{opt} \propto C^{0.27}$
  - 结论："优先扩大模型，数据其次"
- 原始实验的关键设置与潜在问题

#### 3. Chinchilla（2022）: 修正 Scaling Laws
- Kaplan 的偏差分析：
  - 未训练至收敛 → 低估数据的重要性
  - 学习率调度依赖训练步数 → 引入混淆变量
- Chinchilla 三种估计方法：
  - 方法 1: 固定模型，变数据量
  - 方法 2: 固定 FLOPs，变 N/D 分配
  - 方法 3: 参数化损失函数拟合
- Chinchilla 最优：$N_{opt} \propto C^{0.5}$, $D_{opt} \propto C^{0.5}$
  - 核心修正：模型和数据应等比例扩大
- 对工业实践的深远影响：从"大模型少数据"到"平衡扩展"

#### 4. 过度训练（Over-training）策略
- **为什么要过度训练？**:
  - 推理成本 vs 训练成本的经济学分析
  - 小模型过度训练 → 推理更便宜
  - Llama 3 的实践：8B 模型用 15T tokens（远超 Chinchilla 最优）
- **过度训练的理论支撑**:
  - 损失在 Chinchilla 最优之后仍在下降，只是效率降低
  - 过度训练的 Scaling Laws：$L(N, rD_{opt})$ 的预测公式
- **何时选择过度训练？** 决策框架（Mermaid 决策图）

#### 5. MoE 模型的 Scaling Laws
- 总参数量 vs 激活参数量的 Scaling 行为
- 稀疏模型与稠密模型的 Scaling Laws 差异
- DeepSeek 的 MoE Scaling Laws 实验方法
  - 小规模 MoE 实验 → 预测大规模 MoE 性能
  - 专家数量、粒度、激活比例对 Scaling 的影响
- MoE 的"有效参数量"概念

#### 6. 涌现能力（Emergent Abilities）与相变
- 涌现能力的定义：小模型没有、大模型突然具备的能力
- 经典案例：算术推理、CoT、多步推理
- **争论：是真正的相变还是评估指标的假象？**
  - Schaeffer et al.（2023）: 非线性度量造成的错觉
  - 连续度量下能力可能是平滑增长
  - 当前共识与未解问题
- 下游任务的 Scaling Laws：Benchmark 性能 vs 预训练 loss

#### 7. 计算最优配置的工程实践
- **小模型实验 → 大模型预测**（DeepSeek 方法论）:
  - 实验设计：模型规模梯度（70M → 300M → 1B → 3B）
  - 拟合方法：最小二乘 + 贝叶斯优化
  - 预测的可靠性与置信区间
- **muP（Maximal Update Parameterization）**:
  - 核心思想：使不同宽度模型有相似的训练动力学
  - 从小模型调参 → 直接迁移到大模型
  - 数学基础：参数更新的规模不变性
  - 与标准参数化（SP）的对比
- **训练成本估算**:
  - FLOPs → GPU-hours → 美元的转换
  - $C \approx 6PD$ 的推导与适用条件
  - 实际 MFU（Model FLOPs Utilization）的影响

### advanced.md

#### 1. Google 的 Scaling 研究
- Kaplan → Chinchilla 的自我修正历程
- PaLM 的 Scaling 分析（540B 参数）
- Gemini 的 Scaling 策略（推测+公开信息）

#### 2. DeepSeek 的 Scaling Laws 实践
- DeepSeek-V2/V3 的小模型 proxy 实验
- MoE Scaling Laws 的实验细节
- 开源 Scaling Laws 数据的价值

#### 3. Anthropic 的 Scaling 贡献
- Anthropic 的 Scaling Laws 研究贡献
- "Scaling is predictable" 的方法论意义
- Scaling 预测在安全规划中的应用

#### 4. 前沿话题
- 超越幂律：是否存在 Scaling Laws 的极限？
- Inference-time Compute Scaling（与模块 13 CoT 衔接）
- Scaling Laws 的理论解释尝试（统计力学视角）
- 数据质量的 Scaling Laws：高质量数据是否有不同的 Scaling 曲线？
- Chinchilla 之后的 Scaling 研究综述

### 代码目录 `code/scaling_laws/`

```
code/scaling_laws/
├── scaling_laws.py           # Scaling Laws 拟合与预测工具
├── chinchilla_optimal.py     # Chinchilla 最优配置计算器
├── flops_calculator.py       # 模型 FLOPs / 训练成本估算
├── visualization.py          # Scaling 曲线可视化
└── utils.py                  # 工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 在小模型上拟合自己的 Scaling Laws 曲线 | ⭐⭐ 进阶 | 实验设计 + 拟合代码 + 可视化模板 | 理解缩放法则的经验性 |
| 2 | 实现 Chinchilla 最优计算器 | ⭐⭐ 进阶 | 数学推导 + 计算器框架代码 | 掌握计算最优分配 |
| 3 | 从 Scaling Laws 预测大模型性能 | ⭐⭐⭐ 挑战 | 数学推导 + 预测方法论 + 置信区间分析 | 掌握 Scaling Laws 工程应用 |
| 4 | 分析涌现能力：连续度量 vs 离散度量 | ⭐⭐⭐ 挑战 | 论文关键结论 + 实验设计思路 | 理解涌现的本质争论 |

---

## 模块 8C: 预训练（下）— 训练工程与实战

### README.md

#### 1. 优化器深入：AdamW
- **Adam 回顾**:
  - 一阶矩估计（动量）：$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
  - 二阶矩估计（自适应学习率）：$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$
  - 偏差修正：$\hat{m}_t = m_t / (1-\beta_1^t)$
  - 更新规则：$\theta_t = \theta_{t-1} - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$
- **AdamW：Weight Decay vs L2 正则化的本质区别**:
  - L2 正则化：将惩罚加入梯度 → 与自适应学习率耦合
  - Weight Decay：直接缩减权重 → 解耦（Mermaid 对比图）
  - 数学证明：在标准 SGD 下等价，在 Adam 下不等价
- **优化器状态的显存分析**:
  - Adam 状态：2× 模型参数（$m_t$ + $v_t$）
  - FP32 精度下：模型 $P$ 参数 → 优化器占 $8P$ 字节
  - 总显存公式（模型 + 梯度 + 优化器）
- **超参数选择原则**:
  - $\beta_1 = 0.9$（通用默认）
  - $\beta_2$：0.95（GPT-3）vs 0.999（原始 Adam）— 原因分析
  - $\epsilon$：1e-8（默认）vs 1e-6（某些训练更稳定）
  - Weight Decay：通常 0.1，与学习率的交互
- **其他优化器简介**:
  - Adafactor：节省显存（不存储完整二阶矩）
  - LION（符号梯度）：更新方向只取 sign
  - 8-bit Adam（bitsandbytes）：量化优化器状态

#### 2. 学习率调度策略
- **Warmup + Cosine Decay**:
  - Warmup 的理论依据：初始梯度不稳定
  - Cosine 衰减：$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\pi t / T))$
  - Warmup 步数的经验选择（通常 0.1%-1% 总步数）
- **WSD（Warmup-Stable-Decay）调度**:
  - MiniCPM / DeepSeek-V3 使用
  - 三阶段：Warmup → 恒定学习率 → 快速衰减
  - 优势：可灵活决定何时停止训练
  - 与 Cosine 调度的训练曲线对比
- **退火阶段（Cooldown / Annealing）**:
  - 训练末期切换高质量数据 + 降低学习率
  - Llama 3 的数据退火实践
  - 退火对最终性能的提升量化
- **学习率与 Batch Size 的关系**:
  - 线性缩放法则（Linear Scaling Rule）：$\eta \propto B$
  - 平方根缩放：$\eta \propto \sqrt{B}$
  - 临界批大小（Critical Batch Size）的概念

#### 3. 权重初始化
- **标准初始化方法**:
  - Xavier/Glorot 初始化：$W \sim U[-\sqrt{6/(n_{in}+n_{out})}, \sqrt{6/(n_{in}+n_{out})}]$
  - He/Kaiming 初始化：适配 ReLU 的方差分析
  - 截断正态分布初始化
- **残差分支的缩放**:
  - GPT-2 风格：输出投影乘以 $1/\sqrt{N_{layers}}$
  - 防止残差路径的方差随深度增长
- **muP 初始化**:
  - 使不同宽度模型有相似的训练动力学
  - 初始化规则 + 学习率规则的联合调整
  - 从小模型调参 → 直接迁移到大模型

#### 4. 梯度管理
- **梯度裁剪（Gradient Clipping）**:
  - L2 范数裁剪：$g = g \cdot \min(1, \frac{c}{\|g\|_2})$
  - 裁剪阈值的选择（通常 1.0）
  - 梯度裁剪的频率与训练稳定性
- **梯度累积**:
  - 模拟大 batch size：$g_{eff} = \frac{1}{K}\sum_{k=1}^K g_k$
  - 等价于大 batch 训练的数学证明
  - 实现注意事项（loss 的归一化位置）
  - （与模块 9 分布式训练衔接）
- **梯度范数监控**:
  - 梯度范数作为训练健康度指标
  - 突然增大 → 可能的训练不稳定
  - 各层梯度范数的分布分析

#### 5. Batch Size 策略
- 固定 batch size vs 逐步增大（Batch Size Warmup）
- 批大小与学习率的协同调整
- 临界批大小（Critical Batch Size）：
  - 定义：梯度噪声 vs 梯度信号的平衡点
  - $B_{crit} = B_{noise} / L$ 的推导
  - 低于临界大小 → 增大 B 几乎不浪费计算
  - 高于临界大小 → 增大 B 回报递减
- DeepSeek-V3 的 batch size 调度实践

#### 6. 训练稳定性专题
- **Loss Spikes 的根因分析**（Mermaid 决策树）:
  - **数据异常**: 异常样本、数据损坏、编码错误
  - **数值精度**: FP16 溢出（$>65504$）、梯度爆炸
  - **学习率过大**: 超出当前 loss landscape 的稳定区间
  - **批次方差**: 极端 mini-batch 的偶然影响
- **PaLM 的 Loss Spike 处理经验**:
  - 从 spike 前 100 步的 checkpoint 重启
  - 跳过导致 spike 的数据批次
  - 200-500 次 spike 的工程代价
- **训练发散的诊断流程**（Mermaid 决策树）:
  - Step 1: 检查梯度范数 → 是否爆炸？
  - Step 2: 检查 logit 范围 → 是否 NaN/Inf？
  - Step 3: 检查数据 → 是否有异常样本？
  - Step 4: 回退学习率 → 是否超出稳定范围？
- **数值精度与训练稳定性**:
  - FP16 vs BF16 vs FP32 的精度-稳定性权衡
  - BF16 的优势：动态范围大，不易溢出
  - FP8 训练（DeepSeek-V3）的额外挑战
  - （详细的混合精度训练实现见模块 9）
- **梯度爆炸/消失**:
  - 深度网络的梯度传播分析
  - Pre-Norm 的稳定性优势（回顾模块 3）
  - 残差连接对梯度流的保护作用

#### 7. 断点续训（Checkpointing）与容错
- **为什么需要 Checkpoint**:
  - 硬件故障是常态：训练数周的模型不能因单点故障重头再来
  - 预训练成本与容错的经济学分析
- **State Dict 的完整解剖**:
  - 模型权重（Model Weights）
  - 优化器状态（Optimizer States）：Momentum, Variance（占大部分显存）
  - 学习率调度器状态（LR Scheduler）
  - 随机数种子（RNG States）：确保数据增强与 Dropout 的可复现性
  - 数据加载器状态（Dataloader State）：已消费的 sample 位置
  - 训练步数 / epoch 计数
- **实现策略**:
  - 频率控制：按 Step 保存 vs 按时间保存
  - 轮换机制（Rotation）：保留最近 N 个 checkpoints
  - 原子写入（Atomic Write）：防止写入过程中崩溃导致文件损坏
  - 异步保存：不阻塞训练循环
- **断点重连（Fault Recovery）**:
  - 精确恢复：所有状态完全恢复 → 训练完全可重现
  - 数据重放的处理：恢复后不重复已消费数据
  - 分布式训练的 checkpoint 特殊处理（与模块 9 衔接）
- **实战代码**: `save_checkpoint()` / `load_checkpoint()` 的完整实现

#### 8. 训练监控与评估
- **损失曲线解读**:
  - 正常训练曲线的特征：平滑下降 + 逐渐减速
  - 异常模式：震荡、突升、平台期
  - 不同阶段的学习行为
- **梯度与权重监控**:
  - 梯度范数：各层分布、时间变化
  - 权重范数：是否持续增长？
  - 参数更新比率：$\|\Delta W\| / \|W\|$
- **训练中的评估**:
  - 验证集 PPL 的周期性计算
  - 下游任务 few-shot 评估（定期抽样）
  - 评估频率的工程权衡
- **工具实践**: WandB / TensorBoard 的最佳实践

#### 9. 多阶段预训练
- **标准预训练 → 继续预训练（Continual Pre-training）**:
  - 动机：领域适配（医疗、法律、金融）
  - 灾难性遗忘的风险与缓解策略
  - 数据混合：通用 + 领域的比例
- **长上下文扩展预训练**:
  - 从 4K → 128K 的上下文窗口扩展
  - RoPE 基底频率的调整（ABF）
  - 长文本数据的收集与质量控制
  - 训练配置：更小的 batch size、更多的梯度累积
- **数据退火（Data Annealing / Cooldown）**:
  - 训练末期切换高质量数据
  - 学习率同步降低
  - Llama 3 / MiniCPM 的退火实践
  - 退火数据的选择标准
- **代码能力的分阶段注入**:
  - 何时加入代码数据？预训练初期 vs 后期
  - 代码数据比例对通用能力的影响

#### 10. MoE 预训练的特殊工程挑战（与模块 6 衔接）
- **路由器训练不稳定性**:
  - 训练早期路由器的随机性
  - 路由器与专家的协同进化
- **负载均衡的动态调整**:
  - 辅助损失系数的调参策略
  - DeepSeek-V3 的辅助损失 free 策略
  - Token dropping 策略在预训练中的应用
- **MoE 模型的显存特殊性**:
  - 总参数量大 → 优化器状态的显存挑战
  - Expert Parallelism 的必要性（指向模块 9）
- **MoE 预训练的超参数选择**:
  - 专家数量、激活比例的配置
  - 与 Dense 模型超参数的差异

#### 11. 三条技术线的训练工程实践
- **Google**:
  - PaLM：6144 TPU v4 的训练经验
  - Loss spike 处理：跳过 + 重启策略
  - Gemma 的训练配置公开细节
- **DeepSeek**:
  - FP8 混合精度训练的工程突破（DeepSeek-V3）
  - DualPipe 计算-通信重叠（指向模块 9 详解）
  - MoE 训练的稳定性工程
- **Anthropic**:
  - 大规模训练的安全考量
  - 训练过程中的异常检测机制
  - 训练可重复性的工程保证

### advanced.md

#### 1. Google 的大规模训练工程
- PaLM 训练的全流程复盘
- Gemma 的训练配置与经验
- TPU 训练的特殊工程考量

#### 2. DeepSeek 的训练工程创新
- FP8 训练的完整技术细节
- MoE 训练的稳定性保障机制
- 多阶段训练的数据策略

#### 3. Anthropic 视角
- 训练过程中的安全监控
- 训练异常与模型行为的关联
- 大规模训练的可重复性挑战

#### 4. 前沿话题
- 弹性训练（Elastic Training）：节点动态加入/退出
- 在线学习与持续预训练
- 训练过程的可重复性研究
- 绿色 AI：训练的碳排放与能效优化

### 代码目录 `code/training_engineering/`

```
code/training_engineering/
├── optimizer.py              # AdamW 从零实现 + 8-bit Adam 简化版
├── lr_scheduler.py           # Cosine / WSD / 多阶段调度器
├── checkpointing.py          # 完整 checkpoint 保存/恢复实现
├── trainer.py                # 预训练训练器（整合以上组件）
├── monitoring.py             # 训练监控工具（损失/梯度/权重追踪）
├── stability_diagnosis.py    # 训练稳定性诊断工具
└── utils.py                  # 工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 在小语料上训练一个 mini-LM（完整训练循环） | ⭐⭐ 进阶 | 训练框架代码 + 超参数指引 | 体验完整预训练流程 |
| 2 | 对比不同学习率调度的训练效果（Cosine vs WSD） | ⭐⭐ 进阶 | 调度器代码 + 实验对比框架 | 理解学习率策略 |
| 3 | 实现完整的 Checkpoint 保存/恢复并验证可重复性 | ⭐⭐ 进阶 | 核心代码 + 验证方法 | 掌握训练鲁棒性工程 |
| 4 | 模拟并诊断训练不稳定性（人造 loss spike） | ⭐⭐⭐ 挑战 | 诊断框架 + 注入异常的方法 | 理解训练调试方法论 |

---

## 模块 9: 分布式训练 — 并行策略与工程优化

### README.md

#### 1. 为什么需要分布式训练
- 单卡显存限制的数学分析
- 模型参数 + 优化器状态 + 激活值的显存占用公式
- 训练时间估算：$T = \frac{6PD}{n \times \text{GPU\_FLOPS} \times \text{MFU}}$

#### 2. 数据并行（Data Parallelism）
- 基本数据并行（DP）
- 分布式数据并行（DDP）
  - All-Reduce 通信原理（Mermaid 示意图）
  - Ring All-Reduce 的带宽效率
- ZeRO 优化器（DeepSpeed）
  - ZeRO-1: 优化器状态分片
  - ZeRO-2: + 梯度分片
  - ZeRO-3: + 参数分片
  - 每阶段的显存节省推导

#### 3. 模型并行（Model Parallelism）
- 张量并行（Tensor Parallelism）
  - 列并行 vs 行并行（Mermaid 图）
  - 通信量分析
- 流水线并行（Pipeline Parallelism）
  - GPipe 方案：micro-batch 流水线
  - 气泡率分析：$\text{bubble} = \frac{p-1}{m+p-1}$
  - 1F1B 调度优化

#### 4. 3D 并行
- DP × TP × PP 的组合策略
- 设备映射的最优配置原则
- 通信拓扑与硬件亲和性

#### 5. 混合精度训练
- FP32 → FP16/BF16 → FP8 的演进
- 损失缩放（Loss Scaling）
- 动态损失缩放的实现

#### 6. 单卡替代方案
- 梯度累积：模拟大 batch size
- 梯度检查点（Activation Checkpointing）
  - 以计算换显存的数学分析
- CPU Offloading

### advanced.md

#### 1. Google 的分布式训练
- Pathways 系统架构
- TPU Pod 的高效利用
- GSPMD 编程模型
- PyTorch vs JAX (Google/Anthropic 视角):
  - 动态图 (Eager) vs 静态图 (XLA) 的计算图差异
  - SPMD (Single Program Multiple Data) 编程范式：`jax.pjit` 与 `sharding`
  - 为什么 Google 偏爱 JAX？数学表达的纯粹性与编译器优化的权衡

#### 2. DeepSeek 的工程创新
- DualPipe（DeepSeek-V3）：
  - 计算-通信重叠的流水线调度
  - 接近零气泡率的实现
  - 详细的调度时间线图
- FP8 训练的工程细节
- 跨节点通信优化

#### 3. Anthropic 视角
- 大规模训练的安全挑战
- 训练过程中的异常检测
- 分布式训练中的可重复性

#### 4. 前沿话题
- 异构计算集群的训练调度
- 弹性训练（Elastic Training）
- 通信压缩与量化梯度

### 代码目录 `code/distributed/`

```
code/distributed/
├── data_parallel.py          # DDP 基本实现
├── tensor_parallel.py        # 简化的张量并行
├── pipeline_parallel.py      # 简化的流水线并行（GPipe + 1F1B）
├── zero_optimizer.py         # ZeRO-1/2 简化实现
├── mixed_precision.py        # 混合精度训练封装
├── gradient_accumulation.py  # 梯度累积（单卡替代方案）
├── activation_checkpoint.py  # 梯度检查点
└── utils.py                  # 分布式工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 使用梯度累积模拟大 batch 训练（单GPU） | ⭐ 入门 | 完整代码 | 理解梯度累积原理 |
| 2 | 使用 PyTorch DDP 进行多卡训练 | ⭐⭐ 进阶 | 代码框架 + 启动命令 | 掌握 DDP 基础 |
| 3 | 计算并可视化 ZeRO 各阶段的显存节省 | ⭐⭐ 进阶 | 计算公式 + 可视化代码 | 理解 ZeRO 原理 |
| 4 | 实现简化版张量并行（单机模拟） | ⭐⭐⭐ 挑战 | 伪代码 + 通信模式图 + 关键代码 | 理解模型并行 |

---

## 模块 10: SFT — 监督微调与参数高效微调

### README.md

#### 1. 预训练→微调的范式转变
- 预训练模型的能力与不足
- 指令遵循（Instruction Following）的目标
- SFT 的数学形式：$L_{SFT} = -\sum_t \log P(y_t | x, y_{<t})$
- 只在回答部分计算损失（prompt masking）

#### 2. 指令微调数据
- 指令数据的格式：(instruction, input, output)
- 高质量指令数据的特征
- Self-Instruct / Evol-Instruct 方法
- 对话格式模板（ChatML, Llama 格式）

#### 3. 全参数微调
- 全量 SFT 的优势与局限
- 灾难性遗忘的数学分析
- 数据量与微调效果的关系

#### 4. LoRA（Low-Rank Adaptation）
- 核心思想：$W' = W + BA$，其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$
- 数学推导：
  - 低秩假设的理论支撑
  - 参数节省分析：$2dr \ll d^2$
  - 初始化策略：$A \sim \mathcal{N}(0, \sigma^2)$, $B = 0$
- 超参数选择：rank, alpha, target modules
- LoRA 的合并与推理

#### 5. QLoRA
- 4-bit NormalFloat 量化
- 双量化（Double Quantization）
- 分页优化器
- 实际显存节省分析

#### 6. 其他 PEFT 方法
- Prefix Tuning
- Prompt Tuning
- Adapter
- 各方法的适用场景对比

### advanced.md

#### 1. Google 的微调实践
- FLAN 系列的指令微调
- Gemma 的微调最佳实践
- T5/PaLM 的 Prompt Tuning 经验

#### 2. DeepSeek 的微调策略
- DeepSeek-V3 的 SFT 阶段细节
- MoE 模型的微调特殊性
- 多任务微调的数据配比

#### 3. Anthropic 视角
- Claude 的 SFT 与 HHH (Helpful, Harmless, Honest) 目标
- 安全微调数据的构建方法
- Red Teaming 在微调中的应用

#### 4. 前沿话题
- LoRA 变体（DoRA, LoRA+, rsLoRA）
- 长上下文微调策略
- 微调 vs In-Context Learning 的理论对比

### 代码目录 `code/sft/`

```
code/sft/
├── dataset.py                # 指令微调数据集处理
├── sft_trainer.py            # SFT 训练器
├── lora.py                   # LoRA 从零实现
├── qlora.py                  # QLoRA 实现（基于 bitsandbytes）
├── chat_template.py          # 对话模板处理
├── merge_lora.py             # LoRA 权重合并
└── utils.py                  # 工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 从零实现 LoRA 并验证数学等价性 | ⭐⭐ 进阶 | 完整代码 + 数学验证 | 深入理解 LoRA |
| 2 | 使用 QLoRA 微调 Llama/Gemma（单GPU） | ⭐⭐ 进阶 | 训练脚本框架 + 超参数建议 | 实战微调体验 |
| 3 | 构建指令微调数据集并评估质量 | ⭐⭐ 进阶 | 数据格式说明 + 评估思路 | 理解数据工程 |
| 4 | 对比 LoRA 不同 rank 和 target modules 的效果 | ⭐⭐⭐ 挑战 | 实验设计 + 评估框架 | 掌握 LoRA 调参 |

---

## 模块 11: RLHF — 基于人类反馈的强化学习

### README.md

#### 1. 为什么需要 RLHF
- SFT 的局限性：模仿学习的天花板
- 人类偏好与交叉熵损失的错位
- RLHF 的三阶段框架（Mermaid 流程图）

#### 2. 奖励模型（Reward Model）
- Bradley-Terry 偏好模型：
  - $P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$
  - 损失函数推导：$L_{RM} = -\log \sigma(r(x, y_w) - r(x, y_l))$
- 偏好数据的收集与标注
- 奖励模型的训练细节
- Reward Hacking 问题

#### 3. PPO（Proximal Policy Optimization）
- 策略梯度的基础：
  - REINFORCE 算法回顾
  - 基线（baseline）的方差缩减作用
- PPO 的核心思想：
  - 重要性采样比率：$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$
  - 裁剪目标函数：$L^{CLIP} = \min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t)$
  - 为什么需要裁剪？信任域的直觉
- RLHF 中的 PPO 适配：
  - KL 散度惩罚：$R_{total} = R_{RM} - \beta \text{KL}(\pi_\theta || \pi_{ref})$
  - KL 惩罚的作用：防止模型偏离过远
  - 完整训练流程（Mermaid 详细流程图）

#### 4. GAE（Generalized Advantage Estimation）
- 优势函数的估计
- $\hat{A}_t^{GAE} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$
- 偏差-方差权衡

#### 5. RLHF 的工程挑战
- 四个模型的显存管理（actor, critic, reward, reference）
- 训练稳定性问题
- 超参数敏感性

### advanced.md

#### 1. Google 的 RLHF 实践
- Gemini 的 RLHF 流程
- RLHF 在多模态模型中的应用
- 大规模人类标注的组织方法

#### 2. DeepSeek 的 RLHF 实践
- DeepSeek-V3 的后训练流程
- GRPO（Group Relative Policy Optimization）
  - 去除 Critic 模型的创新
  - 数学推导
- 从 RLHF 到推理模型（R1 的技术路线）

#### 3. Anthropic 的 RLHF 贡献
- **历史贡献**: Christiano et al. 2017 — RLHF 奠基论文
- **Constitutional AI (CAI)**:
  - 核心理念：用 AI 反馈替代人类反馈（RLAIF）
  - 宪法原则的设计
  - Red Teaming + Revision 流程
  - 数学框架
- **HH-RLHF 数据集**: 开源偏好数据的标杆
- **RLHF 的安全视角**: 如何通过 RLHF 实现 HHH 目标
- Anthropic 对 Reward Hacking 的研究

#### 4. 前沿话题
- Online RLHF vs Offline RLHF
- 过程奖励模型（Process Reward Model）
- RLHF 的理论基础与收敛性分析
- 多目标 RLHF

### 代码目录 `code/rlhf/`

```
code/rlhf/
├── reward_model.py           # 奖励模型训练
├── ppo_trainer.py            # PPO 训练器（简化版）
├── gae.py                    # GAE 优势估计
├── kl_controller.py          # KL 散度控制器
├── preference_dataset.py     # 偏好数据处理
├── rollout.py                # 经验收集（Rollout）
└── utils.py                  # 工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 训练一个文本情感奖励模型 | ⭐⭐ 进阶 | 数据集 + 训练代码 + 评估方法 | 理解奖励模型 |
| 2 | 实现简化版 PPO 训练循环 | ⭐⭐⭐ 挑战 | 算法伪代码 + 核心代码片段 | 理解 PPO 机制 |
| 3 | 分析 KL 惩罚系数对训练的影响 | ⭐⭐ 进阶 | 实验设计 + 分析框架 | 理解 RLHF 超参数 |
| 4 | 实现一个简化版 Constitutional AI 流程 | ⭐⭐⭐ 挑战 | 流程设计 + 伪代码 + 宪法原则示例 | 理解 Anthropic 的安全方法 |

---

## 模块 12: DPO 及变体 — 直接偏好优化

### README.md

#### 1. DPO 的动机
- RLHF 的复杂性：四个模型 + 强化学习不稳定
- 核心洞察：奖励函数可以从策略中隐式恢复
- 从 RL 到分类：简化训练流程（Mermaid 对比图）

#### 2. DPO 的数学推导
- **起点**: RLHF 目标函数
  - $\max_\pi \mathbb{E}_{x \sim D, y \sim \pi}[r(x,y)] - \beta \text{KL}(\pi || \pi_{ref})$
- **关键推导**: 最优策略的闭式解
  - $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$
- **隐式奖励**: 从策略恢复奖励
  - $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$
- **DPO 损失函数**:
  - $L_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$
- 逐步推导过程（每一步的数学动机）

#### 3. DPO 的直觉理解
- 梯度分析：DPO 做了什么？
  - 增加 preferred 回答的概率
  - 降低 dispreferred 回答的概率
  - 隐式奖励差距越大，梯度越小（自适应加权）
- 与 RLHF-PPO 的等价性讨论

#### 4. DPO 变体
- **IPO (Identity Preference Optimization)**
  - 解决 DPO 的过拟合问题
  - 正则化损失的数学形式
- **KTO (Kahneman-Tversky Optimization)**
  - 不需要成对偏好数据
  - 利用前景理论的损失不对称性
- **ORPO (Odds Ratio Preference Optimization)**
  - 将 SFT 和偏好优化统一
  - 优势比（Odds Ratio）的数学形式
- **SimPO (Simple Preference Optimization)**
  - 去除参考模型
  - 长度归一化的奖励
- **GRPO (Group Relative Policy Optimization)** — DeepSeek
  - 去除 Critic 模型
  - 组内相对排序作为优势估计

#### 5. 各方法对比
- 训练成本 / 稳定性 / 数据需求 / 性能的全面对比表
- 选择指南：什么场景用什么方法

### advanced.md

#### 1. Google 的偏好优化实践
- Gemini 中偏好优化的应用
- RLHF vs DPO 在工业中的选择

#### 2. DeepSeek 的 GRPO
- GRPO 的完整推导
- 在 DeepSeek-R1 中的应用
- 与标准 PPO 的对比实验

#### 3. Anthropic 视角
- Anthropic 对 DPO 类方法的评估
- Constitutional AI 与 DPO 的结合可能性
- 偏好优化中的安全对齐

#### 4. 前沿话题
- 在线偏好优化（Online DPO）
- 迭代 DPO 与自我对弈
- 多轮对话的偏好优化
- 偏好优化的理论边界

### 代码目录 `code/dpo/`

```
code/dpo/
├── dpo_trainer.py            # DPO 训练器
├── dpo_loss.py               # DPO 及变体的损失函数
├── kto_loss.py               # KTO 损失函数
├── orpo_loss.py              # ORPO 损失函数
├── simpo_loss.py             # SimPO 损失函数
├── grpo_loss.py              # GRPO 损失函数（DeepSeek）
├── preference_dataset.py     # 偏好数据处理
└── evaluation.py             # 对齐评估工具
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 从零推导并实现 DPO 损失函数 | ⭐⭐ 进阶 | 数学推导框架 + 代码验证 | 深入理解 DPO 数学 |
| 2 | 使用 DPO 对齐一个小模型 | ⭐⭐ 进阶 | 训练框架 + 数据准备指引 | 实战偏好优化 |
| 3 | 对比 DPO/KTO/SimPO 的训练效果 | ⭐⭐⭐ 挑战 | 实验设计 + 评估指标 | 理解各变体差异 |
| 4 | 实现 GRPO 并分析组大小的影响 | ⭐⭐⭐ 挑战 | 论文关键公式 + 伪代码 | 理解 DeepSeek 的创新 |

---

## 模块 13: CoT 与推理 — 思维链与测试时计算

### README.md

#### 1. 思维链（Chain-of-Thought）
- CoT Prompting 的基本思想
- Few-shot CoT vs Zero-shot CoT ("Let's think step by step")
- CoT 为什么有效？数学与实证分析
- CoT 的涌现特性：只在大模型上有效

#### 2. 高级推理策略
- **Self-Consistency**: 多次采样 + 多数投票
  - 数学框架：$\hat{a} = \arg\max_a \sum_i \mathbf{1}[a_i = a]$
  - 采样温度的影响
- **Tree of Thoughts (ToT)**:
  - 搜索树的构建（Mermaid 树状图）
  - BFS vs DFS 搜索策略
  - 评估函数的设计
- **Least-to-Most Prompting**:
  - 问题分解 → 子问题求解 → 组合
- **ReAct**: 推理 + 行动的交替

#### 3. 测试时计算（Test-time Compute Scaling）
- 核心洞察：推理时投入更多计算 → 更好的结果
- 与训练时 Scaling Laws 的互补
- Verifier 模型的作用
- Best-of-N 采样

#### 4. 推理模型
- **DeepSeek-R1**:
  - 纯 RL 训练的推理模型
  - "Aha moment"：模型学会自我反思
  - 训练流程：冷启动 → RL → 拒绝采样 → RL 强化
- 推理 token 的经济学分析
- 长推理链的控制与效率

#### 5. 推理评估
- 数学推理（GSM8K, MATH）
- 代码生成（HumanEval, MBPP）
- 通用推理（MMLU, ARC）

### advanced.md

#### 1. Google 的推理研究
- Gemini 的推理能力
- 搜索与推理的结合
- AlphaCode/AlphaProof 系列

#### 2. DeepSeek-R1 深度分析
- 训练流程的完整解析
- RL 训练中推理能力的涌现
- R1 vs R1-Zero 的对比
- GRPO 在推理训练中的应用
- 蒸馏小模型的方法

#### 3. Anthropic 视角
- Claude 的推理能力演进
- Extended Thinking 功能
- 安全推理：如何避免推理链中的有害内容
- Faithful CoT vs Unfaithful CoT 的研究

#### 4. 前沿话题
- 过程奖励模型（PRM）vs 结果奖励模型（ORM）
- Monte Carlo Tree Search + LLM
- 推理时计算的最优分配策略
- 推理能力的蒸馏

#### 5. 自动化评测流水线 (LLM-as-a-Judge)
- 核心思想：用强模型（如 GPT-4/Claude 3.5）评估弱模型
- 评测一致性分析：
  - Position Bias (位置偏差) 的数学校正
  - Verbosity Bias (话唠偏差) 的去偏方法
- 构造一个具体的 Judge Prompt 模板（Reference-guided grading）
- AlpacaEval 与 MT-Bench 的实现原理

### 代码目录 `code/reasoning/`

```
code/reasoning/
├── cot_prompting.py          # CoT Prompting 框架
├── self_consistency.py       # Self-Consistency 采样
├── tree_of_thoughts.py       # Tree of Thoughts 搜索
├── best_of_n.py              # Best-of-N 采样
├── verifier.py               # 简化版 Verifier 模型
└── evaluation.py             # 推理评估工具
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 实现 Self-Consistency 并评估效果 | ⭐⭐ 进阶 | 完整代码 + 评估框架 | 理解采样+投票策略 |
| 2 | 实现 Tree of Thoughts 搜索框架 | ⭐⭐⭐ 挑战 | 搜索框架伪代码 + 关键代码 | 理解结构化推理 |
| 3 | 分析 CoT 长度与推理质量的关系 | ⭐⭐ 进阶 | 实验设计 + 分析思路 | 理解推理机制 |
| 4 | 实现 Best-of-N + Verifier 推理框架 | ⭐⭐⭐ 挑战 | 框架设计 + 关键代码片段 | 理解测试时计算 |

---

## 模块 14: 推理加速 — KV Cache/量化/系统优化

### README.md

#### 1. LLM 推理的性能瓶颈
- Prefill 阶段 vs Decode 阶段（Mermaid 流程图）
- 计算密集型 vs 访存密集型分析
- Arithmetic Intensity 与 Roofline 模型
- 推理延迟的组成分解

#### 2. KV Cache
- 为什么需要 KV Cache？避免重复计算
- KV Cache 的显存计算：$M_{KV} = 2 \times L \times H \times d_h \times S \times \text{dtype\_size}$
- KV Cache 的管理策略
- 预分配 vs 动态分配

#### 3. PagedAttention（vLLM）
- 物理内存 vs 逻辑内存的类比
- KV Cache 的分页管理（Mermaid 示意图）
- 显存碎片消除
- Continuous Batching 的数学优势

#### 4. 模型量化
- **量化基础**:
  - 线性量化：$x_q = \text{round}(x / s) + z$
  - 对称量化 vs 非对称量化
  - 量化误差分析
- **GPTQ**:
  - 基于 Hessian 的逐层量化
  - OBQ → GPTQ 的推导
- **AWQ（Activation-aware Weight Quantization）**:
  - 激活感知的权重重要性
  - 等效变换技巧
- **INT4 / INT8 / FP8 量化实践**
  - 各精度的适用场景
  - 量化对模型质量的影响

#### 5. Flash Attention
- 标准 Attention 的 HBM 访问分析
- 分块计算（Tiling）的核心思想
- IO 复杂度的改进：$O(N^2 d^2 / M)$ vs $O(N^2)$
- 在线 Softmax 的数学技巧

#### 6. 推理系统优化
- Speculative Decoding（投机解码）
  - 小模型草稿 + 大模型验证
  - 接受概率的数学保证
- Continuous Batching
- 前缀缓存（Prefix Caching）

### advanced.md

#### 1. Google 的推理优化
- TPU 推理优化
- Gemma 的推理部署最佳实践
- 多查询注意力的推理加速效果

#### 2. DeepSeek 的推理策略
- MLA 对 KV Cache 的极致压缩
- MoE 推理的负载均衡
- DeepSeek-V3 的推理成本分析

#### 3. Anthropic 视角
- 大规模推理系统的安全考量
- 推理过程中的实时安全检查
- 延迟与安全性的 trade-off

#### 4. 前沿话题
- 稀疏注意力推理
- 编译优化（torch.compile, TensorRT）
- 硬件感知的模型优化
- LLM 推理的成本经济学

#### 5. 底层算子编程 (Triton)
- GPU 硬件抽象：
  - HBM (全局内存) vs SRAM (共享内存) 的带宽金字塔
  - 并行计算模型：Block, Warp, Thread
- OpenAI Triton 语言入门：
  - 为什么用 Triton 而不是 CUDA C++？
  - 核心概念：`tl.load`, `tl.store`, `tl.dot`
  - 实战案例：手写一个高效的 Vector Add 和 Softmax
  - FlashAttention 的 Triton 简化版实现解析

### 代码目录 `code/inference/`

```
code/inference/
├── kv_cache.py               # KV Cache 实现
├── paged_attention.py        # 简化版 PagedAttention
├── quantization.py           # 线性量化基础实现
├── gptq_simplified.py        # GPTQ 简化实现
├── flash_attention.py        # Flash Attention 概念实现
├── speculative_decoding.py   # 投机解码实现
├── benchmark.py              # 推理性能基准测试
└── utils.py                  # 工具函数
```

### 项目实践

| # | 项目名称 | 难度 | 提供内容 | 核心目标 |
|---|---------|------|---------|---------|
| 1 | 实现 KV Cache 并测量加速效果 | ⭐⭐ 进阶 | 完整代码 + 基准测试 | 理解 KV Cache 原理 |
| 2 | 对比 INT4/INT8/FP16 量化的质量与速度 | ⭐⭐ 进阶 | 量化代码 + 评估框架 | 理解量化的 trade-off |
| 3 | 实现简化版投机解码 | ⭐⭐⭐ 挑战 | 算法伪代码 + 关键实现 + 正确性证明 | 理解投机解码数学 |
| 4 | 使用 vLLM 部署模型并分析吞吐量 | ⭐⭐ 进阶 | 部署指引 + 性能分析思路 | 理解推理系统工程 |

---

## 模块 15: RAG 与 知识增强 — 检索、向量库与 GraphRAG

### README.md

#### 1. RAG (Retrieval-Augmented Generation) 核心范式
- 幻觉问题与知识截止（Knowledge Cutoff）
- RAG 的数学形式：$P(y|x) \approx \sum_{z \in TopK(x)} P(y|x,z) P(z|x)$
- **Naive RAG vs Advanced RAG**:
  - 预检索（Pre-retrieval）：Query 重写、HyDE（假设性文档嵌入）
  - 后检索（Post-retrieval）：重排序（Reranking）、上下文压缩

#### 2. 向量检索算法（Vector Search）
- **稠密检索 (Dense Retrieval)**:
  - 双塔架构（Bi-Encoder）：Query 与 Doc 的内积相似度
  - 对比学习损失：InfoNCE Loss 推导
- **ANN (Approximate Nearest Neighbor) 算法**:
  - 暴力搜索 vs 近似搜索的时间复杂度
  - **HNSW (Hierarchical Navigable Small World)**:
    - 核心图论：跳表（Skip List）+ 小世界网络（Small World Network）
    - 贪心搜索路径的数学证明
    - 插入与搜索过程的复杂度分析 ($O(\log N)$)

#### 3. 稀疏检索与混合搜索
- **BM25 算法回顾**: TF-IDF 的概率改进版
- **Hybrid Search**:
  - 为什么关键词匹配依然重要？（精确匹配 vs 语义匹配）
  - 倒数排名融合 (RRF) 算法：$score = \sum \frac{1}{k + rank_i}$

#### 4. GraphRAG (基于知识图谱的 RAG)
- 动机：解决"Global Question"（跨文档归纳）难题
- 架构流程（DeepSeek/Microsoft 路线）：
  - 文本 → 实体抽取 (LLM) → 构建图谱 (NetworkX)
  - 社区发现 (Leiden Algorithm) 的数学原理
  - 社区摘要生成 → 答案合成
- GraphRAG vs Vector RAG 的覆盖率对比

### advanced.md

#### 1. Google 的 RAG 研究
- **REALM / RETRO**: 将检索引入预训练阶段
- **Infinite Attention vs RAG**: 长上下文是否会杀死 RAG？
  - "Lost in the Middle" 现象的数学解释
  - 上下文窗口与检索精度的 Trade-off

#### 2. DeepSeek 与 工业界实践
- DeepSeek-R1 在 Search 场景的应用
- **Rerank 模型**：Cross-Encoder 的蒸馏与部署
- 向量数据库选型：Faiss vs Milvus vs pgvector 底层差异

#### 3. 前沿话题
- **RAG 对齐 (RAG Alignment)**: 防止检索到有毒内容导致的生成攻击
- **Self-RAG**: 模型学会自我反思"是否需要检索"
- **LongRAG**: 大 chunk 检索与长文本阅读器的结合

### 代码目录 `code/rag/`

```
code/rag/ 
├── dense_retriever.py # 基于 BERT/Embedding 的双塔检索 
├── bm25_retriever.py # 手写 BM25 算法 
├── hnsw_index.py # 简化的 HNSW 图索引实现（Python版）
├── graph_rag_basic.py # 简化的 GraphRAG 流程（实体抽取+建图） 
├── reranker.py # Cross-Encoder 重排序 
└── rag_pipeline.py # 完整的 Retrieve-Read-Generate 管道
```


### 项目实践 
| #   | 项目名称                               | 难度     | 提供内容                       | 核心目标      |
| --- | ---------------------------------- | ------ | -------------------------- | --------- |
| 1   | 从零实现 HNSW 索引构建与搜索                  | ⭐⭐⭐ 挑战 | 论文算法伪代码 + 图结构类             | 理解向量库底层原理 |
| 2   | 构建一个混合检索 (Hybrid Search) 系统        | ⭐⭐ 进阶  | BM25 + Embedding 代码        | 理解互补优势    |
| 3   | 实现简易版 GraphRAG (实体共现图)             | ⭐⭐⭐ 挑战 | 实体抽取 Prompt + NetworkX 图算法 | 掌握前沿 RAG  |
| 4   | 对比 RAG 与 Long Context (128k) 的问答效果 | ⭐⭐ 进阶  | 评测数据集 + 对比脚本               | 理解技术边界    |

---
## 模块 16: 前沿专题 — 可解释性/安全/多模态

### README.md

#### 1. 机械可解释性（Mechanistic Interpretability）
- 什么是可解释性？为什么重要？
- **Superposition 假说**:
  - 神经元多义性（Polysemanticity）
  - 特征在高维空间的稀疏编码
  - 数学框架：$d$ 维空间编码 $n \gg d$ 个特征
- **Sparse Autoencoders (SAE)**:
  - 架构：$f(x) = W_{dec} \cdot \text{ReLU}(W_{enc} x + b_{enc}) + b_{dec}$
  - 训练目标：重建 + L1 稀疏正则
  - 从多义神经元中提取单义特征
- **Circuits 分析**:
  - 注意力头的功能分类
  - Induction Heads 的发现与验证
  - 模型行为的因果分析方法

#### 2. AI 安全
- 对齐问题的定义与重要性
- **Constitutional AI** (Anthropic):
  - 宪法原则 → AI 自我监督 → RLAIF
  - 完整流程（Mermaid 流程图）
- **Red Teaming**: 系统性发现模型弱点
- **Jailbreaking 与防御**
- 安全评估基准

#### 3. 多模态 LLM
- Vision-Language Models 的基本架构
  - 视觉编码器 + 投影层 + LLM
- LLaVA 架构解析
- Gemini 的原生多模态方法
- 多模态训练的数据与策略

#### 4. Agent 与工具使用
- Function Calling 的实现原理
- ReAct 框架
- 多步推理与工具调用的协同
- Agent 安全性挑战

### advanced.md

#### 1. Anthropic 的可解释性研究（重点）
- **Scaling Monosemanticity** (2024):
  - 在 Claude 3 上训练 SAE
  - 发现可解释的高级特征（如"金门大桥"特征）
  - 特征转向（Feature Steering）的可能性
- **Toy Models of Superposition** (Elhage et al. 2022):
  - Superposition 的数学模型
  - 相变行为的分析
- Anthropic 可解释性团队的研究路线图
- 可解释性与安全性的关系

#### 2. Google 的前沿研究
- Gemini 的多模态能力
- Google 的 AI 安全研究
- 模型评估与基准测试

#### 3. DeepSeek 的前沿探索
- DeepSeek-VL 多模态模型
- 开源社区的安全实践

#### 4. 前沿话题
- 世界模型与 LLM
- 模型合并（Model Merging）
- 持续学习（Continual Learning）
- LLM 的理论理解

### 代码目录 `code/advanced_topics/`

```
code/advanced_topics/
├── sparse_autoencoder.py     # SAE 实现
├── feature_visualization.py  # 特征可视化工具
├── activation_patching.py    # 激活值干预实验
├── safety_evaluation.py      # 安全评估框架
├── multimodal_basic.py       # 多模态基础架构
└── function_calling.py       # Function Calling 实现
```

### 项目实践

| #   | 项目名称                         | 难度     | 提供内容              | 核心目标           |
| --- | ---------------------------- | ------ | ----------------- | -------------- |
| 1   | 训练一个简单的 Sparse Autoencoder   | ⭐⭐ 进阶  | 完整代码 + 可视化        | 理解 SAE 原理      |
| 2   | 分析注意力头的功能（Induction Head 检测） | ⭐⭐⭐ 挑战 | 分析方法 + 检测代码框架     | 理解 Circuits 分析 |
| 3   | 实现一个简单的 Red Teaming 框架       | ⭐⭐ 进阶  | 攻击策略 + 评估指标 + 伪代码 | 理解安全评估         |
| 4   | 搭建一个简单的多模态 LLM               | ⭐⭐⭐ 挑战 | 架构设计思路 + 关键代码片段   | 理解多模态架构        |

---

## 终极项目: 从零训练一个完整 LLM

> 独立章节，位于 `final_project/` 目录

### 项目概述

从数据准备到模型部署的完整流程，提供两个版本：

### Version A: 300M 参数（单 GPU 版本）

**目标**: 在服务器单张 24GB GPU 上完成全流程（进阶：local单张 8GB GPU上完成）

#### 架构配置
| 参数 | 值 |
|------|---|
| 层数 | 24 |
| 隐藏维度 | 1024 |
| 注意力头数 | 16 |
| FFN 隐藏维度 | 2730 (SwiGLU) |
| 词汇表大小 | 32,000 |
| 最大序列长度 | 2048 |
| 位置编码 | RoPE |
| 归一化 | RMSNorm |
| 注意力 | GQA (4 KV heads) |

#### 训练流程
1. **数据准备**: 收集约 10B tokens 的中英文混合数据
2. **分词器训练**: 训练 32K 词汇量的 BPE 分词器
3. **预训练**: NTP 目标, Cosine LR schedule, 混合精度
4. **SFT**: 使用开源指令数据进行微调
5. **DPO 对齐**: 使用偏好数据进行对齐
6. **评估**: 在标准 benchmark 上评估
7. **部署**: 量化 + 推理优化

#### 单卡训练策略
- 梯度累积（等效大 batch）
- 混合精度（BF16）
- 梯度检查点
- Flash Attention

### Version B: 1B 参数（多 GPU 版本）

**目标**: 在 4-8 张 GPU 上训练更大模型

#### 架构配置
| 参数 | 值 |
|------|---|
| 层数 | 32 |
| 隐藏维度 | 2048 |
| 注意力头数 | 32 |
| FFN 隐藏维度 | 5462 (SwiGLU) |
| 词汇表大小 | 64,000 |
| 最大序列长度 | 4096 |
| 位置编码 | RoPE |
| 归一化 | RMSNorm |
| 注意力 | GQA (8 KV heads) |

#### 分布式策略
- FSDP (Fully Sharded Data Parallel) 或 DeepSpeed ZeRO-2
- 张量并行（可选）
- 混合精度 + Flash Attention
- 更大的数据集（30-50B tokens）

### 终极项目代码目录

```
final_project/
├── README.md                  # 项目说明与完整指南
├── configs/
│   ├── model_300m.yaml        # 300M 模型配置
│   └── model_1b.yaml          # 1B 模型配置
├── src/
│   ├── model/
│   │   ├── config.py          # 模型配置类
│   │   ├── attention.py       # GQA 注意力
│   │   ├── feedforward.py     # SwiGLU FFN
│   │   ├── block.py           # Transformer Block
│   │   ├── model.py           # 完整模型
│   │   └── generation.py      # 生成策略
│   ├── data/
│   │   ├── tokenizer.py       # 分词器
│   │   ├── dataset.py         # 数据集处理
│   │   └── data_pipeline.py   # 数据管线
│   ├── training/
│   │   ├── trainer.py         # 训练器
│   │   ├── sft_trainer.py     # SFT 训练器
│   │   ├── dpo_trainer.py     # DPO 训练器
│   │   ├── lr_scheduler.py    # 学习率调度
│   │   └── distributed.py     # 分布式训练封装
│   ├── inference/
│   │   ├── engine.py          # 推理引擎
│   │   ├── quantize.py        # 量化工具
│   │   └── serve.py           # 简单推理服务
│   └── evaluation/
│       ├── benchmark.py       # 评估框架
│       └── metrics.py         # 评估指标
├── scripts/
│   ├── train_tokenizer.py     # 训练分词器脚本
│   ├── pretrain.py            # 预训练启动脚本
│   ├── sft.py                 # SFT 启动脚本
│   ├── dpo.py                 # DPO 启动脚本
│   ├── evaluate.py            # 评估脚本
│   └── export.py              # 模型导出脚本
└── docs/
    ├── training_guide.md      # 训练指南
    ├── troubleshooting.md     # 常见问题
    └── scaling_guide.md       # 扩展指南
```

### 提供内容说明

终极项目作为综合实践，提供：
- **完整的代码框架**（但关键逻辑留空，由学生填充）
- **详细的架构设计文档**
- **超参数配置与调参指南**
- **训练流程的 step-by-step 指引**
- **常见问题排查清单**
- **评估方法与预期结果**

**不提供**: 完整的可直接运行的训练代码（学生需要整合前面模块的知识来完成）

---

## 全局代码目录总览

```
code/
├── tokenization/              # 模块 1
│   ├── bpe_tokenizer.py       ✅ 已完成
│   ├── wordpiece_tokenizer.py
│   ├── unigram_tokenizer.py
│   ├── tokenizer_comparison.py
│   └── utils.py
├── embedding/                 # 模块 2
│   ├── word2vec.py
│   ├── glove.py
│   ├── positional_encoding.py
│   ├── rope.py
│   ├── visualize.py
│   └── utils.py
├── transformer/               # 模块 3
│   ├── attention.py
│   ├── normalization.py
│   ├── feedforward.py
│   ├── block.py
│   ├── model.py
│   └── utils.py
├── decoder_only/              # 模块 4
│   ├── config.py
│   ├── model.py
│   ├── generation.py
│   ├── tokenizer_wrapper.py
│   ├── train_simple.py
│   └── utils.py
├── attention_variants/        # 模块 5
│   ├── mha.py
│   ├── mqa.py
│   ├── gqa.py
│   ├── mla.py
│   ├── kv_cache.py
│   ├── benchmark.py
│   └── visualize_attention.py
├── moe/                       # 模块 6
│   ├── router.py
│   ├── expert.py
│   ├── moe_layer.py
│   ├── auxiliary_loss.py
│   ├── moe_transformer.py
│   └── analysis.py
├── data_engineering/          # 模块 7
│   ├── deduplication.py
│   ├── quality_filter.py
│   ├── text_extraction.py
│   ├── data_mixer.py
│   ├── pipeline.py
│   └── analysis.py
├── pretraining_objectives/    # 模块 8A
│   ├── ntp_loss.py
│   ├── mlm_loss.py
│   ├── fim_loss.py
│   ├── perplexity.py
│   └── utils.py
├── scaling_laws/              # 模块 8B
│   ├── scaling_laws.py
│   ├── chinchilla_optimal.py
│   ├── flops_calculator.py
│   ├── visualization.py
│   └── utils.py
├── training_engineering/      # 模块 8C
│   ├── optimizer.py
│   ├── lr_scheduler.py
│   ├── checkpointing.py
│   ├── trainer.py
│   ├── monitoring.py
│   ├── stability_diagnosis.py
│   └── utils.py
├── distributed/               # 模块 9
│   ├── data_parallel.py
│   ├── tensor_parallel.py
│   ├── pipeline_parallel.py
│   ├── zero_optimizer.py
│   ├── mixed_precision.py
│   ├── gradient_accumulation.py
│   ├── activation_checkpoint.py
│   └── utils.py
├── sft/                       # 模块 10
│   ├── dataset.py
│   ├── sft_trainer.py
│   ├── lora.py
│   ├── qlora.py
│   ├── chat_template.py
│   ├── merge_lora.py
│   └── utils.py
├── rlhf/                      # 模块 11
│   ├── reward_model.py
│   ├── ppo_trainer.py
│   ├── gae.py
│   ├── kl_controller.py
│   ├── preference_dataset.py
│   ├── rollout.py
│   └── utils.py
├── dpo/                       # 模块 12
│   ├── dpo_trainer.py
│   ├── dpo_loss.py
│   ├── kto_loss.py
│   ├── orpo_loss.py
│   ├── simpo_loss.py
│   ├── grpo_loss.py
│   ├── preference_dataset.py
│   └── evaluation.py
├── reasoning/                 # 模块 13
│   ├── cot_prompting.py
│   ├── self_consistency.py
│   ├── tree_of_thoughts.py
│   ├── best_of_n.py
│   ├── verifier.py
│   └── evaluation.py
├── inference/                 # 模块 14
│   ├── kv_cache.py
│   ├── paged_attention.py
│   ├── quantization.py
│   ├── gptq_simplified.py
│   ├── flash_attention.py
│   ├── speculative_decoding.py
│   ├── benchmark.py
|   ├── triton_kernels.py      # Triton 算子实现 (Vector Add, Softmax)
│   └── utils.py
├── rag/                       # 模块 15 (新增)
│   ├── dense_retriever.py
│   ├── bm25_retriever.py
│   ├── hnsw_index.py
│   ├── graph_rag_basic.py
│   ├── reranker.py
│   └── rag_pipeline.py
└── advanced_topics/           # 模块 16
    ├── sparse_autoencoder.py
    ├── feature_visualization.py
    ├── activation_patching.py
    ├── safety_evaluation.py
    ├── multimodal_basic.py
    └── function_calling.py
```

---

## 参考文献

1. Vaswani et al. "Attention Is All You Need" (2017)
2. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2019)
3. Radford et al. "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
4. Brown et al. "Language Models are Few-Shot Learners" (GPT-3, 2020)
5. Kaplan et al. "Scaling Laws for Neural Language Models" (2020)
6. Hoffmann et al. "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
7. Touvron et al. "LLaMA: Open and Efficient Foundation Language Models" (2023)
8. Touvron et al. "Llama 2: Open Foundation and Fine-Tuned Chat Models" (2023)
9. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
10. Press et al. "Train Short, Test Long: Attention with Linear Biases" (ALiBi, 2022)
11. Hu et al. "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
12. Dettmers et al. "QLoRA: Efficient Finetuning of Quantized Language Models" (2023)
13. Christiano et al. "Deep Reinforcement Learning from Human Preferences" (2017)
14. Bai et al. "Training a Helpful and Harmless Assistant with RLHF" (Anthropic, 2022)
15. Bai et al. "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)
16. Rafailov et al. "Direct Preference Optimization" (DPO, 2023)
17. Shazeer et al. "Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer" (2017)
18. Fedus et al. "Switch Transformers: Scaling to Trillion Parameter Models" (2022)
19. DeepSeek-AI "DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model" (2024)
20. DeepSeek-AI "DeepSeek-V3 Technical Report" (2024)
21. DeepSeek-AI "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL" (2025)
22. Elhage et al. "Toy Models of Superposition" (Anthropic, 2022)
23. Templeton et al. "Scaling Monosemanticity" (Anthropic, 2024)
24. Olsson et al. "In-context Learning and Induction Heads" (Anthropic, 2022)
25. Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (2022)
26. Dao et al. "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
27. Kwon et al. "Efficient Memory Management for LLM Serving with PagedAttention" (vLLM, 2023)
28. Schulman et al. "Proximal Policy Optimization Algorithms" (PPO, 2017)
29. Frantar et al. "GPTQ: Accurate Post-Training Quantization for GPT" (2022)
30. Lin et al. "AWQ: Activation-aware Weight Quantization" (2023)

---

## 执行计划

### 第一阶段: 更新已有模块（0-3）✅ 已完成

| 步骤 | 内容 | Commit |
|------|------|--------|
| 1. 提交暂存文件 + OUTLINE.md | 初始化 | `49ee1ce` |
| 2. 模块 0: Anthropic 主线 + advanced.md | 完成 | `12182c3` |
| 3. 模块 1: Anthropic 内容 + advanced.md + 项目扩展 | 完成 | `87df7de` |
| 4. 模块 2: Anthropic 内容 + advanced.md + 代码提取 + 项目扩展 | 完成 | `3d4a622` |
| 5. 模块 3: Anthropic 内容 + advanced.md + 代码提取 + 项目扩展 | 完成 | `4433aaf` + `9576fd4` |
| 6. 杂项: BOM 修复 + OUTLINE 扩展 | 完成 | `5cf4c2d` + `16da8f4` |

**已完成的产出物**:
- `00_overview/`: README.md(已有) + advanced.md
- `01_tokenization/`: README.md(已有) + advanced.md
- `02_embedding/`: README.md(已更新) + advanced.md + `code/embedding/{word2vec,positional_encoding,visualize}.py`
- `03_transformer/`: README.md(已更新) + advanced.md + `code/transformer/{attention,normalization,feedforward,block,model,__init__}.py`

### 第二阶段: 编写新模块（4-16）⬅️ 当前阶段

> **重要**: 本阶段开启新会话执行。以下是给新会话的完整上下文。

#### 已完成批次

##### 批次 1: 架构类模块（4, 5, 6）✅ 已完成

| 模块 | 文件夹 | 文件数 | Commit | 说明 |
|------|--------|--------|--------|------|
| 模块 4: Decoder-Only 架构 | `04_decoder_only/` + `code/decoder_only/` | 9 | `7d4591a` | README.md(1153行) + advanced.md + 7个代码文件 |
| 模块 5: 注意力机制变体 | `05_attention_variants/` + `code/attention_variants/` | 9 | `20d52c7` | README.md(807行) + advanced.md + 7个代码文件 |
| 模块 6: MoE 混合专家 | `06_moe/` + `code/moe/` | 8 | `bbb7932` | README.md(1101行) + advanced.md + 6个代码文件 |

**总计**: 26 个文件，约 10,638 行新增内容。

##### 批次 2: 训练类模块（7, 8A, 8B, 8C, 9）✅ 已完成

| 模块 | 文件夹 | 文件数 | Commit | 说明 |
|------|--------|--------|--------|------|
| 模块 7: 数据工程 | `07_data_engineering/` + `code/data_engineering/` | 8 | `c893452` | README.md + advanced.md + 6个代码文件（去重/过滤/清洗/混合/管线/分析） |
| 模块 8A: 预训练目标 | `08a_pretraining_objectives/` + `code/pretraining_objectives/` | 7 | `ba089d8` | README.md + advanced.md + 5个代码文件（NTP/MLM/FIM损失、困惑度、工具） |
| 模块 8B: Scaling Laws | `08b_scaling_laws/` + `code/scaling_laws/` | 8 | `0812092` | README.md + advanced.md + 6个代码文件（拟合器/Chinchilla/FLOPs/可视化/工具） |
| 模块 8C: 训练工程 | `08c_training_engineering/` + `code/training_engineering/` | 10 | `96061e7` | README.md + advanced.md + 8个代码文件（优化器/调度器/检查点/训练器/监控/诊断/工具） |
| 模块 9: 分布式训练 | `09_distributed/` + `code/distributed/` | 10 | `405dc3d` | README.md + advanced.md + 8个代码文件（DP/TP/PP/ZeRO/混合精度/梯度累积/激活检查点/工具） |

**总计**: 43 个文件，约 22,420 行新增内容。

##### 批次 3: 对齐类模块（10, 11, 12）✅ 已完成

| 模块 | 文件夹 | 文件数 | Commit | 说明 |
|------|--------|--------|--------|------|
| 模块 10: SFT 监督微调 | `10_sft/` + `code/sft/` | 10 | `e4b9772` | README.md + advanced.md + 8个代码文件（数据集/训练器/LoRA/QLoRA/对话模板/权重合并/工具） |
| 模块 11: RLHF 对齐 | `11_rlhf/` + `code/rlhf/` | 9 | `52e7234` | README.md + advanced.md + 7个代码文件（奖励模型/PPO训练器/GAE/KL控制器/偏好数据/Rollout/工具） |
| 模块 12: DPO 及变体 | `12_dpo/` + `code/dpo/` | 10 | `67161a3` | README.md + advanced.md + 8个代码文件（DPO/KTO/ORPO/SimPO/GRPO损失/训练器/偏好数据/评估） |

**总计**: 29 个文件，约 13,946 行新增内容。

##### 批次 4: 应用/前沿类模块（13, 14, 15, 16）✅ 已完成

| 模块 | 文件夹 | 文件数 | Commit | 说明 |
|------|--------|--------|--------|------|
| 模块 13: CoT 与推理 | `13_reasoning/` + `code/reasoning/` | 8 | `4653e48` | README.md + advanced.md + 6个代码文件（CoT/Self-Consistency/ToT/Best-of-N/Verifier/评估） |
| 模块 14: 推理加速 | `14_inference/` + `code/inference/` | 10 | `17400bf` | README.md + advanced.md + 8个代码文件（KV Cache/PagedAttention/量化/GPTQ/Flash Attention/投机解码/基准测试/工具） |
| 模块 15: RAG 知识增强 | `15_rag/` + `code/rag/` | 8 | `7b60499` | README.md + advanced.md + 6个代码文件（双塔检索/BM25/HNSW/GraphRAG/重排序/RAG管道） |
| 模块 16: 前沿专题 | `16_frontiers/` + `code/advanced_topics/` | 8 | `80cfb3b` | README.md + advanced.md + 6个代码文件（SAE/特征可视化/激活干预/安全评估/多模态/Function Calling） |

**总计**: 34 个文件，约 20,551 行新增内容。

#### 执行方式

**1. 使用子代理（Sub-agents）直接写入主仓库**

每个模块的工作量大（README.md ~400-1100行 + advanced.md ~300行 + 代码文件），使用 Task 工具的子代理并行处理。

> ⚠️ **注意**: 最初设计使用 Git Worktree 实现并行隔离，但在 Windows 环境下失败。
> 实际采用的方案是：所有子代理直接写入主仓库 `D:\Repos\LLM learning\`，
> 因为每个模块有独立的子文件夹（`04_decoder_only/`、`05_attention_variants/` 等），
> 不会产生文件冲突。**子代理不执行任何 Git 操作**，所有 `git add` / `git commit`
> 由主 Agent（Orchestrator）在子代理完成后统一执行。

##### 1.1 实际工作流

```
1. 主 Agent 启动一批子代理（Task 工具，background 模式）
2. 每个子代理在自己的模块文件夹中创建文件（Write 工具）
3. 主 Agent 监控子代理进度，等待全部完成
4. 主 Agent 验证产出文件（Glob 工具）
5. 主 Agent 逐模块执行 git add + git commit
6. 启动下一批次
```

##### 1.2 子代理分组与调度

| 批次   | 模块               | 子代理分配    | 状态 | 理由                       |
| ---- | ---------------- | -------- | ---- | ------------------------ |
| 批次 1 | 4, 5, 6          | 各 1 个子代理 | ✅ 已完成 | 架构类模块，相互独立               |
| 批次 2 | 7, 8A, 8B, 8C, 9 | 各 1 个子代理 | ✅ 已完成 | 训练类模块，5个子代理并行写入          |
| 批次 3 | 10, 11, 12       | 各 1 个子代理 | ✅ 已完成 | 对齐类模块（SFT/RLHF/DPO），3个子代理并行写入 |
| 批次 4 | 13, 14, 15, 16   | 各 1 个子代理 | ✅ 已完成 | 应用/前沿类模块，4个子代理并行写入 |

每批次内的子代理并行启动，批次间串行执行（等前一批完成并 commit 后再启动下一批）。

##### 1.3 子代理调度 prompt 模板

```
你是 LLM 教程撰写者（Agent-{N}）。

【工作目录】D:\Repos\LLM learning\（主仓库，直接写入）
【重要】不要执行任何 Git 操作（git add/commit/push 等），主 Agent 会统一处理。

请根据 OUTLINE.md 中模块 {X} 的大纲，创建以下文件:
1. {XX_module}/README.md — 核心教程（理论+数学推导+Mermaid图+三条技术线+项目实践）
2. {XX_module}/advanced.md — 进阶工业实践
3. code/{module}/*.py — 代码实现文件

【关键约束】见下方"写作规范"。
```

**2. 项目实践的开放式原则（重要！）**

本教程面向"理论基础较强，渴望提高编程能力和工程能力"的大学生，项目设计为**开放式**：

| 难度 | 提供内容 | 不提供 |
|------|---------|-------|
| ⭐ 入门 | 完整可运行代码 + 详细注释 | — |
| ⭐⭐ 进阶 | 思路 + 关键代码片段 + 评估方法 | 完整实现 |
| ⭐⭐⭐ 挑战 | 思路 + 流程图(Mermaid) + 伪代码 + 论文指引 | 完整实现 |

即：**撰书者（Opus 模型）不需要给出完整代码**，仅提供思路、流程图（Mermaid）、伪代码、关键部分代码片段。让学生自己动手填充。

**3. 写作规范（所有子代理必须遵守）**

- **语言**: 正文中文，代码注释中文，数学公式 LaTeX
- **三条技术线**: Google / DeepSeek / Anthropic 贯穿每个模块
  - Anthropic 内容基于公开信息，推测部分必须标注
- **文档结构**:
  - README.md: 核心理论 → 数学推导 → 代码实现 → 三条技术线的实践 → 项目实践
  - advanced.md: Google 实践 → DeepSeek 实践 → Anthropic 视角 → 前沿话题
- **代码文件**: 放在 `code/{模块名}/` 下，按逻辑职责拆分为独立 .py 文件
  - 每个文件包含完整的类/函数定义、中文 docstring、`if __name__ == "__main__"` 演示
  - PyTorch 为主框架
  - 代码应能独立运行（无需依赖本项目其他模块）
- **Mermaid 图**: 架构图、流程图、对比图，大量使用
- **数学推导**: 不省略关键步骤，给出直觉解释
- **项目实践**: 每模块 3-4 个，难度梯度递进，按上述开放式原则
- **Git**: 每个模块完成后独立 commit，格式 `feat({module}): {描述}`

**4. 代码文件的提供策略**

`code/{module}/` 下的 .py 文件不同于项目实践中的代码：
- **code/ 下的文件**: 提供**完整可运行的参考实现**，是教程正文中代码的提取和整理
- **项目实践中的代码**: 按难度梯度提供（完整/部分/仅思路）

**5. 每个模块的预期产出**

```
{XX_module}/
├── README.md          # ~300-500 行，核心教程
└── advanced.md        # ~200-400 行，进阶内容

code/{module}/
├── *.py               # 若干代码文件，按 OUTLINE.md 中的代码目录规划
└── __init__.py         # 包初始化（可选）
```

**6. 已完成模块可作为参考范例**

新会话可以读取以下已完成文件作为风格参考：
- `03_transformer/README.md` — README 范例（Phase 1，最完整的一个）
- `03_transformer/advanced.md` — advanced.md 范例
- `code/transformer/attention.py` — 代码文件范例
- `02_embedding/advanced.md` — 另一个 advanced.md 范例
- `04_decoder_only/README.md` — Phase 2 README 范例（1153行，最新风格）
- `05_attention_variants/README.md` — Phase 2 README 范例（含 KV Cache 分析）
- `06_moe/README.md` — Phase 2 README 范例（含 MoE 路由数学推导）
- `code/decoder_only/model.py` — Phase 2 代码范例（完整 Decoder-Only 模型）
- `07_data_engineering/README.md` — Batch 2 范例（数据工程，MinHash/LSH去重）
- `08b_scaling_laws/README.md` — Batch 2 范例（Scaling Laws，完整Lagrange推导）
- `09_distributed/README.md` — Batch 2 范例（分布式训练，ZeRO推导 + 3D并行）
- `code/scaling_laws/scaling_laws.py` — Batch 2 代码范例（含Bootstrap置信区间）
- `code/training_engineering/optimizer.py` — Batch 2 代码范例（AdamW从零实现）
- `10_sft/README.md` — Batch 3 范例（SFT/LoRA完整数学推导，23个Mermaid图）
- `11_rlhf/README.md` — Batch 3 范例（RLHF三阶段，Bradley-Terry/PPO推导）
- `12_dpo/README.md` — Batch 3 范例（DPO五步推导，5种变体对比）
- `code/sft/lora.py` — Batch 3 代码范例（LoRA从零实现+数学验证）
- `code/dpo/dpo_loss.py` — Batch 3 代码范例（DPO损失+完整推导注释）

### 第三阶段: 终极项目

- 在模块 4-16 全部完成后执行
- 两个子代理分别完成 Version A（300M 单GPU）和 Version B（1B 多GPU）
- 终极项目代码框架提供关键逻辑留空的脚手架，学生整合前面模块知识来填充
- 整合测试与文档

---

**阶段进度总览**:
- Phase 1（模块 0-3 更新）: ✅ 已完成
- Phase 2 批次 1（模块 4, 5, 6）: ✅ 已完成 — 26 文件
- Phase 2 批次 2（模块 7, 8A, 8B, 8C, 9）: ✅ 已完成 — 43 文件
- Phase 2 批次 3（模块 10, 11, 12）: ✅ 已完成 — 29 文件
- Phase 2 批次 4（模块 13, 14, 15, 16）: ✅ 已完成 — 34 文件
- **Phase 2 全部完成！** 模块 4-16 共 132 文件
- Phase 3（终极项目）: ⬅️ 下一阶段

*最后更新: 2026-02-15*
