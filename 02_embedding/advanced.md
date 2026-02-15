# Embedding进阶：位置编码演进与嵌入空间研究

> 本文是 [模块2: Embedding与位置编码](./README.md) 的进阶补充，深入分析 Google、DeepSeek、Anthropic 三条技术线的位置编码实践，以及嵌入空间的前沿研究方向。

---

## 目录

- [1. Google 的位置编码演进](#1-google-的位置编码演进)
- [2. DeepSeek 的位置编码实践](#2-deepseek-的位置编码实践)
- [3. Anthropic 视角](#3-anthropic-视角)
- [4. 前沿话题](#4-前沿话题)

---

## 1. Google 的位置编码演进

### 1.1 三代位置编码方案

Google 在位置编码上经历了清晰的演进路径：

```mermaid
graph LR
    A[正弦编码<br/>Transformer 2017<br/>绝对位置] --> B[Relative Position Bias<br/>T5 2019<br/>可学习相对偏置]
    B --> C[RoPE<br/>PaLM/Gemma 2022-2024<br/>旋转位置编码]

    A --> A1[固定 + 无参数]
    B --> B1[可学习 + 桶化]
    C --> C1[旋转 + 外推友好]
```

### 1.2 原始 Transformer 的正弦编码

原始 Transformer（Vaswani et al., 2017）使用正弦位置编码的设计考量：

**选择正弦函数的原因**：
1. **唯一性**：每个位置有唯一编码
2. **有界性**：值域 $[-1, 1]$，数值稳定
3. **线性可表示性**：$PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性变换
4. **无需学习**：减少过拟合风险

**论文中的实验结论**：可学习位置编码与正弦编码效果相当，但正弦编码理论上支持外推。实际上，原始 Transformer 在外推方面表现并不理想。

### 1.3 T5 的 Relative Position Bias

T5 做出了一个重要创新：**桶化相对位置偏置**（Bucketed Relative Position Bias）。

**核心思想**：将相对位置 $m - n$ 映射到有限个"桶"中，每个桶对应一个可学习的偏置值。

**桶化函数**：

$$b(m - n) = \begin{cases}
m - n, & |m-n| \leq k_{near} \\
k_{near} + \lfloor \log_{base} \frac{|m-n|}{k_{near}} \rfloor, & |m-n| > k_{near}
\end{cases}$$

**设计动机**：
- 近距离：精确区分每个相对位置（$|m-n| \leq 8$ 时，每个距离一个桶）
- 远距离：对数桶化，粗粒度区分（远距离差异不大）
- 总桶数有限（如32个），参数开销小

```mermaid
graph TB
    subgraph "桶化示意"
        A["距离 0-8: 每个一个桶 (9桶)"]
        B["距离 9-16: 每2个一个桶"]
        C["距离 17-64: 每8个一个桶"]
        D["距离 64+: 共享最后一个桶"]
    end
```

**优势与局限**：
- 优势：参数少、相对位置感知、不同头可学习不同模式
- 局限：桶数固定，超出最大桶的距离无法精细区分

### 1.4 PaLM / Gemma 转向 RoPE

Google 在 PaLM（2022）和 Gemma（2024）中采用了 RoPE，这是一个重要的转向：

**PaLM 的 RoPE 配置**：
- 模型维度 $d = 18432$，头维度 $d_h = 256$
- 使用标准 base $= 10000$
- 上下文窗口 2048 tokens

**Gemma 的改进**：
- Gemma 1: RoPE base $= 10000$，上下文 8192 tokens
- Gemma 2: 部分层使用 Local Attention + Sliding Window，配合 RoPE
  - 交替使用全局注意力和局部滑动窗口
  - 局部窗口大小为 4096
  - 全局层仍使用完整 RoPE

**Gemma 2 的混合注意力策略**：

```mermaid
graph TB
    subgraph "Gemma 2 注意力层交替"
        L1["Layer 1: Local (window=4096)"]
        L2["Layer 2: Global (full RoPE)"]
        L3["Layer 3: Local (window=4096)"]
        L4["Layer 4: Global (full RoPE)"]
        L5["..."]
    end

    L1 --> L2 --> L3 --> L4 --> L5
```

**为什么从 T5 bias 转向 RoPE？**

| 对比维度 | T5 Relative Bias | RoPE |
|---------|-----------------|------|
| 参数量 | 每层每头需要一组偏置参数 | 零额外参数 |
| 外推能力 | 较弱（桶数固定） | 较强（可配合插值） |
| 计算效率 | 需查表 + 加法 | 仅需逐元素乘法 |
| KV Cache兼容 | 良好 | 良好 |
| 理论基础 | 启发式设计 | 旋转群的数学性质 |

### 1.5 RoPE 的旋转矩阵几何直觉

理解 RoPE 的数学本质，需要从**二维旋转的几何意义**出发。

#### 复数视角

将二维实向量 $(x_1, x_2)$ 视为复数 $z = x_1 + i x_2$，则 RoPE 的旋转操作等价于**复数乘法**：

$$\text{RoPE}(z, m) = z \cdot e^{im\theta} = (x_1 + ix_2)(\cos m\theta + i\sin m\theta)$$

展开为实部和虚部：

$$\text{Re}[\text{RoPE}(z, m)] = x_1 \cos m\theta - x_2 \sin m\theta$$
$$\text{Im}[\text{RoPE}(z, m)] = x_1 \sin m\theta + x_2 \cos m\theta$$

这正是旋转矩阵 $R_m$ 的逐元素展开形式。

**几何直觉**：RoPE 将每对维度 $(x_{2i-1}, x_{2i})$ 视为一个二维平面上的点，位置 $m$ 的编码等价于在这个平面上**逆时针旋转** $m\theta_i$ 角度。

```mermaid
graph TB
    subgraph "RoPE 的几何含义"
        A["原始向量 (x₁, x₂)<br/>在二维平面上"]
        B["旋转 mθ 角度<br/>→ (x₁', x₂')"]
        C["位置 m 越大<br/>旋转角度越大"]
    end
    A --> B --> C
```

#### 高维的"分频段旋转"

对于 $d$ 维向量，RoPE 将其分成 $d/2$ 个二维平面，**每个平面使用不同的旋转频率**：

$$\theta_i = \text{base}^{-2(i-1)/d}, \quad i = 1, 2, \ldots, d/2$$

- **第 1 对维度**：$\theta_1 = 1$，旋转频率最高，位置每增加 1 就旋转约 $57.3°$
- **最后一对维度**：$\theta_{d/2} = 1/\text{base}$，旋转频率极低，位置增加 base 才旋转一圈

这种**多尺度设计**使得：
- 高频维度编码局部位置差异（相邻 token 有明显不同）
- 低频维度编码全局位置信息（只有距离很远的 token 才有显著差异）

#### 为什么 base = 10000？

频率基数 base 决定了**最低频率维度的周期长度**：

$$T_{max} = 2\pi \cdot \text{base} \approx 62832 \text{ positions}$$

- 如果 base 太小（如 100），最低频维度的周期仅约 628 个位置，超出后位置编码开始重复
- 如果 base 太大（如 1000000），低频维度变化过于缓慢，在短序列中几乎不携带位置信息

原始论文选择 base = 10000 是一个经验值，使得在 2048-8192 长度范围内各频率维度都能有效工作。后续 Llama 3 和 Gemma 对此进行了调整。

### 1.6 Gemma / Llama 3 的 RoPE 配置对比

不同模型对 RoPE 的 base 和上下文长度进行了不同的配置，反映了各自在长度外推与表达精度之间的权衡：

| 模型 | $d_{model}$ | $d_{head}$ | RoPE Base | 训练上下文长度 | 外推策略 |
|------|:-----------:|:----------:|:---------:|:------------:|---------|
| Llama 2 7B | 4096 | 128 | 10,000 | 4K | 无 / PI |
| Llama 3 8B | 4096 | 128 | 500,000 | 8K | 高 base + 扩展训练 |
| Llama 3.1 8B | 4096 | 128 | 500,000 | 128K | 高 base + 渐进扩展 |
| Gemma 1 7B | 3072 | 256 | 10,000 | 8K | 标准 RoPE |
| Gemma 2 9B | 3584 | 256 | 10,000 | 8K | 局部/全局交替注意力 |
| DeepSeek-V2 | 5120 | 128 | 10,000 | 4K → 128K | 解耦 RoPE + YaRN |
| DeepSeek-V3 | 7168 | 128 | 10,000 | 4K → 128K | 解耦 RoPE + YaRN |

**Llama 3 的关键设计决策**：

Llama 3 将 RoPE base 从 10,000 大幅提升至 **500,000**，这一改变的数学含义是：

$$\theta_i^{(\text{Llama3})} = 500000^{-2(i-1)/d} \ll \theta_i^{(\text{Llama2})} = 10000^{-2(i-1)/d}$$

对于相同的维度 $i$，Llama 3 的旋转频率**更低**，这意味着：
1. 相邻位置的向量差异更小（更平滑）
2. 最低频维度的周期从约 6.3 万个位置扩展到约 314 万个位置
3. 天然支持更长的序列，而不需要依赖后续的 PI 或 NTK 插值

**代价**：高频维度的位置区分能力下降。Llama 3 通过增加训练数据量和训练步数来弥补这一损失。

**Gemma 2 的替代思路**：

Gemma 2 没有修改 RoPE base，而是采用**局部-全局交替注意力**：
- 奇数层使用滑动窗口注意力（窗口 4096），仅关注局部上下文
- 偶数层使用全局注意力，处理长距离依赖
- 这种设计减少了对位置编码长距离外推能力的依赖

---

## 2. DeepSeek 的位置编码实践

### 2.1 DeepSeek-V2 的 MLA 与位置编码协同

DeepSeek-V2 引入了 MLA（Multi-head Latent Attention），其核心创新是将 KV 压缩到低维潜在空间。这对位置编码提出了特殊挑战。

**MLA 的 KV 压缩**：

$$c_{KV} = W_{DKV} h_t \in \mathbb{R}^{d_c}, \quad d_c \ll n_h \cdot d_h$$

其中 $d_c$ 是压缩维度（如 512），远小于原始 KV 维度 $n_h \cdot d_h$（如 $32 \times 128 = 4096$）。

**位置编码的难题**：

如果在压缩前应用 RoPE，位置信息会被压缩丢失：

$$c_{KV} = W_{DKV} (\text{RoPE}(h_t)) \quad \text{← 位置信息被压缩}$$

**DeepSeek 的解决方案 — 解耦 RoPE**：

```mermaid
graph TB
    A[输入 h_t] --> B[低秩压缩]
    A --> C[RoPE 专用投影]

    B --> D["c_KV (无位置信息)"]
    C --> E["q_rope, k_rope (携带位置信息)"]

    D --> F[上投影恢复 KV]
    E --> G[与 Q/K 拼接]

    F --> H[注意力计算]
    G --> H
```

**数学形式**：

$$q_t = [W_{UQ} c_Q^{(t)} ; \text{RoPE}(W_{QR} h_t)]$$

$$k_t = [W_{UK} c_{KV}^{(t)} ; \text{RoPE}(W_{KR} h_t)]$$

其中 $[\cdot ; \cdot]$ 表示拼接。注意力分数变为：

$$a_{mn} = \frac{1}{\sqrt{d}} \left[ (W_{UQ} c_Q^{(m)})^T (W_{UK} c_{KV}^{(n)}) + (\text{RoPE}_m W_{QR} h_m)^T (\text{RoPE}_n W_{KR} h_n) \right]$$

**关键设计理念**：
1. **内容部分**（第一项）：来自压缩表示，不含位置信息，可被缓存
2. **位置部分**（第二项）：通过独立的低维投影携带位置信息，使用 RoPE

### 2.2 DeepSeek 的长上下文策略

DeepSeek 在长上下文扩展方面采用了渐进式方法：

| 版本 | 上下文长度 | 位置编码策略 |
|------|-----------|-------------|
| DeepSeek-V1 | 4K | 标准 RoPE |
| DeepSeek-V2 | 128K | 解耦 RoPE + YaRN |
| DeepSeek-V3 | 128K | 解耦 RoPE + 改进 YaRN |

**YaRN 在 DeepSeek 中的应用**：

YaRN（Yet another RoPE extensioN）通过分频段处理 RoPE 的外推：

$$\theta_i' = \begin{cases}
\theta_i, & \lambda(\theta_i) > \beta \\
\frac{\theta_i}{s}, & \lambda(\theta_i) < \alpha \\
\text{插值}(\theta_i, s), & \text{otherwise}
\end{cases}$$

其中：
- $\lambda(\theta_i) = \frac{2\pi}{\theta_i}$ 是波长
- $\alpha, \beta$ 是频率边界
- $s$ 是缩放因子

**直觉解释**：
- 高频维度（波长短）：直接外推，这些维度编码局部位置，外推影响小
- 低频维度（波长长）：线性插值缩放，这些维度编码全局位置，需要平滑处理
- 中间频率：在两种策略间平滑过渡

### 2.3 YaRN 与长上下文外推的数学细节

YaRN 的设计基于对 RoPE 外推失败原因的深入分析。以下详细推导其核心方法。

#### Position Interpolation (PI) 的局限

Position Interpolation（Chen et al., 2023）是最朴素的 RoPE 外推方案：

$$\theta_i^{PI} = \frac{\theta_i}{s}, \quad s = \frac{L'}{L}$$

其中 $L$ 是原始训练长度，$L'$ 是目标长度。

**问题**：PI 对所有频率维度使用相同的缩放因子 $s$，但不同频率的维度对缩放的敏感度不同：
- 高频维度（编码局部位置）：缩放后局部位置区分能力大幅下降
- 低频维度（编码全局位置）：缩放是合理的

#### NTK-Aware 插值

NTK-Aware 插值（bloc97, 2023）的核心思想是：**修改 RoPE 的 base 而非直接缩放频率**。

$$\text{base}' = \text{base} \cdot s^{d/(d-2)}$$

这等价于对不同频率维度施加不同的缩放：

$$\theta_i^{NTK} = (\text{base}')^{-2(i-1)/d} = \text{base}^{-2(i-1)/d} \cdot s^{-2(i-1)/(d-2)}$$

**关键性质**：

- 对于 $i = 1$（最高频维度）：$\theta_1^{NTK} = 1 \cdot s^0 = 1$，**完全不缩放**
- 对于 $i = d/2$（最低频维度）：$\theta_{d/2}^{NTK} \approx \theta_{d/2} / s$，**接近线性插值**
- 中间维度：缩放程度平滑过渡

```mermaid
graph LR
    subgraph "NTK-Aware 各维度缩放"
        A["高频维度 (i=1)<br/>几乎不缩放<br/>保留局部精度"]
        B["中间维度<br/>中等缩放<br/>平滑过渡"]
        C["低频维度 (i=d/2)<br/>接近 1/s 缩放<br/>外推长距离"]
    end
    A --> B --> C
```

#### 动态 NTK (Dynamic NTK)

动态 NTK 在推理时根据当前序列长度动态调整 base：

$$\text{base}'(n) = \text{base} \cdot \left(\frac{\alpha \cdot n}{L}\right)^{d/(d-2)}$$

其中 $n$ 是当前序列位置，$\alpha$ 是调节因子。

**优势**：序列长度在训练范围内时不做任何修改，仅在超出训练长度后才开始渐进式缩放。

#### YaRN 的完整方案

YaRN 在 NTK-Aware 的基础上引入了两个额外改进：

**1. 分频段策略（已在 2.2 节描述）**：

将频率维度分为三个区域：
- $\lambda(\theta_i) > \beta$：高频区域，直接外推
- $\lambda(\theta_i) < \alpha$：低频区域，线性插值
- 中间区域：线性混合

其中 $\alpha, \beta$ 通常设为 $\alpha = 1, \beta = 32$。

**2. 注意力缩放因子**：

YaRN 发现外推后注意力分数的方差会增大，因此引入温度修正：

$$\text{score}(q_m, k_n) = \frac{(\mathbf{R}_m' \mathbf{q})^T (\mathbf{R}_n' \mathbf{k})}{\sqrt{d_k}} \cdot \frac{1}{\sqrt{t}}$$

其中 $t = 0.1 \ln(s) + 1$ 是与缩放因子 $s$ 相关的温度修正项。

**直觉**：外推使得 Q/K 向量在旋转后的点积分布发生变化，温度修正将其恢复到合理范围。

**YaRN 效果总结**：

| 方法 | 4K→16K | 4K→64K | 4K→128K | 需要微调 |
|------|:------:|:------:|:-------:|:------:|
| PI | 较好 | 差 | 很差 | 是 |
| NTK-Aware | 好 | 中等 | 差 | 可选 |
| Dynamic NTK | 好 | 较好 | 中等 | 否 |
| YaRN | 好 | 好 | 较好 | 少量 |

### 2.4 MLA 的 KV Cache 效率

MLA 的位置编码设计直接影响推理效率：

| 方案 | KV Cache大小 (每层每token) | 说明 |
|------|--------------------------|------|
| 标准 MHA | $2 \times n_h \times d_h$ | 完整 K, V |
| GQA (Llama 3) | $2 \times n_{kv} \times d_h$ | 共享 K, V |
| MQA | $2 \times d_h$ | 单头 K, V |
| MLA (DeepSeek) | $d_c + d_r$ | 压缩表示 + RoPE 部分 |

以 DeepSeek-V2 为例：
- $n_h = 128$, $d_h = 128$, $d_c = 512$, $d_r = 64$
- 标准 MHA: $2 \times 128 \times 128 = 32768$
- MLA: $512 + 64 = 576$（**约 57 倍压缩**）

---

## 3. Anthropic 视角

### 3.1 Claude 的上下文窗口演进

Claude 在上下文窗口方面经历了快速扩展：

```mermaid
graph LR
    A["Claude 1<br/>2023.03<br/>8K tokens"] --> B["Claude 2<br/>2023.07<br/>100K tokens"]
    B --> C["Claude 3<br/>2024.03<br/>200K tokens"]
    C --> D["Claude 3.5<br/>2024.06<br/>200K tokens"]
```

**从 8K 到 200K 的技术挑战**：

$$\text{注意力计算量} = O(n^2 \cdot d), \quad n: \text{序列长度}$$

200K 相对于 8K 的计算量增长：$(200/8)^2 = 625\text{倍}$

**可能的技术手段**（基于公开信息推测）：
1. 位置编码需支持 200K 长度的外推
2. 注意力计算需要高效实现（如 Flash Attention 变体）
3. KV Cache 管理需要特殊优化
4. 可能使用滑动窗口 + 全局注意力的混合策略

> **注**：Claude 的具体位置编码方案未公开。从其 200K 上下文窗口的性能来看，Anthropic 很可能在位置编码的长度外推方面有独特的工程创新。

### 3.2 嵌入空间与可解释性：Superposition 深度分析

Anthropic 在嵌入空间研究方面的核心贡献是**Superposition 假设**，这是理解神经网络内部表示的关键突破。

#### Toy Models of Superposition (Elhage et al., 2022)

**实验设置**：

研究使用了极简模型来理解 Superposition：

$$y = \text{ReLU}(W^T W x + b)$$

其中 $x \in \mathbb{R}^n$ 是输入特征，$W \in \mathbb{R}^{m \times n}$ 是权重矩阵，$m < n$（嵌入维度小于特征数）。

**核心发现**：

1. **特征密度与稀疏性的关系**：

$$\text{Superposition程度} \propto \frac{\text{特征稀疏度}}{1 - \text{特征稀疏度}}$$

特征越稀疏（出现频率越低），模型越倾向于使用 Superposition。

2. **相变现象**：

当特征重要性超过某个阈值时，模型从"不表示该特征"突然跳变到"以Superposition方式表示"。这是一个**一阶相变**。

```mermaid
graph LR
    subgraph "特征重要性 vs 表示方式"
        A["低重要性<br/>不表示"] -->|相变| B["中等重要性<br/>Superposition"]
        B -->|连续过渡| C["高重要性<br/>专用维度"]
    end
```

3. **几何结构**：

在 Superposition 中，特征向量倾向于形成特定的几何结构：
- 2D 中 3 个特征 → 正三角形的顶点（$120°$ 夹角）
- 2D 中 4 个特征 → 正方形的顶点（$90°$ 夹角）
- 高维中 → 各种正多胞体（polytope）的顶点

**数学解释**：

最大化 $n$ 个单位向量在 $m$ 维空间中的最小夹角等价于**Tammes 问题**（球面上的最优分布问题），这是一个经典的组合几何问题。

#### 对词嵌入理解的影响

Superposition 假设解释了词嵌入中的多个现象：

| 现象 | Superposition 解释 |
|------|-------------------|
| 512维能编码10万+词汇的语义 | 稀疏语义特征在嵌入空间中叠加 |
| 类比关系（king - man + woman ≈ queen）| 语义方向在叠加空间中近似线性 |
| 某些方向无法清晰解释 | Superposition 导致的特征干扰 |
| 低频词的表示质量差 | 低频特征的 Superposition 恢复不精确 |

### 3.3 稀疏自编码器（SAE）与嵌入分析

Anthropic 开发了稀疏自编码器（Sparse Autoencoder, SAE）来"解开"Superposition：

**SAE 的数学形式**：

$$f(x) = \text{ReLU}(W_{enc} x + b_{enc})$$

$$\hat{x} = W_{dec} f(x) + b_{dec}$$

其中 $f(x) \in \mathbb{R}^{n_{features}}$，$n_{features} \gg d_{model}$。

**目标函数**：

$$\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda \|f(x)\|_1$$

- 重建损失：保持信息
- L1 稀疏约束：迫使大多数特征不激活

**与嵌入的关系**：

```mermaid
graph TB
    A["嵌入向量 x ∈ R^d<br/>(Superposition 状态)"] --> B["SAE 编码器<br/>W_enc ∈ R^{n×d}"]
    B --> C["稀疏特征 f(x) ∈ R^n<br/>(n >> d, 大部分为0)"]
    C --> D["SAE 解码器<br/>W_dec ∈ R^{d×n}"]
    D --> E["重建向量 x̂ ∈ R^d"]

    C --> F["可解释特征<br/>每个非零维度对应<br/>一个语义概念"]
```

**研究成果示例**：

在 Claude 模型的中间层应用 SAE，Anthropic 发现了可解释的特征，例如：
- "旧金山" 特征：在提到旧金山相关内容时激活
- "代码错误" 特征：在检测代码缺陷时激活
- "礼貌拒绝" 特征：在模型需要拒绝请求时激活

这些发现证明了嵌入空间确实以 Superposition 的方式编码了可解释的概念。

### 3.4 嵌入维度选择的理论指导

Anthropic 的研究为嵌入维度选择提供了理论指导：

**Johnson-Lindenstrauss 引理的类比**：

在随机投影中，$n$ 个点可以在 $O(\log n / \epsilon^2)$ 维空间中保持距离关系。

**Superposition 的类似结论**：

当特征稀疏度为 $S$（每个样本中活跃特征的比例），$n$ 个特征可以在 $m$ 维空间中叠加表示，条件为：

$$m \geq c \cdot S \cdot n$$

其中 $c$ 是与干扰容忍度相关的常数。

**实践意义**：
- 如果语义特征足够稀疏（每个 token 只需少量特征），较低的嵌入维度就足够
- 这解释了为什么 768 维（BERT）或 4096 维（Llama 2）就能表示极其丰富的语义

---

## 4. 前沿话题

### 4.1 上下文窗口的理论极限

**当前的实践上限**：

| 模型 | 上下文长度 | 位置编码 |
|------|-----------|---------|
| GPT-4 Turbo | 128K | 未公开 |
| Claude 3.5 | 200K | 未公开 |
| Gemini 1.5 | 1M+ | 未公开 |
| Yi-6B-200K | 200K | RoPE + NTK |
| Yarn-Llama-2-128K | 128K | YaRN |

**理论瓶颈**：

1. **注意力复杂度**：$O(n^2)$ 的计算和内存

$$\text{内存} = O(n^2 \cdot n_{heads}) \quad \text{（注意力矩阵）}$$

2. **位置编码的精度衰减**：

对于 RoPE，随着距离增大，注意力权重衰减：

$$|\text{score}(m, n)| \propto \text{decay}(|m - n|)$$

当 $|m - n|$ 极大时（如 > 100K），位置编码的区分能力下降。

3. **信息瓶颈**：

模型实际使用的有效信息量可能远小于上下文窗口大小。研究表明，即使提供 200K tokens，模型在中间位置的信息利用率显著低于首尾位置（"Lost in the Middle" 效应）。

### 4.2 无限上下文的探索方向

#### Infini-Attention (Google, 2024)

Infini-Attention 将压缩记忆与标准注意力结合：

$$A_{out} = \sigma(A_{dot}) V + (1 - \sigma) A_{mem}$$

其中：
- $A_{dot}$：标准点积注意力（局部）
- $A_{mem}$：压缩记忆注意力（全局）
- $\sigma$：学习的门控参数

**关键思想**：用有限的压缩记忆表示无限长的历史，避免 $O(n^2)$ 的计算。

#### Ring Attention (UC Berkeley, 2023)

Ring Attention 解决的是**分布式长序列计算**问题：

```mermaid
graph LR
    subgraph "GPU 0"
        A1["块 0 的 Q"]
        A2["计算 Q0×K0"]
    end
    subgraph "GPU 1"
        B1["块 1 的 Q"]
        B2["计算 Q1×K1"]
    end
    subgraph "GPU 2"
        C1["块 2 的 Q"]
        C2["计算 Q2×K2"]
    end

    A2 -->|"传递 KV"| B2 -->|"传递 KV"| C2 -->|"传递 KV"| A2
```

**核心思想**：
1. 将长序列分割到多个 GPU
2. KV 块在 GPU 之间环形传递
3. 每个 GPU 只需要持有自己的 Q 块和当前的 KV 块
4. 总内存：$O(n / p)$，其中 $p$ 是 GPU 数量

#### Streaming LLM (MIT, 2023)

Streaming LLM 发现了"Attention Sink"现象：

- 第一个 token 总是获得大量注意力权重，即使它语义无关
- 保留前几个 token（Attention Sink）+ 最近的 token 窗口 = 无限长度推理

$$\text{保留的 token} = \{\text{前 } k \text{ 个 sink tokens}\} \cup \{\text{最近 } L \text{ 个 tokens}\}$$

### 4.3 嵌入空间的几何结构研究

#### 流形假设

词嵌入空间并非均匀分布，而是集中在低维流形上：

$$\text{嵌入向量} \in \mathcal{M} \subset \mathbb{R}^d, \quad \dim(\mathcal{M}) \ll d$$

**实证支持**：
- 对词嵌入进行 PCA，前几个主成分解释了大部分方差
- 词嵌入的有效维度（用参与比衡量）远小于名义维度
- 不同语义类别的词占据嵌入空间的不同区域

#### 各向异性问题

研究发现，预训练语言模型的嵌入空间存在**各向异性**（anisotropy）问题：

$$\text{各向异性} = \frac{\sigma_{max}^2}{\sum_i \sigma_i^2}$$

其中 $\sigma_i$ 是嵌入矩阵的奇异值。

**影响**：
- 嵌入向量集中在一个窄锥中
- 余弦相似度普遍偏高，区分能力下降
- 解决方案：后处理白化（whitening）或对比学习

#### 嵌入空间的拓扑结构

最新研究开始用**拓扑数据分析**（TDA）方法分析嵌入空间：

- **持续同调**（Persistent Homology）：检测嵌入空间中的"洞"
- **Betti 数**：衡量嵌入空间的拓扑复杂度
- 发现：语义相关的词形成连通的拓扑结构，语义不相关的词被"洞"分隔

### 4.4 多模态嵌入统一

随着多模态模型的发展，将不同模态映射到统一的嵌入空间成为重要趋势：

**CLIP 范式**：

$$\text{similarity}(\text{image}, \text{text}) = \frac{f_{image}(x)^T f_{text}(y)}{\|f_{image}(x)\| \|f_{text}(y)\|}$$

通过对比学习，将图像和文本映射到同一个嵌入空间。

**Gemini 的统一 Embedding**：

Google 的 Gemini 模型将多种模态（文本、图像、音频、视频）映射到同一嵌入空间，使得跨模态推理成为可能：

```mermaid
graph TB
    subgraph "输入模态"
        A1[文本 tokens]
        A2[图像 patches]
        A3[音频 frames]
    end

    subgraph "模态编码器"
        B1[Text Embedding]
        B2[Vision Encoder]
        B3[Audio Encoder]
    end

    subgraph "统一嵌入空间"
        C["共享 Transformer<br/>统一表示空间 R^d"]
    end

    A1 --> B1 --> C
    A2 --> B2 --> C
    A3 --> B3 --> C
```

**挑战**：
1. 不同模态的信息密度差异极大（一个图像 patch ≠ 一个文本 token）
2. 位置编码需要适配不同模态的空间结构
3. 嵌入维度需要同时满足所有模态的表示需求

---

## 参考资料

### 论文
1. Vaswani et al. (2017). *Attention Is All You Need*.
2. Su et al. (2021). *RoFormer: Enhanced Transformer with Rotary Position Embedding*.
3. Press et al. (2022). *Train Short, Test Long: Attention with Linear Biases Enables Input Length Generalization*.
4. Raffel et al. (2020). *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. (T5)
5. Elhage et al. (2022). *Toy Models of Superposition*. Anthropic.
6. Bricken et al. (2023). *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*. Anthropic.
7. Templeton et al. (2024). *Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet*. Anthropic.
8. Munkhdalai et al. (2024). *Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention*. Google.
9. Liu et al. (2023). *Ring Attention with Blockwise Transformers for Near-Infinite Context*.
10. Xiao et al. (2023). *Efficient Streaming Language Models with Attention Sinks*.
11. Peng et al. (2023). *YaRN: Efficient Context Window Extension of Large Language Models*.
12. DeepSeek-AI (2024). *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*.
13. Chen et al. (2023). *Extending Context Window of Large Language Models via Position Interpolation*.
14. Meta AI (2024). *Llama 3 Model Card and Technical Report*.
15. Team Gemma (2024). *Gemma 2: Improving Open Language Models at a Practical Size*. Google.

### 博客
1. [Rotary Positional Embeddings - EleutherAI](https://blog.eleuther.ai/rotary-embeddings/)
2. [Transformer Circuits Thread - Anthropic](https://transformer-circuits.pub/)
3. [Understanding Superposition - Anthropic](https://transformer-circuits.pub/2022/toy_model/index.html)
4. [NTK-Aware Scaled RoPE - bloc97](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/)
5. [Extending Context Window of Llama - kaiokendev](https://kaiokendev.github.io/context)
