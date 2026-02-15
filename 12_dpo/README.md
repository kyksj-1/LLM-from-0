# 模块12：DPO 及变体 -- 直接偏好优化

> RLHF 需要训练四个模型、使用不稳定的强化学习算法，工程复杂度极高。DPO（Direct Preference Optimization）发现了一个深刻的数学等价关系：**奖励函数可以从策略本身隐式恢复**，从而将强化学习问题转化为一个简单的分类问题。本章将完整推导这一数学变换，并系统介绍 DPO 及其主要变体。

---

## 目录

- [1. DPO 的动机](#1-dpo-的动机)
- [2. DPO 的完整数学推导](#2-dpo-的完整数学推导)
- [3. DPO 的直觉理解](#3-dpo-的直觉理解)
- [4. DPO 变体](#4-dpo-变体)
- [5. 各方法全面对比](#5-各方法全面对比)
- [6. Google 的偏好优化实践](#6-google-的偏好优化实践)
- [7. DeepSeek 的 GRPO](#7-deepseek-的-grpo)
- [8. Anthropic 的偏好优化视角](#8-anthropic-的偏好优化视角)
- [9. 项目实践](#9-项目实践)
- [10. 本章小结](#10-本章小结)

---

## 1. DPO 的动机

### 1.1 RLHF 的工程复杂性

回顾模块 11 中介绍的 RLHF 流程，完成一次完整的偏好优化需要：

```mermaid
graph TB
    subgraph "RLHF 完整流程"
        A["1. SFT 模型<br/>（监督微调基座）"] --> B["2. 奖励模型 RM<br/>（学习人类偏好）"]
        B --> C["3. 策略模型 π<br/>（被优化的模型）"]
        A --> D["4. 参考模型 π_ref<br/>（KL 正则化锚点）"]
        C --> E["PPO 训练循环"]
        D --> E
        B --> E
    end

    style E fill:#ff6b6b,color:#fff
```

**RLHF 的四大痛点**：

| 问题 | 具体表现 | 影响 |
|------|---------|------|
| **模型数量多** | 同时维护 SFT、RM、策略、参考 四个模型 | 显存占用巨大 |
| **训练不稳定** | PPO 对超参数敏感，奖励黑客（reward hacking） | 调参困难、结果不可复现 |
| **采样开销大** | 每个训练步都需要从策略模型生成完整回答 | 训练速度慢 |
| **工程复杂** | 需要协调多个模型的前向/反向传播 | 实现门槛高 |

### 1.2 核心洞察：从 RL 到分类

DPO 的核心洞察来自一个关键的数学发现：

> **在 KL 约束的 RLHF 目标函数下，最优策略有闭式解。通过这个闭式解，我们可以将奖励函数表示为策略的函数，从而完全绕过奖励模型的显式训练。**

这意味着：
- 不需要单独训练奖励模型
- 不需要 PPO 等强化学习算法
- 不需要从策略模型在线采样
- 偏好优化变成了一个**二元分类**问题

```mermaid
graph LR
    subgraph "RLHF 流程"
        direction TB
        A1["人类偏好数据"] --> B1["训练奖励模型"]
        B1 --> C1["PPO 强化学习"]
        C1 --> D1["优化后策略"]
    end

    subgraph "DPO 流程"
        direction TB
        A2["人类偏好数据"] --> B2["直接优化策略<br/>（分类损失）"]
        B2 --> D2["优化后策略"]
    end

    style C1 fill:#ff6b6b,color:#fff
    style B2 fill:#4ecdc4,color:#fff
```

**类比**：RLHF 就像是先雇一位评委（奖励模型）学习评分标准，然后让选手（策略模型）反复表演并根据评分改进。DPO 则是直接告诉选手"A 比 B 好"，让选手自己琢磨什么样的表现更受欢迎——跳过了评委这个中间环节。

### 1.3 DPO 论文的关键贡献

Rafailov et al. (2023) 在论文 "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" 中提出了 DPO，其关键贡献包括：

1. **数学等价性证明**：证明 DPO 与 RLHF 在理论上等价
2. **简化训练流程**：从四个模型减少到两个（策略模型 + 参考模型）
3. **稳定的训练**：使用标准的交叉熵损失，避免了 RL 的不稳定性
4. **实验验证**：在多个基准上达到与 RLHF 相当甚至更优的性能

---

## 2. DPO 的完整数学推导

> 本节是整个模块的核心。我们将从 RLHF 的目标函数出发，经过严格的数学推导，一步步得到 DPO 的损失函数。每一步都会解释其数学动机，确保读者能够理解"为什么这样变换"。

### 2.1 起点：RLHF 的目标函数

RLHF 的核心优化目标是：在最大化奖励的同时，不让策略偏离参考模型太远。数学表达为：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)}\left[r(x, y)\right] - \beta \, \text{KL}\left[\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right]$$

其中：
- $\pi$：我们要优化的策略（语言模型）
- $\pi_{\text{ref}}$：参考策略（通常是 SFT 后的模型），作为正则化锚点
- $r(x, y)$：奖励函数，对 prompt $x$ 和回答 $y$ 打分
- $\beta > 0$：KL 惩罚系数，控制策略偏离参考模型的程度
- $\mathcal{D}$：prompt 的分布

**展开 KL 散度**：

$$\text{KL}\left[\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)\right] = \mathbb{E}_{y \sim \pi(\cdot|x)}\left[\log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

将 KL 展开后，目标函数变为：

$$\max_{\pi} \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi(\cdot|x)}\left[r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

**为什么需要 KL 约束？**
- 没有约束时，策略会走向极端：只生成获得最高奖励的回答，失去多样性
- KL 约束保证了策略不会偏离参考模型太远，维持语言质量
- $\beta$ 控制了"探索新回答"和"保持原有能力"之间的平衡

### 2.2 关键推导：最优策略的闭式解

**目标**：对于给定的奖励函数 $r(x,y)$，求解上述优化问题的最优策略 $\pi^*$。

**第一步：将优化问题写成逐 prompt 的形式**

由于期望是关于 $x$ 的，我们可以对每个 $x$ 独立优化：

$$\max_{\pi(\cdot|x)} \sum_{y} \pi(y|x) \left[r(x, y) - \beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}\right]$$

同时满足约束 $\sum_y \pi(y|x) = 1$，$\pi(y|x) \geq 0$。

**第二步：整理被优化的表达式**

将上式展开：

$$\sum_{y} \pi(y|x) \left[r(x, y) - \beta \log \pi(y|x) + \beta \log \pi_{\text{ref}}(y|x)\right]$$

$$= \sum_{y} \pi(y|x) r(x, y) - \beta \sum_{y} \pi(y|x) \log \pi(y|x) + \beta \sum_{y} \pi(y|x) \log \pi_{\text{ref}}(y|x)$$

注意到中间一项 $-\sum_y \pi(y|x) \log \pi(y|x) = H(\pi(\cdot|x))$ 正是策略分布的熵。

**第三步：使用变分法 / Lagrange 乘子法求解**

构造 Lagrangian（引入约束 $\sum_y \pi(y|x) = 1$）：

$$\mathcal{L} = \sum_{y} \pi(y|x) \left[r(x, y) - \beta \log \pi(y|x) + \beta \log \pi_{\text{ref}}(y|x)\right] + \lambda\left(\sum_y \pi(y|x) - 1\right)$$

对 $\pi(y|x)$ 求导并令其为零：

$$\frac{\partial \mathcal{L}}{\partial \pi(y|x)} = r(x, y) - \beta \log \pi(y|x) - \beta + \beta \log \pi_{\text{ref}}(y|x) + \lambda = 0$$

解出 $\pi(y|x)$：

$$\beta \log \pi(y|x) = r(x, y) + \beta \log \pi_{\text{ref}}(y|x) - \beta + \lambda$$

$$\log \pi(y|x) = \frac{1}{\beta} r(x, y) + \log \pi_{\text{ref}}(y|x) + \frac{\lambda - \beta}{\beta}$$

$$\pi(y|x) = \pi_{\text{ref}}(y|x) \cdot \exp\left(\frac{r(x, y)}{\beta}\right) \cdot \exp\left(\frac{\lambda - \beta}{\beta}\right)$$

**第四步：利用归一化约束确定常数**

利用 $\sum_y \pi(y|x) = 1$：

$$\exp\left(\frac{\lambda - \beta}{\beta}\right) \cdot \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right) = 1$$

定义**配分函数**（partition function）：

$$Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x, y)}{\beta}\right)$$

则 $\exp\left(\frac{\lambda - \beta}{\beta}\right) = \frac{1}{Z(x)}$。

**最终结果：最优策略的闭式解**：

$$\boxed{\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)}$$

**直觉理解**：
- 最优策略是参考策略的一个**加权版本**
- 奖励高的回答 $y$，其概率被指数放大
- $\beta$ 越小，放大效果越强（更激进地追求高奖励）
- $Z(x)$ 是归一化常数，保证概率和为 1

```mermaid
graph LR
    A["参考策略 π_ref(y|x)"] --> B["乘以 exp(r(x,y)/β)"]
    B --> C["除以 Z(x) 归一化"]
    C --> D["最优策略 π*(y|x)"]

    style B fill:#f9d56e,color:#000
    style D fill:#4ecdc4,color:#fff
```

### 2.3 核心变换：从策略恢复隐式奖励

**这一步是 DPO 最关键的数学洞察**。

从 2.2 节的闭式解出发，我们要做一个**逆向操作**：既然最优策略可以由奖励函数表示，那么奖励函数也可以由策略表示。

对 $\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$ 两边取对数：

$$\log \pi^*(y|x) = \log \pi_{\text{ref}}(y|x) + \frac{1}{\beta} r(x,y) - \log Z(x)$$

移项，解出 $r(x,y)$：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**这就是隐式奖励公式**。它告诉我们：

$$\boxed{r(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)}$$

> 当 $\pi_\theta = \pi^*$ 时，上式精确成立。DPO 的思路是：用参数化的 $\pi_\theta$ 去逼近 $\pi^*$，此时上式近似成立。

**直觉理解**：
- $\beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是策略相对于参考模型的对数概率比，可以视为"隐式奖励"
- 如果 $\pi_\theta$ 比 $\pi_{\text{ref}}$ 更倾向于生成 $y$，说明 $y$ 的隐式奖励更高
- $\beta \log Z(x)$ 是只依赖于 $x$ 的常数项，在接下来的推导中会被消去

### 2.4 代入 Bradley-Terry 模型：得到 DPO 损失

**Bradley-Terry 偏好模型**回顾（在模块 11 中已介绍）：

给定 prompt $x$、偏好回答 $y_w$ 和非偏好回答 $y_l$，人类偏好概率为：

$$P(y_w \succ y_l | x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

其中 $\sigma$ 是 sigmoid 函数 $\sigma(z) = \frac{1}{1+e^{-z}}$。

**将隐式奖励代入**：

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \cancel{\beta \log Z(x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \cancel{\beta \log Z(x)}$$

**关键观察**：$\beta \log Z(x)$ 只依赖于 $x$，在 $y_w$ 和 $y_l$ 的奖励差中被完美消去。

化简：

$$r(x, y_w) - r(x, y_l) = \beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]$$

**DPO 的损失函数**（最大化人类偏好的负对数似然）：

$$\boxed{\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]}$$

### 2.5 推导流程总结

```mermaid
graph TB
    A["RLHF 目标函数<br/>max E[r(x,y)] - β KL(π||π_ref)"]
    A -->|"Lagrange 乘子法"| B["最优策略闭式解<br/>π*(y|x) = π_ref(y|x)exp(r(x,y)/β) / Z(x)"]
    B -->|"两边取对数，解出 r"| C["隐式奖励<br/>r(x,y) = β log(π/π_ref) + β log Z(x)"]
    C -->|"代入 Bradley-Terry 模型"| D["奖励差<br/>r(y_w) - r(y_l) = β[log(π(y_w)/π_ref(y_w)) - log(π(y_l)/π_ref(y_l))]"]
    D -->|"Z(x) 项消去！"| E["DPO 损失函数<br/>L = -E[log σ(β log(π(y_w)/π_ref(y_w)) - β log(π(y_l)/π_ref(y_l)))]"]

    style A fill:#ff6b6b,color:#fff
    style E fill:#4ecdc4,color:#fff
```

**推导的关键数学步骤回顾**：

| 步骤 | 数学操作 | 动机 |
|------|---------|------|
| 1 | 展开 KL 散度 | 将约束目标写成无约束形式 |
| 2 | Lagrange 乘子法 | 在归一化约束下求最优分布 |
| 3 | 配分函数 $Z(x)$ | 吸收归一化常数 |
| 4 | 取对数 + 移项 | 从最优策略反解奖励函数 |
| 5 | 代入 Bradley-Terry | 将奖励差转化为偏好概率 |
| 6 | $Z(x)$ 消去 | 避免计算不可行的配分函数 |

---

## 3. DPO 的直觉理解

### 3.1 梯度分析：DPO 在做什么？

为了理解 DPO 的训练行为，我们对损失函数求梯度。

DPO 损失为：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma(\hat{r}_w - \hat{r}_l)$$

其中 $\hat{r}_w = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)}$，$\hat{r}_l = \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$ 分别是偏好回答和非偏好回答的隐式奖励。

对 $\theta$ 求梯度：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \cdot \underbrace{\sigma(\hat{r}_l - \hat{r}_w)}_{\text{加权系数}} \cdot \left[\underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{增加偏好回答概率}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{降低非偏好回答概率}}\right]$$

**梯度的三个组成部分**：

1. **增加偏好回答概率**：$\nabla_\theta \log \pi_\theta(y_w|x)$ -- 让模型更倾向于生成好的回答
2. **降低非偏好回答概率**：$-\nabla_\theta \log \pi_\theta(y_l|x)$ -- 让模型远离坏的回答
3. **自适应加权**：$\sigma(\hat{r}_l - \hat{r}_w)$ -- 当模型已经能很好地区分好坏回答时（$\hat{r}_w \gg \hat{r}_l$），这个系数接近 0，梯度自动减小

**类比**：想象一个学生做选择题。当学生已经能自信地选出正确答案时（隐式奖励差距大），继续做同样的题对学习没什么帮助（梯度小）。但当学生还搞不清楚哪个更好时（隐式奖励差距小），这道题就很有学习价值（梯度大）。

```mermaid
graph TB
    subgraph "DPO 梯度的直觉"
        A["偏好数据 (x, y_w, y_l)"]
        A --> B["计算隐式奖励差<br/>Δr = r_w - r_l"]
        B -->|"Δr 大 → 已经学好了"| C["梯度小<br/>（自适应衰减）"]
        B -->|"Δr 小 → 还没学好"| D["梯度大<br/>（加强学习）"]
    end
```

### 3.2 与 RLHF-PPO 的等价性

DPO 和 RLHF 在理论上是等价的：它们在最优解处收敛到相同的策略。但在实践中有所不同：

| 维度 | RLHF (PPO) | DPO |
|------|-----------|-----|
| **奖励模型** | 显式训练，独立优化 | 隐式，嵌入在策略中 |
| **优化方式** | 在线 RL，从策略采样 | 离线，使用固定数据集 |
| **稳定性** | 需要精心调参 | 通常更稳定 |
| **理论保证** | 在分布内最优 | 在离线数据覆盖范围内最优 |
| **分布偏移** | 通过在线采样缓解 | 可能受离线数据分布限制 |
| **数据利用** | 需要大量在线生成 | 高效利用离线偏好数据 |

**关键区别**：

- **在线 vs 离线**：RLHF 在训练过程中不断从策略模型采样新数据，可以探索数据分布；DPO 使用固定的离线偏好数据集，无法探索新的回答空间
- **奖励泛化**：RLHF 的奖励模型可以对未见过的回答打分；DPO 的隐式奖励只在训练数据的支持集上有效

### 3.3 DPO 的潜在问题

1. **分布偏移（Distribution Shift）**：离线数据中的回答可能与当前策略的生成分布不同，导致优化方向偏差
2. **过拟合偏好数据**：DPO 可能过度拟合训练集中的偏好标注，尤其当数据量较小时
3. **长度偏差（Length Bias）**：DPO 可能倾向于生成更长的回答（因为对数概率与长度相关）

这些问题催生了一系列 DPO 变体，我们将在第 4 节详细介绍。

---

## 4. DPO 变体

### 4.1 IPO（Identity Preference Optimization）

#### 动机

DPO 基于 Bradley-Terry 模型的假设：人类偏好概率等于奖励差的 sigmoid。但在实际中，这个假设可能不成立（例如，标注者对非常相似的回答可能有噪声偏好）。IPO 发现 DPO 在训练过程中可能会**过拟合**：当训练足够久时，隐式奖励差趋向无穷大，导致策略退化。

#### 数学形式

IPO 直接优化偏好回答和非偏好回答的隐式奖励差，并添加正则化项：

$$\mathcal{L}_{\text{IPO}} = \mathbb{E}_{(x, y_w, y_l)}\left[\left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

**直觉**：
- IPO 希望隐式奖励差接近一个目标值 $\frac{1}{2\beta}$，而不是无限增大
- 使用**均方误差**损失代替 DPO 的交叉熵损失
- 当隐式奖励差达到目标时，梯度为零，防止过度优化

**与 DPO 的对比**：

| 特性 | DPO | IPO |
|------|-----|-----|
| 损失函数 | 交叉熵（log sigmoid） | 均方误差 |
| 优化目标 | 最大化偏好概率 | 隐式奖励差接近目标值 |
| 过拟合风险 | 可能过拟合 | 内置正则化 |
| 前提假设 | 需要 Bradley-Terry 模型 | 不依赖特定偏好模型 |

### 4.2 KTO（Kahneman-Tversky Optimization）

#### 动机

DPO、IPO 等方法都需要**成对偏好数据**（即对同一个 prompt 有一好一坏两个回答）。但在实际中，收集成对数据成本很高。很多时候我们只有**二元反馈**：某个回答是"好"还是"坏"（如点赞/点踩）。

KTO 的关键洞察：利用行为经济学中的**前景理论**（Prospect Theory，由 Kahneman 和 Tversky 提出），设计一个不需要成对数据的偏好优化方法。

#### 前景理论简介

前景理论的核心发现：人类对损失的敏感度大于对等量收益的敏感度（损失厌恶）。

```mermaid
graph LR
    subgraph "前景理论"
        A["收益 +100"] -->|"感知价值"| B["+v(100)"]
        C["损失 -100"] -->|"感知价值"| D["-λ v(100)"]
    end

    E["λ > 1：损失比等量收益更'痛'"]
```

#### 数学形式

KTO 的损失函数：

$$\mathcal{L}_{\text{KTO}} = \mathbb{E}_{x, y}\left[\lambda_y \cdot \left(1 - v_\theta(x, y)\right)\right]$$

其中：

对于好的回答（$y = y_w$）：

$$v_\theta(x, y_w) = \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - z_{\text{ref}}\right)$$

对于坏的回答（$y = y_l$）：

$$v_\theta(x, y_l) = \sigma\left(z_{\text{ref}} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

其中 $z_{\text{ref}} = \mathbb{E}_{x', y' \sim \mathcal{D}}\left[\beta \text{KL}\left(\pi_\theta(y'|x') \| \pi_{\text{ref}}(y'|x')\right)\right]$ 是参考点（reference point），$\lambda_y$ 对好回答和坏回答采用不对称权重（反映损失厌恶）。

**直觉**：
- 对于好回答：希望 $\pi_\theta$ 比 $\pi_{\text{ref}}$ 更倾向于生成它（增大隐式奖励）
- 对于坏回答：希望 $\pi_\theta$ 比 $\pi_{\text{ref}}$ 更不倾向于生成它（减小隐式奖励）
- 参考点 $z_{\text{ref}}$ 是一个动态基线，帮助模型判断"好到什么程度算好"
- 损失厌恶权重 $\lambda_y$ 让模型对坏回答的惩罚更重

**KTO 的优势**：

| 优势 | 说明 |
|------|------|
| 无需成对数据 | 只需要 $(x, y, \text{label})$ 格式 |
| 更容易收集数据 | 二元反馈比排序容易获取 |
| 理论基础扎实 | 基于经过验证的行为经济学理论 |
| 实际性能优秀 | 论文报告与 DPO 相当的性能 |

### 4.3 ORPO（Odds Ratio Preference Optimization）

#### 动机

DPO 和其变体都需要一个单独的 SFT 阶段来准备参考模型 $\pi_{\text{ref}}$。ORPO 的核心想法是：**将 SFT 和偏好优化统一到一个训练阶段**，直接在语言建模损失上添加偏好信号。

#### 数学形式

ORPO 使用**优势比**（Odds Ratio）来衡量偏好：

首先定义回答 $y$ 在策略 $\pi_\theta$ 下的"赔率"（odds）：

$$\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$$

其中 $P_\theta(y|x) = \exp\left(\frac{1}{|y|}\sum_{t=1}^{|y|}\log \pi_\theta(y_t|x, y_{<t})\right)$ 是回答的平均 token 概率的指数。

优势比（Odds Ratio）定义为：

$$\text{OR}_\theta(y_w, y_l | x) = \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}$$

ORPO 的损失函数：

$$\mathcal{L}_{\text{ORPO}} = \underbrace{\mathcal{L}_{\text{NLL}}}_{\text{标准语言建模损失}} + \lambda \cdot \underbrace{\mathcal{L}_{\text{OR}}}_{\text{优势比偏好损失}}$$

其中：
- $\mathcal{L}_{\text{NLL}} = -\frac{1}{|y_w|}\sum_{t=1}^{|y_w|}\log \pi_\theta(y_{w,t}|x, y_{w,<t})$ 是偏好回答上的交叉熵损失
- $\mathcal{L}_{\text{OR}} = -\log \sigma\left(\log \text{OR}_\theta(y_w, y_l | x)\right)$ 是优势比损失

**直觉**：
- NLL 部分让模型学习语言建模能力（相当于 SFT）
- OR 部分让模型区分好坏回答
- 两者合并后，只需一次训练即可完成 SFT + 偏好优化

**ORPO 的优势**：

| 特性 | 说明 |
|------|------|
| 无需参考模型 | 不需要额外存储 $\pi_{\text{ref}}$ |
| 无需 SFT 阶段 | 一次训练完成两个目标 |
| 显存更省 | 只维护一个模型 |
| 训练更快 | 减少了一整个训练阶段 |

### 4.4 SimPO（Simple Preference Optimization）

#### 动机

SimPO 进一步简化偏好优化方法，解决两个问题：
1. DPO 需要维护参考模型 $\pi_{\text{ref}}$，增加显存开销
2. 对数概率作为隐式奖励存在**长度偏差**：更长的回答通常有更低的对数概率（每多一个 token，概率就要乘一个小于 1 的数）

#### 数学形式

SimPO 使用**长度归一化的对数概率**作为隐式奖励：

$$\hat{r}_{\text{SimPO}}(x, y) = \frac{\beta}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t}) = \frac{\beta}{|y|} \log \pi_\theta(y|x)$$

SimPO 的损失函数：

$$\mathcal{L}_{\text{SimPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\frac{\beta}{|y_w|}\log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|}\log \pi_\theta(y_l|x) - \gamma\right)\right]$$

其中 $\gamma > 0$ 是目标奖励差（target reward margin），确保偏好回答和非偏好回答的隐式奖励之间有一定间隔。

**与 DPO 的关键区别**：

1. **去掉参考模型**：不再需要计算 $\log \pi_{\text{ref}}$，节省显存
2. **长度归一化**：除以 $|y|$ 消除长度偏差
3. **目标奖励差 $\gamma$**：类似于 SVM 中的间隔（margin），防止奖励差过小

**直觉**：SimPO 的隐式奖励就是"模型认为这个回答有多好的平均置信度"——每个 token 的平均对数概率越高，说明模型对这个回答越有信心。

### 4.5 GRPO（Group Relative Policy Optimization）

#### 动机

GRPO 由 DeepSeek 提出，用于 DeepSeek-Math 和 DeepSeek-R1 的训练。它的核心思想是：**去掉 Critic（价值）模型，使用组内相对排序来估计优势函数**。

在标准 PPO 中，需要一个 Critic 模型来估计状态价值函数 $V(s)$，从而计算优势函数 $A(s,a) = Q(s,a) - V(s)$。Critic 模型增加了显存和计算开销。GRPO 的关键洞察是：**对于同一个 prompt，我们可以从策略中采样一组回答，通过组内比较来获得相对优势**。

#### 数学形式

给定 prompt $x$，从当前策略 $\pi_\theta$ 中采样 $G$ 个回答 $\{y_1, y_2, \ldots, y_G\}$，使用外部奖励函数（如基于规则的验证器）获得奖励 $\{r_1, r_2, \ldots, r_G\}$。

**组内归一化优势**：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

**GRPO 的目标函数**：

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G}\left[\min\left(\rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\right) - \beta \, \text{KL}\left(\pi_\theta \| \pi_{\text{ref}}\right)\right]$$

其中 $\rho_i = \frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}$ 是重要性采样比率，$\epsilon$ 是 PPO 的裁剪参数。

**与标准 PPO 的对比**：

```mermaid
graph TB
    subgraph "PPO"
        A1["采样 1 个回答"] --> B1["Critic 模型估计 V(s)"]
        B1 --> C1["A = Q - V"]
        C1 --> D1["PPO 裁剪目标"]
    end

    subgraph "GRPO"
        A2["采样 G 个回答"] --> B2["外部奖励函数打分"]
        B2 --> C2["组内归一化<br/>A = (r - mean) / std"]
        C2 --> D2["PPO 裁剪目标"]
    end

    style B1 fill:#ff6b6b,color:#fff
    style B2 fill:#4ecdc4,color:#fff
```

**GRPO 的优势**：

| 特性 | PPO | GRPO |
|------|-----|------|
| Critic 模型 | 需要 | 不需要 |
| 优势估计 | 通过 Critic | 组内相对排序 |
| 奖励来源 | 奖励模型 | 可用规则验证器 |
| 显存需求 | 高（4个模型） | 较低（2-3个模型） |
| 适用场景 | 通用 | 有可验证奖励的任务（如数学、编程） |

---

## 5. 各方法全面对比

### 5.1 方法对比表

| 方法 | 需要参考模型 | 需要成对数据 | 需要在线采样 | 额外模型 | 训练稳定性 | 主要优势 |
|------|:-----------:|:-----------:|:-----------:|:--------:|:----------:|---------|
| **RLHF (PPO)** | 是 | 是(训练RM) | 是 | Critic + RM | 低 | 理论最优、在线探索 |
| **DPO** | 是 | 是 | 否 | 无 | 高 | 简单稳定 |
| **IPO** | 是 | 是 | 否 | 无 | 高 | 防止过拟合 |
| **KTO** | 是 | 否 | 否 | 无 | 高 | 无需成对数据 |
| **ORPO** | 否 | 是 | 否 | 无 | 高 | SFT+对齐一步完成 |
| **SimPO** | 否 | 是 | 否 | 无 | 高 | 最简化、无长度偏差 |
| **GRPO** | 是 | 否(需要组采样) | 是 | 无Critic | 中 | 去掉Critic、组排序 |

### 5.2 选择指南

```mermaid
graph TB
    A["选择偏好优化方法"] --> B{"有成对偏好数据?"}
    B -->|"是"| C{"计算资源充足?"}
    B -->|"否"| D{"有二元反馈?"}

    C -->|"是"| E{"需要在线探索?"}
    C -->|"否"| F["SimPO<br/>（最省资源）"]

    E -->|"是"| G["RLHF-PPO<br/>（最灵活）"]
    E -->|"否"| H{"担心过拟合?"}

    H -->|"是"| I["IPO"]
    H -->|"否"| J["DPO<br/>（默认选择）"]

    D -->|"是"| K["KTO"]
    D -->|"否"| L{"有可验证奖励?"}

    L -->|"是"| M["GRPO"]
    L -->|"否"| N["回到收集数据"]

    style J fill:#4ecdc4,color:#fff
    style F fill:#f9d56e,color:#000
    style M fill:#a78bfa,color:#fff
```

### 5.3 训练成本对比

| 方法 | 训练阶段 | 模型数量 | 相对显存 | 相对速度 |
|------|---------|:--------:|:-------:|:-------:|
| RLHF (PPO) | SFT + RM + RL | 4 | 4x | 1x (慢) |
| DPO | SFT + DPO | 2 | 2x | 3x |
| IPO | SFT + IPO | 2 | 2x | 3x |
| KTO | SFT + KTO | 2 | 2x | 3x |
| ORPO | ORPO (一步) | 1 | 1x | 5x (快) |
| SimPO | SFT + SimPO | 1 | 1x | 4x |
| GRPO | SFT + GRPO | 2-3 | 2-3x | 2x |

---

## 6. Google 的偏好优化实践

### 6.1 Gemini 中的偏好优化

Google 在 Gemini 系列模型的训练中采用了多阶段偏好优化策略。根据公开技术报告，Gemini 的后训练流程包括：

1. **监督微调（SFT）**：使用高质量指令数据微调基座模型
2. **奖励模型训练**：训练多个奖励模型，覆盖不同维度（如有用性、安全性、准确性）
3. **RLHF**：使用奖励模型指导策略优化

Google 在工业实践中更倾向于 RLHF 而非 DPO，主要原因是：

- **数据量充足**：Google 拥有大规模的人类标注数据，可以支撑奖励模型的训练
- **在线探索**：RLHF 的在线采样可以发现更好的回答分布，避免离线方法的分布偏移
- **多维度奖励**：多个奖励模型可以分别优化不同方面，比 DPO 的单一偏好信号更灵活

### 6.2 RLHF vs DPO 的工业选择

在工业界，RLHF 和 DPO 的选择通常取决于以下因素：

| 因素 | 倾向 RLHF | 倾向 DPO |
|------|----------|---------|
| 数据规模 | 大规模标注团队 | 有限偏好数据 |
| 计算资源 | 充足（多卡、长时间训练） | 受限 |
| 模型规模 | 超大模型（>70B） | 中小模型（<13B） |
| 迭代频率 | 低频、精细调优 | 高频、快速迭代 |
| 安全需求 | 高（需要多维度奖励） | 一般 |

---

## 7. DeepSeek 的 GRPO

### 7.1 GRPO 在 DeepSeek-R1 中的应用

DeepSeek 在其推理模型 DeepSeek-R1 的训练中，采用了 GRPO 作为核心偏好优化方法。这是一个里程碑式的选择，因为它表明：

1. **基于规则的奖励可以替代学习的奖励模型**：对于数学推理和编程任务，可以通过验证答案的正确性来获得奖励信号
2. **无需 Critic 模型**：通过组内相对排序估计优势，大幅降低训练成本
3. **组大小的重要性**：组大小 $G$ 是一个关键超参数，影响优势估计的质量

### 7.2 DeepSeek-R1 的训练流程

```mermaid
graph TB
    A["DeepSeek-V3 基座模型"] --> B["冷启动数据 SFT<br/>（少量高质量推理样本）"]
    B --> C["GRPO 推理训练<br/>（数学/编程任务）"]
    C --> D["拒绝采样 + SFT<br/>（收集高质量推理轨迹）"]
    D --> E["GRPO 全场景训练<br/>（扩展到通用任务）"]
    E --> F["DeepSeek-R1"]

    style C fill:#a78bfa,color:#fff
    style E fill:#a78bfa,color:#fff
```

详细的 GRPO 推导和在 R1 中的具体应用请参见 [advanced.md](./advanced.md)。

---

## 8. Anthropic 的偏好优化视角

### 8.1 Constitutional AI 与偏好优化

Anthropic 的 Constitutional AI（CAI）方法与传统的偏好优化有一个关键区别：**偏好信号来自 AI 自身**（通过宪法原则引导），而非人类标注。

CAI 的偏好优化流程：

1. 使用"宪法原则"（如"回答应该是有帮助的、无害的、诚实的"）生成修改建议
2. AI 根据宪法原则对比原始回答和修改后的回答，产生偏好数据
3. 使用这些 AI 生成的偏好数据进行 RLHF 或 DPO 训练

### 8.2 Anthropic 对 DPO 类方法的评估 [推测]

[推测] 基于 Anthropic 公开的研究方向和技术报告，Anthropic 可能：
- 在内部对比了 RLHF 和 DPO 的效果
- 更倾向于使用在线方法（如 RLHF），因为在线方法可以更好地探索回答空间
- 可能在安全对齐的特定场景中使用 DPO 类方法作为补充

### 8.3 偏好优化中的安全对齐 [推测]

[推测] Anthropic 在安全对齐方面的实践可能包括：
- 使用多轮 CAI 迭代来提高偏好数据的质量
- 结合 DPO/KTO 等方法快速迭代安全相关的偏好优化
- 在有害内容识别方面，可能使用类似 KTO 的二元反馈方法（因为标注"有害/无害"比排序更高效）

> 注意：以上标注 [推测] 的内容基于公开信息的合理推断，不代表 Anthropic 的官方技术路线。具体的技术细节需要参考 Anthropic 发布的技术报告和论文。

---

## 9. 项目实践

### 项目 1：从零推导并实现 DPO 损失函数（难度：⭐⭐ 进阶）

**核心目标**：深入理解 DPO 的数学基础，从推导到代码的完整闭环。

**提供内容：数学推导框架 + 代码验证**

**任务描述**：

1. **手动推导**：在纸上完成 DPO 损失函数的完整推导（参考第 2 节），写出每一步的数学操作和动机
2. **代码实现**：参考 `code/dpo/dpo_loss.py`，实现 DPO 损失函数
3. **数值验证**：构造简单的数值例子，验证以下性质：
   - 当 $\pi_\theta$ 完美区分偏好时（$\pi_\theta(y_w|x) \gg \pi_\theta(y_l|x)$），损失接近 0
   - 当 $\pi_\theta$ 与参考模型相同时（$\pi_\theta = \pi_{\text{ref}}$），损失等于 $\log 2$
   - 当 $\beta$ 增大时，损失对偏好差异更敏感

**关键代码片段**：

```python
import torch
import torch.nn.functional as F

def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # log π_θ(y_w|x)
    policy_rejected_logps: torch.Tensor,  # log π_θ(y_l|x)
    ref_chosen_logps: torch.Tensor,       # log π_ref(y_w|x)
    ref_rejected_logps: torch.Tensor,     # log π_ref(y_l|x)
    beta: float = 0.1,
) -> torch.Tensor:
    """DPO 损失函数核心实现"""
    # 隐式奖励差
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO 损失 = -log sigmoid(r_w - r_l)
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    return loss
```

**验证提示**：
- 使用 `torch.randn` 构造随机的 log probability 张量
- 分别测试偏好回答概率远大于/等于/小于非偏好回答概率的情况
- 画出损失随 $\beta$ 变化的曲线

---

### 项目 2：使用 DPO 对齐一个小模型（难度：⭐⭐ 进阶）

**核心目标**：实战偏好优化，理解完整的 DPO 训练流程。

**提供内容：训练框架 + 数据准备指引**

**任务描述**：

1. **数据准备**：使用 HuggingFace 上的公开偏好数据集（如 `Anthropic/hh-rlhf` 或 `argilla/ultrafeedback-binarized-preferences`）
2. **模型选择**：选择一个小型预训练模型（如 GPT-2 small 或 TinyLlama-1.1B）
3. **训练 DPO**：参考 `code/dpo/dpo_trainer.py` 实现完整的训练循环
4. **评估**：比较 DPO 前后的模型表现

**关键代码片段**：

```python
# 数据准备示意
from datasets import load_dataset

dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")

# 训练循环核心
for batch in dataloader:
    # 前向传播：计算策略和参考模型的 log probability
    policy_chosen_logps = get_logps(model, batch["chosen"])
    policy_rejected_logps = get_logps(model, batch["rejected"])

    with torch.no_grad():
        ref_chosen_logps = get_logps(ref_model, batch["chosen"])
        ref_rejected_logps = get_logps(ref_model, batch["rejected"])

    # 计算 DPO 损失
    loss = dpo_loss(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=0.1
    )

    loss.backward()
    optimizer.step()
```

**评估方法**：
- 使用 win rate（偏好胜率）评估对齐效果
- 对比 DPO 前后在安全性和有用性方面的变化
- 检查 KL 散度，确保策略没有偏离参考模型太远

---

### 项目 3：对比 DPO/KTO/SimPO 的训练效果（难度：⭐⭐⭐ 挑战）

**核心目标**：理解各变体的实际差异。

**提供内容：实验设计 + 评估指标**

**思路**：

1. **实验设计**：
   - 固定模型（如 GPT-2 medium）和数据集
   - 分别使用 DPO、KTO、SimPO 进行偏好优化
   - 控制训练步数和学习率一致

2. **评估维度**：
   - 训练损失曲线
   - 隐式奖励的分布
   - 生成文本的质量（人工评估或 GPT-4 评估）
   - KL 散度（策略偏离程度）
   - 训练速度和显存占用

3. **分析要点**：
   - KTO 在只有二元反馈数据时是否仍然有效？
   - SimPO 是否确实消除了长度偏差？
   - DPO 在小数据集上是否存在过拟合？

```mermaid
graph LR
    A["固定模型 + 数据集"] --> B["DPO 训练"]
    A --> C["KTO 训练"]
    A --> D["SimPO 训练"]
    B --> E["统一评估"]
    C --> E
    D --> E
    E --> F["对比分析报告"]
```

**伪代码**：

```
# 实验主循环
for method in [DPO, KTO, SimPO]:
    model = load_pretrained("gpt2-medium")
    ref_model = copy(model) if method != SimPO else None

    trainer = method.Trainer(
        model=model,
        ref_model=ref_model,
        dataset=preference_data,
        beta=0.1,
        lr=1e-6,
    )

    metrics = trainer.train(num_epochs=3)

    results[method] = {
        "loss_curve": metrics.loss_history,
        "kl_divergence": compute_kl(model, ref_model),
        "generation_quality": evaluate_generations(model),
        "training_time": metrics.total_time,
        "peak_memory": metrics.peak_memory,
    }

compare_and_visualize(results)
```

---

### 项目 4：实现 GRPO 并分析组大小的影响（难度：⭐⭐⭐ 挑战）

**核心目标**：理解 DeepSeek 的 GRPO 创新。

**提供内容：论文关键公式 + 伪代码**

**思路**：

1. **简化场景**：在简单数学题（如加法、乘法）上实现 GRPO
2. **奖励函数**：使用正确性验证作为奖励（答对 +1，答错 -1）
3. **组大小实验**：测试 $G \in \{4, 8, 16, 32, 64\}$ 对训练效果的影响
4. **分析**：
   - 组大小如何影响优势估计的方差？
   - 更大的组是否总是更好？
   - 训练速度和效果之间的权衡

```mermaid
graph TB
    subgraph "GRPO 训练循环"
        A["给定 Prompt x"] --> B["从 π_θ 采样 G 个回答"]
        B --> C["使用验证器给每个回答打分"]
        C --> D["组内归一化：A_i = (r_i - mean) / std"]
        D --> E["PPO 裁剪目标 + KL 正则"]
        E --> F["更新 π_θ"]
        F -->|"下一个 Prompt"| A
    end
```

**关键公式**（来自 DeepSeek-Math 论文）：

$$\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r}) + \epsilon}$$

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G}\sum_{i=1}^{G}\min\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}\hat{A}_i, \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\text{old}}(y_i|x)}, 1-\epsilon, 1+\epsilon\right)\hat{A}_i\right) + \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

**伪代码**：

```
# GRPO 训练循环
for prompt in math_dataset:
    # 1. 采样一组回答
    responses = sample(policy, prompt, num_samples=G)

    # 2. 计算奖励（数学验证）
    rewards = [verify_answer(resp, ground_truth) for resp in responses]

    # 3. 组内归一化
    advantages = (rewards - mean(rewards)) / (std(rewards) + 1e-8)

    # 4. 计算 GRPO 损失
    for (response, advantage) in zip(responses, advantages):
        ratio = policy.prob(response) / old_policy.prob(response)
        clipped_ratio = clip(ratio, 1-eps, 1+eps)
        loss = -min(ratio * advantage, clipped_ratio * advantage)
        loss += beta * kl_penalty(policy, ref_policy, response)

    # 5. 更新策略
    loss.backward()
    optimizer.step()
```

---

## 10. 本章小结

### 核心知识点回顾

1. **DPO 的数学本质**：通过闭式解将奖励函数表示为策略的函数，将 RL 问题转化为分类问题
2. **隐式奖励**：$r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$
3. **DPO 损失函数**：利用 Bradley-Terry 模型和隐式奖励，推导出简洁的损失函数
4. **变体的动机**：每个变体都是为了解决 DPO 的某个具体问题
   - IPO：防止过拟合
   - KTO：去除成对数据需求
   - ORPO：统一 SFT 和偏好优化
   - SimPO：去除参考模型、消除长度偏差
   - GRPO：去除 Critic、使用组排序

### 与相邻模块的关系

```mermaid
graph LR
    A["模块 11: RLHF<br/>（奖励模型 + PPO）"] --> B["模块 12: DPO 及变体<br/>（直接偏好优化）"]
    B --> C["模块 13: CoT 推理<br/>（推理增强）"]

    A -.->|"DPO 简化了 RLHF"| B
    B -.->|"GRPO 用于推理训练"| C
```

### 推荐阅读

| 论文 | 方法 | 年份 |
|------|------|:----:|
| Rafailov et al., "Direct Preference Optimization" | DPO | 2023 |
| Azar et al., "A General Theoretical Paradigm to Understand Learning from Human Feedback" | IPO | 2023 |
| Ethayarajh et al., "KTO: Model Alignment as Prospect Theoretic Optimization" | KTO | 2024 |
| Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model" | ORPO | 2024 |
| Meng et al., "SimPO: Simple Preference Optimization with a Reference-Free Reward" | SimPO | 2024 |
| Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models" | GRPO | 2024 |
| DeepSeek-AI, "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning" | GRPO 应用 | 2025 |
