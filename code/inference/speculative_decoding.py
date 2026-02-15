"""
投机解码（Speculative Decoding）实现

本模块实现了投机解码的核心算法，包括：
- 草稿模型快速生成候选 token
- 目标模型并行验证
- 接受/拒绝采样（保证输出分布与目标模型一致）
- 加速比分析

核心思想:
    用一个小而快的草稿模型（draft model）一次性猜测 gamma 个 token，
    然后用大模型（target model）并行验证。利用接受/拒绝采样保证最终
    输出的分布与纯大模型采样完全一致（零偏差）。

接受概率:
    P(accept x_t) = min(1, p(x_t) / q(x_t))
    其中 p 是目标模型分布，q 是草稿模型分布

正确性保证:
    投机解码产生的分布与目标模型的分布完全一致（数学可证明）

参考论文:
    Leviathan et al., "Fast Inference from Transformers via Speculative Decoding",
    ICML 2023
    Chen et al., "Accelerating Large Language Model Decoding with Speculative
    Sampling", 2023
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
import time


class SimpleLanguageModel:
    """
    简化的语言模型接口

    使用随机概率分布模拟语言模型的行为，用于演示投机解码算法。
    可以通过调整 temperature 和 agreement 参数控制模型行为。
    """

    def __init__(
        self,
        vocab_size: int,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ):
        """
        Args:
            vocab_size: 词汇表大小
            temperature: 温度参数（越低分布越尖锐）
            seed: 随机种子
        """
        self.vocab_size = vocab_size
        self.temperature = temperature
        if seed is not None:
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)
        else:
            self.rng = None

        # 模拟模型的"知识"：一个随机的 logit 偏置矩阵
        self._logit_bias = torch.randn(vocab_size)

    def predict(self, context: torch.Tensor) -> torch.Tensor:
        """
        预测下一个 token 的概率分布

        Args:
            context: 上下文 token 序列 [seq_len]

        Returns:
            概率分布 [vocab_size]
        """
        # 模拟：基于上下文的最后几个 token 生成 logits
        # 实际模型会做完整的前向传播
        logits = self._logit_bias.clone()

        # 基于上下文做简单的偏移（模拟上下文依赖）
        if len(context) > 0:
            last_token = context[-1].item() % self.vocab_size
            logits[last_token] += 2.0  # 自回归偏好
            # 相邻 token 也有较高概率
            for offset in [-1, 1]:
                idx = (last_token + offset) % self.vocab_size
                logits[idx] += 1.0

        # Temperature 缩放
        logits = logits / self.temperature

        # 转为概率分布
        probs = F.softmax(logits, dim=-1)
        return probs

    def predict_batch(
        self,
        context: torch.Tensor,
        n_positions: int,
    ) -> List[torch.Tensor]:
        """
        批量预测多个位置的分布（模拟并行前向传播）

        Args:
            context: 基础上下文 [seq_len]
            n_positions: 需要预测的位置数

        Returns:
            每个位置的概率分布列表
        """
        probs_list = []
        ctx = context.clone()
        for i in range(n_positions):
            probs = self.predict(ctx)
            probs_list.append(probs)
            # 使用 argmax 作为下一个 token（简化）
            next_token = probs.argmax()
            ctx = torch.cat([ctx, next_token.unsqueeze(0)])
        return probs_list

    def sample(self, probs: torch.Tensor) -> int:
        """从分布中采样一个 token"""
        return torch.multinomial(probs, 1).item()


def speculative_decode_step(
    target_model: SimpleLanguageModel,
    draft_model: SimpleLanguageModel,
    context: torch.Tensor,
    gamma: int = 5,
) -> Tuple[List[int], dict]:
    """
    执行一步投机解码

    1. 草稿模型自回归生成 gamma 个候选 token
    2. 目标模型并行验证所有候选
    3. 按照接受/拒绝规则确定最终接受的 token

    Args:
        target_model: 目标模型（大模型）
        draft_model: 草稿模型（小模型）
        context: 当前上下文 [seq_len]
        gamma: 猜测步数

    Returns:
        (accepted_tokens, stats): 接受的 token 列表和统计信息
    """
    # Step 1: 草稿模型自回归生成 gamma 个 token
    draft_tokens = []
    draft_probs = []
    ctx = context.clone()

    for _ in range(gamma):
        q = draft_model.predict(ctx)
        draft_probs.append(q)
        # 从草稿分布采样
        token = torch.multinomial(q, 1).item()
        draft_tokens.append(token)
        ctx = torch.cat([ctx, torch.tensor([token])])

    # Step 2: 目标模型并行验证
    # 一次前向传播计算所有 gamma+1 个位置的分布
    target_probs = []
    ctx = context.clone()
    for i in range(gamma + 1):
        p = target_model.predict(ctx)
        target_probs.append(p)
        if i < gamma:
            ctx = torch.cat([ctx, torch.tensor([draft_tokens[i]])])

    # Step 3: 接受/拒绝
    accepted_tokens = []
    n_accepted = 0

    for i in range(gamma):
        token = draft_tokens[i]
        p_token = target_probs[i][token].item()  # 目标模型概率
        q_token = draft_probs[i][token].item()    # 草稿模型概率

        # 接受概率: min(1, p/q)
        accept_prob = min(1.0, p_token / max(q_token, 1e-10))

        # 均匀采样决定是否接受
        r = torch.rand(1).item()
        if r < accept_prob:
            # 接受
            accepted_tokens.append(token)
            n_accepted += 1
        else:
            # 拒绝：从修正分布中采样
            # adjusted = normalize(max(0, p - q))
            p = target_probs[i]
            q = draft_probs[i]
            adjusted = torch.clamp(p - q, min=0)
            adjusted_sum = adjusted.sum()

            if adjusted_sum > 0:
                adjusted = adjusted / adjusted_sum
                new_token = torch.multinomial(adjusted, 1).item()
            else:
                # 回退到目标分布采样
                new_token = torch.multinomial(p, 1).item()

            accepted_tokens.append(new_token)
            break  # 一旦拒绝，后续 token 全部丢弃

    # 如果所有 gamma 个都被接受，额外从最后一个位置采样一个
    if n_accepted == gamma:
        extra_token = torch.multinomial(target_probs[gamma], 1).item()
        accepted_tokens.append(extra_token)

    stats = {
        "gamma": gamma,
        "n_accepted": n_accepted,
        "acceptance_rate": n_accepted / gamma,
        "tokens_generated": len(accepted_tokens),
    }

    return accepted_tokens, stats


def standard_decode(
    model: SimpleLanguageModel,
    context: torch.Tensor,
    n_tokens: int,
) -> Tuple[List[int], float]:
    """
    标准自回归解码（基线对比）

    Args:
        model: 语言模型
        context: 初始上下文
        n_tokens: 生成的 token 数

    Returns:
        (tokens, time): 生成的 token 列表和耗时
    """
    tokens = []
    ctx = context.clone()

    start = time.perf_counter()
    for _ in range(n_tokens):
        probs = model.predict(ctx)
        token = torch.multinomial(probs, 1).item()
        tokens.append(token)
        ctx = torch.cat([ctx, torch.tensor([token])])
    elapsed = time.perf_counter() - start

    return tokens, elapsed


def speculative_decode_full(
    target_model: SimpleLanguageModel,
    draft_model: SimpleLanguageModel,
    context: torch.Tensor,
    n_tokens: int,
    gamma: int = 5,
) -> Tuple[List[int], float, dict]:
    """
    完整的投机解码流程

    Args:
        target_model: 目标模型
        draft_model: 草稿模型
        context: 初始上下文
        n_tokens: 目标生成 token 数
        gamma: 每步猜测数

    Returns:
        (tokens, time, stats): 生成 token、耗时、统计信息
    """
    tokens = []
    ctx = context.clone()
    total_accepted = 0
    total_rounds = 0

    start = time.perf_counter()
    while len(tokens) < n_tokens:
        new_tokens, step_stats = speculative_decode_step(
            target_model, draft_model, ctx, gamma
        )

        # 只取需要的数量
        remaining = n_tokens - len(tokens)
        new_tokens = new_tokens[:remaining]

        tokens.extend(new_tokens)
        ctx = torch.cat([ctx, torch.tensor(new_tokens)])

        total_accepted += step_stats["n_accepted"]
        total_rounds += 1

    elapsed = time.perf_counter() - start

    stats = {
        "total_rounds": total_rounds,
        "total_accepted": total_accepted,
        "total_tokens": len(tokens),
        "avg_acceptance_rate": total_accepted / max(total_rounds * gamma, 1),
        "avg_tokens_per_round": len(tokens) / max(total_rounds, 1),
    }

    return tokens, elapsed, stats


def verify_distribution(
    target_model: SimpleLanguageModel,
    draft_model: SimpleLanguageModel,
    vocab_size: int,
    context: torch.Tensor,
    n_samples: int = 10000,
    gamma: int = 5,
) -> dict:
    """
    验证投机解码的分布正确性

    通过大量采样，比较投机解码产生的 token 分布
    与直接从目标模型采样的分布是否一致。

    Args:
        target_model: 目标模型
        draft_model: 草稿模型
        vocab_size: 词汇表大小
        context: 上下文
        n_samples: 采样次数
        gamma: 每步猜测数

    Returns:
        包含分布对比指标的字典
    """
    # 目标分布（直接采样）
    target_counts = torch.zeros(vocab_size)
    for _ in range(n_samples):
        probs = target_model.predict(context)
        token = torch.multinomial(probs, 1).item()
        target_counts[token] += 1
    target_dist = target_counts / n_samples

    # 投机解码分布
    spec_counts = torch.zeros(vocab_size)
    for _ in range(n_samples):
        tokens, _ = speculative_decode_step(
            target_model, draft_model, context, gamma
        )
        spec_counts[tokens[0]] += 1  # 只看第一个生成的 token
    spec_dist = spec_counts / n_samples

    # 计算 KL 散度
    # KL(target || spec)
    mask = (target_dist > 0) & (spec_dist > 0)
    if mask.any():
        kl_div = (target_dist[mask] * (target_dist[mask] / spec_dist[mask]).log()).sum().item()
    else:
        kl_div = float("inf")

    # 总变差距离 (Total Variation Distance)
    tv_distance = 0.5 * (target_dist - spec_dist).abs().sum().item()

    return {
        "kl_divergence": kl_div,
        "total_variation": tv_distance,
        "target_dist_top5": target_dist.topk(5),
        "spec_dist_top5": spec_dist.topk(5),
        "n_samples": n_samples,
    }


def analyze_speedup(
    gamma: int,
    alpha: float,
) -> dict:
    """
    理论加速比分析

    Args:
        gamma: 猜测步数
        alpha: 平均接受率

    Returns:
        理论指标字典
    """
    # 期望每轮接受的 token 数: (1 - alpha^{gamma+1}) / (1 - alpha)
    if abs(alpha - 1.0) < 1e-10:
        expected_tokens = gamma + 1
    else:
        expected_tokens = (1 - alpha ** (gamma + 1)) / (1 - alpha)

    # 理论加速比（不考虑草稿模型开销）
    ideal_speedup = expected_tokens

    return {
        "gamma": gamma,
        "alpha": alpha,
        "expected_tokens_per_round": expected_tokens,
        "ideal_speedup": ideal_speedup,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("投机解码实现演示")
    print("=" * 60)

    torch.manual_seed(42)

    vocab_size = 100
    context = torch.tensor([1, 5, 10, 20, 30])

    # 创建目标模型和草稿模型
    # 目标模型：较低温度（分布更尖锐，更"确定"）
    target = SimpleLanguageModel(vocab_size, temperature=0.8, seed=42)
    # 草稿模型：与目标模型有一定相似性
    draft = SimpleLanguageModel(vocab_size, temperature=1.0, seed=42)

    # 1. 单步投机解码演示
    print("\n--- 单步投机解码 ---")
    for gamma in [3, 5, 8]:
        tokens, stats = speculative_decode_step(target, draft, context, gamma)
        print(f"  gamma={gamma}: 接受 {stats['n_accepted']}/{gamma}, "
              f"实际生成 {stats['tokens_generated']} tokens, "
              f"接受率 {stats['acceptance_rate']:.2%}")

    # 2. 完整生成对比
    print("\n--- 完整生成对比 ---")
    n_tokens = 50

    # 标准解码
    std_tokens, std_time = standard_decode(target, context, n_tokens)
    print(f"  标准解码: {n_tokens} tokens, 耗时 {std_time*1000:.1f} ms")

    # 投机解码
    spec_tokens, spec_time, spec_stats = speculative_decode_full(
        target, draft, context, n_tokens, gamma=5
    )
    print(f"  投机解码: {spec_stats['total_tokens']} tokens, "
          f"耗时 {spec_time*1000:.1f} ms")
    print(f"    轮数: {spec_stats['total_rounds']}, "
          f"平均每轮 {spec_stats['avg_tokens_per_round']:.1f} tokens")
    print(f"    平均接受率: {spec_stats['avg_acceptance_rate']:.2%}")

    # 3. 分布验证
    print("\n--- 分布正确性验证 ---")
    result = verify_distribution(target, draft, vocab_size, context, n_samples=5000)
    print(f"  KL 散度: {result['kl_divergence']:.6f} (越接近 0 越好)")
    print(f"  总变差距离: {result['total_variation']:.6f} (越接近 0 越好)")
    # 注意：由于有限采样，KL 散度不会精确为 0，但应该很小

    # 4. 理论加速比分析
    print("\n--- 理论加速比分析 ---")
    print(f"{'gamma':<8} {'alpha':<8} {'期望 tokens/轮':<18} {'理论加速比'}")
    print("-" * 50)
    for gamma in [3, 5, 8]:
        for alpha in [0.5, 0.7, 0.9]:
            analysis = analyze_speedup(gamma, alpha)
            print(f"{gamma:<8} {alpha:<8.1f} "
                  f"{analysis['expected_tokens_per_round']:<18.2f} "
                  f"{analysis['ideal_speedup']:.2f}x")

    # 5. 接受率随 gamma 变化
    print("\n--- 不同 gamma 的实际表现 ---")
    for gamma in [1, 2, 3, 5, 8, 10]:
        n_trials = 200
        total_accepted = 0
        total_tokens = 0
        for _ in range(n_trials):
            tokens, stats = speculative_decode_step(target, draft, context, gamma)
            total_accepted += stats['n_accepted']
            total_tokens += stats['tokens_generated']
        avg_rate = total_accepted / (n_trials * gamma)
        avg_tokens = total_tokens / n_trials
        print(f"  gamma={gamma:<3d}: 接受率={avg_rate:.2%}, "
              f"平均每轮={avg_tokens:.1f} tokens")
