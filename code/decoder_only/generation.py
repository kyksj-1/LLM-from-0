"""
文本生成策略实现

本模块实现了多种自回归文本生成策略:
- Greedy Decoding: 每步选择概率最大的 token
- Temperature Sampling: 调节分布尖锐程度
- Top-K Sampling: 限制候选集大小
- Top-P (Nucleus) Sampling: 动态候选集
- Beam Search: 多路径搜索最优序列
- 组合策略: Temperature + Top-K + Top-P

数学基础:
- Temperature: P_tau(w) = exp(z_w / tau) / sum exp(z_w' / tau)
- Top-K: 只保留概率最高的 K 个 token
- Top-P: 保留概率累积和达到 p 的最小集合
- Beam Search: 维护 beam_width 个候选序列

参考:
- Holtzman et al. (2020). The Curious Case of Neural Text Degeneration. (Nucleus Sampling)
- Fan et al. (2018). Hierarchical Neural Story Generation. (Top-K Sampling)
"""

import torch
import torch.nn.functional as F
from typing import Optional, List

from model import DecoderOnlyModel
from config import ModelConfig


@torch.no_grad()
def generate(
    model: DecoderOnlyModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    自回归文本生成 (支持多种采样策略)

    Args:
        model: Decoder-Only 语言模型
        input_ids: [batch, seq_len] 输入 token IDs (prompt)
        max_new_tokens: 最大生成 token 数
        temperature: 温度参数 (>1 更随机, <1 更确定, =0 为 Greedy)
        top_k: Top-K 采样参数 (None 表示不使用)
        top_p: Top-P (Nucleus) 采样参数 (None 表示不使用)
        repetition_penalty: 重复惩罚系数 (>1 惩罚重复, =1 无惩罚)
        eos_token_id: 终止 token ID (遇到时停止生成)

    Returns:
        [batch, seq_len + generated_len] 包含 prompt 和生成内容的完整序列
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]

    for step in range(max_new_tokens):
        # 截断到最大上下文长度
        context = input_ids[:, -model.config.max_seq_len:]

        # 前向传播获取 logits
        logits = model(context)
        # 只取最后一个位置的 logits
        next_logits = logits[:, -1, :]  # [batch, vocab_size]

        # --- 重复惩罚 ---
        if repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(
                next_logits, input_ids, repetition_penalty
            )

        # --- Temperature 缩放 ---
        if temperature == 0.0:
            # Greedy decoding
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            # 检查 EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            continue

        next_logits = next_logits / temperature

        # --- Top-K 过滤 ---
        if top_k is not None:
            next_logits = top_k_filtering(next_logits, top_k)

        # --- Top-P 过滤 ---
        if top_p is not None:
            next_logits = top_p_filtering(next_logits, top_p)

        # --- 采样 ---
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 拼接到序列
        input_ids = torch.cat([input_ids, next_token], dim=1)

        # 检查 EOS
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break

    return input_ids


def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Top-K 过滤: 只保留概率最高的 K 个 token

    将不在 Top-K 中的 token 的 logits 设为 -inf。

    Args:
        logits: [batch, vocab_size]
        k: 保留的 token 数

    Returns:
        过滤后的 logits
    """
    k = min(k, logits.size(-1))
    # 找到第 K 大的值作为阈值
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[:, -1:]  # [batch, 1]
    # 将低于阈值的 logits 设为 -inf
    logits = logits.masked_fill(logits < threshold, float("-inf"))
    return logits


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Top-P (Nucleus) 过滤: 保留概率累积和达到 p 的最小集合

    按概率从大到小排序, 累积概率超过 p 的 token 被过滤掉。

    Args:
        logits: [batch, vocab_size]
        p: 累积概率阈值 (0.0 ~ 1.0)

    Returns:
        过滤后的 logits
    """
    # 按 logits 降序排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # 计算累积概率
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 标记需要移除的 token (累积概率超过 p 的部分)
    sorted_mask = cumulative_probs > p
    # 保留第一个超过 p 的 token (避免全部被过滤)
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # 将标记映射回原始顺序
    mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
    logits = logits.masked_fill(mask, float("-inf"))
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """
    应用重复惩罚: 降低已出现 token 的概率

    对于已经出现在 input_ids 中的 token:
    - 如果 logit > 0: logit = logit / penalty
    - 如果 logit < 0: logit = logit * penalty

    Args:
        logits: [batch, vocab_size]
        input_ids: [batch, seq_len] 已有的 token 序列
        penalty: 惩罚系数 (>1 降低重复概率)

    Returns:
        惩罚后的 logits
    """
    for batch_idx in range(logits.shape[0]):
        # 获取已出现的唯一 token
        appeared_tokens = input_ids[batch_idx].unique()
        for token_id in appeared_tokens:
            if logits[batch_idx, token_id] > 0:
                logits[batch_idx, token_id] /= penalty
            else:
                logits[batch_idx, token_id] *= penalty
    return logits


@torch.no_grad()
def beam_search(
    model: DecoderOnlyModel,
    input_ids: torch.Tensor,
    max_new_tokens: int = 50,
    beam_width: int = 5,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Beam Search 生成

    维护 beam_width 个候选序列, 每步扩展每个候选并保留得分最高的 beam_width 个。

    Args:
        model: 语言模型
        input_ids: [1, seq_len] 输入 (仅支持 batch_size=1)
        max_new_tokens: 最大生成 token 数
        beam_width: Beam 宽度
        length_penalty: 长度惩罚 (>1 偏好长序列, <1 偏好短序列)
        eos_token_id: 终止 token ID

    Returns:
        [1, seq_len + generated_len] 最优序列
    """
    model.eval()
    assert input_ids.shape[0] == 1, "Beam Search 目前仅支持 batch_size=1"

    device = input_ids.device
    # 初始化: 将单个序列复制 beam_width 份
    beams = input_ids.repeat(beam_width, 1)  # [beam_width, seq_len]
    beam_scores = torch.zeros(beam_width, device=device)  # 每个 beam 的累积 log 概率
    beam_scores[1:] = float("-inf")  # 初始时只有第一个 beam 有效

    finished_beams: List[tuple] = []  # (score, sequence)

    for step in range(max_new_tokens):
        # 截断上下文
        context = beams[:, -model.config.max_seq_len:]

        # 前向传播
        logits = model(context)[:, -1, :]  # [beam_width, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)

        vocab_size = log_probs.shape[-1]

        # 计算所有可能的下一步得分
        # [beam_width, vocab_size]
        next_scores = beam_scores.unsqueeze(-1) + log_probs

        # 展平并选出 top beam_width 个
        next_scores = next_scores.view(-1)  # [beam_width * vocab_size]
        top_scores, top_indices = torch.topk(next_scores, beam_width)

        # 计算来源 beam 和 token
        beam_indices = top_indices // vocab_size
        token_indices = top_indices % vocab_size

        # 更新 beams
        beams = torch.cat([
            beams[beam_indices],
            token_indices.unsqueeze(-1),
        ], dim=1)
        beam_scores = top_scores

        # 检查 EOS
        if eos_token_id is not None:
            for i in range(beam_width):
                if token_indices[i].item() == eos_token_id:
                    # 长度惩罚: score / length^alpha
                    length = beams[i].shape[0]
                    penalized_score = beam_scores[i].item() / (length ** length_penalty)
                    finished_beams.append((penalized_score, beams[i].clone()))

    # 如果没有遇到 EOS, 将当前 beams 加入 finished
    if not finished_beams:
        for i in range(beam_width):
            length = beams[i].shape[0]
            penalized_score = beam_scores[i].item() / (length ** length_penalty)
            finished_beams.append((penalized_score, beams[i].clone()))

    # 返回得分最高的序列
    finished_beams.sort(key=lambda x: x[0], reverse=True)
    return finished_beams[0][1].unsqueeze(0)


# ============================================================
# 辅助函数: 生成质量评估
# ============================================================

def repetition_rate(token_ids: List[int], n: int = 3) -> float:
    """
    计算 n-gram 重复率

    重复率 = 1 - (唯一 n-gram 数 / 总 n-gram 数)

    Args:
        token_ids: token ID 列表
        n: n-gram 的 n

    Returns:
        重复率 (0.0 ~ 1.0, 越低越好)
    """
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def distinct_n(token_ids: List[int], n: int = 2) -> float:
    """
    计算 Distinct-N 指标 (多样性)

    Distinct-N = 唯一 n-gram 数 / 总 n-gram 数

    Args:
        token_ids: token ID 列表
        n: n-gram 的 n

    Returns:
        多样性 (0.0 ~ 1.0, 越高越好)
    """
    if len(token_ids) < n:
        return 0.0
    ngrams = [tuple(token_ids[i:i + n]) for i in range(len(token_ids) - n + 1)]
    if len(ngrams) == 0:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


if __name__ == "__main__":
    from config import mini_config

    # 创建一个小模型进行测试
    config = mini_config()
    model = DecoderOnlyModel(config)

    # 模拟 prompt
    prompt = torch.randint(0, config.vocab_size, (1, 10))

    print("=== Greedy Decoding ===")
    greedy_output = generate(model, prompt.clone(), max_new_tokens=20, temperature=0.0)
    print(f"输出长度: {greedy_output.shape[1]} (prompt=10 + generated={greedy_output.shape[1]-10})")
    print(f"Greedy tokens: {greedy_output[0, 10:].tolist()}")

    print("\n=== Temperature Sampling (T=0.8) + Top-P (0.9) ===")
    sampled_output = generate(
        model, prompt.clone(), max_new_tokens=20,
        temperature=0.8, top_p=0.9,
    )
    print(f"Sampled tokens: {sampled_output[0, 10:].tolist()}")

    print("\n=== Top-K Sampling (K=50) ===")
    topk_output = generate(
        model, prompt.clone(), max_new_tokens=20,
        temperature=1.0, top_k=50,
    )
    print(f"Top-K tokens: {topk_output[0, 10:].tolist()}")

    print("\n=== Beam Search (width=3) ===")
    beam_output = beam_search(
        model, prompt.clone(), max_new_tokens=20, beam_width=3,
    )
    print(f"Beam tokens: {beam_output[0, 10:].tolist()}")

    # 生成质量对比
    print("\n=== 生成质量对比 ===")
    strategies = {
        "Greedy": generate(model, prompt.clone(), max_new_tokens=50, temperature=0.0),
        "T=0.5": generate(model, prompt.clone(), max_new_tokens=50, temperature=0.5),
        "T=1.0": generate(model, prompt.clone(), max_new_tokens=50, temperature=1.0),
        "T=0.8+P=0.9": generate(model, prompt.clone(), max_new_tokens=50, temperature=0.8, top_p=0.9),
    }

    for name, output in strategies.items():
        tokens = output[0, 10:].tolist()
        rep_3 = repetition_rate(tokens, 3)
        dist_2 = distinct_n(tokens, 2)
        print(f"{name:15s}: 重复率(3-gram)={rep_3:.3f}, Distinct-2={dist_2:.3f}")
