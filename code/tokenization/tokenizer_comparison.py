"""
三种子词分词算法对比分析工具

对比 BPE、WordPiece、Unigram 三种分词算法的:
- 分词结果差异
- 压缩效率
- 训练时间
- 词汇表特征

作者: LLM学习教程
模块: 模块1 - Tokenization

核心差异总结:
    ┌──────────────┬─────────────────┬──────────────────┬──────────────────┐
    │              │     BPE         │    WordPiece      │    Unigram       │
    ├──────────────┼─────────────────┼──────────────────┼──────────────────┤
    │ 方向         │ 自底向上 (合并)  │ 自底向上 (合并)   │ 自顶向下 (剪枝)  │
    │ 合并策略     │ 最高频率         │ 最大似然增益      │ 最小损失贡献     │
    │ 解码算法     │ 贪心 (确定性)    │ 贪心 (确定性)     │ Viterbi (最优)   │
    │ 概率模型     │ 无              │ 无               │ 有 (Unigram LM)  │
    │ 多分词支持   │ 否              │ 否               │ 是 (子词正则化)  │
    │ 代表应用     │ GPT系列          │ BERT             │ T5, Llama       │
    └──────────────┴─────────────────┴──────────────────┴──────────────────┘
"""

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

# 导入本地实现的三种分词器
from bpe_tokenizer import BPETokenizer
from wordpiece_tokenizer import WordPieceTokenizer
from unigram_tokenizer import UnigramTokenizer


def create_sample_corpus() -> List[str]:
    """
    创建用于对比实验的示例语料

    包含多种文本模式:
    - 重复短语 (测试 BPE 频率统计)
    - 多种词形 (测试子词拆分能力)
    - 不同长度的句子

    Returns:
        语料文本列表
    """
    return [
        # 基础句型 (高频词)
        "the cat sat on the mat",
        "the cat ate the mouse and the cheese",
        "the dog sat on the log near the house",
        "the dog ate the bone in the garden",
        # 形容词变化
        "a small cat is a nice animal",
        "a big dog is a loyal animal",
        "the small cat sat on the big mat",
        "the loyal dog ate a big bone",
        # 复数和动词变化
        "cats and dogs are popular pets",
        "the cats ate the mice in the house",
        "dogs are loyal and cats are independent",
        # 更长的句子
        "the small cat and the big dog sat on the mat together",
        "the mouse ran away from the cat and hid in the house",
        "a nice animal is a loyal pet that sits on the mat",
        # 新词 (测试未见词处理)
        "the animal kingdom includes cats dogs mice and birds",
        "small animals like mice can run very fast",
        "big animals like dogs need more food and space",
        "the garden has many nice flowers and tall trees",
        "the house near the garden has a small garden too",
        "cats chase mice and dogs chase cats sometimes",
    ]


def train_all_tokenizers(
    corpus: List[str],
    vocab_size: int = 80,
    verbose: bool = False,
) -> Tuple["BPETokenizer", "WordPieceTokenizer", "UnigramTokenizer", Dict]:
    """
    训练三种分词器并记录训练时间

    Args:
        corpus: 训练语料
        vocab_size: 目标词汇表大小
        verbose: 是否打印训练过程

    Returns:
        (bpe_tokenizer, wp_tokenizer, uni_tokenizer, timing_info)
    """
    timings = {}

    # 1. 训练 BPE
    if verbose:
        print("训练 BPE 分词器...")
    bpe = BPETokenizer()
    t0 = time.time()
    bpe.train(corpus, vocab_size=vocab_size, verbose=False)
    timings["BPE"] = time.time() - t0

    # 2. 训练 WordPiece
    if verbose:
        print("训练 WordPiece 分词器...")
    wp = WordPieceTokenizer(vocab_size=vocab_size)
    t0 = time.time()
    wp.train(corpus, verbose=False)
    timings["WordPiece"] = time.time() - t0

    # 3. 训练 Unigram
    if verbose:
        print("训练 Unigram 分词器...")
    uni = UnigramTokenizer(vocab_size=vocab_size, initial_vocab_size=vocab_size * 4)
    t0 = time.time()
    uni.train(corpus, verbose=False)
    timings["Unigram"] = time.time() - t0

    return bpe, wp, uni, timings


def compare_tokenization(
    text: str,
    bpe: "BPETokenizer",
    wp: "WordPieceTokenizer",
    uni: "UnigramTokenizer",
) -> Dict:
    """
    对比三种分词器对同一文本的分词结果

    Args:
        text: 测试文本
        bpe: BPE 分词器
        wp: WordPiece 分词器
        uni: Unigram 分词器

    Returns:
        对比结果字典
    """
    # BPE 分词
    bpe_tokens = bpe.tokenize(text)
    bpe_ids = bpe.encode(text)

    # WordPiece 分词
    wp_ids = wp.encode(text)
    wp_tokens = [wp.id_to_token.get(i, "[?]") for i in wp_ids]

    # Unigram 分词
    uni_tokens = uni.tokenize(text)
    uni_ids = uni.encode(text)

    return {
        "text": text,
        "BPE": {
            "tokens": bpe_tokens,
            "ids": bpe_ids,
            "num_tokens": len(bpe_tokens),
            "compression": len(text) / max(len(bpe_tokens), 1),
        },
        "WordPiece": {
            "tokens": wp_tokens,
            "ids": wp_ids,
            "num_tokens": len(wp_tokens),
            "compression": len(text) / max(len(wp_tokens), 1),
        },
        "Unigram": {
            "tokens": uni_tokens,
            "ids": uni_ids,
            "num_tokens": len(uni_tokens),
            "compression": len(text) / max(len(uni_tokens), 1),
        },
    }


def analyze_vocab(
    bpe: "BPETokenizer",
    wp: "WordPieceTokenizer",
    uni: "UnigramTokenizer",
) -> Dict:
    """
    分析三种分词器的词汇表特征

    对比维度:
    - 词汇表大小
    - 平均 token 长度
    - 单字符 token 比例
    - 多字符 token (>3) 比例
    - 词汇表重叠度

    Args:
        bpe, wp, uni: 三种分词器

    Returns:
        词汇表分析结果
    """
    def vocab_stats(vocab_dict: Dict[str, int], name: str) -> Dict:
        tokens = [t for t in vocab_dict.keys() if not t.startswith("[") and not t.startswith("<")]
        lengths = [len(t.replace("##", "")) for t in tokens]
        single_char = sum(1 for l in lengths if l == 1)
        long_tokens = sum(1 for l in lengths if l > 3)
        avg_len = sum(lengths) / max(len(lengths), 1)

        return {
            "name": name,
            "vocab_size": len(vocab_dict),
            "avg_token_length": round(avg_len, 2),
            "single_char_pct": round(single_char / max(len(tokens), 1) * 100, 1),
            "long_token_pct": round(long_tokens / max(len(tokens), 1) * 100, 1),
        }

    bpe_vocab = {t: i for i, t in bpe.id_to_token.items()} if hasattr(bpe, "id_to_token") else bpe.token_to_id
    wp_vocab = wp.vocab
    uni_vocab = uni.vocab

    stats = {
        "BPE": vocab_stats(bpe_vocab, "BPE"),
        "WordPiece": vocab_stats(wp_vocab, "WordPiece"),
        "Unigram": vocab_stats(uni_vocab, "Unigram"),
    }

    # 计算词汇表重叠
    bpe_set = set(bpe_vocab.keys())
    wp_set = set(wp_vocab.keys())
    uni_set = set(uni_vocab.keys())

    stats["overlap"] = {
        "BPE_WP": len(bpe_set & wp_set),
        "BPE_Uni": len(bpe_set & uni_set),
        "WP_Uni": len(wp_set & uni_set),
        "all_three": len(bpe_set & wp_set & uni_set),
    }

    return stats


def print_comparison_report(
    corpus: List[str],
    test_texts: List[str],
    vocab_size: int = 80,
):
    """
    打印完整的三种分词器对比报告

    Args:
        corpus: 训练语料
        test_texts: 测试文本列表
        vocab_size: 目标词汇表大小
    """
    print("=" * 70)
    print("三种子词分词算法对比分析报告")
    print("=" * 70)

    # 训练
    print(f"\n[1] 训练 (目标词汇量: {vocab_size})")
    print("-" * 50)
    bpe, wp, uni, timings = train_all_tokenizers(
        corpus, vocab_size=vocab_size, verbose=True
    )

    print("\n训练时间:")
    for name, t in timings.items():
        print(f"  {name:12s}: {t:.4f}s")

    # 词汇表分析
    print(f"\n[2] 词汇表分析")
    print("-" * 50)
    vocab_stats = analyze_vocab(bpe, wp, uni)
    print(f"{'指标':<18s} {'BPE':>10s} {'WordPiece':>10s} {'Unigram':>10s}")
    print("-" * 50)

    for key in ["vocab_size", "avg_token_length", "single_char_pct", "long_token_pct"]:
        label = {
            "vocab_size": "词汇表大小",
            "avg_token_length": "平均 token 长度",
            "single_char_pct": "单字符比例(%)",
            "long_token_pct": "长 token 比例(%)",
        }[key]
        vals = [vocab_stats[name][key] for name in ["BPE", "WordPiece", "Unigram"]]
        print(f"{label:<18s} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10}")

    overlap = vocab_stats["overlap"]
    print(f"\n词汇表重叠: BPE∩WP={overlap['BPE_WP']}, "
          f"BPE∩Uni={overlap['BPE_Uni']}, WP∩Uni={overlap['WP_Uni']}, "
          f"三者共有={overlap['all_three']}")

    # 分词对比
    print(f"\n[3] 分词结果对比")
    print("-" * 50)

    total_tokens = {"BPE": 0, "WordPiece": 0, "Unigram": 0}

    for text in test_texts:
        result = compare_tokenization(text, bpe, wp, uni)
        print(f"\n原文: \"{text}\"")
        for name in ["BPE", "WordPiece", "Unigram"]:
            r = result[name]
            tokens_str = " | ".join(r["tokens"][:15])
            if len(r["tokens"]) > 15:
                tokens_str += " | ..."
            print(f"  {name:12s}: [{tokens_str}]  ({r['num_tokens']} tokens, "
                  f"压缩率 {r['compression']:.2f})")
            total_tokens[name] += r["num_tokens"]

    # 总结
    print(f"\n[4] 总体效率对比")
    print("-" * 50)
    total_chars = sum(len(t) for t in test_texts)
    print(f"测试文本总字符数: {total_chars}")
    for name in ["BPE", "WordPiece", "Unigram"]:
        avg_comp = total_chars / max(total_tokens[name], 1)
        print(f"  {name:12s}: 总 token 数 {total_tokens[name]:4d}, "
              f"平均压缩率 {avg_comp:.2f} chars/token")

    # 算法特性总结
    print(f"\n[5] 算法特性总结")
    print("-" * 50)
    summary = [
        ("合并方向", "自底向上", "自底向上", "自顶向下"),
        ("合并/剪枝策略", "最高频率对", "最大似然增益对", "最小损失贡献token"),
        ("解码/推理", "规则回放(贪心)", "贪心最长匹配", "Viterbi(全局最优)"),
        ("概率模型", "无", "无", "Unigram LM"),
        ("子词正则化", "不支持", "不支持", "支持(多分词采样)"),
        ("代表应用", "GPT, Llama", "BERT", "T5, XLNet"),
    ]
    print(f"{'特性':<16s} {'BPE':>14s} {'WordPiece':>16s} {'Unigram':>18s}")
    print("-" * 70)
    for row in summary:
        print(f"{row[0]:<16s} {row[1]:>14s} {row[2]:>16s} {row[3]:>18s}")


if __name__ == "__main__":
    # 创建语料
    corpus = create_sample_corpus()
    print(f"语料大小: {len(corpus)} 条, 总字符: {sum(len(t) for t in corpus)}")

    # 测试文本
    test_texts = [
        "the cat sat on the mat",
        "a small dog ate the bone",
        "cats and dogs are animals",
        "the mouse ran away from the garden",
        "nice loyal pets are popular",
    ]

    # 运行完整对比
    print_comparison_report(corpus, test_texts, vocab_size=80)

    # 单独演示: 编码→解码的往返测试
    print("\n" + "=" * 70)
    print("编码→解码往返测试 (Round-trip)")
    print("=" * 70)

    bpe, wp, uni, _ = train_all_tokenizers(corpus, vocab_size=80)

    for text in ["the cat sat", "small animals"]:
        print(f"\n原文: \"{text}\"")

        # BPE 往返
        bpe_ids = bpe.encode(text)
        bpe_decoded = bpe.decode(bpe_ids)
        print(f"  BPE:       {text} → {bpe_ids} → \"{bpe_decoded}\"")

        # WordPiece 往返
        wp_ids = wp.encode(text)
        wp_decoded = wp.decode(wp_ids)
        print(f"  WordPiece: {text} → {wp_ids} → \"{wp_decoded}\"")

        # Unigram 往返
        uni_ids = uni.encode(text)
        uni_decoded = uni.decode(uni_ids)
        print(f"  Unigram:   {text} → {uni_ids} → \"{uni_decoded}\"")
