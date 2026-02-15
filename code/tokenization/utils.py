"""
分词工具函数集

提供预分词、文本清洗、Unicode规范化等基础功能。
这些工具函数被各分词器实现共享使用。

作者: LLM学习教程
模块: 模块1 - Tokenization

设计原则:
    分词流程通常分为三个阶段:
    1. 文本规范化 (Normalization): Unicode标准化、大小写转换等
    2. 预分词 (Pre-tokenization): 将文本切分为"词"级别的片段
    3. 子词分词 (Subword Tokenization): BPE/WordPiece/Unigram

    本模块提供阶段 1 和 2 的工具函数。
"""

import re
import unicodedata
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional


# ============================================================
# 1. 文本规范化 (Text Normalization)
# ============================================================

def normalize_unicode(text: str, form: str = "NFC") -> str:
    """
    Unicode 规范化

    Unicode 同一字符可能有多种编码方式:
    - NFC (Canonical Decomposition + Composition): 组合形式, 推荐用于分词
    - NFD (Canonical Decomposition): 分解形式
    - NFKC (Compatibility Decomposition + Composition): 兼容组合
    - NFKD (Compatibility Decomposition): 兼容分解

    示例:
        'é' (U+00E9) vs 'e' + '́' (U+0065 + U+0301)
        NFC 会将后者合并为前者

    Args:
        text: 输入文本
        form: 规范化形式, 可选 "NFC", "NFD", "NFKC", "NFKD"

    Returns:
        规范化后的文本
    """
    return unicodedata.normalize(form, text)


def remove_control_characters(text: str) -> str:
    """
    移除 Unicode 控制字符 (保留换行符和制表符)

    控制字符 (category Cc/Cf) 在文本中通常无意义,
    但可能影响分词器的行为, 甚至被用于 prompt injection 攻击。

    Args:
        text: 输入文本

    Returns:
        清理后的文本
    """
    cleaned = []
    for char in text:
        cat = unicodedata.category(char)
        # 保留换行符、制表符、普通空格
        if cat == "Cc" and char not in ("\n", "\r", "\t"):
            continue
        # 移除零宽字符等格式控制符
        if cat == "Cf":
            continue
        cleaned.append(char)
    return "".join(cleaned)


def lowercase_with_special(text: str) -> str:
    """
    智能小写转换: 保留全大写缩写词 (如 API, GPU, NLP)

    完全小写化会丢失缩写词信息, 智能方案保留全大写词:
    - "The GPU computes" → "the GPU computes"
    - "Hello World" → "hello world"
    - "Use the API" → "use the API"

    Args:
        text: 输入文本

    Returns:
        转换后的文本
    """
    words = text.split()
    result = []
    for word in words:
        # 全大写且长度>=2: 保留 (如 GPU, API, NLP)
        if word.isupper() and len(word) >= 2:
            result.append(word)
        else:
            result.append(word.lower())
    return " ".join(result)


# ============================================================
# 2. 预分词 (Pre-tokenization)
# ============================================================

# GPT-2 风格的预分词正则
GPT2_PAT = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+""",
    re.UNICODE,
)

# 简化版预分词正则 (按空白和标点切分)
SIMPLE_PAT = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def pre_tokenize_gpt2(text: str) -> List[str]:
    """
    GPT-2 风格的预分词

    使用 OpenAI 公开的正则表达式, 处理规则:
    - 英文缩写 ('s, 't, 're, ...) 作为整体
    - 连续字母序列作为一个词 (前缀空格保留)
    - 数字按连续序列切分
    - 标点单独切分
    - 保留空白信息

    Args:
        text: 输入文本

    Returns:
        预分词结果列表

    示例:
        >>> pre_tokenize_gpt2("Hello world! It's a test.")
        ['Hello', ' world', '!', " It", "'s", ' a', ' test', '.']
    """
    return re.findall(GPT2_PAT, text)


def pre_tokenize_simple(text: str) -> List[str]:
    """
    简单预分词: 按空白和标点切分

    适用于教学演示, 不保留空白信息。

    Args:
        text: 输入文本

    Returns:
        词列表

    示例:
        >>> pre_tokenize_simple("Hello, world!")
        ['Hello', ',', 'world', '!']
    """
    return re.findall(SIMPLE_PAT, text)


def pre_tokenize_chinese_aware(text: str) -> List[str]:
    """
    中文感知的预分词

    中文没有空格分隔, 需要特殊处理:
    - 英文部分按空格切分
    - 中文字符逐字切分 (每个汉字作为独立单元)
    - 中英文边界处切分
    - 标点单独切分

    这是 BERT 中文版使用的策略 (BasicTokenizer)。

    Args:
        text: 输入文本

    Returns:
        词/字列表

    示例:
        >>> pre_tokenize_chinese_aware("Hello世界！Test测试")
        ['Hello', '世', '界', '！', 'Test', '测', '试']
    """
    output = []
    current_word = []
    current_type = None  # 'latin', 'cjk', 'punct', 'space'

    for char in text:
        char_type = _get_char_type(char)

        if char_type == "space":
            if current_word:
                output.append("".join(current_word))
                current_word = []
                current_type = None
            continue

        if char_type == "cjk":
            # 中文字符: 先输出之前累积的词, 然后每个中文字符单独输出
            if current_word:
                output.append("".join(current_word))
                current_word = []
            output.append(char)
            current_type = None
        elif char_type == "punct":
            # 标点: 单独输出
            if current_word:
                output.append("".join(current_word))
                current_word = []
            output.append(char)
            current_type = None
        else:
            # 拉丁字母/数字: 累积
            if current_type and current_type != char_type:
                if current_word:
                    output.append("".join(current_word))
                    current_word = []
            current_word.append(char)
            current_type = char_type

    if current_word:
        output.append("".join(current_word))

    return output


def _get_char_type(char: str) -> str:
    """判断字符类型"""
    if char.isspace():
        return "space"
    cp = ord(char)
    # CJK 统一汉字范围
    if (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    ):
        return "cjk"
    cat = unicodedata.category(char)
    if cat.startswith("P") or cat.startswith("S"):
        return "punct"
    if cat.startswith("N"):
        return "number"
    return "latin"


# ============================================================
# 3. 词频统计与分析
# ============================================================

def build_word_freqs(
    corpus: List[str],
    pre_tokenize_fn=None,
    lowercase: bool = True,
) -> Dict[str, int]:
    """
    从语料构建词频字典

    Args:
        corpus: 文本列表
        pre_tokenize_fn: 预分词函数, 默认使用简单切分
        lowercase: 是否小写化

    Returns:
        词频字典 {word: count}
    """
    if pre_tokenize_fn is None:
        pre_tokenize_fn = pre_tokenize_simple

    word_freqs: Dict[str, int] = defaultdict(int)
    for text in corpus:
        if lowercase:
            text = text.lower()
        for word in pre_tokenize_fn(text):
            word_freqs[word] += 1
    return dict(word_freqs)


def compute_compression_ratio(
    text: str, tokens: List[str]
) -> float:
    """
    计算压缩率: 平均每个 token 对应的字符数

    压缩率 = len(text) / len(tokens)

    压缩率越高, 分词效率越好:
    - 字符级: ~1.0
    - 子词级: ~3.0-5.0 (英文), ~1.5-3.0 (中文)
    - 词级: ~5.0-7.0 (英文)

    Args:
        text: 原始文本
        tokens: 分词结果

    Returns:
        压缩率 (chars/token)
    """
    if not tokens:
        return 0.0
    return len(text) / len(tokens)


def compute_fertility(
    words: List[str], tokens: List[str]
) -> float:
    """
    计算生育率 (Fertility): 平均每个词被切分为多少个子词

    Fertility = len(tokens) / len(words)

    Fertility 越接近 1, 说明分词器的词汇覆盖率越高:
    - Fertility = 1.0: 每个词都在词汇表中 (完美覆盖)
    - Fertility > 2.0: 很多词需要拆分 (覆盖率低)

    Args:
        words: 原始词列表
        tokens: 子词 token 列表

    Returns:
        生育率
    """
    if not words:
        return 0.0
    return len(tokens) / len(words)


def analyze_vocab_coverage(
    vocab: Dict[str, int],
    corpus: List[str],
    pre_tokenize_fn=None,
) -> Dict[str, float]:
    """
    分析词汇表对语料的覆盖率

    Args:
        vocab: 词汇表 {token: id}
        corpus: 测试语料
        pre_tokenize_fn: 预分词函数

    Returns:
        覆盖率统计 {
            "total_words": 总词数,
            "covered_words": 被完整覆盖的词数,
            "coverage_rate": 覆盖率,
            "oov_rate": OOV率,
        }
    """
    if pre_tokenize_fn is None:
        pre_tokenize_fn = pre_tokenize_simple

    total = 0
    covered = 0
    for text in corpus:
        words = pre_tokenize_fn(text.lower())
        for word in words:
            total += 1
            if word in vocab:
                covered += 1

    coverage = covered / total if total > 0 else 0.0
    return {
        "total_words": total,
        "covered_words": covered,
        "coverage_rate": coverage,
        "oov_rate": 1.0 - coverage,
    }


# ============================================================
# 4. 字节级操作
# ============================================================

def text_to_bytes(text: str) -> List[int]:
    """
    将文本转换为 UTF-8 字节序列

    Byte-level BPE 的基础操作:
    - ASCII 字符: 1 字节 (0x00-0x7F)
    - 中文字符: 通常 3 字节
    - Emoji: 通常 4 字节

    Args:
        text: 输入文本

    Returns:
        UTF-8 字节值列表

    示例:
        >>> text_to_bytes("Hi你")
        [72, 105, 228, 189, 160]
    """
    return list(text.encode("utf-8"))


def bytes_to_text(byte_values: List[int]) -> str:
    """
    将 UTF-8 字节序列还原为文本

    Args:
        byte_values: 字节值列表

    Returns:
        还原的文本 (解码失败的字节用替换字符 U+FFFD)
    """
    return bytes(byte_values).decode("utf-8", errors="replace")


def byte_to_visible_char(byte_val: int) -> str:
    """
    将字节映射为可见字符 (GPT-2 风格)

    GPT-2 的 byte-level BPE 将 256 个字节值映射为可打印 Unicode 字符,
    避免词汇表中出现不可见字符。

    映射规则:
    - 可打印 ASCII (33-126) 和部分 Latin-1 (161-172, 174-255): 直接映射
    - 其他 (0-32, 127-160, 173): 映射到 U+0100 起始的区间

    Args:
        byte_val: 0-255 的字节值

    Returns:
        对应的可见字符
    """
    # GPT-2 的字节到 Unicode 映射
    # 可打印范围直接映射
    printable_ranges = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    if byte_val in printable_ranges:
        return chr(byte_val)
    else:
        # 不可打印字节映射到 U+0100 起始
        offset = printable_ranges.index(byte_val) if byte_val in printable_ranges else 0
        # 按 GPT-2 规则: 从 256 开始分配
        special_bytes = [b for b in range(256) if b not in printable_ranges]
        if byte_val in special_bytes:
            return chr(256 + special_bytes.index(byte_val))
        return chr(byte_val)


# ============================================================
# 5. 可视化辅助
# ============================================================

def visualize_tokenization(
    text: str, tokens: List[str], token_ids: Optional[List[int]] = None
) -> str:
    """
    可视化分词结果: 用 | 分隔各 token, 便于直观查看

    Args:
        text: 原始文本
        tokens: 分词结果
        token_ids: 可选的 token ID 列表

    Returns:
        格式化的可视化字符串

    示例:
        >>> visualize_tokenization("hello world", ["hel", "lo", " world"])
        '原文: hello world
         分词: |hel|lo| world|
         数量: 3 tokens'
    """
    lines = [
        f"原文:   {text}",
        f"分词:   |{'|'.join(tokens)}|",
        f"数量:   {len(tokens)} tokens",
        f"压缩率: {compute_compression_ratio(text, tokens):.2f} chars/token",
    ]
    if token_ids is not None:
        lines.insert(2, f"IDs:    {token_ids}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("分词工具函数演示")
    print("=" * 60)

    # 1. Unicode 规范化演示
    print("\n--- Unicode 规范化 ---")
    s1 = "caf\u00e9"  # NFC 形式
    s2 = "cafe\u0301"  # NFD 形式
    print(f"s1 = {repr(s1)}, len={len(s1)}")
    print(f"s2 = {repr(s2)}, len={len(s2)}")
    print(f"s1 == s2: {s1 == s2}")
    print(f"NFC(s1) == NFC(s2): {normalize_unicode(s1) == normalize_unicode(s2)}")

    # 2. 预分词对比
    print("\n--- 预分词对比 ---")
    test_en = "Hello world! It's a beautiful day."
    test_zh = "大语言模型(LLM)改变了AI领域。"

    print(f"英文测试: {test_en}")
    print(f"  GPT-2 风格: {pre_tokenize_gpt2(test_en)}")
    print(f"  简单切分:   {pre_tokenize_simple(test_en)}")

    print(f"\n中文测试: {test_zh}")
    print(f"  中文感知: {pre_tokenize_chinese_aware(test_zh)}")
    print(f"  简单切分: {pre_tokenize_simple(test_zh)}")

    # 3. 词频统计
    print("\n--- 词频统计 ---")
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "a cat and a dog",
    ]
    freqs = build_word_freqs(corpus)
    top5 = sorted(freqs.items(), key=lambda x: -x[1])[:5]
    print(f"Top 5 词频: {top5}")

    # 4. 字节操作演示
    print("\n--- 字节级操作 ---")
    for text in ["Hi", "你好", "🎉"]:
        byte_vals = text_to_bytes(text)
        restored = bytes_to_text(byte_vals)
        print(f"  '{text}' → {byte_vals} → '{restored}'  ({len(byte_vals)} bytes)")

    # 5. 可视化演示
    print("\n--- 分词可视化 ---")
    tokens = ["hel", "lo", " wo", "rld", "!"]
    ids = [42, 17, 89, 63, 5]
    print(visualize_tokenization("hello world!", tokens, ids))
