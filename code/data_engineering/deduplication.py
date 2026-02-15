"""
MinHash + LSH 近似去重实现

本模块实现了完整的 MinHash（最小哈希）和 LSH（局部敏感哈希）
近似去重算法，用于大规模文本数据集的近似重复文档检测。

核心算法:
- Shingling: 将文档转化为 n-gram 集合
- MinHash: 用多个哈希函数将集合压缩为固定长度签名
- LSH (分带策略): 将相似文档映射到同一桶中，减少比较次数
- Jaccard 相似度: 衡量两个集合的重叠程度

数学基础:
- MinHash 估计定理: Pr[MinHash_h(A) = MinHash_h(B)] = J(A, B)
- LSH 候选概率: P(candidate | s) = 1 - (1 - s^r)^b

参考:
- Broder (1997). On the Resemblance and Containment of Documents.
- Lee et al. (2022). Deduplicating Training Data Makes Language Models Better.
"""

import hashlib
import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple


# 梅森素数，用于通用哈希函数
MERSENNE_PRIME = (1 << 61) - 1
# 最大哈希值（32位无符号整数）
MAX_HASH = (1 << 32) - 1


def get_shingles(text: str, n: int = 5) -> Set[str]:
    """
    将文本转换为 n-gram（shingle）集合

    将文本按空格分词后，取连续 n 个词作为一个 shingle。
    shingle 集合用于后续的 Jaccard 相似度计算。

    Args:
        text: 输入文本
        n: n-gram 的 n 值，通常取 5

    Returns:
        n-gram 字符串集合
    """
    words = text.lower().split()
    if len(words) < n:
        # 文本太短，整体作为一个 shingle
        return {text.lower().strip()}
    return {' '.join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard_similarity(set_a: Set, set_b: Set) -> float:
    """
    计算两个集合的 Jaccard 相似度

    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: 集合 A
        set_b: 集合 B

    Returns:
        Jaccard 相似度，范围 [0, 1]
    """
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


class MinHasher:
    """
    MinHash 签名计算器

    使用多个独立的哈希函数将集合压缩为固定长度的签名向量。
    两个签名向量中对应位置相等的比例是 Jaccard 相似度的无偏估计。

    数学原理:
    - 对每个哈希函数 h_i，MinHash_h_i(A) = min_{a in A} h_i(a)
    - Pr[MinHash_h(A) = MinHash_h(B)] = J(A, B)
    - 估计量 J_hat = (1/k) * sum(I[sig_A[i] == sig_B[i]]) 是无偏的
    - 方差 Var(J_hat) = J(1-J)/k

    Args:
        num_hashes: 哈希函数数量（签名长度），越大估计越精确
        seed: 随机种子，确保可复现性
    """

    def __init__(self, num_hashes: int = 128, seed: int = 42):
        self.num_hashes = num_hashes
        self.seed = seed
        # 生成哈希函数参数: h_i(x) = (a_i * x + b_i) % p
        self._hash_params = self._generate_hash_params()

    def _generate_hash_params(self) -> List[Tuple[int, int]]:
        """生成通用哈希函数的参数 (a, b)"""
        rng = random.Random(self.seed)
        params = []
        for _ in range(self.num_hashes):
            a = rng.randint(1, MERSENNE_PRIME - 1)
            b = rng.randint(0, MERSENNE_PRIME - 1)
            params.append((a, b))
        return params

    def _hash_token(self, token: str) -> int:
        """将字符串哈希为 32 位无符号整数"""
        return int(hashlib.md5(token.encode('utf-8')).hexdigest()[:8], 16)

    def compute_signature(self, shingles: Set[str]) -> List[int]:
        """
        计算一个集合的 MinHash 签名

        对每个哈希函数，取集合中所有元素经哈希后的最小值。

        Args:
            shingles: 元素集合（通常是 n-gram 集合）

        Returns:
            长度为 num_hashes 的签名向量
        """
        if not shingles:
            return [MAX_HASH] * self.num_hashes

        signature = [MAX_HASH] * self.num_hashes

        for token in shingles:
            token_hash = self._hash_token(token)
            for i, (a, b) in enumerate(self._hash_params):
                # 通用哈希: h(x) = (a * x + b) mod p
                val = (a * token_hash + b) % MERSENNE_PRIME
                # 截断到 32 位
                val = val & MAX_HASH
                if val < signature[i]:
                    signature[i] = val

        return signature

    def estimate_jaccard(
        self, sig_a: List[int], sig_b: List[int]
    ) -> float:
        """
        通过 MinHash 签名估计 Jaccard 相似度

        J_hat = (对应位置相等的数量) / (总位置数)

        Args:
            sig_a: 文档 A 的签名
            sig_b: 文档 B 的签名

        Returns:
            估计的 Jaccard 相似度
        """
        assert len(sig_a) == len(sig_b) == self.num_hashes
        matches = sum(1 for a, b in zip(sig_a, sig_b) if a == b)
        return matches / self.num_hashes


class LSHIndex:
    """
    LSH（局部敏感哈希）索引

    使用分带（banding）策略将相似的签名映射到同一桶中，
    从而高效地找到近似重复的文档对。

    分带策略:
    - 将长度为 k 的签名分为 b 个带，每带 r = k/b 行
    - 如果两个签名在至少一个带中完全匹配，则成为候选对
    - 候选概率: P(candidate | s) = 1 - (1 - s^r)^b
    - 近似阈值: t ≈ (1/b)^(1/r)

    Args:
        num_bands: 带的数量 b
        rows_per_band: 每带的行数 r
    """

    def __init__(self, num_bands: int, rows_per_band: int):
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.num_hashes = num_bands * rows_per_band
        # 每个带对应一个桶字典: {band_hash: set(doc_ids)}
        self._buckets: List[Dict[int, Set]] = [
            defaultdict(set) for _ in range(num_bands)
        ]
        # 近似阈值
        self.threshold = (1.0 / num_bands) ** (1.0 / rows_per_band)

    def insert(self, doc_id: int, signature: List[int]) -> None:
        """
        将文档签名插入 LSH 索引

        Args:
            doc_id: 文档标识符
            signature: MinHash 签名向量
        """
        assert len(signature) >= self.num_hashes, (
            f"签名长度 {len(signature)} < 所需长度 {self.num_hashes}"
        )

        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = tuple(signature[start:end])
            band_hash = hash(band)
            self._buckets[band_idx][band_hash].add(doc_id)

    def query_candidates(self) -> Set[Tuple[int, int]]:
        """
        从 LSH 索引中提取所有候选对

        遍历所有桶，将同一桶中的文档两两配对。

        Returns:
            候选文档对的集合 {(doc_id_a, doc_id_b), ...}
        """
        candidates = set()

        for band_buckets in self._buckets:
            for bucket_docs in band_buckets.values():
                if len(bucket_docs) < 2:
                    continue
                doc_list = sorted(bucket_docs)
                for i in range(len(doc_list)):
                    for j in range(i + 1, len(doc_list)):
                        candidates.add((doc_list[i], doc_list[j]))

        return candidates

    def candidate_probability(self, similarity: float) -> float:
        """
        计算给定相似度下成为候选对的概率

        P(candidate | s) = 1 - (1 - s^r)^b

        Args:
            similarity: 真实 Jaccard 相似度

        Returns:
            成为候选对的概率
        """
        return 1.0 - (1.0 - similarity ** self.rows_per_band) ** self.num_bands


class MinHashLSHDeduplicator:
    """
    完整的 MinHash + LSH 去重器

    将 MinHash 签名计算和 LSH 索引整合为一个完整的去重管线。

    使用流程:
    1. 将文档转为 n-gram 集合
    2. 计算 MinHash 签名
    3. 插入 LSH 索引
    4. 提取候选对并验证
    5. 标记重复文档

    Args:
        num_hashes: 哈希函数数量（签名长度）
        num_bands: LSH 带数
        threshold: Jaccard 相似度阈值
        n_gram: n-gram 的 n 值
        seed: 随机种子
    """

    def __init__(
        self,
        num_hashes: int = 128,
        num_bands: int = 16,
        threshold: float = 0.8,
        n_gram: int = 5,
        seed: int = 42,
    ):
        self.threshold = threshold
        self.n_gram = n_gram

        # 确保 num_hashes 能被 num_bands 整除
        rows_per_band = num_hashes // num_bands
        actual_num_hashes = num_bands * rows_per_band

        self.minhasher = MinHasher(num_hashes=actual_num_hashes, seed=seed)
        self.lsh_index = LSHIndex(
            num_bands=num_bands, rows_per_band=rows_per_band
        )

        # 存储中间结果
        self._shingles: Dict[int, Set[str]] = {}
        self._signatures: Dict[int, List[int]] = {}

    def deduplicate(
        self, documents: Dict[int, str], verbose: bool = True
    ) -> List[int]:
        """
        对文档集合执行去重

        Args:
            documents: {doc_id: text} 文档字典
            verbose: 是否打印进度信息

        Returns:
            保留的文档 ID 列表
        """
        if verbose:
            print(f"[去重] 开始处理 {len(documents)} 个文档")
            print(f"  参数: num_hashes={self.minhasher.num_hashes}, "
                  f"bands={self.lsh_index.num_bands}, "
                  f"rows={self.lsh_index.rows_per_band}")
            print(f"  近似阈值: {self.lsh_index.threshold:.4f}, "
                  f"设定阈值: {self.threshold:.4f}")

        # 步骤 1: 计算 shingles 和 MinHash 签名
        if verbose:
            print("[步骤1] 计算 n-gram 和 MinHash 签名...")

        for doc_id, text in documents.items():
            shingles = get_shingles(text, self.n_gram)
            self._shingles[doc_id] = shingles
            signature = self.minhasher.compute_signature(shingles)
            self._signatures[doc_id] = signature
            self.lsh_index.insert(doc_id, signature)

        # 步骤 2: 获取候选对
        if verbose:
            print("[步骤2] LSH 分桶，提取候选对...")
        candidates = self.lsh_index.query_candidates()

        if verbose:
            print(f"  候选对数量: {len(candidates)}")

        # 步骤 3: 验证候选对，标记重复
        if verbose:
            print("[步骤3] 验证候选对...")
        duplicates = set()

        for doc_a, doc_b in candidates:
            if doc_a in duplicates or doc_b in duplicates:
                continue

            # 计算精确 Jaccard 相似度
            sim = jaccard_similarity(
                self._shingles[doc_a], self._shingles[doc_b]
            )

            if sim >= self.threshold:
                # 保留较长的文档
                if len(documents[doc_a]) >= len(documents[doc_b]):
                    duplicates.add(doc_b)
                else:
                    duplicates.add(doc_a)

        # 步骤 4: 返回保留的文档
        kept = [
            doc_id for doc_id in documents.keys()
            if doc_id not in duplicates
        ]

        if verbose:
            print(f"[完成] 去重结果: "
                  f"{len(documents)} -> {len(kept)} 个文档 "
                  f"(移除 {len(duplicates)} 个重复)")

        return kept

    def get_duplicate_pairs(self) -> List[Tuple[int, int, float]]:
        """
        获取所有检测到的重复文档对及其相似度

        Returns:
            [(doc_id_a, doc_id_b, similarity), ...] 列表
        """
        candidates = self.lsh_index.query_candidates()
        pairs = []

        for doc_a, doc_b in candidates:
            sim = jaccard_similarity(
                self._shingles[doc_a], self._shingles[doc_b]
            )
            if sim >= self.threshold:
                pairs.append((doc_a, doc_b, sim))

        return sorted(pairs, key=lambda x: -x[2])


def exact_dedup(documents: Dict[int, str]) -> List[int]:
    """
    精确去重：基于 SHA-256 哈希

    计算每个文档的 SHA-256 哈希值，完全相同的文档只保留一个。

    Args:
        documents: {doc_id: text} 文档字典

    Returns:
        保留的文档 ID 列表
    """
    seen_hashes: Dict[str, int] = {}
    kept = []

    for doc_id, text in documents.items():
        doc_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if doc_hash not in seen_hashes:
            seen_hashes[doc_hash] = doc_id
            kept.append(doc_id)

    return kept


if __name__ == '__main__':
    # ===== 演示: MinHash + LSH 去重 =====
    print("=" * 60)
    print("MinHash + LSH 近似去重 演示")
    print("=" * 60)

    # 构造测试数据: 包含精确重复和近似重复
    documents = {
        0: "The quick brown fox jumps over the lazy dog in the park on a sunny day",
        1: "The quick brown fox jumps over the lazy dog in the park on a sunny day",  # 精确重复
        2: "The quick brown fox leaps over the lazy dog in the park on a sunny afternoon",  # 近似重复
        3: "A completely different document about machine learning and neural networks",
        4: "Another unique document discussing natural language processing techniques",
        5: "The fast brown fox jumps over the lazy dog in the garden on a sunny day",  # 近似重复
        6: "Deep learning models have revolutionized the field of artificial intelligence",
        7: "A completely different document about machine learning and neural networks today",  # 近似重复于 3
    }

    print(f"\n输入文档数: {len(documents)}")
    print("-" * 40)

    # 1. 精确去重
    print("\n--- 精确去重 ---")
    exact_kept = exact_dedup(documents)
    print(f"精确去重后: {len(exact_kept)} 个文档")
    print(f"保留的 ID: {exact_kept}")

    # 2. MinHash + LSH 近似去重
    print("\n--- MinHash + LSH 近似去重 ---")
    deduplicator = MinHashLSHDeduplicator(
        num_hashes=128,
        num_bands=16,
        threshold=0.5,
        n_gram=3,
    )
    kept_ids = deduplicator.deduplicate(documents)
    print(f"保留的 ID: {kept_ids}")

    # 3. 显示检测到的重复对
    print("\n--- 检测到的重复对 ---")
    pairs = deduplicator.get_duplicate_pairs()
    for doc_a, doc_b, sim in pairs:
        print(f"  文档 {doc_a} <-> 文档 {doc_b}: "
              f"Jaccard = {sim:.4f}")

    # 4. Jaccard 相似度验证
    print("\n--- Jaccard 相似度矩阵 ---")
    for i in range(min(5, len(documents))):
        row = []
        for j in range(min(5, len(documents))):
            s_i = get_shingles(documents[i], n=3)
            s_j = get_shingles(documents[j], n=3)
            sim = jaccard_similarity(s_i, s_j)
            row.append(f"{sim:.2f}")
        print(f"  Doc {i}: [{', '.join(row)}]")

    # 5. LSH 概率曲线
    print("\n--- LSH 候选概率 ---")
    lsh = LSHIndex(num_bands=16, rows_per_band=8)
    for s in [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        p = lsh.candidate_probability(s)
        print(f"  s = {s:.2f} -> P(candidate) = {p:.6f}")
