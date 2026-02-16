"""
合成数据语义级去重 (Semantic Deduplication)

本模块实现了针对合成数据的语义级去重工具。与简单的字符串匹配不同，
语义去重基于文本的向量表示计算相似度，能够识别出措辞不同但含义相同的重复内容。

核心方法:
1. 精确去重: 基于 MD5 哈希的完全匹配去重
2. 模糊去重: 基于 MinHash 的近似文本去重
3. 语义去重: 基于 Sentence Transformer 嵌入的语义相似度去重

依赖:
- sentence-transformers: 语义向量编码
- numpy: 向量运算
- datasketch: MinHash 近似去重 (可选)

安装:
    pip install sentence-transformers numpy datasketch

参考:
- Lee et al. (2022). Deduplicating Training Data Makes Language Models Better.
- Abbas et al. (2023). SemDeDup: Data-efficient learning at web-scale through semantic deduplication.
"""

import json
import hashlib
import logging
from typing import Optional
from pathlib import Path
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 精确去重
# ============================================================

class ExactDeduplicator:
    """
    基于 MD5 哈希的精确去重

    特点: 快速、无误报，但只能检测完全相同的文本
    """

    def __init__(self):
        self.seen_hashes: set[str] = set()

    @staticmethod
    def _hash(text: str) -> str:
        """计算文本的 MD5 哈希"""
        # 标准化：去除首尾空格和多余空行
        normalized = "\n".join(line.strip() for line in text.strip().split("\n"))
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def is_duplicate(self, text: str) -> bool:
        """检查是否为重复文本"""
        h = self._hash(text)
        if h in self.seen_hashes:
            return True
        self.seen_hashes.add(h)
        return False

    def deduplicate(self, texts: list[str]) -> list[str]:
        """去重并返回去重后的文本列表"""
        self.seen_hashes.clear()
        result = []
        for text in texts:
            if not self.is_duplicate(text):
                result.append(text)
        logger.info("精确去重: %d → %d (去除 %d)", len(texts), len(result), len(texts) - len(result))
        return result


# ============================================================
# MinHash 近似去重
# ============================================================

class MinHashDeduplicator:
    """
    基于 MinHash + LSH 的近似去重

    原理:
    - 将文本表示为 n-gram 的集合
    - 用 MinHash 将集合压缩为固定大小的签名
    - 两个签名中相同元素的比例近似 Jaccard 相似度
    - 用 LSH (Locality-Sensitive Hashing) 加速近似最近邻搜索

    适用: 检测经过轻微修改（换词、调整顺序）的近重复文本
    """

    def __init__(self, ngram_size: int = 5, num_perm: int = 128, threshold: float = 0.7):
        """
        Args:
            ngram_size: n-gram 大小
            num_perm: MinHash 置换数量（越大越精确，但越慢）
            threshold: Jaccard 相似度阈值，超过则判为重复
        """
        self.ngram_size = ngram_size
        self.num_perm = num_perm
        self.threshold = threshold

    def _get_ngrams(self, text: str) -> set[str]:
        """提取文本的 n-gram 集合"""
        words = text.lower().split()
        if len(words) < self.ngram_size:
            return {text.lower()}
        return {" ".join(words[i:i + self.ngram_size]) for i in range(len(words) - self.ngram_size + 1)}

    def deduplicate(self, texts: list[str]) -> list[str]:
        """
        使用 MinHash + LSH 进行近似去重

        Returns:
            去重后的文本列表
        """
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:
            logger.warning("datasketch 未安装，回退到精确去重。安装: pip install datasketch")
            return ExactDeduplicator().deduplicate(texts)

        # 创建 LSH 索引
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        minhashes = []

        # 计算每个文本的 MinHash
        for i, text in enumerate(texts):
            m = MinHash(num_perm=self.num_perm)
            for ngram in self._get_ngrams(text):
                m.update(ngram.encode("utf-8"))
            minhashes.append(m)

        # 插入 LSH 并检测重复
        keep_indices = set()
        for i, m in enumerate(minhashes):
            # 查找近似重复
            result = lsh.query(m)
            if not result:
                # 没有近似重复，保留并加入索引
                try:
                    lsh.insert(str(i), m)
                    keep_indices.add(i)
                except ValueError:
                    # 键冲突（极罕见），跳过
                    pass
            # 有近似重复则丢弃（保留最先出现的）

        result = [texts[i] for i in sorted(keep_indices)]
        logger.info("MinHash 去重: %d → %d (去除 %d)", len(texts), len(result), len(texts) - len(result))
        return result


# ============================================================
# 语义去重
# ============================================================

class SemanticDeduplicator:
    """
    基于语义向量的去重

    原理:
    - 用 Sentence Transformer 将文本编码为向量
    - 计算向量间的余弦相似度
    - 相似度超过阈值的判为语义重复

    优势: 能识别措辞完全不同但含义相同的重复
    劣势: 计算成本高，需要 GPU 加速
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        threshold: float = 0.85,
        batch_size: int = 64,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_name: Sentence Transformer 模型名称
            threshold: 余弦相似度阈值
            batch_size: 编码批大小
            device: 计算设备 (None=自动选择)
        """
        self.model_name = model_name
        self.threshold = threshold
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load_model(self):
        """延迟加载模型"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("加载语义模型: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name, device=self.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        """将文本列表编码为向量矩阵"""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,  # L2 归一化，余弦相似度变为内积
        )
        return embeddings

    def deduplicate(self, texts: list[str]) -> list[str]:
        """
        语义去重

        算法:
        1. 编码所有文本
        2. 按顺序遍历，每个文本与已保留文本计算余弦相似度
        3. 若最大相似度 > 阈值，则判为重复并丢弃

        时间复杂度: O(N^2 * d)，其中 N 是文本数，d 是向量维度
        对于大规模数据，建议先用 MinHash 粗筛再用语义精筛
        """
        if not texts:
            return []

        embeddings = self.encode(texts)
        n = len(texts)

        # 贪心去重：按顺序保留，丢弃与已保留项相似度超过阈值的
        keep_mask = [True] * n
        kept_embeddings = []
        kept_indices = []

        for i in range(n):
            if not keep_mask[i]:
                continue

            if kept_embeddings:
                # 计算与所有已保留文本的余弦相似度
                kept_matrix = np.stack(kept_embeddings)
                sims = np.dot(kept_matrix, embeddings[i])
                max_sim = np.max(sims)

                if max_sim > self.threshold:
                    keep_mask[i] = False
                    continue

            kept_embeddings.append(embeddings[i])
            kept_indices.append(i)

        result = [texts[i] for i in kept_indices]
        logger.info(
            "语义去重: %d → %d (去除 %d, 阈值=%.2f)",
            n, len(result), n - len(result), self.threshold,
        )
        return result

    def find_clusters(self, texts: list[str], min_cluster_size: int = 2) -> list[list[int]]:
        """
        找到语义相似的文本簇

        Returns:
            簇列表，每个簇包含相似文本的索引
        """
        if not texts:
            return []

        embeddings = self.encode(texts)
        n = len(texts)

        # 计算相似度矩阵
        sim_matrix = np.dot(embeddings, embeddings.T)

        # Union-Find 聚类
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] > self.threshold:
                    union(i, j)

        # 收集簇
        clusters = defaultdict(list)
        for i in range(n):
            clusters[find(i)].append(i)

        # 过滤小簇
        result = [indices for indices in clusters.values() if len(indices) >= min_cluster_size]
        logger.info("找到 %d 个语义重复簇（阈值=%.2f）", len(result), self.threshold)
        return result


# ============================================================
# 综合去重管线
# ============================================================

class DeduplicationPipeline:
    """
    多层去重管线

    推荐流程:
    1. 精确去重（最快，去除完全重复）
    2. MinHash 去重（较快，去除近重复）
    3. 语义去重（最慢，去除语义重复）

    按需选择层数：数据量大时建议三层全开，数据量小时可只用精确+语义
    """

    def __init__(
        self,
        use_exact: bool = True,
        use_minhash: bool = True,
        use_semantic: bool = True,
        minhash_threshold: float = 0.7,
        semantic_threshold: float = 0.85,
        semantic_model: str = "all-MiniLM-L6-v2",
    ):
        self.stages = []

        if use_exact:
            self.stages.append(("精确去重", ExactDeduplicator()))
        if use_minhash:
            self.stages.append(("MinHash去重", MinHashDeduplicator(threshold=minhash_threshold)))
        if use_semantic:
            self.stages.append(("语义去重", SemanticDeduplicator(
                model_name=semantic_model,
                threshold=semantic_threshold,
            )))

    def deduplicate(
        self,
        data: list[dict],
        text_key: str = "instruction",
    ) -> list[dict]:
        """
        对数据列表执行多层去重

        Args:
            data: 数据列表
            text_key: 用于去重比较的文本字段名

        Returns:
            去重后的数据列表
        """
        logger.info("开始多层去重，输入 %d 条数据", len(data))
        current_data = data

        for stage_name, deduplicator in self.stages:
            logger.info("--- 执行 %s ---", stage_name)
            texts = [item[text_key] for item in current_data]
            unique_texts = set(deduplicator.deduplicate(texts))

            # 保留去重后仍存在的数据
            current_data = [item for item in current_data if item[text_key] in unique_texts]
            logger.info("%s 后剩余: %d 条", stage_name, len(current_data))

        logger.info("去重完成: %d → %d (去除 %.1f%%)",
                     len(data), len(current_data),
                     100 * (1 - len(current_data) / max(len(data), 1)))
        return current_data


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="合成数据语义级去重")
    parser.add_argument("input_file", type=str, help="输入 JSON 文件")
    parser.add_argument("--output", type=str, default="deduplicated_output.json", help="输出文件路径")
    parser.add_argument("--text-key", type=str, default="instruction", help="去重文本字段名")
    parser.add_argument("--no-exact", action="store_true", help="跳过精确去重")
    parser.add_argument("--no-minhash", action="store_true", help="跳过 MinHash 去重")
    parser.add_argument("--no-semantic", action="store_true", help="跳过语义去重")
    parser.add_argument("--minhash-threshold", type=float, default=0.7, help="MinHash 相似度阈值")
    parser.add_argument("--semantic-threshold", type=float, default=0.85, help="语义相似度阈值")
    parser.add_argument("--semantic-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence Transformer 模型名称")
    parser.add_argument("--show-clusters", action="store_true", help="显示语义重复簇")
    args = parser.parse_args()

    # 加载数据
    data = json.loads(Path(args.input_file).read_text(encoding="utf-8"))
    logger.info("加载了 %d 条数据", len(data))

    # 显示重复簇（可选）
    if args.show_clusters and not args.no_semantic:
        dedup = SemanticDeduplicator(
            model_name=args.semantic_model,
            threshold=args.semantic_threshold,
        )
        texts = [item[args.text_key] for item in data]
        clusters = dedup.find_clusters(texts)

        print(f"\n=== 发现 {len(clusters)} 个语义重复簇 ===\n")
        for i, cluster in enumerate(clusters[:10]):  # 只显示前 10 个
            print(f"簇 {i + 1} ({len(cluster)} 条):")
            for idx in cluster[:3]:  # 每簇显示前 3 条
                print(f"  [{idx}] {texts[idx][:80]}...")
            if len(cluster) > 3:
                print(f"  ... 还有 {len(cluster) - 3} 条")
            print()

    # 执行去重
    pipeline = DeduplicationPipeline(
        use_exact=not args.no_exact,
        use_minhash=not args.no_minhash,
        use_semantic=not args.no_semantic,
        minhash_threshold=args.minhash_threshold,
        semantic_threshold=args.semantic_threshold,
        semantic_model=args.semantic_model,
    )

    deduped = pipeline.deduplicate(data, text_key=args.text_key)

    # 保存结果
    Path(args.output).write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("去重后数据已保存到 %s", args.output)

    print(f"\n=== 去重统计 ===")
    print(f"输入: {len(data)} 条")
    print(f"输出: {len(deduped)} 条")
    print(f"去除: {len(data) - len(deduped)} 条 ({100 * (1 - len(deduped) / max(len(data), 1)):.1f}%)")
