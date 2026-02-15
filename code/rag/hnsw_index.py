"""
HNSW（Hierarchical Navigable Small World）图索引：简化实现

本模块实现了 HNSW 近似最近邻搜索算法的核心逻辑，包括：
1. 分层图的构建（多层导航结构）
2. 贪心搜索算法（从顶层到底层的逐层搜索）
3. 节点插入（随机层数 + 双向连接 + 连接修剪）

HNSW 的核心思想：
- 结合跳表（Skip List）的分层思想和小世界网络（NSW）的导航性质
- 顶层稀疏节点提供长程跳转，底层密集节点提供精细搜索
- 搜索复杂度: O(d * log N)，其中 d 是向量维度，N 是节点数

关键超参数：
- M: 每层最大连接数（越大精度越高，内存越大）
- ef_construction: 构建时的搜索候选集大小（越大索引质量越好）
- ef_search: 搜索时的候选集大小（越大精度越高，速度越慢）
- m_L: 层间缩放因子，通常为 1/ln(M)

参考论文:
    Malkov & Yashunin (2018). "Efficient and Robust Approximate Nearest
    Neighbor Using Hierarchical Navigable Small World Graphs"
"""

import heapq
import math
import random
import time
from typing import Optional

import numpy as np


class HNSWNode:
    """
    HNSW 图中的节点

    参数:
        node_id: 节点唯一标识
        vector: 节点的向量表示
        level: 该节点所在的最高层（0-indexed）
    """

    def __init__(self, node_id: int, vector: np.ndarray, level: int):
        self.node_id = node_id
        self.vector = vector
        self.level = level
        # neighbors[l] = 第 l 层的邻居节点 ID 集合
        self.neighbors: dict[int, set[int]] = {l: set() for l in range(level + 1)}


class HNSWIndex:
    """
    HNSW 近似最近邻索引

    实现了完整的 HNSW 算法，包括分层图构建和多层贪心搜索。

    参数:
        dim: 向量维度
        M: 每层最大连接数（底层为 2*M）
        ef_construction: 构建时的搜索候选集大小
        m_L: 层间缩放因子（None 则自动计算为 1/ln(M)）
        distance_metric: 距离度量（"l2" 或 "cosine"）
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        m_L: Optional[float] = None,
        distance_metric: str = "l2",
    ):
        self.dim = dim
        self.M = M
        self.M_max = M  # 高层最大连接数
        self.M_max0 = 2 * M  # 底层最大连接数（底层需要更多连接）
        self.ef_construction = ef_construction
        self.m_L = m_L if m_L is not None else 1.0 / math.log(M)
        self.distance_metric = distance_metric

        # 图数据
        self.nodes: dict[int, HNSWNode] = {}
        self.entry_point: Optional[int] = None
        self.max_level: int = -1
        self.num_nodes: int = 0

    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        计算两个向量之间的距离

        参数:
            v1, v2: 两个向量

        返回:
            距离值（越小越相似）
        """
        if self.distance_metric == "l2":
            return float(np.sum((v1 - v2) ** 2))
        elif self.distance_metric == "cosine":
            # 余弦距离 = 1 - 余弦相似度
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 1.0
            return 1.0 - float(np.dot(v1, v2) / (norm1 * norm2))
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")

    def _random_level(self) -> int:
        """
        随机确定新节点的层数

        使用指数分布: level = floor(-ln(rand()) * m_L)
        这确保高层节点数量指数递减。

        返回:
            随机层数（0-indexed）
        """
        return int(-math.log(random.random()) * self.m_L)

    def _search_layer(
        self,
        query: np.ndarray,
        entry_points: list[int],
        layer: int,
        ef: int,
    ) -> list[tuple[float, int]]:
        """
        在指定层执行贪心搜索

        这是 HNSW 的核心搜索子过程。使用优先队列维护候选集和结果集。

        参数:
            query: 查询向量
            entry_points: 入口节点 ID 列表
            layer: 搜索的层级
            ef: 候选集大小（控制搜索精度）

        返回:
            [(distance, node_id), ...] 最近的 ef 个节点（按距离升序）
        """
        visited = set(entry_points)

        # candidates: 最小堆，距离最近的在堆顶（用于扩展搜索）
        candidates = []
        # results: 最大堆，距离最远的在堆顶（用于维护 Top-ef 结果）
        results = []

        for ep in entry_points:
            dist = self._distance(query, self.nodes[ep].vector)
            heapq.heappush(candidates, (dist, ep))
            heapq.heappush(results, (-dist, ep))  # 取负实现最大堆

        while candidates:
            # 获取候选集中距离最近的节点
            c_dist, c_id = heapq.heappop(candidates)

            # 获取结果集中距离最远的节点
            f_dist = -results[0][0]

            # 如果候选集中最近的也比结果集中最远的还远，停止搜索
            if c_dist > f_dist:
                break

            # 遍历当前节点在该层的所有邻居
            node = self.nodes[c_id]
            if layer not in node.neighbors:
                continue

            for neighbor_id in node.neighbors[layer]:
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)

                dist = self._distance(query, self.nodes[neighbor_id].vector)
                f_dist = -results[0][0]

                # 如果这个邻居比当前结果集中最远的更近，或结果集未满
                if dist < f_dist or len(results) < ef:
                    heapq.heappush(candidates, (dist, neighbor_id))
                    heapq.heappush(results, (-dist, neighbor_id))

                    # 如果结果集超过 ef，移除最远的
                    if len(results) > ef:
                        heapq.heappop(results)

        # 将结果转换为按距离升序的列表
        result_list = [(-dist, node_id) for dist, node_id in results]
        result_list.sort(key=lambda x: x[0])
        return result_list

    def _select_neighbors(
        self,
        query: np.ndarray,
        candidates: list[tuple[float, int]],
        M: int,
    ) -> list[tuple[float, int]]:
        """
        从候选集中选择最优的 M 个邻居

        使用简单策略：选择距离最近的 M 个。
        更高级的策略（如启发式修剪）可以提供更好的连通性。

        参数:
            query: 查询向量
            candidates: [(distance, node_id), ...]
            M: 最大选择数量

        返回:
            选中的邻居列表
        """
        # 按距离排序，选择最近的 M 个
        sorted_candidates = sorted(candidates, key=lambda x: x[0])
        return sorted_candidates[:M]

    def _prune_connections(self, node_id: int, layer: int, M_max: int):
        """
        修剪节点的连接数，保持不超过 M_max

        当节点的邻居数超过上限时，移除距离最远的连接。

        参数:
            node_id: 待修剪的节点 ID
            layer: 层级
            M_max: 最大连接数
        """
        node = self.nodes[node_id]
        if len(node.neighbors[layer]) <= M_max:
            return

        # 计算到所有邻居的距离
        neighbor_dists = []
        for n_id in node.neighbors[layer]:
            dist = self._distance(node.vector, self.nodes[n_id].vector)
            neighbor_dists.append((dist, n_id))

        # 按距离排序，保留最近的 M_max 个
        neighbor_dists.sort(key=lambda x: x[0])
        keep = set(n_id for _, n_id in neighbor_dists[:M_max])
        remove = node.neighbors[layer] - keep

        # 移除多余的连接（双向）
        for r_id in remove:
            node.neighbors[layer].discard(r_id)
            if layer in self.nodes[r_id].neighbors:
                self.nodes[r_id].neighbors[layer].discard(node_id)

    def insert(self, node_id: int, vector: np.ndarray):
        """
        插入新节点到 HNSW 索引

        步骤:
        1. 随机确定新节点的层数
        2. 从顶层到新节点层+1层: 贪心搜索找入口点
        3. 从新节点层到第0层: 搜索邻居 + 建立双向连接

        参数:
            node_id: 节点唯一标识
            vector: 节点的向量表示
        """
        # 确保向量是 numpy 数组
        vector = np.asarray(vector, dtype=np.float32)

        # 随机确定层数
        level = self._random_level()

        # 创建节点
        node = HNSWNode(node_id, vector, level)
        self.nodes[node_id] = node
        self.num_nodes += 1

        # 如果是第一个节点，设为入口点
        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = level
            return

        # 从顶层开始搜索
        ep = [self.entry_point]

        # 阶段 1: 从最高层到 level+1 层，每层贪心找最近的 1 个点
        for l in range(self.max_level, level, -1):
            results = self._search_layer(vector, ep, l, ef=1)
            if results:
                ep = [results[0][1]]

        # 阶段 2: 从 min(level, max_level) 层到第 0 层，搜索并连接
        for l in range(min(level, self.max_level), -1, -1):
            # 在当前层搜索最近邻
            results = self._search_layer(vector, ep, l, ef=self.ef_construction)

            # 确定该层的最大连接数
            M_max = self.M_max0 if l == 0 else self.M_max

            # 选择最优的 M 个邻居
            selected = self._select_neighbors(vector, results, M_max)

            # 建立双向连接
            for dist, neighbor_id in selected:
                node.neighbors[l].add(neighbor_id)
                self.nodes[neighbor_id].neighbors.setdefault(l, set())
                self.nodes[neighbor_id].neighbors[l].add(node_id)

                # 如果邻居的连接数超过上限，修剪
                if len(self.nodes[neighbor_id].neighbors[l]) > M_max:
                    self._prune_connections(neighbor_id, l, M_max)

            # 更新入口点列表
            ep = [n_id for _, n_id in selected]
            if not ep:
                ep = [self.entry_point]

        # 如果新节点的层数超过当前最高层，更新入口点
        if level > self.max_level:
            self.entry_point = node_id
            self.max_level = level

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> list[tuple[int, float]]:
        """
        在 HNSW 索引中搜索最近邻

        步骤:
        1. 从顶层到第 1 层: 每层贪心搜索，ef=1
        2. 在第 0 层: 精细搜索，ef=ef_search

        参数:
            query: 查询向量
            k: 返回的近邻数量
            ef_search: 搜索候选集大小（越大越精确，越慢）

        返回:
            [(node_id, distance), ...] 按距离升序排列
        """
        if self.entry_point is None:
            return []

        if ef_search is None:
            ef_search = max(k, self.ef_construction)

        query = np.asarray(query, dtype=np.float32)
        ep = [self.entry_point]

        # 阶段 1: 从顶层到第 1 层，每层贪心找最近的 1 个点
        for l in range(self.max_level, 0, -1):
            results = self._search_layer(query, ep, l, ef=1)
            if results:
                ep = [results[0][1]]

        # 阶段 2: 在第 0 层精细搜索
        results = self._search_layer(query, ep, 0, ef=ef_search)

        # 返回前 K 个
        return [(node_id, dist) for dist, node_id in results[:k]]

    def brute_force_search(
        self, query: np.ndarray, k: int = 10
    ) -> list[tuple[int, float]]:
        """
        暴力搜索（用于验证 HNSW 搜索的准确性）

        参数:
            query: 查询向量
            k: 返回的近邻数量

        返回:
            [(node_id, distance), ...] 按距离升序排列
        """
        query = np.asarray(query, dtype=np.float32)
        distances = []
        for node_id, node in self.nodes.items():
            dist = self._distance(query, node.vector)
            distances.append((node_id, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def get_stats(self) -> dict:
        """
        获取索引统计信息

        返回:
            包含各种统计指标的字典
        """
        if self.num_nodes == 0:
            return {"num_nodes": 0}

        # 统计每层的节点数
        level_counts = {}
        total_connections = 0
        for node in self.nodes.values():
            for l in node.neighbors:
                level_counts[l] = level_counts.get(l, 0) + 1
                total_connections += len(node.neighbors[l])

        avg_connections = total_connections / self.num_nodes if self.num_nodes > 0 else 0

        return {
            "num_nodes": self.num_nodes,
            "max_level": self.max_level,
            "entry_point": self.entry_point,
            "level_distribution": dict(sorted(level_counts.items())),
            "avg_connections_per_node": avg_connections,
            "M": self.M,
            "ef_construction": self.ef_construction,
        }


def compute_recall(
    hnsw_results: list[tuple[int, float]],
    bf_results: list[tuple[int, float]],
) -> float:
    """
    计算 HNSW 搜索结果相对于暴力搜索的 Recall

    Recall@K = |HNSW_TopK ∩ BF_TopK| / K

    参数:
        hnsw_results: HNSW 搜索结果
        bf_results: 暴力搜索结果

    返回:
        Recall 值 [0, 1]
    """
    hnsw_ids = set(node_id for node_id, _ in hnsw_results)
    bf_ids = set(node_id for node_id, _ in bf_results)
    if len(bf_ids) == 0:
        return 1.0
    return len(hnsw_ids & bf_ids) / len(bf_ids)


if __name__ == "__main__":
    print("=" * 60)
    print("HNSW 索引演示")
    print("=" * 60)

    # 设置随机种子
    random.seed(42)
    np.random.seed(42)

    # 1. 创建索引
    print("\n--- 1. 创建 HNSW 索引 ---")
    dim = 32
    M = 16
    ef_construction = 100

    index = HNSWIndex(
        dim=dim,
        M=M,
        ef_construction=ef_construction,
        distance_metric="l2",
    )
    print(f"索引参数: dim={dim}, M={M}, ef_construction={ef_construction}")

    # 2. 插入数据
    print("\n--- 2. 插入向量数据 ---")
    num_vectors = 2000
    data = np.random.randn(num_vectors, dim).astype(np.float32)

    start_time = time.time()
    for i in range(num_vectors):
        index.insert(i, data[i])
        if (i + 1) % 500 == 0:
            print(f"  已插入 {i + 1}/{num_vectors} 个向量")
    build_time = time.time() - start_time
    print(f"索引构建完成, 耗时: {build_time:.2f} 秒")

    # 3. 查看索引统计
    print("\n--- 3. 索引统计 ---")
    stats = index.get_stats()
    print(f"  节点数: {stats['num_nodes']}")
    print(f"  最大层级: {stats['max_level']}")
    print(f"  入口点: {stats['entry_point']}")
    print(f"  层级节点分布: {stats['level_distribution']}")
    print(f"  平均每节点连接数: {stats['avg_connections_per_node']:.1f}")

    # 4. 搜索测试
    print("\n--- 4. 搜索测试 ---")
    num_queries = 50
    k = 10
    queries = np.random.randn(num_queries, dim).astype(np.float32)

    # HNSW 搜索
    start_time = time.time()
    hnsw_results_all = [index.search(q, k=k, ef_search=50) for q in queries]
    hnsw_time = time.time() - start_time

    # 暴力搜索
    start_time = time.time()
    bf_results_all = [index.brute_force_search(q, k=k) for q in queries]
    bf_time = time.time() - start_time

    # 计算 Recall
    recalls = [
        compute_recall(hnsw_res, bf_res)
        for hnsw_res, bf_res in zip(hnsw_results_all, bf_results_all)
    ]
    avg_recall = sum(recalls) / len(recalls)

    print(f"  查询数: {num_queries}, K={k}")
    print(f"  HNSW 搜索耗时: {hnsw_time:.4f} 秒 ({hnsw_time/num_queries*1000:.2f} ms/query)")
    print(f"  暴力搜索耗时: {bf_time:.4f} 秒 ({bf_time/num_queries*1000:.2f} ms/query)")
    print(f"  加速比: {bf_time/hnsw_time:.1f}x")
    print(f"  平均 Recall@{k}: {avg_recall:.4f}")

    # 5. ef_search 参数影响
    print("\n--- 5. ef_search 参数对 Recall 的影响 ---")
    print(f"  {'ef_search':>10s}  {'Recall@10':>10s}  {'Time(ms)':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}")

    for ef in [10, 20, 50, 100, 200]:
        start_time = time.time()
        results_list = [index.search(q, k=k, ef_search=ef) for q in queries]
        search_time = (time.time() - start_time) / num_queries * 1000

        recalls = [
            compute_recall(hnsw_res, bf_res)
            for hnsw_res, bf_res in zip(results_list, bf_results_all)
        ]
        avg_recall = sum(recalls) / len(recalls)
        print(f"  {ef:>10d}  {avg_recall:>10.4f}  {search_time:>10.2f}")

    # 6. 单个查询详细结果
    print("\n--- 6. 单个查询详细结果对比 ---")
    q = queries[0]
    hnsw_result = index.search(q, k=5, ef_search=100)
    bf_result = index.brute_force_search(q, k=5)

    print("HNSW Top-5:")
    for node_id, dist in hnsw_result:
        print(f"  Node {node_id}: dist = {dist:.4f}")

    print("暴力搜索 Top-5:")
    for node_id, dist in bf_result:
        print(f"  Node {node_id}: dist = {dist:.4f}")

    recall = compute_recall(hnsw_result, bf_result)
    print(f"Recall@5 = {recall:.2f}")

    print("\n演示完成!")
