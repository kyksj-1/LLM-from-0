"""
特征可视化工具: 激活分布分析、Top-K 激活样本检索、特征聚类

本模块提供了一套用于分析 SAE (稀疏自编码器) 学到的特征的可视化工具。
这些工具帮助研究者理解模型的内部表示，发现可解释的语义特征。

核心功能:
1. 激活分布分析: 统计每个特征的激活频率、强度分布
2. Top-K 激活样本: 找到最大程度激活某个特征的输入样本
3. 特征聚类: 对学到的特征进行层次聚类，发现语义分组
4. 特征相似度: 计算特征方向之间的余弦相似度

参考:
- Bricken et al. (2023). Towards Monosemanticity.
- Templeton et al. (2024). Scaling Monosemanticity.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class FeatureStats:
    """
    单个特征的统计信息

    Attributes:
        feature_idx: 特征索引
        activation_freq: 激活频率（在所有样本中激活的比例）
        mean_activation: 条件平均激活值（仅在激活时）
        max_activation: 最大激活值
        std_activation: 激活值标准差
        top_k_samples: Top-K 最大激活样本的索引
        top_k_values: Top-K 最大激活值
    """
    feature_idx: int
    activation_freq: float = 0.0
    mean_activation: float = 0.0
    max_activation: float = 0.0
    std_activation: float = 0.0
    top_k_samples: List[int] = field(default_factory=list)
    top_k_values: List[float] = field(default_factory=list)


class FeatureAnalyzer:
    """
    特征分析器

    对 SAE 编码器产生的稀疏特征进行全面分析。

    Args:
        d_sae: SAE 字典大小（特征数量）
        top_k: 每个特征保留的 Top-K 激活样本数
    """

    def __init__(self, d_sae: int, top_k: int = 20):
        self.d_sae = d_sae
        self.top_k = top_k

        # 存储每个特征的激活历史
        self.all_activations: List[torch.Tensor] = []
        self.total_samples = 0

    def collect_activations(self, feature_acts: torch.Tensor):
        """
        收集特征激活值

        Args:
            feature_acts: SAE 编码器输出 [batch, d_sae]
        """
        self.all_activations.append(feature_acts.detach().cpu())
        self.total_samples += feature_acts.shape[0]

    def compute_feature_stats(self) -> List[FeatureStats]:
        """
        计算所有特征的统计信息

        Returns:
            stats_list: 每个特征的 FeatureStats
        """
        if not self.all_activations:
            return []

        # 合并所有激活值
        all_acts = torch.cat(self.all_activations, dim=0)  # [total, d_sae]

        stats_list = []
        for feat_idx in range(self.d_sae):
            feat_acts = all_acts[:, feat_idx]

            # 基本统计
            active_mask = feat_acts > 0
            freq = active_mask.float().mean().item()

            if active_mask.any():
                active_vals = feat_acts[active_mask]
                mean_act = active_vals.mean().item()
                max_act = active_vals.max().item()
                std_act = active_vals.std().item() if len(active_vals) > 1 else 0.0
            else:
                mean_act = max_act = std_act = 0.0

            # Top-K 样本
            top_values, top_indices = feat_acts.topk(
                min(self.top_k, len(feat_acts))
            )

            stats = FeatureStats(
                feature_idx=feat_idx,
                activation_freq=freq,
                mean_activation=mean_act,
                max_activation=max_act,
                std_activation=std_act,
                top_k_samples=top_indices.tolist(),
                top_k_values=top_values.tolist(),
            )
            stats_list.append(stats)

        return stats_list

    def find_dead_features(self, threshold: float = 1e-6) -> List[int]:
        """
        找到死特征（几乎从不激活的特征）

        Args:
            threshold: 激活频率阈值

        Returns:
            dead_indices: 死特征的索引列表
        """
        stats = self.compute_feature_stats()
        return [s.feature_idx for s in stats if s.activation_freq < threshold]

    def find_high_frequency_features(self, threshold: float = 0.5) -> List[int]:
        """
        找到高频特征（过于频繁激活的特征，可能不可解释）

        Args:
            threshold: 频率阈值

        Returns:
            high_freq_indices: 高频特征索引列表
        """
        stats = self.compute_feature_stats()
        return [s.feature_idx for s in stats if s.activation_freq > threshold]


class FeatureClusterer:
    """
    特征聚类工具

    基于解码器权重（特征方向向量）对特征进行聚类，
    发现语义上相近的特征分组。

    Args:
        decoder_weight: 解码器权重矩阵 [d_model, d_sae]
    """

    def __init__(self, decoder_weight: torch.Tensor):
        # 归一化特征方向
        self.directions = decoder_weight.detach().cpu()
        norms = self.directions.norm(dim=0, keepdim=True)
        self.directions = self.directions / (norms + 1e-8)

    def compute_similarity_matrix(self) -> torch.Tensor:
        """
        计算特征间的余弦相似度矩阵

        Returns:
            sim_matrix: [d_sae, d_sae] 余弦相似度矩阵
        """
        # 转置使每行是一个特征方向
        directions_t = self.directions.t()  # [d_sae, d_model]
        sim_matrix = directions_t @ directions_t.t()
        return sim_matrix

    def find_similar_features(
        self, feature_idx: int, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        找到与给定特征最相似的特征

        Args:
            feature_idx: 查询特征索引
            top_k: 返回前 k 个最相似的特征

        Returns:
            similar: [(feature_idx, similarity), ...] 列表
        """
        sim_matrix = self.compute_similarity_matrix()
        sims = sim_matrix[feature_idx]

        # 排除自身
        sims[feature_idx] = -1.0

        values, indices = sims.topk(top_k)
        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]

    def cluster_features(
        self, n_clusters: int = 10
    ) -> Dict[int, List[int]]:
        """
        对特征进行简单的 K-Means 聚类

        使用余弦距离进行聚类，将语义相近的特征分组。

        Args:
            n_clusters: 聚类数量

        Returns:
            clusters: {cluster_id: [feature_indices]}
        """
        directions_t = self.directions.t()  # [d_sae, d_model]
        n_features = directions_t.shape[0]

        # 简单的 K-Means++ 初始化
        centroids = self._kmeans_plus_plus_init(directions_t, n_clusters)

        # 迭代聚类
        max_iters = 100
        for _ in range(max_iters):
            # 分配: 每个特征分配到最近的聚类中心
            sims = directions_t @ centroids.t()  # [n_features, n_clusters]
            assignments = sims.argmax(dim=1)  # [n_features]

            # 更新: 重新计算聚类中心
            new_centroids = torch.zeros_like(centroids)
            for c in range(n_clusters):
                mask = assignments == c
                if mask.any():
                    cluster_dirs = directions_t[mask]
                    new_centroids[c] = cluster_dirs.mean(dim=0)
                    new_centroids[c] = new_centroids[c] / (
                        new_centroids[c].norm() + 1e-8
                    )
                else:
                    new_centroids[c] = centroids[c]

            # 检查收敛
            if torch.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        # 构建聚类结果
        clusters = {}
        for c in range(n_clusters):
            mask = assignments == c
            clusters[c] = torch.where(mask)[0].tolist()

        return clusters

    def _kmeans_plus_plus_init(
        self, data: torch.Tensor, n_clusters: int
    ) -> torch.Tensor:
        """
        K-Means++ 初始化

        选择相互远离的初始聚类中心，改善 K-Means 的收敛性。

        Args:
            data: 数据矩阵 [n, d]
            n_clusters: 聚类数

        Returns:
            centroids: 初始聚类中心 [n_clusters, d]
        """
        n = data.shape[0]
        centroids = []

        # 随机选择第一个中心
        idx = torch.randint(n, (1,)).item()
        centroids.append(data[idx])

        for _ in range(1, n_clusters):
            # 计算每个点到最近中心的距离
            centroid_tensor = torch.stack(centroids)
            sims = data @ centroid_tensor.t()
            min_sims = sims.max(dim=1).values  # 最大相似度 = 最近
            # 用 (1 - sim) 作为距离的代理
            distances = 1.0 - min_sims
            distances = distances.clamp(min=0)

            # 按距离的平方概率采样
            probs = distances.pow(2)
            probs = probs / (probs.sum() + 1e-8)
            idx = torch.multinomial(probs, 1).item()
            centroids.append(data[idx])

        return torch.stack(centroids)


class ActivationHistogram:
    """
    激活值直方图工具

    记录和分析 SAE 特征激活值的分布，
    用于判断特征质量和调整 L1 系数。
    """

    def __init__(self, n_bins: int = 50, max_value: float = 10.0):
        self.n_bins = n_bins
        self.max_value = max_value
        self.bin_edges = torch.linspace(0, max_value, n_bins + 1)
        self.counts: Optional[torch.Tensor] = None

    def update(self, activations: torch.Tensor):
        """
        更新直方图计数

        Args:
            activations: 特征激活值 [batch, d_sae]
        """
        # 只统计正激活
        positive_acts = activations[activations > 0].flatten()

        if len(positive_acts) == 0:
            return

        # 计算直方图
        hist = torch.histc(
            positive_acts.float(),
            bins=self.n_bins,
            min=0.0,
            max=self.max_value,
        )

        if self.counts is None:
            self.counts = hist
        else:
            self.counts += hist

    def get_distribution(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取归一化的分布

        Returns:
            bin_centers: 直方图 bin 中心 [n_bins]
            density: 归一化密度 [n_bins]
        """
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        if self.counts is None:
            return bin_centers, torch.zeros(self.n_bins)

        density = self.counts / (self.counts.sum() + 1e-8)
        return bin_centers, density


if __name__ == "__main__":
    print("=" * 60)
    print("特征可视化工具演示")
    print("=" * 60)

    # ---- 1. 创建模拟数据 ----
    print("\n[1] 创建模拟 SAE 和激活数据...")
    d_model = 64
    d_sae = 256
    n_samples = 2000

    # 模拟解码器权重
    decoder_weight = torch.randn(d_model, d_sae)
    decoder_weight = decoder_weight / decoder_weight.norm(dim=0, keepdim=True)

    # 模拟稀疏特征激活
    feature_acts = torch.zeros(n_samples, d_sae)
    for i in range(n_samples):
        # 每个样本随机激活 5-15 个特征
        n_active = torch.randint(5, 15, (1,)).item()
        active_idx = torch.randperm(d_sae)[:n_active]
        feature_acts[i, active_idx] = torch.abs(torch.randn(n_active)) * 2.0

    # 模拟一些死特征
    feature_acts[:, 200:210] = 0.0

    print(f"    样本数: {n_samples}")
    print(f"    特征数: {d_sae}")

    # ---- 2. 特征统计分析 ----
    print("\n[2] 特征统计分析...")
    analyzer = FeatureAnalyzer(d_sae=d_sae, top_k=10)
    analyzer.collect_activations(feature_acts)

    stats = analyzer.compute_feature_stats()

    # 显示前几个特征的统计
    for s in stats[:5]:
        print(f"    特征 {s.feature_idx}: "
              f"频率={s.activation_freq:.3f}, "
              f"均值={s.mean_activation:.3f}, "
              f"最大值={s.max_activation:.3f}")

    # 死特征检测
    dead_features = analyzer.find_dead_features(threshold=0.001)
    print(f"\n    死特征数: {len(dead_features)}")
    if dead_features:
        print(f"    死特征索引 (前10): {dead_features[:10]}")

    # 高频特征检测
    high_freq = analyzer.find_high_frequency_features(threshold=0.9)
    print(f"    高频特征数 (>90%): {len(high_freq)}")

    # ---- 3. 特征聚类 ----
    print("\n[3] 特征聚类...")
    clusterer = FeatureClusterer(decoder_weight)

    # 相似特征查找
    similar = clusterer.find_similar_features(feature_idx=0, top_k=5)
    print(f"    与特征 0 最相似的特征:")
    for idx, sim in similar:
        print(f"      特征 {idx}: 相似度={sim:.3f}")

    # 聚类
    clusters = clusterer.cluster_features(n_clusters=8)
    print(f"\n    聚类结果 (8 个聚类):")
    for c_id, members in clusters.items():
        print(f"      聚类 {c_id}: {len(members)} 个特征")

    # ---- 4. 激活直方图 ----
    print("\n[4] 激活值分布...")
    histogram = ActivationHistogram(n_bins=20, max_value=8.0)
    histogram.update(feature_acts)

    bin_centers, density = histogram.get_distribution()
    # 打印简易文本直方图
    print("    激活值分布 (文本直方图):")
    for i in range(len(bin_centers)):
        bar_len = int(density[i].item() * 100)
        bar = "#" * bar_len
        if bar_len > 0:
            print(f"    [{bin_centers[i]:.1f}] {bar} ({density[i]:.3f})")

    print("\n演示完成!")
