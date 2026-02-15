"""
数据混合与采样

本模块实现了预训练数据的多源混合策略，包括:
- 按权重从多个数据源采样
- 静态配比与动态调整
- 课程学习（Curriculum Learning）调度
- 数据源统计与分析
- 基于温度的采样调节

数据混合是预训练中的关键决策之一:
- Web 数据提供广泛的语言覆盖
- 书籍数据提供长文本连贯性
- 代码数据提升逻辑推理能力
- 学术数据增强专业知识

参考:
- Xie et al. (2023). DoReMi: Optimizing Data Mixtures by Reweighting.
- Touvron et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
"""

import random
import math
from typing import Dict, Iterator, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class DataSource:
    """
    数据源描述

    Attributes:
        name: 数据源名称
        documents: 文档列表
        weight: 采样权重（归一化前）
        quality_score: 质量评分 [0, 1]
        domain: 领域标签
    """
    name: str
    documents: List[str]
    weight: float = 1.0
    quality_score: float = 1.0
    domain: str = "general"

    @property
    def size(self) -> int:
        """文档数量"""
        return len(self.documents)

    @property
    def total_chars(self) -> int:
        """总字符数"""
        return sum(len(doc) for doc in self.documents)


@dataclass
class MixingConfig:
    """
    数据混合配置

    Attributes:
        source_weights: {数据源名称: 权重}
        temperature: 采样温度，> 1 使分布更均匀，< 1 使分布更集中
        seed: 随机种子
        upsample_factor: 高质量数据的上采样倍数
    """
    source_weights: Dict[str, float] = field(default_factory=dict)
    temperature: float = 1.0
    seed: int = 42
    upsample_factor: float = 1.0


class DataMixer:
    """
    数据混合器

    将多个数据源按指定权重混合为统一的训练数据流。

    支持三种混合策略:
    1. 按比例采样（proportional）: 按权重随机采样
    2. 轮询采样（round-robin）: 交替从各数据源取样
    3. 课程学习（curriculum）: 随训练进度调整权重

    数学原理:
    设有 K 个数据源，权重为 w_1, ..., w_K。
    归一化后: p_i = w_i / sum(w_j)
    每个 batch 中来自第 i 个数据源的期望比例为 p_i。

    温度调节:
    p_i(T) = w_i^{1/T} / sum(w_j^{1/T})
    - T > 1: 分布更均匀（低资源数据获得更多比例）
    - T < 1: 分布更集中（高权重数据更占主导）
    - T = 1: 使用原始权重

    Args:
        sources: 数据源列表
        config: 混合配置
    """

    def __init__(
        self,
        sources: List[DataSource],
        config: Optional[MixingConfig] = None,
    ):
        self.sources = {s.name: s for s in sources}
        self.config = config or MixingConfig()
        self._rng = random.Random(self.config.seed)

        # 设置默认权重
        for source in sources:
            if source.name not in self.config.source_weights:
                self.config.source_weights[source.name] = source.weight

    def _normalize_weights(self) -> Dict[str, float]:
        """
        归一化并应用温度的权重

        p_i(T) = w_i^{1/T} / sum(w_j^{1/T})
        """
        T = self.config.temperature
        weights = {}

        for name, w in self.config.source_weights.items():
            if name in self.sources:
                # 温度调节
                weights[name] = w ** (1.0 / T) if T != 0 else (1.0 if w == max(self.config.source_weights.values()) else 0.0)

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def sample_proportional(
        self, num_samples: int
    ) -> List[Tuple[str, str]]:
        """
        按比例采样

        根据归一化后的权重，从各数据源随机采样文档。

        Args:
            num_samples: 总采样数

        Returns:
            [(数据源名称, 文档文本), ...] 列表
        """
        weights = self._normalize_weights()
        source_names = list(weights.keys())
        source_probs = [weights[name] for name in source_names]

        samples = []
        for _ in range(num_samples):
            # 按权重选择数据源
            source_name = self._rng.choices(
                source_names, weights=source_probs, k=1
            )[0]
            source = self.sources[source_name]

            # 从该数据源随机选择一个文档
            doc = self._rng.choice(source.documents)
            samples.append((source_name, doc))

        return samples

    def sample_round_robin(
        self, num_samples: int
    ) -> List[Tuple[str, str]]:
        """
        轮询采样

        按权重比例确定各数据源的出现频率，交替采样。
        适合需要数据源均匀分布的场景。

        Args:
            num_samples: 总采样数

        Returns:
            [(数据源名称, 文档文本), ...] 列表
        """
        weights = self._normalize_weights()

        # 计算每个数据源的分配数量
        allocations = {}
        remaining = num_samples
        source_names = sorted(weights.keys())

        for i, name in enumerate(source_names):
            if i == len(source_names) - 1:
                allocations[name] = remaining
            else:
                alloc = round(weights[name] * num_samples)
                allocations[name] = alloc
                remaining -= alloc

        # 创建交错序列
        samples = []
        source_indices = {name: 0 for name in source_names}
        source_queues = {}

        for name in source_names:
            # 为每个数据源创建一个打乱的文档索引队列
            indices = list(range(len(self.sources[name].documents)))
            self._rng.shuffle(indices)
            source_queues[name] = indices

        # 按轮询顺序采样
        schedule = []
        for name, count in allocations.items():
            schedule.extend([name] * count)
        self._rng.shuffle(schedule)

        for source_name in schedule:
            source = self.sources[source_name]
            idx = source_indices[source_name] % len(source.documents)
            queue = source_queues[source_name]
            doc_idx = queue[idx % len(queue)]
            doc = source.documents[doc_idx]
            samples.append((source_name, doc))
            source_indices[source_name] += 1

        return samples

    def get_mixing_stats(self) -> Dict[str, Dict]:
        """
        获取数据混合统计信息

        Returns:
            各数据源的统计信息
        """
        weights = self._normalize_weights()

        stats = {}
        for name, source in self.sources.items():
            stats[name] = {
                'num_documents': source.size,
                'total_chars': source.total_chars,
                'avg_doc_length': (
                    source.total_chars / source.size if source.size > 0 else 0
                ),
                'weight': self.config.source_weights.get(name, 0),
                'normalized_weight': weights.get(name, 0),
                'quality_score': source.quality_score,
                'domain': source.domain,
            }

        return stats


class CurriculumScheduler:
    """
    课程学习调度器

    在训练过程中动态调整数据混合比例。

    核心思想: 模仿人类学习过程，先学简单的、再学难的。
    在预训练中，可以:
    - 训练早期: 使用更高比例的高质量、简单数据
    - 训练中期: 逐步增加多样性
    - 训练末期: 提高高质量数据比例（如 Llama 3 的 annealing）

    支持的调度策略:
    - linear: 线性插值
    - cosine: 余弦退火
    - step: 阶梯式调整

    Args:
        initial_weights: 初始权重
        final_weights: 最终权重
        total_steps: 总训练步数
        strategy: 调度策略
        warmup_steps: 预热步数（前 N 步使用初始权重不变）
    """

    def __init__(
        self,
        initial_weights: Dict[str, float],
        final_weights: Dict[str, float],
        total_steps: int,
        strategy: str = 'linear',
        warmup_steps: int = 0,
    ):
        assert set(initial_weights.keys()) == set(final_weights.keys()), \
            "初始权重和最终权重必须包含相同的数据源"
        assert strategy in ('linear', 'cosine', 'step')

        self.initial_weights = initial_weights
        self.final_weights = final_weights
        self.total_steps = total_steps
        self.strategy = strategy
        self.warmup_steps = warmup_steps

    def get_weights(self, step: int) -> Dict[str, float]:
        """
        获取当前步数对应的数据混合权重

        Args:
            step: 当前训练步数

        Returns:
            当前权重 {数据源名称: 权重}
        """
        if step <= self.warmup_steps:
            return dict(self.initial_weights)

        # 计算进度 [0, 1]
        effective_step = step - self.warmup_steps
        effective_total = self.total_steps - self.warmup_steps
        progress = min(1.0, effective_step / max(effective_total, 1))

        # 根据策略计算插值系数
        if self.strategy == 'linear':
            alpha = progress
        elif self.strategy == 'cosine':
            # 余弦退火: 缓慢开始 -> 加速 -> 缓慢结束
            alpha = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.strategy == 'step':
            # 每 25% 切换一次
            alpha = min(1.0, int(progress * 4) / 3)
        else:
            alpha = progress

        # 线性插值权重
        weights = {}
        for name in self.initial_weights:
            w_init = self.initial_weights[name]
            w_final = self.final_weights[name]
            weights[name] = w_init + alpha * (w_final - w_init)

        return weights

    def get_schedule_preview(
        self, num_points: int = 10
    ) -> List[Tuple[int, Dict[str, float]]]:
        """
        预览整个训练过程的权重变化

        Args:
            num_points: 采样点数

        Returns:
            [(步数, 权重字典), ...] 列表
        """
        schedule = []
        for i in range(num_points + 1):
            step = int(i / num_points * self.total_steps)
            weights = self.get_weights(step)
            schedule.append((step, weights))
        return schedule


class DomainBalancer:
    """
    领域平衡器

    当某些领域的数据量过少时，通过上采样（oversampling）
    使其在训练中获得更多的曝光机会。

    上采样策略:
    - proportional: 按自然比例（不做特殊处理）
    - equal: 所有领域等量
    - sqrt: 按数据量的平方根加权（折中方案）
    - temperature: 温度调节

    Args:
        temperature: 上采样温度
            - T=1: 按自然比例
            - T=无穷大: 所有领域等量
            - T 的效果: p_i(T) = n_i^{1/T} / sum(n_j^{1/T})
    """

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def compute_sampling_weights(
        self,
        domain_sizes: Dict[str, int],
    ) -> Dict[str, float]:
        """
        计算各领域的采样权重

        Args:
            domain_sizes: {领域名称: 数据量}

        Returns:
            {领域名称: 采样权重}（已归一化）
        """
        T = self.temperature

        weights = {}
        for domain, size in domain_sizes.items():
            if size > 0:
                weights[domain] = size ** (1.0 / T) if T > 0 else 1.0
            else:
                weights[domain] = 0.0

        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def compute_epochs_per_domain(
        self,
        domain_sizes: Dict[str, int],
        total_samples: int,
    ) -> Dict[str, float]:
        """
        计算每个领域的数据被使用的平均次数（epoch数）

        上采样意味着某些领域的数据会被重复使用多次。

        Args:
            domain_sizes: {领域名称: 数据量}
            total_samples: 总采样数

        Returns:
            {领域名称: 平均 epoch 数}
        """
        weights = self.compute_sampling_weights(domain_sizes)

        epochs = {}
        for domain, size in domain_sizes.items():
            if size > 0:
                samples_for_domain = weights[domain] * total_samples
                epochs[domain] = samples_for_domain / size
            else:
                epochs[domain] = 0.0

        return epochs


if __name__ == '__main__':
    print("=" * 60)
    print("数据混合与采样 演示")
    print("=" * 60)

    # 创建模拟数据源
    web_docs = [f"Web document {i} about various topics." for i in range(100)]
    book_docs = [f"Book excerpt {i} with detailed narrative." for i in range(30)]
    code_docs = [f"def function_{i}(): return {i}" for i in range(20)]
    wiki_docs = [f"Wikipedia article {i} about science." for i in range(15)]

    sources = [
        DataSource("web", web_docs, weight=6.0, quality_score=0.6, domain="web"),
        DataSource("books", book_docs, weight=2.0, quality_score=0.9, domain="books"),
        DataSource("code", code_docs, weight=1.0, quality_score=0.8, domain="code"),
        DataSource("wikipedia", wiki_docs, weight=1.0, quality_score=0.95, domain="wiki"),
    ]

    config = MixingConfig(
        source_weights={"web": 6.0, "books": 2.0, "code": 1.0, "wikipedia": 1.0},
        temperature=1.0,
        seed=42,
    )

    mixer = DataMixer(sources, config)

    # 1. 混合统计
    print("\n--- 数据混合统计 ---")
    stats = mixer.get_mixing_stats()
    for name, info in stats.items():
        print(f"  {name}: "
              f"docs={info['num_documents']}, "
              f"weight={info['normalized_weight']:.2%}, "
              f"quality={info['quality_score']:.2f}")

    # 2. 按比例采样
    print("\n--- 按比例采样 (100 samples) ---")
    samples = mixer.sample_proportional(100)
    source_counts = Counter(name for name, _ in samples)
    for name in sorted(source_counts.keys()):
        print(f"  {name}: {source_counts[name]} 条 ({source_counts[name]}%)")

    # 3. 温度调节效果
    print("\n--- 温度调节效果 ---")
    for temp in [0.5, 1.0, 2.0, 5.0]:
        config_t = MixingConfig(
            source_weights={"web": 6.0, "books": 2.0, "code": 1.0, "wikipedia": 1.0},
            temperature=temp,
        )
        mixer_t = DataMixer(sources, config_t)
        weights = mixer_t._normalize_weights()
        weight_str = ", ".join(
            f"{k}={v:.2%}" for k, v in sorted(weights.items())
        )
        print(f"  T={temp}: {weight_str}")

    # 4. 课程学习调度
    print("\n--- 课程学习调度 (linear) ---")
    scheduler = CurriculumScheduler(
        initial_weights={"web": 7.0, "books": 1.5, "code": 0.5, "wikipedia": 1.0},
        final_weights={"web": 4.0, "books": 2.5, "code": 2.0, "wikipedia": 1.5},
        total_steps=10000,
        strategy='linear',
        warmup_steps=1000,
    )

    schedule = scheduler.get_schedule_preview(num_points=5)
    for step, weights in schedule:
        weight_str = ", ".join(
            f"{k}={v:.2f}" for k, v in sorted(weights.items())
        )
        print(f"  Step {step:>6d}: {weight_str}")

    # 5. 领域平衡
    print("\n--- 领域平衡 ---")
    domain_sizes = {"web": 10000, "books": 500, "code": 300, "wiki": 200}
    total_training_samples = 20000

    for temp in [1.0, 2.0, 5.0]:
        balancer = DomainBalancer(temperature=temp)
        weights = balancer.compute_sampling_weights(domain_sizes)
        epochs = balancer.compute_epochs_per_domain(
            domain_sizes, total_training_samples
        )
        print(f"\n  T={temp}:")
        for domain in sorted(domain_sizes.keys()):
            print(f"    {domain}: weight={weights[domain]:.3f}, "
                  f"epochs={epochs[domain]:.1f}")
