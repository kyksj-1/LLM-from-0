"""
Tree of Thoughts (ToT) 搜索框架

本模块实现了 Tree of Thoughts 的核心搜索算法，包括 BFS 和 DFS
两种搜索策略，以及基于 LLM 的评估函数。

核心思想:
- 将推理过程建模为搜索树
- 每个节点是一个"思维状态"（已有的推理步骤）
- 使用 BFS 或 DFS 遍历搜索树
- 用评估函数剪枝，选择最有前途的路径

参考:
- Yao et al. (2023). Tree of Thoughts: Deliberate Problem Solving with Large Language Models.
"""

from typing import List, Dict, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import copy


class SearchStrategy(Enum):
    """搜索策略枚举"""
    BFS = "bfs"
    DFS = "dfs"


@dataclass
class ThoughtNode:
    """
    思维节点

    搜索树中的一个节点，表示当前的推理状态。

    Attributes:
        state: 当前推理状态（包含问题和已有的推理步骤）
        thought: 当前节点的思维内容
        score: 评估分数（0-1）
        depth: 节点深度（根节点为0）
        children: 子节点列表
        parent: 父节点
        is_terminal: 是否为终止节点（已得出答案）
    """
    state: str
    thought: str = ""
    score: float = 0.0
    depth: int = 0
    children: List["ThoughtNode"] = field(default_factory=list)
    parent: Optional["ThoughtNode"] = None
    is_terminal: bool = False

    def get_path(self) -> List[str]:
        """
        获取从根节点到当前节点的完整推理路径

        Returns:
            推理步骤列表
        """
        path = []
        node = self
        while node is not None:
            if node.thought:
                path.append(node.thought)
            node = node.parent
        return list(reversed(path))

    def get_full_reasoning(self) -> str:
        """
        获取完整的推理文本

        Returns:
            从根到当前节点的推理过程
        """
        return "\n".join(f"Step {i+1}: {step}" for i, step in enumerate(self.get_path()))


class ThoughtGenerator:
    """
    思维生成器

    负责为给定状态生成候选的下一步思维。
    实际使用时应调用 LLM API。
    """

    def __init__(self, generator_fn: Optional[Callable] = None, k: int = 3):
        """
        初始化思维生成器

        Args:
            generator_fn: 自定义生成函数 (state, k) -> List[str]
            k: 每个节点生成的候选思维数量
        """
        self.generator_fn = generator_fn or self._default_generator
        self.k = k

    def _default_generator(self, state: str, k: int) -> List[str]:
        """
        默认生成器（模拟）

        Args:
            state: 当前推理状态
            k: 生成候选数量

        Returns:
            候选思维列表
        """
        return [f"[思维候选 {i+1} for state: {state[:50]}...]" for i in range(k)]

    def generate(self, state: str, k: Optional[int] = None) -> List[str]:
        """
        生成候选思维

        Args:
            state: 当前推理状态
            k: 候选数量（None 使用默认值）

        Returns:
            候选思维列表
        """
        return self.generator_fn(state, k or self.k)


class ThoughtEvaluator:
    """
    思维评估器

    负责评估当前推理状态的"前景"——即沿着这条路径继续推理
    是否有可能到达正确答案。

    评估方式:
    1. 值评估: 让 LLM 对状态打分 (0-1)
    2. 投票评估: 让 LLM 多次判断 "好/坏"，取比例作为分数
    """

    def __init__(self, evaluator_fn: Optional[Callable] = None, method: str = "value"):
        """
        初始化评估器

        Args:
            evaluator_fn: 自定义评估函数 (state) -> float
            method: 评估方法 ("value" 或 "vote")
        """
        self.evaluator_fn = evaluator_fn or self._default_evaluator
        self.method = method

    def _default_evaluator(self, state: str) -> float:
        """
        默认评估器（模拟）

        Args:
            state: 当前推理状态

        Returns:
            评估分数 (0-1)
        """
        # 模拟：随机分数，稍微偏向好的
        return random.uniform(0.3, 0.9)

    def evaluate(self, state: str) -> float:
        """
        评估推理状态

        Args:
            state: 当前推理状态文本

        Returns:
            评估分数 (0-1)
        """
        return self.evaluator_fn(state)


class TreeOfThoughts:
    """
    Tree of Thoughts 搜索引擎

    实现 BFS 和 DFS 两种搜索策略：
    - BFS: 逐层展开，每层保留 top-k 节点
    - DFS: 深度优先探索，低分节点被剪枝
    """

    def __init__(
        self,
        generator: Optional[ThoughtGenerator] = None,
        evaluator: Optional[ThoughtEvaluator] = None,
        max_depth: int = 3,
        breadth_limit: int = 5,
        pruning_threshold: float = 0.3,
    ):
        """
        初始化 ToT 搜索引擎

        Args:
            generator: 思维生成器
            evaluator: 思维评估器
            max_depth: 最大搜索深度
            breadth_limit: BFS 每层保留的最大节点数
            pruning_threshold: DFS 的剪枝阈值（低于此分数的节点不展开）
        """
        self.generator = generator or ThoughtGenerator()
        self.evaluator = evaluator or ThoughtEvaluator()
        self.max_depth = max_depth
        self.breadth_limit = breadth_limit
        self.pruning_threshold = pruning_threshold

        # 搜索统计
        self.nodes_explored = 0
        self.nodes_pruned = 0

    def _create_child(self, parent: ThoughtNode, thought: str) -> ThoughtNode:
        """
        创建子节点

        Args:
            parent: 父节点
            thought: 思维内容

        Returns:
            新创建的子节点
        """
        child_state = f"{parent.state}\n{thought}"
        child = ThoughtNode(
            state=child_state,
            thought=thought,
            depth=parent.depth + 1,
            parent=parent,
        )
        # 评估子节点
        child.score = self.evaluator.evaluate(child_state)
        parent.children.append(child)
        self.nodes_explored += 1
        return child

    def bfs_search(self, problem: str) -> ThoughtNode:
        """
        BFS 搜索策略

        逐层展开搜索树，每层保留分数最高的 breadth_limit 个节点。

        Args:
            problem: 问题描述

        Returns:
            搜索完成后分数最高的叶节点
        """
        self.nodes_explored = 0
        self.nodes_pruned = 0

        # 初始化根节点
        root = ThoughtNode(state=problem, depth=0)
        root.score = 1.0  # 根节点分数为1

        current_layer = [root]

        for depth in range(self.max_depth):
            candidates = []

            for node in current_layer:
                # 为每个节点生成候选思维
                thoughts = self.generator.generate(node.state)

                for thought in thoughts:
                    child = self._create_child(node, thought)
                    candidates.append(child)

            if not candidates:
                break

            # 按分数排序，保留 top-k
            candidates.sort(key=lambda n: n.score, reverse=True)
            current_layer = candidates[:self.breadth_limit]

            # 统计剪枝数量
            self.nodes_pruned += len(candidates) - len(current_layer)

        # 返回最优节点
        if current_layer:
            best = max(current_layer, key=lambda n: n.score)
            return best

        return root

    def dfs_search(self, problem: str) -> ThoughtNode:
        """
        DFS 搜索策略

        深度优先遍历搜索树，低于阈值的节点被剪枝。

        Args:
            problem: 问题描述

        Returns:
            搜索到的最优叶节点
        """
        self.nodes_explored = 0
        self.nodes_pruned = 0

        root = ThoughtNode(state=problem, depth=0)
        root.score = 1.0

        best_node = root
        stack = [root]

        while stack:
            node = stack.pop()

            # 到达最大深度，检查是否是最优
            if node.depth >= self.max_depth:
                if node.score > best_node.score:
                    best_node = node
                continue

            # 生成候选思维
            thoughts = self.generator.generate(node.state)

            for thought in thoughts:
                child = self._create_child(node, thought)

                # 剪枝：分数低于阈值的节点不继续探索
                if child.score >= self.pruning_threshold:
                    stack.append(child)
                else:
                    self.nodes_pruned += 1

        return best_node

    def search(self, problem: str, strategy: SearchStrategy = SearchStrategy.BFS) -> Dict:
        """
        执行搜索

        Args:
            problem: 问题描述
            strategy: 搜索策略 (BFS 或 DFS)

        Returns:
            搜索结果字典
        """
        if strategy == SearchStrategy.BFS:
            best_node = self.bfs_search(problem)
        else:
            best_node = self.dfs_search(problem)

        return {
            "strategy": strategy.value,
            "best_score": best_node.score,
            "best_path": best_node.get_path(),
            "full_reasoning": best_node.get_full_reasoning(),
            "depth_reached": best_node.depth,
            "nodes_explored": self.nodes_explored,
            "nodes_pruned": self.nodes_pruned,
        }


def demonstrate_24_game():
    """
    在 24 点游戏上演示 ToT

    24 点游戏：给定 4 个数字，通过 +、-、*、/ 运算得到 24。
    这是 ToT 论文中使用的经典测试场景。
    """
    print("=" * 70)
    print("Tree of Thoughts: 24 点游戏演示")
    print("=" * 70)

    # 定义 24 点问题
    numbers = [4, 7, 8, 3]
    target = 24
    print(f"\n数字: {numbers}, 目标: {target}")

    # 模拟思维生成器：为 24 点问题生成候选运算步骤
    def game24_generator(state: str, k: int) -> List[str]:
        """为 24 点问题生成候选思维步骤"""
        # 简化模拟：生成几种可能的运算步骤
        candidates = [
            "尝试 8 - 4 = 4, 剩余 [4, 7, 3]",
            "尝试 7 + 3 = 10, 剩余 [4, 8, 10]",
            "尝试 8 * 3 = 24... 但还有 4 和 7 未使用",
            "尝试 4 * 7 = 28, 剩余 [28, 8, 3]",
            "尝试 7 - 3 = 4, 剩余 [4, 4, 8]",
        ]
        random.shuffle(candidates)
        return candidates[:k]

    # 模拟评估器：评估当前状态是否有希望到达 24
    def game24_evaluator(state: str) -> float:
        """评估 24 点游戏的推理状态"""
        # 简化模拟：包含 24 的路径得分更高
        if "= 24" in state:
            return 0.95
        if "剩余" in state and len(state) > 100:
            return 0.6  # 中间状态
        return random.uniform(0.3, 0.7)

    # 创建搜索引擎
    generator = ThoughtGenerator(game24_generator, k=3)
    evaluator = ThoughtEvaluator(game24_evaluator)

    tot = TreeOfThoughts(
        generator=generator,
        evaluator=evaluator,
        max_depth=3,
        breadth_limit=3,
        pruning_threshold=0.2,
    )

    # BFS 搜索
    print("\n--- BFS 搜索 ---")
    bfs_result = tot.search(f"24 点问题: 数字 {numbers}", SearchStrategy.BFS)
    print(f"搜索策略: {bfs_result['strategy']}")
    print(f"最优分数: {bfs_result['best_score']:.3f}")
    print(f"搜索深度: {bfs_result['depth_reached']}")
    print(f"探索节点: {bfs_result['nodes_explored']}")
    print(f"剪枝节点: {bfs_result['nodes_pruned']}")
    print(f"推理路径:")
    for i, step in enumerate(bfs_result["best_path"]):
        print(f"  Step {i+1}: {step}")

    # DFS 搜索
    print("\n--- DFS 搜索 ---")
    dfs_result = tot.search(f"24 点问题: 数字 {numbers}", SearchStrategy.DFS)
    print(f"搜索策略: {dfs_result['strategy']}")
    print(f"最优分数: {dfs_result['best_score']:.3f}")
    print(f"搜索深度: {dfs_result['depth_reached']}")
    print(f"探索节点: {dfs_result['nodes_explored']}")
    print(f"剪枝节点: {dfs_result['nodes_pruned']}")
    print(f"推理路径:")
    for i, step in enumerate(dfs_result["best_path"]):
        print(f"  Step {i+1}: {step}")


def demonstrate_search_comparison():
    """
    对比 BFS 和 DFS 的搜索行为

    展示不同搜索策略在节点探索数、剪枝行为等方面的差异。
    """
    print("\n" + "=" * 70)
    print("BFS vs DFS 搜索策略对比")
    print("=" * 70)

    random.seed(123)

    # 使用默认的模拟生成器和评估器
    configs = [
        {"max_depth": 2, "breadth_limit": 3, "pruning_threshold": 0.3},
        {"max_depth": 3, "breadth_limit": 5, "pruning_threshold": 0.3},
        {"max_depth": 4, "breadth_limit": 3, "pruning_threshold": 0.5},
    ]

    print(f"\n{'配置':>15} | {'策略':>5} | {'探索':>6} | {'剪枝':>6} | {'深度':>4} | {'分数':>6}")
    print("-" * 65)

    for config in configs:
        config_str = f"d={config['max_depth']},b={config['breadth_limit']}"

        for strategy in [SearchStrategy.BFS, SearchStrategy.DFS]:
            tot = TreeOfThoughts(**config)
            result = tot.search("测试问题", strategy)

            print(f"{config_str:>15} | {strategy.value:>5} | "
                  f"{result['nodes_explored']:>6} | {result['nodes_pruned']:>6} | "
                  f"{result['depth_reached']:>4} | {result['best_score']:>6.3f}")


def demonstrate_tot_structure():
    """演示搜索树的结构"""
    print("\n" + "=" * 70)
    print("搜索树结构演示")
    print("=" * 70)

    # 构建一个简单的搜索树来展示结构
    root = ThoughtNode(state="问题: 如何提高代码质量?", depth=0, score=1.0)

    # 第一层
    ideas = [
        ("引入代码审查流程", 0.8),
        ("增加单元测试覆盖率", 0.9),
        ("使用静态分析工具", 0.7),
    ]

    for thought, score in ideas:
        child = ThoughtNode(
            state=f"{root.state}\n{thought}",
            thought=thought,
            score=score,
            depth=1,
            parent=root,
        )
        root.children.append(child)

    # 第二层（只展开分数最高的节点）
    best_child = max(root.children, key=lambda n: n.score)
    sub_ideas = [
        ("选择 pytest 作为测试框架", 0.85),
        ("制定测试覆盖率目标 > 80%", 0.75),
    ]

    for thought, score in sub_ideas:
        grandchild = ThoughtNode(
            state=f"{best_child.state}\n{thought}",
            thought=thought,
            score=score,
            depth=2,
            parent=best_child,
        )
        best_child.children.append(grandchild)

    # 打印树结构
    print("\n搜索树:")
    _print_tree(root, prefix="")


def _print_tree(node: ThoughtNode, prefix: str = "", is_last: bool = True):
    """递归打印搜索树"""
    connector = "+-- " if is_last else "|-- "
    if node.depth == 0:
        print(f"[ROOT] score={node.score:.2f}")
    else:
        print(f"{prefix}{connector}[D{node.depth}] {node.thought} (score={node.score:.2f})")

    new_prefix = prefix + ("    " if is_last else "|   ")
    for i, child in enumerate(node.children):
        _print_tree(child, new_prefix, i == len(node.children) - 1)


if __name__ == "__main__":
    # 演示1: 24 点游戏
    demonstrate_24_game()

    # 演示2: BFS vs DFS 对比
    demonstrate_search_comparison()

    # 演示3: 搜索树结构
    demonstrate_tot_structure()
