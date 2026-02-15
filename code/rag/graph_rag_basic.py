"""
GraphRAG 基础实现：实体抽取 + NetworkX 建图 + 社区发现

本模块实现了简化版的 GraphRAG 流程，包括：
1. 基于规则/模拟的实体与关系抽取
2. 使用 NetworkX 构建知识图谱
3. 社区发现（使用 Louvain/Leiden 算法的简化实现）
4. 社区摘要生成（模拟 LLM 调用）
5. 基于图谱的问答（局部查询 + 全局查询）

GraphRAG 的核心动机：
- 传统向量 RAG 擅长局部事实问题，但难以回答全局归纳性问题
- GraphRAG 通过构建知识图谱和社区发现，实现跨文档的全局推理

参考:
    Edge et al. (2024). "From Local to Global: A Graph RAG Approach
    to Query-Focused Summarization"
"""

import re
from collections import defaultdict
from typing import Optional

try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("警告: 未安装 networkx, 部分功能不可用。请运行: pip install networkx")


class Entity:
    """
    知识图谱中的实体

    参数:
        name: 实体名称
        entity_type: 实体类型（如 "技术", "人物", "组织"）
        description: 实体描述
    """

    def __init__(self, name: str, entity_type: str, description: str = ""):
        self.name = name
        self.entity_type = entity_type
        self.description = description

    def __repr__(self):
        return f"Entity({self.name}, type={self.entity_type})"


class Relation:
    """
    知识图谱中的关系

    参数:
        source: 源实体名称
        target: 目标实体名称
        relation_type: 关系类型描述
        weight: 关系强度 (0-1)
    """

    def __init__(
        self, source: str, target: str, relation_type: str, weight: float = 1.0
    ):
        self.source = source
        self.target = target
        self.relation_type = relation_type
        self.weight = weight

    def __repr__(self):
        return f"Relation({self.source} --[{self.relation_type}]--> {self.target})"


class SimpleEntityExtractor:
    """
    简化版实体关系抽取器

    在实际 GraphRAG 系统中，此步骤使用 LLM 进行实体与关系抽取。
    本实现使用基于规则的模拟方法，用于演示 GraphRAG 的整体流程。

    实际系统的 Prompt 设计示例:
        "你是一个信息抽取专家。请从以下文本中抽取实体和关系。
         实体格式: (实体名, 实体类型, 实体描述)
         关系格式: (实体1, 实体2, 关系描述, 关系强度)"
    """

    def __init__(self, entity_patterns: Optional[dict] = None):
        """
        参数:
            entity_patterns: {实体名: 实体类型} 的预定义映射
        """
        # 预定义的实体模式（在实际系统中由 LLM 动态抽取）
        self.entity_patterns = entity_patterns or {
            "Transformer": "技术",
            "Self-Attention": "技术",
            "BERT": "模型",
            "GPT": "模型",
            "GPT-2": "模型",
            "GPT-3": "模型",
            "GPT-4": "模型",
            "Llama": "模型",
            "DeepSeek": "组织",
            "DeepSeek-V3": "模型",
            "Google": "组织",
            "Anthropic": "组织",
            "Meta": "组织",
            "OpenAI": "组织",
            "MoE": "技术",
            "RoPE": "技术",
            "SwiGLU": "技术",
            "GQA": "技术",
            "RMSNorm": "技术",
            "RLHF": "技术",
            "DPO": "技术",
            "SFT": "技术",
            "RAG": "技术",
            "HNSW": "技术",
            "BM25": "技术",
            "KV Cache": "技术",
            "Flash Attention": "技术",
            "Vaswani": "人物",
        }

    def extract(self, text: str) -> tuple[list[Entity], list[Relation]]:
        """
        从文本中抽取实体和关系

        参数:
            text: 输入文本

        返回:
            (entities, relations) 抽取到的实体和关系列表
        """
        entities = []
        relations = []
        found_entities = set()

        # 实体抽取：基于模式匹配
        for entity_name, entity_type in self.entity_patterns.items():
            if entity_name.lower() in text.lower():
                entity = Entity(
                    name=entity_name,
                    entity_type=entity_type,
                    description=f"在文本中出现的{entity_type}: {entity_name}",
                )
                entities.append(entity)
                found_entities.add(entity_name)

        # 关系抽取：基于共现（同一文本中出现的实体视为相关）
        found_list = list(found_entities)
        for i in range(len(found_list)):
            for j in range(i + 1, len(found_list)):
                # 计算共现强度（简化：基于文本中的位置距离）
                e1, e2 = found_list[i], found_list[j]
                pos1 = text.lower().find(e1.lower())
                pos2 = text.lower().find(e2.lower())
                distance = abs(pos1 - pos2)
                # 距离越近，关系越强
                weight = max(0.1, 1.0 - distance / len(text))

                relation = Relation(
                    source=e1,
                    target=e2,
                    relation_type="相关",
                    weight=weight,
                )
                relations.append(relation)

        return entities, relations


class SimpleCommunityDetector:
    """
    简化版社区发现算法

    实现了 Louvain 算法的简化版本，用于在知识图谱上发现社区结构。

    Louvain 算法的核心思想:
    1. 初始化: 每个节点自成一个社区
    2. 局部移动: 尝试将每个节点移动到使模块度增益最大的邻居社区
    3. 聚合: 将社区缩为超级节点，构建新图
    4. 重复 2-3 直到收敛

    模块度公式:
        Q = (1/2m) * sum_ij [A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)
    """

    def detect_communities(
        self, graph: "nx.Graph", resolution: float = 1.0
    ) -> dict[str, int]:
        """
        在图上执行社区发现

        参数:
            graph: NetworkX 图
            resolution: 分辨率参数（越大社区越细）

        返回:
            {节点名: 社区ID} 映射
        """
        if not HAS_NETWORKX:
            raise ImportError("需要安装 networkx")

        if len(graph.nodes()) == 0:
            return {}

        # 初始化: 每个节点自成一个社区
        node_to_community = {node: i for i, node in enumerate(graph.nodes())}
        nodes = list(graph.nodes())

        # 迭代优化
        improved = True
        max_iterations = 50

        for iteration in range(max_iterations):
            if not improved:
                break
            improved = False

            for node in nodes:
                current_community = node_to_community[node]

                # 找到邻居所在的社区
                neighbor_communities = set()
                for neighbor in graph.neighbors(node):
                    neighbor_communities.add(node_to_community[neighbor])

                if not neighbor_communities:
                    continue

                # 计算移动到每个邻居社区的模块度增益
                best_community = current_community
                best_gain = 0.0

                for target_community in neighbor_communities:
                    if target_community == current_community:
                        continue

                    gain = self._modularity_gain(
                        graph, node, current_community, target_community,
                        node_to_community, resolution,
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_community = target_community

                # 如果有正增益，移动节点
                if best_community != current_community:
                    node_to_community[node] = best_community
                    improved = True

        # 重新编号社区（从 0 开始连续编号）
        unique_communities = sorted(set(node_to_community.values()))
        community_remap = {old: new for new, old in enumerate(unique_communities)}
        return {
            node: community_remap[comm]
            for node, comm in node_to_community.items()
        }

    def _modularity_gain(
        self,
        graph: "nx.Graph",
        node: str,
        current_community: int,
        target_community: int,
        node_to_community: dict,
        resolution: float,
    ) -> float:
        """
        计算将节点从当前社区移动到目标社区的模块度增益

        参数:
            graph: NetworkX 图
            node: 待移动的节点
            current_community: 当前社区 ID
            target_community: 目标社区 ID
            node_to_community: 当前的节点-社区映射
            resolution: 分辨率参数

        返回:
            模块度增益值
        """
        m = graph.number_of_edges()
        if m == 0:
            return 0.0

        ki = graph.degree(node, weight="weight")
        if ki is None:
            ki = graph.degree(node)

        # 计算节点到目标社区的边权和
        ki_target = 0.0
        ki_current = 0.0
        sigma_target = 0.0
        sigma_current = 0.0

        for neighbor in graph.neighbors(node):
            edge_weight = graph[node][neighbor].get("weight", 1.0)
            neighbor_comm = node_to_community[neighbor]

            if neighbor_comm == target_community:
                ki_target += edge_weight
            if neighbor_comm == current_community and neighbor != node:
                ki_current += edge_weight

        # 计算目标社区和当前社区的度数总和
        for n in graph.nodes():
            n_degree = graph.degree(n, weight="weight")
            if n_degree is None:
                n_degree = graph.degree(n)
            if node_to_community[n] == target_community:
                sigma_target += n_degree
            if node_to_community[n] == current_community and n != node:
                sigma_current += n_degree

        # 模块度增益 = 移入增益 - 移出损失
        gain_in = ki_target - resolution * sigma_target * ki / (2 * m)
        gain_out = ki_current - resolution * sigma_current * ki / (2 * m)

        return (gain_in - gain_out) / (2 * m)


class GraphRAG:
    """
    GraphRAG 系统

    整合实体抽取、图谱构建、社区发现、摘要生成和问答功能。

    参数:
        extractor: 实体关系抽取器
        community_detector: 社区发现算法
    """

    def __init__(
        self,
        extractor: Optional[SimpleEntityExtractor] = None,
        community_detector: Optional[SimpleCommunityDetector] = None,
    ):
        if not HAS_NETWORKX:
            raise ImportError("GraphRAG 需要 networkx 库。请运行: pip install networkx")

        self.extractor = extractor or SimpleEntityExtractor()
        self.community_detector = community_detector or SimpleCommunityDetector()

        self.graph: Optional[nx.Graph] = None
        self.entities: dict[str, Entity] = {}
        self.communities: dict[str, int] = {}
        self.community_summaries: dict[int, str] = {}

    def build_index(self, documents: list[str]):
        """
        从文档集合构建 GraphRAG 索引

        步骤:
        1. 实体和关系抽取
        2. 构建知识图谱
        3. 社区发现
        4. 社区摘要生成

        参数:
            documents: 文档列表
        """
        print("GraphRAG 索引构建开始...")

        # 步骤 1: 实体和关系抽取
        print("  [1/4] 实体和关系抽取...")
        all_entities = {}
        all_relations = []

        for i, doc in enumerate(documents):
            entities, relations = self.extractor.extract(doc)
            for e in entities:
                if e.name not in all_entities:
                    all_entities[e.name] = e
            all_relations.extend(relations)

        self.entities = all_entities
        print(f"    抽取到 {len(all_entities)} 个实体, {len(all_relations)} 条关系")

        # 步骤 2: 构建知识图谱
        print("  [2/4] 构建知识图谱...")
        self.graph = nx.Graph()

        for name, entity in all_entities.items():
            self.graph.add_node(
                name,
                entity_type=entity.entity_type,
                description=entity.description,
            )

        for relation in all_relations:
            if relation.source in all_entities and relation.target in all_entities:
                if self.graph.has_edge(relation.source, relation.target):
                    # 累加权重
                    current_weight = self.graph[relation.source][relation.target].get(
                        "weight", 0
                    )
                    self.graph[relation.source][relation.target][
                        "weight"
                    ] = current_weight + relation.weight
                else:
                    self.graph.add_edge(
                        relation.source,
                        relation.target,
                        relation=relation.relation_type,
                        weight=relation.weight,
                    )

        print(f"    图谱: {self.graph.number_of_nodes()} 节点, "
              f"{self.graph.number_of_edges()} 条边")

        # 步骤 3: 社区发现
        print("  [3/4] 社区发现...")
        if self.graph.number_of_nodes() > 0:
            self.communities = self.community_detector.detect_communities(self.graph)
            num_communities = len(set(self.communities.values()))
            print(f"    发现 {num_communities} 个社区")
        else:
            self.communities = {}

        # 步骤 4: 社区摘要
        print("  [4/4] 生成社区摘要...")
        self._generate_community_summaries()

        print("GraphRAG 索引构建完成!")

    def _generate_community_summaries(self):
        """
        为每个社区生成摘要

        在实际系统中，此步骤调用 LLM 生成摘要。
        本实现使用简单的基于实体和关系信息的模板生成。
        """
        if not self.communities:
            return

        # 按社区分组
        community_nodes: dict[int, list[str]] = defaultdict(list)
        for node, comm_id in self.communities.items():
            community_nodes[comm_id].append(node)

        for comm_id, nodes in community_nodes.items():
            # 获取社区内的实体信息
            entity_types = []
            for node in nodes:
                if node in self.entities:
                    entity_types.append(
                        f"{node}({self.entities[node].entity_type})"
                    )

            # 获取社区内的关系
            relations_desc = []
            for i, n1 in enumerate(nodes):
                for n2 in nodes[i + 1 :]:
                    if self.graph.has_edge(n1, n2):
                        rel = self.graph[n1][n2].get("relation", "相关")
                        relations_desc.append(f"{n1} -- {rel} -- {n2}")

            # 生成摘要（模板方式，实际应用中用 LLM 生成）
            summary_parts = [
                f"社区 {comm_id} 包含 {len(nodes)} 个实体:",
                f"实体: {', '.join(entity_types)}",
            ]
            if relations_desc:
                summary_parts.append(f"关键关系: {'; '.join(relations_desc[:5])}")

            self.community_summaries[comm_id] = "\n".join(summary_parts)

    def query_local(self, question: str, top_k: int = 5) -> str:
        """
        局部查询：基于实体和关系检索

        适用于具体的事实性问题。

        参数:
            question: 用户问题
            top_k: 返回的相关实体数量

        返回:
            基于图谱信息的回答
        """
        if self.graph is None:
            return "索引未构建，请先调用 build_index()"

        # 简单的实体匹配（实际系统中使用向量相似度或 LLM）
        matched_entities = []
        question_lower = question.lower()
        for entity_name in self.entities:
            if entity_name.lower() in question_lower:
                matched_entities.append(entity_name)

        if not matched_entities:
            # 如果没有精确匹配，返回所有实体中最相关的
            return "未找到与问题直接相关的实体。请尝试更具体的问题。"

        # 获取匹配实体的上下文（邻居信息）
        context_parts = []
        for entity_name in matched_entities[:top_k]:
            entity = self.entities.get(entity_name)
            if entity:
                context_parts.append(
                    f"- {entity.name} ({entity.entity_type}): {entity.description}"
                )

            # 获取邻居
            if entity_name in self.graph:
                for neighbor in self.graph.neighbors(entity_name):
                    rel = self.graph[entity_name][neighbor].get("relation", "相关")
                    context_parts.append(
                        f"  -> {rel}: {neighbor} ({self.entities.get(neighbor, Entity(neighbor, '未知')).entity_type})"
                    )

        return f"基于图谱的局部检索结果:\n" + "\n".join(context_parts)

    def query_global(self, question: str) -> str:
        """
        全局查询：基于社区摘要的 Map-Reduce

        适用于需要跨文档归纳的全局问题。

        步骤:
        1. Map: 对每个社区摘要生成部分回答
        2. Reduce: 合成最终回答

        参数:
            question: 用户问题

        返回:
            基于社区摘要的综合回答
        """
        if not self.community_summaries:
            return "社区摘要未生成，请先调用 build_index()"

        # Map 阶段: 从每个社区摘要中提取相关信息
        partial_answers = []
        for comm_id, summary in self.community_summaries.items():
            # 检查社区摘要与问题的相关性（简化: 关键词匹配）
            question_terms = set(question.lower().split())
            summary_terms = set(summary.lower().split())
            overlap = question_terms & summary_terms

            if overlap:
                partial_answers.append(
                    f"[社区 {comm_id}] {summary}"
                )

        if not partial_answers:
            # 如果没有关键词匹配，返回所有社区摘要
            partial_answers = [
                f"[社区 {comm_id}] {summary}"
                for comm_id, summary in self.community_summaries.items()
            ]

        # Reduce 阶段: 合成最终回答
        result = f"基于 {len(partial_answers)} 个社区的全局分析:\n\n"
        result += "\n\n".join(partial_answers)

        return result

    def get_graph_stats(self) -> dict:
        """获取图谱统计信息"""
        if self.graph is None:
            return {}

        stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_communities": len(set(self.communities.values())) if self.communities else 0,
            "entity_types": defaultdict(int),
        }

        for entity in self.entities.values():
            stats["entity_types"][entity.entity_type] += 1

        stats["entity_types"] = dict(stats["entity_types"])
        return stats


if __name__ == "__main__":
    if not HAS_NETWORKX:
        print("请先安装 networkx: pip install networkx")
        exit(1)

    print("=" * 60)
    print("GraphRAG 基础实现演示")
    print("=" * 60)

    # 1. 准备文档
    print("\n--- 1. 准备示例文档 ---")
    documents = [
        "Transformer 是由 Vaswani 等人提出的深度学习架构，使用 Self-Attention 机制。"
        "GPT 和 BERT 都基于 Transformer 架构构建。",

        "GPT-2 和 GPT-3 是 OpenAI 开发的自回归语言模型，基于 Transformer 的 Decoder 结构。"
        "GPT-4 进一步扩展了模型能力，据推测使用了 MoE 架构。",

        "Llama 是 Meta 开发的开源大语言模型，使用了 RoPE 位置编码、SwiGLU 激活函数和 GQA 注意力机制。"
        "Llama 的架构改进包括 RMSNorm 归一化。",

        "DeepSeek-V3 是 DeepSeek 开发的大语言模型，采用 MoE 架构。"
        "DeepSeek 在模型效率方面进行了大量创新。",

        "RAG 技术结合了检索和生成，使用 BM25 或 HNSW 等算法进行文档检索。"
        "HNSW 是一种高效的近似最近邻搜索算法，广泛用于向量数据库。",

        "Anthropic 是一家 AI 安全公司，开发了 Claude 系列模型。"
        "Anthropic 在 RLHF 和 DPO 等对齐技术方面有深入研究。",

        "SFT 是有监督微调的缩写，RLHF 和 DPO 是常用的对齐训练方法。"
        "这些技术结合使用可以训练出安全且有用的大语言模型。",

        "KV Cache 和 Flash Attention 是推理优化的关键技术。"
        "它们可以显著降低 Transformer 模型的推理延迟和显存占用。",
    ]
    print(f"文档数量: {len(documents)}")

    # 2. 构建 GraphRAG 索引
    print("\n--- 2. 构建 GraphRAG 索引 ---")
    graph_rag = GraphRAG()
    graph_rag.build_index(documents)

    # 3. 查看图谱统计
    print("\n--- 3. 图谱统计 ---")
    stats = graph_rag.get_graph_stats()
    print(f"  节点数: {stats['num_nodes']}")
    print(f"  边数: {stats['num_edges']}")
    print(f"  社区数: {stats['num_communities']}")
    print(f"  实体类型分布: {stats['entity_types']}")

    # 4. 查看社区结构
    print("\n--- 4. 社区结构 ---")
    community_nodes = defaultdict(list)
    for node, comm_id in graph_rag.communities.items():
        community_nodes[comm_id].append(node)

    for comm_id, nodes in sorted(community_nodes.items()):
        print(f"  社区 {comm_id}: {', '.join(nodes)}")

    # 5. 社区摘要
    print("\n--- 5. 社区摘要 ---")
    for comm_id, summary in graph_rag.community_summaries.items():
        print(f"\n  === 社区 {comm_id} ===")
        for line in summary.split("\n"):
            print(f"  {line}")

    # 6. 局部查询
    print("\n--- 6. 局部查询演示 ---")
    local_questions = [
        "Transformer 使用了什么技术?",
        "DeepSeek-V3 的架构是什么?",
        "HNSW 是什么算法?",
    ]

    for q in local_questions:
        print(f"\n问题: {q}")
        answer = graph_rag.query_local(q)
        print(answer)

    # 7. 全局查询
    print("\n--- 7. 全局查询演示 ---")
    global_questions = [
        "有哪些主要的大语言模型?",
        "AI 领域有哪些关键技术?",
    ]

    for q in global_questions:
        print(f"\n问题: {q}")
        answer = graph_rag.query_global(q)
        print(answer)

    # 8. 图谱可视化信息
    print("\n--- 8. 图谱连接信息 ---")
    print("度数最高的实体 (连接最多):")
    degree_list = sorted(
        graph_rag.graph.degree(), key=lambda x: x[1], reverse=True
    )
    for node, degree in degree_list[:5]:
        entity_type = graph_rag.entities.get(node, Entity(node, "未知")).entity_type
        print(f"  {node} ({entity_type}): 度数 = {degree}")

    print("\n演示完成!")
