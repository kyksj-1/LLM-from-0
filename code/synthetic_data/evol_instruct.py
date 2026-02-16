"""
Evol-Instruct: 指令进化数据生成管线

本模块实现了 WizardLM 论文中提出的 Evol-Instruct 流程，通过深度进化和广度进化
两种策略，将简单的种子指令逐步升级为复杂、多样的指令数据集。

核心流程:
1. 从种子指令集出发
2. 随机选择深度进化或广度进化策略
3. 用 LLM 执行进化操作
4. 过滤低质量/重复指令
5. 迭代扩展指令池

依赖:
- openai: OpenAI 兼容 API 客户端（支持 vLLM/Ollama/OpenAI）
- rouge_score: ROUGE-L 去重

安装:
    pip install openai rouge-score

参考:
- Xu et al. (2023). WizardLM: Empowering Large Language Models to Follow Complex Instructions.
"""

import json
import random
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
from pathlib import Path

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 进化 Prompt 模板
# ============================================================

# 深度进化：增加指令的复杂度和难度
DEPTH_EVOLUTION_PROMPTS = {
    "add_constraints": """请将以下指令改写为更复杂的版本，通过增加约束条件来提高难度。

要求：
1. 增加 1-3 个具体的约束条件
2. 约束条件应合理且不矛盾
3. 改写后的指令仍然可以被人类理解和完成

原始指令: {instruction}

改写后的指令:""",

    "deepen_reasoning": """请将以下指令改写为需要更深层推理的版本。

要求：
1. 要求多步推理或分析
2. 加入"为什么"或"如何"的追问
3. 改写后的指令不能通过简单查找回答

原始指令: {instruction}

改写后的指令:""",

    "concretize": """请将以下指令改写为更具体的版本，替换通用概念为特定领域概念。

要求：
1. 将通用词汇替换为专业领域术语
2. 加入具体的数字、场景或案例
3. 保持指令的可回答性

原始指令: {instruction}

改写后的指令:""",

    "add_nesting": """请将以下指令改写为包含嵌套条件的更复杂版本。

要求：
1. 增加条件判断（如"如果...则...否则..."）
2. 或增加多个子任务的组合
3. 改写后的指令逻辑应清晰、无歧义

原始指令: {instruction}

改写后的指令:""",

    "rare_concepts": """请将以下指令改写为涉及更低频/更专业概念的版本。

要求：
1. 引入该领域中更高级或更冷门的知识点
2. 保持指令的可回答性（有明确答案或解决方案）
3. 不要简单替换关键词，而是整体提升专业度

原始指令: {instruction}

改写后的指令:""",
}

# 广度进化：保持难度，改变领域/主题
BREADTH_EVOLUTION_PROMPT = """请基于以下指令的复杂度和风格，生成一条全新的指令。

要求：
1. 新指令应与原指令在不同的主题领域
2. 复杂度和长度应与原指令相当
3. 新指令必须合理且可回答
4. 不要简单改写原指令，而是创造全新的内容

原始指令: {instruction}

新指令:"""

# 回答生成模板
ANSWER_GENERATION_PROMPT = """请对以下指令给出高质量的回答。

要求：
1. 回答应准确、完整、有条理
2. 适当使用分步骤的格式
3. 包含具体的例子或解释

指令: {instruction}

回答:"""


# ============================================================
# 数据结构
# ============================================================

@dataclass
class EvolNode:
    """进化节点：记录一条指令及其进化路径"""
    instruction: str
    response: Optional[str] = None
    depth: int = 0
    evolution_type: str = "seed"  # seed / depth_* / breadth
    parent_id: Optional[str] = None
    node_id: str = ""
    children_ids: list = field(default_factory=list)

    def __post_init__(self):
        if not self.node_id:
            # 基于指令内容生成唯一 ID
            self.node_id = hashlib.md5(self.instruction.encode()).hexdigest()[:12]


# ============================================================
# 过滤器
# ============================================================

class InstructionFilter:
    """指令过滤器：去重 + 质量检查"""

    def __init__(self, rouge_threshold: float = 0.7):
        """
        Args:
            rouge_threshold: ROUGE-L 相似度阈值，超过则判为重复
        """
        self.rouge_threshold = rouge_threshold
        self.existing_instructions: list[str] = []
        self._scorer = None

    @property
    def scorer(self):
        """延迟加载 ROUGE 评分器"""
        if self._scorer is None:
            from rouge_score import rouge_scorer
            self._scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        return self._scorer

    def is_duplicate(self, instruction: str) -> bool:
        """检查指令是否与已有指令重复（基于 ROUGE-L）"""
        for existing in self.existing_instructions:
            score = self.scorer.score(existing, instruction)
            if score["rougeL"].fmeasure > self.rouge_threshold:
                return True
        return False

    def passes_heuristic(self, instruction: str) -> bool:
        """启发式质量检查"""
        # 长度检查
        if len(instruction) < 10 or len(instruction) > 2000:
            return False

        # 拒绝以某些模式开头的指令（通常是 LLM 的元表述）
        bad_prefixes = [
            "作为一个AI", "作为AI", "我是一个", "我无法",
            "As an AI", "I'm an AI", "I cannot",
        ]
        for prefix in bad_prefixes:
            if instruction.strip().startswith(prefix):
                return False

        # 拒绝纯问候或过于简单的指令
        too_simple = ["你好", "谢谢", "再见", "hello", "hi", "thanks"]
        if instruction.strip().lower() in too_simple:
            return False

        return True

    def filter(self, instruction: str) -> bool:
        """综合过滤：通过返回 True"""
        if not self.passes_heuristic(instruction):
            logger.debug("启发式过滤拒绝: %s", instruction[:50])
            return False
        if self.is_duplicate(instruction):
            logger.debug("重复过滤拒绝: %s", instruction[:50])
            return False
        return True

    def add(self, instruction: str):
        """将指令加入已有集合"""
        self.existing_instructions.append(instruction)


# ============================================================
# Evol-Instruct 主引擎
# ============================================================

class EvolInstructEngine:
    """
    Evol-Instruct 指令进化引擎

    使用 OpenAI 兼容 API 驱动 LLM 执行指令进化操作。
    支持 vLLM / Ollama / OpenAI 等后端。
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        model: str = "deepseek-chat",
        depth_prob: float = 0.7,
        max_depth: int = 4,
        temperature: float = 0.9,
    ):
        """
        Args:
            api_base: OpenAI 兼容 API 地址
            api_key: API 密钥
            model: 模型名称
            depth_prob: 选择深度进化的概率（1 - depth_prob 为广度进化概率）
            max_depth: 最大进化深度
            temperature: 生成温度
        """
        self.client = OpenAI(base_url=api_base, api_key=api_key)
        self.model = model
        self.depth_prob = depth_prob
        self.max_depth = max_depth
        self.temperature = temperature
        self.filter = InstructionFilter()
        self.nodes: dict[str, EvolNode] = {}

    def _call_llm(self, prompt: str) -> str:
        """调用 LLM 生成文本"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()

    def _evolve_depth(self, instruction: str) -> tuple[str, str]:
        """
        深度进化：随机选择一种策略增加指令复杂度

        Returns:
            (进化后的指令, 策略名称)
        """
        strategy_name = random.choice(list(DEPTH_EVOLUTION_PROMPTS.keys()))
        prompt_template = DEPTH_EVOLUTION_PROMPTS[strategy_name]
        prompt = prompt_template.format(instruction=instruction)

        evolved = self._call_llm(prompt)
        return evolved, f"depth_{strategy_name}"

    def _evolve_breadth(self, instruction: str) -> tuple[str, str]:
        """
        广度进化：生成同等复杂度但不同领域的指令

        Returns:
            (进化后的指令, 策略名称)
        """
        prompt = BREADTH_EVOLUTION_PROMPT.format(instruction=instruction)
        evolved = self._call_llm(prompt)
        return evolved, "breadth"

    def _generate_response(self, instruction: str) -> str:
        """为指令生成对应的回答"""
        prompt = ANSWER_GENERATION_PROMPT.format(instruction=instruction)
        return self._call_llm(prompt)

    def evolve_one(self, parent_node: EvolNode) -> Optional[EvolNode]:
        """
        对一条指令执行一次进化操作

        Args:
            parent_node: 父节点

        Returns:
            新的进化节点，若进化失败则返回 None
        """
        # 检查深度限制
        if parent_node.depth >= self.max_depth:
            logger.info("达到最大深度 %d，跳过进化", self.max_depth)
            return None

        # 选择进化策略
        if random.random() < self.depth_prob:
            evolved_instruction, evo_type = self._evolve_depth(parent_node.instruction)
        else:
            evolved_instruction, evo_type = self._evolve_breadth(parent_node.instruction)

        # 过滤检查
        if not self.filter.filter(evolved_instruction):
            logger.info("进化结果未通过过滤: %s...", evolved_instruction[:50])
            return None

        # 创建新节点
        child = EvolNode(
            instruction=evolved_instruction,
            depth=parent_node.depth + 1,
            evolution_type=evo_type,
            parent_id=parent_node.node_id,
        )

        # 生成回答
        child.response = self._generate_response(evolved_instruction)

        # 更新关系
        parent_node.children_ids.append(child.node_id)
        self.filter.add(evolved_instruction)
        self.nodes[child.node_id] = child

        logger.info(
            "进化成功: [%s] depth=%d, type=%s, instruction=%s...",
            child.node_id, child.depth, evo_type, evolved_instruction[:40],
        )
        return child

    def run(
        self,
        seed_instructions: list[str],
        target_count: int = 100,
        max_attempts: int = 300,
    ) -> list[EvolNode]:
        """
        运行 Evol-Instruct 流程

        Args:
            seed_instructions: 种子指令列表
            target_count: 目标指令数量
            max_attempts: 最大尝试次数（防止无限循环）

        Returns:
            所有生成的 EvolNode 列表
        """
        logger.info("开始 Evol-Instruct，种子数=%d，目标数=%d", len(seed_instructions), target_count)

        # 初始化种子节点
        for inst in seed_instructions:
            node = EvolNode(instruction=inst, depth=0, evolution_type="seed")
            self.nodes[node.node_id] = node
            self.filter.add(inst)

        attempts = 0
        while len(self.nodes) < target_count and attempts < max_attempts:
            attempts += 1

            # 随机选择一个现有节点作为进化起点
            parent = random.choice(list(self.nodes.values()))
            child = self.evolve_one(parent)

            if child is not None:
                logger.info("当前指令数: %d / %d", len(self.nodes), target_count)

        logger.info(
            "Evol-Instruct 完成: 生成 %d 条指令（%d 次尝试）",
            len(self.nodes), attempts,
        )
        return list(self.nodes.values())

    def save(self, output_path: str):
        """保存结果到 JSON 文件"""
        data = [asdict(node) for node in self.nodes.values()]
        Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("结果已保存到 %s", output_path)

    def print_evolution_tree(self):
        """打印进化树结构"""
        # 找到所有根节点（种子）
        roots = [n for n in self.nodes.values() if n.parent_id is None]

        def _print_tree(node: EvolNode, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            print(f"{prefix}{connector}[{node.evolution_type}] {node.instruction[:60]}...")
            new_prefix = prefix + ("    " if is_last else "│   ")
            children = [self.nodes[cid] for cid in node.children_ids if cid in self.nodes]
            for i, child in enumerate(children):
                _print_tree(child, new_prefix, i == len(children) - 1)

        for root in roots:
            print(f"\n种子: {root.instruction[:60]}...")
            children = [self.nodes[cid] for cid in root.children_ids if cid in self.nodes]
            for i, child in enumerate(children):
                _print_tree(child, "", i == len(children) - 1)


# ============================================================
# 默认种子指令集
# ============================================================

DEFAULT_SEEDS = [
    "解释什么是机器学习中的过拟合，以及如何避免它。",
    "写一个 Python 函数来实现二分查找算法。",
    "比较关系型数据库和 NoSQL 数据库的优缺点。",
    "解释 TCP 三次握手的过程。",
    "设计一个简单的待办事项应用的数据库表结构。",
    "解释什么是 RESTful API，并给出一个设计示例。",
    "写一段代码来解析 JSON 数据并提取特定字段。",
    "解释深度学习中梯度消失问题的原因和解决方案。",
    "描述如何使用 Git 进行团队协作开发。",
    "解释什么是 Docker 容器，以及它与虚拟机的区别。",
]


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evol-Instruct 指令进化数据生成")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000/v1",
                        help="OpenAI 兼容 API 地址")
    parser.add_argument("--api-key", type=str, default="not-needed",
                        help="API 密钥")
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="模型名称")
    parser.add_argument("--target", type=int, default=100,
                        help="目标生成指令数量")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="最大进化深度")
    parser.add_argument("--output", type=str, default="evol_instruct_output.json",
                        help="输出文件路径")
    parser.add_argument("--seed-file", type=str, default=None,
                        help="自定义种子指令文件（每行一条指令），不指定则使用默认种子")
    args = parser.parse_args()

    # 加载种子指令
    if args.seed_file:
        seeds = Path(args.seed_file).read_text(encoding="utf-8").strip().split("\n")
        seeds = [s.strip() for s in seeds if s.strip()]
    else:
        seeds = DEFAULT_SEEDS

    # 运行 Evol-Instruct
    engine = EvolInstructEngine(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
        max_depth=args.max_depth,
    )

    nodes = engine.run(seed_instructions=seeds, target_count=args.target)

    # 保存结果
    engine.save(args.output)

    # 打印进化树
    engine.print_evolution_tree()

    # 统计信息
    depth_counts = {}
    type_counts = {}
    for node in nodes:
        depth_counts[node.depth] = depth_counts.get(node.depth, 0) + 1
        type_counts[node.evolution_type] = type_counts.get(node.evolution_type, 0) + 1

    print("\n=== 统计信息 ===")
    print(f"总指令数: {len(nodes)}")
    print(f"深度分布: {depth_counts}")
    print(f"类型分布: {type_counts}")
