"""
Chain-of-Thought (CoT) Prompting 框架

本模块实现了 CoT Prompting 的核心功能，包括 Few-shot CoT、Zero-shot CoT
和 Prompt 模板管理。

核心思想:
- Few-shot CoT: 在 prompt 中提供带推理步骤的示例
- Zero-shot CoT: 添加 "Let's think step by step" 触发推理
- 模板管理: 统一管理不同任务类型的 CoT prompt 模板

参考:
- Wei et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.
- Kojima et al. (2022). Large Language Models are Zero-Shot Reasoners.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class CoTExample:
    """
    CoT 示例数据类

    Attributes:
        question: 问题文本
        reasoning: 推理过程
        answer: 最终答案
    """
    question: str
    reasoning: str
    answer: str

    def format(self) -> str:
        """格式化为 prompt 文本"""
        return f"Q: {self.question}\nA: {self.reasoning} 答案是{self.answer}。"


@dataclass
class CoTPromptTemplate:
    """
    CoT Prompt 模板

    Attributes:
        task_type: 任务类型（如 math、logic、code）
        system_instruction: 系统指令
        examples: Few-shot 示例列表
        zero_shot_trigger: Zero-shot 触发语
        answer_extraction_pattern: 答案提取正则表达式
    """
    task_type: str
    system_instruction: str = ""
    examples: List[CoTExample] = field(default_factory=list)
    zero_shot_trigger: str = "让我们一步一步思考。"
    answer_extraction_pattern: str = r"答案[是为：:]\s*(.+?)[\n。$]"


class CoTPromptManager:
    """
    CoT Prompt 模板管理器

    管理不同任务类型的 prompt 模板，支持 Few-shot 和 Zero-shot 两种模式。
    """

    def __init__(self):
        """初始化模板管理器，注册默认模板"""
        self.templates: Dict[str, CoTPromptTemplate] = {}
        self._register_default_templates()

    def _register_default_templates(self):
        """注册默认的数学和逻辑推理模板"""
        # 数学推理模板
        math_template = CoTPromptTemplate(
            task_type="math",
            system_instruction="你是一个数学推理专家。请仔细分析问题，展示每一步计算过程。",
            examples=[
                CoTExample(
                    question="Roger有5个网球。他又买了2罐网球，每罐有3个。现在他有多少个网球？",
                    reasoning="Roger一开始有5个网球。2罐网球共有2×3=6个。5+6=11。",
                    answer="11"
                ),
                CoTExample(
                    question="食堂有23个苹果。如果他们用了20个做午餐，又买了6个，现在有多少个苹果？",
                    reasoning="食堂一开始有23个苹果。用了20个后剩23-20=3个。又买了6个，3+6=9。",
                    answer="9"
                ),
                CoTExample(
                    question="一个人有12块钱，花了5块买面包，又花了3块买水，最后找回2块。他还有多少钱？",
                    reasoning="一开始有12块。花了5块买面包：12-5=7。花了3块买水：7-3=4。找回2块：4+2=6。",
                    answer="6"
                ),
            ],
            answer_extraction_pattern=r"答案[是为：:]\s*(-?\d+\.?\d*)"
        )
        self.register_template(math_template)

        # 逻辑推理模板
        logic_template = CoTPromptTemplate(
            task_type="logic",
            system_instruction="你是一个逻辑推理专家。请仔细分析条件，逐步推导结论。",
            examples=[
                CoTExample(
                    question="所有的猫都是动物。Tom是一只猫。Tom是动物吗？",
                    reasoning="前提1：所有的猫都是动物。前提2：Tom是一只猫。根据前提1，猫属于动物类别。Tom是猫，所以Tom是动物。",
                    answer="是"
                ),
                CoTExample(
                    question="如果下雨，地面就会湿。地面是湿的。一定下过雨吗？",
                    reasoning="这是一个逻辑推理问题。'如果下雨→地面湿'是正确的，但反过来'地面湿→一定下雨'不一定成立。地面可能因为其他原因而湿（如洒水、管道泄漏）。这是肯定后件的谬误。",
                    answer="不一定"
                ),
            ],
            answer_extraction_pattern=r"答案[是为：:]\s*(.+?)[\n。$]"
        )
        self.register_template(logic_template)

    def register_template(self, template: CoTPromptTemplate):
        """
        注册新的 prompt 模板

        Args:
            template: CoT prompt 模板对象
        """
        self.templates[template.task_type] = template

    def get_template(self, task_type: str) -> Optional[CoTPromptTemplate]:
        """
        获取指定任务类型的模板

        Args:
            task_type: 任务类型

        Returns:
            模板对象，若不存在返回 None
        """
        return self.templates.get(task_type)

    def list_templates(self) -> List[str]:
        """列出所有已注册的模板类型"""
        return list(self.templates.keys())


class CoTPrompter:
    """
    CoT Prompting 执行器

    根据模板生成 Few-shot 或 Zero-shot CoT prompt，
    并从模型输出中提取答案。
    """

    def __init__(self, template: CoTPromptTemplate):
        """
        初始化 CoT Prompter

        Args:
            template: 要使用的 prompt 模板
        """
        self.template = template

    def build_few_shot_prompt(self, question: str, n_examples: Optional[int] = None) -> str:
        """
        构建 Few-shot CoT prompt

        将系统指令、示例和目标问题组合为完整的 prompt。

        Args:
            question: 目标问题
            n_examples: 使用的示例数量（None 表示使用全部）

        Returns:
            完整的 Few-shot CoT prompt
        """
        parts = []

        # 系统指令
        if self.template.system_instruction:
            parts.append(self.template.system_instruction)
            parts.append("")

        # Few-shot 示例
        examples = self.template.examples
        if n_examples is not None:
            examples = examples[:n_examples]

        for example in examples:
            parts.append(example.format())
            parts.append("")

        # 目标问题
        parts.append(f"Q: {question}")
        parts.append("A:")

        return "\n".join(parts)

    def build_zero_shot_prompt(self, question: str) -> str:
        """
        构建 Zero-shot CoT prompt

        在问题后添加 "让我们一步一步思考" 触发推理。

        Args:
            question: 目标问题

        Returns:
            Zero-shot CoT prompt
        """
        parts = []

        # 系统指令
        if self.template.system_instruction:
            parts.append(self.template.system_instruction)
            parts.append("")

        # 问题 + 触发语
        parts.append(f"Q: {question}")
        parts.append(f"A: {self.template.zero_shot_trigger}")

        return "\n".join(parts)

    def extract_answer(self, response: str) -> str:
        """
        从模型输出中提取最终答案

        按优先级尝试多种提取模式:
        1. 模板定义的正则模式
        2. \\boxed{} 格式
        3. "答案是" / "The answer is" 格式
        4. 最后一个数字

        Args:
            response: 模型的完整输出

        Returns:
            提取到的答案文本
        """
        # 方法1: 模板定义的正则模式
        matches = re.findall(self.template.answer_extraction_pattern, response)
        if matches:
            return matches[-1].strip()

        # 方法2: \boxed{} 格式（常见于数学输出）
        boxed = re.findall(r'\\boxed\{([^}]+)\}', response)
        if boxed:
            return boxed[-1].strip()

        # 方法3: "The answer is" 格式
        en_pattern = re.findall(r'[Tt]he answer is\s*(.+?)[\n.]', response)
        if en_pattern:
            return en_pattern[-1].strip()

        # 方法4: 回退到最后一个数字
        numbers = re.findall(r'-?\d+\.?\d*', response)
        if numbers:
            return numbers[-1]

        # 无法提取时返回整个响应的最后一行
        lines = response.strip().split('\n')
        return lines[-1].strip() if lines else ""


class CoTTwoStagePrompter:
    """
    两阶段 Zero-shot CoT 执行器

    实现 Kojima et al. (2022) 提出的两阶段 Zero-shot CoT:
    阶段1: 生成推理过程
    阶段2: 提取最终答案

    注意: 本实现使用模拟的 LLM 调用来演示流程。
    实际使用时需要替换为真实的 LLM API 调用。
    """

    def __init__(self, template: CoTPromptTemplate):
        """
        初始化两阶段执行器

        Args:
            template: 使用的 prompt 模板
        """
        self.template = template

    def stage1_generate_reasoning(self, question: str) -> str:
        """
        阶段1：生成推理过程

        Args:
            question: 输入问题

        Returns:
            阶段1的 prompt（实际使用时应发送给 LLM）
        """
        prompt = (
            f"Q: {question}\n"
            f"A: {self.template.zero_shot_trigger}\n"
        )
        return prompt

    def stage2_extract_answer(self, question: str, reasoning: str) -> str:
        """
        阶段2：从推理过程中提取最终答案

        Args:
            question: 原始问题
            reasoning: 阶段1生成的推理过程

        Returns:
            阶段2的 prompt（实际使用时应发送给 LLM）
        """
        prompt = (
            f"Q: {question}\n"
            f"A: {reasoning}\n"
            f"因此，答案是"
        )
        return prompt

    def run(self, question: str, simulate_reasoning: Optional[str] = None) -> Dict[str, str]:
        """
        执行两阶段推理流程

        Args:
            question: 输入问题
            simulate_reasoning: 模拟的推理过程（用于演示）

        Returns:
            包含各阶段 prompt 和结果的字典
        """
        # 阶段1
        stage1_prompt = self.stage1_generate_reasoning(question)

        # 模拟推理结果（实际使用时由 LLM 生成）
        reasoning = simulate_reasoning or "[此处应为 LLM 生成的推理过程]"

        # 阶段2
        stage2_prompt = self.stage2_extract_answer(question, reasoning)

        return {
            "question": question,
            "stage1_prompt": stage1_prompt,
            "reasoning": reasoning,
            "stage2_prompt": stage2_prompt,
        }


def demonstrate_cot_comparison():
    """
    演示 Standard Prompting vs CoT Prompting 的对比

    展示相同问题在两种 prompting 策略下的 prompt 构建差异。
    """
    print("=" * 70)
    print("Standard Prompting vs CoT Prompting 对比演示")
    print("=" * 70)

    question = "小明有15颗糖果，他给了小红5颗，又从小华那里得到了8颗。然后他把手上糖果的一半给了妈妈。小明现在有多少颗糖果？"

    # Standard Prompting（直接回答）
    print("\n--- Standard Prompting ---")
    standard_prompt = f"Q: {question}\nA:"
    print(standard_prompt)
    print("\n模型可能直接输出: 9")

    # Few-shot CoT Prompting
    print("\n--- Few-shot CoT Prompting ---")
    manager = CoTPromptManager()
    template = manager.get_template("math")
    prompter = CoTPrompter(template)
    few_shot_prompt = prompter.build_few_shot_prompt(question, n_examples=2)
    print(few_shot_prompt)

    # 模拟 CoT 输出
    simulated_cot_output = (
        "小明一开始有15颗糖果。"
        "给了小红5颗后：15-5=10颗。"
        "从小华那里得到8颗：10+8=18颗。"
        "把一半给了妈妈：18÷2=9颗。"
        "答案是9。"
    )
    print(f"\n模型 CoT 输出: {simulated_cot_output}")

    # 答案提取
    answer = prompter.extract_answer(simulated_cot_output)
    print(f"提取的答案: {answer}")

    # Zero-shot CoT
    print("\n--- Zero-shot CoT Prompting ---")
    zero_shot_prompt = prompter.build_zero_shot_prompt(question)
    print(zero_shot_prompt)


def demonstrate_two_stage_cot():
    """演示两阶段 Zero-shot CoT"""
    print("\n" + "=" * 70)
    print("两阶段 Zero-shot CoT 演示")
    print("=" * 70)

    manager = CoTPromptManager()
    template = manager.get_template("math")
    two_stage = CoTTwoStagePrompter(template)

    question = "一个班有30个学生，其中男生比女生多6人。男生有多少人？"
    simulated_reasoning = (
        "设女生有x人，则男生有x+6人。"
        "总人数: x + (x+6) = 30。"
        "2x + 6 = 30。"
        "2x = 24。"
        "x = 12。"
        "所以男生有12+6=18人。"
    )

    result = two_stage.run(question, simulate_reasoning=simulated_reasoning)

    print(f"\n问题: {result['question']}")
    print(f"\n阶段1 Prompt:\n{result['stage1_prompt']}")
    print(f"\n模拟推理过程:\n{result['reasoning']}")
    print(f"\n阶段2 Prompt:\n{result['stage2_prompt']}")


def demonstrate_answer_extraction():
    """演示答案提取功能"""
    print("\n" + "=" * 70)
    print("答案提取演示")
    print("=" * 70)

    manager = CoTPromptManager()
    template = manager.get_template("math")
    prompter = CoTPrompter(template)

    # 测试不同格式的答案
    test_responses = [
        "经过计算，5+3=8。答案是8。",
        "Step 1: 10-3=7. Step 2: 7*2=14. The answer is 14.",
        "计算过程如下：\\boxed{42}",
        "最终结果为 3.14159",
        "所以答案为：256",
    ]

    for response in test_responses:
        answer = prompter.extract_answer(response)
        print(f"\n输入: {response}")
        print(f"提取答案: {answer}")


def demonstrate_template_management():
    """演示模板管理功能"""
    print("\n" + "=" * 70)
    print("模板管理演示")
    print("=" * 70)

    manager = CoTPromptManager()

    # 查看已注册模板
    print(f"\n已注册模板: {manager.list_templates()}")

    # 注册自定义模板
    code_template = CoTPromptTemplate(
        task_type="code",
        system_instruction="你是一个编程专家。请分析需求，逐步设计解决方案。",
        examples=[
            CoTExample(
                question="写一个函数计算列表的平均值",
                reasoning="1. 需要计算列表所有元素的和。2. 除以列表的长度。3. 需要处理空列表的情况。",
                answer="def average(lst): return sum(lst)/len(lst) if lst else 0"
            ),
        ],
        zero_shot_trigger="让我分析这个编程问题，逐步设计解决方案。",
        answer_extraction_pattern=r"答案[是为：:]\s*(.+)"
    )
    manager.register_template(code_template)

    print(f"注册后模板: {manager.list_templates()}")

    # 使用代码模板
    code_prompter = CoTPrompter(manager.get_template("code"))
    prompt = code_prompter.build_few_shot_prompt("写一个函数判断一个数是否是回文数")
    print(f"\n代码任务 Few-shot Prompt:\n{prompt}")


if __name__ == "__main__":
    # 演示1: Standard vs CoT 对比
    demonstrate_cot_comparison()

    # 演示2: 两阶段 Zero-shot CoT
    demonstrate_two_stage_cot()

    # 演示3: 答案提取
    demonstrate_answer_extraction()

    # 演示4: 模板管理
    demonstrate_template_management()
