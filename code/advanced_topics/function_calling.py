"""
Function Calling 实现: 工具注册、参数解析、调用执行、ReAct 循环

本模块实现了 LLM Agent 的核心能力 -- 工具调用 (Function Calling)。
包括工具注册系统、参数解析验证、调用执行引擎，以及 ReAct 推理循环。

核心概念:
1. Function Calling: LLM 输出结构化的工具调用指令，而非纯文本
2. ReAct: Reasoning + Acting 框架，交替进行思考和行动
3. 工具注册: 统一的工具描述格式，使 LLM 了解可用工具
4. Agent 循环: 观察 -> 思考 -> 行动 -> 观察 的循环

ReAct 循环流程:
  Thought: 我需要查询天气信息
  Action: get_weather(city="Beijing")
  Observation: 北京今天晴，25度
  Thought: 我已经获得了天气信息，可以回答用户了
  Answer: 北京今天天气晴朗，温度 25 度。

参考:
- Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in LLMs.
- Schick et al. (2023). Toolformer: Language Models Can Teach Themselves to Use Tools.
- OpenAI (2023). Function Calling API Documentation.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum


class ParameterType(Enum):
    """工具参数类型"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """
    工具参数描述

    Attributes:
        name: 参数名称
        param_type: 参数类型
        description: 参数描述
        required: 是否必需
        default: 默认值
        enum: 可选的枚举值列表
    """
    name: str
    param_type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None


@dataclass
class ToolDefinition:
    """
    工具定义

    Attributes:
        name: 工具名称
        description: 工具功能描述
        parameters: 参数列表
        function: 实际的 Python 函数
    """
    name: str
    description: str
    parameters: List[ToolParameter]
    function: Callable

    def to_schema(self) -> Dict:
        """转换为 JSON Schema 格式（类似 OpenAI 的 Function Calling Schema）"""
        properties = {}
        required = []

        for param in self.parameters:
            prop = {
                "type": param.param_type.value,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class FunctionCall:
    """
    函数调用请求

    Attributes:
        function_name: 要调用的函数名
        arguments: 参数字典
    """
    function_name: str
    arguments: Dict[str, Any]


@dataclass
class FunctionResult:
    """
    函数调用结果

    Attributes:
        function_name: 被调用的函数名
        result: 返回结果
        success: 是否成功
        error: 错误信息（如果失败）
    """
    function_name: str
    result: Any = None
    success: bool = True
    error: str = ""


class ToolRegistry:
    """
    工具注册中心

    管理所有可用的工具，提供注册、查找和验证功能。
    """

    def __init__(self):
        self.tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition):
        """
        注册一个工具

        Args:
            tool: 工具定义
        """
        if tool.name in self.tools:
            raise ValueError(f"工具 '{tool.name}' 已经注册")
        self.tools[tool.name] = tool

    def register_function(
        self,
        name: str,
        description: str,
        parameters: List[ToolParameter],
        function: Callable,
    ):
        """
        便捷注册方法

        Args:
            name: 工具名称
            description: 描述
            parameters: 参数列表
            function: 实际函数
        """
        tool = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            function=function,
        )
        self.register(tool)

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        """获取工具定义"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """列出所有已注册的工具名称"""
        return list(self.tools.keys())

    def get_all_schemas(self) -> List[Dict]:
        """获取所有工具的 JSON Schema"""
        return [tool.to_schema() for tool in self.tools.values()]

    def format_tools_for_prompt(self) -> str:
        """
        格式化工具描述，用于放入 LLM 的 system prompt 中

        Returns:
            formatted: 工具描述文本
        """
        lines = ["Available tools:"]
        for name, tool in self.tools.items():
            lines.append(f"\n  {name}: {tool.description}")
            lines.append("    Parameters:")
            for param in tool.parameters:
                req = "required" if param.required else "optional"
                lines.append(
                    f"      - {param.name} ({param.param_type.value}, {req}): "
                    f"{param.description}"
                )
        return "\n".join(lines)


class FunctionCallParser:
    """
    函数调用解析器

    从 LLM 的输出文本中提取函数调用指令。
    支持多种格式:
    1. JSON 格式: {"name": "func", "arguments": {"arg1": "val1"}}
    2. 类函数调用: func(arg1="val1", arg2=42)
    3. XML 风格: <function_call name="func"><arg name="arg1">val1</arg></function_call>
    """

    @staticmethod
    def parse_json_format(text: str) -> Optional[FunctionCall]:
        """
        解析 JSON 格式的函数调用

        Args:
            text: LLM 输出文本

        Returns:
            call: 解析的函数调用，失败返回 None
        """
        # 尝试找到 JSON 块
        json_pattern = r'\{[^{}]*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^{}]*\}[^{}]*\}'
        match = re.search(json_pattern, text)

        if not match:
            return None

        try:
            data = json.loads(match.group())
            return FunctionCall(
                function_name=data["name"],
                arguments=data.get("arguments", {}),
            )
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def parse_function_call_format(text: str) -> Optional[FunctionCall]:
        """
        解析类函数调用格式: func_name(arg1="val1", arg2=42)

        Args:
            text: LLM 输出文本

        Returns:
            call: 解析的函数调用
        """
        pattern = r'(\w+)\((.*?)\)'
        match = re.search(pattern, text)

        if not match:
            return None

        func_name = match.group(1)
        args_str = match.group(2)

        # 解析参数
        arguments = {}
        if args_str.strip():
            # 简单的参数解析
            arg_pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|(\d+(?:\.\d+)?)|(\w+))'
            for arg_match in re.finditer(arg_pattern, args_str):
                arg_name = arg_match.group(1)
                # 按优先级: 字符串 > 数字 > 标识符
                if arg_match.group(2) is not None:
                    arguments[arg_name] = arg_match.group(2)
                elif arg_match.group(3) is not None:
                    val = arg_match.group(3)
                    arguments[arg_name] = (
                        float(val) if "." in val else int(val)
                    )
                elif arg_match.group(4) is not None:
                    val = arg_match.group(4)
                    if val.lower() == "true":
                        arguments[arg_name] = True
                    elif val.lower() == "false":
                        arguments[arg_name] = False
                    else:
                        arguments[arg_name] = val

        return FunctionCall(function_name=func_name, arguments=arguments)

    def parse(self, text: str) -> Optional[FunctionCall]:
        """
        自动检测格式并解析

        按优先级尝试: JSON > 函数调用格式

        Args:
            text: LLM 输出文本

        Returns:
            call: 解析的函数调用
        """
        # 尝试 JSON 格式
        result = self.parse_json_format(text)
        if result:
            return result

        # 尝试函数调用格式
        result = self.parse_function_call_format(text)
        if result:
            return result

        return None


class FunctionCallExecutor:
    """
    函数调用执行器

    验证参数并安全地执行工具函数。

    Args:
        registry: 工具注册中心
        max_retries: 最大重试次数
    """

    def __init__(
        self, registry: ToolRegistry, max_retries: int = 3
    ):
        self.registry = registry
        self.max_retries = max_retries

    def validate_arguments(
        self, tool: ToolDefinition, arguments: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        验证函数调用参数

        Args:
            tool: 工具定义
            arguments: 传入的参数

        Returns:
            (is_valid, error_message)
        """
        # 检查必需参数
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                return False, f"缺少必需参数: {param.name}"

        # 检查参数类型
        for param in tool.parameters:
            if param.name in arguments:
                value = arguments[param.name]
                if not self._check_type(value, param.param_type):
                    return False, (
                        f"参数 '{param.name}' 类型错误: "
                        f"期望 {param.param_type.value}, "
                        f"实际 {type(value).__name__}"
                    )

                # 检查枚举值
                if param.enum and value not in param.enum:
                    return False, (
                        f"参数 '{param.name}' 的值 '{value}' "
                        f"不在允许范围内: {param.enum}"
                    )

        return True, ""

    @staticmethod
    def _check_type(value: Any, expected: ParameterType) -> bool:
        """检查值的类型是否匹配"""
        type_map = {
            ParameterType.STRING: str,
            ParameterType.INTEGER: int,
            ParameterType.FLOAT: (int, float),
            ParameterType.BOOLEAN: bool,
            ParameterType.ARRAY: (list, tuple),
            ParameterType.OBJECT: dict,
        }
        expected_type = type_map.get(expected)
        if expected_type is None:
            return True
        return isinstance(value, expected_type)

    def execute(self, call: FunctionCall) -> FunctionResult:
        """
        执行函数调用

        Args:
            call: 函数调用请求

        Returns:
            result: 执行结果
        """
        # 查找工具
        tool = self.registry.get_tool(call.function_name)
        if tool is None:
            return FunctionResult(
                function_name=call.function_name,
                success=False,
                error=f"未知工具: {call.function_name}",
            )

        # 验证参数
        is_valid, error = self.validate_arguments(tool, call.arguments)
        if not is_valid:
            return FunctionResult(
                function_name=call.function_name,
                success=False,
                error=error,
            )

        # 填充默认值
        full_args = {}
        for param in tool.parameters:
            if param.name in call.arguments:
                full_args[param.name] = call.arguments[param.name]
            elif param.default is not None:
                full_args[param.name] = param.default

        # 执行函数
        try:
            result = tool.function(**full_args)
            return FunctionResult(
                function_name=call.function_name,
                result=result,
                success=True,
            )
        except Exception as e:
            return FunctionResult(
                function_name=call.function_name,
                success=False,
                error=f"执行错误: {str(e)}",
            )


@dataclass
class ReActStep:
    """
    ReAct 循环的单个步骤

    Attributes:
        step_type: 步骤类型 (thought/action/observation/answer)
        content: 步骤内容
    """
    step_type: str  # "thought", "action", "observation", "answer"
    content: str


class ReActAgent:
    """
    ReAct Agent

    实现 Reasoning + Acting 的交互式 Agent 框架。
    Agent 交替进行思考 (Thought) 和行动 (Action)，
    根据观察结果 (Observation) 决定下一步。

    Args:
        registry: 工具注册中心
        max_steps: 最大推理步数（防止无限循环）
    """

    def __init__(
        self,
        registry: ToolRegistry,
        max_steps: int = 10,
    ):
        self.registry = registry
        self.executor = FunctionCallExecutor(registry)
        self.parser = FunctionCallParser()
        self.max_steps = max_steps

    def build_system_prompt(self) -> str:
        """
        构建 Agent 的系统提示

        Returns:
            prompt: 系统提示文本
        """
        tools_desc = self.registry.format_tools_for_prompt()

        return (
            "You are a helpful assistant with access to the following tools.\n"
            f"{tools_desc}\n\n"
            "To use a tool, follow this format:\n"
            "Thought: [your reasoning about what to do]\n"
            "Action: tool_name(param1=\"value1\", param2=42)\n"
            "Observation: [tool result will appear here]\n\n"
            "You can repeat Thought/Action/Observation as needed.\n"
            "When you have enough information, respond with:\n"
            "Thought: I now have enough information.\n"
            "Answer: [your final response to the user]\n"
        )

    def simulate_step(
        self, query: str, history: List[ReActStep]
    ) -> ReActStep:
        """
        模拟 Agent 的一个推理步骤

        注意: 这是一个教学用的模拟。实际应用中，
        Thought 和 Action 由 LLM 生成。

        Args:
            query: 用户查询
            history: 历史步骤

        Returns:
            step: 下一步骤
        """
        # 如果没有历史，生成初始思考
        if not history:
            tools = self.registry.list_tools()
            return ReActStep(
                step_type="thought",
                content=(
                    f"用户问了: '{query}'。"
                    f"我有以下工具可用: {tools}。"
                    f"让我思考应该使用哪个工具。"
                ),
            )

        last_step = history[-1]

        # 如果上一步是思考，生成行动
        if last_step.step_type == "thought":
            # 简单的工具选择逻辑
            tools = self.registry.list_tools()
            if tools:
                tool = self.registry.get_tool(tools[0])
                if tool and tool.parameters:
                    param = tool.parameters[0]
                    return ReActStep(
                        step_type="action",
                        content=f'{tools[0]}({param.name}="{query}")',
                    )

        # 如果上一步是观察，生成总结
        if last_step.step_type == "observation":
            return ReActStep(
                step_type="answer",
                content=f"根据工具返回的结果: {last_step.content}",
            )

        # 默认结束
        return ReActStep(
            step_type="answer",
            content="抱歉，我无法处理这个请求。",
        )

    def run(self, query: str) -> Tuple[str, List[ReActStep]]:
        """
        运行 ReAct 循环

        Args:
            query: 用户查询

        Returns:
            (final_answer, steps): 最终回答和推理过程
        """
        history: List[ReActStep] = []

        for step_num in range(self.max_steps):
            # 生成下一步
            step = self.simulate_step(query, history)
            history.append(step)

            # 如果是行动步骤，执行工具调用
            if step.step_type == "action":
                call = self.parser.parse(step.content)
                if call:
                    result = self.executor.execute(call)
                    obs = ReActStep(
                        step_type="observation",
                        content=(
                            str(result.result) if result.success
                            else f"错误: {result.error}"
                        ),
                    )
                    history.append(obs)
                else:
                    obs = ReActStep(
                        step_type="observation",
                        content="无法解析工具调用",
                    )
                    history.append(obs)

            # 如果是回答步骤，结束循环
            if step.step_type == "answer":
                return step.content, history

        return "达到最大步数限制", history

    def format_trace(self, steps: List[ReActStep]) -> str:
        """
        格式化推理轨迹

        Args:
            steps: 推理步骤列表

        Returns:
            formatted: 格式化的推理过程
        """
        lines = []
        for i, step in enumerate(steps):
            prefix = {
                "thought": "Thought",
                "action": "Action",
                "observation": "Observation",
                "answer": "Answer",
            }.get(step.step_type, step.step_type)

            lines.append(f"[Step {i+1}] {prefix}: {step.content}")
        return "\n".join(lines)


# ---- 示例工具函数 ----


def calculator(expression: str) -> str:
    """
    简易计算器工具

    Args:
        expression: 数学表达式字符串

    Returns:
        计算结果
    """
    # 安全的表达式评估（仅允许数学运算）
    allowed_chars = set("0123456789+-*/().% ")
    if not all(c in allowed_chars for c in expression):
        return "错误: 表达式包含不允许的字符"

    try:
        result = eval(expression)  # 注: 教学用途，实际应用需更安全的实现
        return str(result)
    except Exception as e:
        return f"计算错误: {str(e)}"


def get_weather(city: str) -> str:
    """
    模拟天气查询工具

    Args:
        city: 城市名

    Returns:
        天气信息
    """
    # 模拟数据
    weather_data = {
        "Beijing": "晴天，25度，湿度40%",
        "Shanghai": "多云，28度，湿度65%",
        "Guangzhou": "小雨，30度，湿度80%",
        "Shenzhen": "阴天，29度，湿度70%",
    }
    return weather_data.get(city, f"未找到城市 '{city}' 的天气数据")


def search_knowledge(query: str) -> str:
    """
    模拟知识搜索工具

    Args:
        query: 搜索查询

    Returns:
        搜索结果
    """
    # 模拟搜索结果
    return f"关于 '{query}' 的搜索结果: 这是一个模拟的搜索返回内容。"


if __name__ == "__main__":
    print("=" * 60)
    print("Function Calling 与 ReAct Agent 演示")
    print("=" * 60)

    # ---- 1. 注册工具 ----
    print("\n[1] 注册工具...")
    registry = ToolRegistry()

    registry.register_function(
        name="calculator",
        description="执行数学计算",
        parameters=[
            ToolParameter(
                name="expression",
                param_type=ParameterType.STRING,
                description="数学表达式",
                required=True,
            ),
        ],
        function=calculator,
    )

    registry.register_function(
        name="get_weather",
        description="查询城市天气信息",
        parameters=[
            ToolParameter(
                name="city",
                param_type=ParameterType.STRING,
                description="城市名（英文）",
                required=True,
            ),
        ],
        function=get_weather,
    )

    registry.register_function(
        name="search_knowledge",
        description="搜索知识库",
        parameters=[
            ToolParameter(
                name="query",
                param_type=ParameterType.STRING,
                description="搜索查询",
                required=True,
            ),
        ],
        function=search_knowledge,
    )

    print(f"    已注册工具: {registry.list_tools()}")

    # ---- 2. 工具 Schema 展示 ----
    print("\n[2] 工具 Schema (JSON)...")
    schemas = registry.get_all_schemas()
    for schema in schemas:
        print(f"    {json.dumps(schema, indent=2, ensure_ascii=False)[:100]}...")

    # ---- 3. 函数调用解析 ----
    print("\n[3] 函数调用解析...")
    parser = FunctionCallParser()

    test_texts = [
        'calculator(expression="2 + 3 * 4")',
        '{"name": "get_weather", "arguments": {"city": "Beijing"}}',
        'search_knowledge(query="quantum computing")',
    ]

    for text in test_texts:
        call = parser.parse(text)
        if call:
            print(f"    输入: {text}")
            print(f"    解析: {call.function_name}({call.arguments})")
        else:
            print(f"    解析失败: {text}")

    # ---- 4. 函数执行 ----
    print("\n[4] 函数执行...")
    executor = FunctionCallExecutor(registry)

    calls = [
        FunctionCall("calculator", {"expression": "2 + 3 * 4"}),
        FunctionCall("get_weather", {"city": "Beijing"}),
        FunctionCall("get_weather", {"city": "Tokyo"}),
        FunctionCall("unknown_tool", {"arg": "val"}),
    ]

    for call in calls:
        result = executor.execute(call)
        status = "成功" if result.success else "失败"
        output = result.result if result.success else result.error
        print(f"    {call.function_name}({call.arguments}) -> [{status}] {output}")

    # ---- 5. ReAct Agent ----
    print("\n[5] ReAct Agent 运行...")
    agent = ReActAgent(registry, max_steps=6)

    # 打印系统提示
    print(f"\n    系统提示 (前200字):")
    prompt = agent.build_system_prompt()
    print(f"    {prompt[:200]}...")

    # 运行 Agent
    answer, steps = agent.run("What is the weather in Beijing?")

    print(f"\n    推理轨迹:")
    print(f"    {agent.format_trace(steps)}")
    print(f"\n    最终答案: {answer}")

    # ---- 6. 参数验证 ----
    print("\n[6] 参数验证...")
    tool = registry.get_tool("calculator")
    if tool:
        # 正确参数
        valid, err = executor.validate_arguments(tool, {"expression": "1+1"})
        print(f"    正确参数: valid={valid}")

        # 缺少必需参数
        valid, err = executor.validate_arguments(tool, {})
        print(f"    缺少参数: valid={valid}, error='{err}'")

        # 类型错误
        valid, err = executor.validate_arguments(tool, {"expression": 123})
        print(f"    类型错误: valid={valid}, error='{err}'")

    print("\n演示完成!")
