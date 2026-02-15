"""
对话模板处理

本模块实现了多种 LLM 对话格式模板，将原始对话数据
转换为模型可接受的格式化文本。

支持的模板:
- ChatML: OpenAI 提出的标准格式，被 Qwen/Yi 等采用
- Llama: Meta Llama 2/3 的对话格式
- Alpaca: Stanford Alpaca 的指令格式

核心功能:
- 将多轮对话格式化为单个字符串
- 计算每个角色文本的起止位置（用于 Prompt Masking）
- 支持 System Prompt

参考:
- ChatML: https://github.com/openai/openai-python/blob/main/chatml.md
- Llama Chat: https://llama.meta.com/docs/model-cards-and-prompt-formats/
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Message:
    """
    单条消息

    Attributes:
        role: 角色 ("system", "user", "assistant")
        content: 消息内容
    """
    role: str
    content: str


class ChatTemplate(ABC):
    """
    对话模板基类

    所有模板子类需要实现:
    - format_messages: 将消息列表格式化为字符串
    - get_response_positions: 返回 assistant 回答的起止位置
    """

    @abstractmethod
    def format_messages(self, messages: List[Message]) -> str:
        """
        将消息列表格式化为单个字符串

        Args:
            messages: 消息列表

        Returns:
            格式化后的字符串
        """
        pass

    @abstractmethod
    def get_response_positions(
        self, messages: List[Message]
    ) -> List[Tuple[int, int]]:
        """
        获取 assistant 回答在格式化文本中的字符位置

        用于 Prompt Masking: 只有这些位置的 token 参与损失计算。

        Args:
            messages: 消息列表

        Returns:
            [(start, end), ...] 字符位置列表
        """
        pass

    def format_for_training(
        self, messages: List[Message]
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """
        格式化并返回训练所需的信息

        Args:
            messages: 消息列表

        Returns:
            (formatted_text, response_positions)
        """
        text = self.format_messages(messages)
        positions = self.get_response_positions(messages)
        return text, positions


class ChatMLTemplate(ChatTemplate):
    """
    ChatML 格式模板

    格式:
        <|im_start|>system
        {system_message}<|im_end|>
        <|im_start|>user
        {user_message}<|im_end|>
        <|im_start|>assistant
        {assistant_message}<|im_end|>

    特殊 Token:
        - <|im_start|>: 消息开始标记
        - <|im_end|>: 消息结束标记

    使用模型: Qwen, Yi, ChatGLM 等
    """

    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"

    def format_messages(self, messages: List[Message]) -> str:
        """
        将消息格式化为 ChatML 格式

        Args:
            messages: 消息列表

        Returns:
            ChatML 格式的字符串
        """
        parts = []
        for msg in messages:
            parts.append(f"{self.IM_START}{msg.role}\n{msg.content}{self.IM_END}\n")
        return "".join(parts)

    def get_response_positions(
        self, messages: List[Message]
    ) -> List[Tuple[int, int]]:
        """
        获取 assistant 回答的字符位置

        Returns:
            assistant 回答内容的 (start, end) 列表
        """
        positions = []
        current_pos = 0

        for msg in messages:
            # 消息头: "<|im_start|>{role}\n"
            header = f"{self.IM_START}{msg.role}\n"
            content_start = current_pos + len(header)

            # 消息尾: "<|im_end|>\n"
            content_end = content_start + len(msg.content)

            if msg.role == "assistant":
                positions.append((content_start, content_end))

            # 完整消息长度
            full_msg = f"{self.IM_START}{msg.role}\n{msg.content}{self.IM_END}\n"
            current_pos += len(full_msg)

        return positions

    def format_prompt(self, messages: List[Message]) -> str:
        """
        格式化为推理用的 prompt（最后加上 assistant 开始标记）

        Args:
            messages: 消息列表（最后一条应为 user）

        Returns:
            推理用的 prompt 字符串
        """
        text = self.format_messages(messages)
        # 添加 assistant 开始标记
        text += f"{self.IM_START}assistant\n"
        return text


class LlamaTemplate(ChatTemplate):
    """
    Llama 2/3 格式模板

    Llama 2 格式:
        <s>[INST] <<SYS>>
        {system_message}
        <</SYS>>

        {user_message} [/INST] {assistant_message} </s>
        <s>[INST] {user_message_2} [/INST] {assistant_message_2} </s>

    特殊 Token:
        - <s>, </s>: 序列起止
        - [INST], [/INST]: 指令标记
        - <<SYS>>, <</SYS>>: 系统提示标记

    使用模型: Llama 2, Llama 3 (格式略有不同)
    """

    def format_messages(self, messages: List[Message]) -> str:
        """
        将消息格式化为 Llama 格式

        Args:
            messages: 消息列表

        Returns:
            Llama 格式的字符串
        """
        parts = []
        system_msg = None

        # 提取系统消息
        msg_list = list(messages)
        if msg_list and msg_list[0].role == "system":
            system_msg = msg_list[0].content
            msg_list = msg_list[1:]

        # 处理对话轮次
        is_first = True
        i = 0
        while i < len(msg_list):
            if msg_list[i].role == "user":
                user_content = msg_list[i].content

                # 第一轮包含系统消息
                if is_first and system_msg:
                    inst_content = f"<<SYS>>\n{system_msg}\n<</SYS>>\n\n{user_content}"
                    is_first = False
                else:
                    inst_content = user_content

                parts.append(f"<s>[INST] {inst_content} [/INST]")

                # 检查是否有对应的 assistant 回复
                if i + 1 < len(msg_list) and msg_list[i + 1].role == "assistant":
                    parts.append(f" {msg_list[i + 1].content} </s>\n")
                    i += 2
                else:
                    parts.append(" ")
                    i += 1
            else:
                i += 1

        return "".join(parts)

    def get_response_positions(
        self, messages: List[Message]
    ) -> List[Tuple[int, int]]:
        """
        获取 assistant 回答的字符位置

        Returns:
            assistant 回答内容的 (start, end) 列表
        """
        text = self.format_messages(messages)
        positions = []

        # 在格式化文本中查找 [/INST] 后的内容直到 </s>
        search_start = 0
        while True:
            inst_end = text.find("[/INST]", search_start)
            if inst_end == -1:
                break

            # assistant 回答开始位置（跳过 "[/INST] "）
            resp_start = inst_end + len("[/INST] ")

            # 查找回答结束标记
            eos_pos = text.find("</s>", resp_start)
            if eos_pos == -1:
                # 最后一个回答可能没有 </s>
                resp_end = len(text)
            else:
                resp_end = eos_pos

            if resp_start < resp_end:
                positions.append((resp_start, resp_end))

            search_start = resp_end + 1

        return positions


class AlpacaTemplate(ChatTemplate):
    """
    Alpaca 格式模板

    格式 (有 input):
        Below is an instruction that describes a task, paired with an input.
        Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Input:
        {input}

        ### Response:
        {response}

    格式 (无 input):
        Below is an instruction that describes a task.
        Write a response that appropriately completes the request.

        ### Instruction:
        {instruction}

        ### Response:
        {response}

    使用模型: Alpaca, Vicuna (变体)
    """

    PROMPT_WITH_INPUT = (
        "Below is an instruction that describes a task, paired with an input. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n"
    )

    PROMPT_WITHOUT_INPUT = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )

    def format_messages(self, messages: List[Message]) -> str:
        """
        将消息格式化为 Alpaca 格式

        注意: Alpaca 格式主要用于单轮指令，这里取第一条 user 消息
        作为 instruction，第一条 assistant 消息作为 response。

        Args:
            messages: 消息列表

        Returns:
            Alpaca 格式的字符串
        """
        instruction = ""
        input_text = ""
        response = ""

        for msg in messages:
            if msg.role == "user":
                instruction = msg.content
            elif msg.role == "assistant":
                response = msg.content
            elif msg.role == "input":
                input_text = msg.content

        if input_text:
            prompt = self.PROMPT_WITH_INPUT.format(
                instruction=instruction, input=input_text
            )
        else:
            prompt = self.PROMPT_WITHOUT_INPUT.format(instruction=instruction)

        return prompt + response

    def get_response_positions(
        self, messages: List[Message]
    ) -> List[Tuple[int, int]]:
        """
        获取 response 在格式化文本中的字符位置

        Returns:
            response 的 (start, end) 列表
        """
        text = self.format_messages(messages)
        marker = "### Response:\n"
        resp_start = text.find(marker)
        if resp_start == -1:
            return []
        resp_start += len(marker)
        return [(resp_start, len(text))]


def build_labels_from_positions(
    input_ids: List[int],
    response_char_positions: List[Tuple[int, int]],
    formatted_text: str,
    tokenizer: object,
) -> List[int]:
    """
    根据 assistant 回答的字符位置构建 labels

    将非 response 部分的 label 设为 -100（不计算损失）

    Args:
        input_ids: token ID 列表
        response_char_positions: assistant 回答的字符位置列表
        formatted_text: 格式化后的完整文本
        tokenizer: 分词器

    Returns:
        labels 列表（-100 表示不计算损失）
    """
    # 简化实现: 基于字符位置近似映射到 token 位置
    # 生产环境需要更精确的 offset mapping
    labels = [-100] * len(input_ids)

    if not response_char_positions:
        return labels

    # 估算每个 token 的平均字符数
    avg_chars_per_token = len(formatted_text) / max(len(input_ids), 1)

    for char_start, char_end in response_char_positions:
        # 近似映射字符位置到 token 位置
        token_start = int(char_start / avg_chars_per_token)
        token_end = int(char_end / avg_chars_per_token)

        # 裁剪到有效范围
        token_start = max(0, min(token_start, len(input_ids)))
        token_end = max(0, min(token_end, len(input_ids)))

        for i in range(token_start, token_end):
            labels[i] = input_ids[i]

    return labels


if __name__ == "__main__":
    print("=" * 60)
    print("对话模板处理演示")
    print("=" * 60)

    # === 1. ChatML 格式 ===
    print("\n--- 1. ChatML 格式 ---")
    chatml = ChatMLTemplate()

    messages = [
        Message(role="system", content="你是一个有帮助的AI助手。"),
        Message(role="user", content="什么是光合作用？"),
        Message(role="assistant", content="光合作用是植物利用阳光、水和二氧化碳合成有机物的过程。"),
        Message(role="user", content="它的化学方程式是什么？"),
        Message(role="assistant", content="6CO2 + 6H2O -> C6H12O6 + 6O2"),
    ]

    formatted = chatml.format_messages(messages)
    print(formatted)

    positions = chatml.get_response_positions(messages)
    print(f"\nassistant 回答位置: {positions}")
    for start, end in positions:
        print(f"  [{start}:{end}] = '{formatted[start:end]}'")

    # 推理用 prompt
    inference_messages = [
        Message(role="system", content="你是一个有帮助的AI助手。"),
        Message(role="user", content="什么是光合作用？"),
    ]
    prompt = chatml.format_prompt(inference_messages)
    print(f"\n推理 Prompt:\n{prompt}")

    # === 2. Llama 格式 ===
    print("\n--- 2. Llama 格式 ---")
    llama = LlamaTemplate()

    formatted_llama = llama.format_messages(messages)
    print(formatted_llama)

    positions_llama = llama.get_response_positions(messages)
    print(f"\nassistant 回答位置: {positions_llama}")
    for start, end in positions_llama:
        content = formatted_llama[start:end].strip()
        print(f"  [{start}:{end}] = '{content}'")

    # === 3. Alpaca 格式 ===
    print("\n--- 3. Alpaca 格式 ---")
    alpaca = AlpacaTemplate()

    # 无 input 的情况
    messages_no_input = [
        Message(role="user", content="什么是机器学习？"),
        Message(role="assistant", content="机器学习是AI的一个分支，让计算机从数据中学习规律。"),
    ]
    formatted_alpaca = alpaca.format_messages(messages_no_input)
    print(formatted_alpaca)

    positions_alpaca = alpaca.get_response_positions(messages_no_input)
    print(f"\nresponse 位置: {positions_alpaca}")
    for start, end in positions_alpaca:
        print(f"  [{start}:{end}] = '{formatted_alpaca[start:end]}'")

    # 有 input 的情况
    print("\n--- Alpaca 格式 (带 input) ---")
    messages_with_input = [
        Message(role="user", content="将以下文本翻译成英文"),
        Message(role="input", content="今天天气很好"),
        Message(role="assistant", content="The weather is nice today."),
    ]
    formatted_alpaca_input = alpaca.format_messages(messages_with_input)
    print(formatted_alpaca_input)

    # === 4. 模板对比 ===
    print("\n--- 4. 模板格式对比 ---")
    print(f"{'模板':>10} {'特殊Token':>25} {'多轮支持':>10} {'适用模型':>15}")
    print("-" * 65)
    templates = [
        ("ChatML", "<|im_start|>/<|im_end|>", "原生支持", "Qwen/Yi"),
        ("Llama", "[INST]/[/INST]", "拼接轮次", "Llama 2/3"),
        ("Alpaca", "### Instruction:", "有限", "Alpaca"),
    ]
    for name, tokens, multi, models in templates:
        print(f"{name:>10} {tokens:>25} {multi:>10} {models:>15}")
