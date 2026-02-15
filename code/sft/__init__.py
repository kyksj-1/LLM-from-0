"""
SFT (Supervised Fine-Tuning) 模块

本包实现了监督微调的核心组件：
- dataset: 指令微调数据集处理
- sft_trainer: SFT 训练器
- lora: LoRA 从零实现
- qlora: QLoRA 实现（NF4 量化 + 双量化）
- chat_template: 对话模板处理
- merge_lora: LoRA 权重合并
- utils: 工具函数
"""

from .lora import LoRALinear, apply_lora_to_model
from .chat_template import ChatTemplate, ChatMLTemplate, LlamaTemplate
from .dataset import InstructDataset
from .utils import count_parameters, print_trainable_parameters
