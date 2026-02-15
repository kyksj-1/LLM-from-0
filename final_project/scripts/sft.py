"""
SFT 指令微调启动脚本

使用方式:
    python scripts/sft.py \
        --config configs/model_300m.yaml \
        --pretrain_ckpt checkpoints/300m/step_50000.pt \
        --data sft_data.json \
        --tokenizer tokenizer/my_tokenizer.model \
        --use_lora

知识依赖: 模块 10（SFT 监督微调）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="SFT 指令微调")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--pretrain_ckpt", type=str, required=True, help="预训练 checkpoint 路径")
    parser.add_argument("--data", type=str, required=True, help="SFT 数据路径 (JSON)")
    parser.add_argument("--tokenizer", type=str, required=True, help="分词器路径")
    parser.add_argument("--use_lora", action="store_true", help="使用 LoRA")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA 秩")
    args = parser.parse_args()

    # SFT 主流程:
    #
    # 1. 加载配置和预训练模型
    # 2. (可选) 注入 LoRA 适配器
    # 3. 加载 SFT 数据
    # 4. 创建 SFTTrainer 并训练
    # 5. (可选) 合并 LoRA 权重并保存
    #
    raise NotImplementedError(
        "TODO: 实现 SFT 启动脚本\n"
        "前置: 先完成预训练阶段，再进行 SFT"
    )


if __name__ == "__main__":
    main()
