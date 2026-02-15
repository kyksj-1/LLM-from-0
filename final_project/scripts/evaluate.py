"""
模型评估脚本

使用方式:
    python scripts/evaluate.py \
        --config configs/model_300m.yaml \
        --checkpoint checkpoints/300m/final.pt \
        --tokenizer tokenizer/my_tokenizer.model \
        --eval_data data/val.bin

知识依赖: 模块 8B（困惑度）、模块 13（推理评估）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="模型评估")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--tokenizer", type=str, required=True, help="分词器路径")
    parser.add_argument("--eval_data", type=str, default=None, help="评估数据路径")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, help="评估任务列表")
    args = parser.parse_args()

    # 评估主流程:
    #
    # 1. 加载模型
    # 2. 计算困惑度（如果提供了 eval_data）
    # 3. 运行下游任务评估（如果指定了 tasks）
    # 4. 打印评估结果
    #
    # 推荐: 使用 lm-eval-harness 进行标准评估
    # pip install lm-eval
    # lm_eval --model hf --model_args pretrained=./checkpoints/300m/final \
    #         --tasks hellaswag,arc_easy --batch_size 8
    #
    raise NotImplementedError(
        "TODO: 实现评估脚本\n"
        "前置: 先完成训练流程"
    )


if __name__ == "__main__":
    main()
