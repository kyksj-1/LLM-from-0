"""
模型导出脚本

使用方式:
    # 导出为 HuggingFace 格式
    python scripts/export.py \
        --checkpoint checkpoints/300m/final.pt \
        --output_dir exported_model/ \
        --format huggingface

    # 导出量化版本
    python scripts/export.py \
        --checkpoint checkpoints/300m/final.pt \
        --output_dir exported_model_int8/ \
        --quantize int8

知识依赖: 模块 14（推理加速 / 量化与部署）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="模型导出")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--tokenizer", type=str, required=True, help="分词器路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--format", type=str, default="pytorch", choices=["pytorch", "huggingface"],
                        help="导出格式")
    parser.add_argument("--quantize", type=str, default=None, choices=["int8", "int4"],
                        help="量化方式")
    args = parser.parse_args()

    # 导出主流程:
    #
    # 1. 加载模型
    # 2. (可选) 量化
    # 3. 根据 format 导出:
    #    - pytorch: 直接保存 state_dict
    #    - huggingface: 转换为 HF 格式（config.json + model.safetensors）
    # 4. 复制分词器到输出目录
    #
    raise NotImplementedError(
        "TODO: 实现模型导出\n"
        "前置: 完成训练后导出最终模型"
    )


if __name__ == "__main__":
    main()
