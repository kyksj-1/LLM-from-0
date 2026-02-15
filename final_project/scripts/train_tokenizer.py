"""
训练分词器脚本

使用方式:
    python scripts/train_tokenizer.py \
        --input data/train.txt \
        --output tokenizer/my_tokenizer \
        --vocab_size 32000

知识依赖: 模块 1（分词与词汇表构建）
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="训练 BPE 分词器")
    parser.add_argument("--input", type=str, required=True, help="训练数据文件路径（纯文本）")
    parser.add_argument("--output", type=str, required=True, help="输出模型前缀")
    parser.add_argument("--vocab_size", type=int, default=32000, help="词汇表大小")
    parser.add_argument("--character_coverage", type=float, default=0.9995, help="字符覆盖率")
    parser.add_argument("--model_type", type=str, default="bpe", help="分词算法类型")
    args = parser.parse_args()

    # 确保输出目录存在
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # TODO: 调用分词器训练
    # from src.data.tokenizer import Tokenizer
    # Tokenizer.train(
    #     input_files=[args.input],
    #     model_prefix=args.output,
    #     vocab_size=args.vocab_size,
    #     character_coverage=args.character_coverage,
    #     model_type=args.model_type,
    # )
    # print(f"分词器训练完成: {args.output}.model")
    raise NotImplementedError(
        "TODO: 调用 Tokenizer.train() 训练分词器\n"
        "前置: 先实现 src/data/tokenizer.py 中的 train() 方法"
    )


if __name__ == "__main__":
    main()
