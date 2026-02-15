"""
预训练启动脚本

使用方式:
    # 单卡 (Version A)
    python scripts/pretrain.py --config configs/model_300m.yaml

    # 多卡 (Version B)
    torchrun --nproc_per_node=4 scripts/pretrain.py --config configs/model_1b.yaml

知识依赖: 模块 8C（训练工程）、模块 9（分布式训练）
"""

import argparse
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="LLM 预训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径 (YAML)")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint 路径（断点续训）")
    parser.add_argument("--data_dir", type=str, default="./data", help="数据目录")
    parser.add_argument("--tokenizer", type=str, required=True, help="分词器模型路径")
    args = parser.parse_args()

    # 预训练主流程:
    #
    # 1. 加载配置
    #    from src.model.config import from_yaml
    #    model_config, train_config = from_yaml(args.config)
    #
    # 2. (可选) 初始化分布式环境
    #    from src.training.distributed import setup_distributed
    #    如果 YAML 中有 distributed 配置:
    #        local_rank = setup_distributed()
    #
    # 3. 创建模型
    #    from src.model.model import DecoderOnlyLM
    #    model = DecoderOnlyLM(model_config)
    #    print(f"模型参数量: {model.count_parameters():,}")
    #
    # 4. (可选) 分布式包装
    #    from src.training.distributed import wrap_model_ddp, wrap_model_fsdp
    #
    # 5. 准备数据
    #    from src.data.data_pipeline import create_dataloaders
    #    train_loader, val_loader = create_dataloaders(...)
    #
    # 6. 创建训练器并训练
    #    from src.training.trainer import Trainer
    #    trainer = Trainer(model, train_loader, val_loader, model_config, train_config)
    #    if args.resume:
    #        trainer.load_checkpoint(args.resume)
    #    trainer.train()
    #
    raise NotImplementedError(
        "TODO: 实现预训练启动脚本\n"
        "前置: 先实现 src/model/ 和 src/training/ 中的各模块"
    )


if __name__ == "__main__":
    main()
