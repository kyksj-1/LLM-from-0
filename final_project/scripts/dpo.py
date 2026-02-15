"""
DPO 偏好优化启动脚本

使用方式:
    python scripts/dpo.py \
        --config configs/model_300m.yaml \
        --sft_ckpt checkpoints/300m/sft_final.pt \
        --data dpo_data.json \
        --tokenizer tokenizer/my_tokenizer.model

知识依赖: 模块 12（DPO 及变体）
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="DPO 偏好优化")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--sft_ckpt", type=str, required=True, help="SFT checkpoint 路径")
    parser.add_argument("--data", type=str, required=True, help="偏好数据路径 (JSON)")
    parser.add_argument("--tokenizer", type=str, required=True, help="分词器路径")
    parser.add_argument("--beta", type=float, default=None, help="DPO β 参数（覆盖配置值）")
    args = parser.parse_args()

    # DPO 主流程:
    #
    # 1. 加载配置和 SFT 模型
    # 2. 创建参考模型（SFT 模型的冻结副本）
    #    import copy
    #    ref_model = copy.deepcopy(model)
    #    ref_model.eval()
    #    ref_model.requires_grad_(False)
    # 3. 加载偏好数据
    # 4. 创建 DPOTrainer 并训练
    # 5. 保存最终模型
    #
    raise NotImplementedError(
        "TODO: 实现 DPO 启动脚本\n"
        "前置: 先完成 SFT 阶段，再进行 DPO"
    )


if __name__ == "__main__":
    main()
