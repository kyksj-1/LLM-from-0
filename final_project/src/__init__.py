"""
终极项目: 从零训练一个完整 LLM

本包包含完整的 LLM 训练框架，涵盖:
- model/: 模型架构（GQA Attention + SwiGLU FFN + RoPE + RMSNorm）
- data/: 数据处理（分词器、数据集、数据管线）
- training/: 训练流程（预训练、SFT、DPO、分布式）
- inference/: 推理部署（KV Cache、量化、服务）
- evaluation/: 评估框架（困惑度、下游任务）

项目提供两个版本:
- Version A: 300M 参数（单卡 24GB GPU）
- Version B: 1B 参数（4-8 卡 GPU）
"""
