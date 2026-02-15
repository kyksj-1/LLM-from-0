"""
简单推理服务

知识依赖:
- 模块 14（推理加速）: 推理部署

本模块提供一个极简的 HTTP 推理服务，用于测试训练好的模型。
生产环境推荐使用 vLLM 或 TGI。

使用方式:
    python -m scripts.serve --model_path checkpoints/300m/final --port 8000

    curl -X POST http://localhost:8000/generate \\
         -H "Content-Type: application/json" \\
         -d '{"prompt": "你好", "max_tokens": 100}'
"""

import json
from typing import Optional


def load_model_for_serving(
    model_path: str,
    tokenizer_path: str,
    device: str = "cuda",
    quantize: Optional[str] = None,
):
    """
    加载模型用于推理服务

    Args:
        model_path: 模型 checkpoint 路径
        tokenizer_path: 分词器路径
        device: 推理设备
        quantize: 量化方式 ("int8" / "int4" / None)

    Returns:
        (model, tokenizer)

    实现步骤:
        1. 加载模型配置
        2. 初始化模型架构
        3. 加载 checkpoint 权重
        4. 如果 quantize: 应用量化
        5. model.eval()
        6. 加载分词器
    """
    raise NotImplementedError(
        "TODO: 实现模型加载\n"
        "参考: 模块 14 的部署章节"
    )


def handle_generate_request(
    request_body: dict,
    model,
    tokenizer,
) -> dict:
    """
    处理生成请求

    Args:
        request_body: {
            "prompt": str,
            "max_tokens": int (默认 128),
            "temperature": float (默认 0.8),
            "top_p": float (默认 0.9),
        }
        model: 语言模型
        tokenizer: 分词器

    Returns:
        {"text": str, "tokens_generated": int}
    """
    raise NotImplementedError(
        "TODO: 实现请求处理\n"
        "提示: 调用 model/generation.py 中的 TextGenerator"
    )


def start_server(model_path: str, tokenizer_path: str, port: int = 8000):
    """
    启动 HTTP 推理服务

    使用 Python 内置的 http.server 模块实现最简服务。
    生产环境请使用 FastAPI + uvicorn 或直接使用 vLLM。

    Args:
        model_path: 模型路径
        tokenizer_path: 分词器路径
        port: 服务端口
    """
    raise NotImplementedError(
        "TODO: 实现简单 HTTP 服务\n"
        "提示: 可以使用 http.server.HTTPServer 或 flask/fastapi\n"
        "生产建议: 直接使用 vLLM serve（参考模块 14 的 vLLM 章节）"
    )
