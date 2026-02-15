"""
GQA 注意力机制 + RoPE 旋转位置编码

知识依赖:
- 模块 3（Transformer 核心架构）: 标准 Scaled Dot-Product Attention
- 模块 5（注意力变体）: MHA → MQA → GQA 的演进，KV Cache 原理
- 模块 2（嵌入与位置编码）: RoPE 旋转位置编码的数学推导

参考实现:
- code/decoder_only/model.py 中的 Attention 类
- code/attention_variants/gqa.py
- code/embedding/rope.py

核心数学:

1. Scaled Dot-Product Attention:
   Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

2. GQA (Grouped-Query Attention):
   将 n_heads 个 Q 头分成 n_kv_heads 组，每组共享一对 KV 头。
   GQA(4 KV heads, 16 Q heads) → 每个 KV 头服务 4 个 Q 头。
   KV Cache 大小减少为 MHA 的 n_kv_heads / n_heads。

3. RoPE (Rotary Position Embedding):
   对 Q, K 向量的每对相邻维度施加旋转:
   [q_{2i}, q_{2i+1}] @ [[cos(m*theta_i), -sin(m*theta_i)],
                          [sin(m*theta_i),  cos(m*theta_i)]]
   其中 theta_i = 1 / (base ^ (2i/d))，m 是位置索引。
   旋转后 Q @ K^T 自然编码了相对位置信息。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# 假定从同包导入配置
from .config import ModelConfig


class RotaryPositionEmbedding(nn.Module):
    """
    RoPE 旋转位置编码

    预计算 cos 和 sin 频率表，在前向传播时对 Q/K 施加旋转。

    数学:
        theta_i = 1 / (base ^ (2i / d_head))  对 i = 0, 1, ..., d_head/2 - 1
        cos_cache[m, i] = cos(m * theta_i)
        sin_cache[m, i] = sin(m * theta_i)

    Args:
        head_dim: 每个注意力头的维度
        max_seq_len: 最大序列长度（决定预计算表的大小）
        base: 基底频率（默认 10000.0）
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10000.0):
        super().__init__()
        # TODO: 预计算 cos 和 sin 频率表
        # 提示:
        #   1. 计算频率向量 theta: shape [head_dim // 2]
        #   2. 生成位置索引 m: shape [max_seq_len]
        #   3. 外积得到角度表 m * theta: shape [max_seq_len, head_dim // 2]
        #   4. 分别计算 cos 和 sin，用 register_buffer 缓存
        raise NotImplementedError(
            "TODO: 预计算 RoPE 频率表\n"
            "参考: 模块 2 (嵌入与位置编码) 的 RoPE 章节\n"
            "参考实现: code/embedding/rope.py"
        )

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对 Q 和 K 施加旋转位置编码

        Args:
            q: Query 张量 [batch, n_heads, seq_len, head_dim]
            k: Key 张量 [batch, n_kv_heads, seq_len, head_dim]
            position_ids: 位置索引 [batch, seq_len]，None 时使用 0..seq_len-1

        Returns:
            (q_rotated, k_rotated): 旋转后的 Q, K，形状不变

        实现步骤:
            1. 将 q, k 拆成偶数和奇数维度对: [..., d//2, 2]
            2. 查表获取对应位置的 cos, sin
            3. 对每对维度施加 2D 旋转矩阵
            4. 恢复原始形状
        """
        raise NotImplementedError(
            "TODO: 对 Q, K 施加 RoPE 旋转\n"
            "参考: code/embedding/rope.py 中的 apply_rotary_emb()"
        )


class GQAAttention(nn.Module):
    """
    Grouped-Query Attention

    核心: n_heads 个 Q 头分成 n_kv_heads 组，每组共享一对 KV 投影。
    当 n_kv_heads == n_heads 时退化为标准 MHA。
    当 n_kv_heads == 1 时退化为 MQA。

    Args:
        config: 模型配置
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # TODO: 定义以下线性投影层（均无 bias）
        #   - W_q: [d_model, n_heads * head_dim]
        #   - W_k: [d_model, n_kv_heads * head_dim]
        #   - W_v: [d_model, n_kv_heads * head_dim]
        #   - W_o: [n_heads * head_dim, d_model]
        # 以及 RoPE 实例
        raise NotImplementedError(
            "TODO: 初始化 GQA 注意力层\n"
            "参考: 模块 5 (注意力变体) 的 GQA 章节\n"
            "参考实现: code/attention_variants/gqa.py"
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        GQA 前向传播

        Args:
            x: 输入 [batch, seq_len, d_model]
            mask: 因果掩码 [seq_len, seq_len] 或 None
            kv_cache: 推理时的 KV Cache (k_cache, v_cache)，训练时为 None

        Returns:
            (output, new_kv_cache):
                output: [batch, seq_len, d_model]
                new_kv_cache: 更新后的 KV Cache（训练时为 None）

        实现步骤:
            1. 计算 Q, K, V 投影
            2. reshape 为 [batch, n_heads/n_kv_heads, seq_len, head_dim]
            3. 对 Q, K 施加 RoPE
            4. 如果有 kv_cache，拼接历史 KV
            5. 将 KV 头广播到与 Q 头匹配（GQA 的关键步骤）
            6. 计算注意力分数: Q @ K^T / sqrt(d_k)
            7. 应用因果掩码
            8. Softmax + 加权求和
            9. 输出投影
        """
        raise NotImplementedError(
            "TODO: 实现 GQA 前向传播\n"
            "关键: 步骤 5 中 KV 头的广播（repeat_interleave 或 expand）\n"
            "参考: 模块 5 的 code/attention_variants/gqa.py"
        )
