"""
MoE Transformer 完整模型实现

本模块将 MoE 层集成到完整的 Transformer 模型中，
实现一个可训练的 MoE 语言模型。

架构:
- Token Embedding + 位置编码
- N 个 Transformer Block（Attention + MoE FFN）
- 最终 RMSNorm + LM Head

参考:
- Shazeer et al. (2017). Outrageously Large Neural Networks.
- DeepSeek-AI (2024). DeepSeek-V2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import math


# ===== 基础组件 =====

class RMSNorm(nn.Module):
    """RMS 归一化（与 Llama/DeepSeek 一致）"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.gamma * (x / rms)


class MultiHeadAttention(nn.Module):
    """多头注意力（简化版本，支持因果掩码）"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, is_causal: bool = True
    ) -> torch.Tensor:
        B, T, D = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 注意力分数: [B, n_heads, T, T]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if is_causal:
            mask = torch.triu(
                torch.ones(T, T, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


# ===== MoE 组件 =====

class TopKRouter(nn.Module):
    """Top-K 路由器"""

    def __init__(self, d_model: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x)
        top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        return weights, top_k_indices, logits


class MoEFFN(nn.Module):
    """MoE 前馈网络层"""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        d_ff = d_ff or 4 * d_model

        self.router = TopKRouter(d_model, num_experts, top_k)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(d_ff, d_model, bias=False),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        B, T_seq, D = x.shape
        T = B * T_seq

        weights, indices, logits = self.router(x)
        flat_x = x.view(T, D)
        flat_w = weights.view(T, self.top_k)
        flat_idx = indices.view(T, self.top_k)

        output = torch.zeros(T, D, device=x.device, dtype=x.dtype)

        for k in range(self.top_k):
            eidx = flat_idx[:, k]
            ew = flat_w[:, k:k+1]
            for i in range(self.num_experts):
                mask = (eidx == i)
                if mask.any():
                    out_i = self.experts[i](flat_x[mask])
                    output[mask] += ew[mask] * out_i

        return output.view(B, T_seq, D), logits


# ===== Transformer Block =====

class MoETransformerBlock(nn.Module):
    """
    MoE Transformer Block

    结构（Pre-Norm）:
    x -> RMSNorm -> Multi-Head Attention -> 残差 ->
         RMSNorm -> MoE FFN -> 残差 -> 输出

    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        num_experts: MoE 专家数
        top_k: Top-K 路由
        d_ff: 专家 FFN 隐藏维度
        dropout: Dropout 概率
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.moe_ffn = MoEFFN(d_model, num_experts, top_k, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            logits: [batch, seq_len, num_experts] 路由 logits
        """
        # Attention 子层
        residual = x
        x = self.attn_norm(x)
        x = self.attn(x, is_causal=True)
        x = self.dropout(x) + residual

        # MoE FFN 子层
        residual = x
        x = self.ffn_norm(x)
        x, logits = self.moe_ffn(x)
        x = self.dropout(x) + residual

        return x, logits


# ===== 完整模型 =====

class MoETransformer(nn.Module):
    """
    MoE Transformer 语言模型

    将 MoE 层集成到完整的 Decoder-only Transformer 中。
    每个 Transformer Block 的 FFN 被替换为 MoE FFN。

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: Transformer 层数
        num_experts: 每层的专家数量
        top_k: 每个 token 激活的专家数
        d_ff: 专家 FFN 隐藏维度
        max_seq_len: 最大序列长度
        dropout: Dropout 概率
        aux_loss_alpha: 辅助损失系数
        aux_loss_beta: Router z-loss 系数
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: Optional[int] = None,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        aux_loss_alpha: float = 0.01,
        aux_loss_beta: float = 0.001,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.num_experts = num_experts
        self.top_k = top_k
        self.aux_loss_alpha = aux_loss_alpha
        self.aux_loss_beta = aux_loss_beta

        # Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # Transformer Blocks（带 MoE）
        self.layers = nn.ModuleList([
            MoETransformerBlock(
                d_model, n_heads, num_experts, top_k, d_ff, dropout
            )
            for _ in range(n_layers)
        ])

        # 最终归一化
        self.final_norm = RMSNorm(d_model)

        # 语言模型头
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重绑定
        self.lm_head.weight = self.token_embedding.weight

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """参数初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=self.d_model ** -0.5)

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len] 词元 ID

        Returns:
            logits: [batch, seq_len, vocab_size] 预测 logits
            aux_losses: 字典包含辅助损失信息
        """
        x = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        x = self.dropout(x)

        # 收集所有层的路由信息
        all_router_logits = []

        for layer in self.layers:
            x, router_logits = layer(x)
            all_router_logits.append(router_logits)

        x = self.final_norm(x)
        lm_logits = self.lm_head(x)

        # 计算辅助损失
        aux_losses = self._compute_aux_losses(all_router_logits)

        return lm_logits, aux_losses

    def _compute_aux_losses(
        self, all_router_logits: list
    ) -> Dict[str, torch.Tensor]:
        """
        计算所有层的辅助损失

        Args:
            all_router_logits: 列表，每个元素 [batch, seq_len, num_experts]

        Returns:
            辅助损失字典
        """
        total_lb_loss = 0.0
        total_z_loss = 0.0

        for logits in all_router_logits:
            flat_logits = logits.view(-1, self.num_experts)
            T = flat_logits.shape[0]

            # Top-K 索引
            _, top_k_idx = logits.topk(self.top_k, dim=-1)

            # 负载均衡损失
            one_hot = F.one_hot(
                top_k_idx, self.num_experts
            ).float()                          # [B, T, K, N]
            f = one_hot.sum(dim=-2).view(-1, self.num_experts).mean(dim=0)  # [N]
            P = F.softmax(flat_logits, dim=-1).mean(dim=0)                   # [N]
            lb_loss = self.num_experts * torch.sum(f * P)
            total_lb_loss += lb_loss

            # Router z-loss
            log_z = torch.logsumexp(flat_logits, dim=-1)
            z_loss = torch.mean(log_z ** 2)
            total_z_loss += z_loss

        # 对所有层取平均
        n = len(all_router_logits)
        total_lb_loss = self.aux_loss_alpha * total_lb_loss / n
        total_z_loss = self.aux_loss_beta * total_z_loss / n

        return {
            "total": total_lb_loss + total_z_loss,
            "load_balance": total_lb_loss,
            "z_loss": total_z_loss,
        }

    def count_parameters(self) -> Dict[str, int]:
        """统计模型参数量"""
        total = sum(p.numel() for p in self.parameters())
        embedding = sum(p.numel() for p in self.token_embedding.parameters())
        attention = 0
        moe = 0

        for layer in self.layers:
            attention += sum(p.numel() for p in layer.attn.parameters())
            attention += sum(p.numel() for p in layer.attn_norm.parameters())
            moe += sum(p.numel() for p in layer.moe_ffn.parameters())
            moe += sum(p.numel() for p in layer.ffn_norm.parameters())

        return {
            "total": total,
            "embedding": embedding,
            "attention": attention,
            "moe": moe,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        自回归文本生成

        Args:
            input_ids: [batch, seq_len] 输入 token ID
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: Top-K 采样

        Returns:
            [batch, seq_len + max_new_tokens] 生成的 token ID
        """
        for _ in range(max_new_tokens):
            # 前向传播
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


if __name__ == "__main__":
    # 创建 MoE Transformer 模型
    model = MoETransformer(
        vocab_size=32000,
        d_model=256,
        n_heads=8,
        n_layers=4,
        num_experts=8,
        top_k=2,
        d_ff=512,
        dropout=0.1,
    )

    # 统计参数量
    param_info = model.count_parameters()
    print("=" * 60)
    print("MoE Transformer 模型信息")
    print("=" * 60)
    print(f"总参数量: {param_info['total']:,}")
    print(f"  Embedding 参数: {param_info['embedding']:,}")
    print(f"  Attention 参数: {param_info['attention']:,}")
    print(f"  MoE 参数: {param_info['moe']:,}")

    # 前向传播测试
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, 32000, (batch_size, seq_len))

    logits, aux_losses = model(input_ids)

    print(f"\n输入形状: {input_ids.shape}")
    print(f"输出 logits 形状: {logits.shape}")
    print(f"辅助损失:")
    print(f"  总辅助损失: {aux_losses['total']:.6f}")
    print(f"  负载均衡损失: {aux_losses['load_balance']:.6f}")
    print(f"  Router z-loss: {aux_losses['z_loss']:.6f}")

    # 训练循环演示
    print(f"\n" + "=" * 60)
    print("训练循环演示（3步）")
    print("=" * 60)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    targets = torch.randint(0, 32000, (batch_size, seq_len))

    for step in range(3):
        logits, aux_losses = model(input_ids)

        # 主损失: 交叉熵
        main_loss = F.cross_entropy(
            logits.view(-1, 32000),
            targets.view(-1),
        )

        # 总损失 = 主损失 + 辅助损失
        total_loss = main_loss + aux_losses["total"]

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        print(
            f"Step {step+1}: "
            f"main_loss={main_loss.item():.4f}, "
            f"aux_loss={aux_losses['total'].item():.6f}, "
            f"total={total_loss.item():.4f}"
        )

    # 生成演示
    print(f"\n" + "=" * 60)
    print("文本生成演示")
    print("=" * 60)
    prompt = torch.randint(0, 32000, (1, 8))
    generated = model.generate(prompt, max_new_tokens=10, temperature=0.8)
    print(f"输入长度: {prompt.shape[1]}")
    print(f"生成后长度: {generated.shape[1]}")
    print(f"生成的 token IDs: {generated[0].tolist()}")
