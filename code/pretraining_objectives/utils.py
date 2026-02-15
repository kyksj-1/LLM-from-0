"""
工具函数: 预训练目标模块的公共工具

本模块提供了预训练目标实现中常用的工具函数,
包括因果掩码生成、序列填充、简单分词器等。

参考:
- 本模块服务于 ntp_loss.py, mlm_loss.py, fim_loss.py, perplexity.py
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple


def create_causal_mask(seq_len: int, device: torch.device = None) -> torch.Tensor:
    """
    创建因果掩码 (上三角为 True, 表示需要被遮蔽的位置)

    Args:
        seq_len: 序列长度
        device: 设备

    Returns:
        [seq_len, seq_len] 的 bool 张量, True 表示需要被遮蔽
    """
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return mask


def create_prefix_mask(
    seq_len: int, prefix_len: int, device: torch.device = None
) -> torch.Tensor:
    """
    创建 Prefix LM 的注意力掩码

    前缀部分: 双向注意力 (全部可见)
    后缀部分: 因果注意力 (只能看到自身及之前的位置)

    Args:
        seq_len: 总序列长度
        prefix_len: 前缀长度
        device: 设备

    Returns:
        [seq_len, seq_len] 的 bool 张量, True 表示需要被遮蔽
    """
    # 初始化全 False (全部可见)
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    # 后缀部分使用因果掩码
    if prefix_len < seq_len:
        suffix_len = seq_len - prefix_len
        # 后缀行、后缀列: 因果掩码
        causal_part = torch.triu(
            torch.ones(suffix_len, suffix_len, dtype=torch.bool, device=device),
            diagonal=1,
        )
        mask[prefix_len:, prefix_len:] = causal_part

    return mask


class CharTokenizer:
    """
    字符级分词器 (用于教学演示)

    将文本按字符分词, 每个字符对应一个 token id。
    在实际中, LLM 使用 BPE/WordPiece 等子词分词器,
    但字符级分词器足够简单, 适合理解预训练目标的核心逻辑。
    """

    def __init__(self, text: str = ""):
        """
        Args:
            text: 用于构建词汇表的文本
        """
        # 特殊 token
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.mask_token = "<MASK>"
        self.pre_token = "<PRE>"
        self.suf_token = "<SUF>"
        self.mid_token = "<MID>"

        special_tokens = [
            self.pad_token, self.bos_token, self.eos_token,
            self.mask_token, self.pre_token, self.suf_token, self.mid_token,
        ]

        # 构建词汇表
        chars = sorted(set(text))
        all_tokens = special_tokens + chars

        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}
        self.vocab_size = len(all_tokens)

        # 特殊 token 的 id
        self.pad_id = self.token_to_id[self.pad_token]
        self.bos_id = self.token_to_id[self.bos_token]
        self.eos_id = self.token_to_id[self.eos_token]
        self.mask_id = self.token_to_id[self.mask_token]
        self.pre_id = self.token_to_id[self.pre_token]
        self.suf_id = self.token_to_id[self.suf_token]
        self.mid_id = self.token_to_id[self.mid_token]

    def encode(self, text: str) -> List[int]:
        """将文本编码为 token id 列表"""
        return [self.token_to_id.get(c, self.pad_id) for c in text]

    def decode(self, ids: List[int]) -> str:
        """将 token id 列表解码为文本"""
        tokens = [self.id_to_token.get(i, "") for i in ids]
        # 过滤特殊 token
        special = {self.pad_token, self.bos_token, self.eos_token,
                   self.mask_token, self.pre_token, self.suf_token, self.mid_token}
        return "".join(t for t in tokens if t not in special)

    def __len__(self) -> int:
        return self.vocab_size


def pad_sequences(
    sequences: List[List[int]],
    pad_value: int = 0,
    max_len: Optional[int] = None,
) -> torch.Tensor:
    """
    将不等长序列填充到相同长度

    Args:
        sequences: token id 序列列表
        pad_value: 填充值
        max_len: 最大长度 (None 则自动取最长序列的长度)

    Returns:
        [batch, max_len] 的 LongTensor
    """
    if max_len is None:
        max_len = max(len(s) for s in sequences)

    padded = []
    for seq in sequences:
        if len(seq) >= max_len:
            padded.append(seq[:max_len])
        else:
            padded.append(seq + [pad_value] * (max_len - len(seq)))

    return torch.tensor(padded, dtype=torch.long)


class SimpleTransformerLM(nn.Module):
    """
    简单的 Transformer 语言模型 (用于教学演示)

    结构: Embedding -> Transformer Encoder (因果掩码) -> Linear Head

    Args:
        vocab_size: 词汇表大小
        d_model: 隐藏维度
        n_heads: 注意力头数
        n_layers: Transformer 层数
        max_seq_len: 最大序列长度
        dropout: Dropout 比率
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-Norm (现代 LLM 标准)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """标准权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            input_ids: [batch, seq_len] 输入 token id
            attention_mask: [seq_len, seq_len] 注意力掩码 (True 表示被遮蔽)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Token + Position Embedding
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        h = self.token_embed(input_ids) + self.pos_embed(positions)
        h = self.dropout(h)

        # 因果掩码
        if attention_mask is None:
            attention_mask = create_causal_mask(seq_len, device=device)

        # Transformer 编码
        h = self.transformer(h, mask=attention_mask, is_causal=(attention_mask is None))
        h = self.ln_f(h)

        # 输出 logits
        logits = self.head(h)
        return logits


def compute_num_parameters(model: nn.Module) -> Dict[str, int]:
    """
    计算模型参数量

    Args:
        model: PyTorch 模型

    Returns:
        包含总参数量和可训练参数量的字典
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("预训练目标工具函数演示")
    print("=" * 60)

    # 1. 字符级分词器
    text = "Hello, World! This is a test."
    tokenizer = CharTokenizer(text)
    print(f"\n词汇表大小: {tokenizer.vocab_size}")
    encoded = tokenizer.encode("Hello")
    decoded = tokenizer.decode(encoded)
    print(f"编码 'Hello' -> {encoded}")
    print(f"解码 {encoded} -> '{decoded}'")

    # 2. 因果掩码
    mask = create_causal_mask(5)
    print(f"\n因果掩码 (5x5):\n{mask.int()}")

    # 3. Prefix LM 掩码
    prefix_mask = create_prefix_mask(6, prefix_len=3)
    print(f"\nPrefix LM 掩码 (prefix=3, total=6):\n{prefix_mask.int()}")

    # 4. 序列填充
    seqs = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
    padded = pad_sequences(seqs, pad_value=0)
    print(f"\n填充序列:\n{padded}")

    # 5. 简单模型
    model = SimpleTransformerLM(vocab_size=tokenizer.vocab_size)
    params = compute_num_parameters(model)
    print(f"\n模型参数量: {params['total']:,}")

    # 前向传播测试
    input_ids = torch.tensor([encoded])
    logits = model(input_ids)
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {logits.shape}")
