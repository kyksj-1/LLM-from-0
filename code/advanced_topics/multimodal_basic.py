"""
多模态 LLM 基础架构: 视觉编码器 + 投影层 + LLM

本模块实现了 Vision-Language Model (VLM) 的简化版本，
展示多模态 LLM 的核心架构设计。

架构概述:
1. 视觉编码器 (Vision Encoder): 将图像转化为视觉 token 序列
2. 投影层 (Projection Layer): 将视觉特征对齐到语言模型的嵌入空间
3. LLM 主干 (Language Model): 处理视觉 + 文本的混合 token 序列

数学框架:
- 视觉特征提取: V = ViT(image) in R^{n_patches x d_vision}
- 视觉-语言对齐: V' = Projection(V) in R^{n_patches x d_model}
- 混合序列: [V'; text_tokens] -> LLM -> response

两种主流架构:
- LLaVA 式: 冻结视觉编码器 + 可训练投影层 + 微调 LLM
- Gemini 式: 原生多模态，所有模态统一训练

参考:
- Liu et al. (2023). Visual Instruction Tuning (LLaVA).
- Alayrac et al. (2022). Flamingo: a Visual Language Model.
- Google (2023). Gemini: A Family of Highly Capable Multimodal Models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math


class PatchEmbedding(nn.Module):
    """
    图像分块嵌入 (Patch Embedding)

    将图像分割为固定大小的 patch 并线性投影到嵌入空间。
    这是 Vision Transformer (ViT) 的第一步。

    输入: [batch, channels, height, width]
    输出: [batch, n_patches, d_vision]

    其中 n_patches = (H / patch_size) * (W / patch_size)

    Args:
        image_size: 输入图像大小（假设正方形）
        patch_size: patch 大小
        in_channels: 输入通道数（RGB = 3）
        d_vision: 视觉嵌入维度
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_vision: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2

        # 使用卷积实现分块 + 线性投影
        self.projection = nn.Conv2d(
            in_channels, d_vision,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # CLS token (可选，用于分类任务)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_vision))

        # 位置编码
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, d_vision)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, channels, height, width]

        Returns:
            patch_embeddings: [batch, n_patches + 1, d_vision]
        """
        batch_size = images.shape[0]

        # 分块投影: [batch, d_vision, h/p, w/p]
        x = self.projection(images)

        # 展平为序列: [batch, d_vision, n_patches] -> [batch, n_patches, d_vision]
        x = x.flatten(2).transpose(1, 2)

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = x + self.position_embedding

        return x


class SimpleViTEncoder(nn.Module):
    """
    简化的 Vision Transformer 编码器

    包含多层 Transformer 块，对图像 patch 序列进行编码。

    Args:
        d_vision: 视觉嵌入维度
        n_heads: 注意力头数
        n_layers: 编码器层数
        d_ff: FFN 中间维度
        dropout: Dropout 率
    """

    def __init__(
        self,
        d_vision: int = 768,
        n_heads: int = 12,
        n_layers: int = 6,
        d_ff: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "ln1": nn.LayerNorm(d_vision),
                "attn": nn.MultiheadAttention(
                    d_vision, n_heads,
                    dropout=dropout, batch_first=True,
                ),
                "ln2": nn.LayerNorm(d_vision),
                "ff": nn.Sequential(
                    nn.Linear(d_vision, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_vision),
                    nn.Dropout(dropout),
                ),
            })
            self.layers.append(layer)

        self.final_ln = nn.LayerNorm(d_vision)

    def forward(
        self, x: torch.Tensor, return_all_layers: bool = False
    ) -> torch.Tensor:
        """
        Args:
            x: patch 嵌入序列 [batch, n_patches + 1, d_vision]
            return_all_layers: 是否返回所有层的输出

        Returns:
            encoded: 编码后的视觉特征 [batch, n_patches + 1, d_vision]
        """
        all_outputs = []

        for layer in self.layers:
            # 注意力子层 (Pre-Norm)
            ln_out = layer["ln1"](x)
            attn_out, _ = layer["attn"](ln_out, ln_out, ln_out)
            x = x + attn_out

            # FFN 子层 (Pre-Norm)
            ln_out = layer["ln2"](x)
            ff_out = layer["ff"](ln_out)
            x = x + ff_out

            if return_all_layers:
                all_outputs.append(x)

        x = self.final_ln(x)

        if return_all_layers:
            return x, all_outputs

        return x


class VisionLanguageProjector(nn.Module):
    """
    视觉-语言投影层

    将视觉编码器的输出对齐到 LLM 的嵌入空间。
    这是多模态融合的关键桥梁。

    支持多种投影方式:
    1. Linear: 简单线性投影 (LLaVA v1)
    2. MLP: 两层 MLP + GELU (LLaVA v1.5)
    3. Cross-Attention: 使用可学习查询向量 (Flamingo 风格)

    Args:
        d_vision: 视觉特征维度
        d_model: LLM 嵌入维度
        projector_type: 投影方式 ("linear", "mlp", "cross_attention")
        n_query_tokens: Cross-Attention 模式的查询 token 数
    """

    def __init__(
        self,
        d_vision: int = 768,
        d_model: int = 512,
        projector_type: str = "mlp",
        n_query_tokens: int = 64,
    ):
        super().__init__()
        self.projector_type = projector_type

        if projector_type == "linear":
            self.projector = nn.Linear(d_vision, d_model)

        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(d_vision, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )

        elif projector_type == "cross_attention":
            self.query_tokens = nn.Parameter(
                torch.randn(1, n_query_tokens, d_model)
            )
            self.cross_attn = nn.MultiheadAttention(
                d_model, num_heads=8,
                kdim=d_vision, vdim=d_vision,
                batch_first=True,
            )
            self.ln = nn.LayerNorm(d_model)
        else:
            raise ValueError(f"未知投影类型: {projector_type}")

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: 视觉编码器输出 [batch, n_vis_tokens, d_vision]

        Returns:
            projected: 投影后的视觉 token [batch, n_tokens, d_model]
        """
        if self.projector_type in ("linear", "mlp"):
            return self.projector(visual_features)

        elif self.projector_type == "cross_attention":
            batch_size = visual_features.shape[0]
            queries = self.query_tokens.expand(batch_size, -1, -1)
            attn_out, _ = self.cross_attn(
                queries, visual_features, visual_features
            )
            return self.ln(attn_out)

        return visual_features


class SimpleLLMDecoder(nn.Module):
    """
    简化的 LLM 解码器

    接收混合的视觉 + 文本 token 序列，进行自回归生成。

    Args:
        vocab_size: 词汇表大小
        d_model: 模型维度
        n_heads: 注意力头数
        n_layers: 解码器层数
        d_ff: FFN 中间维度
        max_seq_len: 最大序列长度
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn": nn.MultiheadAttention(
                    d_model, n_heads, batch_first=True,
                ),
                "ln2": nn.LayerNorm(d_model),
                "ff": nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Linear(d_ff, d_model),
                ),
            })
            self.layers.append(layer)

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            inputs_embeds: 输入嵌入 [batch, seq_len, d_model]
            attention_mask: 注意力掩码 [batch, seq_len]

        Returns:
            logits: 输出 logits [batch, seq_len, vocab_size]
        """
        seq_len = inputs_embeds.shape[1]
        device = inputs_embeds.device

        # 添加位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_embeds = self.position_embedding(positions)
        x = inputs_embeds + pos_embeds

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

        for layer in self.layers:
            # 注意力
            ln_out = layer["ln1"](x)
            attn_out, _ = layer["attn"](
                ln_out, ln_out, ln_out,
                attn_mask=causal_mask,
            )
            x = x + attn_out

            # FFN
            ln_out = layer["ln2"](x)
            ff_out = layer["ff"](ln_out)
            x = x + ff_out

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """将 token ID 转换为嵌入"""
        return self.token_embedding(token_ids)


class VisionLanguageModel(nn.Module):
    """
    完整的 Vision-Language Model (VLM)

    整合视觉编码器、投影层和 LLM，实现多模态理解与生成。

    架构流程:
    1. 图像 -> PatchEmbedding -> ViT 编码器 -> 视觉特征
    2. 视觉特征 -> 投影层 -> 视觉 token (对齐到 LLM 空间)
    3. [视觉 token; 文本 token] -> LLM 解码器 -> 生成回复

    Args:
        image_size: 输入图像大小
        patch_size: patch 大小
        d_vision: 视觉编码器维度
        d_model: LLM 维度
        vocab_size: 词汇表大小
        n_vision_layers: 视觉编码器层数
        n_llm_layers: LLM 解码器层数
        projector_type: 投影层类型
        freeze_vision: 是否冻结视觉编码器
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        d_vision: int = 768,
        d_model: int = 512,
        vocab_size: int = 32000,
        n_vision_layers: int = 6,
        n_llm_layers: int = 6,
        projector_type: str = "mlp",
        freeze_vision: bool = True,
    ):
        super().__init__()

        # 视觉编码器
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            d_vision=d_vision,
        )
        self.vision_encoder = SimpleViTEncoder(
            d_vision=d_vision,
            n_layers=n_vision_layers,
        )

        # 投影层
        self.projector = VisionLanguageProjector(
            d_vision=d_vision,
            d_model=d_model,
            projector_type=projector_type,
        )

        # LLM 解码器
        self.llm = SimpleLLMDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_llm_layers,
        )

        # 冻结视觉编码器
        if freeze_vision:
            self._freeze_vision_encoder()

        self.d_model = d_model

    def _freeze_vision_encoder(self):
        """冻结视觉编码器的参数"""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        编码图像为视觉 token

        Args:
            images: [batch, channels, height, width]

        Returns:
            visual_tokens: [batch, n_vis_tokens, d_model]
        """
        # 分块嵌入
        patch_embeds = self.patch_embed(images)

        # ViT 编码
        visual_features = self.vision_encoder(patch_embeds)

        # 投影到 LLM 空间
        visual_tokens = self.projector(visual_features)

        return visual_tokens

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            images: 输入图像 [batch, channels, H, W]，可选
            text_ids: 文本 token ID [batch, text_len]，可选
            labels: 标签 token ID [batch, text_len]，用于计算损失

        Returns:
            outputs: 包含 logits 和 loss 的字典
        """
        batch_size = (
            images.shape[0] if images is not None
            else text_ids.shape[0]
        )
        device = (
            images.device if images is not None
            else text_ids.device
        )

        # 准备嵌入序列
        embeds_list = []

        # 编码图像
        if images is not None:
            visual_tokens = self.encode_image(images)
            embeds_list.append(visual_tokens)

        # 编码文本
        if text_ids is not None:
            text_embeds = self.llm.embed_tokens(text_ids)
            embeds_list.append(text_embeds)

        # 拼接: [视觉 token; 文本 token]
        if embeds_list:
            combined_embeds = torch.cat(embeds_list, dim=1)
        else:
            raise ValueError("必须提供 images 或 text_ids 中的至少一个")

        # LLM 解码
        logits = self.llm(combined_embeds)

        outputs = {"logits": logits}

        # 计算损失（仅对文本部分）
        if labels is not None and images is not None:
            # 视觉 token 不参与损失计算
            n_vis = visual_tokens.shape[1]
            text_logits = logits[:, n_vis:, :]

            # 移位预测: 用 token_t 预测 token_{t+1}
            shift_logits = text_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            outputs["loss"] = loss

        return outputs

    def count_parameters(self) -> Dict[str, int]:
        """统计各组件的参数量"""
        vision_params = sum(
            p.numel() for p in self.patch_embed.parameters()
        ) + sum(
            p.numel() for p in self.vision_encoder.parameters()
        )
        projector_params = sum(
            p.numel() for p in self.projector.parameters()
        )
        llm_params = sum(p.numel() for p in self.llm.parameters())

        trainable = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )

        return {
            "vision_encoder": vision_params,
            "projector": projector_params,
            "llm_decoder": llm_params,
            "total": vision_params + projector_params + llm_params,
            "trainable": trainable,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("多模态 LLM 基础架构演示")
    print("=" * 60)

    # ---- 1. 创建 VLM ----
    print("\n[1] 创建 Vision-Language Model...")
    vlm = VisionLanguageModel(
        image_size=224,
        patch_size=16,
        d_vision=256,        # 简化版，实际一般 768 或 1024
        d_model=256,         # 简化版，实际一般 4096+
        vocab_size=1000,     # 简化版
        n_vision_layers=4,   # 简化版，实际一般 24+
        n_llm_layers=4,      # 简化版，实际一般 32+
        projector_type="mlp",
        freeze_vision=True,
    )

    param_stats = vlm.count_parameters()
    print(f"    参数量统计:")
    for name, count in param_stats.items():
        print(f"      {name}: {count:,}")

    # ---- 2. 前向传播测试 ----
    print("\n[2] 前向传播测试...")
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224)
    text_ids = torch.randint(0, 1000, (batch_size, 32))
    labels = torch.randint(0, 1000, (batch_size, 32))

    with torch.no_grad():
        outputs = vlm(images=images, text_ids=text_ids, labels=labels)

    print(f"    输入图像: {images.shape}")
    print(f"    输入文本: {text_ids.shape}")
    print(f"    输出 logits: {outputs['logits'].shape}")
    print(f"    损失值: {outputs['loss'].item():.4f}")

    # ---- 3. 视觉编码测试 ----
    print("\n[3] 视觉编码测试...")
    with torch.no_grad():
        visual_tokens = vlm.encode_image(images)
    print(f"    视觉 token 形状: {visual_tokens.shape}")
    print(f"    n_patches = (224/16)^2 + 1(CLS) = {(224//16)**2 + 1}")

    # ---- 4. 投影层对比 ----
    print("\n[4] 不同投影层类型对比...")
    for proj_type in ["linear", "mlp", "cross_attention"]:
        projector = VisionLanguageProjector(
            d_vision=256, d_model=256,
            projector_type=proj_type,
        )
        n_params = sum(p.numel() for p in projector.parameters())
        dummy_input = torch.randn(1, 197, 256)  # 196 patches + 1 CLS
        with torch.no_grad():
            output = projector(dummy_input)
        print(f"    {proj_type:20s}: "
              f"输入={dummy_input.shape}, "
              f"输出={output.shape}, "
              f"参数={n_params:,}")

    # ---- 5. 纯文本模式 ----
    print("\n[5] 纯文本模式（无图像输入）...")
    with torch.no_grad():
        text_only = vlm(text_ids=text_ids)
    print(f"    输出 logits: {text_only['logits'].shape}")

    print("\n演示完成!")
