"""
激活值干预实验: 零化 (Ablation)、替换 (Patching)、因果分析

本模块实现了机械可解释性中的激活值干预方法。
通过干预模型内部的激活值来建立因果关系，
从而理解特定组件（注意力头、MLP层、特征方向）对模型行为的贡献。

核心方法:
1. 零化 (Ablation): 将某个组件的输出设为零，观察性能变化
2. 均值消融 (Mean Ablation): 用平均激活值替换，控制基线方差
3. 激活替换 (Activation Patching): 用另一个输入的激活值替换
4. 因果追踪 (Causal Tracing): 系统性地定位关键组件

数学框架:
- 干预效果 = metric(原始) - metric(干预后)
- 正效果: 该组件对输出有正面贡献
- 负效果: 该组件对输出有负面贡献
- 直接效果 vs 间接效果 (通过下游组件传递)

参考:
- Vig et al. (2020). Causal Mediation Analysis for Interpreting Neural NLP.
- Meng et al. (2022). Locating and Editing Factual Associations in GPT.
- Conmy et al. (2023). Towards Automated Circuit Discovery.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import copy


@dataclass
class PatchingResult:
    """
    干预实验结果

    Attributes:
        component_name: 被干预的组件名称
        layer_idx: 层索引
        head_idx: 注意力头索引（如适用）
        original_metric: 干预前的指标值
        patched_metric: 干预后的指标值
        effect: 干预效果 (original - patched)
        normalized_effect: 归一化效果 (effect / original)
    """
    component_name: str
    layer_idx: int
    head_idx: Optional[int] = None
    original_metric: float = 0.0
    patched_metric: float = 0.0
    effect: float = 0.0
    normalized_effect: float = 0.0


class SimpleTransformerForPatching(nn.Module):
    """
    简化的 Transformer 模型（用于激活干预实验）

    这是一个教学用的简化模型，保留了关键的结构（残差流、注意力头、MLP），
    同时暴露了所有中间激活值以便干预。

    Args:
        vocab_size: 词汇表大小
        d_model: 隐藏层维度
        n_heads: 注意力头数
        n_layers: 层数
        d_ff: FFN 中间维度
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_head = d_model // n_heads

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(512, d_model)

        # Transformer 层
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.ModuleDict({
                "ln1": nn.LayerNorm(d_model),
                "attn_qkv": nn.Linear(d_model, 3 * d_model),
                "attn_out": nn.Linear(d_model, d_model),
                "ln2": nn.LayerNorm(d_model),
                "ff_up": nn.Linear(d_model, d_ff),
                "ff_down": nn.Linear(d_ff, d_model),
            })
            self.layers.append(layer)

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 存储中间激活值（用于干预）
        self.activations: Dict[str, torch.Tensor] = {}

    def forward(
        self,
        input_ids: torch.Tensor,
        hooks: Optional[Dict[str, Callable]] = None,
    ) -> torch.Tensor:
        """
        前向传播，支持激活值钩子

        Args:
            input_ids: 输入 token ID [batch, seq_len]
            hooks: 干预钩子 {组件名: 修改函数}

        Returns:
            logits: 输出 logits [batch, seq_len, vocab_size]
        """
        if hooks is None:
            hooks = {}

        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # 嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        self.activations["embed"] = x.detach()

        # Transformer 层
        for layer_idx, layer in enumerate(self.layers):
            # 注意力子层
            ln_out = layer["ln1"](x)

            # 计算 Q/K/V
            qkv = layer["attn_qkv"](ln_out)
            q, k, v = qkv.chunk(3, dim=-1)

            # 重塑为多头格式
            q = q.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
            k = k.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
            v = v.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

            # 注意力计算
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

            # 因果掩码
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(causal_mask, float("-inf"))
            attn_weights = torch.softmax(scores, dim=-1)

            # 存储注意力权重
            self.activations[f"attn_weights_L{layer_idx}"] = attn_weights.detach()

            # 注意力输出（逐头存储）
            attn_out = torch.matmul(attn_weights, v)
            self.activations[f"attn_out_L{layer_idx}"] = attn_out.detach()

            # 应用干预钩子（注意力输出层面）
            for head_idx in range(self.n_heads):
                hook_name = f"attn_head_L{layer_idx}_H{head_idx}"
                if hook_name in hooks:
                    attn_out[:, head_idx] = hooks[hook_name](
                        attn_out[:, head_idx], layer_idx, head_idx
                    )

            # 合并多头并投影
            attn_out = attn_out.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.d_model
            )
            attn_out = layer["attn_out"](attn_out)

            # 应用注意力层级的钩子
            hook_name = f"attn_layer_L{layer_idx}"
            if hook_name in hooks:
                attn_out = hooks[hook_name](attn_out, layer_idx, None)

            # 残差连接
            x = x + attn_out
            self.activations[f"resid_mid_L{layer_idx}"] = x.detach()

            # MLP 子层
            ln_out = layer["ln2"](x)
            ff_out = layer["ff_down"](torch.relu(layer["ff_up"](ln_out)))

            # 应用 MLP 层级的钩子
            hook_name = f"mlp_L{layer_idx}"
            if hook_name in hooks:
                ff_out = hooks[hook_name](ff_out, layer_idx, None)

            self.activations[f"mlp_out_L{layer_idx}"] = ff_out.detach()

            # 残差连接
            x = x + ff_out
            self.activations[f"resid_post_L{layer_idx}"] = x.detach()

        # 输出层
        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits


class ActivationPatcher:
    """
    激活值干预实验框架

    提供多种干预方法来建立模型组件与输出之间的因果关系。

    Args:
        model: 支持钩子的 Transformer 模型
    """

    def __init__(self, model: SimpleTransformerForPatching):
        self.model = model

    def zero_ablation(
        self,
        input_ids: torch.Tensor,
        target_pos: int,
        target_token: int,
        components: List[str],
    ) -> List[PatchingResult]:
        """
        零化消融: 将指定组件的输出设为零

        这是最简单的因果干预: 如果零化某个组件导致性能大幅下降，
        说明该组件对当前任务至关重要。

        Args:
            input_ids: 输入 token ID [1, seq_len]
            target_pos: 目标位置索引（我们关注的输出位置）
            target_token: 正确的目标 token ID
            components: 要干预的组件名称列表

        Returns:
            results: 每个组件的干预结果
        """
        # 1. 计算原始 logits
        with torch.no_grad():
            original_logits = self.model(input_ids)
            original_prob = torch.softmax(original_logits[0, target_pos], dim=-1)
            original_metric = original_prob[target_token].item()

        results = []

        for component in components:
            # 2. 创建零化钩子
            def zero_hook(activation, layer_idx, head_idx):
                return torch.zeros_like(activation)

            hooks = {component: zero_hook}

            # 3. 计算干预后 logits
            with torch.no_grad():
                patched_logits = self.model(input_ids, hooks=hooks)
                patched_prob = torch.softmax(
                    patched_logits[0, target_pos], dim=-1
                )
                patched_metric = patched_prob[target_token].item()

            # 4. 计算效果
            effect = original_metric - patched_metric

            # 解析组件名
            parts = component.split("_")
            layer_idx = int(parts[-1][1:]) if parts[-1].startswith("L") else -1
            head_idx_val = None
            for p in parts:
                if p.startswith("H"):
                    head_idx_val = int(p[1:])

            result = PatchingResult(
                component_name=component,
                layer_idx=layer_idx,
                head_idx=head_idx_val,
                original_metric=original_metric,
                patched_metric=patched_metric,
                effect=effect,
                normalized_effect=effect / (abs(original_metric) + 1e-8),
            )
            results.append(result)

        return results

    def mean_ablation(
        self,
        input_ids: torch.Tensor,
        target_pos: int,
        target_token: int,
        reference_activations: Dict[str, torch.Tensor],
        components: List[str],
    ) -> List[PatchingResult]:
        """
        均值消融: 用参考激活值（通常是数据集均值）替换

        比零化消融更合理的基线:
        - 零化会引入分布外的激活模式
        - 均值消融保持激活值在合理范围内

        Args:
            input_ids: 输入 token ID
            target_pos: 目标位置
            target_token: 正确 token ID
            reference_activations: 参考激活值字典
            components: 要干预的组件列表

        Returns:
            results: 干预结果列表
        """
        # 计算原始 logits
        with torch.no_grad():
            original_logits = self.model(input_ids)
            original_prob = torch.softmax(original_logits[0, target_pos], dim=-1)
            original_metric = original_prob[target_token].item()

        results = []

        for component in components:
            # 均值替换钩子
            if component in reference_activations:
                mean_act = reference_activations[component]

                def mean_hook(activation, layer_idx, head_idx, _mean=mean_act):
                    return _mean.expand_as(activation)

                hooks = {component: mean_hook}
            else:
                continue

            with torch.no_grad():
                patched_logits = self.model(input_ids, hooks=hooks)
                patched_prob = torch.softmax(
                    patched_logits[0, target_pos], dim=-1
                )
                patched_metric = patched_prob[target_token].item()

            effect = original_metric - patched_metric

            parts = component.split("_")
            layer_idx = -1
            for p in parts:
                if p.startswith("L") and p[1:].isdigit():
                    layer_idx = int(p[1:])

            result = PatchingResult(
                component_name=component,
                layer_idx=layer_idx,
                original_metric=original_metric,
                patched_metric=patched_metric,
                effect=effect,
                normalized_effect=effect / (abs(original_metric) + 1e-8),
            )
            results.append(result)

        return results

    def activation_patching(
        self,
        clean_input: torch.Tensor,
        corrupted_input: torch.Tensor,
        target_pos: int,
        target_token: int,
    ) -> Dict[str, List[PatchingResult]]:
        """
        激活替换实验（因果追踪）

        步骤:
        1. 用干净输入获取正确的激活值
        2. 用损坏输入获取基线（模型无法正确预测）
        3. 逐个组件将干净激活值"注入"到损坏运行中
        4. 如果注入某个组件恢复了正确预测，说明该组件是关键的

        Args:
            clean_input: 干净输入（模型能正确预测的）
            corrupted_input: 损坏输入（模型无法正确预测的）
            target_pos: 目标位置
            target_token: 正确 token ID

        Returns:
            results: 按组件类型分组的干预结果
        """
        # 1. 干净运行: 收集所有激活值
        with torch.no_grad():
            clean_logits = self.model(clean_input)
            clean_activations = dict(self.model.activations)
            clean_prob = torch.softmax(clean_logits[0, target_pos], dim=-1)
            clean_metric = clean_prob[target_token].item()

        # 2. 损坏运行: 获取基线
        with torch.no_grad():
            corrupted_logits = self.model(corrupted_input)
            corrupted_prob = torch.softmax(
                corrupted_logits[0, target_pos], dim=-1
            )
            corrupted_metric = corrupted_prob[target_token].item()

        results = {"attention": [], "mlp": []}

        # 3. 逐层、逐组件替换
        for layer_idx in range(self.model.n_layers):
            # 替换注意力输出
            for head_idx in range(self.model.n_heads):
                component = f"attn_head_L{layer_idx}_H{head_idx}"
                clean_act = clean_activations.get(f"attn_out_L{layer_idx}")
                if clean_act is None:
                    continue

                def patch_hook(
                    activation, l_idx, h_idx,
                    _clean=clean_act, _head=head_idx
                ):
                    return _clean[:, _head]

                hooks = {component: patch_hook}
                with torch.no_grad():
                    patched_logits = self.model(corrupted_input, hooks=hooks)
                    patched_prob = torch.softmax(
                        patched_logits[0, target_pos], dim=-1
                    )
                    patched_metric = patched_prob[target_token].item()

                # 恢复效果 = 从损坏恢复了多少
                recovery = patched_metric - corrupted_metric
                total_gap = clean_metric - corrupted_metric

                result = PatchingResult(
                    component_name=component,
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    original_metric=corrupted_metric,
                    patched_metric=patched_metric,
                    effect=recovery,
                    normalized_effect=(
                        recovery / (abs(total_gap) + 1e-8)
                    ),
                )
                results["attention"].append(result)

            # 替换 MLP 输出
            component = f"mlp_L{layer_idx}"
            clean_mlp = clean_activations.get(f"mlp_out_L{layer_idx}")
            if clean_mlp is not None:
                def mlp_hook(activation, l_idx, h_idx, _clean=clean_mlp):
                    return _clean

                hooks = {component: mlp_hook}
                with torch.no_grad():
                    patched_logits = self.model(corrupted_input, hooks=hooks)
                    patched_prob = torch.softmax(
                        patched_logits[0, target_pos], dim=-1
                    )
                    patched_metric = patched_prob[target_token].item()

                recovery = patched_metric - corrupted_metric
                result = PatchingResult(
                    component_name=component,
                    layer_idx=layer_idx,
                    original_metric=corrupted_metric,
                    patched_metric=patched_metric,
                    effect=recovery,
                    normalized_effect=recovery / (abs(total_gap) + 1e-8),
                )
                results["mlp"].append(result)

        return results


def format_patching_results(
    results: List[PatchingResult],
    sort_by: str = "effect",
    top_k: int = 10,
) -> str:
    """
    格式化干预结果为可读的表格

    Args:
        results: 干预结果列表
        sort_by: 排序字段
        top_k: 显示前 k 个结果

    Returns:
        formatted: 格式化的结果字符串
    """
    sorted_results = sorted(
        results,
        key=lambda r: abs(getattr(r, sort_by)),
        reverse=True,
    )[:top_k]

    lines = []
    lines.append(f"{'组件':<30} {'原始':>10} {'干预后':>10} {'效果':>10} {'归一化':>10}")
    lines.append("-" * 72)

    for r in sorted_results:
        lines.append(
            f"{r.component_name:<30} "
            f"{r.original_metric:>10.4f} "
            f"{r.patched_metric:>10.4f} "
            f"{r.effect:>10.4f} "
            f"{r.normalized_effect:>10.4f}"
        )

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("激活值干预实验演示")
    print("=" * 60)

    # ---- 1. 创建简化模型 ----
    print("\n[1] 创建简化 Transformer...")
    model = SimpleTransformerForPatching(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_layers=4,
        d_ff=128,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"    模型参数量: {n_params:,}")

    # ---- 2. 创建测试输入 ----
    print("\n[2] 创建测试输入...")
    clean_input = torch.randint(0, 100, (1, 16))
    corrupted_input = torch.randint(0, 100, (1, 16))

    # 保持前半部分相同，后半部分不同
    corrupted_input[0, :8] = clean_input[0, :8]

    target_pos = 10
    target_token = 42

    print(f"    序列长度: {clean_input.shape[1]}")
    print(f"    目标位置: {target_pos}")
    print(f"    目标 token: {target_token}")

    # ---- 3. 零化消融 ----
    print("\n[3] 零化消融实验...")
    patcher = ActivationPatcher(model)

    # 对所有注意力头进行零化
    components = []
    for layer in range(4):
        for head in range(4):
            components.append(f"attn_head_L{layer}_H{head}")

    results = patcher.zero_ablation(
        input_ids=clean_input,
        target_pos=target_pos,
        target_token=target_token,
        components=components,
    )

    print("\n    零化消融结果 (Top-10 效果最大的注意力头):")
    print(format_patching_results(results, top_k=10))

    # ---- 4. 因果追踪 ----
    print("\n[4] 因果追踪 (激活替换)...")
    causal_results = patcher.activation_patching(
        clean_input=clean_input,
        corrupted_input=corrupted_input,
        target_pos=target_pos,
        target_token=target_token,
    )

    print("\n    注意力层恢复效果:")
    print(format_patching_results(causal_results["attention"], top_k=5))

    print("\n    MLP 层恢复效果:")
    print(format_patching_results(causal_results["mlp"], top_k=5))

    print("\n演示完成!")
