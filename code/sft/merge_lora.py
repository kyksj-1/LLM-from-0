"""
LoRA 权重合并工具

本模块实现了 LoRA 权重的合并、保存和加载功能:
- 将 LoRA 适配器合并回基座模型
- 保存/加载 LoRA 权重
- 多 LoRA 切换

数学基础:
- 合并公式: W_merged = W_0 + (alpha/r) * B @ A
- 合并后模型结构与原始模型完全相同，推理无额外开销
- 未合并时，可以为同一基座加载不同 LoRA 实现任务切换

参考:
- Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List
import os

from .lora import LoRALinear, get_lora_state_dict


def merge_lora_weights(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    将模型中所有 LoRA 权重合并回原始权重

    合并后:
    - W_merged = W_0 + (alpha/r) * B @ A
    - LoRA 参数被保留（可选择删除以节省显存）
    - 模型行为与合并前数学等价

    Args:
        model: 包含 LoRALinear 层的模型
        inplace: 是否就地修改（True 修改原始权重，False 返回新模型）

    Returns:
        合并后的模型
    """
    if not inplace:
        import copy
        model = copy.deepcopy(model)

    merged_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merged_count += 1

    print(f"已合并 {merged_count} 个 LoRA 层")
    return model


def unload_lora(model: nn.Module) -> nn.Module:
    """
    将 LoRALinear 层替换回普通的 nn.Linear 层（合并后）

    先合并权重，然后用合并后的 nn.Linear 替换 LoRALinear，
    完全移除 LoRA 结构以节省显存。

    Args:
        model: 包含 LoRALinear 层的模型

    Returns:
        只包含 nn.Linear 的模型
    """
    # 收集需要替换的模块
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # 获取合并后的 Linear 层
            merged_linear = module.get_merged_linear()
            replacements.append((name, merged_linear))

    # 执行替换
    for name, new_module in replacements:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    print(f"已卸载 {len(replacements)} 个 LoRA 层，替换为 nn.Linear")
    return model


def save_lora_weights(
    model: nn.Module,
    save_path: str,
    metadata: Optional[Dict] = None,
) -> None:
    """
    保存 LoRA 权重到文件

    只保存 LoRA 的 A 和 B 矩阵（不保存冻结的基座权重），
    文件大小远小于完整模型。

    Args:
        model: 包含 LoRA 层的模型
        save_path: 保存路径 (.pt 文件)
        metadata: 可选的元数据（如 rank, alpha, target_modules 等）
    """
    lora_state = get_lora_state_dict(model)

    save_dict = {
        "lora_state_dict": lora_state,
        "metadata": metadata or {},
    }

    # 添加 LoRA 配置信息
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            save_dict["metadata"]["rank"] = module.rank
            save_dict["metadata"]["alpha"] = module.alpha
            save_dict["metadata"]["scaling"] = module.scaling
            break  # 假设所有 LoRA 层配置相同

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save(save_dict, save_path)

    # 统计文件大小
    file_size = os.path.getsize(save_path)
    num_params = sum(v.numel() for v in lora_state.values())
    print(f"LoRA 权重已保存到: {save_path}")
    print(f"  参数量: {num_params:,}")
    print(f"  文件大小: {file_size / 1024:.1f} KB")


def load_lora_weights(
    model: nn.Module,
    load_path: str,
    strict: bool = True,
) -> nn.Module:
    """
    加载 LoRA 权重到模型

    Args:
        model: 已添加 LoRA 层的模型
        load_path: LoRA 权重文件路径
        strict: 是否严格匹配键名

    Returns:
        加载了 LoRA 权重的模型
    """
    checkpoint = torch.load(load_path, map_location="cpu", weights_only=True)
    lora_state = checkpoint["lora_state_dict"]
    metadata = checkpoint.get("metadata", {})

    # 加载参数
    model_state = model.state_dict()
    loaded_count = 0
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key].copy_(value)
            loaded_count += 1
        elif strict:
            raise KeyError(f"LoRA 权重中的键 '{key}' 在模型中未找到")

    print(f"已加载 {loaded_count} 个 LoRA 参数")
    if metadata:
        print(f"  配置: rank={metadata.get('rank')}, alpha={metadata.get('alpha')}")

    return model


class LoRAManager:
    """
    LoRA 多适配器管理器

    支持为同一个基座模型加载和切换不同的 LoRA 适配器，
    无需重新加载基座模型。

    使用场景:
    - 同一基座模型服务多个任务
    - A/B 测试不同的 LoRA 配置
    - 动态切换对话风格

    Args:
        base_model: 基座模型（已添加 LoRA 层）
    """

    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.adapters: Dict[str, Dict[str, torch.Tensor]] = {}
        self.active_adapter: Optional[str] = None

        # 保存初始 LoRA 状态（零初始化状态）
        self._zero_state = self._get_current_lora_state()

    def _get_current_lora_state(self) -> Dict[str, torch.Tensor]:
        """获取当前 LoRA 参数的副本"""
        state = {}
        for name, param in self.base_model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                state[name] = param.data.clone()
        return state

    def _set_lora_state(self, state: Dict[str, torch.Tensor]) -> None:
        """设置 LoRA 参数"""
        for name, param in self.base_model.named_parameters():
            if name in state:
                param.data.copy_(state[name])

    def register_adapter(self, name: str, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        注册一个新的 LoRA 适配器

        Args:
            name: 适配器名称
            state_dict: LoRA 参数字典
        """
        self.adapters[name] = {k: v.clone() for k, v in state_dict.items()}
        print(f"已注册适配器 '{name}' (参数量: {sum(v.numel() for v in state_dict.values()):,})")

    def load_adapter(self, name: str, path: str) -> None:
        """
        从文件加载并注册适配器

        Args:
            name: 适配器名称
            path: 权重文件路径
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        self.register_adapter(name, checkpoint["lora_state_dict"])

    def switch_adapter(self, name: str) -> None:
        """
        切换到指定的 LoRA 适配器

        Args:
            name: 适配器名称
        """
        if name not in self.adapters:
            raise ValueError(f"适配器 '{name}' 未注册。已注册: {list(self.adapters.keys())}")

        self._set_lora_state(self.adapters[name])
        self.active_adapter = name
        print(f"已切换到适配器 '{name}'")

    def reset_adapter(self) -> None:
        """重置 LoRA 为零（等效于无适配器）"""
        self._set_lora_state(self._zero_state)
        self.active_adapter = None
        print("已重置 LoRA 适配器")

    def list_adapters(self) -> List[str]:
        """列出所有已注册的适配器"""
        return list(self.adapters.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("LoRA 权重合并演示")
    print("=" * 60)

    torch.manual_seed(42)

    # === 1. 创建带 LoRA 的模型 ===
    from .lora import apply_lora_to_model

    class SimpleModel(nn.Module):
        def __init__(self, d_model=256):
            super().__init__()
            self.q_proj = nn.Linear(d_model, d_model)
            self.v_proj = nn.Linear(d_model, d_model)
            self.out_proj = nn.Linear(d_model, d_model)

        def forward(self, x):
            q = self.q_proj(x)
            v = self.v_proj(x)
            return self.out_proj(v + q)

    model = SimpleModel(d_model=256)
    apply_lora_to_model(model, {"q_proj", "v_proj"}, rank=8, alpha=16.0)

    # 模拟训练后的 LoRA 参数
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.normal_(0, 0.01)
                module.lora_B.normal_(0, 0.01)

    # === 2. 验证合并等价性 ===
    print("\n--- 合并等价性验证 ---")
    x = torch.randn(2, 16, 256)

    # 合并前输出
    output_before = model(x).detach().clone()

    # 合并
    merge_lora_weights(model, inplace=True)

    # 合并后输出
    output_after = model(x).detach()

    diff = (output_before - output_after).abs().max().item()
    print(f"合并前后最大差异: {diff:.2e}")
    print(f"验证: {'通过' if diff < 1e-5 else '失败'}")

    # === 3. 保存和加载 LoRA 权重 ===
    print("\n--- LoRA 权重保存/加载 ---")

    # 重新创建模型
    model2 = SimpleModel(d_model=256)
    apply_lora_to_model(model2, {"q_proj", "v_proj"}, rank=8, alpha=16.0)

    with torch.no_grad():
        for module in model2.modules():
            if isinstance(module, LoRALinear):
                module.lora_A.normal_(0, 0.01)
                module.lora_B.normal_(0, 0.01)

    # 保存
    save_path = "demo_lora_weights.pt"
    save_lora_weights(model2, save_path, metadata={"task": "demo"})

    # 加载到新模型
    model3 = SimpleModel(d_model=256)
    apply_lora_to_model(model3, {"q_proj", "v_proj"}, rank=8, alpha=16.0)
    load_lora_weights(model3, save_path)

    # 验证
    x2 = torch.randn(2, 16, 256)
    out2 = model2(x2).detach()
    out3 = model3(x2).detach()
    diff2 = (out2 - out3).abs().max().item()
    print(f"保存/加载后输出差异: {diff2:.2e}")

    # 清理临时文件
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"已删除临时文件: {save_path}")

    # === 4. 多 LoRA 切换演示 ===
    print("\n--- 多 LoRA 切换演示 ---")

    base_model = SimpleModel(d_model=256)
    apply_lora_to_model(base_model, {"q_proj", "v_proj"}, rank=8, alpha=16.0)

    manager = LoRAManager(base_model)

    # 创建两个不同的适配器
    adapter_a = {}
    adapter_b = {}
    for name, param in base_model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            adapter_a[name] = torch.randn_like(param) * 0.01
            adapter_b[name] = torch.randn_like(param) * 0.02

    manager.register_adapter("task_a", adapter_a)
    manager.register_adapter("task_b", adapter_b)

    print(f"已注册适配器: {manager.list_adapters()}")

    # 切换并推理
    x3 = torch.randn(1, 8, 256)

    manager.switch_adapter("task_a")
    out_a = base_model(x3).detach()

    manager.switch_adapter("task_b")
    out_b = base_model(x3).detach()

    diff_ab = (out_a - out_b).abs().mean().item()
    print(f"不同适配器的输出差异: {diff_ab:.6f}")
    print(f"验证: {'通过 (输出不同)' if diff_ab > 1e-6 else '失败'}")
