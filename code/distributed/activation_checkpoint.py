"""
激活检查点 (Activation Checkpointing / Gradient Checkpointing)

本模块实现激活检查点技术，以计算换显存:
    - 标准反向传播需要保存所有中间层的激活值
    - 激活检查点只保存部分层的激活值，反向传播时重新计算其余层

显存节省:
    标准: O(L) -- 保存每一层的激活
    检查点: O(sqrt(L)) -- 只保存 sqrt(L) 个检查点的激活

计算开销:
    增加约一次前向传播的计算量 (约 33% 额外时间)

核心原理:
    1. 将 L 层网络分为 sqrt(L) 个段
    2. 只保存每个段的输入（检查点）
    3. 反向传播到某一段时，从该段的检查点重新执行前向传播

作者: LLM Tutorial
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Callable, Optional
import math
import time


class CheckpointFunction(torch.autograd.Function):
    """
    激活检查点的自定义自动求导函数

    前向传播时不保存中间激活值（使用 no_grad），
    反向传播时重新执行前向传播以恢复激活值。
    """

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        """
        前向传播

        Args:
            ctx: 上下文对象
            run_function: 要检查点的函数
            preserve_rng_state: 是否保存随机数生成器状态
            *args: 输入张量

        Returns:
            run_function 的输出
        """
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state

        # 保存输入（用于反向传播时重新计算）
        ctx.save_for_backward(*args)

        # 保存 RNG 状态（确保 Dropout 等随机操作可重复）
        if preserve_rng_state:
            ctx.fwd_rng_state = torch.random.get_rng_state()

        # 在 no_grad 模式下执行前向传播
        # 不会保存中间激活值
        with torch.no_grad():
            outputs = run_function(*args)

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        反向传播

        重新执行前向传播以恢复中间激活值，然后计算梯度。
        """
        inputs = ctx.saved_tensors

        # 恢复 RNG 状态
        if ctx.preserve_rng_state:
            rng_state_backup = torch.random.get_rng_state()
            torch.random.set_rng_state(ctx.fwd_rng_state)

        # 重新执行前向传播（这次需要梯度）
        detached_inputs = tuple(x.detach().requires_grad_(x.requires_grad) for x in inputs)

        with torch.enable_grad():
            outputs = ctx.run_function(*detached_inputs)

        # 恢复 RNG 状态
        if ctx.preserve_rng_state:
            torch.random.set_rng_state(rng_state_backup)

        # 计算梯度
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)

        torch.autograd.backward(outputs, grad_outputs)

        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) and inp.requires_grad else None
            for inp in detached_inputs
        )

        return (None, None) + grads


def checkpoint(function: Callable, *args, preserve_rng_state: bool = True):
    """
    对一个函数应用激活检查点

    使用方式:
        output = checkpoint(layer, input)
        等价于 output = layer(input)
        但不保存 layer 内部的中间激活值

    Args:
        function: 要检查点的函数或模块
        *args: 输入参数
        preserve_rng_state: 是否保存随机数状态

    Returns:
        function 的输出
    """
    return CheckpointFunction.apply(function, preserve_rng_state, *args)


class SimpleLayer(nn.Module):
    """简单的网络层，用于演示"""

    def __init__(self, dim: int, layer_id: int = 0):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.norm(x))


class SequentialWithCheckpoint(nn.Module):
    """
    支持激活检查点的顺序模型

    将模型的层分为若干段（segment），只在段的边界保存激活值。
    反向传播到某一段时，从该段的输入重新执行前向传播。

    分段策略:
        - 每 checkpoint_every 层设置一个检查点
        - 推荐值: sqrt(总层数)
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        checkpoint_every: Optional[int] = None,
        use_checkpoint: bool = True,
    ):
        """
        Args:
            layers: 模型层列表
            checkpoint_every: 每多少层设置一个检查点 (None = sqrt(L))
            use_checkpoint: 是否启用激活检查点
        """
        super().__init__()
        self.layers = layers
        self.use_checkpoint = use_checkpoint

        num_layers = len(layers)
        if checkpoint_every is None:
            self.checkpoint_every = max(1, int(math.sqrt(num_layers)))
        else:
            self.checkpoint_every = checkpoint_every

        # 将层分成段
        self.segments = []
        for i in range(0, num_layers, self.checkpoint_every):
            end = min(i + self.checkpoint_every, num_layers)
            self.segments.append(list(range(i, end)))

        print(f"激活检查点配置:")
        print(f"  总层数: {num_layers}")
        print(f"  检查点间隔: {self.checkpoint_every}")
        print(f"  段数: {len(self.segments)}")
        print(f"  标准显存: O({num_layers}) 层激活")
        print(f"  检查点显存: O({len(self.segments)} + {self.checkpoint_every}) "
              f"= O({len(self.segments) + self.checkpoint_every}) 层激活")

    def _run_segment(self, segment_idx: int, x: torch.Tensor) -> torch.Tensor:
        """执行一个段的前向传播"""
        for layer_idx in self.segments[segment_idx]:
            x = self.layers[layer_idx](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        如果启用激活检查点，则对每个段使用检查点。
        """
        for seg_idx in range(len(self.segments)):
            if self.use_checkpoint and self.training:
                # 使用激活检查点
                # 需要用 lambda 包装来正确捕获 seg_idx
                def create_forward_fn(idx):
                    def forward_fn(input_tensor):
                        return self._run_segment(idx, input_tensor)
                    return forward_fn

                x = checkpoint(create_forward_fn(seg_idx), x)
            else:
                x = self._run_segment(seg_idx, x)

        return x


def analyze_memory_savings(num_layers_list: List[int] = None):
    """
    分析激活检查点的显存节省

    对比不同层数下标准方式和检查点方式的显存占用。

    Args:
        num_layers_list: 要分析的层数列表
    """
    if num_layers_list is None:
        num_layers_list = [4, 9, 16, 25, 36, 49, 64, 100, 144]

    print("=" * 60)
    print("激活检查点显存分析")
    print("=" * 60)

    print(f"\n{'层数(L)':>8} {'标准 O(L)':>12} {'检查点 O(sqrt(L))':>20} {'节省比例':>10} {'额外计算':>10}")
    print("-" * 64)

    for L in num_layers_list:
        sqrt_L = int(math.sqrt(L))
        # 检查点方式: 保存 sqrt(L) 个段的输入 + 当前段的 sqrt(L) 层激活
        checkpoint_mem = sqrt_L + sqrt_L  # 近似

        saving = 1 - checkpoint_mem / L
        extra_compute = "~33%"

        print(
            f"{L:>8} {L:>12} {checkpoint_mem:>20} {saving:>10.1%} {extra_compute:>10}"
        )


def benchmark_checkpoint(
    dim: int = 256,
    num_layers: int = 16,
    batch_size: int = 8,
    seq_len: int = 64,
):
    """
    基准测试: 对比有无激活检查点的训练

    注意: CPU 上的基准测试可能无法准确反映 GPU 上的性能差异。

    Args:
        dim: 隐藏维度
        num_layers: 层数
        batch_size: 批次大小
        seq_len: 序列长度
    """
    print("\n" + "=" * 60)
    print("激活检查点基准测试")
    print("=" * 60)

    print(f"\n配置: dim={dim}, layers={num_layers}, batch={batch_size}, seq={seq_len}")

    # 创建输入
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_len, dim, requires_grad=True)

    # --- 无检查点 ---
    layers_no_ckpt = nn.ModuleList([SimpleLayer(dim, i) for i in range(num_layers)])
    model_no_ckpt = SequentialWithCheckpoint(layers_no_ckpt, use_checkpoint=False)
    model_no_ckpt.train()

    # 前向 + 反向
    start = time.time()
    out = model_no_ckpt(x)
    loss = out.sum()
    loss.backward()
    time_no_ckpt = time.time() - start

    # --- 有检查点 ---
    layers_ckpt = nn.ModuleList([SimpleLayer(dim, i) for i in range(num_layers)])
    # 复制权重
    for l1, l2 in zip(layers_no_ckpt, layers_ckpt):
        l2.load_state_dict(l1.state_dict())

    model_ckpt = SequentialWithCheckpoint(layers_ckpt, use_checkpoint=True)
    model_ckpt.train()

    x_ckpt = x.detach().clone().requires_grad_(True)

    start = time.time()
    out_ckpt = model_ckpt(x_ckpt)
    loss_ckpt = out_ckpt.sum()
    loss_ckpt.backward()
    time_ckpt = time.time() - start

    # 对比结果
    print(f"\n无检查点:")
    print(f"  时间: {time_no_ckpt * 1000:.1f} ms")
    print(f"  输出范数: {out.detach().norm().item():.4f}")
    print(f"  梯度范数: {x.grad.norm().item():.4f}")

    print(f"\n有检查点:")
    print(f"  时间: {time_ckpt * 1000:.1f} ms")
    print(f"  输出范数: {out_ckpt.detach().norm().item():.4f}")
    print(f"  梯度范数: {x_ckpt.grad.norm().item():.4f}")

    # 验证正确性
    output_diff = (out.detach() - out_ckpt.detach()).abs().max().item()
    grad_diff = (x.grad - x_ckpt.grad).abs().max().item()

    print(f"\n正确性验证:")
    print(f"  输出最大差异: {output_diff:.2e}")
    print(f"  梯度最大差异: {grad_diff:.2e}")

    overhead = (time_ckpt - time_no_ckpt) / time_no_ckpt
    print(f"\n时间开销: {overhead:+.1%}")
    print(f"(注: CPU 基准测试可能不准确，GPU 上典型开销为 +30%~40%)")


if __name__ == "__main__":
    analyze_memory_savings()
    benchmark_checkpoint()
