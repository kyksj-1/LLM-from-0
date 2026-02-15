"""
分布式数据并行 (DDP) 基本实现

本模块演示 PyTorch DistributedDataParallel 的核心原理，
包括 Ring All-Reduce 的模拟实现和 DDP 训练的基本流程。

核心概念:
    - All-Reduce: 所有节点获得梯度的聚合结果
    - Ring All-Reduce: 带宽最优的 All-Reduce 算法
    - DDP: 每个 GPU 持有完整模型副本，通过 All-Reduce 同步梯度

作者: LLM Tutorial
"""

import torch
import torch.nn as nn
from typing import List, Optional
import math


class RingAllReduce:
    """
    Ring All-Reduce 模拟实现

    在单进程中模拟多个 GPU 节点之间的 Ring All-Reduce 操作。
    实际分布式环境中应使用 torch.distributed.all_reduce。

    Ring All-Reduce 分为两个阶段:
        1. Reduce-Scatter: 每个节点得到一个分片的完整归约结果
        2. All-Gather: 将归约后的分片广播到所有节点

    通信量分析:
        每个节点总共发送 2 * (n-1)/n * P 数据，
        其中 n 为节点数，P 为参数总量。
        当 n 较大时，每个节点的通信量趋近于 2P，与节点数无关。
    """

    def __init__(self, num_workers: int):
        """
        初始化 Ring All-Reduce

        Args:
            num_workers: 模拟的工作节点数量
        """
        self.num_workers = num_workers

    def reduce_scatter(self, tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reduce-Scatter 阶段

        将每个节点的张量分为 num_workers 个分片，
        经过 num_workers-1 步环形传递后，
        每个节点持有一个分片的完整归约结果。

        Args:
            tensors: 每个工作节点的梯度张量列表

        Returns:
            每个工作节点的部分归约结果
        """
        n = self.num_workers
        chunk_size = tensors[0].numel() // n

        # 将每个张量分为 n 个分片
        chunks = []
        for t in tensors:
            flat = t.flatten()
            cs = [flat[i * chunk_size: (i + 1) * chunk_size].clone() for i in range(n)]
            chunks.append(cs)

        # 执行 n-1 步环形传递
        for step in range(n - 1):
            new_chunks = [list(c) for c in chunks]  # 深拷贝
            for worker in range(n):
                # 每个 worker 向下一个 worker 发送一个分片
                send_idx = (worker - step) % n
                recv_from = (worker - 1) % n
                recv_idx = (recv_from - step) % n

                # 接收并累加
                new_chunks[worker][recv_idx] = (
                    chunks[worker][recv_idx] + chunks[recv_from][recv_idx]
                )
            chunks = new_chunks

        return chunks

    def all_gather(self, chunks: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """
        All-Gather 阶段

        将 Reduce-Scatter 的结果广播到所有节点，
        使每个节点都拥有完整的归约结果。

        Args:
            chunks: Reduce-Scatter 后每个节点的分片

        Returns:
            每个工作节点的完整归约结果
        """
        n = self.num_workers

        # 执行 n-1 步环形传递
        for step in range(n - 1):
            new_chunks = [list(c) for c in chunks]
            for worker in range(n):
                send_idx = (worker - step + 1) % n
                recv_from = (worker - 1) % n
                recv_idx = (recv_from - step + 1) % n

                # 直接复制（不再累加）
                new_chunks[worker][recv_idx] = chunks[recv_from][recv_idx].clone()
            chunks = new_chunks

        # 将分片拼接为完整张量
        results = []
        for worker_chunks in chunks:
            results.append(torch.cat(worker_chunks))

        return results

    def all_reduce(self, tensors: List[torch.Tensor], op: str = "sum") -> List[torch.Tensor]:
        """
        完整的 Ring All-Reduce 操作

        Args:
            tensors: 每个工作节点的张量列表
            op: 归约操作类型 ("sum" 或 "mean")

        Returns:
            每个工作节点的归约结果（所有节点结果相同）
        """
        # 阶段 1: Reduce-Scatter
        chunks = self.reduce_scatter(tensors)

        # 阶段 2: All-Gather
        results = self.all_gather(chunks)

        if op == "mean":
            results = [r / self.num_workers for r in results]

        return results


class SimpleModel(nn.Module):
    """
    用于演示 DDP 的简单模型

    一个三层全连接网络，用于分类任务。
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 10):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            output_dim: 输出类别数
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.layers(x)


class SimulatedDDP:
    """
    模拟的分布式数据并行 (DDP)

    在单进程中模拟多个 GPU 的 DDP 训练过程:
        1. 每个"GPU"持有模型的完整副本
        2. 数据被均匀分割到各个 GPU
        3. 各 GPU 独立计算梯度
        4. 通过 All-Reduce 同步梯度
        5. 各 GPU 使用相同的梯度更新参数

    实际分布式环境中应使用 torch.nn.parallel.DistributedDataParallel。
    """

    def __init__(self, model: nn.Module, num_gpus: int = 4, lr: float = 1e-3):
        """
        Args:
            model: 要并行化的模型
            num_gpus: 模拟的 GPU 数量
            lr: 学习率
        """
        self.num_gpus = num_gpus
        self.lr = lr
        self.ring_all_reduce = RingAllReduce(num_gpus)

        # 创建模型副本（模拟每个 GPU 上的模型）
        self.models = []
        for i in range(num_gpus):
            model_copy = type(model)(
                *self._get_model_init_args(model)
            )
            model_copy.load_state_dict(model.state_dict())
            self.models.append(model_copy)

        # 创建优化器
        self.optimizers = [
            torch.optim.SGD(m.parameters(), lr=lr)
            for m in self.models
        ]

    def _get_model_init_args(self, model: nn.Module):
        """获取模型初始化参数（简化处理）"""
        if isinstance(model, SimpleModel):
            first_layer = model.layers[0]
            last_layer = model.layers[-1]
            hidden_layer = model.layers[2]
            return (first_layer.in_features, hidden_layer.in_features, last_layer.out_features)
        raise ValueError("不支持的模型类型")

    def train_step(self, data: torch.Tensor, targets: torch.Tensor) -> float:
        """
        执行一步 DDP 训练

        Args:
            data: 输入数据 [batch_size, input_dim]
            targets: 目标标签 [batch_size]

        Returns:
            平均损失值
        """
        batch_size = data.shape[0]
        chunk_size = batch_size // self.num_gpus
        criterion = nn.CrossEntropyLoss()

        # 步骤 1: 分割数据到各个 GPU
        data_chunks = torch.chunk(data, self.num_gpus, dim=0)
        target_chunks = torch.chunk(targets, self.num_gpus, dim=0)

        # 步骤 2: 各 GPU 独立前向传播和反向传播
        losses = []
        for i in range(self.num_gpus):
            self.optimizers[i].zero_grad()
            output = self.models[i](data_chunks[i])
            loss = criterion(output, target_chunks[i])
            loss.backward()
            losses.append(loss.item())

        # 步骤 3: 收集所有 GPU 的梯度并进行 All-Reduce
        for param_group in zip(*[m.parameters() for m in self.models]):
            grads = [p.grad.flatten() for p in param_group]

            # Ring All-Reduce 求平均
            avg_grads = self.ring_all_reduce.all_reduce(grads, op="mean")

            # 将平均后的梯度写回各个模型
            for p, avg_g in zip(param_group, avg_grads):
                p.grad = avg_g.reshape(p.grad.shape)

        # 步骤 4: 各 GPU 使用相同的梯度更新参数
        for opt in self.optimizers:
            opt.step()

        return sum(losses) / len(losses)

    def verify_sync(self) -> float:
        """
        验证所有模型副本的参数是否一致

        Returns:
            所有模型副本之间参数的最大差异
        """
        max_diff = 0.0
        for param_group in zip(*[m.parameters() for m in self.models]):
            for i in range(1, len(param_group)):
                diff = (param_group[0].data - param_group[i].data).abs().max().item()
                max_diff = max(max_diff, diff)
        return max_diff


def demo_ring_all_reduce():
    """演示 Ring All-Reduce 的正确性"""
    print("=" * 60)
    print("Ring All-Reduce 演示")
    print("=" * 60)

    num_workers = 4
    tensor_size = 8  # 必须能被 num_workers 整除

    # 创建每个 worker 的梯度
    tensors = [torch.randn(tensor_size) for _ in range(num_workers)]
    print("\n各 Worker 的原始梯度:")
    for i, t in enumerate(tensors):
        print(f"  Worker {i}: {t.tolist()}")

    # 计算期望的求和结果
    expected_sum = sum(tensors)
    print(f"\n期望的求和结果: {expected_sum.tolist()}")

    # 执行 Ring All-Reduce
    ring_ar = RingAllReduce(num_workers)
    results = ring_ar.all_reduce(tensors, op="sum")

    print("\nRing All-Reduce 结果:")
    for i, r in enumerate(results):
        diff = (r - expected_sum).abs().max().item()
        print(f"  Worker {i}: {r.tolist()}")
        print(f"    与期望值的最大差异: {diff:.2e}")

    # 验证所有 worker 的结果一致
    max_cross_diff = max(
        (results[i] - results[j]).abs().max().item()
        for i in range(num_workers)
        for j in range(i + 1, num_workers)
    )
    print(f"\n各 Worker 结果之间的最大差异: {max_cross_diff:.2e}")
    print("All-Reduce 正确性验证通过!" if max_cross_diff < 1e-6 else "验证失败!")


def demo_simulated_ddp():
    """演示模拟 DDP 训练"""
    print("\n" + "=" * 60)
    print("模拟 DDP 训练演示")
    print("=" * 60)

    # 创建数据
    torch.manual_seed(42)
    num_samples = 256
    input_dim = 64
    num_classes = 10

    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))

    # 创建模型和 DDP 封装
    model = SimpleModel(input_dim=input_dim, output_dim=num_classes)
    num_gpus = 4
    ddp = SimulatedDDP(model, num_gpus=num_gpus, lr=0.01)

    # 训练
    num_epochs = 5
    batch_size = 64
    print(f"\n配置: {num_gpus} GPUs, batch_size={batch_size}, epochs={num_epochs}")
    print(f"每个 GPU 的 micro-batch size: {batch_size // num_gpus}")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_x = X[start:end]
            batch_y = y[start:end]

            # 确保 batch 能被 GPU 数量整除
            if batch_x.shape[0] % num_gpus != 0:
                continue

            loss = ddp.train_step(batch_x, batch_y)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)
        sync_diff = ddp.verify_sync()
        print(f"  Epoch {epoch + 1}: 损失={avg_loss:.4f}, 模型同步差异={sync_diff:.2e}")

    print("\nDDP 模拟训练完成!")
    print(f"最终模型同步差异: {ddp.verify_sync():.2e}")


if __name__ == "__main__":
    demo_ring_all_reduce()
    demo_simulated_ddp()
