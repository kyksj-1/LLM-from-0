"""
简化版 PagedAttention 实现

本模块实现了 PagedAttention 的核心概念，包括：
- 物理块管理（Block Allocator）
- 页表（Block Table）
- 分页 KV Cache 的读写操作
- 简化版的注意力计算（通过页表间接访问 KV Cache）

PagedAttention 借鉴了操作系统的虚拟内存分页机制，将 KV Cache 从连续内存分配
改为分页分配，消除了内存碎片，将显存利用率从 20-40% 提升到接近 100%。

参考论文:
    Kwon et al., "Efficient Memory Management for Large Language Model Serving
    with PagedAttention", SOSP 2023
"""

import torch
import torch.nn.functional as F
import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class PagedKVConfig:
    """PagedAttention 配置"""
    n_layers: int           # 层数
    n_kv_heads: int         # KV 头数
    head_dim: int           # 每头维度
    block_size: int = 16    # 每个物理块存储的 token 数
    max_num_blocks: int = 256  # 最大物理块数
    dtype: torch.dtype = torch.float32


class BlockAllocator:
    """
    物理块分配器

    管理一个固定大小的物理块池，支持分配和释放操作。
    类似于操作系统的页帧分配器。
    """

    def __init__(self, max_num_blocks: int):
        """
        Args:
            max_num_blocks: 最大物理块数
        """
        self.max_num_blocks = max_num_blocks
        # 空闲块列表（用栈实现，后进先出）
        self.free_blocks: List[int] = list(range(max_num_blocks))
        # 已分配块集合
        self.allocated_blocks: set = set()

    def allocate(self) -> int:
        """
        分配一个物理块

        Returns:
            物理块 ID

        Raises:
            RuntimeError: 没有可用的空闲块
        """
        if not self.free_blocks:
            raise RuntimeError("物理块已耗尽！无法分配新块。")
        block_id = self.free_blocks.pop()
        self.allocated_blocks.add(block_id)
        return block_id

    def free(self, block_id: int):
        """
        释放一个物理块

        Args:
            block_id: 要释放的物理块 ID
        """
        if block_id not in self.allocated_blocks:
            raise ValueError(f"块 {block_id} 未被分配，无法释放")
        self.allocated_blocks.remove(block_id)
        self.free_blocks.append(block_id)

    @property
    def num_free(self) -> int:
        """可用空闲块数"""
        return len(self.free_blocks)

    @property
    def num_allocated(self) -> int:
        """已分配块数"""
        return len(self.allocated_blocks)

    @property
    def utilization(self) -> float:
        """显存利用率"""
        return self.num_allocated / self.max_num_blocks


class PagedKVCache:
    """
    基于分页的 KV Cache 管理器

    将 KV Cache 存储在固定大小的物理块中，通过页表（block table）
    实现逻辑地址到物理地址的映射。

    物理块形状: [n_kv_heads, block_size, head_dim]
    """

    def __init__(self, config: PagedKVConfig, device: str = "cpu"):
        self.config = config
        self.device = device

        # 块分配器
        self.allocator = BlockAllocator(config.max_num_blocks)

        # 物理块存储
        # k_blocks[layer][block_id] 形状: [n_kv_heads, block_size, head_dim]
        self.k_blocks = [
            torch.zeros(
                config.max_num_blocks, config.n_kv_heads,
                config.block_size, config.head_dim,
                dtype=config.dtype, device=device,
            )
            for _ in range(config.n_layers)
        ]
        self.v_blocks = [
            torch.zeros(
                config.max_num_blocks, config.n_kv_heads,
                config.block_size, config.head_dim,
                dtype=config.dtype, device=device,
            )
            for _ in range(config.n_layers)
        ]

        # 每个请求的页表: request_id -> List[block_id]
        self.block_tables: Dict[int, List[int]] = {}
        # 每个请求的当前序列长度
        self.seq_lens: Dict[int, int] = {}

    def add_request(self, request_id: int) -> bool:
        """
        注册新请求，分配初始块

        Args:
            request_id: 请求 ID

        Returns:
            是否成功添加
        """
        if request_id in self.block_tables:
            return False  # 请求已存在

        if self.allocator.num_free == 0:
            return False  # 无可用块

        # 分配第一个块
        first_block = self.allocator.allocate()
        self.block_tables[request_id] = [first_block]
        self.seq_lens[request_id] = 0
        return True

    def remove_request(self, request_id: int):
        """
        移除请求，释放其所有块

        Args:
            request_id: 请求 ID
        """
        if request_id not in self.block_tables:
            return
        for block_id in self.block_tables[request_id]:
            self.allocator.free(block_id)
        del self.block_tables[request_id]
        del self.seq_lens[request_id]

    def append_token(
        self,
        request_id: int,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> bool:
        """
        向请求的 KV Cache 追加一个 token

        Args:
            request_id: 请求 ID
            layer_idx: 层索引
            new_k: 新的 Key [n_kv_heads, 1, head_dim]
            new_v: 新的 Value [n_kv_heads, 1, head_dim]

        Returns:
            是否成功追加
        """
        if request_id not in self.block_tables:
            return False

        # 只在第一层时更新序列长度和分配新块
        seq_len = self.seq_lens[request_id]
        block_idx = seq_len // self.config.block_size  # 当前应写入的逻辑块号
        slot_idx = seq_len % self.config.block_size    # 块内偏移

        # 如果需要新块（当前块已满）
        if block_idx >= len(self.block_tables[request_id]):
            if self.allocator.num_free == 0:
                return False  # 无可用块
            new_block = self.allocator.allocate()
            self.block_tables[request_id].append(new_block)

        # 获取物理块 ID
        physical_block_id = self.block_tables[request_id][block_idx]

        # 写入 KV
        self.k_blocks[layer_idx][physical_block_id, :, slot_idx, :] = new_k.squeeze(1)
        self.v_blocks[layer_idx][physical_block_id, :, slot_idx, :] = new_v.squeeze(1)

        # 只在最后一层更新序列长度
        if layer_idx == self.config.n_layers - 1:
            self.seq_lens[request_id] = seq_len + 1

        return True

    def get_kv(
        self,
        request_id: int,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取请求在指定层的完整 KV Cache

        通过页表将分散在各物理块中的 KV 收集为连续张量。

        Args:
            request_id: 请求 ID
            layer_idx: 层索引

        Returns:
            (k, v): 形状 [n_kv_heads, seq_len, head_dim]
        """
        seq_len = self.seq_lens[request_id]
        if seq_len == 0:
            return (
                torch.empty(self.config.n_kv_heads, 0, self.config.head_dim,
                           device=self.device),
                torch.empty(self.config.n_kv_heads, 0, self.config.head_dim,
                           device=self.device),
            )

        # 按逻辑块顺序收集物理块中的数据
        k_parts = []
        v_parts = []
        remaining = seq_len

        for block_id in self.block_tables[request_id]:
            n_tokens = min(remaining, self.config.block_size)
            k_parts.append(self.k_blocks[layer_idx][block_id, :, :n_tokens, :])
            v_parts.append(self.v_blocks[layer_idx][block_id, :, :n_tokens, :])
            remaining -= n_tokens
            if remaining <= 0:
                break

        k = torch.cat(k_parts, dim=1)  # [n_kv_heads, seq_len, head_dim]
        v = torch.cat(v_parts, dim=1)
        return k, v

    def status(self) -> dict:
        """返回当前状态摘要"""
        return {
            "total_blocks": self.config.max_num_blocks,
            "allocated_blocks": self.allocator.num_allocated,
            "free_blocks": self.allocator.num_free,
            "utilization": f"{self.allocator.utilization * 100:.1f}%",
            "active_requests": len(self.block_tables),
            "request_details": {
                rid: {
                    "seq_len": self.seq_lens[rid],
                    "num_blocks": len(self.block_tables[rid]),
                    "internal_fragmentation": (
                        len(self.block_tables[rid]) * self.config.block_size
                        - self.seq_lens[rid]
                    ),
                }
                for rid in self.block_tables
            },
        }


def paged_attention(
    query: torch.Tensor,
    paged_cache: PagedKVCache,
    request_id: int,
    layer_idx: int,
) -> torch.Tensor:
    """
    使用分页 KV Cache 计算注意力

    Args:
        query: [n_heads, 1, head_dim]（Decode 阶段单 token query）
        paged_cache: 分页 KV Cache 管理器
        request_id: 请求 ID
        layer_idx: 层索引

    Returns:
        注意力输出 [n_heads, 1, head_dim]
    """
    # 从分页缓存中收集 KV
    k, v = paged_cache.get_kv(request_id, layer_idx)  # [n_kv_heads, seq_len, head_dim]

    if k.shape[1] == 0:
        return torch.zeros_like(query)

    # 简化处理：假设 n_heads == n_kv_heads（无 GQA）
    # 计算注意力分数
    scale = math.sqrt(query.shape[-1])
    scores = torch.matmul(query, k.transpose(-2, -1)) / scale  # [n_heads, 1, seq_len]
    attn = F.softmax(scores, dim=-1)
    output = torch.matmul(attn, v)  # [n_heads, 1, head_dim]

    return output


if __name__ == "__main__":
    print("=" * 60)
    print("PagedAttention 简化实现演示")
    print("=" * 60)

    # 配置
    config = PagedKVConfig(
        n_layers=2, n_kv_heads=4, head_dim=64,
        block_size=4, max_num_blocks=32,
    )
    cache = PagedKVCache(config)

    # 1. 添加多个请求
    print("\n--- 多请求管理 ---")
    for rid in range(3):
        success = cache.add_request(rid)
        print(f"添加请求 {rid}: {'成功' if success else '失败'}")

    # 2. 模拟 token 生成
    print("\n--- 模拟 token 生成 ---")
    request_lengths = {0: 10, 1: 5, 2: 8}  # 每个请求生成的 token 数

    for rid, n_tokens in request_lengths.items():
        for t in range(n_tokens):
            # 每层都要追加
            for layer_idx in range(config.n_layers):
                new_k = torch.randn(config.n_kv_heads, 1, config.head_dim)
                new_v = torch.randn(config.n_kv_heads, 1, config.head_dim)
                cache.append_token(rid, layer_idx, new_k, new_v)

    # 3. 查看状态
    print("\n--- 缓存状态 ---")
    status = cache.status()
    print(f"总块数: {status['total_blocks']}")
    print(f"已分配: {status['allocated_blocks']}")
    print(f"空闲: {status['free_blocks']}")
    print(f"利用率: {status['utilization']}")

    for rid, details in status["request_details"].items():
        print(f"\n请求 {rid}:")
        print(f"  序列长度: {details['seq_len']}")
        print(f"  使用块数: {details['num_blocks']}")
        print(f"  内部碎片: {details['internal_fragmentation']} slots")

    # 4. 分页注意力计算
    print("\n--- 分页注意力计算 ---")
    query = torch.randn(config.n_kv_heads, 1, config.head_dim)
    output = paged_attention(query, cache, request_id=0, layer_idx=0)
    print(f"Query 形状: {query.shape}")
    print(f"Output 形状: {output.shape}")

    # 5. 请求完成后释放
    print("\n--- 请求完成释放 ---")
    cache.remove_request(1)
    status_after = cache.status()
    print(f"释放请求 1 后:")
    print(f"  已分配块: {status_after['allocated_blocks']}")
    print(f"  空闲块: {status_after['free_blocks']}")
    print(f"  利用率: {status_after['utilization']}")

    # 6. 碎片分析
    print("\n--- 碎片分析（与传统连续分配对比）---")
    total_tokens = sum(request_lengths.values())
    max_len = max(request_lengths.values())
    n_requests = len(request_lengths)

    # 传统方式：每个请求按最大长度分配连续空间
    traditional_waste = n_requests * max_len - total_tokens
    traditional_util = total_tokens / (n_requests * max_len)

    # PagedAttention：浪费仅来自最后一个不满的块
    paged_waste = sum(
        (config.block_size - (l % config.block_size)) % config.block_size
        for l in request_lengths.values()
    )
    paged_util = total_tokens / (total_tokens + paged_waste)

    print(f"传统连续分配:")
    print(f"  浪费的 slot 数: {traditional_waste}")
    print(f"  利用率: {traditional_util*100:.1f}%")
    print(f"\nPagedAttention (block_size={config.block_size}):")
    print(f"  浪费的 slot 数: {paged_waste}")
    print(f"  利用率: {paged_util*100:.1f}%")
