"""
Checkpoint 管理：保存/恢复/轮换机制

大规模 LLM 预训练中，checkpoint 是训练容错的核心机制。
训练可能持续数周甚至数月，硬件故障是常态而非例外。

Checkpoint 需保存的完整状态 (State Dict 解剖):
1. 模型权重 (Model Weights) - 所有可学习参数
2. 优化器状态 (Optimizer States) - 一阶矩 m, 二阶矩 v (占主要显存)
3. 学习率调度器状态 (LR Scheduler) - 当前步数等
4. 随机数种子 (RNG States) - 确保 Dropout 等的可复现性
5. 数据加载器状态 (Dataloader State) - 已消费的数据位置
6. 训练元信息 - 步数/epoch/最佳指标等

实现策略:
- 频率控制: 按步数保存 vs 按时间保存
- 轮换机制: 保留最近 N 个 checkpoint，自动删除旧的
- 原子写入: 先写临时文件，再重命名，防止写入中崩溃
- 异步保存: 后台线程保存，不阻塞训练循环

参考:
- PyTorch Checkpoint Best Practices
- Megatron-LM Checkpoint System
"""

import os
import time
import shutil
import json
import threading
from typing import Dict, Optional, Any, List
from pathlib import Path

import torch
import torch.nn as nn


class CheckpointManager:
    """
    训练 Checkpoint 管理器

    提供完整的 checkpoint 保存/恢复/轮换功能。

    Args:
        save_dir: checkpoint 保存目录
        max_keep: 最多保留的 checkpoint 数量（轮换机制）
        save_interval_steps: 每隔多少步保存一次
        save_interval_seconds: 每隔多少秒保存一次（与步数取较早者）
        async_save: 是否异步保存（不阻塞训练）
    """

    def __init__(
        self,
        save_dir: str,
        max_keep: int = 5,
        save_interval_steps: int = 1000,
        save_interval_seconds: Optional[float] = None,
        async_save: bool = False,
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_keep = max_keep
        self.save_interval_steps = save_interval_steps
        self.save_interval_seconds = save_interval_seconds
        self.async_save = async_save

        self._last_save_time = time.time()
        self._last_save_step = 0
        self._save_thread = None
        self._checkpoint_list: List[str] = []

        # 扫描已有 checkpoint
        self._scan_existing_checkpoints()

    def _scan_existing_checkpoints(self):
        """扫描目录中已有的 checkpoint"""
        if self.save_dir.exists():
            ckpts = sorted(
                [d.name for d in self.save_dir.iterdir()
                 if d.is_dir() and d.name.startswith("step_")],
                key=lambda x: int(x.split("_")[1])
            )
            self._checkpoint_list = ckpts

    def should_save(self, step: int) -> bool:
        """
        判断是否需要保存 checkpoint

        触发条件（满足任一）:
        1. 距上次保存已超过 save_interval_steps 步
        2. 距上次保存已超过 save_interval_seconds 秒

        Args:
            step: 当前训练步数

        Returns:
            是否需要保存
        """
        # 按步数检查
        if step - self._last_save_step >= self.save_interval_steps:
            return True

        # 按时间检查
        if self.save_interval_seconds is not None:
            elapsed = time.time() - self._last_save_time
            if elapsed >= self.save_interval_seconds:
                return True

        return False

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        dataloader_state: Optional[Dict] = None,
        rng_states: Optional[Dict] = None,
        extra: Optional[Dict] = None,
    ) -> str:
        """
        保存完整的训练 checkpoint

        实现原子写入:
        1. 创建临时目录 step_XXXX_tmp
        2. 将所有状态保存到临时目录
        3. 重命名为正式目录 step_XXXX
        4. 如果步骤 1-2 中间崩溃，临时目录不会被当作有效 checkpoint

        Args:
            step: 当前训练步数
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器（可选）
            dataloader_state: 数据加载器状态（可选）
            rng_states: 随机数生成器状态（可选）
            extra: 额外信息（可选）

        Returns:
            保存路径
        """
        ckpt_name = f"step_{step:08d}"
        ckpt_dir = self.save_dir / ckpt_name
        tmp_dir = self.save_dir / f"{ckpt_name}_tmp"

        def _do_save():
            # 创建临时目录
            tmp_dir.mkdir(parents=True, exist_ok=True)

            try:
                # 1. 模型权重
                torch.save(
                    model.state_dict(),
                    tmp_dir / "model.pt"
                )

                # 2. 优化器状态
                torch.save(
                    optimizer.state_dict(),
                    tmp_dir / "optimizer.pt"
                )

                # 3. 学习率调度器状态
                if scheduler is not None:
                    scheduler_state = (
                        scheduler.state_dict()
                        if hasattr(scheduler, "state_dict")
                        else {}
                    )
                    torch.save(scheduler_state, tmp_dir / "scheduler.pt")

                # 4. 随机数种子状态
                if rng_states is not None:
                    torch.save(rng_states, tmp_dir / "rng_states.pt")

                # 5. 数据加载器状态
                if dataloader_state is not None:
                    torch.save(dataloader_state, tmp_dir / "dataloader.pt")

                # 6. 训练元信息
                meta = {
                    "step": step,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_class": model.__class__.__name__,
                }
                if extra:
                    meta.update(extra)
                with open(tmp_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2, ensure_ascii=False)

                # 原子重命名: tmp -> 正式
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)
                tmp_dir.rename(ckpt_dir)

                # 更新 checkpoint 列表
                self._checkpoint_list.append(ckpt_name)
                self._last_save_step = step
                self._last_save_time = time.time()

                # 轮换: 删除多余的旧 checkpoint
                self._rotate_checkpoints()

            except Exception as e:
                # 保存失败，清理临时目录
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir)
                raise RuntimeError(f"Checkpoint 保存失败: {e}") from e

        if self.async_save:
            # 异步保存: 在后台线程中执行
            if self._save_thread is not None and self._save_thread.is_alive():
                self._save_thread.join()  # 等待上一次保存完成
            self._save_thread = threading.Thread(target=_do_save, daemon=True)
            self._save_thread.start()
        else:
            _do_save()

        return str(ckpt_dir)

    def _rotate_checkpoints(self):
        """
        轮换机制: 保留最近 max_keep 个 checkpoint

        策略: 按步数排序，删除最旧的 checkpoint。
        保留策略可以更灵活（如保留里程碑 checkpoint），
        这里实现最简单的 FIFO 策略。
        """
        while len(self._checkpoint_list) > self.max_keep:
            oldest = self._checkpoint_list.pop(0)
            oldest_path = self.save_dir / oldest
            if oldest_path.exists():
                shutil.rmtree(oldest_path)

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_rng: bool = True,
        map_location: Optional[str] = None,
    ) -> Dict:
        """
        从 checkpoint 恢复训练状态

        精确恢复: 所有状态完全恢复 -> 训练完全可重现

        Args:
            model: 模型（将被加载权重）
            optimizer: 优化器（可选，将被加载状态）
            scheduler: 学习率调度器（可选）
            checkpoint_path: 指定 checkpoint 路径（默认加载最新的）
            load_optimizer: 是否加载优化器状态
            load_scheduler: 是否加载调度器状态
            load_rng: 是否恢复随机数状态
            map_location: 设备映射（如 "cpu"）

        Returns:
            包含元信息的字典（step, timestamp 等）
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise FileNotFoundError(
                    f"目录 {self.save_dir} 中没有找到 checkpoint"
                )

        ckpt_dir = Path(checkpoint_path)

        # 1. 加载模型权重
        model_path = ckpt_dir / "model.pt"
        if model_path.exists():
            state_dict = torch.load(model_path, map_location=map_location, weights_only=True)
            model.load_state_dict(state_dict)

        # 2. 加载优化器状态
        if load_optimizer and optimizer is not None:
            opt_path = ckpt_dir / "optimizer.pt"
            if opt_path.exists():
                opt_state = torch.load(opt_path, map_location=map_location, weights_only=True)
                optimizer.load_state_dict(opt_state)

        # 3. 加载调度器状态
        if load_scheduler and scheduler is not None:
            sched_path = ckpt_dir / "scheduler.pt"
            if sched_path.exists():
                sched_state = torch.load(sched_path, map_location=map_location, weights_only=True)
                if hasattr(scheduler, "load_state_dict"):
                    scheduler.load_state_dict(sched_state)

        # 4. 恢复随机数状态
        if load_rng:
            rng_path = ckpt_dir / "rng_states.pt"
            if rng_path.exists():
                rng_states = torch.load(rng_path, map_location="cpu", weights_only=False)
                from .utils import set_rng_states
                set_rng_states(rng_states)

        # 5. 加载元信息
        meta = {}
        meta_path = ckpt_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

        # 6. 数据加载器状态
        dl_path = ckpt_dir / "dataloader.pt"
        if dl_path.exists():
            meta["dataloader_state"] = torch.load(
                dl_path, map_location="cpu", weights_only=True
            )

        return meta

    def get_latest_checkpoint(self) -> Optional[str]:
        """
        获取最新的 checkpoint 路径

        Returns:
            最新 checkpoint 的完整路径，或 None
        """
        if not self._checkpoint_list:
            self._scan_existing_checkpoints()
        if self._checkpoint_list:
            return str(self.save_dir / self._checkpoint_list[-1])
        return None

    def list_checkpoints(self) -> List[Dict]:
        """
        列出所有可用的 checkpoint

        Returns:
            checkpoint 信息列表
        """
        self._scan_existing_checkpoints()
        results = []
        for name in self._checkpoint_list:
            ckpt_dir = self.save_dir / name
            meta_path = ckpt_dir / "meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            # 计算 checkpoint 大小
            total_size = sum(
                f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file()
            )
            meta["path"] = str(ckpt_dir)
            meta["size_MB"] = total_size / 1e6
            results.append(meta)
        return results

    def wait_for_save(self):
        """等待异步保存完成"""
        if self._save_thread is not None and self._save_thread.is_alive():
            self._save_thread.join()


def save_checkpoint_simple(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    **kwargs,
) -> None:
    """
    简单的 checkpoint 保存函数（不使用 CheckpointManager）

    适用于小规模实验和快速原型。

    Args:
        path: 保存路径
        model: 模型
        optimizer: 优化器
        step: 当前步数
        **kwargs: 额外状态
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    checkpoint.update(kwargs)

    # 原子写入
    tmp_path = path + ".tmp"
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint_simple(
    path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Optional[str] = None,
) -> Dict:
    """
    简单的 checkpoint 加载函数

    Args:
        path: checkpoint 路径
        model: 模型
        optimizer: 优化器（可选）
        map_location: 设备映射

    Returns:
        checkpoint 中的额外信息
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 返回除模型和优化器状态外的信息
    meta = {k: v for k, v in checkpoint.items()
            if k not in ("model_state_dict", "optimizer_state_dict")}
    return meta


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("Checkpoint 管理器 演示")
    print("=" * 60)

    # 创建简单模型和优化器
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    # 1. 简单版 checkpoint
    print("\n--- 简单版 checkpoint ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "simple_ckpt.pt")

        # 模拟训练
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # 保存
        save_checkpoint_simple(
            ckpt_path, model, optimizer, step=100, loss=3.14
        )
        print(f"已保存 checkpoint: {ckpt_path}")
        print(f"文件大小: {os.path.getsize(ckpt_path) / 1024:.1f} KB")

        # 加载
        model2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        opt2 = torch.optim.AdamW(model2.parameters(), lr=3e-4)
        meta = load_checkpoint_simple(ckpt_path, model2, opt2)
        print(f"恢复的元信息: {meta}")

        # 验证权重一致
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert torch.equal(p1, p2), f"权重不一致: {n1}"
        print("权重验证通过!")

    # 2. CheckpointManager
    print("\n--- CheckpointManager ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            save_dir=tmpdir,
            max_keep=3,
            save_interval_steps=5,
        )

        # 模拟训练循环
        for step in range(1, 21):
            x = torch.randn(4, 64)
            loss = model(x).sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if manager.should_save(step):
                path = manager.save(
                    step=step,
                    model=model,
                    optimizer=optimizer,
                    extra={"loss": float(loss)},
                )
                print(f"  step {step}: 已保存 checkpoint -> {path}")

        # 列出所有 checkpoint
        print("\n可用的 checkpoint:")
        for info in manager.list_checkpoints():
            print(f"  step={info.get('step', '?')}, "
                  f"size={info.get('size_MB', 0):.2f} MB, "
                  f"time={info.get('timestamp', '?')}")

        # 恢复最新 checkpoint
        model3 = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
        opt3 = torch.optim.AdamW(model3.parameters(), lr=3e-4)
        meta = manager.load(model3, opt3)
        print(f"\n恢复的 checkpoint: step = {meta.get('step', '?')}")
