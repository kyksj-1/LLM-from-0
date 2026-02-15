"""
稀疏自编码器 (Sparse Autoencoder, SAE) 实现

本模块实现了用于机械可解释性研究的稀疏自编码器。
SAE 的核心思想是将模型激活值分解为可解释的稀疏特征。

数学基础:
- 编码: z = ReLU(W_enc @ x + b_enc)
- 解码: x_hat = W_dec @ z + b_dec
- 训练目标: L = ||x - x_hat||^2 + lambda * ||z||_1
  - 重建损失确保信息保留
  - L1 正则促进稀疏性，使每个特征只在少数样本上激活

关键设计:
- 字典大小 d_sae 远大于输入维度 d_model (通常 4x-64x)
- 这种过完备表示使得 SAE 能从多义神经元中提取单义特征
- 解码器列向量归一化，防止特征缩放退化

参考:
- Bricken et al. (2023). Towards Monosemanticity.
- Cunningham et al. (2023). Sparse Autoencoders Find Highly Interpretable Features.
- Templeton et al. (2024). Scaling Monosemanticity.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, Optional
import math


class SparseAutoencoder(nn.Module):
    """
    稀疏自编码器

    将 d_model 维激活值映射到 d_sae 维稀疏表示，再重建回 d_model 维。
    通过 L1 正则化强制稀疏性，使每个字典特征具有明确语义。

    Args:
        d_model: 输入激活值的维度（模型隐藏层维度）
        d_sae: SAE 字典大小（特征数量），通常为 d_model 的 4-64 倍
        l1_coeff: L1 正则化系数，控制稀疏程度
        tied_weights: 是否使用绑定权重（W_dec = W_enc^T）
    """

    def __init__(
        self,
        d_model: int,
        d_sae: int,
        l1_coeff: float = 1e-3,
        tied_weights: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff
        self.tied_weights = tied_weights

        # 编码器权重和偏置
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # 解码器权重和偏置
        if not tied_weights:
            self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """
        使用 Kaiming 均匀初始化权重

        编码器和解码器的初始化对 SAE 训练至关重要:
        - 过大的初始化会导致所有特征同时激活
        - 过小的初始化会导致特征难以学习
        """
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        if not self.tied_weights:
            nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))

        # 归一化解码器列（每个特征方向为单位向量）
        with torch.no_grad():
            dec_weight = self.W_enc.t() if self.tied_weights else self.W_dec
            dec_weight.div_(dec_weight.norm(dim=0, keepdim=True) + 1e-8)

    @property
    def decoder_weight(self) -> torch.Tensor:
        """获取解码器权重（支持绑定权重模式）"""
        if self.tied_weights:
            return self.W_enc.t()
        return self.W_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码: 将输入激活值映射为稀疏特征表示

        z = ReLU(W_enc @ (x - b_dec) + b_enc)

        注意: 先减去解码器偏置(centering)，使编码器处理零均值数据

        Args:
            x: 输入激活值 [batch, d_model]

        Returns:
            z: 稀疏特征激活 [batch, d_sae]
        """
        # 中心化: 减去解码器偏置
        x_centered = x - self.b_dec

        # 线性变换 + ReLU 激活
        z = torch.relu(x_centered @ self.W_enc.t() + self.b_enc)

        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码: 从稀疏表示重建原始激活值

        x_hat = W_dec @ z + b_dec

        Args:
            z: 稀疏特征激活 [batch, d_sae]

        Returns:
            x_hat: 重建的激活值 [batch, d_model]
        """
        x_hat = z @ self.decoder_weight.t() + self.b_dec
        return x_hat

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播: 编码 -> 解码 -> 计算损失

        Args:
            x: 输入激活值 [batch, d_model]

        Returns:
            x_hat: 重建的激活值
            z: 稀疏特征激活
            loss_dict: 包含各项损失的字典
        """
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, self.compute_loss(x, x_hat, z)

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练损失

        L_total = L_reconstruct + lambda * L_sparsity

        其中:
        - L_reconstruct = ||x - x_hat||^2 / ||x||^2  (归一化重建误差)
        - L_sparsity = mean(||z||_1)  (L1 范数促进稀疏性)

        Args:
            x: 原始激活值
            x_hat: 重建的激活值
            z: 稀疏特征激活

        Returns:
            loss_dict: 包含 total_loss, reconstruction_loss, sparsity_loss
        """
        # 重建损失: 归一化 MSE
        reconstruction_loss = (
            (x - x_hat).pow(2).sum(dim=-1).mean()
            / (x.pow(2).sum(dim=-1).mean() + 1e-8)
        )

        # 稀疏性损失: L1 范数
        sparsity_loss = z.abs().sum(dim=-1).mean()

        # 总损失
        total_loss = reconstruction_loss + self.l1_coeff * sparsity_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "sparsity_loss": sparsity_loss,
        }

    @torch.no_grad()
    def normalize_decoder(self):
        """
        归一化解码器列向量

        确保每个特征方向为单位向量，防止模型通过缩放解码器来降低 L1 损失
        （如果不归一化，模型可以增大解码器权重同时减小编码器输出来作弊）
        """
        if not self.tied_weights:
            norms = self.W_dec.norm(dim=0, keepdim=True)
            self.W_dec.div_(norms + 1e-8)

    def get_feature_activations(
        self, x: torch.Tensor, top_k: int = 10
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取每个样本的 Top-K 激活特征

        Args:
            x: 输入激活值 [batch, d_model]
            top_k: 返回前 k 个最活跃的特征

        Returns:
            values: Top-K 激活值 [batch, top_k]
            indices: Top-K 特征索引 [batch, top_k]
        """
        z = self.encode(x)
        values, indices = z.topk(top_k, dim=-1)
        return values, indices

    def get_sparsity_stats(self, x: torch.Tensor) -> Dict[str, float]:
        """
        计算稀疏性统计信息

        Args:
            x: 输入激活值 [batch, d_model]

        Returns:
            stats: 包含各种稀疏性指标的字典
        """
        z = self.encode(x)

        # 活跃特征数（非零激活的平均数）
        n_active = (z > 0).float().sum(dim=-1).mean().item()

        # 稀疏度（零激活的比例）
        sparsity = 1.0 - (z > 0).float().mean().item()

        # 每个特征的激活频率
        feature_freq = (z > 0).float().mean(dim=0)

        # 死特征数（从未激活的特征）
        n_dead = (feature_freq == 0).sum().item()

        return {
            "avg_active_features": n_active,
            "sparsity_ratio": sparsity,
            "n_dead_features": n_dead,
            "dead_ratio": n_dead / self.d_sae,
            "max_activation": z.max().item(),
            "mean_activation": z[z > 0].mean().item() if (z > 0).any() else 0.0,
        }


class SAETrainer:
    """
    SAE 训练器

    封装训练循环和常用训练技巧:
    - 解码器列归一化（每步或定期）
    - 死特征重激活
    - 学习率预热

    Args:
        sae: 稀疏自编码器模型
        lr: 学习率
        warmup_steps: 预热步数
        resample_dead_every: 每隔多少步重新激活死特征
        dead_threshold: 判断死特征的激活频率阈值
    """

    def __init__(
        self,
        sae: SparseAutoencoder,
        lr: float = 1e-4,
        warmup_steps: int = 1000,
        resample_dead_every: int = 5000,
        dead_threshold: float = 1e-6,
    ):
        self.sae = sae
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.resample_dead_every = resample_dead_every
        self.dead_threshold = dead_threshold

        self.optimizer = optim.Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999))
        self.step_count = 0

        # 跟踪每个特征的激活频率（用于检测死特征）
        self.feature_activation_count = torch.zeros(sae.d_sae)

    def get_lr_scale(self) -> float:
        """线性预热学习率调度"""
        if self.step_count < self.warmup_steps:
            return self.step_count / max(self.warmup_steps, 1)
        return 1.0

    def train_step(self, activations: torch.Tensor) -> Dict[str, float]:
        """
        单步训练

        Args:
            activations: 模型激活值 [batch, d_model]

        Returns:
            metrics: 训练指标字典
        """
        self.sae.train()
        self.step_count += 1

        # 调整学习率
        lr_scale = self.get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr * lr_scale

        # 前向传播
        x_hat, z, loss_dict = self.sae(activations)

        # 反向传播
        self.optimizer.zero_grad()
        loss_dict["total_loss"].backward()
        self.optimizer.step()

        # 归一化解码器
        self.sae.normalize_decoder()

        # 更新特征激活计数
        with torch.no_grad():
            self.feature_activation_count += (
                (z > 0).float().sum(dim=0).cpu()
            )

        # 定期重新激活死特征
        if (
            self.resample_dead_every > 0
            and self.step_count % self.resample_dead_every == 0
        ):
            n_resampled = self._resample_dead_features(activations)
        else:
            n_resampled = 0

        return {
            "total_loss": loss_dict["total_loss"].item(),
            "reconstruction_loss": loss_dict["reconstruction_loss"].item(),
            "sparsity_loss": loss_dict["sparsity_loss"].item(),
            "lr": self.lr * lr_scale,
            "step": self.step_count,
            "n_resampled": n_resampled,
        }

    @torch.no_grad()
    def _resample_dead_features(self, activations: torch.Tensor) -> int:
        """
        重新激活死特征

        死特征是训练过程中从不激活的特征方向。
        重激活策略: 将死特征的编码器/解码器权重重新初始化为
        当前高重建误差样本的方向。

        Args:
            activations: 当前批次的激活值

        Returns:
            n_resampled: 重新激活的特征数量
        """
        # 计算平均激活频率
        avg_freq = self.feature_activation_count / max(self.step_count, 1)

        # 找到死特征
        dead_mask = avg_freq < self.dead_threshold
        n_dead = dead_mask.sum().item()

        if n_dead == 0:
            return 0

        dead_indices = torch.where(dead_mask)[0]

        # 计算重建误差，选择高误差样本
        x_hat, _, _ = self.sae(activations)
        reconstruction_errors = (activations - x_hat).pow(2).sum(dim=-1)

        # 按误差大小采样
        probs = reconstruction_errors / reconstruction_errors.sum()
        n_to_sample = min(n_dead, len(activations))
        sampled_indices = torch.multinomial(probs, n_to_sample, replacement=True)

        # 用高误差样本的方向重新初始化死特征
        for i, dead_idx in enumerate(dead_indices[:n_to_sample]):
            sample_idx = sampled_indices[i % n_to_sample]
            direction = activations[sample_idx] - self.sae.b_dec

            # 归一化方向
            direction = direction / (direction.norm() + 1e-8)

            # 设置编码器权重
            self.sae.W_enc.data[dead_idx] = direction * 0.1

            # 设置解码器权重
            if not self.sae.tied_weights:
                self.sae.W_dec.data[:, dead_idx] = direction

            # 重置偏置
            self.sae.b_enc.data[dead_idx] = 0.0

        # 重置激活计数
        self.feature_activation_count.zero_()

        return n_to_sample


def generate_synthetic_data(
    n_samples: int = 10000,
    d_model: int = 64,
    n_true_features: int = 128,
    sparsity: float = 0.05,
    noise_std: float = 0.01,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    生成合成数据用于 SAE 实验

    模拟 Superposition 场景: n_true_features 个真实特征编码在 d_model 维空间中
    (n_true_features > d_model)

    Args:
        n_samples: 样本数量
        d_model: 激活值维度
        n_true_features: 真实特征数量 (> d_model, 模拟过完备)
        sparsity: 每个特征的激活概率
        noise_std: 噪声标准差

    Returns:
        activations: 合成的激活值 [n_samples, d_model]
        feature_coeffs: 真实特征系数 [n_samples, n_true_features]
        feature_directions: 特征方向矩阵 [d_model, n_true_features]
    """
    # 随机生成特征方向（模拟超完备基）
    feature_directions = torch.randn(d_model, n_true_features)
    feature_directions = feature_directions / feature_directions.norm(dim=0, keepdim=True)

    # 生成稀疏特征系数
    # 每个特征以概率 sparsity 激活，激活值服从指数分布
    active_mask = (torch.rand(n_samples, n_true_features) < sparsity).float()
    feature_magnitudes = torch.exponential(torch.ones(n_samples, n_true_features))
    feature_coeffs = active_mask * feature_magnitudes

    # 合成激活值 = 特征方向 @ 特征系数 + 噪声
    activations = feature_coeffs @ feature_directions.t()
    activations += torch.randn_like(activations) * noise_std

    return activations, feature_coeffs, feature_directions


if __name__ == "__main__":
    print("=" * 60)
    print("稀疏自编码器 (SAE) 演示")
    print("=" * 60)

    # ---- 1. 生成合成数据 ----
    print("\n[1] 生成合成数据...")
    d_model = 64
    n_true_features = 256  # 4x 过完备
    activations, true_coeffs, true_directions = generate_synthetic_data(
        n_samples=5000,
        d_model=d_model,
        n_true_features=n_true_features,
        sparsity=0.05,
    )
    print(f"    激活值形状: {activations.shape}")
    print(f"    真实特征数: {n_true_features}")
    print(f"    每样本平均活跃特征: {(true_coeffs > 0).float().sum(dim=-1).mean():.1f}")

    # ---- 2. 创建并训练 SAE ----
    print("\n[2] 创建 SAE...")
    d_sae = 512  # 8x 字典大小
    sae = SparseAutoencoder(
        d_model=d_model,
        d_sae=d_sae,
        l1_coeff=5e-3,
    )
    print(f"    输入维度: {d_model}")
    print(f"    字典大小: {d_sae}")
    print(f"    参数量: {sum(p.numel() for p in sae.parameters()):,}")

    trainer = SAETrainer(
        sae=sae,
        lr=3e-4,
        warmup_steps=100,
        resample_dead_every=500,
    )

    print("\n[3] 训练 SAE...")
    batch_size = 256
    n_epochs = 5
    for epoch in range(n_epochs):
        # 随机打乱数据
        perm = torch.randperm(len(activations))
        epoch_losses = []

        for i in range(0, len(activations), batch_size):
            batch = activations[perm[i : i + batch_size]]
            metrics = trainer.train_step(batch)
            epoch_losses.append(metrics["total_loss"])

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"    Epoch {epoch + 1}/{n_epochs}: "
              f"loss={avg_loss:.4f}, "
              f"recon={metrics['reconstruction_loss']:.4f}, "
              f"sparsity={metrics['sparsity_loss']:.4f}")

    # ---- 3. 分析结果 ----
    print("\n[4] 分析稀疏性...")
    stats = sae.get_sparsity_stats(activations[:1000])
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    # ---- 4. 查看 Top-K 特征 ----
    print("\n[5] Top-K 特征激活示例...")
    sample = activations[:5]
    values, indices = sae.get_feature_activations(sample, top_k=5)
    for i in range(min(3, len(sample))):
        print(f"    样本 {i}: 特征索引={indices[i].tolist()}, "
              f"激活值={[f'{v:.3f}' for v in values[i].tolist()]}")

    print("\n演示完成!")
