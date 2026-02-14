# 环境配置指南

> 本指南帮助你搭建完整的LLM开发环境。建议使用Linux或WSL2环境，Windows用户推荐使用WSL2。

---

## 目录

- [1. 硬件要求](#1-硬件要求)
- [2. 软件环境](#2-软件环境)
- [3. PyTorch安装](#3-pytorch安装)
- [4. 依赖安装](#4-依赖安装)
- [5. 验证安装](#5-验证安装)
- [6. 常见问题](#6-常见问题)

---

## 1. 硬件要求

### 1.1 最低配置

| 组件 | 要求 |
|------|------|
| GPU | NVIDIA GPU, 计算能力 >= 7.0 |
| 显存 | >= 8GB |
| 内存 | >= 16GB |
| 存储 | >= 100GB SSD |

### 1.2 推荐配置

| 组件 | 推荐 |
|------|------|
| GPU | RTX 3090/4090 或 A100 |
| 显存 | >= 24GB |
| 内存 | >= 64GB |
| 存储 | >= 500GB NVMe SSD |

### 1.3 各模块GPU需求

| 模块 | 最小显存 | 推荐显存 |
|------|----------|----------|
| Tokenization | CPU即可 | CPU |
| Embedding | 4GB | 8GB |
| Transformer | 8GB | 16GB |
| Decoder-only | 8GB | 24GB |
| MoE | 16GB | 48GB |
| 预训练 | 24GB | 多卡 |
| SFT/DPO | 8GB | 24GB |
| 量化推理 | 4GB | 8GB |

---

## 2. 软件环境

### 2.1 操作系统

- **推荐**: Ubuntu 22.04 LTS 或 WSL2
- **支持**: macOS (Apple Silicon), Windows 11

### 2.2 CUDA工具包

```bash
# 检查CUDA版本
nvidia-smi

# 推荐CUDA 11.8或12.1
# 安装CUDA Toolkit（如果未安装）
# 参考: https://developer.nvidia.com/cuda-downloads
```

### 2.3 Python环境

```bash
# 推荐使用Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建环境
conda create -n llm-learning python=3.10
conda activate llm-learning
```

---

## 3. PyTorch安装

### 3.1 根据CUDA版本选择

```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3.2 验证PyTorch

```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"GPU数量: {torch.cuda.device_count()}")
```

---

## 4. 依赖安装

### 4.1 核心依赖

```bash
# Transformers生态
pip install transformers>=4.36.0
pip install tokenizers>=0.14.0
pip install datasets>=2.14.0
pip install sentencepiece>=0.1.99
pip install safetensors>=0.4.0

# 训练加速
pip install accelerate>=0.24.0
pip install deepspeed>=0.12.0
pip install peft>=0.7.0
pip install trl>=0.7.0

# 实验追踪
pip install wandb>=0.16.0
pip install tensorboard>=2.15.0

# 工具库
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install tqdm>=4.66.0
pip install matplotlib>=3.7.0
pip install pandas>=2.0.0
```

### 4.2 Flash Attention

```bash
# 需要CUDA 11.6+和GCC 9+
pip install flash-attn --no-build-isolation
```

如果编译失败，可以从预编译wheel安装：

```bash
# 访问 https://github.com/Dao-AILab/flash-attention/releases
# 下载对应CUDA版本的wheel
pip install flash_attn-2.x.x+cu118-cp310-linux_x86_64.whl
```

### 4.3 推理加速

```bash
# vLLM
pip install vllm>=0.3.0

# 量化工具
pip install auto-gptq>=0.6.0
pip install autoawq>=0.2.0
pip install bitsandbytes>=0.41.0
```

### 4.4 JAX支持（可选）

```bash
# CPU版本
pip install jax jaxlib

# GPU版本
pip install jax[cuda11_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## 5. 验证安装

### 5.1 完整验证脚本

创建 `verify_env.py`：

```python
#!/usr/bin/env python3
"""验证LLM学习环境"""

import sys

def check_python():
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    assert version.major == 3 and version.minor >= 8, "需要Python 3.8+"
    print("✓ Python版本检查通过")

def check_pytorch():
    import torch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"cuDNN版本: {torch.backends.cudnn.version()}")
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("✓ PyTorch检查通过")

def check_transformers():
    import transformers
    print(f"Transformers版本: {transformers.__version__}")
    print("✓ Transformers检查通过")

def check_flash_attention():
    try:
        import flash_attn
        print(f"Flash Attention版本: {flash_attn.__version__}")
        print("✓ Flash Attention检查通过")
    except ImportError:
        print("⚠ Flash Attention未安装，部分功能可能较慢")

def check_deepspeed():
    try:
        import deepspeed
        print(f"DeepSpeed版本: {deepspeed.__version__}")
        print("✓ DeepSpeed检查通过")
    except ImportError:
        print("⚠ DeepSpeed未安装")

def check_vllm():
    try:
        import vllm
        print(f"vLLM版本: {vllm.__version__}")
        print("✓ vLLM检查通过")
    except ImportError:
        print("⚠ vLLM未安装")

def main():
    print("="*50)
    print("LLM学习环境验证")
    print("="*50)
    
    check_python()
    print()
    check_pytorch()
    print()
    check_transformers()
    print()
    check_flash_attention()
    print()
    check_deepspeed()
    print()
    check_vllm()
    
    print("\n" + "="*50)
    print("环境验证完成！")
    print("="*50)

if __name__ == "__main__":
    main()
```

运行验证：

```bash
python verify_env.py
```

### 5.2 快速GPU测试

```python
import torch

# 测试CUDA
x = torch.randn(1000, 1000, device="cuda")
y = torch.randn(1000, 1000, device="cuda")

# 预热
z = torch.matmul(x, y)
torch.cuda.synchronize()

# 计时
import time
start = time.time()
for _ in range(100):
    z = torch.matmul(x, y)
torch.cuda.synchronize()
end = time.time()

print(f"100次矩阵乘法耗时: {(end-start)*1000:.2f}ms")
print(f"单次耗时: {(end-start)*10:.2f}ms")
```

---

## 6. 常见问题

### 6.1 CUDA版本不匹配

**问题**: `RuntimeError: CUDA out of memory` 或版本错误

**解决**:

```bash
# 检查CUDA版本
nvidia-smi  # 驱动支持的CUDA版本
nvcc --version  # 安装的CUDA版本

# 重新安装匹配的PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 6.2 Flash Attention安装失败

**问题**: 编译错误

**解决**:

```bash
# 确保GCC版本
gcc --version  # 需要 >= 9.0

# Ubuntu安装GCC 9
sudo apt install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 1

# 或使用预编译版本
pip install flash-attn --no-build-isolation
```

### 6.3 DeepSpeed安装失败

**问题**: 缺少依赖

**解决**:

```bash
# 安装系统依赖
sudo apt install libaio-dev

# 重新安装
pip install deepspeed --no-build-isolation
```

### 6.4 Windows环境问题

**问题**: Windows上部分库不兼容

**解决**:

推荐使用WSL2:

```bash
# 安装WSL2
wsl --install -d Ubuntu-22.04

# 在WSL2中配置CUDA
# 参考: https://docs.nvidia.com/cuda/wsl-user-guide/index.html
```

---

## 7. 开发工具推荐

### 7.1 IDE

- **VS Code**: 配合Python、Jupyter插件
- **PyCharm**: 专业Python IDE
- **Cursor**: AI辅助编程

### 7.2 Jupyter配置

```bash
pip install jupyter ipython

# 生成配置
jupyter notebook --generate-config

# 设置远程访问（可选）
# 编辑 ~/.jupyter/jupyter_notebook_config.py
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
```

### 7.3 实验追踪

```bash
# wandb配置
wandb login

# tensorboard
tensorboard --logdir ./logs --port 6006
```

---

## 附录：完整requirements.txt

```txt
# Core
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
numpy>=1.24.0
scipy>=1.10.0

# Transformers ecosystem
transformers>=4.36.0
tokenizers>=0.14.0
datasets>=2.14.0
sentencepiece>=0.1.99
safetensors>=0.4.0

# Training
accelerate>=0.24.0
deepspeed>=0.12.0
peft>=0.7.0
trl>=0.7.0

# Experiment tracking
wandb>=0.16.0
tensorboard>=2.15.0

# Inference
vllm>=0.3.0
flash-attn>=2.3.0

# Quantization
auto-gptq>=0.6.0
autoawq>=0.2.0
bitsandbytes>=0.41.0

# Utilities
tqdm>=4.66.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
requests>=2.31.0
```
