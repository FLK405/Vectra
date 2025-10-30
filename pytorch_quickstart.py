"""
PyTorch 2.9 CPU版本 - 快速参考指南
适用于Intel Core i5-12500H (AVX2支持)
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 60)
print("PyTorch 2.9 快速参考指南")
print("=" * 60)
print(f"\nPyTorch版本: {torch.__version__}")
print(f"CPU加速: {torch.backends.cpu.get_cpu_capability()}")
print("状态: ✅ 已就绪\n")

# 示例1: 创建和训练简单模型
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

print("✅ 模型创建成功")
print("📝 提示: 查看此文件学习PyTorch基础用法")
