# EIT 数据格式说明

本文档详细说明 `data/` 目录下的数据格式，以及 EIM（Electrical Impedance Map）数据的含义。

## 1. 目录结构

```
data/
├── train/          # 训练数据集
├── valid/          # 验证数据集
├── test/           # 测试数据集
├── test2017/       # UEF2017 真实数据集（22个样本）
├── test2023/       # KTC2023 真实数据集（25个样本）
├── mean.pth        # EIM 数据均值（用于归一化）
└── std.pth         # EIM 数据标准差（用于归一化）
```

## 2. NPZ 文件格式

每个 `.npz` 文件包含以下字段：

| 字段 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| `xs` | (128, 128) | float32 | **电导率分布 (Ground Truth)** - 用作训练标签 |
| `xs_gn` | (128, 128) | float32 | Gauss-Newton 方法重建结果（参考用） |
| `TR` | (128, 128) | float32 | Tikhonov 正则化重建结果（参考用） |
| `ys` | (208, 1) | float32 | **EIM 测量数据** - 用作网络输入 |

### 2.1 字段详细说明

#### `xs` - 电导率分布 (Ground Truth)
- 128×128 像素的电导率图像
- 值范围通常为 [-1, 1] 或 [0, 1]
- 表示成像区域内的电导率变化
- **这是我们要重建的目标**

#### `ys` - EIM 测量数据
- 208 维向量（实际存储为 208×1）
- 包含边界电压测量值
- **这是网络的输入**
- 详见下文 "EIM 数据格式" 部分

#### `xs_gn` 和 `TR`
- 传统算法的重建结果
- 仅供参考和对比
- 不参与深度学习训练

## 3. EIM 数据格式详解

### 3.1 什么是 EIM？

**EIM (Electrical Impedance Map)** 是将一维边界电压测量数据重组为二维矩阵的表示方法。这种表示保留了电压测量与电极位置之间的空间关系，有助于深度学习模型学习。

### 3.2 测量系统配置

本项目使用 **16 电极系统**，采用 **相邻激励-相邻测量** 模式：

```
        电极 0
          ●
      ●       ●  电极 1
   15
  ●               ●  2

 14●     成像区域    ●3

  ●               ●  4
   13
      ●       ●  5
          ●
        电极 8
```

### 3.3 测量原理

1. **电流激励**：在相邻两个电极之间注入电流（如电极 0-1）
2. **电压测量**：在其他相邻电极对之间测量电压
3. **排除的电极**：每个激励模式下，3 个电极的数据被排除：
   - 2 个激励电极（电流注入点）
   - 1 个参考电极（通常是激励电极的相邻电极）

### 3.4 数据维度计算

```
总测量数 = 激励模式数 × 每模式有效测量数
       = 16 × 13
       = 208
```

- **16**：电流激励模式数量（每对相邻电极轮流作为激励对）
- **13**：每个激励模式下的有效测量数（16 - 3 = 13）

### 3.5 向量到矩阵的转换

208 维向量可以重塑为 16×16 的 EIM 矩阵：

```
EIM 矩阵结构（16×16）：
        测量电极位置
        0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
激励  0  [0   0   0   V   V   V   V   V   V   V   V   V   V   V   V   0 ]
模式  1  [0   0   0   0   V   V   V   V   V   V   V   V   V   V   V   V ]
      2  [V   0   0   0   0   V   V   V   V   V   V   V   V   V   V   V ]
      3  [V   V   0   0   0   0   V   V   V   V   V   V   V   V   V   V ]
      ...
     15  [0   V   V   V   V   V   V   V   V   V   V   V   V   V   0   0 ]
```

其中：
- `V` 表示有效电压测量值
- `0` 表示被排除的位置（激励电极或参考电极）

每行对应一个激励模式，被排除的 3 个位置为：
- 当前激励电极 `i`
- 前一个电极 `(i-1) % 16`
- 后一个电极 `(i+1) % 16`

### 3.6 代码示例

```python
import numpy as np

def eim_vector_to_matrix(ys_vector, num_electrodes=16):
    """将 208 维 EIM 向量转换为 16×16 矩阵"""
    eim_matrix = np.zeros((num_electrodes, num_electrodes))
    idx = 0

    for i in range(num_electrodes):
        # 被排除的电极位置
        skip_positions = {(i - 1) % num_electrodes,
                          i,
                          (i + 1) % num_electrodes}

        for j in range(num_electrodes):
            if j not in skip_positions:
                eim_matrix[i, j] = ys_vector[idx]
                idx += 1

    return eim_matrix

# 使用示例
data = np.load('data/test2023/0_1.npz')
ys = data['ys'].squeeze()  # (208,)
eim = eim_vector_to_matrix(ys)  # (16, 16)
```

## 4. KTC2023 数据的特殊处理

KTC2023 数据集原始使用 **32 电极系统** 和 **skip-1 激励模式**。转换为 16 电极等效数据的方法：

1. **激励模式转换**：
   - 32 电极 skip-1 模式：电极 1-3, 3-5, ..., 31-1（共 16 个模式）
   - 等效于 16 电极相邻激励模式

2. **电压合并**：
   - 将相邻两对电极的电压测量值相加
   - 这在物理上等价于将两个相邻电极视为一个更大的电极

参考论文：*"A Conditional Diffusion Model for Electrical Impedance Tomography Image Reconstruction"* (CDEIT)

## 5. 归一化参数

`mean.pth` 和 `std.pth` 存储了 EIM 数据的归一化参数：

```python
import torch

mean = torch.load('data/mean.pth')  # shape: (1, 16, 13)
std = torch.load('data/std.pth')    # shape: (1, 16, 13)

# 归一化
ys_normalized = (ys - mean) / std
```

注意：这些参数是按照 (16, 13) 的格式存储的，每行 13 个有效测量值。

## 6. 数据加载示例

```python
import numpy as np
import torch
from pathlib import Path

def load_eit_sample(file_path):
    """加载单个 EIT 数据样本"""
    data = np.load(file_path)

    # 测量数据（网络输入）
    measurements = data['ys'].squeeze()  # (208,)

    # 电导率分布（Ground Truth，网络标签）
    conductivity = data['xs']  # (128, 128)

    return measurements, conductivity

# 批量加载
data_dir = Path('data/train')
for npz_file in data_dir.glob('*.npz'):
    ys, xs = load_eit_sample(npz_file)
    print(f'{npz_file.name}: ys={ys.shape}, xs={xs.shape}')
```

## 7. 可视化示例

```python
import matplotlib.pyplot as plt
import numpy as np

def visualize_sample(file_path):
    """可视化单个样本"""
    data = np.load(file_path)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # EIM 矩阵
    ys = data['ys'].squeeze()
    eim = eim_vector_to_matrix(ys)
    axes[0].imshow(eim, cmap='jet')
    axes[0].set_xlabel('Measurement electrode')
    axes[0].set_ylabel('Excitation pattern')

    # Ground Truth
    axes[1].imshow(data['xs'], cmap='jet')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')

    # Tikhonov 重建
    axes[2].imshow(data['TR'], cmap='jet')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')

    plt.tight_layout()
    plt.show()
```

## 8. 参考文献

1. **CDEIT 论文**: Shi, S., Kang, R., & Liatsis, P. (2024). "A Conditional Diffusion Model for Electrical Impedance Tomography Image Reconstruction."

2. **KTC2023 数据集**: Räsänen, M., et al. (2024). "Kuopio Tomography Challenge 2023." https://zenodo.org/records/10418802

3. **UEF2017 数据集**: Hauptmann, A., et al. (2017). "Open 2D Electrical Impedance Tomography Data Archive." https://zenodo.org/records/1203914

4. **DeepDbar 论文**: Hamilton, S. J., & Hauptmann, A. (2018). "Deep D-bar: Real-time Electrical Impedance Tomography Imaging with Deep Neural Networks."
