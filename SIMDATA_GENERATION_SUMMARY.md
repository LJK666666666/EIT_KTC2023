# EIT仿真数据集生成 - 项目总结

## 完成的工作

### 1. 创建的核心文件

#### `src/sim_dataset.py`
实现了 `SimulatedEITDataset` 类，这是一个PyTorch Dataset类：
- ✅ 使用 `src/ktc_methods` 中的KTC2023官方组件
- ✅ 加载有限元网格（`load_mesh()` 函数）
- ✅ 生成随机导电率分布（圆形、矩形、多边形）
- ✅ 使用EIT正向求解器计算测量数据
- ✅ 添加可配置的测量噪声
- ✅ 支持PyTorch DataLoader

#### `src/generate_simdata.py`
批量生成仿真数据的主脚本：
- ✅ 生成10000条level1数据（不移除电极）
- ✅ 自动创建文件夹结构
- ✅ 保存ground truth和measurements
- ✅ 显示进度条（tqdm）
- ✅ 数据验证功能
- ✅ 支持测试模式（--test）
- ✅ 可配置样本数量（--num_samples）

#### `src/monitor_progress.py`
进度监控脚本：
- ✅ 实时显示生成进度
- ✅ 计算生成速度
- ✅ 估计剩余时间（ETA）
- ✅ 可配置检查间隔

#### `src/example_use_simdataset.py`
使用示例脚本：
- ✅ 展示如何使用Dataset类
- ✅ DataLoader集成示例
- ✅ 数据可视化
- ✅ 数据统计分析

### 2. 数据生成规格

**Level 1 (不移除电极)**

- **样本数量**: 10,000条
- **网格**: Mesh_dense.mat (3766个节点)
- **电极**: 32个（完整配置）
- **测量维度**: 992 (32×31)
- **噪声水平**: 1% (noise_std1=0.1)
- **分割类别**: 3类（背景、低电导率、高电导率）

**Ground Truth格式**:
- 形状: (256, 256)
- 类型: float32
- 值: 0/1/2

**Measurements格式**:
- 形状: (992,)
- 类型: float32
- 单位: 电压差分值

### 3. 文件结构

```
EIT_KTC2023/
├── src/
│   ├── sim_dataset.py              # 数据集类
│   ├── generate_simdata.py         # 批量生成脚本
│   ├── monitor_progress.py         # 进度监控
│   ├── example_use_simdataset.py   # 使用示例
│   └── ktc_methods/                # KTC2023官方方法
│       ├── KTCFwd.py               # 正向求解器
│       ├── KTCMeshing.py           # 网格处理
│       ├── KTCAux.py               # 辅助函数
│       ├── Mesh_dense.mat          # 密集网格
│       └── Mesh_sparse.mat         # 稀疏网格
│
└── SimData/                        # 生成的数据
    ├── README.md                   # 数据说明文档
    └── level_1/
        ├── gt/                     # Ground truth
        │   ├── gt_00000.npy
        │   ├── gt_00001.npy
        │   └── ... (10000个文件)
        └── measurements/           # EIT测量
            ├── u_00000.npy
            ├── u_00001.npy
            └── ... (10000个文件)
```

## 使用方法

### 快速开始

```python
from src.sim_dataset import SimulatedEITDataset
from torch.utils.data import DataLoader

# 创建数据集
dataset = SimulatedEITDataset(length=100)

# 获取一个样本
phantom, measurements = dataset[0]
print(f"Phantom shape: {phantom.shape}")        # (256, 256)
print(f"Measurements shape: {measurements.shape}") # (992,)

# 使用DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_phantom, batch_meas in dataloader:
    # 训练模型
    pass
```

### 加载已生成的数据

```python
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SimDataLoader(Dataset):
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        gt = np.load(f'SimData/level_1/gt/gt_{idx:05d}.npy')
        meas = np.load(f'SimData/level_1/measurements/u_{idx:05d}.npy')
        return torch.from_numpy(gt).float(), torch.from_numpy(meas).float()

dataset = SimDataLoader()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 生成数据

```bash
# 完整生成（10000条）
python src/generate_simdata.py --num_samples 10000

# 测试生成（10条）
python src/generate_simdata.py --test

# 监控进度（另一个终端）
python src/monitor_progress.py --target 10000 --interval 60
```

## 数据生成状态

**当前进度**: 47/10000 (0.47%)
**当前速度**: 约4.5秒/样本
**预计总时间**: 约12.5小时
**开始时间**: 2025-10-28 15:23
**预计完成**: 2025-10-29 03:53

## 技术细节

### 导电率生成
- 背景导电率: 1.0
- 低电导率区域: 0.5
- 高电导率区域: 1.5
- 包含物形状: 随机（圆形/矩形/多边形）
- 包含物数量: 1-3个

### EIT仿真
- 正向求解器: KTCFwd.EITFEM
- 有限元方法: 二阶三角形单元
- 测量模式: 相邻电流注入
- 电极接触阻抗: 1.0 Ω

### 数据质量
- ✅ 物理一致性（使用官方正向求解器）
- ✅ 噪声模型（1%测量噪声）
- ✅ 几何约束（圆形域，半径0.098m）
- ✅ 碰撞检测（包含物不重叠）

## 磁盘占用

- 单个样本: ~261 KB (257KB GT + 4KB 测量)
- 10000个样本: **~2.6 GB**

## 后续工作

可以使用这些数据进行：
1. 训练深度学习重建模型
2. 测试不同的逆问题求解算法
3. 评估重建质量
4. 生成不同noise level的数据
5. 生成level 2-5的数据（带电极移除）

## 注意事项

1. 数据生成正在后台运行（shell ID: e5d1d6）
2. 可以使用 `BashOutput` 工具检查进度
3. 生成日志保存在 `simdata_generation.log`
4. 数据保存在Google Drive上，确保有足够空间
5. 如果中断，可以修改脚本从断点继续生成

## 验证

运行以下命令验证数据：

```bash
# 检查文件数量
ls SimData/level_1/gt/*.npy | wc -l

# 验证数据格式
python -c "
import numpy as np
gt = np.load('SimData/level_1/gt/gt_00000.npy')
meas = np.load('SimData/level_1/measurements/u_00000.npy')
print(f'GT shape: {gt.shape}, dtype: {gt.dtype}')
print(f'Meas shape: {meas.shape}, dtype: {meas.dtype}')
"
```

## 相关参考

- KTC2023比赛官网: [链接]
- 正向求解器文档: `src/ktc_methods/Main_SimData.py`
- 原始Dataset参考: `programs/ktc2023_postprocessing/src/dataset/SimDataset.py`
