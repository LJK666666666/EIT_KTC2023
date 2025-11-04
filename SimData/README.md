# SimData 仿真数据生成

本文件夹包含使用KTC2023正向求解器生成的EIT仿真数据。

## 文件结构

```
SimData/
└── level_1/              # Level 1 数据（不移除电极）
    ├── gt/               # Ground truth（导电率分布）
    │   ├── gt_00000.npy
    │   ├── gt_00001.npy
    │   └── ...
    └── measurements/     # EIT测量数据
        ├── u_00000.npy
        ├── u_00001.npy
        └── ...
```

## 数据格式

### Ground Truth (gt_*.npy)
- **形状**: (256, 256)
- **类型**: float32
- **值**:
  - 0: 背景（均匀导电率）
  - 1: 低电导率区域
  - 2: 高电导率区域
- **单位**: 像素网格（256×256）

### Measurements (u_*.npy)
- **形状**: (992,)
- **类型**: float32
- **含义**: EIT电压差分测量值
- **维度**: 32电极 × 31测量模式 = 992个测量值
- **噪声水平**: 1% (noise_std1=0.1)

## 生成数据

### 完整生成（10000条数据）

```bash
python src/generate_simdata.py --num_samples 10000
```

预计时间：约12-13小时（每个样本约4.5秒）

### 测试生成（10条数据）

```bash
python src/generate_simdata.py --test
```

### 监控进度

在另一个终端中运行：

```bash
python src/monitor_progress.py --target 10000 --interval 60
```

## 使用生成的数据

### 加载数据

```python
import numpy as np

# 加载单个样本
gt = np.load('SimData/level_1/gt/gt_00000.npy')
measurements = np.load('SimData/level_1/measurements/u_00000.npy')

print(f'Ground truth shape: {gt.shape}')
print(f'Measurements shape: {measurements.shape}')
```

### 使用Dataset类

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class SimDataLoader(Dataset):
    def __init__(self, data_path='SimData/level_1', num_samples=10000):
        self.data_path = data_path
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        gt = np.load(f'{self.data_path}/gt/gt_{idx:05d}.npy')
        measurements = np.load(f'{self.data_path}/measurements/u_{idx:05d}.npy')

        return torch.from_numpy(gt).float(), torch.from_numpy(measurements).float()

# 使用
dataset = SimDataLoader()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for gt_batch, meas_batch in dataloader:
    # gt_batch: [batch_size, 256, 256]
    # meas_batch: [batch_size, 992]
    pass
```

## 数据生成参数

生成时使用的参数：

- **网格**: Mesh_dense.mat (3766个节点)
- **电极数量**: 32
- **噪声标准差1**: 0.1 (1%的测量值)
- **噪声标准差2**: 0 (不使用固定噪声)
- **分割类别**: 3 (背景 + 2种异常)
- **包含物形状**: 圆形、矩形、多边形（随机生成）
- **包含物数量**: 1-3个（随机）

## 数据统计

运行以下命令查看数据统计：

```bash
python -c "
import numpy as np
import os

gt_path = 'SimData/level_1/gt'
meas_path = 'SimData/level_1/measurements'

num_files = len([f for f in os.listdir(gt_path) if f.endswith('.npy')])
print(f'Total samples: {num_files}')

# 加载前100个样本进行统计
measurements = []
for i in range(min(100, num_files)):
    meas = np.load(f'{meas_path}/u_{i:05d}.npy')
    measurements.append(meas)

measurements = np.array(measurements)
print(f'Measurements mean: {measurements.mean():.6f}')
print(f'Measurements std: {measurements.std():.6f}')
print(f'Measurements range: [{measurements.min():.6f}, {measurements.max():.6f}]')
"
```

## 相关文件

- `src/sim_dataset.py`: 数据集生成器类
- `src/generate_simdata.py`: 批量生成脚本
- `src/monitor_progress.py`: 进度监控脚本
- `src/example_use_simdataset.py`: 使用示例

## 注意事项

1. 生成的数据使用KTC2023官方的正向求解器和网格
2. Level 1表示不移除电极（完整的32个电极）
3. 导电率分布在圆形区域内（半径0.098m）
4. 测量噪声已添加（1%水平）
5. 数据保存为NumPy的.npy格式，便于快速加载

## 磁盘空间

每个样本约占用：
- Ground truth: 257 KB
- Measurements: 4 KB
- 总计: ~261 KB/样本

10000个样本总计约: **2.6 GB**
