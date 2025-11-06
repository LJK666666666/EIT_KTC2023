# EIT 重建项目 - 推荐架构设计

## 项目目标
整合多种 EIT 图像重建方法（CNN、扩散模型、传统算法等），通过命令行参数灵活切换，进行统一的训练、评估和推理。

---

## 推荐的目录结构

```
项目根目录/
├── src/                            # 源代码目录
│   ├── core/                       # 核心模块
│   │   ├── __init__.py
│   │   ├── base.py                 # 基础类（BaseMethod）
│   │   ├── data_loader.py          # 统一的数据加载模块
│   │   ├── evaluator.py            # 统一的评估模块
│   │   └── config.py               # 配置管理
│   │
│   ├── methods/                    # 各种重建方法
│   │   ├── __init__.py
│   │   ├── cnn/                    # CNN 方法
│   │   │   ├── __init__.py
│   │   │   ├── models.py           # UNet、SimpleNet 等
│   │   │   ├── trainer.py          # CNN 训练器
│   │   │   └── config.yaml         # CNN 默认配置
│   │   │
│   │   ├── diffusion/              # 扩散模型方法（从 CDEIT 迁移）
│   │   │   ├── __init__.py
│   │   │   ├── models.py           # DiT 模型
│   │   │   ├── diffusion_utils.py  # 扩散过程
│   │   │   ├── trainer.py          # 扩散模型训练器
│   │   │   └── config.yaml         # 扩散模型默认配置
│   │   │
│   │   └── traditional/            # 传统方法（如 Tikhonov 正则化）
│   │       ├── __init__.py
│   │       ├── solvers.py          # 传统算法实现
│   │       └── config.yaml
│   │
│   ├── utils/                      # 工具函数
│   │   ├── __init__.py
│   │   ├── visualization.py        # 可视化函数
│   │   ├── metrics.py              # 评估指标（集成 KTC scoring）
│   │   ├── io.py                   # 文件 I/O 操作
│   │   └── logging.py              # 日志配置
│   │
│   ├── ktc_methods/                # KTC 官方方法（保持不变）
│   │   ├── __init__.py
│   │   ├── KTCFwd.py
│   │   ├── KTCMeshing.py
│   │   └── ...
│   │
│   ├── datasets/                   # 数据集管理
│   │   ├── __init__.py
│   │   ├── sim_dataset.py          # 仿真数据集
│   │   ├── ktc_dataset.py          # KTC 真实数据集
│   │   └── data_utils.py           # 数据处理工具
│   │
│   ├── configs/                    # 配置文件（YAML）
│   │   ├── defaults.yaml           # 全局默认配置
│   │   ├── cnn.yaml                # CNN 配置
│   │   ├── diffusion.yaml          # 扩散模型配置
│   │   └── traditional.yaml        # 传统方法配置
│   │
│   ├── train.py                    # 统一的训练入口
│   ├── evaluate.py                 # 统一的评估入口
│   ├── inference.py                # 推理脚本
│   └── test/                       # 测试脚本
│
├── results/                        # ⚠️ 所有结果保存在根目录（非 src/ 下）
│   └── {exp_name}_{timestamp}/
│       ├── config.yaml             # 实验配置
│       ├── checkpoints/            # 模型检查点
│       │   ├── best_model.pth
│       │   ├── checkpoint_epoch_10.pth
│       │   └── final_model.pth
│       ├── logs/                   # 训练日志
│       │   ├── train.log
│       │   └── tensorboard/
│       ├── predictions/            # 预测结果
│       │   ├── train_samples/
│       │   ├── val_samples/
│       │   └── test_samples/
│       └── figures/                # 可视化图表
│           ├── training_curves.png
│           ├── sample_predictions.png
│           └── comparison.png
│
├── data/                           # 数据目录（延用 CDEIT 结构）
│   ├── train/                      # 仿真训练数据
│   ├── valid/                      # 仿真验证数据
│   ├── test/                       # 仿真测试数据
│   ├── test2017/                   # UEF2017 真实数据
│   │   ├── 1_1.npz
│   │   ├── 1_2.npz
│   │   └── ...
│   ├── test2023/                   # KTC2023 真实数据
│   │   ├── 0_1.npz
│   │   ├── 0_2.npz
│   │   └── ...
│   ├── mean.pth                    # 数据归一化参数
│   └── std.pth
│
├── EvaluationData/                 # KTC2023 官方评估数据集（级别 1-7）（根据难度级别移除相应数量的电极）
│   ├── evaluation_datasets/
│   │   ├── level1/
│   │   │   ├── data1.mat
│   │   │   ├── data2.mat
│   │   │   ├── data3.mat
│   │   │   └── ref.mat
│   │   ├── level2/
│   │   └── ...
│   └── GroundTruths/
│       ├── level_1/
│       │   ├── 1_true.mat
│       │   └── 1_true.png
│       └── ...
│
├── EvaluationData_full/            # KTC2023 完整电极测量的数据集
│   └── ... (与 EvaluationData 相同结构)
│
├── SimData/                        # ⚠️ 自己生成的仿真数据（暂不使用）
│   └── ... (保留但不使用)
│
└── programs/                       # ⚠️ 其他 EIT 研究相关代码库（参考用）
    ├── CDEIT/                      # CNN + Diffusion 方法（参考）
    └── ... (其他研究代码)
```

---

## 核心设计模式

### 1. 基础类 (`core/base.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn

class BaseReconstructionMethod(ABC):
    """所有重建方法的基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cuda')
        self.model = self._build_model()

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """构建模型"""
        pass

    @abstractmethod
    def forward(self, measurements):
        """前向传播"""
        pass

    @abstractmethod
    def train_step(self, batch):
        """单个训练步骤，返回 loss"""
        pass

    @abstractmethod
    def val_step(self, batch):
        """验证步骤，返回 metrics"""
        pass

    @abstractmethod
    def inference(self, measurements):
        """推理，返回重建结果"""
        pass

    def save_checkpoint(self, path: str):
        """保存检查点"""
        pass

    def load_checkpoint(self, path: str):
        """加载检查点"""
        pass
```

### 2. 数据加载模块 (`core/data_loader.py`)

```python
class EITDataModule:
    """
    统一的数据加载接口

    支持的数据集：
    - 仿真数据：data/train, data/valid, data/test (延用 CDEIT 结构)
    - 真实数据：data/test2017 (UEF2017), data/test2023 (KTC2023)
    - 官方数据：EvaluationData/evaluation_datasets/level{1-7}
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_path = config.get('data_path', './data')
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # 数据归一化参数
        self.mean = None
        self.std = None

    def setup(self, mode='train'):
        """
        设置数据集

        Args:
            mode: 'train' (使用 data/train, data/valid)
                  'test' (使用 data/test)
                  'real' (使用 data/test2017 或 data/test2023)
                  'eval' (使用 EvaluationData)
        """
        if mode == 'train':
            self._setup_training_data()
        elif mode == 'test':
            self._setup_test_data()
        elif mode == 'real':
            self._setup_real_data()
        elif mode == 'eval':
            self._setup_evaluation_data()

    def _load_normalization_params(self):
        """从 data/mean.pth 和 data/std.pth 加载归一化参数"""
        import torch
        self.mean = torch.load(f'{self.data_path}/mean.pth')
        self.std = torch.load(f'{self.data_path}/std.pth')

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader
```

### 3. 评估模块 (`core/evaluator.py`)

```python
class EITEvaluator:
    """统一的评估接口"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # 使用 KTC 官方的 scoring function
        from ktc_methods import KTCScoring
        self.scorer = KTCScoring.scoringFunction

    def evaluate(self, predictions, ground_truth):
        """评估模型性能"""
        pass

    def visualize_results(self, predictions, ground_truth, save_dir):
        """可视化结果"""
        pass
```

### 4. 统一训练器 (`core/trainer.py`)

```python
class UnifiedTrainer:
    """统一的训练器，支持任意重建方法"""

    def __init__(self, method: BaseReconstructionMethod, data_module: EITDataModule, config: Dict):
        self.method = method
        self.data_module = data_module
        self.config = config
        self.evaluator = EITEvaluator(config)

    def train(self):
        """统一的训练循环"""
        for epoch in range(self.config['epochs']):
            train_loss = self._train_epoch()
            val_metrics = self._val_epoch()
            # 统一的检查点保存、提前停止等逻辑

    def _train_epoch(self):
        """单个 epoch 的训练"""
        pass

    def _val_epoch(self):
        """单个 epoch 的验证"""
        pass
```

---

## 方法实现示例

### CNN 方法实现 (`methods/cnn/trainer.py`)

```python
from core.base import BaseReconstructionMethod

class CNNReconstructor(BaseReconstructionMethod):
    def __init__(self, config):
        super().__init__(config)
        self.optimizer = None
        self.criterion = None

    def _build_model(self):
        """构建 CNN 模型（UNet、SimpleNet 等）"""
        if self.config['model_name'] == 'unet':
            from .models import UNet
            return UNet(...)
        elif self.config['model_name'] == 'simple':
            from .models import SimpleNet
            return SimpleNet(...)

    def train_step(self, batch):
        measurements, ground_truth = batch
        predictions = self.model(measurements)
        loss = self.criterion(predictions, ground_truth)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def val_step(self, batch):
        measurements, ground_truth = batch
        with torch.no_grad():
            predictions = self.model(measurements)
        # 计算指标
        metrics = self.evaluator.evaluate(predictions, ground_truth)
        return metrics
```

### 扩散模型实现 (`methods/diffusion/trainer.py`)

```python
from core.base import BaseReconstructionMethod

class DiffusionReconstructor(BaseReconstructionMethod):
    def __init__(self, config):
        super().__init__(config)
        from diffusion import create_diffusion
        self.diffusion = create_diffusion()

    def _build_model(self):
        """构建 DiT 模型（从 CDEIT 迁移）"""
        from .models import DiT
        return DiT(**self.config['model_params'])

    def train_step(self, batch):
        measurements, ground_truth = batch
        # 扩散训练逻辑
        t = torch.randint(0, self.diffusion.num_timesteps, (measurements.shape[0],), device=self.device)
        loss_dict = self.diffusion.training_losses(self.model, ground_truth, t, {'y': measurements})
        loss = loss_dict['loss']
        loss.backward()
        return loss.item()

    def val_step(self, batch):
        measurements, ground_truth = batch
        # 扩散模型验证（使用 DDIM 采样）
        samples = self.diffusion.ddim_sample(self.model, ...)
        metrics = self.evaluator.evaluate(samples, ground_truth)
        return metrics
```

### 传统方法实现 (`methods/traditional/solvers.py`)

```python
from core.base import BaseReconstructionMethod

class TraditionalReconstructor(BaseReconstructionMethod):
    def __init__(self, config):
        super().__init__(config)
        # 传统方法不需要神经网络

    def _build_model(self):
        """传统方法不需要模型"""
        return None

    def inference(self, measurements):
        """直接求解（无需训练）"""
        if self.config['method'] == 'tikhonov':
            return self._tikhonov(measurements)
        elif self.config['method'] == 'lsqr':
            return self._lsqr(measurements)
```

---

## 命令行使用方式

### 训练示例

```bash
# 使用 CNN 方法训练（使用 data/ 仿真数据）
python src/train.py \
    --method cnn \
    --model unet \
    --config src/configs/cnn.yaml \
    --data-path ./data \
    --exp-name cnn_unet_exp1

# 使用扩散模型训练（使用 data/ 仿真数据）
python src/train.py \
    --method diffusion \
    --config src/configs/diffusion.yaml \
    --data-path ./data \
    --exp-name diffusion_dit_exp1

# 使用传统方法（无需训练）
python src/train.py \
    --method traditional \
    --solver tikhonov \
    --data-path ./data
```

### 评估示例

```bash
# 评估 CNN 模型（仿真测试集）
python src/evaluate.py \
    --method cnn \
    --checkpoint results/cnn_unet_exp1/checkpoints/best_model.pth \
    --data-path ./data \
    --test-set test

# 评估 CNN 模型（KTC2023 真实数据）
python src/evaluate.py \
    --method cnn \
    --checkpoint results/cnn_unet_exp1/checkpoints/best_model.pth \
    --data-path ./data \
    --test-set test2023

# 评估扩散模型（官方评估数据集 level 1-7）
python src/evaluate.py \
    --method diffusion \
    --checkpoint results/diffusion_dit_exp1/checkpoints/best_model.pth \
    --data-path ./EvaluationData \
    --levels 1 2 3 4 5 6 7
```

### 推理示例

```bash
# 使用训练好的模型进行推理
python src/inference.py \
    --method cnn \
    --checkpoint results/cnn_unet_exp1/best_model.pth \
    --input measurements.npy \
    --output predictions.npy
```

---

## 迁移步骤

### 第 1 步：创建基础框架
1. 创建 `core/` 目录和基础类
2. 创建 `methods/` 目录结构
3. 创建配置文件系统

### 第 2 步：迁移现有代码
1. 将 `programs/CDEIT/` 迁移到 `methods/diffusion/`
2. 将 `src/train.py` 中的 CNN 逻辑迁移到 `methods/cnn/`
3. 将 KTC 官方方法保留在 `ktc_methods/`

### 第 3 步：实现统一接口
1. 为 CNN 方法实现 `BaseReconstructionMethod` 接口
2. 为扩散模型实现 `BaseReconstructionMethod` 接口
3. 为传统方法实现 `BaseReconstructionMethod` 接口

### 第 4 步：创建统一的训练/评估脚本
1. 重写 `src/train.py` 为统一入口
2. 重写 `src/evaluate.py` 为统一入口
3. 创建 `src/inference.py` 推理脚本

### 第 5 步：测试和文档
1. 编写单元测试验证各方法
2. 编写使用文档和 API 文档
3. 提供示例配置和使用脚本

---

## 配置文件示例

### `src/configs/cnn.yaml`

```yaml
# 全局设置
method: cnn
device: cuda
seed: 42

# 模型设置
model_name: unet
model_params:
  input_dim: 2356
  output_size: 256
  channels: [64, 128, 256, 512]

# 数据设置
data:
  # 数据根目录（延用 CDEIT 结构）
  data_path: ./data
  train_path: ./data/train          # 仿真训练数据
  val_path: ./data/valid            # 仿真验证数据
  test_path: ./data/test            # 仿真测试数据
  real_2017_path: ./data/test2017   # UEF2017 真实数据
  real_2023_path: ./data/test2023   # KTC2023 真实数据

  # 归一化参数
  mean_path: ./data/mean.pth
  std_path: ./data/std.pth

  # DataLoader 设置
  batch_size: 32
  num_workers: 4
  train_split: 0.9                  # 如果没有单独的验证集

# 训练设置
training:
  epochs: 200
  learning_rate: 1e-3
  weight_decay: 1e-5
  optimizer: adam
  scheduler:
    type: ReduceLROnPlateau
    patience: 10

# 损失函数
loss:
  type: combined  # ce + dice
  weight_ce: 1.0
  weight_dice: 1.0

# 检查点和保存
checkpoint:
  save_interval: 10
  best_metric: val_loss
```

### `src/configs/diffusion.yaml`

```yaml
# 全局设置
method: diffusion
device: cuda
seed: 42

# 模型设置
model_name: dit
model_params:
  hidden_size: 384
  depth: 12
  num_heads: 6
  num_classes: 2

# 扩散过程
diffusion:
  num_timesteps: 1000
  schedule: linear
  sampling_method: ddim
  num_sample_steps: 50

# 其他设置...
```

---

## 数据说明

### 数据组织结构

本项目使用三类数据集：

1. **仿真数据** (`data/` 目录，延用 CDEIT 结构)
   - `train/`: 训练集（npz 格式）
   - `valid/`: 验证集（npz 格式）
   - `test/`: 测试集（npz 格式）
   - `mean.pth`, `std.pth`: 归一化参数

2. **真实数据** (`data/` 目录)
   - `test2017/`: UEF2017 真实测量数据
   - `test2023/`: KTC2023 真实测量数据

3. **官方评估数据** (`EvaluationData/` 和 `EvaluationData_full/`)
   - `evaluation_datasets/level{1-7}/`: KTC2023 官方评估数据（7 个难度级别）
   - `GroundTruths/level_{1-7}/`: 对应的真值标签
   - 用于最终提交和排名

### 数据格式

**仿真数据 (.npz)**：
```python
data = np.load('data/train/00001.npz')
measurements = data['measurements']  # 形状: [2356] 或 [992]
conductivity = data['conductivity']  # 形状: [256, 256]
```

**真实数据 (.npz)**：
```python
data = np.load('data/test2023/0_1.npz')
measurements = data['U']             # 电压测量
# 无真值标签（需要提交到官方评估）
```

**官方数据 (.mat)**：
```python
import scipy.io as sio
data = sio.loadmat('EvaluationData/evaluation_datasets/level1/data1.mat')
ref = sio.loadmat('EvaluationData/evaluation_datasets/level1/ref.mat')
```

### 数据使用建议

- **训练**: 使用 `data/train` 和 `data/valid` (仿真数据)
- **验证**: 使用 `data/test` (仿真测试集)
- **实际测试**: 使用 `data/test2017` 或 `data/test2023` (真实数据)
- **官方评估**: 使用 `EvaluationData/` (提交排名用)

---

## 主要优势

1. **模块化**：各种方法相互独立，易于添加新方法
2. **可维护**：统一的接口和代码结构
3. **可复现**：配置文件管理所有超参数
4. **可扩展**：轻松集成新的重建方法
5. **易用**：单一的命令行接口
6. **代码复用**：共享的数据加载、评估、可视化等模块

---

## 注意事项

- 保持 `ktc_methods/` 不变（官方代码）
- 使用 YAML 配置文件管理超参数（避免修改代码）
- 为每个方法编写独立的测试脚本
- 使用统一的日志系统追踪所有实验
- 文档和注释必须清晰完整
