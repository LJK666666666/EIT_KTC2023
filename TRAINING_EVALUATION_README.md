# EIT重建深度学习系统

基于深度学习的电阻抗层析成像（EIT）重建系统。该系统使用仿真数据训练神经网络，并在KTC2023评估数据集上进行测试。

## 项目结构

```
EIT_KTC2023/
├── src/
│   ├── model.py                    # 神经网络模型定义
│   ├── sim_dataset.py              # 仿真数据集生成器
│   ├── eval_dataset.py             # 评估数据集加载器
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估和可视化脚本
│   ├── generate_simdata.py         # 批量生成仿真数据
│   ├── test_pipeline.py            # 完整流程测试
│   └── ktc_methods/                # KTC2023官方方法
│       ├── KTCFwd.py               # 正向求解器
│       ├── KTCMeshing.py           # 网格处理
│       ├── KTCScoring.py           # 评分函数
│       └── ...
│
├── SimData/                        # 仿真训练数据
│   └── level_1/
│       ├── gt/                     # Ground truth
│       └── measurements/           # 测量数据
│
├── EvaluationData_full/            # KTC2023评估数据
│   └── evaluation_datasets/
│       ├── level1/
│       ├── level2/
│       └── ...
│
├── EvaluationData/                 # Ground truth
│   └── GroundTruths/
│       ├── level_1/
│       ├── level_2/
│       └── ...
│
└── results/                        # 训练和评估结果
    ├── experiment_name_timestamp/
    │   ├── best_model.pth
    │   ├── training_curves.png
    │   ├── config.json
    │   ├── history.json
    │   └── evaluation/
    │       ├── level1/
    │       │   ├── data1_comparison.png
    │       │   ├── data2_comparison.png
    │       │   └── ...
    │       ├── evaluation_results.json
    │       └── summary_report.png
    └── ...
```

## 功能模块

### 1. 神经网络模型 (`model.py`)

**EITReconstructionNet**
- 简单的编码器-解码器架构
- 输入：992维测量数据
- 输出：256×256×3的分割图（3类）
- 参数量：~38.5M

**UNetEITReconstruction**
- 基于U-Net的架构，带跳跃连接
- 适合空间信息处理
- 参数量更多，但性能更好

**SegmentationLoss**
- 结合交叉熵损失和Dice损失
- 适合不平衡的分割任务

### 2. 数据集

**SimulatedEITDataset** (`sim_dataset.py`)
- 生成仿真EIT数据
- 使用KTC2023官方正向求解器
- 随机生成导电率分布（圆形、矩形、多边形）
- 添加可配置噪声

**EvaluationDataset** (`eval_dataset.py`)
- 加载KTC2023评估数据集
- 支持7个评估级别
- 自动加载对应的ground truth

### 3. 训练 (`train.py`)

功能：
- 从SimData加载仿真数据
- 训练/验证集划分
- 自动保存最佳模型
- Early stopping
- 学习率调度
- 保存训练曲线

### 4. 评估 (`evaluate.py`)

功能：
- 在所有评估级别上测试模型
- 使用KTC2023官方评分函数
- 生成对比图（GT vs 重建）
- 保存每个样本的得分
- 生成总结报告

## 使用方法

### 步骤1：生成仿真数据

```bash
# 生成10000条训练数据（需要约12小时）
python src/generate_simdata.py --num_samples 10000

# 测试模式（生成10条）
python src/generate_simdata.py --test

# 监控进度
python src/monitor_progress.py --target 10000 --interval 60
```

### 步骤2：训练模型

```bash
python src/train.py
```

或者在代码中配置：

```python
config = {
    'exp_name': 'my_experiment',
    'model_name': 'simple',  # 或 'unet'
    'input_dim': 992,
    'data_path': 'SimData/level_1',
    'num_samples': 1000,  # 使用的样本数
    'train_split': 0.8,
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'weight_ce': 1.0,
    'weight_dice': 1.0,
    'patience': 10,
    'early_stopping': 20,
}

from train import train_model
model, history, save_dir = train_model(config)
```

### 步骤3：评估模型

```bash
python src/evaluate.py --model_path results/experiment_name/best_model.pth
```

### 一键测试完整流程

```bash
python src/test_pipeline.py
```

## 评估结果

评估会生成以下内容：

### 1. 对比图

每个测试样本生成一张对比图，包含：
- Ground Truth（左）
- 重建结果（中），显示KTC评分
- 测量数据曲线（右）

### 2. 评分文件

**evaluation_results.json**
```json
{
  "level1": {
    "level": 1,
    "scores": [0.7234, 0.6891, 0.7012],
    "data_indices": [1, 2, 3],
    "average_score": 0.7046
  },
  ...
}
```

### 3. 总结报告

- **summary_report.png**: 各级别平均分柱状图
- **summary_report.txt**: 详细的文本报告

示例：
```
======================================================================
EIT RECONSTRUCTION EVALUATION REPORT
======================================================================

Level 1:
  Average Score: 0.7046
  Number of Samples: 3
  Individual Scores: ['0.7234', '0.6891', '0.7012']

Level 2:
  Average Score: 0.6523
  Number of Samples: 3
  Individual Scores: ['0.6234', '0.6712', '0.6623']

...

----------------------------------------------------------------------
Overall Average Score: 0.6785
Total Samples: 21
======================================================================
```

## KTC评分方法

使用KTC2023官方评分函数 (`KTCScoring.py`)：

- 基于SSIM（结构相似性指数）
- 分别计算高电导率区域(2)和低电导率区域(1)的得分
- 最终得分 = 0.5 × (score_conductive + score_resistive)
- 分数范围：[0, 1]，越高越好

## 训练技巧

### 1. 数据增强
- 旋转（已实现）
- 可添加平移、缩放等

### 2. 模型选择
- 简单任务：使用 `EITReconstructionNet`
- 复杂任务：使用 `UNetEITReconstruction`

### 3. 超参数调整
- 学习率：1e-3 到 1e-4
- Batch size：16-64（取决于GPU内存）
- 损失权重：调整 `weight_ce` 和 `weight_dice`

### 4. Early Stopping
- patience: 5-10 epochs
- early_stopping: 15-20 epochs

## 依赖项

```bash
torch>=1.10.0
numpy>=1.20.0
scipy>=1.7.0
matplotlib>=3.3.0
tqdm>=4.60.0
Pillow>=8.0.0
```

## 性能

### 训练性能
- CPU：~2秒/batch（batch_size=16）
- GPU (T4)：~0.5秒/batch

### 评估性能
- 单个样本：~0.1秒
- 完整评估（21个样本）：~3秒

### 模型大小
- EITReconstructionNet：~147MB
- UNetEITReconstruction：~250MB

## 已知问题

1. **测量维度不匹配**
   - 仿真数据：992维（32电极，相邻注入）
   - 评估数据：2356维（可能使用不同的测量模式）
   - 解决方案：需要调整模型输入维度或转换测量数据

2. **类别不平衡**
   - 背景类别占比最大
   - 解决方案：使用Dice loss + 类别权重

## TODO

- [ ] 适配不同测量维度的评估数据
- [ ] 实现数据增强
- [ ] 添加更多模型架构
- [ ] 支持多GPU训练
- [ ] 添加TensorBoard可视化
- [ ] 实现模型ensemble

## 参考文献

1. KTC2023 Challenge: [链接]
2. U-Net: Convolutional Networks for Biomedical Image Segmentation
3. EIT Reconstruction using Deep Learning

## 联系方式

如有问题，请提交Issue。
