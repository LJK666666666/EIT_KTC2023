# EIT Reconstruction Framework

统一的电阻抗层析成像(EIT)重建框架，支持多种重建方法。

## 快速开始

### 安装依赖

### 训练模型

#### CNN 方法
```bash
python main.py train --method cnn --num_epochs 200 --result_dir cnn --batch_size 128
```

#### Diffusion 方法
```bash
python main.py train --method diffusion --num_epochs 200 --result_dir diffusion --batch_size 128
```

### 推理

#### CNN 方法
```bash
python main.py inference --method cnn --num_epochs 200 --result_dir cnn --batch_size 128
```

#### Diffusion 方法
```bash
python main.py inference --method diffusion --num_epochs 200 --result_dir diffusion --batch_size 128
```

## 项目结构

```
.
├── src/                       # 源代码目录
│   ├── core/                  # 核心模块
│   │   ├── base.py            # 基础重建方法类
│   │   ├── data_loader.py     # 数据加载
│   │   ├── evaluator.py       # 评估指标
│   │   ├── trainer.py         # 训练器
│   │   └── config.py          # 配置管理
│   ├── methods/               # 重建方法实现
│   │   ├── cnn/               # CNN 方法（UNet）
│   │   ├── diffusion/         # Diffusion 方法（DiT）
│   │   └── traditional/       # 传统方法（Tikhonov）
│   ├── utils/                 # 工具函数
│   ├── datasets/              # 数据集
│   ├── configs/               # 配置文件
│   ├── train.py               # 训练脚本
│   ├── inference.py           # 推理脚本
│   └── ktc_methods/           # KTC 官方代码
├── data/                      # 数据目录
│   ├── train/                 # 训练数据
│   ├── valid/                 # 验证数据
│   ├── test/                  # 测试数据
│   ├── test2017/              # 2017 真实数据
│   └── test2023/              # 2023 真实数据
├── results/                   # 结果保存目录
├── main.py                    # 入口脚本（可选）
├── test_framework.py          # 测试脚本
├── README.md                  # 本文件
├── EXAMPLES.md                # 使用示例
├── ARCHITECTURE.md            # 架构设计
└── REFACTORING_PLAN.md        # 重构计划
```

## 支持的方法

### 1. CNN (UNet)
基于 UNet 架构的卷积神经网络方法
- 配置文件: `src/configs/cnn_config.yaml`
- 特点: 快速训练，实时推理

### 2. Diffusion (DiT)
基于扩散模型的 DiT (Diffusion Transformer) 方法
- 配置文件: `src/configs/diffusion_config.yaml`
- 特点: 高质量重建，来自 CDEIT 论文
- 参考: [CDEIT GitHub](https://github.com/...)

### 3. Traditional (Tikhonov)
传统 Tikhonov 正则化方法
- 配置文件: `src/configs/traditional_config.yaml`
- 特点: 无需训练，直接推理

## 配置文件说明

所有配置文件位于 `src/configs/` 目录下：

### 基础配置 (`base_config.yaml`)
- 数据配置: `data_dir`, `batch_size`, `num_workers`
- 训练配置: `num_epochs`, `device`
- 优化器配置: `type` (adam/sgd), `lr`, `weight_decay`
- 调度器配置: `type` (reduce_on_plateau/step)

### CNN 配置 (`cnn_config.yaml`)
- 模型配置:
  - `input_channels`: 输入通道数
  - `output_channels`: 输出通道数
  - `base_channels`: 基础通道数
  - `num_layers`: U-Net 层数
  - `dropout`: Dropout 率

### Diffusion 配置 (`diffusion_config.yaml`)
- 模型配置:
  - `input_size`: 输入图像大小
  - `patch_size`: Patch 大小
  - `hidden_size`: 隐藏层维度
  - `num_heads`: 注意力头数
- Diffusion 参数:
  - `num_timesteps`: 扩散步数
  - `beta_schedule`: 噪声调度
- 推理配置:
  - `num_sampling_steps`: DDIM 采样步数

## 命令行参数

### 训练参数
- `--method`: 方法类型 (cnn/diffusion/traditional)
- `--config`: 配置文件路径
- `--data_dir`: 数据目录
- `--num_epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--lr`: 学习率
- `--device`: 设备 (cuda/cpu)
- `--result_dir`: 结果保存目录
- `--resume`: 恢复训练的检查点路径

### 推理参数
- `--method`: 方法类型
- `--checkpoint`: 模型检查点路径
- `--dataset`: 数据集 (test/test2017/test2023)
- `--output_dir`: 输出目录
- `--save_mat`: 保存为 .mat 格式
- `--save_images`: 保存为 .png 格式

## 数据格式

### 训练/验证/测试数据
- 格式: MATLAB `.mat` 文件
- 包含字段:
  - `measurements`: 测量电压数据
  - `conductivity`: 真实电导率图像

### 真实数据 (2017/2023)
- 格式: MATLAB `.mat` 文件
- 包含字段:
  - `measurements`: 测量电压数据

## 评估指标

框架自动计算以下指标：
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Relative Error

## 输出结果

训练结果保存在 `results/{method_name}_{timestamp}/`:
- `config.json`: 训练配置
- `best_model.pth`: 最佳模型
- `checkpoint_epoch_*.pth`: 定期检查点
- `history.json`: 训练历史
- `training_curves.png`: 训练曲线图

推理结果保存在指定的 `output_dir`:
- `reconstruction_*.mat`: 重建结果 (.mat 格式)
- `reconstruction_*.png`: 重建结果可视化
- `metrics.json`: 评估指标

## 扩展新方法

1. 在 `src/methods/` 下创建新方法目录
2. 实现继承自 `BaseReconstructionMethod` 的方法类
3. 实现必需的抽象方法:
   - `_build_model()`
   - `train_step()`
   - `val_step()`
   - `inference()`
4. 在 `src/methods/__init__.py` 中注册新方法
5. 创建对应的配置文件

## 引用

如果使用了 Diffusion 方法，请引用 CDEIT 论文：
```
[待补充]
```

## License

[待补充]
