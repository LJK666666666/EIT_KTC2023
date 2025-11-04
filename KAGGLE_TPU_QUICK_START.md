# Kaggle TPU v5e-8 快速开始指南

## 🚀 最简单的运行方式（推荐）

### 在 Kaggle Notebook 中，直接运行这个命令：

```bash
!cd /kaggle/working && python CDEIT_TPU/main.py --mode train \
  --data-path /kaggle/input/simdata-cdeit \
  --global-batch-size 64 \
  --epochs 100
```

**就这样！不需要 `accelerate launch`，不需要 `--use_tpu`。**

---

## 为什么这样运行？

| 方法 | 是否有效 | 说明 |
|------|--------|------|
| `accelerate launch --use_tpu main.py` | ❌ 错误 | `--use_tpu` 参数已被移除 |
| `accelerate config && accelerate launch main.py` | ⚠️ 复杂 | 需要额外配置步骤 |
| `python main.py` | ✅ **推荐** | 直接运行，自动检测设备 |

---

## 在 Kaggle Notebook 中的完整步骤

### 单元格 1：安装依赖
```bash
!pip install --upgrade torch-xla
!pip install -q accelerate ema-pytorch scipy timm pillow
```

### 单元格 2：设置目录
```python
import os
os.chdir('/kaggle/working')

# 检查是否有 TPU
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"✅ TPU 已就绪: {device}")
except:
    print("⚠️ TPU 不可用，使用 GPU")
```

### 单元格 3：运行训练
```bash
!python CDEIT_TPU/main.py --mode train \
  --data-path /kaggle/input/simdata-cdeit \
  --global-batch-size 64 \
  --global-seed 0 \
  --epochs 100 \
  --log-every 10 \
  --ckpt-every 100
```

### 单元格 4：运行测试
```bash
!python CDEIT_TPU/main.py --mode test \
  --data-path /kaggle/input/simdata-cdeit \
  --global-batch-size 64 \
  --data simulated
```

---

## 命令行参数说明

```bash
python main.py [OPTIONS]

必需参数:
  --mode {train,test}           运行模式（默认: test）
  --data-path PATH              数据文件夹路径

可选参数:
  --results-dir PATH            结果保存目录（默认: results）
  --global-batch-size INT       批大小（默认: 16）
  --epochs INT                  训练轮数（默认: 1400）
  --global-seed INT             随机种子（默认: 0）
  --num-workers INT             数据加载进程数（默认: 4）
  --log-every INT               多少步输出一次日志（默认: 500）
  --ckpt-every INT              多少步保存一次检查点（默认: 1000）
  --data {simulated,uef2017,ktc2023}  测试数据集（默认: simulated）
  --samplingsteps INT           采样步数（默认: 5）
```

---

## 常见问题

### Q1: 遇到 `ValueError: fp16 mixed precision requires a GPU (not 'xla')`

**原因**: 代码尝试在 TPU 上使用 FP16，但 TPU 只支持 BF16

**✅ 已修复**: 新代码会自动检测设备并选择正确的精度

### Q2: 需要指定 TPU 吗？

**不需要！** 代码会自动检测：
- 如果有 TPU，使用 TPU + BF16
- 如果只有 GPU，使用 GPU + FP16

### Q3: 显存/内存不够？

尝试减小 `--global-batch-size`：

```bash
!python main.py --mode train \
  --global-batch-size 32  # 从 64 改为 32
```

### Q4: 为什么不使用 `accelerate launch`?

因为：
1. Kaggle TPU 有特殊的初始化方式
2. `accelerate` 的 `--use_tpu` 参数已被移除
3. 直接 `python main.py` 更简单，代码自动处理

### Q5: 可以用 `accelerate` 吗？

可以，但需要先配置：

```bash
# 第一次运行时配置一次
!accelerate config  # 选择 TPU
!accelerate launch main.py --mode train --data-path /kaggle/input/simdata-cdeit
```

但推荐直接用 `python`，更简单。

---

## 性能提示

### TPU v5e-8 优化
- **批大小**: 64-128（TPU 内存大）
- **精度**: 自动使用 BF16（快 2-3 倍）
- **数据加载**: `--num-workers 4` 足够

### 预期训练速度
```
批大小 64, TPU v5e-8:
- 每个 epoch: ~2-3 分钟（假设 1000 张图片）
- 100 个 epoch: ~200-300 分钟 (~4 小时)
```

---

## 完整示例命令

### 最小训练（快速测试）
```bash
!python CDEIT_TPU/main.py --mode train \
  --data-path /kaggle/input/simdata-cdeit \
  --global-batch-size 64 \
  --epochs 5 \
  --log-every 10
```

### 完整训练（推荐）
```bash
!python CDEIT_TPU/main.py --mode train \
  --data-path /kaggle/input/simdata-cdeit \
  --global-batch-size 64 \
  --epochs 100 \
  --global-seed 0 \
  --log-every 50 \
  --ckpt-every 500 \
  --results-dir /kaggle/working/results
```

### 测试各个数据集
```bash
# 测试模拟数据
!python CDEIT_TPU/main.py --mode test --data simulated --global-batch-size 64

# 测试 UEF2017
!python CDEIT_TPU/main.py --mode test --data uef2017 --global-batch-size 64

# 测试 KTC2023
!python CDEIT_TPU/main.py --mode test --data ktc2023 --global-batch-size 64
```

---

## 数据路径配置

假设你的 Kaggle 数据结构：

```
/kaggle/input/
  simdata-cdeit/
    train/         ← 训练数据
    valid/         ← 验证数据
    test/          ← 测试数据
    mean.pth       ← 标准化参数
    std.pth
```

运行命令：
```bash
python main.py --mode train --data-path /kaggle/input/simdata-cdeit
```

代码会自动寻找：
- `/kaggle/input/simdata-cdeit/train/`
- `/kaggle/input/simdata-cdeit/valid/`
- 标准化参数如果找不到，从 `./data/` 加载

---

## 遇到错误怎么办？

### 错误 1: `No such file or directory: ...`
检查 `--data-path` 是否正确，用实际的 Kaggle 数据集名称替换

### 错误 2: 内存不足
减小 `--global-batch-size` 或 `--num-workers`

### 错误 3: 缓慢的数据加载
增加 `--num-workers`（但不要超过 CPU 核数）

---

## 下一步

1. ✅ 运行 `python main.py --mode train` 开始训练
2. 📊 在 `/kaggle/working/results/deit/checkpoints/` 查看结果
3. 📈 查看 `loss1.mat` 文件中的损失曲线

---

## 参考资源

- [PyTorch XLA 文档](https://pytorch.org/xla/)
- [Kaggle TPU 指南](https://www.kaggle.com/docs/TPU)
- [原始 CDEIT 论文](https://arxiv.org/abs/2412.16979)
