# TPU v5e-8 内存优化指南

## 问题

```
ValueError: Allocation would exceed memory (size=17179869184)
```

**含义**: 尝试分配超过 TPU 可用内存（16GB）的内存空间。

---

## ✅ 解决方案

### 1. 立即修复 - 使用更小的批大小

```bash
# 方案 A：批大小 16（推荐开始）
python main.py --mode train \
  --data-path /kaggle/input/simdata-cdeit \
  --global-batch-size 16

# 方案 B：批大小 8（更保险）
python main.py --mode train \
  --data-path /kaggle/input/simdata-cdeit \
  --global-batch-size 8

# 方案 C：批大小 4（如果仍然内存不足）
python main.py --mode train \
  --data-path /kaggle/input/simdata-cdeit \
  --global-batch-size 4
```

### 2. 代码修改（已做）

验证集批大小已修改：
```python
# 原来（内存多）
batch_size=batch_size * 4

# 现在（为 TPU 优化）
batch_size=batch_size
```

---

## 📊 TPU v5e-8 内存规格

| 项目 | 规格 |
|------|------|
| 总内存 | 16 GB |
| 可用内存 | ~14-15 GB（系统占用） |
| 单精度（FP32） | ~4 亿参数 |
| 混合精度（BF16） | ~8 亿参数 |

---

## 🎯 推荐配置

### 保守配置（最安全）
```bash
python main.py --mode train \
  --global-batch-size 8 \
  --epochs 100 \
  --num-workers 2 \
  --log-every 50
```

### 平衡配置（推荐）
```bash
python main.py --mode train \
  --global-batch-size 16 \
  --epochs 100 \
  --num-workers 4 \
  --log-every 50
```

### 激进配置（需要足够数据）
```bash
python main.py --mode train \
  --global-batch-size 32 \
  --epochs 50 \
  --num-workers 4
```

---

## 内存优化技巧

### 1. 减小批大小（最有效）

| 批大小 | 内存占用 | 速度 | 收敛 |
|--------|---------|------|------|
| 4 | 很低 | 慢 | 稳定 |
| 8 | 低 | 中 | 良好 |
| 16 | 中 | 快 | 很好 |
| 32 | 高 | 很快 | 需要调 LR |
| 64 | 很高 ❌ | 超快 | 容易 OOM |

### 2. 减小验证集批大小

已修改为 `batch_size`（原为 `batch_size * 4`）

### 3. 减少数据加载进程

```bash
python main.py --num-workers 2  # 改从 4 为 2
```

### 4. 启用梯度累积（高级）

如果需要大有效批大小但内存有限，可以使用梯度累积：

```python
# 在训练循环中
accumulated_steps = 4
if train_steps % accumulated_steps == 0:
    opt.step()
    opt.zero_grad()
else:
    # 梯度缩放
    (loss / accumulated_steps).backward()
```

---

## 诊断步骤

### 1. 检查 TPU 内存使用

```python
import torch_xla.core.xla_model as xm

# 在训练循环中打印内存
print(f"TPU 内存: {xm.get_metrics()}")
```

### 2. 从小到大尝试批大小

```bash
# 先试 4
python main.py --global-batch-size 4 --epochs 1

# 再试 8
python main.py --global-batch-size 8 --epochs 1

# 再试 16
python main.py --global-batch-size 16 --epochs 1

# 找到最大可用批大小
```

### 3. 监控错误信息

如果出现类似错误，记下 allocation size：
- **< 4GB**: 批大小可以更大
- **4-8GB**: 批大小合适
- **8-16GB**: 接近极限
- **> 16GB**: 内存溢出

---

## 快速修复检查清单

- [ ] 修改 `--global-batch-size` 为 16 或更小
- [ ] 修改 `loaderVal batch_size` 从 `batch_size * 4` 为 `batch_size` ✅ 已做
- [ ] 修改 `--num-workers` 为 2-4
- [ ] 确保 `--epochs` 合理（不要太大）

---

## 如果还是内存不足

### 检查模型大小

```python
import torch
model = DiT()
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数: {total_params / 1e6:.1f}M")  # 单位百万
```

### 可能的原因

1. **数据变量没有清理** - 检查训练循环是否有内存泄漏
2. **输入尺寸太大** - 检查输入 shape 是否正确
3. **累积梯度** - 确保梯度在反向传播后清除

### 极端解决方案

```bash
# 如果模型真的太大，只能用 1 的批大小
python main.py --global-batch-size 1 --epochs 1
```

---

## 性能预期

使用不同批大小的训练速度：

```
批大小 8：  ~10-15 秒/step
批大小 16： ~12-18 秒/step （推荐）
批大小 32： 内存不足或很慢
批大小 64： ❌ 内存溢出
```

---

## 验证修复成功

```bash
# 1. 先跑 1 个 epoch 测试
python main.py --mode train \
  --global-batch-size 16 \
  --epochs 1 \
  --log-every 5

# 2. 如果成功，跑完整训练
python main.py --mode train \
  --global-batch-size 16 \
  --epochs 100
```

---

## 总结

| 问题 | 解决方案 |
|------|---------|
| 内存溢出 | 减小 `--global-batch-size` |
| 验证太慢 | ✅ 已减小验证集批大小 |
| 仍然内存不足 | 进一步减小批大小或 `--num-workers` |
| 训练太慢 | 增大批大小（如果内存允许） |

**立即尝试**：
```bash
python main.py --mode train --data-path /kaggle/input/simdata-cdeit --global-batch-size 16
```

这应该能解决问题！
