# Solver图像重建方法详解

## 概述

`solver.py` 实现了基于深度图像先验（Deep Image Prior, DIP）的电阻抗层析成像（EIT）重建算法。该方法使用神经网络作为图像表示，通过优化网络参数来重建电导率分布图像。

## 核心算法：深度图像先验（DIP）

### 基本原理

深度图像先验是一种无监督的图像重建方法，其核心思想是：
- **不需要预训练**：网络随机初始化，不使用任何训练数据
- **网络结构作为先验**：卷积神经网络的架构本身就能提供良好的图像先验
- **优化网络参数**：通过优化网络参数来拟合测量数据，而非优化图像本身

### 在EIT中的应用

在电阻抗成像中，DIP方法解决逆问题的流程如下：

```
随机噪声输入 → DIP网络 → 电导率图像 → 正向模型 → 预测电压
                ↑                                      ↓
                └──────── 反向传播（最小化损失） ──────┘
```

## 主要函数详解

### 1. solve() - 主求解函数

这是整个重建过程的入口函数，实现了完整的DIP优化流程。

#### 参数设置

```python
img_resolution = 128          # 图像分辨率
linpoint = 0.7927            # 线性化点（背景电导率）
deltaU_fac = 2               # 电压差放大因子（增强信号）
dist_mode = 'lowpass'        # 距离插值模式
dist_sigma = 8e-3            # 高斯核标准差
```

#### DIP网络架构

采用U-Net风格的跳跃连接架构：
- **输入深度**：32通道随机噪声
- **尺度数**：6层多尺度处理
- **跳跃连接**：下采样和上采样路径都有跳跃连接
- **输出**：单通道电导率图像

```python
skip_n11=32                                    # 1×1卷积通道数
num_scales=6                                   # 网络尺度数
skip_n33d=[16, 16, 32, 32, 64, 64]           # 下采样跳跃连接通道数
skip_n33u=[16, 16, 32, 32, 64, 64]           # 上采样跳跃连接通道数
```

#### 关键组件

##### a) 求解矩阵获取
```python
solver_dict = get_solver_matrices(inputData, categoryNbr, linpoint)
J = torch.from_numpy(solver_dict['J']).type(dtype)        # 雅可比矩阵
deltaU = torch.from_numpy(solver_dict['deltaU']).type(dtype) * deltaU_fac
```

- **雅可比矩阵 J**：描述电导率变化与电压测量之间的线性关系
- **电压差 deltaU**：实际测量值与参考值的差异

##### b) 距离矩阵
```python
dist = calc_dist(solver_dict['coordinates'], x_d=img_resolution,
                y_d=img_resolution, mode=dist_mode, sigma=dist_sigma)
```

距离矩阵用于将有限元网格节点的电导率值映射到规则像素网格：
- 输入：不规则的FEM节点坐标
- 输出：节点到像素的权重矩阵
- 模式：高斯加权插值（lowpass）

##### c) 圆形掩膜
```python
c_mask = get_mask(img_resolution)
```

限制重建区域在圆形域内（EIT测量区域是圆形）。

#### 优化过程

```python
LR = 3.5e-3              # 学习率
reg_noise_std = 0.04     # 正则化噪声
num_iter = 3000          # 迭代次数
w_tv = 5e-8              # 总变差权重
```

**优化循环**：

```python
for i in range(num_iter):
    optimizer.zero_grad()
    
    # 1. 添加正则化噪声（防止过拟合）
    if reg_noise_std > 0:
        deblur_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    # 2. 网络前向传播
    cond_pixels = c_mask * deblur_net(deblur_input)
    
    # 3. 像素到节点映射
    cond_nodes = dist @ cond_pixels[0,0].reshape((-1,1))
    
    # 4. 计算损失
    total_loss = F.mse_loss(J @ cond_nodes, deltaU)  # 数据拟合项
    total_loss += w_tv * tv(cond_pixels)              # 总变差正则化
    
    # 5. 反向传播和优化
    total_loss.backward()
    optimizer.step()
```

**损失函数组成**：

1. **数据拟合项**：`MSE(J @ cond_nodes, deltaU)`
   - 使正向模型的预测与实际测量匹配
   
2. **总变差正则化**：`w_tv * TV(cond_pixels)`
   - 促进图像平滑，同时保留边缘
   - 权重很小（5e-8），起微调作用

#### 后处理

```python
# 1. 调整大小到256×256
cond_pixels_np = np.array(Image.fromarray(cond_pixels_np).resize((256,256)))

# 2. 图像分割
cond_pixels_np_segmented = segment(cond_pixels_np)
```

---

### 2. get_solver_matrices() - 构建求解矩阵

此函数准备EIT逆问题所需的所有矩阵和数据。

#### 电极配置

```python
Nel = 32                  # 32个电极
z = (1e-6) * np.ones((Nel, 1))  # 接触阻抗（很小，近似完美接触）
```

#### 测量模式

从参考文件加载：
- **Injref**：电流注入模式（哪些电极对注入电流）
- **Uelref**：参考电压（水槽测量，已知电导率）
- **Mpat**：电压测量模式（哪些电极对测量电压）

#### 难度级别处理

```python
rmind = np.arange(0, 2 * (categoryNbr - 1), 1)
```

根据`categoryNbr`（难度级别）移除部分电极数据：
- **Level 1**：使用所有32个电极
- **Level 2**：移除2个电极数据
- **Level 3**：移除4个电极数据
- **Level 4**：移除6个电极数据

#### 有限元网格

加载预生成的网格（Gmsh生成）：

**一阶网格**：
- `g`：节点坐标
- `H`：三角形单元的节点索引
- `elfaces`：边界电极的节点索引

**二阶网格**（用于更精确的正向求解）：
- `H2`、`g2`、`elfaces2`：二阶单元数据
- 包含单元中点，提供更高精度

#### 正向求解器设置

```python
solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)
```

使用二阶网格创建有限元求解器。

#### 噪声模型

```python
noise_std1 = 0.05  # 相对于每个测量值的5%噪声
noise_std2 = 0.01  # 相对于最大测量值的1%噪声
```

这两个噪声分量模拟实际测量中的误差。

#### 雅可比矩阵计算

```python
J = solver.Jacobian(sigma0, z)
```

在线性化点`sigma0`处计算雅可比矩阵：
- 大小：(测量数) × (节点数)
- 含义：J[i,j] = ∂U_i / ∂σ_j（第j个节点的电导率变化对第i个电压测量的影响）

---

### 3. calc_dist() - 计算距离矩阵

将不规则FEM网格节点映射到规则像素网格的关键函数。

#### 坐标转换

```python
nodes_coordinates = nodes_coordinates.copy().T
nodes_coordinates = nodes_coordinates[[1,0],:]  # 交换x和y
```

#### 像素网格生成

```python
for i in range(x_d):
    for j in range(y_d):
        xp[i,j] = a[0]*i + b[0]  # 计算每个像素中心的坐标
        yp[i,j] = a[1]*j + b[1]
```

#### 两种插值模式

##### a) 最近邻模式 (mode='nearest')
```python
dist[i,:] = dist[i,:] == dist[i,:].min()  # 只有最近的像素为1，其余为0
```

每个节点只映射到最近的一个像素。

##### b) 高斯加权模式 (mode='lowpass')
```python
dist = np.exp(-dist**2/(2*sigma**2))  # 高斯权重
dist /= (dist.sum(axis=1)+eps).reshape(-1,1)  # 归一化
```

每个节点按距离加权映射到多个像素，权重服从高斯分布：
- **优点**：平滑，减少锯齿效应
- **sigma**：控制平滑程度，越大越平滑

**距离矩阵的使用**：
```python
cond_pixels = dist.T @ cond_nodes  # 节点到像素
cond_nodes = dist @ cond_pixels    # 像素到节点
```

---

### 4. get_mask() - 生成圆形掩膜

EIT测量区域是圆形，需要掩膜限制重建区域。

```python
c_mask = (X-xc)**2 + (Y-yc)**2 < r**2
```

- 圆心：`(xc, yc) = (n_img/2-1, n_img/2-1)`
- 半径：`r = n_img/2 + 1`
- 结果：圆内为True，圆外为False

**应用**：
```python
cond_pixels = c_mask * deblur_net(deblur_input)  # 圆外强制为0
```

---

### 5. segment() - 图像分割

将连续的电导率图像分割为离散的区域标签。

#### Otsu多阈值分割

```python
level, x = KTCScoring.Otsu2(cond_pixels_np.flatten(), 256, 7)
```

- 使用Otsu方法找到两个最优阈值
- 将图像分为3类：低、中、高电导率

#### 三分类

```python
ind0 = cond_pixels_np < x[level[0]]          # 低值区域
ind1 = (x[level[0]] <= cond_pixels_np <= x[level[1]])  # 中值区域
ind2 = cond_pixels_np > x[level[1]]          # 高值区域
```

#### 背景识别

```python
bgclass = inds.index(max(inds))  # 像素最多的类作为背景
```

根据背景类，将前景分为两个标签（1和2）：

- **背景 → 0**
- **前景类型1 → 1**
- **前景类型2 → 2**

#### 形态学处理

```python
opening_mask = sp.ndimage.binary_opening(cond_pixels_np_segmented, iterations=10)
cond_pixels_np_segmented = opening_mask * cond_pixels_np_segmented
```

**开运算**（腐蚀+膨胀）的作用：
- 去除小的噪声点
- 平滑边界
- iterations=10：较强的滤波效果

---

## 算法流程图

```
输入：测量电压数据 (Uel)
     难度级别 (categoryNbr)
     
步骤1：准备求解矩阵
  ├─ 加载参考数据和网格
  ├─ 根据难度级别选择电极
  ├─ 计算雅可比矩阵 J
  └─ 计算电压差 deltaU

步骤2：初始化DIP网络
  ├─ 生成随机噪声输入
  ├─ 构建U-Net架构
  ├─ 计算距离矩阵（FEM节点→像素）
  └─ 生成圆形掩膜

步骤3：优化循环 (3000次迭代)
  对于每次迭代：
    ├─ 添加正则化噪声
    ├─ 网络前向：噪声 → 电导率图像
    ├─ 应用圆形掩膜
    ├─ 像素 → FEM节点（通过距离矩阵）
    ├─ 正向模型：σ → 预测电压 (J @ σ)
    ├─ 计算损失：MSE + TV正则化
    └─ 反向传播优化网络参数

步骤4：后处理
  ├─ 提取最终图像
  ├─ 调整大小到256×256
  ├─ Otsu分割（3类）
  ├─ 识别背景
  ├─ 标记前景（1和2）
  └─ 形态学开运算去噪

输出：分割后的电导率图像
```

## 关键技术点

### 1. 为什么使用DIP？

**传统方法的问题**：
- 正则化参数难以选择
- 需要手工设计先验
- 对噪声敏感

**DIP的优势**：
- 无需训练数据
- 网络架构自动提供图像先验
- 鲁棒性好

### 2. 线性化近似

```python
J @ cond_nodes ≈ deltaU
```

这是EIT逆问题的线性化形式：
- 在背景电导率`linpoint`处泰勒展开
- 只保留一阶项
- 适用于小扰动情况

### 3. 正则化策略

**多层正则化**：

a) **网络架构**：U-Net结构本身就是先验
b) **噪声注入**：`reg_noise_std = 0.04`，防止过拟合
c) **总变差**：`w_tv = 5e-8`，平滑图像
d) **圆形掩膜**：几何约束

### 4. 多尺度处理

```python
num_scales=6
```

6层多尺度结构能捕捉不同尺度的特征：
- 粗尺度：整体结构
- 细尺度：边缘细节

### 5. 电压差放大

```python
deltaU_fac = 2
```

将电压差放大2倍，原因：
- 增强微弱信号
- 改善优化景观
- 加快收敛

## 参数调优建议

### 网络参数

| 参数 | 当前值 | 影响 | 调整建议 |
|------|--------|------|----------|
| `input_depth` | 32 | 网络表达能力 | 增大→更复杂图像，但可能过拟合 |
| `num_scales` | 6 | 多尺度能力 | 增大→更细致，但计算慢 |
| `skip_n33d/u` | [16,16,32,32,64,64] | 特征通道数 | 增大→更强表达，但计算慢 |

### 优化参数

| 参数 | 当前值 | 影响 | 调整建议 |
|------|--------|------|----------|
| `LR` | 3.5e-3 | 收敛速度 | 太大→不稳定；太小→收敛慢 |
| `num_iter` | 3000 | 优化充分性 | 观察损失曲线决定 |
| `reg_noise_std` | 0.04 | 正则化强度 | 增大→更平滑，但可能欠拟合 |
| `w_tv` | 5e-8 | 平滑程度 | 增大→更平滑，但可能丢失细节 |

### 问题相关参数

| 参数 | 当前值 | 影响 | 调整建议 |
|------|--------|------|----------|
| `linpoint` | 0.7927 | 线性化精度 | 应接近真实背景电导率 |
| `deltaU_fac` | 2 | 信号强度 | 根据信噪比调整 |
| `dist_sigma` | 8e-3 | 插值平滑度 | 越大→越平滑 |

## 常见问题

### Q1: 为什么需要距离矩阵？

**A**: EIT正向求解使用有限元方法，节点分布不规则；但DIP生成的是规则像素网格。距离矩阵实现两者之间的转换。

### Q2: 总变差权重为什么这么小？

**A**: `w_tv = 5e-8` 看起来很小，但：
- TV值通常很大（所有梯度的和）
- 主要靠网络架构提供先验
- TV只是微调作用

### Q3: 为什么要添加噪声？

**A**: 正则化噪声（`reg_noise_std`）是DIP的关键技巧：
- 防止网络记忆测量噪声
- 提供隐式正则化
- 在合适的迭代次数停止可以得到最佳结果

### Q4: 如何判断收敛？

**A**: 可以监控：
- 损失函数值
- 重建图像的变化
- 通常3000次迭代足够，但可根据具体情况调整

### Q5: 分割为什么用3类？

**A**: EIT问题通常有：
- 背景（均匀介质）
- 一个或两个异常（不同电导率的目标）

3类分割能区分背景和不同类型的异常。

## 性能优化

### GPU加速

代码已支持GPU：
```python
if torch.cuda.is_available():
    dtype = torch.cuda.DoubleTensor
```

**加速比**：GPU通常比CPU快5-10倍。

### 批处理

当前代码处理单个案例。对于多个案例，可以：
```python
# 批处理伪代码
for case in cases:
    result = solve(case.data, case.category)
    save(result)
```

### 内存优化

- 使用`detach()`避免不必要的梯度计算
- 及时释放中间结果

## 扩展方向

### 1. 非线性求解

当前使用线性化近似。可以改进为：
```python
# 在优化循环中重新计算雅可比矩阵
J = solver.Jacobian(cond_nodes, z)
```

### 2. 自适应参数

```python
# 动态调整TV权重
w_tv = w_tv_init * decay_factor ** (i / num_iter)
```

### 3. 更复杂的网络

- 使用注意力机制
- 残差连接
- 批归一化

### 4. 多模态融合

结合其他成像模态（如超声、光学）的先验信息。

## 参考文献

1. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2018). Deep image prior. CVPR.
2. 深度图像先验在逆问题中的应用
3. EIT正向和逆问题理论
4. U-Net架构及其变体

## 总结

`solver.py` 实现了一个优雅的无监督EIT重建方法：

**核心创新**：
- 使用DIP避免训练数据需求
- 结合物理模型（雅可比矩阵）
- 多层正则化策略

**优势**：
- 无需大量训练数据
- 重建质量高
- 适应不同难度级别

**适用场景**：
- 医学成像
- 工业无损检测
- 地球物理勘探

这个实现为EIT图像重建提供了一个强大而灵活的框架。
