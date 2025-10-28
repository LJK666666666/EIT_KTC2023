# KTC2023 后处理重建方法详解

## 概述

本项目采用**两阶段重建策略**，结合线性化重建和深度学习后处理，实现高质量的EIT图像分割：

1. **第一阶段**：线性化Gauss-Newton重建（Linearised Reconstruction）
2. **第二阶段**：基于U-Net的深度学习后处理与分割

这种混合方法充分利用了传统物理模型的可解释性和深度学习的强大表示能力。

## 整体流程图

```
输入数据 (Uel)
    ↓
┌─────────────────────────────────────────────┐
│  第一阶段: 线性化重建                          │
│  - 使用5组不同的正则化参数                      │
│  - 生成5个不同的重建结果                        │
│  - 每个结果对应不同的正则化策略                  │
└─────────────────────────────────────────────┘
    ↓
    5个初始重建图像 (256×256)
    ↓
┌─────────────────────────────────────────────┐
│  第二阶段: U-Net深度学习后处理                  │
│  - 输入: 5通道图像堆叠                         │
│  - 条件输入: 难度级别                          │
│  - 输出: 3类分割结果 (0, 1, 2)                 │
└─────────────────────────────────────────────┘
    ↓
最终分割结果 (256×256)
```

---

## 第一阶段: 线性化Gauss-Newton重建

### 理论基础

#### EIT前向问题

EIT前向问题是求解椭圆型偏微分方程：

$$
\nabla \cdot (\sigma \nabla u) = 0 \quad \text{in } \Omega
$$

其中：
- $\Omega$：成像区域（圆形，半径0.115m）
- $\sigma$：电导率分布（待重建）
- $u$：电位分布

边界条件由电极的电流注入和电压测量决定。

#### 线性化策略

对于小扰动 $\delta\sigma$，电压变化可以近似为：

$$
\delta U \approx \mathbf{J} \delta\sigma
$$

其中 $\mathbf{J}$ 是Jacobian矩阵（灵敏度矩阵），表示电导率微小变化对测量电压的影响。

#### 逆问题求解

重建问题转化为求解正则化线性系统：

$$
\min_{\delta\sigma} \| \mathbf{B}\mathbf{J}\delta\sigma - \delta U \|_{\Gamma_n^{-1}}^2 + \alpha_{\text{TV}} \|\mathbf{R}_{\text{TV}} \delta\sigma\|^2 + \alpha_{\text{SM}} \|\mathbf{R}_{\text{SM}} \delta\sigma\|^2 + \alpha_{\text{LM}} \|\text{diag}(\mathbf{J}^T\Gamma_n^{-1}\mathbf{J})\delta\sigma\|^2
$$

其中：
- $\mathbf{B}$：测量模式矩阵（31×32，`Mpat.T`）
- $\delta U = U_{\text{meas}} - U_{\text{ref}}$：差分电压
- $\Gamma_n^{-1}$：噪声协方差矩阵的逆（加权矩阵）
- $\mathbf{R}_{\text{TV}}$：全变分正则化矩阵
- $\mathbf{R}_{\text{SM}}$：平滑正则化矩阵
- $\alpha_{\text{TV}}, \alpha_{\text{SM}}, \alpha_{\text{LM}}$：正则化参数

### 核心组件

#### 1. Jacobian矩阵（灵敏度矩阵）

**定义**：
$$
J_{ij} = \frac{\partial U_i}{\partial \sigma_j}
$$

**物理意义**：
- 表示第 $j$ 个网格单元的电导率变化对第 $i$ 个电压测量的影响
- 预先使用FEniCS有限元软件计算并保存
- 在背景电导率 $\sigma_0 = 0.745$ S/m 下计算

**代码实现**：
```python
# 加载预计算的Jacobian矩阵
J = np.load(os.path.join(base_path, f"jac_{mesh_name}.npy"))
J = J.reshape(76*32, J.shape[-1])  # 重塑为 (2432, n_elements)

# 应用测量模式矩阵B
self.B = block_diag(*[np.array(B) for i in range(76)])
BJ = self.B[self.vincl_flatten,:] @ J
self.BJ = BJ  # 最终的组合矩阵
```

**维度说明**：
- 76个注入模式
- 每个模式32个电极测量
- 经过测量矩阵B后变为31个独立测量
- 总共 76×31 = 2356 个测量值（完整数据）

#### 2. 噪声建模

采用**两分量噪声模型**：

$$
\text{Var}(U_i) = \left(\frac{\sigma_1}{100} |\delta U_i|\right)^2 + \left(\frac{\sigma_2}{100} \max(|\delta U|)\right)^2
$$

其中：
- $\sigma_1 = 0.05$：相对噪声水平（5%）
- $\sigma_2 = 0.01$：绝对噪声水平（1%）

**代码实现**：
```python
# 构建噪声协方差矩阵的逆
var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))), 2)
var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)), 2)
GammaInv = 1. / var_meas  # 对角矩阵的对角元素
GammaInv = np.diag(GammaInv[:, 0])
```

**物理意义**：
- 信号越强的测量，权重越高
- 自动适应不同幻影的信号强度
- 避免噪声放大

#### 3. 正则化项

本方法使用**三种正则化策略**的组合：

##### 3.1 全变分(TV)正则化

**矩阵构建**：
```python
# 加载网格邻接矩阵
Rtv = np.load(os.path.join(base_path, f"mesh_neighbour_matrix_{mesh_name}.npy"))
# 构建拉普拉斯矩阵
np.fill_diagonal(Rtv, -Rtv.sum(axis=0))
self.Rtv = -Rtv
```

**作用**：
- 惩罚相邻网格单元之间的电导率差异
- 促进分片常数解
- 保持边缘锐度

**数学表达**：
$$
\|\mathbf{R}_{\text{TV}} \delta\sigma\|^2 \approx \sum_{(i,j) \in \text{edges}} (\delta\sigma_i - \delta\sigma_j)^2
$$

##### 3.2 平滑先验(SM)正则化

**加载预计算矩阵**：
```python
self.Rsm = np.load(os.path.join(base_path, f"smoothnessR_{mesh_name}.npy"))
```

**作用**：
- 基于Sobolev范数的平滑先验
- 惩罚电导率的剧烈变化
- 促进整体平滑性

**物理意义**：
- 假设真实电导率分布相对平滑
- 避免高频噪声

##### 3.3 Levenberg-Marquardt (LM)正则化

**定义**：
$$
\alpha_{\text{LM}} \cdot \text{diag}(\mathbf{J}^T\Gamma_n^{-1}\mathbf{J})
$$

**代码实现**：
```python
A = JGJ + alphas[0]*self.Rtv + alphas[1]*self.Rsm + alphas[2] * np.diag(np.diag(JGJ))
```

**作用**：
- 添加对角加载，改善条件数
- 稳定求解过程
- 类似于信赖域方法

#### 4. 难度级别处理

根据难度级别移除部分电极的数据：

```python
Nel = 32
vincl_level = np.ones(((Nel - 1), 76), dtype=bool)
rmind = np.arange(0, 2 * (level - 1), 1)  # 要移除的电极索引

# 移除测量数据
for ii in range(0, 75):
    for jj in rmind:
        if Injref[jj, ii]:
            vincl_level[:, ii] = 0  # 移除整个注入模式
        vincl_level[jj, :] = 0      # 移除电极的所有测量
```

**难度级别与可用电极**：

| Level | 移除电极数 | 可用电极数 | 数据完整度 |
|-------|-----------|-----------|-----------|
| 1     | 0         | 32        | 100%      |
| 2     | 2         | 30        | ~94%      |
| 3     | 4         | 28        | ~88%      |
| 4     | 6         | 26        | ~81%      |
| 5     | 8         | 24        | ~75%      |
| 6     | 10        | 22        | ~69%      |
| 7     | 12        | 20        | ~63%      |

### 重建求解

#### 法方程

最优解满足：

$$
\left[\mathbf{J}^T\Gamma_n^{-1}\mathbf{J} + \alpha_{\text{TV}}\mathbf{R}_{\text{TV}} + \alpha_{\text{SM}}\mathbf{R}_{\text{SM}} + \alpha_{\text{LM}}\text{diag}(\mathbf{J}^T\Gamma_n^{-1}\mathbf{J})\right] \delta\sigma = \mathbf{J}^T\Gamma_n^{-1}\delta U
$$

**代码实现**：
```python
def reconstruct(self, Uel, alpha_tv, alpha_sm, alpha_lm):
    # 计算差分电压
    deltaU = Uel - np.array(self.Uref)
    deltaU = deltaU[self.vincl_flatten]

    # 构建噪声逆协方差矩阵
    var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))), 2)
    var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)), 2)
    GammaInv = 1. / var_meas
    GammaInv = np.diag(GammaInv[:, 0])

    # 构建法方程左侧
    JGJ = self.BJ.T @ GammaInv @ self.BJ
    # 构建法方程右侧
    b = self.BJ.T @ GammaInv @ deltaU

    # 添加正则化项
    A = JGJ + alpha_tv*self.Rtv + alpha_sm*self.Rsm + alpha_lm * np.diag(np.diag(JGJ))

    # 求解线性系统
    delta_sigma = np.linalg.solve(A, b)

    return delta_sigma
```

### 多参数重建策略

为了提供丰富的先验信息给深度学习模型，对每个测量数据使用**5组不同的正则化参数**进行重建。

#### 参数配置

以Level 1为例：

```python
level_to_alphas = {
    1: [
        [1956315.789, 0., 0.],              # 纯TV正则化
        [0., 656.842, 0.],                  # 纯平滑正则化
        [0., 0.1, 6.105],                   # 平滑+LM正则化
        [1956315.789/3., 656.842/3., 6.105/3.],  # 混合正则化（弱）
        [1e4, 0.1, 5.]                      # 平衡正则化
    ],
    # ... 其他级别
}
```

#### 不同参数组的作用

1. **纯TV** `[1956315.789, 0., 0.]`
   - 强调边缘保持
   - 生成分片常数结果
   - 适合检测物体边界

2. **纯平滑** `[0., 656.842, 0.]`
   - 强调整体平滑
   - 抑制高频噪声
   - 可能模糊边缘

3. **平滑+LM** `[0., 0.1, 6.105]`
   - 轻度平滑
   - 稳定求解
   - 折中方案

4. **混合正则化（弱）** `[α_TV/3, α_SM/3, α_LM/3]`
   - 所有正则化项的弱版本
   - 更接近数据
   - 可能包含更多细节

5. **平衡正则化** `[1e4, 0.1, 5.]`
   - 经验调优的参数组合
   - 平衡各种特性

#### 批量重建实现

```python
def reconstruct_list(self, Uel, alpha_list):
    """
    对同一测量数据使用多组参数进行重建

    参数:
        Uel: 测量电压
        alpha_list: 正则化参数列表 [[α_TV, α_SM, α_LM], ...]

    返回:
        delta_sigma_list: 多个重建结果的列表
    """
    deltaU = Uel - np.array(self.Uref)
    deltaU = deltaU[self.vincl_flatten]

    # 只计算一次GammaInv和JGJ（优化性能）
    var_meas = np.power(((self.noise_std1 / 100) * (np.abs(deltaU))), 2)
    var_meas = var_meas + np.power((self.noise_std2 / 100) * np.max(np.abs(deltaU)), 2)
    GammaInv = 1. / var_meas
    GammaInv = np.diag(GammaInv[:, 0])

    JGJ = self.BJ.T @ GammaInv @ self.BJ
    b = self.BJ.T @ GammaInv @ deltaU

    # 对每组参数求解
    delta_sigma_list = []
    for alphas in alpha_list:
        A = JGJ + alphas[0]*self.Rtv + alphas[1]*self.Rsm + alphas[2] * np.diag(np.diag(JGJ))
        delta_sigma = np.linalg.solve(A, b)
        delta_sigma_list.append(delta_sigma)

    return delta_sigma_list
```

### 网格到像素的插值

重建结果定义在有限元网格上，需要插值到256×256像素网格。

#### 网格类型

- **稀疏网格** (`mesh_name="sparse"`)：较少节点，计算快
- **密集网格** (`mesh_name="dense"`)：更多节点，精度高

#### 插值方法

使用**线性插值**（Linear Interpolation）：

```python
def interpolate_to_image(self, sigma):
    """
    将有限元网格上的解插值到256×256像素网格

    参数:
        sigma: 有限元网格上的电导率分布

    返回:
        256×256的像素网格图像
    """
    pixwidth = 0.23 / 256  # 像素宽度

    # 创建像素中心坐标
    pixcenter_x = np.linspace(-0.115 + pixwidth/2,
                              0.115 - pixwidth/2 + pixwidth, 256)
    pixcenter_y = pixcenter_x
    X, Y = np.meshgrid(pixcenter_x, pixcenter_y)
    pixcenters = np.column_stack((X.ravel(), Y.ravel()))

    # 使用线性插值
    interp = LinearNDInterpolator(self.pos, sigma, fill_value=0)
    sigma_grid = interp(pixcenters)

    # 翻转以匹配图像坐标系
    return np.flipud(sigma_grid.reshape(256, 256))
```

**关键点**：
- 网格外的点（圆形区域外）填充为0
- 使用三角剖分的线性插值
- 翻转y轴以匹配图像坐标

---

## 第二阶段: U-Net深度学习后处理

### 模型架构

#### OpenAI U-Net

基于OpenAI的扩散模型U-Net架构，但用于分割任务。

**网络配置**：
```python
config.model.in_channels = 5          # 输入：5个初始重建结果
config.model.out_channels = 3         # 输出：3类分割（0, 1, 2）
config.model.model_channels = 64      # 基础通道数
config.model.num_res_blocks = 2       # 每层残差块数量
config.model.attention_resolutions = [16, 32]  # 注意力层分辨率
config.model.channel_mult = (1., 1., 2., 2., 4., 4.)  # 通道倍增因子
```

#### 网络结构

```
输入: (Batch, 5, 256, 256) + 难度级别

编码器路径:
256×256 (64 channels)  ──┐
    ↓ ResBlock + Downsample   │
128×128 (64 channels)  ──┤
    ↓ ResBlock + Downsample   │
64×64 (128 channels)    ──┤  Skip Connections
    ↓ ResBlock + Attention    │
32×32 (128 channels)    ──┤
    ↓ ResBlock + Downsample   │
16×16 (256 channels)    ──┤
    ↓ ResBlock + Attention    │
8×8 (256 channels)      ──┘
    ↓
  Bottleneck
    ↓
解码器路径:
8×8 (256 channels)      ←─┐
    ↑ Upsample + ResBlock    │
16×16 (256 channels)    ←─┤
    ↑ Upsample + Attention   │  Skip Connections
32×32 (128 channels)    ←─┤
    ↑ Upsample + ResBlock    │
64×64 (128 channels)    ←─┤
    ↑ Upsample + ResBlock    │
128×128 (64 channels)   ←─┤
    ↑ Upsample + ResBlock    │
256×256 (64 channels)   ←─┘
    ↓
输出: (Batch, 3, 256, 256)
```

#### 关键组件

1. **残差块（ResBlock）**
   - 包含两个卷积层
   - GroupNorm归一化
   - SiLU激活函数
   - 跳跃连接

2. **注意力机制（Attention）**
   - 仅在分辨率16和32上使用
   - 多头自注意力
   - 捕获长程依赖关系

3. **条件输入**
   - 难度级别通过正弦位置编码嵌入
   - 嵌入向量注入到每个残差块
   - 使得模型适应不同难度级别

### 推理流程

```python
# 1. 堆叠5个初始重建结果
sigma_reco = np.stack([delta_sigma_0, delta_sigma_1,
                       delta_sigma_2, delta_sigma_3, delta_sigma_4])
# sigma_reco.shape = (5, 256, 256)

# 2. 转换为PyTorch张量并添加batch维度
reco = torch.from_numpy(sigma_reco).float().to(device).unsqueeze(0)
# reco.shape = (1, 5, 256, 256)

# 3. 准备条件输入（难度级别）
level = torch.tensor([level]).to("cuda")

# 4. 前向传播
with torch.no_grad():
    pred = model(reco, level)
    # pred.shape = (1, 3, 256, 256)

# 5. Softmax + Argmax得到分割结果
pred_softmax = F.softmax(pred, dim=1)
pred_argmax = torch.argmax(pred_softmax, dim=1).cpu().numpy()[0, :, :]
# pred_argmax.shape = (256, 256), 值为 0, 1, 或 2
```

### 输出类别

分割结果包含3个类别：

- **类别0**：背景（水/盐水）
- **类别1**：低电导率物体（如塑料）
- **类别2**：高电导率物体（如金属）

---

## 完整流程代码解析

### main.py 主流程

```python
def coordinator(args):
    level = int(args.level)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 创建输出目录
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========== 加载U-Net模型 ==========
    config = get_configs()
    model = get_model(config)
    model.load_state_dict(torch.load(level_to_model_path[level]))
    model.eval()
    model.to(device)

    # ========== 读取参考数据 ==========
    y_ref = loadmat(os.path.join(args.input_folder, "ref.mat"))
    Injref = y_ref["Injref"]  # 注入模式
    Mpat = y_ref["Mpat"]      # 测量模式
    Uelref = y_ref["Uelref"]  # 参考电压

    # ========== 设置线性重建器 ==========
    mesh_name = "sparse"
    B = Mpat.T

    # 根据难度级别构建测量包含矩阵
    Nel = 32
    vincl_level = np.ones(((Nel - 1), 76), dtype=bool)
    rmind = np.arange(0, 2 * (level - 1), 1)

    for ii in range(0, 75):
        for jj in rmind:
            if Injref[jj, ii]:
                vincl_level[:, ii] = 0
            vincl_level[jj, :] = 0

    # 创建重建器实例
    reconstructor = LinearisedRecoFenics(Uelref, B, vincl_level,
                                         mesh_name=mesh_name)

    # ========== 处理每个数据文件 ==========
    alphas = level_to_alphas[level]
    f_list = [f for f in os.listdir(args.input_folder) if f != "ref.mat"]

    for f in f_list:
        print("Start processing ", f)

        # 1. 加载测量数据
        y = np.array(loadmat(os.path.join(args.input_folder, f))["Uel"])

        # 2. 第一阶段：多参数线性重建
        delta_sigma_list = reconstructor.reconstruct_list(y, alphas)

        # 3. 插值到像素网格
        delta_sigma_0 = reconstructor.interpolate_to_image(delta_sigma_list[0])
        delta_sigma_1 = reconstructor.interpolate_to_image(delta_sigma_list[1])
        delta_sigma_2 = reconstructor.interpolate_to_image(delta_sigma_list[2])
        delta_sigma_3 = reconstructor.interpolate_to_image(delta_sigma_list[3])
        delta_sigma_4 = reconstructor.interpolate_to_image(delta_sigma_list[4])

        # 4. 堆叠为5通道输入
        sigma_reco = np.stack([delta_sigma_0, delta_sigma_1,
                               delta_sigma_2, delta_sigma_3, delta_sigma_4])

        # 5. 第二阶段：U-Net后处理
        reco = torch.from_numpy(sigma_reco).float().to(device).unsqueeze(0)
        level_tensor = torch.tensor([level]).to("cuda")

        with torch.no_grad():
            pred = model(reco, level_tensor)
            pred_softmax = F.softmax(pred, dim=1)
            pred_argmax = torch.argmax(pred_softmax, dim=1).cpu().numpy()[0, :, :]

        # 6. 保存结果
        mdic = {"reconstruction": pred_argmax.astype(int)}
        objectno = f.split(".")[-2][-1]  # 提取文件编号
        savemat(os.path.join(output_path, str(objectno) + ".mat"), mdic)
```

---

## 方法优势

### 1. 多参数策略的优势

**问题**：单一正则化参数难以适应所有情况
- 强正则化：平滑但可能丢失细节
- 弱正则化：保留细节但可能噪声大

**解决方案**：提供多种重建结果
- 深度学习模型自动选择和融合信息
- 不同参数捕获不同特征
- 提高鲁棒性

### 2. 两阶段架构的优势

**第一阶段**（物理模型）：
- 可解释性强
- 利用物理先验
- 无需训练数据

**第二阶段**（深度学习）：
- 强大的模式识别能力
- 学习从重建到分割的映射
- 克服线性化误差

**协同效应**：
- 物理模型提供初始估计
- 深度学习模型修正和分割
- 结合两者优势

### 3. 条件U-Net的优势

**问题**：不同难度级别数据质量差异大

**解决方案**：条件输入
- 模型学习难度级别的影响
- 自适应调整分割策略
- 单一模型处理所有级别

---

## 计算复杂度分析

### 第一阶段：线性重建

**单次重建**：
- Jacobian矩阵乘法: $O(m \times n)$，其中 $m$ 是测量数，$n$ 是网格元素数
- 求解线性系统: $O(n^3)$（直接法）或 $O(kn^2)$（迭代法，$k$ 是迭代次数）
- 插值到像素网格: $O(256^2 \times n)$

**5次重建总时间**：约1-5秒（CPU）

### 第二阶段：U-Net推理

**前向传播**：
- 参数量: 约50M（取决于具体配置）
- FLOPs: 约100G（256×256输入）
- 推理时间: 约0.1秒（GPU）

**总体时间**：约2-10秒/幻影（取决于硬件）

---

## 参数调优指南

### 线性重建参数

#### TV正则化参数 $\alpha_{\text{TV}}$

**推荐范围**：$10^4$ - $10^7$

**调优策略**：
- 过小：噪声放大，边缘不清晰
- 过大：过度分片，丢失细节
- 根据信噪比调整：信噪比高→小；信噪比低→大

#### 平滑正则化参数 $\alpha_{\text{SM}}$

**推荐范围**：$0.1$ - $1000$

**调优策略**：
- 与TV正则化配合使用
- 可以弥补TV的过度锐化
- 一般小于TV参数

#### LM正则化参数 $\alpha_{\text{LM}}$

**推荐范围**：$1$ - $100$

**调优策略**：
- 主要用于稳定求解
- 条件数不好时增大
- 一般较小

### 难度级别特定调优

**原则**：难度越高，正则化越强

| Level | 策略 |
|-------|------|
| 1-2   | 弱正则化，保留细节 |
| 3-4   | 中等正则化 |
| 5-7   | 强正则化，抑制噪声 |

**经验公式**（粗略）：
$$
\alpha_{\text{TV}}(\text{level}) \approx \alpha_{\text{TV}}(\text{level}=1) \times \left(1 + 0.2 \times (\text{level}-1)\right)
$$

---

## 训练深度学习模型

### 数据准备

**训练数据**：
- 仿真数据（FEM正向求解）
- 实验数据（如果有）
- 数据增强：噪声注入、旋转等

**标签**：
- 真实分割（ground truth）
- 3类：0（背景），1（低电导率），2（高电导率）

### 训练配置

```python
config.training.batch_size = 6
config.training.epochs = 1000
config.training.lr = 1e-4
config.training.save_model_every_n_epoch = 10
```

### 损失函数

**常用选择**：
1. **交叉熵损失**（Cross-Entropy Loss）
   $$
   \mathcal{L}_{\text{CE}} = -\sum_{i=1}^{N} \sum_{c=0}^{2} y_{i,c} \log(\hat{y}_{i,c})
   $$

2. **Dice损失**（Dice Loss）
   $$
   \mathcal{L}_{\text{Dice}} = 1 - \frac{2\sum_{i} y_i \hat{y}_i}{\sum_{i} y_i + \sum_{i} \hat{y}_i}
   $$

3. **组合损失**
   $$
   \mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \mathcal{L}_{\text{Dice}}
   $$

### 评估指标

- **像素准确率**（Pixel Accuracy）
- **IoU**（Intersection over Union）
- **Dice系数**
- **KTC评分**（竞赛特定）

---

## 实际应用示例

### 基本用法

```bash
python main.py input_data/ output_results/ 1
```

参数说明：
- `input_data/`：包含`ref.mat`和`data*.mat`的输入文件夹
- `output_results/`：输出文件夹（自动创建）
- `1`：难度级别（1-7）

### 批处理脚本

```bash
#!/bin/bash

for level in {1..7}; do
    python main.py \
        data/level${level}/ \
        results/level${level}/ \
        ${level}
done
```

### Python API使用

```python
from src import LinearisedRecoFenics, get_model
from configs.postprocessing_config import get_configs
import torch

# 加载数据
Uelref = ...  # 参考电压
B = ...       # 测量矩阵
vincl = ...   # 测量包含矩阵
Uel = ...     # 目标电压

# 第一阶段：线性重建
reconstructor = LinearisedRecoFenics(Uelref, B, vincl, mesh_name="sparse")
alphas = [[1e6, 0, 0], [0, 500, 0], [0, 0.1, 5], [1e6/3, 500/3, 5/3], [1e4, 0.1, 5]]
delta_sigma_list = reconstructor.reconstruct_list(Uel, alphas)

# 插值到像素网格
images = [reconstructor.interpolate_to_image(ds) for ds in delta_sigma_list]
sigma_reco = np.stack(images)

# 第二阶段：U-Net后处理
config = get_configs()
model = get_model(config)
model.load_state_dict(torch.load("model.pt"))
model.eval()

reco = torch.from_numpy(sigma_reco).float().unsqueeze(0)
level = torch.tensor([1])

with torch.no_grad():
    pred = model(reco, level)
    pred_argmax = torch.argmax(F.softmax(pred, dim=1), dim=1).cpu().numpy()[0]

# pred_argmax 是最终的分割结果
```

---

## 局限性与改进方向

### 当前局限性

1. **线性化误差**
   - Jacobian在背景电导率处计算
   - 大扰动时误差增大
   - 解决方案：迭代更新Jacobian

2. **网格依赖**
   - 重建质量受网格质量影响
   - 粗网格丢失细节
   - 解决方案：自适应网格细化

3. **数据需求**
   - 深度学习模型需要大量训练数据
   - 实验数据有限
   - 解决方案：数据增强、迁移学习

4. **计算成本**
   - 多次重建和深度学习推理
   - 实时应用有挑战
   - 解决方案：模型压缩、硬件加速

### 可能的改进

1. **非线性重建**
   - 使用Gauss-Newton迭代
   - 更新Jacobian
   - 提高大扰动情况下的精度

2. **端到端学习**
   - 直接从测量数据到分割
   - 跳过线性重建
   - 需要大量训练数据

3. **物理信息神经网络**
   - 将PDE约束嵌入网络
   - 减少数据需求
   - 提高可解释性

4. **不确定性量化**
   - 提供置信区间
   - 贝叶斯方法
   - 集成学习

---

## 理论深入

### Jacobian矩阵的计算

#### 伴随法（Adjoint Method）

Jacobian矩阵的第 $j$ 列可以通过伴随法高效计算：

$$
\frac{\partial U}{\partial \sigma_j} = -\int_{\Omega_j} \nabla u \cdot \nabla w \, d\Omega
$$

其中 $w$ 是伴随场，满足：
$$
\nabla \cdot (\sigma \nabla w) = \text{测量函数}
$$

**优势**：
- 一次伴随求解获得一个测量的所有导数
- 计算复杂度 $O(m \times \text{FEM求解})$
- 比有限差分快得多

### 正则化理论

#### Tikhonov正则化的贝叶斯解释

Tikhonov正则化等价于最大后验估计（MAP）：

$$
\delta\sigma_{\text{MAP}} = \arg\max_{\delta\sigma} p(\delta\sigma | \delta U)
$$

假设：
- 似然：$p(\delta U | \delta\sigma) \propto \exp\left(-\frac{1}{2}\|\mathbf{J}\delta\sigma - \delta U\|_{\Gamma_n^{-1}}^2\right)$
- 先验：$p(\delta\sigma) \propto \exp\left(-\frac{1}{2}\alpha\|\mathbf{R}\delta\sigma\|^2\right)$

则：
$$
\delta\sigma_{\text{MAP}} = \arg\min_{\delta\sigma} \|\mathbf{J}\delta\sigma - \delta U\|_{\Gamma_n^{-1}}^2 + \alpha\|\mathbf{R}\delta\sigma\|^2
$$

这正是Tikhonov正则化！

#### L-曲线准则

选择最优正则化参数 $\alpha$：

```python
import numpy as np
import matplotlib.pyplot as plt

alphas = np.logspace(-2, 6, 50)
residual_norms = []
solution_norms = []

for alpha in alphas:
    delta_sigma = reconstructor.reconstruct(Uel, alpha, 0, 0)
    residual = np.linalg.norm(BJ @ delta_sigma - deltaU)
    solution = np.linalg.norm(delta_sigma)
    residual_norms.append(residual)
    solution_norms.append(solution)

# 绘制L-曲线
plt.loglog(residual_norms, solution_norms)
plt.xlabel('Residual Norm')
plt.ylabel('Solution Norm')
plt.title('L-curve')
```

最优 $\alpha$ 在L-曲线的"拐角"处。

---

## 参考文献

1. **线性化重建**：
   - Vauhkonen, M., et al. "Tikhonov regularization and prior information in electrical impedance tomography." *IEEE TMI*, 1998.

2. **深度学习后处理**：
   - Ronneberger, O., et al. "U-net: Convolutional networks for biomedical image segmentation." *MICCAI*, 2015.
   - Dhariwal, P., & Nichol, A. "Diffusion models beat GANs on image synthesis." *NeurIPS*, 2021.

3. **EIT综述**：
   - Holder, D. S. "Electrical impedance tomography: methods, history and applications." *IOP Publishing*, 2004.

4. **正则化理论**：
   - Hansen, P. C. "Rank-deficient and discrete ill-posed problems." *SIAM*, 1998.

---

## 附录

### A. 关键数据文件

| 文件名 | 内容 | 维度 |
|-------|------|------|
| `jac_sparse.npy` | Jacobian矩阵 | (2432, n_elements) |
| `mesh_neighbour_matrix_sparse.npy` | 网格邻接矩阵 | (n_elements, n_elements) |
| `smoothnessR_sparse.npy` | 平滑正则化矩阵 | (n_elements, n_elements) |
| `mesh_coordinates_sparse.npy` | 网格节点坐标 | (n_nodes, 2) |
| `cells_sparse.npy` | 网格单元定义 | (n_elements, 3) |

### B. 常见问题

**Q1: 为什么使用5个重建结果？**
- A: 提供多样化的先验信息，让深度学习模型自动选择和融合。实验表明5个是性能和计算成本的良好折中。

**Q2: 可以只用一个重建结果吗？**
- A: 可以，但性能会下降。单一参数难以适应所有情况。

**Q3: 如何选择网格密度？**
- A: 稀疏网格（sparse）计算快，适合实时应用；密集网格（dense）精度高，适合离线分析。

**Q4: U-Net的难度级别嵌入有多重要？**
- A: 非常重要。不同级别的数据质量差异大，条件输入让模型自适应调整。

**Q5: 可以用其他深度学习架构吗？**
- A: 可以，如DeepLab、SegNet等。U-Net因其优秀的性能和训练稳定性而常用。

### C. 代码组织结构

```
ktc2023_postprocessing/
├── main.py                          # 主入口
├── configs/
│   └── postprocessing_config.py    # 模型配置
├── src/
│   ├── __init__.py
│   ├── reconstruction/
│   │   ├── __init__.py
│   │   └── linearisedRecoFenics.py # 线性重建类
│   ├── postprocessing/
│   │   ├── __init__.py
│   │   └── utils.py                # 模型加载工具
│   ├── third_party_models/
│   │   ├── __init__.py
│   │   └── openai_unet/           # U-Net实现
│   └── ktc_methods/
│       └── ...                     # 辅助函数
├── data/
│   ├── jac_sparse.npy
│   ├── mesh_neighbour_matrix_sparse.npy
│   └── ...                         # 其他数据文件
└── postprocessing_model/
    └── version_01/                 # 预训练模型
```

---

## 总结

**KTC2023后处理方法**通过巧妙结合物理模型和深度学习，实现了高质量的EIT图像分割：

1. **第一阶段**：线性化Gauss-Newton重建
   - 快速、可解释
   - 多参数策略提供丰富信息
   - 利用物理先验

2. **第二阶段**：U-Net深度学习后处理
   - 强大的模式识别
   - 自动融合多重建结果
   - 适应不同难度级别

3. **协同效应**：
   - 物理模型的可靠性 + 深度学习的灵活性
   - 在KTC2023竞赛中取得优异成绩

这种混合方法为解决病态逆问题提供了有效范式，值得在其他成像领域推广。
