# main.py EIT反向图像重建实现详解

## 目录

- [概述](#概述)
- [算法核心流程](#算法核心流程)
- [数学理论基础](#数学理论基础)
  - [线性化差分反演模型](#线性化差分反演模型)
  - [线性化近似](#线性化近似)
  - [贝叶斯正则化框架](#贝叶斯正则化框架)
- [代码实现详解](#代码实现详解)
  - [第一部分：数据加载和预处理](#第一部分数据加载和预处理)
  - [第二部分：有限元网格加载](#第二部分有限元网格加载)
  - [第三部分：正则化先验和正向求解器设置](#第三部分正则化先验和正向求解器设置)
  - [第四部分：正则化反问题求解](#第四部分正则化反问题求解)
  - [第五部分：图像后处理和分割](#第五部分图像后处理和分割)
- [算法特点和优化](#算法特点和优化)
- [数学公式汇总](#数学公式汇总)
- [应用场景和限制](#应用场景和限制)
- [总结](#总结)

## 概述

本文档详细解释 `main.py` 中实现的电阻抗断层成像（EIT）反向图像重建算法。该代码采用线性化差分方法结合贝叶斯正则化，从电压测量数据重建电导率分布。

## 算法核心流程

### 总体架构

```python
def main(inputFolder, outputFolder, categoryNbr):
    # 1. 加载参考数据和设置测量参数
    # 2. 加载有限元网格
    # 3. 设置正则化先验和正向求解器
    # 4. 对每个输入文件进行重建
    # 5. 插值、分割和保存结果
```

## 数学理论基础

### 线性化差分反演模型

EIT 反问题的基本数学模型为：

```
U = F(σ) + ε
```

其中：
- U: 测量电压向量
- F(σ): 非线性正向算子
- σ: 电导率分布
- ε: 测量噪声

### 线性化近似

在参考电导率 σ₀ 附近进行一阶泰勒展开：

```
U ≈ F(σ₀) + J(σ₀)(σ - σ₀) + ε
```

其中 J(σ₀) 是雅可比矩阵（灵敏度矩阵）。

定义电压差和电导率差：
```
ΔU = U - F(σ₀)
Δσ = σ - σ₀
```

得到线性化模型：
```
ΔU = J(σ₀)Δσ + ε
```

### 贝叶斯正则化框架

在贝叶斯框架下，反问题转化为最大后验概率估计：

```
argmin_Δσ { ||L_n(ΔU - JΔσ)||² + ||L_pr(Δσ - μ_pr)||² }
```

其中：
- L_n: 噪声精度矩阵的 Cholesky 分解
- L_pr: 先验精度矩阵的 Cholesky 分解
- μ_pr: 先验均值

## 代码实现详解

### 第一部分：数据加载和预处理

```python
# 加载参考数据和设置测量参数
Nel = 32  # 电极数量
z = (1e-6) * np.ones((Nel, 1))  # 接触阻抗

# 加载参考数据
mat_dict = sp.io.loadmat(os.path.join(inputFolder, 'ref.mat'))
Injref = mat_dict["Injref"]  # 参考电流注入模式
Uelref = mat_dict["Uelref"]  # 水箱参考电压测量值
Mpat = mat_dict["Mpat"]      # 电压测量模式

# 根据难度级别移除电极数据
vincl = np.ones(((Nel - 1), 76), dtype=bool)
rmind = np.arange(0, 2 * (categoryNbr - 1), 1)
```

**关键步骤：**
- 加载均匀介质的参考测量数据
- 根据难度级别移除部分电极数据，模拟实际测量限制
- 构建测量包含掩码 `vincl`

### 第二部分：有限元网格加载

```python
# 加载预先生成的稀疏网格
mat_dict_mesh = sp.io.loadmat('Mesh_sparse.mat')
g = mat_dict_mesh['g']        # 节点坐标
H = mat_dict_mesh['H']        # 单元拓扑
elfaces = mat_dict_mesh['elfaces'][0].tolist()  # 边界电极节点

# 构建一阶和二阶网格对象
Mesh = KTCMeshing.Mesh(H, g, elfaces, nodes, elements)
Mesh2 = KTCMeshing.Mesh(H2, g2, elfaces2, nodes2, elements2)
```

**网格特点：**
- 一阶网格用于反演计算
- 二阶网格用于高精度正向求解
- 电极边界条件精确建模

### 第三部分：正则化先验和正向求解器设置

```python
# 设置正则化先验
sigma0 = np.ones((len(Mesh.g), 1))      # 初始电导率猜测（均匀分布）
corrlength = 1 * 0.115                  # 相关长度
var_sigma = 0.05 ** 2                   # 先验方差
mean_sigma = sigma0                     # 先验均值
smprior = KTCRegularization.SMPrior(Mesh.g, corrlength, var_sigma, mean_sigma)

# 设置正向求解器
solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)

# 设置噪声模型
noise_std1 = 0.05  # 第一噪声分量（相对测量值）
noise_std2 = 0.01  # 第二噪声分量（相对最大测量值）
solver.SetInvGamma(noise_std1, noise_std2, Uelref)
```

**先验参数说明：**
- `sigma0`: 线性化点，假设为均匀电导率分布
- `corrlength`: 控制电导率场空间平滑度的相关长度
- `var_sigma`: 先验方差，控制正则化强度

**噪声模型：**
```
Γ_n⁻¹ = diag(1/((0.05·|U_meas|)² + (0.01·max(|U_meas|))²))
```

### 第四部分：正则化反问题求解

```python
# 对每个数据文件进行重建
for objectno in range(0, len(mat_files)):
    # 加载测量数据
    mat_dict2 = sp.io.loadmat(mat_files[objectno])
    Inj = mat_dict2["Inj"]      # 电流注入
    Uel = mat_dict2["Uel"]      # 电压测量
    deltaU = Uel - Uelref       # 计算电压差

    # 正向求解和雅可比计算
    Usim = solver.SolveForward(sigma0, z)  # 线性化点正向解
    J = solver.Jacobian(sigma0, z)         # 灵敏度矩阵

    # 正则化反问题求解
    mask = np.array(vincl, bool)
    deltareco = np.linalg.solve(
        J.T @ solver.InvGamma_n[np.ix_(mask, mask)] @ J + smprior.L.T @ smprior.L,
        J.T @ solver.InvGamma_n[np.ix_(mask, mask)] @ deltaU[vincl]
    )
```

**正则化方程推导：**

最小化目标函数：
```
min_Δσ ||L_n(ΔU - JΔσ)||² + ||L_prΔσ||²
```

对应的正规方程：
```
(JᵀΓ_n⁻¹J + Γ_pr⁻¹) Δσ = JᵀΓ_n⁻¹ΔU
```

其中：
- Γ_pr⁻¹ = L_prᵀL_pr
- Γ_n⁻¹ = L_nᵀL_n

**数值实现细节：**
- 使用 `np.linalg.solve` 直接求解线性系统
- 利用测量掩码 `vincl` 选择有效测量数据
- 精度矩阵 Γ_n⁻¹ 考虑了相对和绝对噪声分量

### 第五部分：图像后处理和分割

#### 插值到像素网格

```python
# 将重建结果插值到规则的像素网格
deltareco_pixgrid = KTCAux.interpolateRecoToPixGrid(deltareco, Mesh)
```

**插值目的：**
- 将不规则网格上的重建结果转换为规则像素网格
- 便于可视化和后续处理
- 统一图像尺寸和格式

#### Otsu 多阈值分割

```python
# 使用Otsu2方法进行三类阈值分割
level, x = KTCScoring.Otsu2(deltareco_pixgrid.flatten(), 256, 7)

# 根据阈值分类像素
deltareco_pixgrid_segmented = np.zeros_like(deltareco_pixgrid)
ind0 = deltareco_pixgrid < x[level[0]]      # 低电导率区域
ind1 = np.logical_and(deltareco_pixgrid >= x[level[0]], 
                      deltareco_pixgrid <= x[level[1]])  # 中等电导率区域
ind2 = deltareco_pixgrid > x[level[1]]      # 高电导率区域

# 确定背景类别（像素数最多的类别）
inds = [np.count_nonzero(ind0), np.count_nonzero(ind1), np.count_nonzero(ind2)]
bgclass = inds.index(max(inds))

# 重新标记分割图像
match bgclass:
    case 0:  # 背景为低电导率
        deltareco_pixgrid_segmented[ind1] = 2
        deltareco_pixgrid_segmented[ind2] = 2
    case 1:  # 背景为中等电导率
        deltareco_pixgrid_segmented[ind0] = 1
        deltareco_pixgrid_segmented[ind2] = 2
    case 2:  # 背景为高电导率
        deltareco_pixgrid_segmented[ind0] = 1
        deltareco_pixgrid_segmented[ind1] = 1
```

**Otsu 多阈值分割原理：**

Otsu 方法通过最大化类间方差来自动确定最佳阈值。对于三类分割：

1. 将直方图分为三个区间：低值、中值、高值
2. 计算每个可能阈值组合的类间方差
3. 选择使类间方差最大的阈值组合

**分割策略：**
- 将像素数最多的类别标记为背景（0）
- 其他两个类别标记为包含物（1 和 2）
- 适应不同的背景电导率情况

#### 结果保存和可视化

```python
# 保存分割结果
reconstruction = deltareco_pixgrid_segmented
mdic = {"reconstruction": reconstruction}
sp.io.savemat(outpath, mdic)

# 保存连续重建图像
fig, ax = plt.subplots()
cax = ax.imshow(deltareco_pixgrid, cmap='jet')
fig.savefig(reco_png, bbox_inches='tight')

# 保存分割图像
fig2, ax2 = plt.subplots()
cax2 = ax2.imshow(deltareco_pixgrid_segmented, cmap='gray')
fig2.savefig(seg_png, bbox_inches='tight')
```

## 算法特点和优化

### 1. 线性化差分方法优势
- **计算效率高**: 避免非线性迭代
- **数值稳定**: 线性系统求解稳定
- **实时性**: 适合在线监测应用

### 2. 贝叶斯正则化特点
- **物理意义明确**: 基于统计推断理论
- **自适应平滑**: 空间相关先验控制平滑度
- **噪声鲁棒**: 精确的噪声模型

### 3. 多级处理流程
- **数据预处理**: 电极数据选择和掩码处理
- **网格适配**: 一阶反演、二阶正向计算
- **后处理优化**: 插值、分割、可视化

### 4. 可扩展性设计
- **模块化架构**: 各功能模块独立
- **参数可调**: 噪声、先验参数可配置
- **难度分级**: 支持不同测量配置

## 数学公式汇总

### 反问题模型
```
ΔU = JΔσ + ε
```

### 正则化目标函数
```
min_Δσ { ||L_n(ΔU - JΔσ)||² + ||L_prΔσ||² }
```

### 正规方程
```
(JᵀΓ_n⁻¹J + Γ_pr⁻¹) Δσ = JᵀΓ_n⁻¹ΔU
```

### 噪声模型
```
Γ_n⁻¹ = diag(1/((α·|U_meas|)² + (β·max(|U_meas|))²))
```

## 应用场景和限制

### 适用场景
- 电导率变化较小的成像问题
- 实时监测和快速重建应用
- 医学成像、工业过程监测

### 算法限制
- 线性化近似在强非线性情况下失效
- 对初始猜测敏感
- 空间分辨率受正则化强度影响

## 总结

`main.py` 实现了一个完整的 EIT 线性化差分反演系统，具有以下特点：

1. **理论严谨**: 基于贝叶斯统计和有限元方法
2. **实现完整**: 从数据加载到结果输出的完整流程
3. **数值稳健**: 多重正则化和噪声处理机制
4. **实用性强**: 支持难度分级和自动分割

该实现为 EIT 图像重建提供了可靠的基础框架，特别适合电导率变化较小的应用场景。