# KTCRegularization.py 正则化先验实现详解

## 目录

- [概述](#概述)
- [数学理论基础](#数学理论基础)
  - [贝叶斯反演框架](#贝叶斯反演框架)
  - [高斯过程先验](#高斯过程先验)
  - [协方差函数](#协方差函数)
- [代码实现详解](#代码实现详解)
  - [SMPrior 类结构](#smprior-类结构)
  - [协方差矩阵计算](#协方差矩阵计算)
  - [样本生成方法](#样本生成方法)
  - [先验函数评估](#先验函数评估)
  - [梯度和海森矩阵计算](#梯度和海森矩阵计算)
- [数值实现特点](#数值实现特点)
- [算法复杂度分析](#算法复杂度分析)
- [在 EIT 逆问题中的应用](#在-eit-逆问题中的应用)
- [总结](#总结)

## 概述

本文档详细解释 `KTCRegularization.py` 中实现的高斯过程先验（Gaussian Process Prior）正则化方法。该代码为电阻抗断层成像（EIT）逆问题提供贝叶斯框架下的先验分布建模。

## 数学理论基础

### 贝叶斯反演框架

在贝叶斯反演中，后验分布由似然函数和先验分布组成：

```
p(σ|U) ∝ p(U|σ) · p(σ)
```

其中：
- p(σ|U): 后验分布
- p(U|σ): 似然函数（测量噪声模型）
- p(σ): 先验分布（正则化项）

### 高斯过程先验

代码实现的高斯过程先验假设电导率场 σ 服从多元高斯分布：

```
σ ~ N(μ, Γ_pr)
```

其中：
- μ: 先验均值向量
- Γ_pr: 先验协方差矩阵

### 协方差函数

#### 1. 平方距离协方差（Squared Distance）

```
Γ_pr(i,j) = a · exp(-‖g_i - g_j‖² / (2b²)) + c·δ_ij
```

其中：
- a = var - c
- b = √(-corrlength² / (2·ln(0.01)))
- c: 噪声项（默认 1e-9）
- δ_ij: Kronecker delta

#### 2. Ornstein-Uhlenbeck 协方差

```
Γ_pr(i,j) = a · exp(-‖g_i - g_j‖ / corrlength) + c·δ_ij
```

## 代码实现详解

### SMPrior 类结构

```python
class SMPrior:
    def __init__(self, ginv, corrlength, var, mean, covariancetype=None)
    def compute_L(self, g, corrlength, var)
    def draw_samples(self, nsamples)
    def eval_fun(self, args)
    def compute_hess_and_grad(self, args, nparam)
```

### 初始化方法

```python
def __init__(self, ginv, corrlength, var, mean, covariancetype=None):
    self.corrlength = corrlength  # 相关长度
    self.mean = mean              # 先验均值
    self.c = 1e-9                 # 噪声项
    if covariancetype is not None:
        self.covariancetype = covariancetype
    else:
        self.covariancetype = 'Squared Distance'  # 默认协方差类型
    self.compute_L(ginv, corrlength, var)
```

### 协方差矩阵计算

`compute_L` 方法构建协方差矩阵并计算其 Cholesky 分解：

```python
def compute_L(self, g, corrlength, var):
    ng = g.shape[0]  # 节点数量
    a = var - self.c  # 方差参数
    b = np.sqrt(-corrlength**2 / (2 * np.log(0.01)))  # 缩放参数
    Gamma_pr = np.zeros((ng, ng))  # 协方差矩阵初始化

    # 构建协方差矩阵
    for ii in range(ng):
        for jj in range(ii, ng):
            dist_ij = np.linalg.norm(g[ii, :] - g[jj, :])  # 节点间距离
            if self.covariancetype == 'Squared Distance':
                gamma_ij = a * np.exp(-dist_ij**2 / (2 * b**2))
            elif self.covariancetype == 'Ornstein-Uhlenbeck':
                gamma_ij = a * np.exp(-dist_ij / corrlength)
            else:
                raise ValueError('Unrecognized prior covariance type')
            
            # 对角线元素添加噪声项
            if ii == jj:
                gamma_ij = gamma_ij + self.c
            
            Gamma_pr[ii, jj] = gamma_ij
            Gamma_pr[jj, ii] = gamma_ij  # 对称性

    # 计算精度矩阵的 Cholesky 分解
    self.L = np.linalg.cholesky(np.linalg.inv(Gamma_pr)).T
```

**关键参数说明：**
- `a = var - self.c`: 方差参数，控制协方差幅值
- `b = √(-corrlength² / (2·ln(0.01)))`: 缩放参数，确保在相关长度处协方差衰减到 1%
- `c = 1e-9`: 噪声项，保证矩阵正定性

**协方差函数特性：**
- 距离衰减：节点距离越远，相关性越小
- 对称性：Γ_pr(i,j) = Γ_pr(j,i)
- 正定性：通过噪声项 c 保证

### 参数 b 的数学推导

对于平方距离协方差函数：
```
Γ(d) = a · exp(-d²/(2b²))
```
当 d = corrlength 时，我们希望 Γ(corrlength) = 0.01a，因此：
```
0.01 = exp(-corrlength²/(2b²))
ln(0.01) = -corrlength²/(2b²)
b² = -corrlength²/(2·ln(0.01))
b = √(-corrlength²/(2·ln(0.01)))
```

### 样本生成方法

`draw_samples` 方法用于从先验分布生成随机样本：

```python
def draw_samples(self, nsamples):
    samples = self.mean + np.linalg.solve(self.L, np.random.randn(self.L.shape[0], nsamples))
    return samples
```

**数学推导：**

设 L 是精度矩阵 Γ_pr⁻¹ 的 Cholesky 分解：
```
Γ_pr⁻¹ = L·Lᵀ
```

则从高斯分布 N(μ, Γ_pr) 生成样本的公式为：
```
σ = μ + L⁻¹·ε, 其中 ε ~ N(0, I)
```

由于 L 是下三角矩阵，使用 `np.linalg.solve(L, ε)` 高效计算 L⁻¹·ε。

### 先验函数评估

`eval_fun` 方法计算负对数先验概率：

```python
def eval_fun(self, args):
    sigma = args[0]
    res = 0.5 * np.linalg.norm(self.L @ (sigma - self.mean))**2
    return res
```

**数学推导：**

高斯先验的负对数概率密度为：
```
-log p(σ) ∝ 0.5 * (σ - μ)ᵀ Γ_pr⁻¹ (σ - μ)
```

由于 Γ_pr⁻¹ = L·Lᵀ，因此：
```
(σ - μ)ᵀ Γ_pr⁻¹ (σ - μ) = (σ - μ)ᵀ L Lᵀ (σ - μ) = ‖Lᵀ(σ - μ)‖²
```

### 梯度和海森矩阵计算

`compute_hess_and_grad` 方法计算先验项的梯度和海森矩阵：

```python
def compute_hess_and_grad(self, args, nparam):
    sigma = args[0]
    Hess = self.L.T @ self.L  # Γ_pr⁻¹
    grad = Hess @ (sigma - self.mean)

    # 处理参数维度不匹配的情况
    if nparam > len(sigma):
        Hess = np.block([[Hess, np.zeros((len(sigma), nparam - len(sigma)))],
                         [np.zeros((nparam - len(sigma), len(sigma))), 
                          np.zeros((nparam - len(sigma), nparam - len(sigma)))]])
        grad = np.concatenate([grad, np.zeros(nparam - len(sigma))])

    return Hess, grad
```

**数学推导：**

高斯先验的梯度：
```
∇[-log p(σ)] = Γ_pr⁻¹ (σ - μ)
```

高斯先验的海森矩阵：
```
∇²[-log p(σ)] = Γ_pr⁻¹
```

**维度处理：**
当参数空间维度大于电导率参数数量时，自动填充零矩阵和零向量，确保维度匹配。

## 数值实现特点

### 1. 空间相关性建模
- 基于节点间欧氏距离构建协方差
- 支持两种协方差函数：平方距离和 Ornstein-Uhlenbeck
- 相关长度控制空间平滑度

### 2. 数值稳定性
- 添加小噪声项 (c = 1e-9) 保证协方差矩阵正定性
- 使用 Cholesky 分解确保数值稳定性
- 对称矩阵构造避免数值误差

### 3. 计算效率
- 利用协方差矩阵对称性减少计算量
- Cholesky 分解支持高效采样和求逆
- 稀疏性考虑（虽然当前实现为稠密矩阵）

### 4. 贝叶斯框架集成
- 提供完整的先验分布建模
- 支持 MAP（最大后验概率）估计
- 便于 MCMC 采样

## 算法复杂度分析

- **协方差矩阵构建**: O(n²) - 需要计算所有节点对的距离
- **Cholesky 分解**: O(n³) - 矩阵求逆的主要开销
- **样本生成**: O(n²) - 三角矩阵求解
- **函数评估**: O(n²) - 矩阵向量乘法

## 在 EIT 逆问题中的应用

### 目标函数构造

EIT 逆问题的贝叶斯目标函数：
```
J(σ) = 0.5‖U_meas - F(σ)‖²_Γ_n⁻¹ + 0.5‖σ - μ‖²_Γ_pr⁻¹
```

其中：
- 第一项：数据拟合项（似然函数）
- 第二项：正则化项（先验分布）

### 参数选择指南

1. **相关长度 (corrlength)**: 
   - 控制空间平滑度
   - 较大值 → 更平滑的重建
   - 较小值 → 更细节的重建

2. **方差 (var)**:
   - 控制先验不确定性
   - 较大值 → 更弱的正则化
   - 较小值 → 更强的正则化

3. **均值 (mean)**:
   - 通常设为均匀背景电导率
   - 提供重建的基准参考

## 总结

`KTCRegularization.py` 实现了一个灵活且数学严谨的高斯过程先验模型，具有以下特点：

1. **理论基础扎实**: 基于贝叶斯统计和空间统计理论
2. **实现完整**: 支持先验评估、采样、梯度和海森矩阵计算
3. **数值稳健**: 多重机制保证数值稳定性
4. **应用广泛**: 适用于各种 EIT 逆问题求解方法

该实现为 EIT 图像重建提供了强大的正则化工具，能够有效处理逆问题的不适定性。