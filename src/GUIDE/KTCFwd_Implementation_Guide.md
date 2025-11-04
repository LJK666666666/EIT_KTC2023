# KTCFwd.py 正向电极模型实现详解

## 目录

- [概述](#概述)
- [数学理论基础](#数学理论基础)
  - [控制方程](#控制方程)
  - [变分形式](#变分形式)
  - [有限元离散化](#有限元离散化)
  - [矩阵分块结构](#矩阵分块结构)
- [代码实现详解](#代码实现详解)
  - [EITFEM 类结构](#eitfem-类结构)
  - [SolveForward 方法核心流程](#solveforward-方法核心流程)
  - [关键数值方法](#关键数值方法)
  - [雅可比矩阵计算](#雅可比矩阵计算)
  - [特殊数据结构](#特殊数据结构)
- [实现特点与优化](#实现特点与优化)
- [算法复杂度分析](#算法复杂度分析)
- [应用场景](#应用场景)
- [总结](#总结)

## 概述

本文档详细解释 `KTCFwd.py` 中实现的电阻抗断层成像（EIT）正向电极模型。该代码使用有限元方法（FEM）求解 EIT 问题的正问题，即给定电导率分布和电极阻抗，计算电极上的测量电压。

## 数学理论基础

### 控制方程

EIT 正问题由以下椭圆型偏微分方程描述：

```
∇ · (σ∇u) = 0, 在域 Ω 内
```

边界条件采用完整的电极模型（Complete Electrode Model, CEM）：

```
σ ∂u/∂n = 0, 在非电极边界

u + z_l σ ∂u/∂n = V_l, 在电极边界 E_l

∫_E_l σ ∂u/∂n ds = I_l, 在电极边界 E_l
```

其中：
- σ: 电导率分布
- u: 电势场
- z_l: 第 l 个电极的接触阻抗
- V_l: 第 l 个电极的电势
- I_l: 第 l 个电极注入的电流

### 变分形式

通过伽辽金方法，上述边值问题转化为以下弱形式：

```
∫_Ω σ∇u · ∇v dΩ + ∑_{l=1}^L (1/z_l) ∫_{E_l} (u - V_l)(v - V_l) ds = 0
```

其中 v 是测试函数。

### 有限元离散化

将电势场 u 和测试函数 v 用基函数展开：

```
u(x) = ∑_{j=1}^N u_j φ_j(x), v(x) = ∑_{i=1}^N v_i φ_i(x)
```

代入弱形式得到线性系统：

```
∑_{j=1}^N u_j [∫_Ω σ∇φ_i · ∇φ_j dΩ + ∑_{l=1}^L (1/z_l) ∫_{E_l} φ_i φ_j ds] 
- ∑_{l=1}^L V_l (1/z_l) ∫_{E_l} φ_i ds = 0
```

定义系统矩阵和右端项：

```
A_{ij} = ∫_Ω σ∇φ_i · ∇φ_j dΩ + ∑_{l=1}^L (1/z_l) ∫_{E_l} φ_i φ_j ds

b_i = ∑_{l=1}^L V_l (1/z_l) ∫_{E_l} φ_i ds
```

### 矩阵分块结构

由于包含电极电势未知量，系统矩阵采用分块形式：

```
[A] [u] = [b]
```

其中：
```
A = [K + A_σ   M]
    [M^T      S]

u = [u_nodes]
    [V_electrodes]

b = [0]
    [I]
```

- **A_σ**: 电导率相关的刚度矩阵
- **K**: 电极阻抗相关的边界积分矩阵
- **M**: 电极-节点耦合矩阵
- **S**: 电极-电极耦合矩阵
- **I**: 电流注入向量

## 有限元离散化

### 网格和基函数

代码使用二阶三角形单元（6节点三角形）进行离散化。每个单元有：
- 3个角节点
- 3个边中点节点

基函数采用二次拉格朗日多项式。

### 系统矩阵组装

通过有限元离散化，得到线性系统：

```
[A] [u] = [b]
```

其中：
- A = A_σ + S_z（电导率刚度矩阵 + 电极阻抗矩阵）
- u: 节点电势和电极电势向量
- b: 电流注入向量

## 代码实现详解

### EITFEM 类结构

```python
class EITFEM:
    def __init__(self, Mesh2, Inj, Mpat=None, vincl=None, sigmamin=None, sigmamax=None)
    def SolveForward(self, sigma, z)
    def Jacobian(self, sigma=None, z=None)
    def Jacobianz(self, sigma=None, z=None)
    # ... 其他辅助方法
```

### SolveForward 方法核心流程

#### 1. 电导率约束处理
```python
sigma[sigma < self.sigmamin] = self.sigmamin
sigma[sigma > self.sigmamax] = self.sigmamax
z[z < self.zmin] = self.zmin
```

将电导率和接触阻抗限制在合理范围内，避免数值不稳定。

#### 2. 电导率刚度矩阵 A0 组装

使用高斯积分计算每个单元的局部刚度矩阵：

```python
for ii in range(HN):  # 遍历所有三角形单元
    ind = self.Mesh2.H[ii, :]  # 节点索引
    gg = self.Mesh2.g[ind, :]   # 节点坐标
    ss = sigma[ind[[0, 2, 4]]]  # 角节点的电导率值
    int = self.grinprod_gauss_quad_node(gg, ss)  # 高斯积分计算局部矩阵
```

`grinprod_gauss_quad_node` 方法使用 3 点高斯积分计算：
- 积分点：`ip = [[1/2, 0], [1/2, 1/2], [0, 1/2]]`
- 权重：`w = [1/6, 1/6, 1/6]`

局部刚度矩阵元素计算公式：
```
∫_Ω σ∇φ_i · ∇φ_j dΩ = ∑_k w_k σ(x_k) ∇φ_i(x_k) · ∇φ_j(x_k) |det(J)| 
```

#### 3. 电极边界条件处理

对于电极边界上的单元：

```python
if self.Mesh2.Element[ii].Electrode:
    # 计算电极积分项
    s[InE] += (1/z[InE]) * self.electrlen(np.array([a, c]))
    bb1 = self.bound_quad1(np.array([a, b, c]))
    bb2 = self.bound_quad2(np.array([a, b, c]))
```

- `electrlen`: 计算电极长度
- `bound_quad1`: 边界积分第一类（线性项）
- `bound_quad2`: 边界积分第二类（二次项）

#### 4. 系统矩阵组装

最终系统矩阵结构：
```python
S0 = sp.sparse.bmat([
    [K, M],
    [M.T, S]
])
self.A = A0 + S0
```

其中：
- **A0**: 电导率相关的刚度矩阵
- **K**: 电极阻抗相关的边界积分矩阵
- **M**: 电极-节点耦合矩阵  
- **S**: 电极-电极耦合矩阵

#### 5. 右端项构建

```python
self.b = np.concatenate((np.zeros((self.ng2, self.Inj.shape[1])), 
                        self.C.T * self.Inj), axis=0)
```

- 上半部分：节点电势对应的零向量
- 下半部分：电流注入向量（通过 C 矩阵变换）

#### 6. 求解和结果提取

```python
UU = sp.sparse.linalg.spsolve(self.A, self.b)  # 求解线性系统
self.Pot = UU[0:self.ng2, :]  # 节点电势
self.Umeas = self.Mpat.T * self.C * self.theta[self.ng2:, :]  # 测量电压
```

### 关键数值方法

#### 高斯积分实现

**体积积分（grinprod_gauss_quad_node）:**
- 使用 3 点高斯积分规则
- 积分点：`[(1/2, 0), (1/2, 1/2), (0, 1/2)]`
- 权重：`[1/6, 1/6, 1/6]`

积分公式：
```python
for ii in range(3):
    S = [1 - ξ - η, ξ, η]  # 线性形函数
    L = [[4(ξ+η)-3, -8ξ-4η+4, 4ξ-1, 4η, 0, -4η],
         [4(ξ+η)-3, -4ξ, 0, 4ξ, 4η-1, -8η-4ξ+4]]
    Jt = L @ g  # 雅可比矩阵
    G = inv(Jt) @ L  # 形函数梯度
    int_sum += w[ii] * (S.T @ sigma) * G.T @ G * |det(Jt)|
```

**边界积分（bound_quad1 和 bound_quad2）:**
- `bound_quad1`: 2 点高斯积分，用于线性项
- `bound_quad2`: 3 点高斯积分，用于二次项

### 雅可比矩阵计算

#### 对电导率的雅可比

```python
def Jacobian(self, sigma=None, z=None):
    # 使用伴随变量法计算灵敏度
    Jleft = sp.sparse.linalg.spsolve(self.A.T, self.QC.T)
    Jright = self.theta
    
    for ii in range(m):
        Jid = self.dA[ii].i
        Jtemp = -Jleft.T[:, Jid] @ self.dA[ii].mat @ Jright[Jid, :]
        Js[:, ii] = Jtemp.T[self.mincl.T]
```

数学推导：
```
∂U/∂σ_j = -Q_C A^{-1} (∂A/∂σ_j) A^{-1} b
```

#### 对电极阻抗的雅可比

```python
def Jacobianz(self, sigma=None, z=None):
    dA_dz = self.ComputedA_dz(z)
    Jleft = -self.QC @ sp.sparse.linalg.inv(self.A)
    Jright = self.theta
    
    for ii in range(m):
        Jtemp = Jleft @ dA_dz[ii] @ Jright
        Jz[:, ii] = Jtemp.T[self.mincl.T]
```

### 特殊数据结构

#### CMATRIX 类

用于高效存储稀疏矩阵：
```python
class CMATRIX:
    def __init__(self, mat, indi, indj=None):
        self.mat = mat  # 稠密子矩阵
        self.i = indi   # 行索引
        self.j = indj   # 列索引（可选）
```

这种结构特别适合雅可比矩阵计算，因为只有部分节点受电导率变化影响。

## 实现特点与优化

### 1. 稀疏矩阵优化
- 使用 `scipy.sparse` 库处理大型稀疏系统
- 分块矩阵组装减少内存占用
- CSR 格式存储提高求解效率

### 2. 数值稳定性
- 电导率截断避免奇异矩阵
- 高斯积分保证精度
- 条件数控制

### 3. 模块化设计
- 分离的积分函数便于维护
- 清晰的矩阵组装流程
- 可扩展的雅可比计算

## 算法复杂度分析

- **矩阵组装**: O(N_elements × N_gauss_points)
- **线性求解**: O(N_nodes^{1.5}~2.5) 取决于稀疏求解器
- **雅可比计算**: O(N_measurements × N_parameters)

## 应用场景

该实现适用于：
- EIT 正问题仿真
- 灵敏度分析
- 逆问题求解的预处理
- 电极优化设计

## 总结

`KTCFwd.py` 实现了一个完整且高效的 EIT 正向电极模型，具有以下特点：

1. **理论完备**: 采用完整的电极模型，准确描述实际物理过程
2. **数值稳健**: 多重保护机制确保计算稳定性
3. **高效实现**: 稀疏矩阵和优化算法保证计算效率
4. **功能完善**: 支持正问题求解和灵敏度分析

该代码为 EIT 逆问题求解提供了可靠的数值基础，是电阻抗断层成像研究的重要工具。