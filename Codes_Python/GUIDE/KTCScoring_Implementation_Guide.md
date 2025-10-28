# KTCScoring.py 图像评分算法实现详解

## 目录

- [概述](#概述)
- [算法整体架构](#算法整体架构)
- [1. Otsu 多阈值分割算法](#1-otsu-多阈值分割算法)
  - [1.1 二分类 Otsu 算法](#11-二分类-otsu-算法)
  - [1.2 三分类 Otsu 算法](#12-三分类-otsu-算法)
- [2. 结构相似性指数 (SSIM)](#2-结构相似性指数-ssim)
  - [2.1 SSIM 基本公式](#21-ssim-基本公式)
  - [2.2 代码实现](#22-代码实现)
  - [2.3 高斯加权窗口](#23-高斯加权窗口)
- [3. 主评分函数](#3-主评分函数)
  - [3.1 评分流程](#31-评分流程)
  - [3.2 图像类别定义](#32-图像类别定义)
  - [3.3 评分策略](#33-评分策略)
- [4. 算法特点分析](#4-算法特点分析)
  - [4.1 Otsu分割的优势](#41-otsu分割的优势)
  - [4.2 SSIM评分的特点](#42-ssim评分的特点)
  - [4.3 整体评分策略](#43-整体评分策略)
- [5. 数学公式汇总](#5-数学公式汇总)
- [6. 在 EIT 重建中的应用](#6-在-eit-重建中的应用)
  - [6.1 在 main.py 中的使用](#61-在-mainpy-中的使用)
  - [6.2 评分指标解读](#62-评分指标解读)
- [7. 总结](#7-总结)

## 概述

本文档详细解释 `KTCScoring.py` 中实现的 EIT 图像质量评分算法。该代码包含 Otsu 多阈值分割和结构相似性（SSIM）评分功能，用于评估重建图像与真实图像之间的相似度。

## 算法整体架构

```python
# 主要功能函数
def Otsu(image, nvals, figno)          # 二分类Otsu分割
def Otsu2(image, nvals, figno)         # 三分类Otsu分割  
def scoringFunction(groundtruth, reconstruction)  # 主评分函数
def ssim(truth, reco)                  # 结构相似性计算
```

## 1. Otsu 多阈值分割算法

### 1.1 二分类 Otsu 算法

**数学原理：**
Otsu 方法通过最大化类间方差来自动确定最佳阈值。

```python
def Otsu(image, nvals, figno):
    histogramCounts, x = np.histogram(image.ravel(), nvals)
    total = np.sum(histogramCounts)
    
    sumB = 0      # 背景类累积和
    wB = 0        # 背景类权重
    maximum = 0.0
    sum1 = np.dot(np.arange(top), histogramCounts)  # 总均值计算
    
    for ii in range(1, top):
        wF = total - wB  # 前景类权重
        if wB > 0 and wF > 0:
            mF = (sum1 - sumB) / wF  # 前景类均值
            # 类间方差计算
            val = wB * wF * (((sumB / wB) - mF) ** 2)
            if val >= maximum:
                level = ii
                maximum = val
        
        wB = wB + histogramCounts[ii]
        sumB = sumB + (ii - 1) * histogramCounts[ii]
```

**关键公式：**
- **类间方差**: `σ²_b = w₀w₁(μ₀ - μ₁)²`
- **权重计算**: `w₀ = ∑_{i=0}^t p(i)`, `w₁ = 1 - w₀`
- **均值计算**: `μ₀ = ∑_{i=0}^t i·p(i)/w₀`, `μ₁ = ∑_{i=t+1}^{L-1} i·p(i)/w₁`

### 1.2 三分类 Otsu 算法

**扩展原理：**
将图像分为三个类别（低、中、高值），寻找两个最优阈值。

```python
def Otsu2(image, nvals, figno):
    histogramCounts, tx = np.histogram(image.ravel(), nvals)
    x = (tx[0:-1] + tx[1:])/2  # 直方图中心点
    
    muT = np.dot(np.arange(1, nvals+1), histogramCounts) / np.sum(histogramCounts)
    
    for ii in range(1, nvals):
        for jj in range(1, ii):
            w1 = np.sum(histogramCounts[:jj])     # 第一类权重
            w2 = np.sum(histogramCounts[jj:ii])   # 第二类权重  
            w3 = np.sum(histogramCounts[ii:])     # 第三类权重
            
            if w1 > 0 and w2 > 0 and w3 > 0:
                mu1 = np.dot(np.arange(1, jj+1), histogramCounts[:jj]) / w1
                mu2 = np.dot(np.arange(jj+1, ii+1), histogramCounts[jj:ii]) / w2
                mu3 = np.dot(np.arange(ii+1, nvals+1), histogramCounts[ii:]) / w3
                
                # 三分类类间方差
                val = w1 * ((mu1 - muT) ** 2) + w2 * ((mu2 - muT) ** 2) + w3 * ((mu3 - muT) ** 2)
                if val >= maximum:
                    level = [jj-1, ii-1]
                    maximum = val
```

**数学公式：**
```
σ²_b = w₁(μ₁ - μ_T)² + w₂(μ₂ - μ_T)² + w₃(μ₃ - μ_T)²
```
其中 μ_T 是全局均值。

## 2. 结构相似性指数 (SSIM)

### 2.1 SSIM 基本公式

结构相似性指数衡量两幅图像在亮度、对比度和结构三个方面的相似性：

```
SSIM(x,y) = [l(x,y)]^α · [c(x,y)]^β · [s(x,y)]^γ
```

其中：
- **亮度比较**: `l(x,y) = (2μ_xμ_y + C₁)/(μ_x² + μ_y² + C₁)`
- **对比度比较**: `c(x,y) = (2σ_xσ_y + C₂)/(σ_x² + σ_y² + C₂)`  
- **结构比较**: `s(x,y) = (σ_xy + C₃)/(σ_xσ_y + C₃)`

### 2.2 代码实现

```python
def ssim(truth, reco):
    c1 = 1e-4  # 亮度稳定性常数
    c2 = 9e-4  # 对比度稳定性常数
    r = 80     # 高斯核半径

    # 创建高斯加权窗口
    ws = np.ceil(2*r)
    wr = np.arange(-ws, ws+1)
    X, Y = np.meshgrid(wr, wr)
    ker = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * np.divide((np.square(X) + np.square(Y)), r**2))
    
    # 归一化因子
    correction = sps.convolve2d(np.ones(truth.shape), ker, mode='same')

    # 计算局部均值（高斯加权）
    gt = np.divide(sps.convolve2d(truth, ker, mode='same'), correction)
    gr = np.divide(sps.convolve2d(reco, ker, mode='same'), correction)

    # 计算局部方差和协方差
    mu_t2 = np.square(gt)      # 真实图像均值平方
    mu_r2 = np.square(gr)      # 重建图像均值平方
    mu_t_mu_r = np.multiply(gt, gr)  # 均值乘积

    sigma_t2 = np.divide(sps.convolve2d(np.square(truth), ker, mode='same'), correction) - mu_t2
    sigma_r2 = np.divide(sps.convolve2d(np.square(reco), ker, mode='same'), correction) - mu_r2
    sigma_tr = np.divide(sps.convolve2d(np.multiply(truth, reco), ker, mode='same'), correction) - mu_t_mu_r

    # SSIM 计算
    num = np.multiply((2*mu_t_mu_r + c1), (2*sigma_tr + c2))
    den = np.multiply((mu_t2 + mu_r2 + c1), (sigma_t2 + sigma_r2 + c2))
    ssimimage = np.divide(num, den)

    score = np.mean(ssimimage)  # 全局平均SSIM
```

### 2.3 高斯加权窗口

使用圆形对称高斯核进行局部统计计算：
```
ker(x,y) = (1/√(2π)) · exp(-0.5·(x² + y²)/r²)
```

**参数说明：**
- `r = 80`: 控制局部窗口大小
- `c1 = 1e-4`: 防止除零，稳定亮度比较
- `c2 = 9e-4`: 防止除零，稳定对比度比较

## 3. 主评分函数

### 3.1 评分流程

```python
def scoringFunction(groundtruth, reconstruction):
    # 1. 输入验证
    if (np.any(groundtruth.shape != np.array([256, 256])):
        raise Exception('The shape of the given ground truth is not 256 x 256!')
    
    if (np.any(reconstruction.shape != np.array([256, 256])):
        return 0  # 尺寸不匹配返回0分
    
    # 2. 提取导电物体（类别2）
    truth_c = np.zeros(groundtruth.shape)
    truth_c[np.abs(groundtruth - 2) < 0.1] = 1
    reco_c = np.zeros(reconstruction.shape)
    reco_c[np.abs(reconstruction - 2) < 0.1] = 1
    score_c = ssim(truth_c, reco_c)  # 导电物体SSIM评分

    # 3. 提取绝缘物体（类别1）
    truth_d = np.zeros(groundtruth.shape)
    truth_d[np.abs(groundtruth - 1) < 0.1] = 1
    reco_d = np.zeros(reconstruction.shape)
    reco_d[np.abs(reconstruction - 1) < 0.1] = 1
    score_d = ssim(truth_d, reco_d)  # 绝缘物体SSIM评分

    # 4. 综合评分
    score = 0.5*(score_c + score_d)
    return score
```

### 3.2 图像类别定义

在 EIT 重建中：
- **类别 0**: 背景介质
- **类别 1**: 绝缘物体（低电导率）
- **类别 2**: 导电物体（高电导率）

### 3.3 评分策略

**双类别独立评分：**
1. **导电物体评分** (`score_c`): 专门评估导电区域的定位精度
2. **绝缘物体评分** (`score_d`): 专门评估绝缘区域的定位精度  
3. **综合评分**: 两个评分的平均值

**容差处理：**
```python
np.abs(groundtruth - 2) < 0.1  # 允许0.1的数值容差
```

## 4. 算法特点分析

### 4.1 Otsu分割的优势
- **自适应阈值**: 无需人工设定阈值
- **多类别支持**: 可处理背景、绝缘体、导体三类
- **统计最优**: 基于直方图统计特性

### 4.2 SSIM评分的特点
- **结构感知**: 考虑图像结构信息，优于简单像素比较
- **局部评估**: 高斯窗口提供局部结构相似性
- **稳健性**: 对亮度、对比度变化不敏感

### 4.3 整体评分策略
- **类别平衡**: 导电和绝缘物体权重相等
- **容错处理**: 尺寸不匹配时返回0分
- **数值稳定**: 防止除零和边界情况

## 5. 数学公式汇总

### Otsu 阈值分割
**二分类类间方差**:
```
σ²_b(t) = w₀(t)w₁(t)[μ₀(t) - μ₁(t)]²
```

**三分类类间方差**:
```
σ²_b(t₁,t₂) = w₁(μ₁ - μ_T)² + w₂(μ₂ - μ_T)² + w₃(μ₃ - μ_T)²
```

### SSIM 结构相似性
**基本公式**:
```
SSIM(x,y) = (2μ_xμ_y + C₁)(2σ_xy + C₂) / ((μ_x² + μ_y² + C₁)(σ_x² + σ_y² + C₂))
```

**高斯加权统计量**:
```
μ_x = (x * w) / sum(w)
σ_x² = (x² * w) / sum(w) - μ_x²
σ_xy = (x·y * w) / sum(w) - μ_xμ_y
```

## 6. 在 EIT 重建中的应用

### 6.1 在 main.py 中的使用
```python
# 使用Otsu2进行三类分割
level, x = KTCScoring.Otsu2(deltareco_pixgrid.flatten(), 256, 7)

# 根据阈值分类
ind0 = deltareco_pixgrid < x[level[0]]      # 低电导率
ind1 = (deltareco_pixgrid >= x[level[0]]) & (deltareco_pixgrid <= x[level[1]])  # 中等
ind2 = deltareco_pixgrid > x[level[1]]      # 高电导率

# 确定背景并重新标记
inds = [np.count_nonzero(ind0), np.count_nonzero(ind1), np.count_nonzero(ind2)]
bgclass = inds.index(max(inds))  # 像素最多的为背景
```

### 6.2 评分指标解读
- **SSIM 范围**: [0, 1]，1表示完全相似
- **评分意义**: 衡量重建图像与真实图像的结构相似性
- **应用场景**: 算法比较、参数调优、质量控制

## 7. 总结

`KTCScoring.py` 实现了一个完整的 EIT 图像质量评估系统：

1. **自适应分割**: 使用 Otsu 方法自动确定最优阈值
2. **结构感知**: 采用 SSIM 评估图像结构相似性
3. **类别平衡**: 对导电和绝缘物体分别评分
4. **稳健性**: 包含容错处理和数值稳定性措施

该评分系统为 EIT 重建算法提供了客观、定量的性能评估标准。