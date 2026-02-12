## EIT Quick Notes (Project Memory)
### 1) 数据源与路径
- KTC2023评测数据有两套：`EvaluationData`（按难度删测量）与`EvaluationData_full`（完整测量）。
- 参考数据：`EvaluationData_full/evaluation_datasets/level1/ref.mat`，含`Injref(32,76)`、`Mpat(32,31)`、`Uelref(2356,1)`。
- 单样本测量：`EvaluationData_full/evaluation_datasets/level1/data1.mat`，含`Uel(2356,1)`。
- 真值示例：`EvaluationData_full/GroundTruths/level_1/1_true.mat`，键`truth`（分割标签图）。
### 2) 训练/推理数据格式
- `data/*/*.npz`常见字段：`ys(208,1)`、`xs(128,128)`、`xs_gn(128,128)`、`TR(128,128)`。
- 在本项目中，网络输入链路通常为：`ys -> reshape(16,13) -> /voltage -> normalize(mean/std) -> to_eim(16,16)`。
- 电压缩放系数：`test2017`用`1.040856e3`，`test2023`用`1978`，仿真通常`1.0`。
### 3) EIM映射要点
- 16电极EIM每行保留13个值，3个位置置零：`(i-1)%16`、`i`、`(i+1)%16`。
- `to_eim`实现已在`src/core/data_loader.py`与`programs/CDEIT/dataset.py`中验证一致。
### 4) 指标速记
- Relative Error：`||pred-target||2 / ||target||2`，越小越好。
- Corr（Pearson相关）：范围`[-1,1]`，越接近`1`越好；高Corr不等于幅值正确。
### 5) 前馈/物理模型调试结论（当前）
- 物理后端应优先对齐真实测量协议：`Injref/Mpat/Uelref`，不要自造注入模式。
- 线性化参考建议与KTC背景一致（常见`0.745`）；但标签到物理电导率映射仍需按数据生成规则继续校准。
- 仅看`U`域相关性可能偏乐观，必须同时看`deltaU = U-Uref`域误差。
### 6) 参考实现与资料
- 官方代码示例：`src/ktc_methods`。
- 冠军方案：`paper/other/DATA-DRIVEN APPROACHES FOR ELECTRICAL IMPEDANCE__TOMOGRAPHY IMAGE SEGMENTATION FROM PARTIAL__BOUNDAR.pdf`，代码：`programs/ktc2023_postprocessing`、`programs/ktc2023_fcunet`、`programs/ktc2023_conditional_diffusion`。
- 亚军代码：`programs/KTC2023-ABC2`。
- 季军代码：`programs/KTC2023-CUQI9`。
- CDEIT论文：`paper/other/A Conditional Diffusion Model for Electrical__Impedance Tomography Image Reconstruction.pdf`，代码：`programs/CDEIT`。
- EIM解释论文：`paper/other/Image_reconstruction_for_electrical_impedance_tomography_based_on_spatial_invariant_feature_maps_and_convolutional_neural_network.pdf`。
- FEM参考实现：`programs/eit_fenicsx`。
- EIT代码索引：`programs/awesome-eit/README.md`。
