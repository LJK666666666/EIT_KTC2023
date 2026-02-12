  ## 计划标题
  在 inference 中新增“可微物理层测试时优化（TTO）”方案（线性化后端，可插拔架构）

  ## 摘要

  首版采用你确认的路线：线性化可微层 + 优化 sigma 图像 + inference 开关启用 + 数据一致性+L2平滑 + 默认20步 + 未来可插拔后端。


  3. 支持 --dataset test|test2017|test2023 全部测试集。
  4. 结果目录、图像保存、指标汇总沿用现有机制，不改训练产物结构。

  ## 公开接口与命令行变更

  在 src/inference.py 增加以下参数（只在 --method cnn 且开启后使用）：

  1. --test_opt_physics
     说明：启用测试时物理优化（默认关闭）。                                                                                                                                
  2. --test_opt_steps（默认 20）                                                                                                                                           
     说明：每样本优化步数。
  3. --test_opt_lr（默认 1e-2）                                                                                                                                            
     说明：优化学习率。                                                                                                                                                    
  4. --test_opt_lambda_smooth（默认 1e-4）                                                                                                                                 
     说明：L2 平滑权重。                                                                                                                                                   
  5. --test_opt_backend（默认 linearized_ktc）                                                                                                                             
     说明：后端类型，先实现一个值，接口为可扩展。                                                                                                                          
  6. --test_opt_save_curve（默认开启）                                                                                                                                     
     说明：保存每样本迭代 loss 曲线（你要求“保存每步 loss 曲线”）。                                                                                                        
                                                                                                                                                                           
  默认权重规则：                                                                                                                                                           
                                                                                                                                                                           
  3. 若未传 --checkpoint，自动设为 results/cnn_01/best_model.pth。

  ## 架构设计（可插拔）
  新增一个统一后端接口层，首版只实现 linearized_ktc：
     方法：

  - prepare(sample_context)：构建该样本的线性化物理算子。
  - predict(sigma_img)：输出与观测同域的预测测量（EIM 域）。
  2. LinearizedKTCBackend(PhysicsBackend)
     职责：用 KTC 前向与雅可比构造仿射可微层
     [
     其中 sigma_0 为 CNN 初始重建。

  1. src/methods/cnn/physics_backend.py：接口与后端注册。
  2. src/methods/cnn/physics_linearized_ktc.py：线性化后端实现。
  3. src/methods/cnn/test_time_opt.py：测试时优化循环（与后端解耦）。

  ## 关键数据流（决策完备）

  ### A. 推理主流程改造（仅 cnn + 开关）
  2. 对当前样本构建 LinearizedKTCBackend.prepare(...)。
     [
     \mathcal{L} = |y_{lin}(\sigma)-y_{obs}|_2^2 + \lambda |\nabla \sigma|_2^2
     ]
  5. 每样本保存 loss_curve_*.json 与 loss_curve_*.png（步数、data_loss、smooth_loss、total_loss）。
  ### B. 线性化后端构造

  1. 读取 KTC 网格与测量模式（src/ktc_methods 现有资产）。
  2. 将 sigma_0（128x128）映射到 FEM 节点域（固定映射矩阵，缓存复用）。
  - 计算 y0_raw（电压预测）
  - 计算 J_mesh = d y_raw / d sigma_mesh
  4. 通过固定线性映射转换到 EIM 观测域，得到：

  - y0_eim
  - J_img（链式映射后，对 128x128 像素展开变量）

  5. 转为 torch 常量张量后，predict(sigma) 只需矩阵乘法，自动可微，优化阶段极快。

  ### C. 与观测对齐
  1. y_obs 直接使用 dataloader 给出的当前样本测量张量（EIM 域）展开向量。
  2. predict(sigma) 输出与 y_obs 同维度、同归一化域，避免额外后处理歧义。
  3. 维度不匹配立即报错（不做替代策略）。

  ## 只测不训保证
  2. 不新增训练阶段参数，不改 checkpoint 保存策略。
  3. 新逻辑仅在 inference 路径且开关打开时执行。

  ## 性能与稳定性策略

  3. 使用进度条显示样本与迭代进度，满足长时任务可视化要求。
  4. 任何关键资产缺失（mesh/ref/checkpoint）直接抛错并停止。

  ## 测试用例与验收标准

  ### 功能测试

  1. 基线回归
     命令：python main.py inference --method cnn --dataset test2023 --checkpoint results/cnn_01/best_model.pth
     期望：行为与当前一致（不开启优化）。
  2. 默认权重回退
     命令：python main.py inference --method cnn --dataset test2023 --test_opt_physics
  3. 命令行优先
     期望：使用传入 checkpoint，不使用默认回退。
  4. 多数据集支持
     分别跑 test/test2017/test2023 + --test_opt_physics。
     期望：全部可运行，输出目录规则不变。


  1. 每样本 loss 曲线文件存在且步数=test_opt_steps。
  2. 至少 total_loss 有下降趋势（允许个别步震荡）。
  3. 汇总 metrics.json 仍能生成，字段完整。
  4. 未开开关时，输出与历史脚本一致（兼容性）。

  ### 失败场景测试

  1. 默认 checkpoint 不存在时，应明确报错。
  2. KTC 资产缺失/维度不一致时，应明确报错。
  3. 非 cnn 方法开启开关时，应明确报错说明“不支持”。

  ## 实现顺序

  1. 在 inference 加参数与默认 checkpoint 规则。
  2. 落地 PhysicsBackend 抽象与 LinearizedKTCBackend。
  3. 落地 test_time_opt 优化器（20 步 + 曲线保存）。
  4. 接入推理主循环（按样本调用优化）。
  5. 跑小样本冒烟与三组命令验收。

  ## 明确假设与默认值

  1. 默认优化仅面向 --method cnn。
  2. 默认后端 linearized_ktc，并保留可插拔接口。
  3. 默认损失：data_consistency + L2_smooth。
  4. 默认步数 20，并允许命令行覆盖。
  5. 默认权重规则：--checkpoint 优先，否则回退 results/cnn_01/best_model.pth。
  6. 只在测试时优化，不改训练。

