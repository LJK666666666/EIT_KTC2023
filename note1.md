  ## 1) DataLoader：建议加 persistent_workers / prefetch_factor                                                                                                              
                                                                                                                                                                             
  位置：src/core/data_loader.py 里 train_dataloader/val_dataloader/test_dataloader(...)                                                                                      
                                                                                                                                                                             
  目前 DataLoader 只有：                                                                                                                                                     
                                                                                                                                                                             
  - num_workers=self.num_workers                                                                                                                                             
  - pin_memory=self.pin_memory                                                                                                                                               
                                                                                                                                                                             
  建议：                                                                                                                                                                     
                                                                                                                                                                             
  - 当 num_workers > 0 时加：persistent_workers=True（每个 epoch 不重启 worker，一般能明显省时间）                                                                           
  - 适当加：prefetch_factor=2~4（加大预取，减少 GPU 等数据）                                                                                                                 
  - 训练集可加：drop_last=True（可选，小幅提升吞吐，代价是丢掉最后不足 batch 的样本）                                                                                        
                                                                                                                                                                             
  ## 2) num_workers：要根据环境调，不是越大越好                                                                                                                              
                                                                                                                                                                             
  你 config 里现在常见是 num_workers: 4。                                                                                                                                    
                                                                                                                                                                             
  - Colab：通常 2~8 比较合适（取决于 CPU 核数、npz 解压/IO 压力）；过大可能反而更慢或触发资源争用。                                                                          
  - Windows 本地：有时多进程 DataLoader 会更麻烦，但你主要在 Colab 训练的话问题不大。                                                                                        
                                                                                                                                                                             
  ## 3) 数据预处理：to_eim() 的双重 for 循环可能是数据侧大头                                                                                                                 
                                                                                                                                                                             
  位置：src/core/data_loader.py → EITDataset.to_eim()                                                                                                                        
                                                                                                                                                                             
  它每个样本都会做 Python 循环+逐元素赋值（16×16 级别，但仍然是 Python 循环），在 batch 小、模型快（或 IO 快）时会拖慢整体。                                                 
  建议优化方向：                                                                                                                                                             
                                                                                                                                                                             
  - 预先构建一个 index 映射表（把 16×13 映射到 16×16 的位置），用张量索引/一次性赋值替代循环。                                                                               
                                                                                                                                                                             
  这通常比单纯调 num_workers 更“本质”。                                                                                                                                      
                                                                                                                                                                             
  ## 4) GPU 侧：可以考虑 AMP / TF32（如果你用 A100/T4 之类）                                                                                                                 
                                                                                                                                                                             
  位置：src/core/trainer.py 训练循环（目前是纯 FP32）                                                                                                                        
                                                                                                                                                                             
  - 对 CNN / FNO 一般都能提速：AMP（torch.cuda.amp.autocast + GradScaler）                                                                                                   
  - 若是 Ampere（A100/T4 也是），可开 TF32（对 matmul/conv 有提速，精度影响通常可接受）                                                                                      
                                                                                                                                                                             
  这属于“更大改动”，但收益也很可观。                                                                                                                                         
                                                                                                                                                                             
  ———                                                                                                                                                                        
                                                                                                                                                                             
  你更想先做哪一类？                                                                                                                                                         
                                                                                                                                                                             
  1. 只改 DataLoader 参数（persistent_workers/prefetch_factor）                                                                                                              
  2. 优化 to_eim() 映射（去掉 Python 循环）                                                                                                                                  
  3. 加 AMP/TF32（训练器改动更大）                                                                                                                                           


› src\inference.py 和 src\evaluate.py 有什么区别？                                                                                                                          
                                                                                                                                                                            
                                                                                                                                                                            
• 我理解你的问题是想区分这两个脚本在项目中的职责。我先快速对比 src/inference.py 和 src/evaluate.py 的入口参数、数据来源和输出内容，然后给你一个直接可用的结论。             
                                                                                                                                                                            
• 核心区别是：src/inference.py 是当前通用推理入口，src/evaluate.py 是旧的专项评估脚本（偏单一模型+KTC评分）。                                                               
                                                                                                                                                                            
  1. src/inference.py                                                                                                                                                       
                                                                                                                                                                            
  - 面向你现在的统一框架：create_method(...)，支持 cnn/diffusion/traditional/deepdbar/fno/dbar。                                                                            
  - 支持 test/test2017/test2023 数据集。                                                                                                                                    
  - 可加载 checkpoint 做批量推理，保存重建图、.mat、metrics.json。
  - 评估指标用 EITEvaluator（MSE/MAE/PSNR/SSIM/relative_error）。

  2. src/evaluate.py

  - 不是走统一 methods 框架，而是直接 from model import get_model。
  - 使用 eval_dataset.py 的 level 数据集，按 level 评估。
  - 核心指标是 KTCScoring.scoringFunction，并生成 summary report。
  - 更像历史/专项评测工具，不是你现在 main.py 工作流里的主路径。

  简化理解：

  - 日常训练后推理与结果导出：用 src/inference.py
  - 想跑 KTC 评分体系和 level 报告：才用 src/evaluate.py




python src/generate_simdata_32.py --output_dir data32 --num_train 40000 --num_valid 5000 --workers 8
================================================================================
Generate 32-electrode simulated dataset
================================================================================
Output dir: data32
Train samples: 40000
Valid samples: 5000
Noise: on (std1=0.1, std2=0.0)
Protocol: challenge76
Use evaluation pattern: True
Workers: 8
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
D:\020_Software\M\miniconda\Miniconda3\lib\site-packages\scipy\sparse\_index.py:168: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil and dok are more efficient.
  self._set_intXint(row, col, x.flat[0])
Measurement dimension: 2356
Injection patterns: 76
Measurements per injection: 31
Generating train:   0%|                                                   | 0/40000 [00:00<?, ?it/s]Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Generating train: 100%|████████████████████████████████████| 40000/40000 [12:55:39<00:00,  1.16s/it]
Generating valid:   0%|                                                    | 0/5000 [00:00<?, ?it/s]Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
Loaded evaluation measurement pattern:
  Measurements per injection: 31
  Injection patterns: 76
  Total measurements: 2356
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Loaded evaluation measurement pattern:
  Injection patterns: 76
  Measurements per injection: 31
  Total measurements: 2356
Generating valid: 100%|███████████████████████████████████████| 5000/5000 [1:43:22<00:00,  1.24s/it]
--------------------------------------------------------------------------------
Done. Total samples: 45000
Total time: 52744.11s
Average per sample: 1.1721s
Saved to: data32


• 可以按下面流程直接做。                                                                                                                                                
                                                                                                                                                                        
  1. 在 data32 上训练（你刚生成的仿真集）                                                                                                                               
                                                                                                                                                                        
  python main.py train --method fno --config src/configs/fno_config.yaml --data_dir data32 --num_epochs 200 --batch_size 64 --result_dir fno_sim32                      
                                                                                                                                                                        
  或 CNN：                                                                                                                                                              
                                                                                                                                                                        
  python main.py train --method cnn --config src/configs/cnn_config.yaml --data_dir data32 --num_epochs 200 --batch_size 64 --result_dir cnn_sim32                      
                                                                                                                                                                        
  训练完成后权重在：                                                                                                                                                    
                                                                                                                                                                        
  - results/fno_sim32_XX/best_model.pth                                                                                                                                 
  - results/fno_sim32_XX/last.pth                                                                                                                                       
                                                                                                                                                                        
  ———                                                                                                                                                                   
                                                                                                                                                                        
  2. 在 KTC2023 EvaluationData_full 上测试（对应 test2023）                                                                                                             
                                                                                                                                                                        
  python main.py inference --method fno --config src/configs/fno_config.yaml --checkpoint results/fno_sim32_XX/best_model.pth --data_dir data --dataset test2023        
  --output_dir fno_sim32_test2023                                                                                                                                       
                                                                                                                                                                        
  ———                                                                                                                                                                   
                                                                                                                                                                        
  3. 在 KTC2023 EvaluationData（删测量版本）上测试（对应 test2017）                                                                                                     
                                                                                                                                                                        
  python main.py inference --method fno --config src/configs/fno_config.yaml --checkpoint results/fno_sim32_XX/best_model.pth --data_dir data --dataset test2017        
  --output_dir fno_sim32_test2017                                                                                                                                       
                                                                                                                                                                        
  ———                                                                                                                                                                   
                                                                                                                                                                        
  4. 只测单个样本（快速调参）                                                                                                                                           
                                                                                                                                                                        
  python main.py inference --method fno --config src/configs/fno_config.yaml --checkpoint results/fno_sim32_XX/best_model.pth --data_dir data --dataset test2023        
  --sample_idx 1 --output_dir fno_sim32_test2023_single                                                                                                                 
                                                                                                                                                                        
  ———                                                                                                                                                                   
                                                                                                                                                                        
  5. 若要开启物理可微优化（已支持通用方法）                                                                                                                             
                                                                                                                                                                        
  python main.py inference --method fno --config src/configs/fno_config.yaml --checkpoint results/fno_sim32_XX/best_model.pth --data_dir data --dataset test2023        
  --sample_idx 1 --test_opt_physics --test_opt_mode contour_step --test_opt_steps 200 --test_opt_stage2_steps 30 --output_dir fno_sim32_test2023_opt                    
                                                                                                                                                                        
  ———                                                                                                                                                                   
                                                                                                                                                                        
  说明：                                                                                                                                                                
                                                                                                                                                                        
  - 你训练用的是 data32，但测试 KTC 官方数据要用 --data_dir data，因为 test2023/test2017 在 data/ 下。                                                                  
  - test2023 对应 EvaluationData_full，test2017 对应 EvaluationData。                                                                                                   
 
