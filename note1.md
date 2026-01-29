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
 