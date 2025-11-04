#!/usr/bin/env python
"""
Kaggle TPU v5e-8 运行脚本
直接在 Kaggle TPU 上运行，无需 accelerate launch
"""

import os
import sys
import argparse

def main():
    # Kaggle TPU 环境自动检测
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.xla_multiprocessing as xmp
        HAS_TPU = True
        print("✅ 检测到 TPU 环境，使用 TPU 运行")
    except ImportError:
        HAS_TPU = False
        print("⚠️ 未检测到 TPU，使用 GPU 运行")

    # 添加项目路径
    sys.path.insert(0, '/kaggle/working/CDEIT_TPU')

    # 导入原始 main 模块
    from main import main as original_main, test as original_test

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='CDEIT TPU Training')
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--data", type=str, choices=["simulated", "uef2017", "ktc2023"], default="simulated")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--samplingsteps", type=int, default=5)

    args = parser.parse_args()

    if HAS_TPU:
        # TPU 多进程运行
        def _mp_fn(index):
            """多进程函数，在每个 TPU 核上运行"""
            if args.mode == 'train':
                original_main(args)
            else:
                original_test(args)

        # 运行在所有可用的 TPU 核上
        xmp.spawn(_mp_fn, args=(), nprocs=8)  # v5e-8 有 8 个核
    else:
        # GPU 模式直接运行
        if args.mode == 'train':
            original_main(args)
        else:
            original_test(args)

if __name__ == "__main__":
    main()
