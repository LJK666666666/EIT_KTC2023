#!/usr/bin/env python
"""
EIT 重建框架入口脚本
在项目根目录执行训练和推理
"""
import sys
import os
from pathlib import Path

# 确保可以导入 src 模块
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == '__main__':
    # 获取命令
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|inference|test] [options]")
        print("\nExamples:")
        print("  python main.py train --method cnn --data_dir data --num_epochs 200")
        print("  python main.py inference --method cnn --checkpoint results/.../best_model.pth")
        print("  python main.py test")
        sys.exit(1)

    command = sys.argv[1]

    if command == 'train':
        # 训练命令
        from src.train import main
        sys.argv.pop(1)  # 移除 'train' 命令
        main()

    elif command == 'inference':
        # 推理命令
        from src.inference import main
        sys.argv.pop(1)  # 移除 'inference' 命令
        main()

    elif command == 'test':
        # 测试命令
        from test_framework import main
        sys.argv.pop(1)  # 移除 'test' 命令
        sys.exit(main())

    else:
        print(f"Unknown command: {command}")
        print("Available commands: train, inference, test")
        sys.exit(1)
