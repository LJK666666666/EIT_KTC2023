"""
src 模块入口
支持 python -m src.train 等方式运行
"""
import sys

if __name__ == '__main__':
    print("Please run specific modules directly:")
    print("  python -m src.train --method cnn")
    print("  python -m src.train --method fno")
    print("  python -m src.inference --method cnn --checkpoint ...")
    print("  python -m src.inference --method fno --checkpoint ...")
    sys.exit(1)
