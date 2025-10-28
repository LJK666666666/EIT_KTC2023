import os
os.chdir('D:/010_CodePrograms/E/EIT_KTC2023/Codes_Python')
print('当前目录:', os.getcwd())
import sys
sys.path.append('.')

try:
    import numpy
    print('numpy导入成功')
    import scipy
    print('scipy导入成功')
    import KTCScoring
    print('KTCScoring导入成功')
    print('所有依赖库导入成功！')
except ImportError as e:
    print(f'导入错误: {e}')