"""
评估数据加载器

加载KTC2023评估数据集和ground truth
"""

import os
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ktc_methods'))
import KTCFwd
import KTCMeshing
import KTCAux


class EvaluationDataset(Dataset):
    """
    KTC2023评估数据集加载器（使用完整电极测量数据）

    加载EvaluationData_full中的测量数据和GroundTruths中的真实图像
    """

    def __init__(self, level=1,
                 eval_data_path="EvaluationData_full/evaluation_datasets",
                 gt_path="EvaluationData/GroundTruths"):
        """
        Args:
            level: 评估级别 (1-7，这里level表示数据的难度级别，但都使用完整电极)
            eval_data_path: 评估数据路径（完整电极测量）
            gt_path: Ground truth路径
        """
        self.level = level
        self.eval_data_path = os.path.join(eval_data_path, f"level{level}")
        self.gt_path = os.path.join(gt_path, f"level_{level}")

        print(f"Loading evaluation data from: {self.eval_data_path}")
        print(f"Loading ground truth from: {self.gt_path}")

        # 加载参考数据和测量数据
        self._load_data()

    def _load_data(self):
        """加载数据文件"""
        # 加载参考测量
        ref_file = os.path.join(self.eval_data_path, "ref.mat")
        ref_data = sio.loadmat(ref_file)
        # 尝试不同的键名
        if 'Uref' in ref_data:
            self.Uref = ref_data['Uref']
        elif 'Uelref' in ref_data:
            self.Uref = ref_data['Uelref']
        else:
            raise KeyError(f"Cannot find reference voltage in {ref_file}. Keys: {list(ref_data.keys())}")

        # 查找所有data文件
        self.data_files = []
        idx = 1
        while True:
            data_file = os.path.join(self.eval_data_path, f"data{idx}.mat")
            if os.path.exists(data_file):
                self.data_files.append(data_file)
                idx += 1
            else:
                break

        self.num_samples = len(self.data_files)
        print(f"Found {self.num_samples} evaluation samples for level {self.level}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            measurements: EIT测量数据差分 [992]
            gt_image: Ground truth图像 [256, 256]
            data_idx: 数据索引 (1-based)
        """
        # 加载测量数据
        data_file = self.data_files[idx]
        data = sio.loadmat(data_file)

        # 尝试不同的键名
        if 'Umeas' in data:
            Umeas = data['Umeas']
        elif 'Uel' in data:
            Umeas = data['Uel']
        else:
            raise KeyError(f"Cannot find measurement voltage in {data_file}. Keys: {list(data.keys())}")

        # 计算差分测量
        deltaU = Umeas - self.Uref

        # 加载ground truth
        data_idx = idx + 1
        gt_file = os.path.join(self.gt_path, f"{data_idx}_true.mat")
        if os.path.exists(gt_file):
            gt_data = sio.loadmat(gt_file)
            if 'truth' in gt_data:
                gt_image = gt_data['truth']
            elif 'reconstruction' in gt_data:
                gt_image = gt_data['reconstruction']
            else:
                # 尝试第一个非系统键
                keys = [k for k in gt_data.keys() if not k.startswith('__')]
                if keys:
                    gt_image = gt_data[keys[0]]
                else:
                    gt_image = np.zeros((256, 256))
        else:
            # 如果没有ground truth，返回全零图像
            gt_image = np.zeros((256, 256))

        measurements = torch.from_numpy(deltaU).float().squeeze()
        gt_image = torch.from_numpy(gt_image).float()

        return measurements, gt_image, data_idx


def load_all_evaluation_levels(eval_data_path="EvaluationData_full/evaluation_datasets",
                                gt_path="EvaluationData/GroundTruths"):
    """
    加载所有评估级别的数据集

    Returns:
        dict: {level: dataset}
    """
    datasets = {}
    for level in range(1, 8):
        level_path = os.path.join(eval_data_path, f"level{level}")
        if os.path.exists(level_path):
            try:
                dataset = EvaluationDataset(level=level,
                                           eval_data_path=eval_data_path,
                                           gt_path=gt_path)
                datasets[level] = dataset
                print(f"Level {level}: {len(dataset)} samples")
            except Exception as e:
                print(f"Failed to load level {level}: {e}")

    return datasets


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing EvaluationDataset...")
    print("="*70)

    # 测试level 1
    dataset = EvaluationDataset(level=1)

    print(f"\nDataset length: {len(dataset)}")

    # 获取第一个样本
    measurements, gt_image, data_idx = dataset[0]

    print(f"\nSample {data_idx}:")
    print(f"  Measurements shape: {measurements.shape}")
    print(f"  GT image shape: {gt_image.shape}")
    print(f"  GT unique values: {torch.unique(gt_image)}")

    # 可视化
    fig, axes = plt.subplots(1, len(dataset), figsize=(5*len(dataset), 5))
    if len(dataset) == 1:
        axes = [axes]

    for i in range(len(dataset)):
        meas, gt, idx = dataset[i]
        axes[i].imshow(gt.numpy().T, origin='lower', cmap='gray')
        axes[i].set_title(f'Data {idx} GT')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('evaluation_dataset_preview.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved preview to 'evaluation_dataset_preview.png'")

    # 测试加载所有级别
    print("\n" + "="*70)
    print("Loading all evaluation levels...")
    all_datasets = load_all_evaluation_levels()
    print(f"\nSuccessfully loaded {len(all_datasets)} levels")
