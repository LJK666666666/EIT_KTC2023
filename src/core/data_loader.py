"""
数据加载模块
支持 CDEIT 数据结构：train, valid, test, test2017, test2023
"""
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import scipy.io as sio


class EITDataset(Dataset):
    """EIT 数据集类"""

    def __init__(self, data_dir: str, dataset_type: str = 'train', use_eim: bool = True):
        """
        Args:
            data_dir: 数据根目录
            dataset_type: 数据集类型 ('train', 'valid', 'test', 'test2017', 'test2023')
            use_eim: 是否使用EIM格式
                    - True: 转换为EIM格式 [1, 16, 16] (用于CDEIT等)
                    - False: 保持原始格式 [1, 16, 13] (用于PyDbar等传统方法)
        """
        self.data_dir = Path(data_dir)
        self.dataset_type = dataset_type
        self.data_path = self.data_dir / dataset_type
        self.use_eim = use_eim

        if not self.data_path.exists():
            raise ValueError(f"Data path {self.data_path} does not exist")

        # 支持 .npz 和 .mat 格式
        self.data_files = sorted(list(self.data_path.glob('*.npz')))
        if len(self.data_files) == 0:
            self.data_files = sorted(list(self.data_path.glob('*.mat')))

        if len(self.data_files) == 0:
            raise ValueError(f"No .npz or .mat files found in {self.data_path}")

        # 确定文件格式
        if self.data_files[0].suffix == '.npz':
            self.file_format = 'npz'
        else:
            self.file_format = 'mat'

        # 加载归一化参数（用于EIM格式）
        if self.use_eim:
            mean_path = self.data_dir / 'mean.pth'
            std_path = self.data_dir / 'std.pth'

            if mean_path.exists() and std_path.exists():
                self.mean = torch.load(mean_path)  # [1, 16, 13]
                self.std = torch.load(std_path)    # [1, 16, 13]
            else:
                print(f"Warning: mean.pth or std.pth not found in {self.data_dir}, skipping normalization")
                self.mean = None
                self.std = None

        # 数据集特定的voltage系数（用于真实数据）
        if dataset_type == 'test2017':
            self.voltage = 1.040856e3
        elif dataset_type == 'test2023':
            self.voltage = 1978
        else:
            self.voltage = 1.0

        print(f"Loaded {len(self.data_files)} {self.file_format} files from {self.data_path}")

    def to_eim(self, voltage: torch.Tensor) -> torch.Tensor:
        """
        将16x13的电压数据转换为16x16的EIM (Electrode Imaging Matrix)

        Args:
            voltage: [1, 16, 13] tensor

        Returns:
            eim: [1, 16, 16] tensor
        """
        num = 16  # 电极数量
        eim = torch.zeros(1, num, num)

        # 遍历每一行
        for i in range(num):
            # 确定本行零的位置
            zero_positions = [(i + j) % num for j in range(3)]
            row = zero_positions[1]

            # 填充非零元素
            non_zero_index = 0
            for j in range(num):
                idx = j % num
                if idx not in zero_positions:
                    eim[:, row, idx] = voltage[:, row, non_zero_index]
                    non_zero_index += 1

        return eim

    def normalize(self, ys: torch.Tensor) -> torch.Tensor:
        """
        归一化电压数据

        Args:
            ys: [1, 16, 13] tensor

        Returns:
            normalized: [1, 16, 13] tensor
        """
        if self.mean is not None and self.std is not None:
            return (ys - self.mean) / self.std
        else:
            return ys

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            measurements: 测量数据
                - 如果use_eim=True: [1, 16, 16] EIM格式
                - 如果use_eim=False: [1, 16, 13] 原始电压格式
            conductivity: 电导率图像 [1, 128, 128] (如果存在)
        """
        file_path = self.data_files[idx]

        if self.file_format == 'npz':
            data = np.load(file_path)
            # NPZ 格式: ys=measurements, xs=conductivity
            if self.use_eim:
                # EIM 格式需要归一化：除以 voltage 系数
                measurements = data['ys'][:, 0] / self.voltage  # (208,)
            else:
                # 传统方法（如 PyDbar）使用原始电压差值，不除以 voltage
                measurements = data['ys'][:, 0]  # (208,) 保持原始范围

            if 'xs' in data:
                conductivity = data['xs']  # (128, 128)
            else:
                conductivity = None
        else:
            data = sio.loadmat(str(file_path))
            # MAT 格式
            if self.use_eim:
                measurements = data['measurements'].flatten() / self.voltage
            else:
                measurements = data['measurements'].flatten()

            if 'conductivity' in data:
                conductivity = data['conductivity']
            else:
                conductivity = None

        # 转换为 torch tensor
        measurements = torch.from_numpy(measurements.astype(np.float32))
        if conductivity is not None:
            conductivity = torch.from_numpy(conductivity.astype(np.float32))

        # 处理测量数据
        if measurements.dim() == 1:
            # Reshape to (16, 13)
            h, w = 16, 13
            if len(measurements) == h * w:
                measurements = measurements.view(h, w)
            else:
                # Pad or truncate
                if len(measurements) < h * w:
                    pad_size = h * w - len(measurements)
                    measurements = torch.nn.functional.pad(measurements, (0, pad_size))
                else:
                    measurements = measurements[:h*w]
                measurements = measurements.view(h, w)

            # Add channel dimension: [1, 16, 13]
            measurements = measurements.unsqueeze(0)

            if self.use_eim:
                # EIM格式: 归一化 -> 转换为EIM [1, 16, 16]
                measurements = self.normalize(measurements)  # [1, 16, 13]
                measurements = self.to_eim(measurements)      # [1, 16, 16]
            # 否则保持原始格式 [1, 16, 13]，不做任何处理

        # 处理电导率数据
        if conductivity is not None:
            if conductivity.dim() == 2:
                conductivity = conductivity.unsqueeze(0)  # [1, 128, 128]
            return measurements, conductivity
        else:
            return measurements, None


class EITDataModule:
    """EIT 数据模块 - 管理所有数据集的加载"""

    def __init__(self, config: Dict):
        """
        Args:
            config: 配置字典，包含：
                - data_dir: 数据根目录
                - batch_size: 批次大小
                - num_workers: 数据加载线程数
                - pin_memory: 是否使用锁页内存
                - use_eim: 是否使用EIM格式（默认True，用于CDEIT）
        """
        self.config = config
        self.data_dir = config['data_dir']
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.pin_memory = config.get('pin_memory', True)
        self.use_eim = config.get('use_eim', True)  # 默认使用EIM格式

        self.train_dataset: Optional[EITDataset] = None
        self.val_dataset: Optional[EITDataset] = None
        self.test_dataset: Optional[EITDataset] = None
        self.test2017_dataset: Optional[EITDataset] = None
        self.test2023_dataset: Optional[EITDataset] = None

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage in ('fit', 'train', None):
            self.train_dataset = EITDataset(self.data_dir, 'train', use_eim=self.use_eim)
            self.val_dataset = EITDataset(self.data_dir, 'valid', use_eim=self.use_eim)

        if stage in ('test', None):
            # 尝试加载测试数据集（如果存在）
            try:
                self.test_dataset = EITDataset(self.data_dir, 'test', use_eim=self.use_eim)
            except ValueError:
                print("Warning: No test dataset found")

            try:
                self.test2017_dataset = EITDataset(self.data_dir, 'test2017', use_eim=self.use_eim)
            except ValueError:
                print("Warning: No test2017 dataset found")

            try:
                self.test2023_dataset = EITDataset(self.data_dir, 'test2023', use_eim=self.use_eim)
            except ValueError:
                print("Warning: No test2023 dataset found")

    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        if self.train_dataset is None:
            raise ValueError("Train dataset not set up. Call setup('fit') first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        if self.val_dataset is None:
            raise ValueError("Validation dataset not set up. Call setup('fit') first.")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """测试数据加载器"""
        if self.test_dataset is None:
            return None

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test2017_dataloader(self) -> Optional[DataLoader]:
        """2017年真实数据加载器"""
        if self.test2017_dataset is None:
            return None

        return DataLoader(
            self.test2017_dataset,
            batch_size=self.batch_size,  # 使用配置中的 batch_size
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test2023_dataloader(self) -> Optional[DataLoader]:
        """2023年真实数据加载器"""
        if self.test2023_dataset is None:
            return None

        return DataLoader(
            self.test2023_dataset,
            batch_size=self.batch_size,  # 使用配置中的 batch_size
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
