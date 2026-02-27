"""
数据加载模块
支持 CDEIT 数据结构：train, valid, test, test2017, test2023
以及 KTC2023 官方 .mat 测试集（EvaluationData_full / EvaluationData）。
"""
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import scipy.io as sio
import re
from tqdm import tqdm


def _compute_matrix32_stats(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 matrix32 测量格式的逐位置均值/标准差，shape 为 [1, 31, 76]。
    使用 train 子目录数据，要求样本包含至少 31*76 个测量值。
    """
    train_path = data_dir / 'train'
    if not train_path.exists():
        raise ValueError(f"Train path not found for matrix32 stats: {train_path}")

    files = sorted(list(train_path.glob('*.npz')))
    if len(files) == 0:
        files = sorted(list(train_path.glob('*.mat')))
    if len(files) == 0:
        raise ValueError(f"No train files found for matrix32 stats in {train_path}")

    target_len = 31 * 76
    sum_arr = np.zeros((31, 76), dtype=np.float64)
    sum_sq_arr = np.zeros((31, 76), dtype=np.float64)
    count = 0

    for file_path in tqdm(files, desc="Computing matrix32 mean/std", leave=False):
        if file_path.suffix == '.npz':
            data = np.load(file_path)
            ys = np.asarray(data['ys'], dtype=np.float32).reshape(-1)
        else:
            data = sio.loadmat(str(file_path))
            if 'measurements' in data:
                ys = np.asarray(data['measurements'], dtype=np.float32).reshape(-1)
            elif 'ys' in data:
                ys = np.asarray(data['ys'], dtype=np.float32).reshape(-1)
            else:
                raise ValueError(f"Cannot find measurements in {file_path}")

        if ys.size < target_len:
            ys = np.pad(ys, (0, target_len - ys.size), mode='constant')
        else:
            ys = ys[:target_len]

        mat = ys.reshape(76, 31).T.astype(np.float64)
        sum_arr += mat
        sum_sq_arr += mat * mat
        count += 1

    mean = sum_arr / max(count, 1)
    var = sum_sq_arr / max(count, 1) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)

    mean_t = torch.from_numpy(mean.astype(np.float32)).unsqueeze(0)
    std_t = torch.from_numpy(std.astype(np.float32)).unsqueeze(0)
    return mean_t, std_t


class EITDataset(Dataset):
    """EIT 数据集类"""

    def __init__(
        self,
        data_dir: str,
        dataset_type: str = 'train',
        use_eim: bool = True,
        measurement_format: str = 'eim16'
    ):
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
        self.measurement_format = measurement_format
        self.mean = None
        self.std = None

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

        # 加载归一化参数（EIM 或 matrix32）
        if self.measurement_format == 'eim16':
            mean_path = self.data_dir / 'mean.pth'
            std_path = self.data_dir / 'std.pth'

            if mean_path.exists() and std_path.exists():
                self.mean = torch.load(mean_path)  # [1, 16, 13]
                self.std = torch.load(std_path)    # [1, 16, 13]
            else:
                print(f"Warning: mean.pth or std.pth not found in {self.data_dir}, skipping normalization")
        elif self.measurement_format == 'matrix32':
            mean_path = self.data_dir / 'mean_matrix32.pth'
            std_path = self.data_dir / 'std_matrix32.pth'
            if mean_path.exists() and std_path.exists():
                self.mean = torch.load(mean_path)  # [1, 31, 76]
                self.std = torch.load(std_path)    # [1, 31, 76]
            else:
                print(
                    f"Warning: mean_matrix32.pth or std_matrix32.pth not found in {self.data_dir}, "
                    "skipping matrix32 normalization"
                )

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
        num = 16

        # 预先构建映射索引，避免每个样本都进行 Python 循环填充
        if not hasattr(self, '_eim_col_index'):
            rows = torch.arange(num, dtype=torch.long).unsqueeze(1)  # [16,1]
            cols = torch.arange(num, dtype=torch.long).unsqueeze(0)  # [1,16]

            # 本行 3 个 0 的位置：row-1, row, row+1
            mask = (
                (cols != rows)
                & (cols != (rows - 1) % num)
                & (cols != (rows + 1) % num)
            )  # [16,16] bool

            # rank: 各行 True 位置在 0..12 的顺序索引；False 位置设为 -1
            rank = torch.cumsum(mask.to(torch.long), dim=1) - 1  # [16,16]
            col_index = torch.where(mask, rank, torch.full_like(rank, -1))  # [16,16]
            self._eim_col_index = col_index  # -1 表示该位置为 0

        col_index = self._eim_col_index.to(device=voltage.device)

        # 对于 -1 的位置用 0 作为安全索引，并在最后乘 mask 置零
        safe_index = col_index.clamp(min=0)  # [16,16]
        src = voltage[0]  # [16,13]
        gathered = src.gather(1, safe_index)  # [16,16]
        mask = (col_index >= 0).to(gathered.dtype)
        eim = gathered * mask
        return eim.unsqueeze(0)

    def normalize(self, ys: torch.Tensor) -> torch.Tensor:
        """
        归一化电压数据

        Args:
            ys: [1, H, W] tensor

        Returns:
            normalized: [1, H, W] tensor
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
            # 仅 eim16/raw16x13 路线使用 historical voltage scaling
            if self.measurement_format in ('eim16', 'raw16x13') and self.use_eim:
                measurements = data['ys'][:, 0] / self.voltage
            else:
                measurements = data['ys'][:, 0]

            if 'xs' in data:
                conductivity = data['xs']  # (128, 128)
            else:
                conductivity = None
        else:
            data = sio.loadmat(str(file_path))
            # MAT 格式
            if self.measurement_format in ('eim16', 'raw16x13') and self.use_eim:
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
            if self.measurement_format == 'matrix32':
                # 32电极挑战协议：76个注入，每个31个测量 => [1, 31, 76]
                target_len = 31 * 76
                if len(measurements) < target_len:
                    pad_size = target_len - len(measurements)
                    measurements = torch.nn.functional.pad(measurements, (0, pad_size))
                else:
                    measurements = measurements[:target_len]
                measurements = measurements.view(76, 31).transpose(0, 1).contiguous()
                measurements = measurements.unsqueeze(0)
                measurements = self.normalize(measurements)
            else:
                # 16电极EIM链路：先变成[1,16,13]
                h, w = 16, 13
                if len(measurements) == h * w:
                    measurements = measurements.view(h, w)
                else:
                    if len(measurements) < h * w:
                        pad_size = h * w - len(measurements)
                        measurements = torch.nn.functional.pad(measurements, (0, pad_size))
                    else:
                        measurements = measurements[:h * w]
                    measurements = measurements.view(h, w)
                measurements = measurements.unsqueeze(0)
                if self.measurement_format == 'eim16':
                    measurements = self.normalize(measurements)
                    measurements = self.to_eim(measurements)

        # 处理电导率数据
        if conductivity is not None:
            if conductivity.dim() == 2:
                conductivity = conductivity.unsqueeze(0)  # [1, 128, 128]
            return measurements, conductivity
        else:
            return measurements, None


class KTCChallengeDataset(Dataset):
    """KTC 官方挑战数据集（32电极全协议）"""

    def __init__(
        self,
        source: str = 'full',
        level: int = 1,
        measurement_format: str = 'matrix32',
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None
    ):
        if source not in ('full', 'eval'):
            raise ValueError(f"source must be 'full' or 'eval', got {source}")
        if measurement_format != 'matrix32':
            raise ValueError(
                f"KTC challenge dataset requires measurement_format='matrix32', got {measurement_format}"
            )
        self.source = source
        self.level = int(level)
        self.measurement_format = measurement_format
        self.mean = mean
        self.std = std
        self.voltage = 1.0

        project_root = Path(__file__).resolve().parents[2]
        base_name = 'EvaluationData_full' if source == 'full' else 'EvaluationData'
        self.eval_path = project_root / base_name / 'evaluation_datasets' / f'level{self.level}'
        self.gt_path = project_root / base_name / 'GroundTruths' / f'level_{self.level}'

        if not self.eval_path.exists():
            raise ValueError(f"Evaluation path not found: {self.eval_path}")

        ref_path = self.eval_path / 'ref.mat'
        if not ref_path.exists():
            raise ValueError(f"Reference file not found: {ref_path}")

        ref_data = sio.loadmat(str(ref_path))
        if 'Uelref' not in ref_data:
            raise ValueError(f"'Uelref' not found in {ref_path}")
        self.uel_ref = np.asarray(ref_data['Uelref'], dtype=np.float32).reshape(-1)

        pattern = re.compile(r'^data(\d+)\.mat$')
        indexed_files = []
        for p in self.eval_path.glob('data*.mat'):
            m = pattern.match(p.name)
            if m is None:
                continue
            indexed_files.append((int(m.group(1)), p))
        indexed_files.sort(key=lambda x: x[0])
        self.sample_ids = [sid for sid, _ in indexed_files]
        self.data_files = [p for _, p in indexed_files]

        if len(self.data_files) == 0:
            raise ValueError(f"No data*.mat files found in {self.eval_path}")

        print(f"Loaded {len(self.data_files)} .mat files from {self.eval_path}")

    def __len__(self) -> int:
        return len(self.data_files)

    def _load_ground_truth(self, sample_id: int) -> Optional[torch.Tensor]:
        gt_file = self.gt_path / f'{sample_id}_true.mat'
        if not gt_file.exists():
            return None
        gt_data = sio.loadmat(str(gt_file))
        key_candidates = ['truth', 'groundtruth', 'gt', 'reconstruction']
        gt_array = None
        for key in key_candidates:
            if key in gt_data:
                gt_array = np.asarray(gt_data[key], dtype=np.float32)
                break
        if gt_array is None:
            keys = [k for k in gt_data.keys() if not k.startswith('__')]
            if len(keys) == 0:
                return None
            gt_array = np.asarray(gt_data[keys[0]], dtype=np.float32)
        gt_array = np.squeeze(gt_array)
        gt_tensor = torch.from_numpy(gt_array)
        if gt_tensor.dim() == 2:
            gt_tensor = gt_tensor.unsqueeze(0)
        return gt_tensor

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        file_path = self.data_files[idx]
        sample_id = self.sample_ids[idx]
        data = sio.loadmat(str(file_path))
        if 'Uel' not in data:
            raise ValueError(f"'Uel' not found in {file_path}")
        uel = np.asarray(data['Uel'], dtype=np.float32).reshape(-1)
        delta_u = uel - self.uel_ref

        expected_len = 31 * 76
        if delta_u.size != expected_len:
            raise ValueError(
                f"Unexpected measurement size in {file_path}: {delta_u.size}, expected {expected_len}"
            )

        measurements = torch.from_numpy(delta_u).view(76, 31).transpose(0, 1).contiguous().unsqueeze(0)
        if self.mean is not None and self.std is not None:
            measurements = (measurements - self.mean) / self.std
        conductivity = self._load_ground_truth(sample_id)
        return measurements, conductivity


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
        self.measurement_format = config.get(
            'measurement_format',
            'eim16' if self.use_eim else 'raw16x13'
        )

        self.train_dataset: Optional[EITDataset] = None
        self.val_dataset: Optional[EITDataset] = None
        self.test_dataset: Optional[EITDataset] = None
        self.test2017_dataset: Optional[EITDataset] = None
        self.test2023_dataset: Optional[EITDataset] = None
        self.ktc_full_dataset: Optional[KTCChallengeDataset] = None
        self.ktc_eval_dataset: Optional[KTCChallengeDataset] = None
        self.matrix32_mean: Optional[torch.Tensor] = None
        self.matrix32_std: Optional[torch.Tensor] = None

    def _ensure_matrix32_stats(self):
        if self.measurement_format != 'matrix32':
            return
        mean_path = Path(self.data_dir) / 'mean_matrix32.pth'
        std_path = Path(self.data_dir) / 'std_matrix32.pth'

        if mean_path.exists() and std_path.exists():
            self.matrix32_mean = torch.load(mean_path)
            self.matrix32_std = torch.load(std_path)
            return

        mean_t, std_t = _compute_matrix32_stats(Path(self.data_dir))
        mean_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(mean_t, mean_path)
        torch.save(std_t, std_path)
        print(f"Saved matrix32 normalization stats to: {mean_path} and {std_path}")
        self.matrix32_mean = mean_t
        self.matrix32_std = std_t

    def setup(self, stage: Optional[str] = None):
        """设置数据集"""
        if stage in ('fit', 'train', None):
            self._ensure_matrix32_stats()
            self.train_dataset = EITDataset(
                self.data_dir, 'train',
                use_eim=self.use_eim,
                measurement_format=self.measurement_format
            )
            self.val_dataset = EITDataset(
                self.data_dir, 'valid',
                use_eim=self.use_eim,
                measurement_format=self.measurement_format
            )

        if stage in ('test', None):
            if self.measurement_format == 'matrix32':
                self._ensure_matrix32_stats()
            # 尝试加载测试数据集（如果存在）
            try:
                self.test_dataset = EITDataset(
                    self.data_dir,
                    'test',
                    use_eim=self.use_eim,
                    measurement_format=self.measurement_format
                )
            except ValueError:
                print("Warning: No test dataset found")

            try:
                self.test2017_dataset = EITDataset(
                    self.data_dir,
                    'test2017',
                    use_eim=self.use_eim,
                    measurement_format=self.measurement_format
                )
            except ValueError:
                print("Warning: No test2017 dataset found")

            try:
                self.test2023_dataset = EITDataset(
                    self.data_dir,
                    'test2023',
                    use_eim=self.use_eim,
                    measurement_format=self.measurement_format
                )
            except ValueError:
                print("Warning: No test2023 dataset found")

    def train_dataloader(self) -> DataLoader:
        """训练数据加载器"""
        if self.train_dataset is None:
            raise ValueError("Train dataset not set up. Call setup('fit') first.")

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        if self.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2
            )
        return DataLoader(self.train_dataset, **loader_kwargs)

    def val_dataloader(self) -> DataLoader:
        """验证数据加载器"""
        if self.val_dataset is None:
            raise ValueError("Validation dataset not set up. Call setup('fit') first.")

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        if self.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2
            )
        return DataLoader(self.val_dataset, **loader_kwargs)

    def test_dataloader(self) -> Optional[DataLoader]:
        """测试数据加载器"""
        if self.test_dataset is None:
            return None

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        if self.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2
            )
        return DataLoader(self.test_dataset, **loader_kwargs)

    def test2017_dataloader(self) -> Optional[DataLoader]:
        """2017年真实数据加载器"""
        if self.test2017_dataset is None:
            return None

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        if self.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2
            )
        return DataLoader(self.test2017_dataset, **loader_kwargs)

    def test2023_dataloader(self) -> Optional[DataLoader]:
        """2023年真实数据加载器"""
        if self.test2023_dataset is None:
            return None

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
        if self.num_workers > 0:
            loader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=2
            )
        return DataLoader(self.test2023_dataset, **loader_kwargs)

    def ktc_full_dataloader(self, level: int = 1) -> DataLoader:
        """KTC2023 官方 full 数据加载器"""
        self.ktc_full_dataset = KTCChallengeDataset(
            source='full',
            level=level,
            measurement_format=self.measurement_format,
            mean=self.matrix32_mean,
            std=self.matrix32_std
        )
        return DataLoader(
            self.ktc_full_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def ktc_eval_dataloader(self, level: int = 1) -> DataLoader:
        """KTC2023 官方 eval 数据加载器"""
        self.ktc_eval_dataset = KTCChallengeDataset(
            source='eval',
            level=level,
            measurement_format=self.measurement_format,
            mean=self.matrix32_mean,
            std=self.matrix32_std
        )
        return DataLoader(
            self.ktc_eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
