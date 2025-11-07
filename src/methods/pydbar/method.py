"""
PyDbar 传统重建方法包装器
基于 D-bar 方法的 EIT 重建

这是一个传统的数学方法，不涉及深度学习
"""
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple, Optional

# 添加 pydbar 到路径
pydbar_path = Path(__file__).parent.parent.parent.parent / 'programs' / 'pydbar'
if str(pydbar_path) not in sys.path:
    sys.path.insert(0, str(pydbar_path))

# 导入 pydbar 模块
try:
    from py_dbar import k_grid, scattering, dBar, read_data
    PYDBAR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pydbar modules not available: {e}")
    PYDBAR_AVAILABLE = False

from ...core.base import BaseReconstructionMethod


class PyDbarReconstruction(BaseReconstructionMethod):
    """
    PyDbar 传统重建方���（基于 D-bar 方程）

    这是一个传统的数学方法，不需要训练
    """

    def __init__(self, config: Dict):
        """
        初始化 PyDbar 重建方法

        Args:
            config: 配置字典
        """
        if not PYDBAR_AVAILABLE:
            raise ImportError("pydbar modules are not available. Please install or check the path.")

        # PyDbar 配置
        self.k_radius = config.get('pydbar', {}).get('k_radius', 4)
        self.k_power = config.get('pydbar', {}).get('k_power', 6)
        self.z_power = config.get('pydbar', {}).get('z_power', 6)
        self.scattering_type = config.get('pydbar', {}).get('scattering_type', 'exp')

        # 电极配置
        self.num_electrodes = config.get('pydbar', {}).get('num_electrodes', 16)
        self.radius = config.get('pydbar', {}).get('radius', 1.0)
        self.electrode_area = config.get('pydbar', {}).get('electrode_area', 1.0)

        # 参考测量（均匀电导率）
        self.reference_current = None
        self.reference_voltage = None
        self.num_patterns = 16  # 激励模式数量（16个电极，16个激励模式）
        self.auto_generate_reference = config.get('pydbar', {}).get('auto_generate_reference', True)

        super().__init__(config)

        print(f"[PyDbar] D-bar 传统重建方法")
        print(f"  - k_radius: {self.k_radius}, k_power: {self.k_power}")
        print(f"  - z_power: {self.z_power}")
        print(f"  - scattering_type: {self.scattering_type}")

        # 自动生成参考测量（均匀电导率）
        if self.auto_generate_reference:
            self._generate_default_reference()
            print(f"  - 已自动生成均匀电导率参考测量")

    def _build_model(self) -> Optional[torch.nn.Module]:
        """
        PyDbar 是传统方法，不需要神经网络模型
        """
        return None

    def _generate_default_reference(self):
        """
        生成默认的参考测量（均匀电导率）

        EIT测量原理：
        - 16个电极，16个激励模式
        - 每次在相邻两电极间施加电流 (ei, ei+1)
        - 测量其他电极对之间的差分电压
        - 每个激励模式有13个有效电压测量（排除激励电极对和参考）
        - 数据格式: [16 激励模式, 13 电压测量]
        """
        L = self.num_electrodes
        num_patterns = L  # 16个激励模式

        # 生成电流模式矩阵 [L, num_patterns]
        # 第i个激励模式：在电极i和电极(i+1)%L之间施加电流
        current = np.zeros((L, num_patterns))
        for i in range(num_patterns):
            current[i, i] = 1.0           # 电极 i 流入电流
            current[(i+1) % L, i] = -1.0  # 电极 i+1 流出电流

        # 生成电压模式矩阵 [L, num_patterns]
        # 对于均匀电导率，使用解析解
        voltage_full = np.zeros((L, num_patterns))
        for pattern_idx in range(num_patterns):
            # 激励在电极 pattern_idx 和 (pattern_idx+1)%L 之间
            inject_pos = pattern_idx
            inject_neg = (pattern_idx + 1) % L

            # 对于圆形均匀导体，电势分布可以用调和函数近似
            for elec_idx in range(L):
                # 计算电极相对位置的角度
                angle_elec = 2 * np.pi * elec_idx / L
                angle_inject_pos = 2 * np.pi * inject_pos / L
                angle_inject_neg = 2 * np.pi * inject_neg / L

                # 使用点电流源的格林函数近似
                # V ∝ log|r - r_source+| - log|r - r_source-|
                dist_to_pos = 2 * np.sin((angle_elec - angle_inject_pos) / 2)
                dist_to_neg = 2 * np.sin((angle_elec - angle_inject_neg) / 2)

                # 避免除零
                eps = 1e-10
                voltage_full[elec_idx, pattern_idx] = (
                    np.log(np.abs(dist_to_pos) + eps) -
                    np.log(np.abs(dist_to_neg) + eps)
                )

        # 从完整的16×16电压矩阵中提取13列
        # 这是实际数据的格式（排除了某些测量）
        # 我们需要选择合适的13列
        voltage_selected = np.zeros((L, 13))
        for pattern_idx in range(num_patterns):
            # 对于每个激励模式，选择13个测量电极
            # 排除激励电极对 (pattern_idx, (pattern_idx+1)%L)
            # 以及参考电极 ((pattern_idx+2)%L)

            selected_indices = []
            for elec_idx in range(L):
                # 排除激励电极和它们相邻的电极
                if elec_idx not in [pattern_idx, (pattern_idx + 1) % L, (pattern_idx + 2) % L]:
                    selected_indices.append(elec_idx)

            # 应该正好有13个（16 - 3 = 13）
            assert len(selected_indices) == 13

            # 填充到参考电压矩阵
            for col_idx, elec_idx in enumerate(selected_indices):
                voltage_selected[pattern_idx, col_idx] = voltage_full[elec_idx, pattern_idx]

        # 归一化并添加小噪声确保矩阵满秩
        voltage_selected = voltage_selected / (np.max(np.abs(voltage_selected)) + 1e-10)
        voltage_selected += np.random.randn(L, 13) * 1e-6

        self.reference_current = current
        self.reference_voltage = voltage_selected
        self.num_patterns = num_patterns

        print(f"[PyDbar] 参考测量形状: 电流{current.shape}, 电压{voltage_selected.shape}")

    def set_reference(self, current: np.ndarray, voltage: np.ndarray):
        """
        设置参考测量（均匀电导率的测量）

        Args:
            current: 参考电流模式 [L, L-1]
            voltage: 参考电压测量 [L, L-1]
        """
        self.reference_current = current
        self.reference_voltage = voltage

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        PyDbar 是传统方法，不需要训练

        返回零损失以兼容训练框架
        """
        return {'loss': 0.0}

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        PyDbar 是传统方法，不需要验证

        返回零损失以兼容训练框架
        """
        return {'loss': 0.0}

    def inference(self, measurements: torch.Tensor) -> torch.Tensor:
        """
        使用 D-bar 方法进行重建

        Args:
            measurements: 测量数据 [batch, 1, 16, 13] - 原始电压测量

        Returns:
            重建的电导率图像 [batch, 1, H, W]
        """
        # 检查参考测量是否设置
        if self.reference_current is None or self.reference_voltage is None:
            print("[PyDbar] 参考测量未设置，正在自动生成...")
            self._generate_default_reference()

        measurements_np = measurements.cpu().numpy()
        batch_size = measurements_np.shape[0]

        # 验证输入格式
        if measurements_np.shape[-2:] != (16, 13):
            raise ValueError(
                f"PyDbar 期望输入格式为 [batch, 1, 16, 13] (原始电压数据)，"
                f"但得到了 {measurements_np.shape}。"
                f"请确保配置文件中设置了 use_eim: false"
            )

        print(f"[PyDbar] 输入数据形状: {measurements_np.shape}")

        # 输出尺寸（2^z_power x 2^z_power）
        output_size = 2 ** self.z_power
        reconstructions = np.zeros((batch_size, 1, output_size, output_size))

        L = self.num_electrodes

        # 处理每个样本
        for i in range(batch_size):
            # 提取电压数据 [16, 13]
            voltage = measurements_np[i, 0, :, :]  # [16, 13]

            # 生成对应的电流激励模式 [16, 16]
            # 但实际使用时需要对应到13列的电压测量
            current_full = self._generate_current_patterns(16)

            # 为了匹配pydbar的格式，我们需要将13列的电压数据
            # 对应到完整的16个激励模式
            # 这里我们保持电压数据为[16, 13]格式

            # 重建单个样本
            reconstructions[i, 0, :, :] = self._reconstruct_single(current_full, voltage)

        # 转换回 PyTorch tensor
        return torch.from_numpy(reconstructions).float().to(self.device)

    def _generate_current_patterns(self, num_patterns: int) -> np.ndarray:
        """
        生成相邻电极激励模式

        Args:
            num_patterns: 激励模式数量

        Returns:
            current: [L, num_patterns] 电流模式矩阵
        """
        L = self.num_electrodes
        current = np.zeros((L, num_patterns))

        for i in range(num_patterns):
            current[i, i] = 1.0           # 电极 i 流入电流
            current[(i+1) % L, i] = -1.0  # 电极 i+1 流出电流

        return current

    def _reconstruct_single(self, current: np.ndarray, voltage: np.ndarray) -> np.ndarray:
        """
        对单个样本进行重建

        Args:
            current: 电流数据 [L, L-1]
            voltage: 电压数据 [L, L-1]

        Returns:
            重建的电导率图像 [H, W]
        """
        L = self.num_electrodes

        # 创建 read_data 对象
        Now = read_data.read_data(
            Current=current,
            Voltage=voltage,
            r=self.radius,
            AE=self.electrode_area,
            L=L
        )

        Ref = read_data.read_data(
            Current=self.reference_current,
            Voltage=self.reference_voltage,
            r=self.radius,
            AE=self.electrode_area,
            L=L
        )

        # 创建 k-grid
        Kp = k_grid.k_grid(self.k_radius, self.k_power)

        # 计算散射变换
        tK = scattering.scattering(self.scattering_type, Kp, Now, Ref)

        # 创建 D-bar 求解器
        solver = dBar.dBar(R_z=1.0, m_z=self.z_power)

        # 求解
        solver.solve(Kp, tK)

        # 获取重建结果
        return solver.sigma


def create_pydbar_method(config: Dict) -> PyDbarReconstruction:
    """
    创建 PyDbar 重建方法

    Args:
        config: 配置字典

    Returns:
        PyDbar 重建方法实例
    """
    return PyDbarReconstruction(config)
