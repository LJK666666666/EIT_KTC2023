"""
D-bar 方法实现
基于 pyDbar 库的 EIT 图像重建

D-bar 方法是一种直接重建算法，区别于迭代优化方法。
核心思想是将 EIT 逆问题转化为求解频谱空间中的 D-bar 方程。

算法流程：
1. 从边界测量构建 Dirichlet-to-Neumann (DN) 映射
2. 计算散射变换 t(k)
3. 求解 D-bar 方程得到 μ(z, k)
4. 从 μ(z, 0) 计算电导率 σ(z) = [μ(z, 0)]²
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import math
import sys
from pathlib import Path
from tqdm import tqdm

from ...core.base import BaseReconstructionMethod

# 添加 pyDbar 到路径
pydbar_path = Path(__file__).parent.parent.parent.parent / 'programs' / 'pydbar'
if str(pydbar_path) not in sys.path:
    sys.path.insert(0, str(pydbar_path))


class EIMConverter:
    """
    EIM (Electrical Impedance Map) 数据格式转换器

    将 208 维 EIM 向量转换为 D-bar 算法所需的电流/电压矩阵格式

    EIM 格式说明（来自论文）：
    - 16 个电极，相邻激励模式
    - 每个激励模式测量 13 个差分电压（排除 2 个激励电极和 1 个参考电极）
    - 总共 16 × 13 = 208 个测量值
    - 排列成 16×16 矩阵，其中对角线附近为 0（激励电极位置）

    D-bar 格式说明：
    - Current: (L-1, L) 矩阵，每行是一个归一化的电流模式
    - Voltage: (L-1, L) 矩阵，每行是对应电流模式下测量的电压
    """

    def __init__(self, num_electrodes: int = 16):
        """
        Args:
            num_electrodes: 电极数量，默认 16
        """
        self.L = num_electrodes

        # 预计算相邻电流模式矩阵
        self._build_adjacent_current_patterns()

    def _build_adjacent_current_patterns(self):
        """构建相邻电流模式矩阵"""
        # 相邻电流模式：电流从电极 i 流入，从电极 i+1 流出
        self.current_patterns = np.zeros((self.L, self.L))
        for i in range(self.L):
            self.current_patterns[i, i] = 1.0
            self.current_patterns[i, (i + 1) % self.L] = -1.0

    def eim_vector_to_matrix(self, eim_vector: np.ndarray) -> np.ndarray:
        """
        将 208 维 EIM 向量转换为 16×16 EIM 矩阵

        Args:
            eim_vector: shape (208,) 或 (208, 1) 的 EIM 向量

        Returns:
            shape (16, 16) 的 EIM 矩阵
        """
        eim_vector = np.squeeze(eim_vector)
        assert len(eim_vector) == 208, f"Expected 208 elements, got {len(eim_vector)}"

        # EIM 矩阵布局：
        # 行 i = 激励电极对 (i, i+1)
        # 列 j = 测量电极对 (j, j+1)
        # 但排除激励电极位置的测量

        eim_matrix = np.zeros((self.L, self.L))
        idx = 0

        for i in range(self.L):  # 激励模式
            for j in range(self.L):  # 测量位置
                # 跳过激励电极位置（相邻的 3 个位置）
                # 当激励 (i, i+1) 时，不测量 j = i-1, i, i+1 位置
                skip_positions = [(i - 1) % self.L, i, (i + 1) % self.L]
                if j not in skip_positions:
                    eim_matrix[i, j] = eim_vector[idx]
                    idx += 1

        return eim_matrix

    def eim_matrix_to_vector(self, eim_matrix: np.ndarray) -> np.ndarray:
        """
        将 16×16 EIM 矩阵转换为 208 维 EIM 向量

        这是 eim_vector_to_matrix 的逆操作

        Args:
            eim_matrix: shape (16, 16) 的 EIM 矩阵

        Returns:
            shape (208,) 的 EIM 向量
        """
        eim_vector = []

        for i in range(self.L):  # 激励模式
            for j in range(self.L):  # 测量位置
                # 跳过激励电极位置（相邻的 3 个位置）
                skip_positions = [(i - 1) % self.L, i, (i + 1) % self.L]
                if j not in skip_positions:
                    eim_vector.append(eim_matrix[i, j])

        return np.array(eim_vector)

    def eim_to_dbar_format(self, eim_vector: np.ndarray,
                           reference_eim: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        将 EIM 数据转换为 D-bar 所需的电流/电压格式。
        电压采用“已知差分约束+均值为 0”的最小二乘恢复，以确保与电流模式同基。

        Returns:
            current_matrix: shape (L-1, L) 的电流模式矩阵（相邻注流）
            voltage_matrix: shape (L-1, L) 的电压矩阵（均值为 0，按电极顺序）
        """
        eim_vector = np.squeeze(eim_vector)
        assert len(eim_vector) == 208, f"Expected 208 elements, got {len(eim_vector)}"

        # 相邻电流模式，取前 L-1 个（与 pyDbar 假设一致）
        current_matrix = self.current_patterns[: self.L - 1, :].copy()
        voltage_matrix = np.zeros((self.L - 1, self.L))

        # 将参考向量转换为矩阵便于取对应差分（如果未给出则全零）
        if reference_eim is None:
            ref_vector = np.zeros_like(eim_vector)
        else:
            ref_vector = np.squeeze(reference_eim)
            assert len(ref_vector) == 208, f"Expected reference 208 elements, got {len(ref_vector)}"

        ref_matrix = self.eim_vector_to_matrix(ref_vector)

        # 将输入向量重排为矩阵，保留测量顺序
        eim_matrix = self.eim_vector_to_matrix(eim_vector)

        # 最小二乘恢复每个模式的电位（加约束电位和为 0）
        for i in range(self.L - 1):
            skip_positions = {(i - 1) % self.L, i, (i + 1) % self.L}
            A_rows = []
            b_vals = []

            for j in range(self.L):
                if j in skip_positions:
                    continue
                row = np.zeros(self.L)
                row[j] = 1.0
                row[(j + 1) % self.L] = -1.0
                A_rows.append(row)
                # 使用差分（当前 - 参考）保证 tdEIT 设置
                b_vals.append(eim_matrix[i, j] - ref_matrix[i, j])

            # 均值为 0 约束
            A_rows.append(np.ones(self.L))
            b_vals.append(0.0)

            A = np.vstack(A_rows)
            b = np.array(b_vals)

            voltages, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            voltage_matrix[i, :] = voltages

        return current_matrix, voltage_matrix

    def create_trigonometric_current_patterns(self) -> np.ndarray:
        """
        创建三角函数电流模式（D-bar 算法的理想输入）

        Returns:
            shape (L-1, L) 的三角电流模式矩阵
        """
        L = self.L
        theta = np.linspace(0, 2 * np.pi, L, endpoint=False)

        current_matrix = np.zeros((L - 1, L))

        # 前 L/2 个模式：cos(n*theta)
        for n in range(1, L // 2 + 1):
            current_matrix[n - 1, :] = np.cos(n * theta)

        # 后 L/2-1 个模式：sin(n*theta)
        for n in range(1, L // 2):
            current_matrix[L // 2 + n - 1, :] = np.sin(n * theta)

        return current_matrix


class DbarCore:
    """
    D-bar 算法核心实现

    基于 pyDbar 库，但针对 EIM 数据格式进行了适配
    """

    def __init__(self,
                 num_electrodes: int = 16,
                 R: float = 4.0,
                 m: int = 6,
                 scattering_type: str = 'exp',
                 radius: float = 1.0,
                 electrode_area: float = 0.1):
        """
        Args:
            num_electrodes: 电极数量
            R: k 空间截断半径（正则化参数）
            m: k 空间网格分辨率，N = 2^m
            scattering_type: 散射变换类型，'exp' 或 'partial'
            radius: 成像区域半径
            electrode_area: 电极面积
        """
        self.L = num_electrodes
        self.R = R
        self.m = m
        self.scattering_type = scattering_type
        self.radius = radius
        self.electrode_area = electrode_area

        # 初始化 k 空间网格
        self._init_k_grid()

    def _init_k_grid(self):
        """初始化 k 空间网格"""
        self.N = 2 ** self.m
        self.s = 2.3 * self.R
        self.h = 2 * self.s / self.N

        # 创建复数 k 网格
        self.k = np.zeros((self.N, self.N), dtype=complex)
        self.pos_x = []
        self.pos_y = []
        self.index = -1

        for j in range(self.N):
            for jj in range(self.N):
                self.k[j, jj] = complex(-self.s + j * self.h, -self.s + jj * self.h)

                if abs(self.k[j, jj]) < self.R:
                    self.pos_x.append(j)
                    self.pos_y.append(jj)

                if abs(self.k[j, jj]) < 1e-7:
                    self.index = len(self.pos_x) - 1

        # 计算基本解的 FFT
        self._compute_fundamental_solution()

    def _compute_fundamental_solution(self):
        """计算 D-bar 算子基本解的 FFT"""
        from scipy.fft import fft2, fftshift

        eps = self.s / 10
        RR = (self.s - eps) / 2

        G = np.zeros((self.N, self.N), dtype=complex)

        for j in range(self.N):
            for jj in range(self.N):
                abs_k = abs(self.k[j, jj])

                if abs_k < self.s and abs_k > 1e-7:
                    G[j, jj] = 1 / (self.k[j, jj] * math.pi)

                    # 平滑截断
                    if abs_k >= 2 * RR:
                        G[j, jj] = G[j, jj] * (1 - (abs_k - 2 * RR) / eps)

        self.FG = fft2(fftshift(G))

    def compute_dn_map(self, current: np.ndarray, voltage: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算 Dirichlet-to-Neumann 映射

        Args:
            current: shape (L-1, L) 电流模式矩阵
            voltage: shape (L-1, L) 电压测量矩阵

        Returns:
            DN 映射矩阵 (L-1, L-1) 以及正交归一后的 current/voltage（行形式）
        """
        L = current.shape[1]

        # 转置为列向量形式
        current = current.T  # (L, L-1)
        voltage = voltage.T  # (L, L-1)

        # Gram-Schmidt 正交化
        for j in range(L - 1):
            # 归一化第一个
            if j == 0:
                norm = np.linalg.norm(current[:, 0])
                voltage[:, 0] = voltage[:, 0] / norm
                current[:, 0] = current[:, 0] / norm
            else:
                # 正交化
                for jj in range(j):
                    coef = np.inner(current[:, j], current[:, jj])
                    current[:, j] = current[:, j] - coef * current[:, jj]
                    voltage[:, j] = voltage[:, j] - coef * voltage[:, jj]

                norm = np.linalg.norm(current[:, j])
                voltage[:, j] = voltage[:, j] / norm
                current[:, j] = current[:, j] / norm

        # 计算 Neumann-to-Dirichlet 映射
        R_gamma = np.zeros((L - 1, L - 1))
        for n in range(L - 1):
            for m in range(L - 1):
                R_gamma[n, m] = np.inner(current[:, n], voltage[:, m])

        # DN 映射 = (AE/r) * R^{-1}
        DN_map = (self.electrode_area / self.radius) * np.linalg.inv(R_gamma)

        # 返回行向量形式的正交化电流/电压
        current_rows = current.T
        voltage_rows = voltage.T

        return DN_map, current_rows, voltage_rows

    def compute_scattering_transform(self,
                                     dn_map_now: np.ndarray,
                                     dn_map_ref: np.ndarray,
                                     current_ortho: np.ndarray) -> np.ndarray:
        """
        计算散射变换

        Args:
            dn_map_now: 当前 DN 映射 (L-1, L-1)
            dn_map_ref: 参考 DN 映射（均匀介质）(L-1, L-1)
            current_ortho: 正交化后的电流模式矩阵 (L-1, L) - 每行是一个电流模式

        Returns:
            shape (N, N) 的散射变换矩阵
        """
        tK = np.zeros((self.N, self.N), dtype=complex)

        dt = (2 * math.pi) / self.L
        zt = np.exp(1j * np.arange(0, 2 * math.pi, dt))  # 电极位置 (L,)

        dL = dn_map_now - dn_map_ref  # (L-1, L-1)

        # current 形状: (L-1, L)，转置后 (L, L-1) 用于 lstsq
        current_T = current_ortho.T  # (L, L-1)

        if self.scattering_type == 'exp':
            # 指数近似
            for j in range(self.N):
                for jj in range(self.N):
                    if abs(self.k[j, jj]) < self.R and abs(self.k[j, jj]) > 1e-7:
                        # Ez: (L,) - 在电极位置的指数函数值
                        Ez = np.exp(1j * self.k[j, jj] * zt)
                        conj_Ez = np.exp(1j * self.k[j, jj].conjugate() * np.conjugate(zt))

                        # 用最小二乘法将指数展开为电流模式的线性组合
                        # current_T @ ck ≈ Ez
                        # current_T: (L, L-1), Ez: (L,), ck: (L-1,)
                        ck, _, _, _ = np.linalg.lstsq(current_T, Ez, rcond=None)
                        dk, _, _, _ = np.linalg.lstsq(current_T, conj_Ez, rcond=None)

                        # 计算散射变换
                        for l in range(self.L - 1):
                            for ll in range(self.L - 1):
                                tK[j, jj] += ck[l] * dL[ll, l] * dk[ll]

                        tK[j, jj] /= (4 * math.pi * self.k[j, jj].conjugate())
        else:
            # partial 近似（更精确但更慢）
            G0 = np.zeros((self.L, self.L), dtype=complex)

            for l in range(self.L):
                for ll in range(self.L):
                    if l != ll:
                        G0[l, ll] = -(1 / (2 * math.pi)) * np.log(abs(zt[l] - zt[ll]))

            # Phi: (L-1, L-1), PhidL: (L, L-1)
            Phi = np.matmul(current_T.T, current_T)  # (L-1, L-1)
            PhidL = np.matmul(current_T, dL)         # (L, L-1)
            M = Phi + np.matmul(current_T.T, np.matmul(G0, PhidL))  # (L-1, L-1)

            for j in range(self.N):
                for jj in range(self.N):
                    if abs(self.k[j, jj]) < self.R and abs(self.k[j, jj]) > 1e-7:
                        Ez = np.exp(1j * self.k[j, jj] * zt)
                        rhs = np.matmul(current_T.T, Ez)
                        psi_b, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

                        for l in range(self.L):
                            c = np.exp(1j * (self.k[j, jj] * zt[l]).conjugate())
                            for ll in range(self.L - 1):
                                tK[j, jj] += c * PhidL[l, ll] * psi_b[ll]

                        tK[j, jj] /= (4 * math.pi * self.k[j, jj].conjugate())

        return tK

    def solve_dbar_equation(self, tK: np.ndarray, z_grid_size: int = 64) -> np.ndarray:
        """
        求解 D-bar 方程

        Args:
            tK: 散射变换矩阵
            z_grid_size: z 空间网格大小

        Returns:
            shape (z_grid_size, z_grid_size) 的电导率重建
        """
        from scipy.fft import fft2, ifft2
        import scipy.sparse.linalg as spla

        # 创建 z 空间网格
        m_z = int(np.log2(z_grid_size))
        N_z = 2 ** m_z
        h_z = 2.0 / N_z

        Z = np.zeros((N_z, N_z), dtype=complex)
        for l in range(N_z):
            for ll in range(N_z):
                Z[l, ll] = complex(-1 + l * h_z, -1 + ll * h_z)

        sigma = np.zeros((N_z, N_z))

        N_k = len(self.pos_x)

        # 定义 D-bar 算子
        def dbar_operator(mu, zz):
            RHS = np.zeros((self.N, self.N), dtype=complex)

            for l in range(N_k):
                px, py = self.pos_x[l], self.pos_y[l]
                exp_term = np.exp(-2j * (self.k[px, py] * zz).real)
                RHS[px, py] = exp_term * tK[px, py] * complex(mu[l], -mu[l + N_k])

            F_RHS = fft2(RHS)
            F_RHS = F_RHS * self.FG
            RHS = ifft2(F_RHS)

            result = mu.copy()
            for l in range(N_k):
                px, py = self.pos_x[l], self.pos_y[l]
                result[l] -= (self.h ** 2) * RHS[px, py].real
                result[l + N_k] -= (self.h ** 2) * RHS[px, py].imag

            return result

        # 初始解
        b = np.concatenate([np.ones(N_k), np.zeros(N_k)])
        mu = b.copy()

        # 对每个 z 点求解
        for j in tqdm(range(N_z), desc="Solving D-bar"):
            for jj in range(N_z):
                zz = Z[j, jj]

                def Op(mu_vec):
                    return dbar_operator(mu_vec, zz)

                A = spla.LinearOperator((2 * N_k, 2 * N_k), matvec=Op)

                # 使用 GMRES 求解
                mu, _ = spla.gmres(A, b, x0=mu, maxiter=5, atol=1e-6)

                if abs(zz) <= 1:
                    # σ(z) = [μ(z, 0)]²
                    sigma[j, jj] = mu[self.index] ** 2 - mu[self.index + N_k] ** 2

        return sigma


class DbarReconstruction(BaseReconstructionMethod):
    """
    D-bar 重建方法

    D-bar 是一种基于数学理论的直接重建算法，不需要训练。
    优点：理论严谨，有正则化保证
    缺点：计算量较大，对数据格式有要求
    """

    def __init__(self, config: Dict):
        super().__init__(config)

        # D-bar 参数
        method_config = config.get('method', {})
        self.num_electrodes = method_config.get('num_electrodes', 16)
        self.R = method_config.get('R', 4.0)  # k 空间截断半径
        self.m = method_config.get('m', 6)  # k 空间网格分辨率
        self.scattering_type = method_config.get('scattering_type', 'exp')
        self.output_size = method_config.get('output_size', 128)

        # 初始化转换器和核心算法
        self.converter = EIMConverter(self.num_electrodes)
        self.dbar_core = DbarCore(
            num_electrodes=self.num_electrodes,
            R=self.R,
            m=self.m,
            scattering_type=self.scattering_type
        )

        # 参考数据（均匀介质的 DN 映射）
        self.reference_dn_map = None
        self._compute_reference_dn_map()

        print(f"[D-bar] D-bar 重建方法初始化完成")
        print(f"  - 电极数量: {self.num_electrodes}")
        print(f"  - k 空间截断半径 R: {self.R}")
        print(f"  - k 空间分辨率: {2**self.m}x{2**self.m}")
        print(f"  - 散射变换类型: {self.scattering_type}")
        print(f"  - 输出尺寸: {self.output_size}x{self.output_size}")

    def _compute_reference_dn_map(self):
        """计算参考 DN 映射（均匀介质 σ=1）"""
        # 对于差分测量（tdEIT），参考 DN 映射应该为零
        # 因为我们计算的是 (DN_now - DN_ref)，
        # 而 EIM 数据本身就是差分电压，已经包含了与参考的差异
        L = self.num_electrodes

        # 使用零矩阵作为参考 DN 映射
        # 这等价于假设参考状态的 DN 映射已经在测量数据中被减去
        self.reference_dn_map = np.zeros((L - 1, L - 1))

    def _build_model(self) -> Optional[nn.Module]:
        """D-bar 方法不需要神经网络模型"""
        return None

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """D-bar 方法不需要训练"""
        return {'loss': 0.0}

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """D-bar 方法不需要验证"""
        return {'loss': 0.0}

    def inference(self, measurements: torch.Tensor) -> torch.Tensor:
        """
        使用 D-bar 方法进行重建

        Args:
            measurements: 测量数据，支持多种格式：
                - [batch_size, 1, 16, 16]: EIM 矩阵格式（来自 DataLoader）
                - [batch_size, 208, 1] 或 [batch_size, 208]: EIM 向量格式

        Returns:
            重建的电导率图像 [batch_size, 1, H, W]
        """
        print(f"\n[D-bar] 开始 D-bar 重建...")

        # 转换为 numpy
        if isinstance(measurements, torch.Tensor):
            measurements_np = measurements.cpu().numpy()
        else:
            measurements_np = measurements

        batch_size = measurements_np.shape[0]
        reconstructions = []

        for i in tqdm(range(batch_size), desc="D-bar reconstruction"):
            sample = measurements_np[i]

            # 根据输入形状判断数据格式
            if sample.ndim == 3 and sample.shape == (1, 16, 16):
                # EIM 矩阵格式 [1, 16, 16] -> 转换为向量
                eim_matrix = sample[0]  # [16, 16]
                eim_vector = self.converter.eim_matrix_to_vector(eim_matrix)
            elif sample.ndim == 2 and sample.shape == (16, 16):
                # EIM 矩阵格式 [16, 16] -> 转换为向量
                eim_matrix = sample
                eim_vector = self.converter.eim_matrix_to_vector(eim_matrix)
            elif sample.ndim == 2 and sample.shape[0] == 208:
                # [208, 1] 格式
                eim_vector = sample.squeeze()
            elif sample.ndim == 1 and len(sample) == 208:
                # [208] 格式
                eim_vector = sample
            else:
                raise ValueError(f"Unsupported input shape: {sample.shape}")

            # 转换数据格式
            current, voltage = self.converter.eim_to_dbar_format(eim_vector)

            # 计算 DN 映射
            dn_map, current_ortho, voltage_ortho = self.dbar_core.compute_dn_map(current, voltage)

            # 计算散射变换
            # 使用正交化后的 current
            tK = self.dbar_core.compute_scattering_transform(
                dn_map,
                self.reference_dn_map,
                current_ortho
            )

            # 求解 D-bar 方程
            sigma = self.dbar_core.solve_dbar_equation(tK, z_grid_size=64)

            # 归一化输出到合理范围
            # D-bar 输出的是电导率变化，需要归一化
            # 应用圆形掩码（只保留圆盘内的值）
            h = 2.0 / 64
            mask = np.zeros_like(sigma)
            for ii in range(64):
                for jj in range(64):
                    z = complex(-1 + ii * h, -1 + jj * h)
                    if abs(z) <= 1:
                        mask[ii, jj] = 1

            sigma = sigma * mask

            # 归一化到 [-1, 1] 范围
            sigma_max = np.max(np.abs(sigma))
            if sigma_max > 0:
                sigma = sigma / sigma_max

            # 调整到目标尺寸
            from PIL import Image
            sigma_resized = np.array(
                Image.fromarray(sigma.astype(np.float32)).resize(
                    (self.output_size, self.output_size)
                )
            )

            reconstructions.append(sigma_resized)

        # 转换为 tensor
        reconstructions = np.stack(reconstructions, axis=0)  # [B, H, W]
        reconstructions = reconstructions[:, np.newaxis, :, :]  # [B, 1, H, W]

        return torch.from_numpy(reconstructions).float().to(self.device)


def create_dbar_method(config: Dict) -> DbarReconstruction:
    """
    创建 D-bar 重建方法

    Args:
        config: 配置字典

    Returns:
        D-bar 重建方法实例
    """
    return DbarReconstruction(config)
