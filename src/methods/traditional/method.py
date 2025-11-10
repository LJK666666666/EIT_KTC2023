"""
传统 Tikhonov 正则化方法实现
使用 KTC 官方代码进行重建
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import sys
import os
from pathlib import Path

from ...core.base import BaseReconstructionMethod

# 添加 ktc_methods 到路径
ktc_path = Path(__file__).parent.parent.parent / 'ktc_methods'
if str(ktc_path) not in sys.path:
    sys.path.insert(0, str(ktc_path))

import KTCFwd
import KTCMeshing
import KTCRegularization
import KTCAux


class TikhonovReconstruction(BaseReconstructionMethod):
    """Tikhonov 正则化重建方法"""

    def __init__(self, config: Dict):
        super().__init__(config)

        # Tikhonov 参数
        method_config = config.get('method', {})
        self.alpha = method_config.get('alpha', 0.01)
        self.max_iter = method_config.get('max_iter', 100)
        self.tolerance = method_config.get('tolerance', 1e-6)

        # KTC 配置
        self.Nel = 32  # 电极数量
        self.z = (1e-6) * np.ones((self.Nel, 1))  # 接触阻抗

        # 数据路径配置 - 强制使用 EvaluationData_full
        self.data_root = Path(__file__).parent.parent.parent.parent / 'EvaluationData_full'

        # 默认使用 level1（可以通过参数覆盖）
        self.category_nbr = config.get('ktc', {}).get('category', 1)
        self.input_folder = self.data_root / 'evaluation_datasets' / f'level{self.category_nbr}'

        print(f"[Traditional] Tikhonov 正则化重建方法")
        print(f"  - alpha: {self.alpha}")
        print(f"  - 电极数量: {self.Nel}")
        print(f"  - 难度级别: {self.category_nbr}")
        print(f"  - 数据路径: {self.input_folder}")

        # 加载网格和参考数据
        self._load_mesh_and_reference()

    def _load_mesh_and_reference(self):
        """加载有限元网格和参考测量数据"""
        print("[Traditional] 加载有限元网格和参考数据...")

        # 加载参考数据
        ref_path = self.input_folder / 'ref.mat'
        if not ref_path.exists():
            raise FileNotFoundError(f"参考数据文件不存在: {ref_path}")

        mat_dict = sio.loadmat(str(ref_path))
        self.Injref = mat_dict["Injref"]  # [32, 76]
        self.Uelref = mat_dict["Uelref"]  # [2356, 1]
        self.Mpat = mat_dict["Mpat"]      # [32, 31]

        print(f"  - Injref: {self.Injref.shape}")
        print(f"  - Uelref: {self.Uelref.shape}, 范围: [{self.Uelref.min():.4f}, {self.Uelref.max():.4f}]")
        print(f"  - Mpat: {self.Mpat.shape}")

        # 设置测量包含掩码（根据难度级别移除电极数据）
        self.vincl = np.ones(((self.Nel - 1), 76), dtype=bool)
        rmind = np.arange(0, 2 * (self.category_nbr - 1), 1)

        for ii in range(0, 75):
            for jj in rmind:
                if self.Injref[jj, ii]:
                    self.vincl[:, ii] = 0
                self.vincl[jj, :] = 0

        # 加载网格文件
        mesh_path = ktc_path / 'Mesh_sparse.mat'
        if not mesh_path.exists():
            raise FileNotFoundError(f"网格文件不存在: {mesh_path}")

        mat_dict_mesh = sio.loadmat(str(mesh_path))

        # 构建一阶网格
        g = mat_dict_mesh['g']
        H = mat_dict_mesh['H']
        elfaces = mat_dict_mesh['elfaces'][0].tolist()

        ElementT = mat_dict_mesh['Element']['Topology'].tolist()
        for k in range(len(ElementT)):
            ElementT[k] = ElementT[k][0].flatten()
        ElementE = mat_dict_mesh['ElementE'].tolist()
        for k in range(len(ElementE)):
            if len(ElementE[k][0]) > 0:
                ElementE[k] = [ElementE[k][0][0][0], ElementE[k][0][0][1:len(ElementE[k][0][0])]]
            else:
                ElementE[k] = []

        NodeC = mat_dict_mesh['Node']['Coordinate']
        NodeE = mat_dict_mesh['Node']['ElementConnection']
        nodes = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC]
        for k in range(NodeC.shape[0]):
            nodes[k].ElementConnection = NodeE[k][0].flatten()
        elements = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT]
        for k in range(len(ElementT)):
            elements[k].Electrode = ElementE[k]

        # 构建二阶网格
        H2 = mat_dict_mesh['H2']
        g2 = mat_dict_mesh['g2']
        elfaces2 = mat_dict_mesh['elfaces2'][0].tolist()
        ElementT2 = mat_dict_mesh['Element2']['Topology'].tolist()
        for k in range(len(ElementT2)):
            ElementT2[k] = ElementT2[k][0].flatten()
        ElementE2 = mat_dict_mesh['Element2E'].tolist()
        for k in range(len(ElementE2)):
            if len(ElementE2[k][0]) > 0:
                ElementE2[k] = [ElementE2[k][0][0][0], ElementE2[k][0][0][1:len(ElementE2[k][0][0])]]
            else:
                ElementE2[k] = []

        NodeC2 = mat_dict_mesh['Node2']['Coordinate']
        NodeE2 = mat_dict_mesh['Node2']['ElementConnection']
        nodes2 = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC2]
        for k in range(NodeC2.shape[0]):
            nodes2[k].ElementConnection = NodeE2[k][0].flatten()
        elements2 = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT2]
        for k in range(len(ElementT2)):
            elements2[k].Electrode = ElementE2[k]

        self.Mesh = KTCMeshing.Mesh(H, g, elfaces, nodes, elements)
        self.Mesh2 = KTCMeshing.Mesh(H2, g2, elfaces2, nodes2, elements2)

        print(f"  - 一阶网格节点数: {len(self.Mesh.g)}")
        print(f"  - 二阶网格节点数: {len(self.Mesh2.g)}")

        # 设置正则化先验
        sigma0 = np.ones((len(self.Mesh.g), 1))
        corrlength = 1 * 0.115
        var_sigma = 0.05 ** 2
        mean_sigma = sigma0
        self.smprior = KTCRegularization.SMPrior(self.Mesh.g, corrlength, var_sigma, mean_sigma)

        # 设置正向求解器
        self.solver = KTCFwd.EITFEM(self.Mesh2, self.Injref, self.Mpat, self.vincl)

        # 设置噪声模型
        noise_std1 = 0.05
        noise_std2 = 0.01
        self.solver.SetInvGamma(noise_std1, noise_std2, self.Uelref)

        # 计算线性化点的雅可比矩阵（只需计算一次）
        self.sigma0 = sigma0
        self.vincl_flat = self.vincl.T.flatten()

        print(f"  - 正则化相关长度: {corrlength}")
        print("[Traditional] 网格和参考数据加载完成")

    def _build_model(self) -> Optional[nn.Module]:
        """传统方法不需要神经网络模型"""
        return None

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        传统方法不需要训练

        Returns:
            空损失字典
        """
        return {'loss': 0.0}

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        传统方法不需要验证

        Returns:
            空损失字典
        """
        return {'loss': 0.0}

    def inference(self, measurements: torch.Tensor) -> torch.Tensor:
        """
        使用 Tikhonov 正则化进行重建

        注意：此方法会忽略输入的 measurements 参数，
        直接从 EvaluationData_full 加载 .mat 数据进行重建

        Args:
            measurements: 测量数据（此参数被忽略）

        Returns:
            重建的电导率图像
        """
        print(f"\n[Traditional] 开始 Tikhonov 重建...")
        print(f"[Traditional] 注意：忽略输入的 measurements，直接从 {self.input_folder} 加载 .mat 数据")

        # 获取所有数据文件
        data_files = sorted(self.input_folder.glob('data*.mat'))

        if len(data_files) == 0:
            raise FileNotFoundError(f"未找到数据文件: {self.input_folder}/data*.mat")

        print(f"[Traditional] 找到 {len(data_files)} 个数据文件")

        # 处理每个数据文件
        reconstructions = []

        for idx, data_file in enumerate(data_files):
            print(f"\n[Traditional] 处理文件 {idx+1}/{len(data_files)}: {data_file.name}")

            # 加载数据
            mat_dict = sio.loadmat(str(data_file))
            Uel = mat_dict["Uel"]

            # 重建单个样本
            reco = self._reconstruct_single(Uel)
            reconstructions.append(reco)

        # 转换为 torch tensor
        reconstructions = np.stack(reconstructions, axis=0)  # [N, H, W]
        reconstructions = reconstructions[:, np.newaxis, :, :]  # [N, 1, H, W]

        return torch.from_numpy(reconstructions).float().to(self.device)

    def _reconstruct_single(self, Uel: np.ndarray) -> np.ndarray:
        """
        重建单个样本

        Args:
            Uel: 电压测量 [2356, 1]

        Returns:
            重建的电导率图像 [128, 128]
        """
        # 计算电压差
        deltaU = Uel - self.Uelref

        # 计算雅可比矩阵
        J = self.solver.Jacobian(self.sigma0, self.z)

        # 求解正则化问题
        mask = np.array(self.vincl_flat, bool)
        deltareco = np.linalg.solve(
            J.T @ self.solver.InvGamma_n[np.ix_(mask, mask)] @ J + self.smprior.L.T @ self.smprior.L,
            J.T @ self.solver.InvGamma_n[np.ix_(mask, mask)] @ deltaU[self.vincl_flat]
        )

        # 插值到像素网格
        deltareco_pixgrid = KTCAux.interpolateRecoToPixGrid(deltareco, self.Mesh)

        # 调整大小到 256x256（与 ground truth 保持一致）
        from PIL import Image
        deltareco_pixgrid = np.flipud(deltareco_pixgrid)
        deltareco_pixgrid = np.array(Image.fromarray(deltareco_pixgrid).resize((256, 256)))

        return deltareco_pixgrid


def create_traditional_method(config: Dict) -> TikhonovReconstruction:
    """
    创建传统重建方法

    Args:
        config: 配置字典

    Returns:
        传统重建方法实例
    """
    return TikhonovReconstruction(config)
