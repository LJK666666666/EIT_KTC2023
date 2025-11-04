import os
import sys
import numpy as np
import scipy as sp
from scipy.ndimage import rotate
from scipy.io import loadmat
import scipy.io as sio

import torch
from torch.utils.data import Dataset

import math
import random
from typing import List, Tuple
from PIL import Image, ImageDraw

# 添加ktc_methods到系统路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ktc_methods'))

import KTCFwd
import KTCMeshing
import KTCAux


def load_mesh(mesh_name="Mesh_dense.mat"):
    """
    加载预先生成的有限元网格（密集或稀疏网格）

    Args:
        mesh_name: 网格文件名，"Mesh_dense.mat" 或 "Mesh_sparse.mat"

    Returns:
        mesh: 一阶网格对象
        mesh2: 二阶网格对象
    """
    mesh_path = os.path.join(os.path.dirname(__file__), 'ktc_methods', mesh_name)
    mat_dict_mesh = sp.io.loadmat(mesh_path)

    g = mat_dict_mesh['g']
    H = mat_dict_mesh['H']
    elfaces = mat_dict_mesh['elfaces'][0].tolist()

    # 构建一阶网格的单元结构
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

    # 构建二阶网格数据
    H2 = mat_dict_mesh['H2']
    g2 = mat_dict_mesh['g2']
    elfaces2 = mat_dict_mesh['elfaces2'][0].tolist()
    ElementT2 = mat_dict_mesh['Element2']['Topology']
    ElementT2 = ElementT2.tolist()
    for k in range(len(ElementT2)):
        ElementT2[k] = ElementT2[k][0].flatten()
    ElementE2 = mat_dict_mesh['Element2E']
    ElementE2 = ElementE2.tolist()
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

    mesh = KTCMeshing.Mesh(H, g, elfaces, nodes, elements)
    mesh2 = KTCMeshing.Mesh(H2, g2, elfaces2, nodes2, elements2)

    return mesh, mesh2


class SimulatedEITDataset(Dataset):
    """
    使用KTC方法生成EIT仿真数据集
    包含合成的导电率分布和对应的测量数据

    Args:
        length: 数据集大小
        mesh_name: 使用的网格文件名
        noise_std1: 噪声标准差（测量值的百分比）
        noise_std2: 第二噪声分量的标准差（与最大测量值成比例）
        segments: 分割类别数量（2或3）
    """
    def __init__(self, length=1000, mesh_name="Mesh_dense.mat",
                 noise_std1=0.1, noise_std2=0, segments=3,
                 use_evaluation_pattern=True):
        self.length = length
        self.Nel = 32  # 电极数量
        self.noise_std1 = noise_std1
        self.noise_std2 = noise_std2
        self.segments = segments

        # 加载网格
        self.mesh, self.mesh2 = load_mesh(mesh_name=mesh_name)

        # 设置接触阻抗
        self.z = np.ones((self.Nel, 1))

        # 设置测量模式
        if use_evaluation_pattern:
            # 使用评估数据的测量模式
            self._load_evaluation_pattern()
        else:
            # 使用简单的相邻注入模式
            self.Inj, self.Mpat, self.vincl = KTCAux.setMeasurementPattern(self.Nel)

        # 初始化正向求解器
        self.solver = KTCFwd.EITFEM(self.mesh2, self.Inj, self.Mpat, self.vincl)

        # 像素网格参数
        pixwidth = 0.23 / 256
        pixcenter_x = np.linspace(-0.115 + pixwidth / 2, 0.115 - pixwidth / 2 + pixwidth, 256)
        pixcenter_y = pixcenter_x
        self.X, self.Y = np.meshgrid(pixcenter_x, pixcenter_y)

    def _load_evaluation_pattern(self):
        """加载评估数据的测量模式"""
        ref_path = 'EvaluationData_full/evaluation_datasets/level1/ref.mat'
        if not os.path.exists(ref_path):
            print(f"Warning: Cannot find {ref_path}, using default pattern")
            self.Inj, self.Mpat, self.vincl = KTCAux.setMeasurementPattern(self.Nel)
            return

        ref_data = sio.loadmat(ref_path)
        self.Inj = ref_data['Injref']
        self.Mpat = ref_data['Mpat']

        # 计算vincl（所有测量都包含）
        self.vincl = np.ones(self.Inj.shape[1] * (self.Nel - 1), dtype=bool)

        print(f"Loaded evaluation measurement pattern:")
        print(f"  Injection patterns: {self.Inj.shape[1]}")
        print(f"  Measurements per injection: {self.Nel - 1}")
        print(f"  Total measurements: {self.Inj.shape[1] * (self.Nel - 1)}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        生成一个数据样本

        Returns:
            phantom_pix: 像素网格上的导电率分布 (256, 256)
            measurements: EIT测量数据（电压差）
        """
        # 生成随机的导电率分布
        sigma_pix = self._create_phantoms()

        # 将像素导电率插值到网格节点
        sigma = self._interpolate_to_mesh(sigma_pix)

        # 仿真参考测量（均匀分布）
        sigma_ref = np.ones((len(self.mesh.g), 1))
        Uref = self.solver.SolveForward(sigma_ref, self.z)

        # 仿真带有异常的测量
        Umeas = self.solver.SolveForward(sigma, self.z)

        # 添加测量噪声
        self.solver.SetInvGamma(self.noise_std1, self.noise_std2, Uref)
        noise = self.solver.InvLn @ np.random.randn(Uref.shape[0], 1)
        Umeas_noisy = Umeas + noise

        # 计算电压差分
        deltaU = Umeas_noisy - Uref

        phantom_pix = torch.from_numpy(sigma_pix).float()
        measurements = torch.from_numpy(deltaU).float()[:, 0]

        return phantom_pix, measurements

    def _create_phantoms(self, min_inclusions=1, max_inclusions=4,
                        max_iter=80, distance_between=25, p=[0.7, 0.15, 0.15]):
        """
        创建随机的导电率分布（类别标签：0=背景, 1=低电导率, 2=高电导率）

        Args:
            min_inclusions: 最少包含物数量
            max_inclusions: 最多包含物数量
            max_iter: 最大迭代次数
            distance_between: 包含物之间的最小距离（像素）
            p: 形状概率分布 [多边形, 圆形, 矩形]

        Returns:
            sigma_pix: 256x256的导电率分布数组
        """
        rectangle_dict = {'min_width': 25, 'max_width': 50,
                         "min_height": 40, "max_height": 120}

        I = np.zeros((256, 256))
        im = Image.fromarray(np.uint8(I))
        draw = ImageDraw.Draw(im)

        num_forms = np.random.randint(min_inclusions, max_inclusions)
        circle_list = []

        iter_count = 0
        while len(circle_list) < num_forms:
            object_type = np.random.choice(["polygon", "circle", "rectangle"], p=p)

            if object_type == "rectangle":
                lower_x = 50 + np.random.randint(-24, 24)
                lower_y = 50 + np.random.randint(-24, 24)
                width = np.random.randint(rectangle_dict["min_width"],
                                         rectangle_dict["max_width"])
                height = np.random.randint(rectangle_dict["min_height"],
                                          rectangle_dict["max_height"])
                center_x = lower_x + width / 2
                center_y = lower_y + height / 2
                avg_radius = max(width / 2, height / 2)
            else:
                avg_radius = np.random.randint(25, 50)
                center_x = 128 + np.random.randint(-54, 54)
                center_y = 128 + np.random.randint(-54, 54)

            # 碰撞检测
            collide = False
            for x, y, r in circle_list:
                d = (center_x - x)**2 + (center_y - y)**2
                if d < (avg_radius + r + distance_between)**2:
                    collide = True
                    break

            if not collide:
                if object_type == "rectangle":
                    fill_val = 1 if np.random.rand() < 0.5 else 2
                    draw.rectangle([lower_x, lower_y,
                                  lower_x + width, lower_y + height],
                                 fill=fill_val)
                elif object_type == "circle":
                    fill_val = 1 if np.random.rand() < 0.5 else 2
                    draw.ellipse((center_x - avg_radius, center_y - avg_radius,
                                center_x + avg_radius, center_y + avg_radius),
                               fill=fill_val)
                elif object_type == "polygon":
                    num_vertices = np.random.randint(5, 9)
                    vertices = self._generate_polygon(
                        center=(center_x, center_y),
                        avg_radius=avg_radius,
                        irregularity=0.4,
                        spikiness=0.3,
                        num_vertices=num_vertices)
                    fill_val = 1 if np.random.rand() < 0.5 else 2
                    draw.polygon(vertices, fill=fill_val)

                circle_list.append((center_x, center_y, avg_radius))

            iter_count += 1
            if iter_count > max_iter:
                break

        sigma_pix = np.array(np.asarray(im))
        # 限制在圆形区域内
        sigma_pix[self.X**2 + self.Y**2 > 0.098**2] = 0.0
        # 随机旋转
        angle = np.random.randint(0, 180)
        sigma_pix = np.round(rotate(sigma_pix, angle, mode="constant",
                                   cval=0.0, reshape=False, order=0))

        return sigma_pix

    def _generate_polygon(self, center: Tuple[float, float], avg_radius: float,
                         irregularity: float, spikiness: float,
                         num_vertices: int) -> List[Tuple[float, float]]:
        """生成随机多边形"""
        irregularity *= 2 * math.pi / num_vertices
        spikiness *= avg_radius
        angle_steps = self._random_angle_steps(num_vertices, irregularity)

        points = []
        angle = random.uniform(0, 2 * math.pi)
        for i in range(num_vertices):
            radius = self._clip(random.gauss(avg_radius, spikiness),
                              0, 2 * avg_radius)
            point = (center[0] + radius * math.cos(angle),
                    center[1] + radius * math.sin(angle))
            points.append(point)
            angle += angle_steps[i]

        return points

    def _random_angle_steps(self, steps: int, irregularity: float) -> List[float]:
        """生成随机角度步长"""
        angles = []
        lower = (2 * math.pi / steps) - irregularity
        upper = (2 * math.pi / steps) + irregularity
        cumsum = 0
        for i in range(steps):
            angle = random.uniform(lower, upper)
            angles.append(angle)
            cumsum += angle

        cumsum /= (2 * math.pi)
        for i in range(steps):
            angles[i] /= cumsum
        return angles

    def _clip(self, value, lower, upper):
        """限制值在指定区间内"""
        return min(upper, max(value, lower))

    def _interpolate_to_mesh(self, sigma_pix):
        """
        将像素网格上的导电率插值到有限元网格节点

        Args:
            sigma_pix: 像素网格上的导电率分布 (256, 256)

        Returns:
            sigma: 网格节点上的导电率分布
        """
        # 将类别标签转换为导电率值
        # 0 -> 1.0 (背景)
        # 1 -> 0.5 (低电导率)
        # 2 -> 1.5 (高电导率)
        sigma_map = np.zeros_like(sigma_pix, dtype=np.float64)
        sigma_map[sigma_pix == 0] = 1.0
        sigma_map[sigma_pix == 1] = 0.5
        sigma_map[sigma_pix == 2] = 1.5

        # 像素中心坐标
        pixwidth = 0.23 / 256
        pixcenter_x = np.linspace(-0.115 + pixwidth / 2,
                                 0.115 - pixwidth / 2 + pixwidth, 256)
        pixcenter_y = pixcenter_x
        X, Y = np.meshgrid(pixcenter_x, pixcenter_y, indexing="ij")
        pixcenters = np.column_stack((X.ravel(), Y.ravel()))

        # 反向插值：从网格节点到像素的插值已经实现，这里需要从像素到网格
        # 使用最近邻插值
        from scipy.interpolate import NearestNDInterpolator
        interp = NearestNDInterpolator(pixcenters, sigma_map.ravel())
        sigma = interp(self.mesh.g)

        return sigma.reshape(-1, 1)


if __name__ == "__main__":
    # 测试数据集
    import matplotlib.pyplot as plt

    dataset = SimulatedEITDataset(length=10, mesh_name="Mesh_dense.mat")

    print(f"Dataset length: {len(dataset)}")
    print(f"Mesh nodes: {len(dataset.mesh.g)}")

    # 获取一个样本
    phantom, measurements = dataset[0]

    print(f"Phantom shape: {phantom.shape}")
    print(f"Measurements shape: {measurements.shape}")
    print(f"Phantom unique values: {torch.unique(phantom)}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(phantom.numpy().T, origin='lower', cmap='viridis')
    axes[0].set_title('Phantom')

    axes[1].plot(measurements.numpy())
    axes[1].set_title('Measurements')
    axes[1].set_xlabel('Measurement index')
    axes[1].set_ylabel('Voltage difference')

    plt.tight_layout()
    plt.savefig('test_simulated_dataset.png', dpi=150, bbox_inches='tight')
    print("Saved test figure to test_simulated_dataset.png")
