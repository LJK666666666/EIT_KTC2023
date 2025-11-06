import numpy as np
import KTCFwd  # EIT正向求解器模块
import KTCMeshing  # 网格生成与处理模块
import KTCRegularization  # 正则化先验模块
import KTCPlotting  # 绘图工具模块
import KTCScoring  # 评分与分割算法模块
import KTCAux  # 辅助工具函数模块
import matplotlib.pyplot as plt
import scipy as sp
# import warnings


# Replace MakePaths with appropriate imports
# from your_library import create2Dmesh_circ, setMeasurementPattern, simulateConductivity, EITFEM, SMprior, sigmaplotter, interpolateRecoToPixGrid, Otsu, Otsu2, scoringFunction

# 图像分割的类别数量（2 = 背景 + 一种包含物；3 = 背景 + 两种包含物）
segments = 3

# ========== 第一部分：设置数据仿真网格（密集网格） ==========
Nel = 32  # 电极数量
#Mesh2sim, Meshsim,elcenterangles = KTCMeshing.create2Dmesh_circ(Nel, 6, 1, 1)

# 加载预先生成的有限元网格（使用Gmsh生成，导出到Matlab并保存为.mat文件）
mat_dict_mesh = sp.io.loadmat('Mesh_dense.mat')
g = mat_dict_mesh['g']  # 节点坐标
H = mat_dict_mesh['H']  # 组成三角形单元的节点索引
elfaces = mat_dict_mesh['elfaces'][0].tolist()  # 边界电极的节点索引

# 构建一阶网格的单元结构
ElementT = mat_dict_mesh['Element']['Topology'].tolist()
for k in range(len(ElementT)):
    ElementT[k] = ElementT[k][0].flatten()
ElementE = mat_dict_mesh['ElementE'].tolist()  # 标记邻近边界电极的单元
for k in range(len(ElementE)):
    if len(ElementE[k][0]) > 0:
        ElementE[k] = [ElementE[k][0][0][0], ElementE[k][0][0][1:len(ElementE[k][0][0])]]
    else:
        ElementE[k] = []

# 构建节点结构
NodeC = mat_dict_mesh['Node']['Coordinate']
NodeE = mat_dict_mesh['Node']['ElementConnection']  # 标记每个节点所属的单元
nodes = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC]
for k in range(NodeC.shape[0]):
    nodes[k].ElementConnection = NodeE[k][0].flatten()
elements = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT]
for k in range(len(ElementT)):
    elements[k].Electrode = ElementE[k]

# 构建二阶网格数据（提供更高精度的数值计算）
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

# 创建仿真用的一阶和二阶网格对象
Meshsim = KTCMeshing.Mesh(H,g,elfaces,nodes,elements)
Mesh2sim = KTCMeshing.Mesh(H2,g2,elfaces2,nodes2,elements2)

# ========== 第二部分：设置测量模式并仿真数据 ==========
z = np.ones((Nel, 1))  # 接触阻抗（电极与组织接触界面的阻抗）
Inj, Mpat, vincl = KTCAux.setMeasurementPattern(Nel)  # 设置电流注入模式和电压测量模式

# 仿真初始电导率和电导率变化
sigma, delta_sigma, sigma2 = KTCAux.simulateConductivity(Meshsim, segments)
sgplot = KTCPlotting.SigmaPlotter(Meshsim, [2, 3], 'jet')
sgplot.basic2Dplot(sigma, delta_sigma, ['initial conductivity', 'conductivity change'])

# 建立EIT正向求解器
solver = KTCFwd.EITFEM(Mesh2sim, Inj, Mpat, vincl)

# 仿真测量数据
Iel2_true = solver.SolveForward(sigma, z)  # 初始状态的电压测量值
Iel_true = solver.SolveForward(sigma + delta_sigma, z)  # 变化后的电压测量值

# 添加测量噪声
noise_std1 = 0.1  # 噪声标准差，为每个电压测量值的百分比
noise_std2 = 0  # 第二噪声分量的标准差（与最大测量值成比例）
solver.SetInvGamma(noise_std1, noise_std2, Iel2_true)  # 计算噪声精度矩阵
tmp = np.random.randn(Iel2_true.shape[0])
Iel2_noisy = Iel2_true  # 初始状态的带噪声测量值（此处未添加噪声）
Iel_noisy = Iel_true  # 变化后的带噪声测量值（此处未添加噪声）
deltaI = Iel_noisy - Iel2_noisy  # 电压测量值的差分

# ========== 第三部分：创建反演网格（稀疏网格） ==========
#Mesh2, Mesh, elcenterangles = KTCMeshing.create2Dmesh_circ(Nel, 5, 1, 4)

# 加载预先生成的稀疏网格用于图像重建（比仿真网格更粗糙，计算效率更高）
mat_dict_mesh = sp.io.loadmat('Mesh_sparse.mat')
g = mat_dict_mesh['g']
H = mat_dict_mesh['H']
elfaces = mat_dict_mesh['elfaces'][0].tolist()

# 构建反演网格的一阶单元结构（与仿真网格结构相同，此处省略详细注释）
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

# 构建反演网格的二阶网格数据
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

# 创建反演用的一阶和二阶网格对象
Mesh = KTCMeshing.Mesh(H,g,elfaces,nodes,elements)
Mesh2 = KTCMeshing.Mesh(H2,g2,elfaces2,nodes2,elements2)


print(f'Nodes in simulation 1st order mesh: {len(Meshsim.g)}')
print(f'Nodes in inversion 1st order mesh: {len(Mesh.g)}')

# ========== 第四部分：设置正则化先验并执行线性差分重建 ==========
# 为电导率变化设置高斯平滑先验
sigma0 = np.ones((len(Mesh.g), 1))  # 初始电导率猜测（均匀分布）
corrlength = 1 * 0.115  # 相关长度（控制平滑程度）
var_sigma = 0.05**2  # 电导率的方差
mean_sigma = sigma0  # 电导率的均值
smprior = KTCRegularization.SMPrior(Mesh.g, corrlength, var_sigma, mean_sigma)

# 为反演设置正向求解器
solver = KTCFwd.EITFEM(Mesh2, Inj, Mpat, vincl)
solver.SetInvGamma(noise_std1, noise_std2, deltaI)

# 计算电导率变化的线性差分重建
J = solver.Jacobian(sigma0, z)  # 计算雅可比矩阵（灵敏度矩阵）
HtLtLH = J.T @ solver.InvGamma_n @ J  # 海森矩阵的数据项
RtR = smprior.L.T @ smprior.L  # 正则化项
Ht = J.T @ solver.InvGamma_n
# 求解正则化最小二乘问题：(J^T * InvGamma * J + L^T * L) * delta_sigma = J^T * InvGamma * deltaI
deltareco = np.linalg.solve(J.T @ solver.InvGamma_n @ J + smprior.L.T @ smprior.L, J.T @ solver.InvGamma_n @ deltaI)
sgplot = KTCPlotting.SigmaPlotter(Mesh, [5], 'jet')
sgplot.basic2Dplot(deltareco,[], ['linear difference reconstruction'])

# ========== 第五部分：将重建结果插值到像素网格并进行图像分割 ==========
# 将重建结果插值到规则的像素网格
deltareco_pixgrid = KTCAux.interpolateRecoToPixGrid(deltareco, Mesh)
fig, ax = plt.subplots()
cax = ax.imshow(deltareco_pixgrid)
plt.colorbar(cax)
plt.axis('image')

# 使用Otsu方法对图像直方图进行阈值分割
if segments == 2:
    level, x = KTCScoring.Otsu(deltareco_pixgrid.flatten(), 256, 7)
elif segments == 3:
    level, x = KTCScoring.Otsu2(deltareco_pixgrid.flatten(), 256, 7)  # Otsu2用于三类分割

# 根据阈值对重建图像进行分割
deltareco_pixgrid_segmented = np.zeros_like(deltareco_pixgrid)
if segments == 2:
    ind = deltareco_pixgrid < x[level[0]]
    deltareco_pixgrid_segmented[ind] = 1
elif segments == 3:
    ind = deltareco_pixgrid < x[level[0]]  # 低电导率区域标记为1
    deltareco_pixgrid_segmented[ind] = 1
    ind = deltareco_pixgrid > x[level[1]]  # 高电导率区域标记为2
    deltareco_pixgrid_segmented[ind] = 2

# 显示分割后的重建图像
fig, ax = plt.subplots()
cax = ax.imshow(deltareco_pixgrid_segmented, cmap='gray')
plt.colorbar(cax)
plt.axis('image')
plt.title('segmented linear difference reconstruction')

# 显示重建图像（灰度）
fig, ax = plt.subplots()
cax = ax.imshow(deltareco_pixgrid, cmap='gray')
plt.colorbar(cax)
plt.axis('image')

# ========== 第六部分：创建真实值的分割图像并计算评分 ==========
# 将仿真的真实电导率变化插值到像素网格
delta_pixgrid = KTCAux.interpolateRecoToPixGrid(delta_sigma, Meshsim)

# 对真实值使用Otsu方法进行阈值分割
if segments == 2:
    level, x = KTCScoring.Otsu(delta_pixgrid, 256, 9)
elif segments == 3:
    # with warnings.catch_warnings():
    #     warnings.simplefilter('error')
    level, x = KTCScoring.Otsu2(delta_pixgrid, 256, 9)

# 根据阈值对真实图像进行分割
delta_pixgrid_segmented = np.zeros_like(deltareco_pixgrid)
if segments == 2:
    ind = delta_pixgrid < x[level[0]]
    delta_pixgrid_segmented[ind] = 1
elif segments == 3:
    ind = delta_pixgrid < x[level[0]]
    delta_pixgrid_segmented[ind] = 1
    ind = delta_pixgrid > x[level[1]]
    delta_pixgrid_segmented[ind] = 2

# 显示真实值的分割图像
fig, ax = plt.subplots()
cax = ax.imshow(delta_pixgrid_segmented, cmap='gray')
plt.colorbar(cax)
plt.axis('image')
plt.title('ground truth image')

# 计算重建结果与真实值之间的评分（用于评估重建质量）
score = KTCScoring.scoringFunction(delta_pixgrid_segmented, deltareco_pixgrid_segmented)
print(f'SCORE = {score}')

input('Press any key to continue...')
