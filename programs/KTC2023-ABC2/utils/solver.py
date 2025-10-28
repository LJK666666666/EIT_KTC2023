import numpy as np
import scipy as sp
import scipy.io
import scipy.ndimage
import torch
import torch.nn.functional as F
# from torchmetrics.image import TotalVariation  # torchmetrics 0.8.2 版本不支持
from PIL import Image
from utils import KTCMeshing
from utils import KTCFwd
from utils import KTCScoring
from utils import DIPAux

# 检查并使用GPU进行计算
if torch.cuda.is_available():
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  dtype = torch.cuda.DoubleTensor  # 使用GPU张量类型（双精度）
  map_location = 'cuda:0'  # 指定GPU设备
  print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
  torch.backends.cudnn.enabled = False
  torch.backends.cudnn.benchmark = False
  dtype = torch.DoubleTensor  # 使用CPU张量类型（双精度）
  map_location = 'cpu'  # 指定CPU
  print("GPU not available, using CPU")

class TotalVariation(torch.nn.Module):
    """
    计算图像的总变差 (Total Variation)

    兼容 torchmetrics.image.TotalVariation 的自定义实现
    """
    def __init__(self, reduction='sum'):
        super().__init__()
        self.reduction = reduction

    def forward(self, x):
        # x 期望形状为 (N, C, H, W)
        if x.dim() != 4:
            raise ValueError('TotalVariation expects a 4D tensor (N,C,H,W)')
        # 计算水平方向的差分
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        # 计算垂直方向的差分
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        tv = dh.sum() + dw.sum()
        if self.reduction == 'mean':
            return tv / x.numel()
        return tv

def solve(inputData, categoryNbr):
    # 图像分辨率参数
    img_resolution = 128
    linpoint = 0.7927  # 线性化点
    deltaU_fac = 2  # 电压差放大因子
    dist_mode = 'lowpass'  # 距离计算模式
    dist_sigma = 8e-3  # 距离高斯滤波标准差

    # DIP网络架构参数
    skip_n11=32
    num_scales=6
    skip_n33d=[16, 16, 32, 32, 64, 64]
    skip_n33u=[16, 16, 32, 32, 64, 64]

    # 获取求解矩阵（雅可比矩阵和测量数据）
    solver_dict = get_solver_matrices(inputData, categoryNbr, linpoint)
    J = torch.from_numpy(solver_dict['J']).type(dtype)  # 雅可比矩阵
    deltaU = torch.from_numpy(solver_dict['deltaU']).type(dtype) * deltaU_fac  # 电压差

    # 计算节点到像素的距离矩阵
    dist = calc_dist(solver_dict['coordinates'], x_d=img_resolution,
                    y_d=img_resolution, mode=dist_mode, sigma=dist_sigma)
    dist = torch.from_numpy(dist).type(dtype)

    # 获取圆形掩膜（限制计算区域在圆形区域内）
    c_mask = get_mask(img_resolution)
    c_mask = torch.from_numpy(c_mask).type(dtype)
   
    # 网络输入参数
    input_depth = 32  # 输入噪声深度
    input_type = 'noise'  # 输入类型
    OPT_OVER = 'net'  # 优化对象（网络参数）

    # 优化参数
    OPTIMIZER = 'adam'  # 优化器类型
    pad = 'reflection'  # 填充模式
    NET_TYPE = 'skip'  # 网络类型（跳跃连接）
    LR = 3.5e-3  # 学习率
    reg_noise_std = 0.04  # 正则化噪声标准差
    # num_iter=3000  # 迭代次数
    num_iter=500  # 迭代次数
    w_tv = 5e-8  # 总变差权重
    tv = TotalVariation().to(map_location)  # 总变差计算器

    # 初始化随机输入
    deblur_input = DIPAux.get_noise(input_depth,input_type,
                (img_resolution,img_resolution)).type(dtype).detach()

    # 构建DIP网络
    deblur_net = DIPAux.get_net(input_depth, NET_TYPE, pad,
            skip_n33d = skip_n33d,
            skip_n33u = skip_n33u,
            skip_n11 = skip_n11,
            n_channels=1,
            num_scales = num_scales,
            upsample_mode='bilinear',
            need_sigmoid=False).type(dtype)

    # 保存原始网络输入和噪声副本
    net_input_saved = deblur_input.detach().clone()
    noise = deblur_input.detach().clone()

    # 获取要优化的参数
    p = DIPAux.get_params(OPT_OVER,deblur_net,deblur_input)

    # 创建优化器
    optimizer = torch.optim.Adam(p, lr=LR)

    # 优化迭代过程
    for i in range(num_iter):
        optimizer.zero_grad()

        # 添加正则化噪声
        if reg_noise_std > 0:
            deblur_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            deblur_input = net_input_saved

        # 前向传播：获得电导率像素值
        cond_pixels = c_mask * deblur_net(deblur_input)
        # 将像素值映射到网格节点
        cond_nodes = dist @ cond_pixels[0,0].reshape((-1,1))

        # 计算总损失（MSE损失+总变差正则化）
        total_loss = F.mse_loss(J @ cond_nodes, deltaU)
        total_loss += w_tv * tv(cond_pixels)

        # 反向传播和优化步
        total_loss.backward()
        optimizer.step()

        # 每500次迭代输出一次进度
        if (i + 1) % 500 == 0:
            print(f"Iteration {i+1}/{num_iter}, Loss: {total_loss.item():.6f}")

    # 将结果转为numpy并调整尺寸
    cond_pixels_np = np.flipud(cond_pixels[0,0].clone().detach().cpu().numpy())
    cond_pixels_np = np.array(Image.fromarray(cond_pixels_np).resize((256,256)))

    # 对重建结果进行分割
    cond_pixels_np_segmented = segment(cond_pixels_np)

    return cond_pixels_np_segmented

def get_solver_matrices(inputData, categoryNbr, lin_point=1):
    # 电极数量设置
    Nel = 32  # 电极数量
    z = (1e-6) * np.ones((Nel, 1))  # 接触阻抗
    # 加载参考数据
    mat_dict = sp.io.loadmat('utils/reference_files/ref.mat') # 加载参考数据
    Injref = mat_dict["Injref"] # 参考电流注入
    Uelref = mat_dict["Uelref"] # 水槽测量的参考电压
    Mpat = mat_dict["Mpat"] # 电压测量模式
    vincl = np.ones(((Nel - 1),76), dtype=bool) # 指定反演中包含的测量值
    rmind = np.arange(0,2 * (categoryNbr - 1),1) # 需要移除的电极数据

    # 根据难度级别移除相应的测量数据
    for ii in range(0,75):
        for jj in rmind:
            if Injref[jj,ii]:
                vincl[:,ii] = 0
            vincl[jj,:] = 0

    # 加载预生成的有限元网格（使用Gmsh生成，导出为Matlab格式）
    mat_dict_mesh = sp.io.loadmat('utils/reference_files/Mesh_sparse.mat')
    g = mat_dict_mesh['g'] # 节点坐标
    H = mat_dict_mesh['H'] # 组成三角形单元的节点索引
    elfaces = mat_dict_mesh['elfaces'][0].tolist() # 组成边界电极的节点索引

    # 单元结构
    ElementT = mat_dict_mesh['Element']['Topology'].tolist()
    for k in range(len(ElementT)):
        ElementT[k] = ElementT[k][0].flatten()
    # 标记与边界电极相邻的单元
    ElementE = mat_dict_mesh['ElementE'].tolist()
    for k in range(len(ElementE)):
        if len(ElementE[k][0]) > 0:
            ElementE[k] = [ElementE[k][0][0][0], ElementE[k][0][0][1:len(ElementE[k][0][0])]]
        else:
            ElementE[k] = []

    # 节点结构
    NodeC = mat_dict_mesh['Node']['Coordinate']
    NodeE = mat_dict_mesh['Node']['ElementConnection'] # 标记节点所属的单元
    nodes = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC]
    for k in range(NodeC.shape[0]):
        nodes[k].ElementConnection = NodeE[k][0].flatten()
    elements = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT]
    for k in range(len(ElementT)):
        elements[k].Electrode = ElementE[k]

    # 二阶网格数据
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

    # 二阶网格节点坐标和单元连接
    NodeC2 = mat_dict_mesh['Node2']['Coordinate']
    NodeE2 = mat_dict_mesh['Node2']['ElementConnection']
    nodes2 = [KTCMeshing.NODE(coord[0].flatten(), []) for coord in NodeC2]
    for k in range(NodeC2.shape[0]):
        nodes2[k].ElementConnection = NodeE2[k][0].flatten()
    elements2 = [KTCMeshing.ELEMENT(ind, []) for ind in ElementT2]
    for k in range(len(ElementT2)):
        elements2[k].Electrode = ElementE2[k]

    # 创建一阶和二阶网格对象
    Mesh = KTCMeshing.Mesh(H,g,elfaces,nodes,elements)
    Mesh2 = KTCMeshing.Mesh(H2,g2,elfaces2,nodes2,elements2)

    # 线性化参数设置
    sigma0 = np.ones((len(Mesh.g), 1))*lin_point # 线性化点
    corrlength = 1 * 0.115 # 先验中使用的相关长度
    var_sigma = 0.05 ** 2 # 先验方差
    mean_sigma = sigma0

    # 为反演设置正向求解器
    solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)

    vincl = vincl.T.flatten()

    # 为反演设置噪声模型
    noise_std1 = 0.05  # 第一个噪声分量的标准差（相对于每个电压测量值）
    noise_std2 = 0.01  # 第二个噪声分量的标准差（相对于最大电压测量值）
    solver.SetInvGamma(noise_std1, noise_std2, Uelref)

    # 加载输入数据
    mat_dict2 = sp.io.loadmat(inputData)
    Uel = mat_dict2["Uel"]
    deltaU = Uel[vincl] - Uelref[vincl]  # 计算电压差

    # 在线性化点处计算正向解和雅可比矩阵
    Usim = solver.SolveForward(sigma0, z) # 线性化点处的正向解
    J = solver.Jacobian(sigma0, z)  # 雅可比矩阵

    # 返回求解所需的矩阵和数据
    outputs = {"J": J,
                "deltaU": deltaU,
                "coordinates":np.array([coord[0].flatten() for coord in NodeC])
              }

    return outputs

def calc_dist(nodes_coordinates, x_d=256, y_d=256, mode='nearest', sigma=5e-7):
    # 转置并交换坐标顺序
    nodes_coordinates = nodes_coordinates.copy().T
    nodes_coordinates = nodes_coordinates[[1,0],:]  # 交换x和y坐标
    n_elements = nodes_coordinates.shape[1]

    # 获取坐标范围
    x_min = np.min(nodes_coordinates[0,:])
    y_min = np.min(nodes_coordinates[1,:])

    x_max = np.max(nodes_coordinates[0,:])
    y_max = np.max(nodes_coordinates[1,:])

    # 计算网格间距
    b = [x_min, y_min]
    a = [0, 0]
    a[0] = (x_max - x_min) / x_d
    a[1] = (y_max - y_min) / y_d

    # 生成像素网格坐标
    xp = np.zeros((x_d,y_d))
    yp = np.zeros((x_d,y_d))

    for i in range(x_d):
        for j in range(y_d):
            xp[i,j] = a[0]*i + b[0]
            yp[i,j] = a[1]*j + b[1]

    xp = xp.reshape(-1)
    yp = yp.reshape(-1)

    # 初始化距离矩阵
    dist = np.zeros((n_elements, y_d*x_d))

    if(mode=='nearest'):
        # 最近邻插值模式：找最近的像素
        for i in range(n_elements):
            dist[i,:] = ((nodes_coordinates[:,i][:,None] - [xp,yp])**2).sum(axis=0)
            dist[i,:] = dist[i,:] == dist[i,:].min()

    elif(mode=='lowpass'):
        # 高斯加权插值模式：用高斯函数平滑分配权重
        for i in range(n_elements):
            dist[i,:] = ((nodes_coordinates[:,i][:,None] - [xp,yp])**2).sum(axis=0)

        dist = np.exp(-dist**2/(2*sigma**2))  # 高斯加权
        dist /= (dist.sum(axis=1)+np.finfo(float).eps).reshape(-1,1)  # 归一化

    return dist

def get_mask(n_img):
    # 生成网格坐标
    Y, X = np.mgrid[0:n_img,0:n_img]

    # 圆形掩膜参数
    r = n_img//2 + 1
    xc = n_img//2 - 1
    yc = n_img//2 - 1

    # 创建圆形掩膜（保留圆形区域内的像素）
    c_mask = (X-xc)**2 + (Y-yc)**2 < r**2

    # 调整为4D张量形状 (batch, channel, height, width)
    c_mask = c_mask[None,None,:,:]

    return c_mask

def segment(cond_pixels_np):
    # 使用Otsu多阈值分割方法
    level, x = KTCScoring.Otsu2(cond_pixels_np.flatten(), 256, 7)

    # 初始化分割结果数组
    cond_pixels_np_segmented = np.zeros_like(cond_pixels_np)

    # 根据阈值将像素分为三类
    ind0 = cond_pixels_np < x[level[0]]
    ind1 = np.logical_and(cond_pixels_np >= x[level[0]],cond_pixels_np <= x[level[1]])
    ind2 = cond_pixels_np > x[level[1]]
    # 统计各类像素数量
    inds = [np.count_nonzero(ind0),np.count_nonzero(ind1),np.count_nonzero(ind2)]
    # 确定背景类（像素数最多的类）
    bgclass = inds.index(max(inds))

    # match bgclass:
    #     case 0:  # 第一类为背景
    #         cond_pixels_np_segmented[ind1] = 2
    #         cond_pixels_np_segmented[ind2] = 2
    #     case 1:  # 第二类为背景
    #         cond_pixels_np_segmented[ind0] = 1
    #         cond_pixels_np_segmented[ind2] = 2
    #     case 2:  # 第三类为背景
    #         cond_pixels_np_segmented[ind0] = 1
    #         cond_pixels_np_segmented[ind1] = 1

    # 根据背景类标记前景像素（使用 if/elif/else 以兼容 Python <3.10）
    if bgclass == 0:  # 第一类为背景
        cond_pixels_np_segmented[ind1] = 2
        cond_pixels_np_segmented[ind2] = 2
    elif bgclass == 1:  # 第二类为背景
        cond_pixels_np_segmented[ind0] = 1
        cond_pixels_np_segmented[ind2] = 2
    else:  # 第三类为背景或其他情况
        cond_pixels_np_segmented[ind0] = 1
        cond_pixels_np_segmented[ind1] = 1

    # 使用形态学开运算去除噪声
    opening_mask = sp.ndimage.binary_opening(cond_pixels_np_segmented, iterations=10)
    cond_pixels_np_segmented = opening_mask * cond_pixels_np_segmented
  
    return cond_pixels_np_segmented
