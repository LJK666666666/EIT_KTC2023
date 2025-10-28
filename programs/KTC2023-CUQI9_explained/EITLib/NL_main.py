
# %%

# 导入FEniCS库（用于有限元计算）
from dolfin import *
# from dolfinx import *
# 导入matplotlib绘图库
from matplotlib import pyplot as plt
# 导入工具函数模块
from .utils import *
# 导入scipy的IO模块（用于读写.mat文件）
import scipy.io as io
# 导入平滑先验正则化类
from .KTCRegularization_NLOpt import SMPrior
# 导入pickle模块（用于加载序列化对象）
import pickle
# 导入高斯滤波器
from scipy.ndimage import gaussian_filter
# 导入Chan-Vese分割和评分函数
from .segmentation import cv, scoring_function
# 导入scipy优化器
from scipy.optimize import minimize

def NL_main(Uel_ref, background_Uel_ref, Imatr, difficulty_level, niter=50, output_dir_name=None):
    """
    非线性EIT图像重建主函数（第一版本）

    参数:
        Uel_ref: 参考电压测量值
        background_Uel_ref: 背景电压测量值
        Imatr: 电流注入矩阵
        difficulty_level: 难度级别
        niter: 迭代次数
        output_dir_name: 输出目录名称
    """
    # 设置参数
    # 高电导率值
    high_conductivity = 1e1
    # 低电导率值
    low_conductivity = 1e-2
    # 背景电导率值
    background_conductivity = 0.8
    # 圆形区域半径
    radius = 0.115
    # 将参考电压数据展平为一维数组
    Uel_data =  Uel_ref.flatten()
    # 将背景参考电压数据展平为一维数组
    background_Uel_ref = background_Uel_ref.flatten()

    # %% 构建EIT-FEniCS模型
    # 电极数量
    L = 32
    # 创建网格对象
    mesh = Mesh()
    # 从XDMF文件读取32电极、300节点的网格
    with XDMFFile("./EITLib/mesh_file_32_300.xdmf") as infile:
        infile.read(mesh)
    # 创建EIT FEniCS对象
    myeit = EITFenics(mesh=mesh, L=L, background_conductivity=background_conductivity)


    # %% 计算背景幻影的电压
    # 创建背景幻影（256x256的均匀电导率分布）
    background_phantom_float = np.zeros((256,256)) + background_conductivity
    # 求解背景幻影的前向问题，得到模拟电压
    background_Uel_sim, background_Q, background_q_list = myeit.solve_forward(Imatr, background_phantom_float, 76)

    # %% 准备模拟/伪数据
    # 添加背景解信息到EIT对象
    myeit.add_background_sol_info(background_Uel_sim, background_Uel_ref, background_q_list)
    # 设置逆问题的噪声参数（Gamma分布参数）
    myeit.SetInvGamma( 0.05, 0.01, meas_data=Uel_data- background_Uel_ref)


    #%% 目标函数定义
    # 加载平滑先验对象（从预先保存的文件中）
    # load smprior object
    file = open("./EITLib/smprior_32_300.p", 'rb')

    # 反序列化加载平滑先验对象
    smprior = pickle.load(file)

    # 来自CUQI1提交的正则化项
    # Regularization term from CUQI1 submission
    #############################  Changed code

    # 定义CUQI1正则化类
    class reg_CUQI1:
        def __init__(self, radius, mesh, difficulty_level, H) -> None:
            """
            初始化CUQI1正则化项

            参数:
                radius: 圆形区域半径
                mesh: 有限元网格
                difficulty_level: 难度级别
                H: 函数空间
            """

            # 获取自由度到顶点的映射
            d2v = dof_to_vertex_map(H)

            radius = radius
            # 电极数量
            Nel = 32
            # 网格顶点数量
            m = mesh.num_vertices()
            # 根据难度级别计算实际可用电极数量
            num_el =  Nel - (2*difficulty_level - 1)
            # 初始化电极位置数组
            electrodes = np.zeros((num_el, 2))
            # 计算电极之间的角度间隔
            angle = 2*np.pi/Nel
            # 计算每个电极的位置（圆周上均匀分布）
            for i in range(num_el):
                electrodes[i] = radius*np.array([np.sin(i*angle), np.cos(i*angle)])
            # 初始化距离权重矩阵的对角元素
            D = np.zeros(m)
            # 获取网格坐标
            mesh_coordinate = mesh.coordinates()[d2v]
            # 对每个网格顶点计算到所有电极的距离权重
            for i in range(m):
                v = mesh_coordinate[i,:]
                dist = np.zeros(num_el)
                # 计算到每个电极的距离
                for k, e in enumerate(electrodes):
                    dist[k] = np.linalg.norm(v - e)
                # 计算权重：使用3-范数的立方乘以到原点距离的4次方
                D[i] = (np.linalg.norm(dist, ord = 3)**3)*np.linalg.norm(v)**4
            # 将权重向量转换为对角矩阵
            self.D = np.diag(D)


        def evaluate_target_external(self, x):
            """计算CUQI1正则化项的值"""
            return 0.5*np.linalg.norm(self.D@x)**2

        def evaluate_grad_external(self, x):
            """计算CUQI1正则化项的梯度"""
            return self.D.T@self.D@x

    # 创建CUQI1正则化对象
    # define regularization term
    reg_CUQI1_obj = reg_CUQI1(radius, myeit.mesh, difficulty_level, myeit.H_sigma)


    # 以下是注释掉的代码（备选的正则化参数设置）
    #reg = 0.75
    #reg2 = 1e15
    # 以下是注释掉的代码（用于绘制CUQI1正则化权重分布）
    # plot the diagonal of reg_CUQI1_obj


    #plt.figure()
    #reg_CUQI1_fun = Function(myeit.H_sigma)
    #reg_CUQI1_fun.vector().set_local( reg_CUQI1_obj.D.diagonal())
    #im =plot(reg_CUQI1_fun)
    #plt.colorbar(im)
    #plt.show()
    #exit()


########


    # 定义用于TV（全变分）正则化的目标函数类
    # Class Target_scipy_TV for TV regularization that uses TV_reg
    class Target_scipy_TV:
        def __init__(self, myeit, tv_reg, reg_CUQI1_obj, smprior, Imatr, Uel_data, factor=1, factor_sm=1, factor_CUQI1=1e15) -> None:
            """
            初始化复合目标函数（数据拟合项 + TV正则化 + Tikhonov正则化 + CUQI1正则化）

            参数:
                myeit: EIT FEniCS对象
                tv_reg: TV正则化对象
                reg_CUQI1_obj: CUQI1正则化对象
                smprior: 平滑先验对象
                Imatr: 电流注入矩阵
                Uel_data: 测量电压数据
                factor: TV正则化因子
                factor_sm: 平滑先验（Tikhonov）正则化因子
                factor_CUQI1: CUQI1正则化因子
            """
            self.myeit = myeit
            self.tv_reg = tv_reg
            self.Imatr = Imatr
            self.Uel_data = Uel_data

            # 初始化各项目标函数值
            self.v1 = None  # 数据拟合项
            self.v2 = None  # TV正则化项
            self.v3 = None  # Tikhonov正则化项
            self.v4 = None  # CUQI1正则化项
            # 初始化各项梯度
            self.g1 = None  # 数据拟合项梯度
            self.g2 = None  # TV正则化项梯度
            self.g3 = None  # Tikhonov正则化项梯度
            self.g4 = None  # CUQI1正则化项梯度

            # 各项正则化因子
            self.factor = factor
            self.factor_sm = factor_sm
            self.smprior = smprior

            self.reg_CUQI1_obj = reg_CUQI1_obj
            self.factor_CUQI1 = factor_CUQI1

            # 迭代计数器
            self.counter = 0
            # 记录各项目标函数值的列表（用于监控优化过程）
            self.list_v1 = []
            self.list_v2 = []
            self.list_v3 = []
            self.list_v4 = []


        def obj_scipy(self,x):
            """
            计算目标函数值（用于scipy优化器）

            参数:
                x: 当前的电导率分布向量

            返回:
                目标函数总值 = v1 + factor*v2 + factor_sm*v3 + factor_CUQI1*v4
            """
            # 将向量x转换为FEniCS函数
            x_fun = Function(self.myeit.H_sigma)
            x_fun.vector().set_local(x)

            # 更新迭代计数器
            self.counter +=1
            # 计算数据拟合项及其梯度
            self.v1, self.g1 =  self.myeit.evaluate_target_external(self.Imatr, x, self.Uel_data, compute_grad=True)
            # 计算Tikhonov正则化项及其梯度
            self.v3, self.g3 = self.smprior.evaluate_target_external(x,  compute_grad=True)
            factor = self.factor
            # 计算TV正则化项
            self.v2 = self.tv_reg.cost_reg(x_fun )

            factor_CUQI1 = self.factor_CUQI1
            # 计算CUQI1正则化项
            self.v4 = self.reg_CUQI1_obj.evaluate_target_external(x)


            # 以下是注释掉的自适应正则化参数代码
            #self.factor = 0.6*((self.v1/2)/self.v2)
            #self.factor_sm = 0.6*((self.v1/2)/self.v3)
            # 记录各项的对数值（用于后续分析）
            self.list_v1.append(np.log(self.v1))
            self.list_v2.append(np.log(self.factor*self.v2))
            self.list_v3.append(np.log(self.factor_sm*self.v3))
            self.list_v4.append(np.log(self.factor_CUQI1*self.v4))




            # 打印当前迭代次数
            print(self.counter)
            # 绘图标志（默认关闭）
            plot_flag = False
            # 每5次迭代绘制一次中间结果（如果开启绘图）
            if self.counter % 5 == 0 and plot_flag:
                plt.figure()
                im = plot(self.myeit.inclusion)
                plt.colorbar(im)
                plt.title("sigma "+str(self.counter))
                plt.show()
                plt.figure()
                plt.plot(self.list_v1)
                plt.title("v1")
                plt.show()
                plt.figure()
                plt.plot(self.list_v2)
                plt.title("v2")
                plt.show()
                plt.figure()
                plt.plot(self.list_v3)
                plt.title("v3")
                plt.show()
                plt.figure()
                plt.plot(self.list_v4)
                plt.title("v4")
                plt.show()

            # 打印目标函数各项的值
            print(self.v1+factor*self.v2, "(", self.v1, "+", factor*self.v2,  "+", self.factor_sm*self.v3, "+", self.factor_CUQI1*self.v4, ")")

            # 返回目标函数总值
            return self.v1+factor*self.v2+self.factor_sm*self.v3+self.factor_CUQI1*self.v4

        def obj_scipy_grad(self, x):
            """
            计算目标函数的梯度（用于scipy优化器）

            参数:
                x: 当前的电导率分布向量

            返回:
                目标函数的梯度 = g1 + factor*g2 + factor_sm*g3 + factor_CUQI1*g4
            """
            # 将向量x转换为FEniCS函数
            x_fun = Function(self.myeit.H_sigma)
            x_fun.vector().set_local(x)
            # 获取数据拟合项的梯度
            g1 = self.g1
            # 计算TV正则化项的梯度
            g2 = self.tv_reg.grad_reg(x_fun).get_local()
            self.g2 = g2
            # 获取Tikhonov正则化项的梯度
            g3 = self.g3
            # 计算CUQI1正则化项的梯度
            g4 = self.reg_CUQI1_obj.evaluate_grad_external(x)

            factor = self.factor
            factor_sm = self.factor_sm
            factor_CUQI1 = self.factor_CUQI1

            # 绘图标志（默认关闭）
            plot_flag = False
            # 每20次迭代绘制一次梯度信息（如果开启绘图）
            if self.counter % 20 == 0 and plot_flag:
              # 将梯度转换为FEniCS函数用于可视化
              g1_fenics = Function(self.myeit.H_sigma)
              g1_fenics.vector()[:] = g1.flatten()
              g2_fenics = Function(self.myeit.H_sigma)
              g2_fenics.vector()[:] = g2.flatten()
              g3_fenics = Function(self.myeit.H_sigma)
              g3_fenics.vector()[:] = g3.flatten()
              g_fenics = Function(self.myeit.H_sigma)
              g_fenics.vector()[:] = g1.flatten()+factor*g2.flatten()+factor_sm*g3.flatten()
              plt.figure()
              im = plot(g1_fenics)
              plt.colorbar(im)
              plt.title("grad 1")
              plt.show()
              plt.figure()
              im = plot(g2_fenics)
              plt.colorbar(im)
              plt.title("grad 2")
              plt.show()
              plt.figure()
              im = plot(g3_fenics)
              plt.colorbar(im)
              plt.title("grad 3")
              plt.figure()
              im = plot(g_fenics)
              plt.colorbar(im)
              plt.title("grad (g1 + factor*g2 + factor_sm*g3)")
              plt.show()


            # 返回目标函数的总梯度
            return g1.flatten()+factor*g2.flatten()+factor_sm*g3.flatten()+factor_CUQI1*g4.flatten()

    #%% 初始化和优化
    # 创建初始猜测x0
    recon_background_flag = False

    if recon_background_flag:
        # 从文件加载之前的重建结果作为初始值
        recon_KTC = io.loadmat(recon_KTC_file)["reconstruction"]
        recon_KTC_float = np.zeros_like(recon_KTC)
        recon_KTC_float[:] = recon_KTC
        # 将分割值映射到电导率值
        recon_KTC_float[recon_KTC_float == 0] = background_conductivity
        recon_KTC_float[recon_KTC_float == 1] = low_conductivity
        recon_KTC_float[recon_KTC_float == 2] = high_conductivity

        im = plt.imshow(np.flipud(recon_KTC_float))
        plt.title('KTC reconstruction, flipped for interpolation')
        plt.colorbar(im)

        # 对初始猜测应用高斯平滑
        recon_KTC_float_smoothed = gaussian_filter(recon_KTC_float, sigma=30)
        plt.figure()
        im = plt.imshow(recon_KTC_float_smoothed)
        plt.colorbar(im)


        # 将平滑后的结果插值到有限元函数空间
        x0_exp = Inclusion(np.fliplr(recon_KTC_float_smoothed.T), radius, degree=1)
        x0_fun = interpolate(x0_exp, myeit.H_sigma)
        plot(x0_fun)
        x0 = x0_fun.vector().get_local()
    else:
        # 使用背景电导率作为初始猜测
        x0 = 0.8 * np.ones(myeit.H_sigma.dim())

    # 创建TV正则化对象
    tv_reg = TV_reg(myeit.H_sigma, None, 1, 1e-4)
    # 创建复合目标函数对象
    target_scipy_TV = Target_scipy_TV( myeit, tv_reg, reg_CUQI1_obj, smprior=smprior, Imatr=Imatr, Uel_data=Uel_data, factor=5e5, factor_sm=0.5, factor_CUQI1=1e10)

    #%% 运行优化
    # 导入时间模块
    import time
    # 记录优化开始时间
    start = time.time()
    # 设置变量的边界约束
    bounds = [(1e-5,100)]*myeit.H_sigma.dim()
    # 使用L-BFGS-B优化算法求解最小化问题
    res = minimize(target_scipy_TV.obj_scipy, x0, method='L-BFGS-B', jac=target_scipy_TV.obj_scipy_grad, options={'disp': True, 'maxiter':niter} , bounds=bounds)
    # 记录优化结束时间
    end = time.time()
    print("time elapsed: ", end-start)
    print("time elapsed in minutes: ", (end-start)/60)

    # 保存各项目标函数值的列表
    v1_list = np.array(target_scipy_TV.list_v1)
    v2_list = np.array(target_scipy_TV.list_v2)
    v3_list = np.array(target_scipy_TV.list_v3)
    np.savez(output_dir_name+"/v_list.npz", v1_list=v1_list, v2_list=v2_list, v3_list=v3_list)

    #%% 将结果投影到像素网格
    # 将优化结果转换为FEniCS函数
    res_fenics = Function(myeit.H_sigma)
    res_fenics.vector().set_local( res['x'])

    # 绘图标志（默认关闭）
    plot_flag = False
    if plot_flag:
        plt.figure()
        im = plot(res_fenics)
        plt.colorbar(im)

    #%% 投影和分割
    # 创建256x256的像素网格
    X, Y = np.meshgrid(np.linspace(-radius,radius,256),np.linspace(-radius,radius,256) )
    Z = np.zeros_like(X)

    # 在网格上进行插值：逐个像素计算有限元解的值
    for i in range(256):
        for j in range(256):
            try:
                Z[i,j] = res_fenics(X[i,j], Y[i,j])
            except:
                # 插值失败时使用背景电导率值
                Z[i,j] = background_conductivity

    #%% 翻转结果以匹配标准坐标系
    deltareco_pixgrid = np.flipud(Z)

    return deltareco_pixgrid


def NL_main_2(Uel_ref, background_Uel_ref, Imatr, difficulty_level, niter=50, output_dir_name=None, TV_factor=1, Tikhonov_factor=1, CUQI1_factor=1e15):
    """
    非线性EIT图像重建主函数（第二版本，改进版）

    参数:
        Uel_ref: 参考电压测量值
        background_Uel_ref: 背景电压测量值
        Imatr: 电流注入矩阵
        difficulty_level: 难度级别
        niter: 迭代次数
        output_dir_name: 输出目录名称
        TV_factor: TV正则化因子
        Tikhonov_factor: Tikhonov正则化因子
        CUQI1_factor: CUQI1正则化因子
    """
    # 设置参数
    high_conductivity = 1e1
    low_conductivity = 1e-2
    background_conductivity = 0.8
    radius = 0.115
    Uel_data =  Uel_ref.flatten()
    background_Uel_ref = background_Uel_ref.flatten()

    # %% 构建EIT-FEniCS模型
    L = 32
    mesh = Mesh()
    with XDMFFile("./EITLib/mesh_file_32_300.xdmf") as infile:
        infile.read(mesh)
    myeit = EITFenics(mesh=mesh, L=L, background_conductivity=background_conductivity)

    # %% 计算背景幻影的电压
    background_phantom_float = np.zeros((256,256)) + background_conductivity
    background_Uel_sim, background_Q, background_q_list = myeit.solve_forward(Imatr, background_phantom_float, 76)

    # %% 准备模拟/伪数据
    myeit.add_background_sol_info(background_Uel_sim, background_Uel_ref, background_q_list)
    myeit.SetInvGamma( 0.05, 0.01, meas_data=Uel_data- background_Uel_ref)

    
    #%% 目标函数定义
    # 加载平滑先验对象
    # load smprior object
    file = open("./EITLib/smprior_32_300.p", 'rb')

    # 反序列化加载平滑先验对象
    smprior = pickle.load(file)

    # 来自CUQI1提交的正则化项
    # Regularization term from CUQI1 submission
    #############################  Changed code

    # 定义CUQI1正则化类（与NL_main中的类相同）
    class reg_CUQI1:
        def __init__(self, radius, mesh, difficulty_level, H) -> None:
            # 获取自由度到顶点的映射
            d2v = dof_to_vertex_map(H)

            radius = radius
            # 电极数量
            Nel = 32
            # 网格顶点数量
            m = mesh.num_vertices()
            # 根据难度级别计算实际可用电极数量
            num_el =  Nel - (2*difficulty_level - 1)
            # 初始化电极位置数组
            electrodes = np.zeros((num_el, 2))
            # 计算电极之间的角度间隔
            angle = 2*np.pi/Nel
            # 计算每个电极的位置
            for i in range(num_el):
                electrodes[i] = radius*np.array([np.sin(i*angle), np.cos(i*angle)])
            # 初始化距离权重矩阵的对角元素
            D = np.zeros(m)
            # 获取网格坐标
            mesh_coordinate = mesh.coordinates()[d2v]
            # 对每个网格顶点计算距离权重
            for i in range(m):
                v = mesh_coordinate[i,:]
                dist = np.zeros(num_el)
                # 计算到每个电极的距离
                for k, e in enumerate(electrodes):
                    dist[k] = np.linalg.norm(v - e)
                # 计算权重：距离的3-范数的立方乘以到原点距离的4次方
                D[i] = (np.linalg.norm(dist, ord = 3)**3)*np.linalg.norm(v)**4
            # 将权重向量转换为对角矩阵
            self.D = np.diag(D)


        def evaluate_target_external(self, x):
            """计算CUQI1正则化项的值"""
            return 0.5*np.linalg.norm(self.D@x)**2

        def evaluate_grad_external(self, x):
            """计算CUQI1正则化项的梯度"""
            return self.D.T@self.D@x

    # 创建CUQI1正则化对象
    # define regularization term
    reg_CUQI1_obj = reg_CUQI1(radius, myeit.mesh, difficulty_level, myeit.H_sigma)


    # 以下是注释掉的代码
    #reg = 0.75
    #reg2 = 1e15
    # plot the diagonal of reg_CUQI1_obj

    #plt.figure()
    #reg_CUQI1_fun = Function(myeit.H_sigma)
    #reg_CUQI1_fun.vector().set_local( reg_CUQI1_obj.D.diagonal())
    #im =plot(reg_CUQI1_fun)
    #plt.colorbar(im)
    #plt.show()
    #exit()


########

    # 定义用于TV正则化的目标函数类（改进版，使用动态正则化因子）
    # Class Target_scipy_TV for TV regularization that uses TV_reg
    class Target_scipy_TV:
        def __init__(self, myeit, tv_reg, reg_CUQI1_obj, smprior, Imatr, Uel_data, factor=1, factor_sm=1, factor_CUQI1=1e15) -> None:
            """
            初始化复合目标函数（改进版）

            参数:
                myeit: EIT FEniCS对象
                tv_reg: TV正则化对象
                reg_CUQI1_obj: CUQI1正则化对象
                smprior: 平滑先验对象
                Imatr: 电流注入矩阵
                Uel_data: 测量电压数据
                factor: TV正则化因���
                factor_sm: Tikhonov正则化因子
                factor_CUQI1: CUQI1正则化因子
            """
            self.myeit = myeit
            self.tv_reg = tv_reg
            self.Imatr = Imatr
            self.Uel_data = Uel_data

            # 初始化各项目标函数值
            self.v1 = None
            self.v2 = None
            self.v3 = None
            self.v4 = None
            # 初始化各项梯度
            self.g1 = None
            self.g2 = None
            self.g3 = None
            self.g4 = None

            # 各项正则化因子（改进版：可动态调整）
            self.factor = factor
            self.factor_sm = factor_sm
            self.smprior = smprior

            self.reg_CUQI1_obj = reg_CUQI1_obj
            self.factor_CUQI1 = factor_CUQI1

            # 迭代计数器
            self.counter = 0
            # 记录各项目标函数值的列表
            self.list_v1 = []
            self.list_v2 = []
            self.list_v3 = []
            self.list_v4 = []


        def obj_scipy(self,x):
            """计算目标函数值（改进版）"""
            # 将向量x转换为FEniCS函数
            x_fun = Function(self.myeit.H_sigma)
            x_fun.vector().set_local(x)

            # 更新迭代计数器
            self.counter +=1
            # 计算数据拟合项及其梯度
            self.v1, self.g1 =  self.myeit.evaluate_target_external(self.Imatr, x, self.Uel_data, compute_grad=True)
            # 计算Tikhonov正则化项及其梯度
            self.v3, self.g3 = self.smprior.evaluate_target_external(x,  compute_grad=True)
            factor = self.factor
            # 计算TV正则化项
            self.v2 = self.tv_reg.cost_reg(x_fun )

            factor_CUQI1 = self.factor_CUQI1
            # 计算CUQI1正则化项
            self.v4 = self.reg_CUQI1_obj.evaluate_target_external(x)

            # 自适应正则化参数代码（注释掉）
            #self.factor = 0.6*((self.v1/2)/self.v2)
            #self.factor_sm = 0.6*((self.v1/2)/self.v3)
            # 记录各项的对数值
            self.list_v1.append(np.log(self.v1))
            self.list_v2.append(np.log(self.factor*self.v2))
            self.list_v3.append(np.log(self.factor_sm*self.v3))
            self.list_v4.append(np.log(self.factor_CUQI1*self.v4))

            # 打印迭代次数
            print(self.counter)
            # 绘图标志（默认关闭）
            plot_flag = False
            if self.counter % 5 == 0 and plot_flag:
                # 绘制中间结果和目标函数值变化
                plt.figure()
                im = plot(self.myeit.inclusion)
                plt.colorbar(im)
                plt.title("sigma "+str(self.counter))
                plt.show()
                plt.figure()
                plt.plot(self.list_v1)
                plt.title("v1")
                plt.show()
                plt.figure()
                plt.plot(self.list_v2)
                plt.title("v2")
                plt.show()
                plt.figure()
                plt.plot(self.list_v3)
                plt.title("v3")
                plt.show()
                plt.figure()
                plt.plot(self.list_v4)
                plt.title("v4")
                plt.show()

            # 打印目标函数各项的值
            print(self.v1+factor*self.v2, "(", self.v1, "+", factor*self.v2,  "+", self.factor_sm*self.v3, "+", self.factor_CUQI1*self.v4, ")")

            # 返回目标函数总值
            return self.v1+factor*self.v2+self.factor_sm*self.v3+self.factor_CUQI1*self.v4

        def obj_scipy_grad(self, x):
            """计算目标函数的梯度"""
            # 将向量x转换为FEniCS函数
            x_fun = Function(self.myeit.H_sigma)
            x_fun.vector().set_local(x)
            # 获取各项的梯度
            g1 = self.g1
            g2 = self.tv_reg.grad_reg(x_fun).get_local()
            self.g2 = g2
            g3 = self.g3
            g4 = self.reg_CUQI1_obj.evaluate_grad_external(x)

            factor = self.factor
            factor_sm = self.factor_sm
            factor_CUQI1 = self.factor_CUQI1

            # 绘图标志（默认关闭）
            plot_flag = False
            if self.counter % 20 == 0 and plot_flag:
                # 可视化各项梯度
                g1_fenics = Function(self.myeit.H_sigma)
                g1_fenics.vector()[:] = g1.flatten()
                g2_fenics = Function(self.myeit.H_sigma)
                g2_fenics.vector()[:] = g2.flatten()
                g3_fenics = Function(self.myeit.H_sigma)
                g3_fenics.vector()[:] = g3.flatten()
                g_fenics = Function(self.myeit.H_sigma)
                g_fenics.vector()[:] = g1.flatten()+factor*g2.flatten()+factor_sm*g3.flatten()
                plt.figure()
                im = plot(g1_fenics)
                plt.colorbar(im)
                plt.title("grad 1")
                plt.show()
                plt.figure()
                im = plot(g2_fenics)
                plt.colorbar(im)
                plt.title("grad 2")
                plt.show()
                plt.figure()
                im = plot(g3_fenics)
                plt.colorbar(im)
                plt.title("grad 3")
                plt.figure()
                im = plot(g_fenics)
                plt.colorbar(im)
                plt.title("grad (g1 + factor*g2 + factor_sm*g3)")
                plt.show()

            # 返回目标函数的总梯度
            return g1.flatten()+factor*g2.flatten()+factor_sm*g3.flatten()+factor_CUQI1*g4.flatten()

    #%% 初始化和优化（NL_main_2版本）
    # 创建初始猜测x0
    recon_background_flag = False

    if recon_background_flag:
        # 从文件加载之前的重建结果
        recon_KTC = io.loadmat(recon_KTC_file)["reconstruction"]
        recon_KTC_float = np.zeros_like(recon_KTC)
        recon_KTC_float[:] = recon_KTC
        # 将分割值映射到电导率值
        recon_KTC_float[recon_KTC_float == 0] = background_conductivity
        recon_KTC_float[recon_KTC_float == 1] = low_conductivity
        recon_KTC_float[recon_KTC_float == 2] = high_conductivity

        im = plt.imshow(np.flipud(recon_KTC_float))
        plt.title('KTC reconstruction, flipped for interpolation')
        plt.colorbar(im)

        # 应用高斯平滑
        recon_KTC_float_smoothed = gaussian_filter(recon_KTC_float, sigma=30)
        plt.figure()
        im = plt.imshow(recon_KTC_float_smoothed)
        plt.colorbar(im)

        # 插值到有限元函数空间
        x0_exp = Inclusion(np.fliplr(recon_KTC_float_smoothed.T), radius, degree=1)
        x0_fun = interpolate(x0_exp, myeit.H_sigma)
        plot(x0_fun)
        x0 = x0_fun.vector().get_local()
    else:
        # 使用背景电导率作为初始猜测
        x0 = 0.8 * np.ones(myeit.H_sigma.dim())

    # 创建TV正则化对象
    tv_reg = TV_reg(myeit.H_sigma, None, 1, 1e-4)
    # 创建复合目标函数对象，使用传入的正则化因子
    target_scipy_TV = Target_scipy_TV( myeit, tv_reg, reg_CUQI1_obj, smprior=smprior, Imatr=Imatr, Uel_data=Uel_data, factor=TV_factor, factor_sm=Tikhonov_factor, factor_CUQI1=CUQI1_factor)


    #%% 运行优化
    # 导入时间模块
    import time
    # 记录开始时间
    start = time.time()
    # 设置变量的边界约束
    bounds = [(1e-5,100)]*myeit.H_sigma.dim()
    # 使用L-BFGS-B优化算法求解
    res = minimize(target_scipy_TV.obj_scipy, x0, method='L-BFGS-B', jac=target_scipy_TV.obj_scipy_grad, options={'disp': True, 'maxiter':niter} , bounds=bounds)
    # 记录结束时间
    end = time.time()
    print("time elapsed: ", end-start)
    print("time elapsed in minutes: ", (end-start)/60)

    # 保存各项目标函数值的列表
    v1_list = np.array(target_scipy_TV.list_v1)
    v2_list = np.array(target_scipy_TV.list_v2)
    v3_list = np.array(target_scipy_TV.list_v3)
    np.savez(output_dir_name+"/v_list.npz", v1_list=v1_list, v2_list=v2_list, v3_list=v3_list)

    #%% 将结果投影到像素网格
    # 将优化结果转换为FEniCS函数
    res_fenics = Function(myeit.H_sigma)
    res_fenics.vector().set_local( res['x'])

    # 绘图标志（默认关闭）
    plot_flag = False
    if plot_flag:
        plt.figure()
        im = plot(res_fenics)
        plt.colorbar(im)

    #%% 投影到像素网格
    # 创建256x256的像素网格
    X, Y = np.meshgrid(np.linspace(-radius,radius,256),np.linspace(-radius,radius,256) )
    Z = np.zeros_like(X)

    # 在网格上进行插值：计算有限元解在每个像素位置的值
    for i in range(256):
        for j in range(256):
            try:
                Z[i,j] = res_fenics(X[i,j], Y[i,j])
            except:
                # 插值失败时使用背景电导率值
                Z[i,j] = background_conductivity

    #%% 翻转结果以匹配标准坐标系
    deltareco_pixgrid = np.flipud(Z)

    # 返回重建的电导率分布
    return deltareco_pixgrid
     