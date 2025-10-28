#%%
# 导入命令行参数解析库
import argparse
# 导入数值计算库
import numpy as np
# 导入科学计算库
import scipy as sp
# 导入KTC竞赛相关的前向模型库
import KTCFwd
# 导入KTC竞赛相关的网格处理库
import KTCMeshing
# 导入KTC竞赛相关的绘图库
import KTCPlotting
# 导入KTC竞赛相关的评分库
import KTCScoring
# 导入KTC竞赛相关的辅助函数库
import KTCAux
# 导入绘图库
import matplotlib.pyplot as plt
# 导入文件路径匹配库
import glob
# 导入Chan-Vese图像分割算法
from skimage.segmentation import chan_vese
# 导入EIT非线性主函数
from EITLib import NL_main
# 导入CUQI正则化相关的平滑先验
from EITLib.KTCRegularization_NLOpt import SMPrior
# 导入分割评分函数和Otsu分割算法
from EITLib.segmentation import scoring_function, otsu
# 导入操作系统接口库
import os

#%%
def main():
    """主函数：执行EIT图像重建和分割"""

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    # 添加输入文件夹路径参数
    parser.add_argument("inputFolder")
    # 添加输出文件夹路径参数
    parser.add_argument("outputFolder")
    # 添加类别编号参数（难度级别，整数类型）
    parser.add_argument("categoryNbr", type=int)
    # 添加迭代次数参数（整数类型）
    parser.add_argument("niter", type=int)

    # 添加TV（全变分）正则化因子参数（浮点类型）
    parser.add_argument("TV_factor", type=float)
    # 添加Tikhonov正则化因子参数（浮点类型）
    parser.add_argument("Tikhonov_factor", type=float)
    # 添加CUQI1正则化因子参数（浮点类型）
    parser.add_argument("CUQI1_factor", type=float)
    # 添加分割方法参数（字符串类型）
    parser.add_argument("segmentation_method", type=str)

    # 解析命令行参数
    args = parser.parse_args()

    # 获取输入文件夹路径
    inputFolder = args.inputFolder
    # 获取输出文件夹路径
    outputFolder = args.outputFolder
    # 获取类别编号（难度级别）
    categoryNbr = args.categoryNbr

    # 获取迭代次数
    niter = args.niter

    # 获取TV正则化因子
    TV_factor = args.TV_factor
    # 获取Tikhonov正则化因子
    Tikhonov_factor = args.Tikhonov_factor
    # 获取CUQI1正则化因子
    CUQI1_factor = args.CUQI1_factor

    # 获取分割方法类型
    segmentation_method = args.segmentation_method

    # 如果输出文件夹不存在，则创建它
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    # 设置电极数量为32个
    Nel = 32  # number of electrodes
    # 设置接触阻抗（每个电极的接触阻抗都是1e-6）
    z = (1e-6) * np.ones((Nel, 1))  # contact impedances
    # 从输入文件夹加载参考数据（背景测量数据）
    mat_dict = sp.io.loadmat(inputFolder + '/ref.mat') #load the reference data
    # 提取参考电流注入模式
    Injref = mat_dict["Injref"] #current injections
    # 提取参考电压测量值（水箱测量值）
    Uelref = mat_dict["Uelref"] #measured voltages from water chamber
    # 提取电压测量模式
    Mpat = mat_dict["Mpat"] #voltage measurement pattern
    # 创建测量包含标志矩阵（哪些测量值包含在反演中）
    vincl = np.ones(((Nel - 1),76), dtype=bool) #which measurements to include in the inversion
    # 根据难度级别确定要移除的电极索引
    rmind = np.arange(0,2 * (categoryNbr - 1),1) #electrodes whose data is removed

    # 根据难度级别移除测量数据
    #remove measurements according to the difficulty level
    for ii in range(0,75):
        for jj in rmind:
            # 如果该电极在当前注入模式中被使用，则移除整个注入模式的测量
            if Injref[jj,ii]:
                vincl[:,ii] = 0
            # 移除该电极对应的所有测量
            vincl[jj,:] = 0

    # 将测量包含标志矩阵转置并展平为一维数组
    vincl = vincl.T.flatten()
    #recon = NL_main(Uelref, Uelref, Mpat, categoryNbr)

    # 获取输入文件夹中所有.mat数据文件的列表并排序
    # Get a list of .mat files in the input folder
    mat_files = sorted(glob.glob(inputFolder + '/data*.mat'))
    # 遍历每个输入文件，对每个文件进行图像重建
    for objectno in range (0,len(mat_files)): #compute the reconstruction for each input file
        # 加载当前数据文件
        mat_dict2 = sp.io.loadmat(mat_files[objectno])
        # 提取电流注入模式
        Inj = mat_dict2["Inj"]
        # 提取电压测量值
        Uel = mat_dict2["Uel"]
        # 提取电压测量模式
        Mpat = mat_dict2["Mpat"]
        # 计算电压差（目标测量值 - 参考测量值）
        deltaU = Uel - Uelref
        #############################  Changed code

        # 调用非线性主函数进行图像重建，返回像素网格上的电导率变化
        deltareco_pixgrid = NL_main.NL_main_2(Uel, Uelref, Inj, categoryNbr, niter=niter, output_dir_name=outputFolder, TV_factor=TV_factor, Tikhonov_factor=Tikhonov_factor, CUQI1_factor=CUQI1_factor)

        # 保存重建结果到.npz文件
        # save deltareco_pixgrid
        np.savez(outputFolder + '/' + str(objectno + 1) + '.npz', deltareco_pixgrid=deltareco_pixgrid)

        # 根据选择的分割方法进行图像分割
        if segmentation_method == 'otsu':
            # 使用Otsu阈值分割方法
            deltareco_pixgrid_segmented = otsu(deltareco_pixgrid)
        elif segmentation_method == 'cv':
            # 使用Chan-Vese水平集分割方法
            deltareco_pixgrid_segmented = KTCScoring.cv_NLOpt(deltareco_pixgrid, log_par=1.5, linear_par=1, exp_par=0)

        ###################################  End of changed code
        # 将分割后的结果作为最终重建结果
        reconstruction = deltareco_pixgrid_segmented
        # 创建包含重建结果的字典
        mdic = {"reconstruction": reconstruction}
        # 打印保存路径
        print(outputFolder + '/' + str(objectno + 1) + '.mat')
        # 保存重建结果到.mat文件
        sp.io.savemat( outputFolder + '/' + str(objectno + 1) + '.mat',mdic)

        # 将重建结果保存为PNG图像
        # save reconstruction as png
        plt.imshow(reconstruction)
        # 从文件中读取真实的幻影图像
        # read real phantom from file
        phantom = sp.io.loadmat('GroundTruths/true'+str(objectno+1)+'.mat')['truth']
        # 添加标题，显示类别编号和分割评分
        # add title that shows the category number and score
        plt.title('Category ' + str(categoryNbr) + ', score = ' + str(scoring_function(reconstruction, phantom)))
        # 保存图像到输出文件夹
        plt.savefig(outputFolder + '/' + str(objectno + 1) + '.png')

# 主程序入口：当脚本作为主程序运行时，调用main函数
if __name__ == "__main__":
    main()
