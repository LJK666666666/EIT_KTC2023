import numpy as np
import scipy as sp
import os
from pathlib import Path
import KTCFwd  # EIT正向求解器模块
import KTCMeshing  # 网格生成与处理模块
import KTCRegularization  # 正则化先验模块
import KTCPlotting  # 绘图工具模块
import KTCScoring  # 评分与分割算法模块
import KTCAux  # 辅助工具函数模块
import matplotlib.pyplot as plt
import glob
import csv

def main(inputFolder, outputFolder, categoryNbr):
    """
    EIT图像重建主函数

    参数:
        inputFolder: 输入数据文件夹路径（包含ref.mat和data*.mat文件）
        outputFolder: 输出结果文件夹路径
        categoryNbr: 难度类别编号（用于确定移除哪些电极的数据）
    """
    # ========== 第一部分：加载参考数据和设置测量参数 ==========
    Nel = 32  # 电极数量
    z = (1e-6) * np.ones((Nel, 1))  # 接触阻抗（极小值，近似理想接触）
    # ensure output folder exists
    os.makedirs(outputFolder, exist_ok=True)

    # load reference data (use os.path.join to avoid relative path issues)
    mat_dict = sp.io.loadmat(os.path.join(inputFolder, 'ref.mat'))  # 加载参考数据
    Injref = mat_dict["Injref"]  # 参考电流注入模式
    Uelref = mat_dict["Uelref"]  # 水箱参考电压测量值（均匀介质）
    Mpat = mat_dict["Mpat"]  # 电压测量模式
    vincl = np.ones(((Nel - 1), 76), dtype=bool)  # 标记哪些测量值用于反演（初始全部包含）
    rmind = np.arange(0, 2 * (categoryNbr - 1), 1)  # 根据难度级别确定要移除数据的电极索引

    # 根据难度级别移除指定电极的测量数据
    for ii in range(0, 75):
        for jj in rmind:
            if Injref[jj, ii]:  # 如果该电极在当前注入模式中被使用
                vincl[:, ii] = 0  # 移除该注入模式的所有测量
            vincl[jj, :] = 0  # 移除该电极的所有测量数据

    # ========== 第二部分：加载有限元网格 ==========
    # 加载预先生成的稀疏网格用于图像重建
    mat_dict_mesh = sp.io.loadmat('Mesh_sparse.mat')
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

    # 创建一阶和二阶网格对象
    Mesh = KTCMeshing.Mesh(H, g, elfaces, nodes, elements)
    Mesh2 = KTCMeshing.Mesh(H2, g2, elfaces2, nodes2, elements2)

    # print(f'Nodes in inversion 1st order mesh: {len(Mesh.g)}')

    # ========== 第三部分：设置正则化先验和正向求解器 ==========
    sigma0 = np.ones((len(Mesh.g), 1))  # 线性化点（初始电导率猜测，均匀分布）
    corrlength = 1 * 0.115  # 先验中的相关长度（控制平滑程度）
    var_sigma = 0.05 ** 2  # 先验方差
    mean_sigma = sigma0  # 先验均值
    smprior = KTCRegularization.SMPrior(Mesh.g, corrlength, var_sigma, mean_sigma)

    # 为反演设置正向求解器
    solver = KTCFwd.EITFEM(Mesh2, Injref, Mpat, vincl)

    vincl = vincl.T.flatten()  # 将测量包含掩码展平为一维数组

    # 设置反演的噪声模型
    noise_std1 = 0.05  # 第一噪声分量的标准差（相对于每个电压测量值）
    noise_std2 = 0.01  # 第二噪声分量的标准差（相对于最大电压测量值）
    solver.SetInvGamma(noise_std1, noise_std2, Uelref)  # 计算噪声精度矩阵

    # ========== 第四部分：对每个输入文件进行重建 ==========
    # 获取输入文件夹中所有.mat数据文件的列表
    mat_files = glob.glob(os.path.join(inputFolder, 'data*.mat'))
    
    # 初始化评分结果列表
    scoring_results = []
    
    for objectno in range(0, len(mat_files)):  # 对每个输入文件计算重建
        mat_dict2 = sp.io.loadmat(mat_files[objectno])
        Inj = mat_dict2["Inj"]  # 电流注入数据
        Uel = mat_dict2["Uel"]  # 电压测量数据
        Mpat = mat_dict2["Mpat"]  # 测量模式
        deltaU = Uel - Uelref  # 计算电压差（相对于参考测量）

        Usim = solver.SolveForward(sigma0, z)  # 在线性化点进行正向求解
        J = solver.Jacobian(sigma0, z)  # 计算雅可比矩阵（灵敏度矩阵）
        #Jz = solver.Jacobianz(sigma0, z)  # 接触阻抗雅可比矩阵 - 简单重建算法未使用

        # 求解正则化反问题以重建电导率变化
        mask = np.array(vincl, bool)  # 创建布尔掩码
        # 求解：(J^T * InvGamma * J + L^T * L) * delta_sigma = J^T * InvGamma * deltaU
        deltareco = np.linalg.solve(J.T @ solver.InvGamma_n[np.ix_(mask, mask)] @ J + smprior.L.T @ smprior.L,
                                     J.T @ solver.InvGamma_n[np.ix_(mask, mask)] @ deltaU[vincl])
        #sgplot = KTCPlotting.SigmaPlotter(Mesh, [5], 'jet')
        # sgplot.basic2Dplot(deltareco, [], ['linear difference reconstruction'])

        # ========== 第五部分：插值、分割和保存结果 ==========
        # 将重建结果插值到规则的像素网格
        deltareco_pixgrid = KTCAux.interpolateRecoToPixGrid(deltareco, Mesh)
        # fig, ax = plt.subplots()
        # cax = ax.imshow(deltareco_pixgrid, cmap="jet")
        # plt.colorbar(cax)
        # plt.axis('image')

        # 使用Otsu2方法对图像直方图进行三类阈值分割
        level, x = KTCScoring.Otsu2(deltareco_pixgrid.flatten(), 256, 7)

        deltareco_pixgrid_segmented = np.zeros_like(deltareco_pixgrid)

        # 根据阈值将像素分为三类
        ind0 = deltareco_pixgrid < x[level[0]]  # 低电导率区域
        ind1 = np.logical_and(deltareco_pixgrid >= x[level[0]], deltareco_pixgrid <= x[level[1]])  # 中等电导率区域
        ind2 = deltareco_pixgrid > x[level[1]]  # 高电导率区域
        inds = [np.count_nonzero(ind0), np.count_nonzero(ind1), np.count_nonzero(ind2)]
        bgclass = inds.index(max(inds))  # 确定背景类别（像素数最多的类别）

        # 根据背景类别重新标记分割图像（背景为0，包含物为1和2）
        match bgclass:
            case 0:  # 如果背景是低电导率区域
                deltareco_pixgrid_segmented[ind1] = 2
                deltareco_pixgrid_segmented[ind2] = 2
            case 1:  # 如果背景是中等电导率区域
                deltareco_pixgrid_segmented[ind0] = 1
                deltareco_pixgrid_segmented[ind2] = 2
            case 2:  # 如果背景是高电导率区域
                deltareco_pixgrid_segmented[ind0] = 1
                deltareco_pixgrid_segmented[ind1] = 1

        # fig, ax = plt.subplots()
        # cax = ax.imshow(deltareco_pixgrid_segmented, cmap='gray')
        # plt.colorbar(cax)
        # plt.axis('image')
        # plt.title('segmented linear difference reconstruction')

        # fig, ax = plt.subplots()
        # cax = ax.imshow(deltareco_pixgrid, cmap='gray')
        # plt.colorbar(cax)
        # plt.axis('image')

        # 保存重建结果到输出文件夹
        reconstruction = deltareco_pixgrid_segmented
        mdic = {"reconstruction": reconstruction}
        outpath = os.path.join(outputFolder, f"{objectno + 1}.mat")
        print(outpath)
        sp.io.savemat(outpath, mdic)

        # 加载ground truth图像
        groundtruth_folder = Path(__file__).resolve().parents[1] / 'EvaluationData' / 'GroundTruths' / f'level_{categoryNbr}'
        groundtruth_path = groundtruth_folder / f"{objectno + 1}_true.mat"

        if groundtruth_path.exists():
            groundtruth_data = sp.io.loadmat(groundtruth_path)
            groundtruth = groundtruth_data['truth']

            # 计算评分
            score = KTCScoring.scoringFunction(groundtruth, deltareco_pixgrid_segmented)
            
            # 记录评分结果
            scoring_results.append({
                'file_id': objectno + 1,
                'score': score,
                'groundtruth_path': str(groundtruth_path),
                'reconstruction_path': outpath
            })
            
            print(f"File {objectno + 1}: Score = {score:.4f}")

            # 保存对比图像：ground truth, 原始重建, 分割后重建
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Ground Truth
            cax0 = axes[0].imshow(groundtruth, cmap='gray')
            axes[0].axis('image')
            axes[0].axis('off')
            axes[0].set_title(f'truth (Score: {score:.4f})')
            plt.colorbar(cax0, ax=axes[0])

            # 原始重建结果
            cax1 = axes[1].imshow(deltareco_pixgrid, cmap='jet')
            axes[1].axis('image')
            axes[1].axis('off')
            axes[1].set_title('reconstruction_original')
            plt.colorbar(cax1, ax=axes[1])

            # 分割后的重建结果
            cax2 = axes[2].imshow(deltareco_pixgrid_segmented, cmap='gray')
            axes[2].axis('image')
            axes[2].axis('off')
            axes[2].set_title('reconstruction_segmented')
            plt.colorbar(cax2, ax=axes[2])
        else:
            # 如果没有ground truth，只显示重建结果
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # 原始重建结果
            cax1 = axes[0].imshow(deltareco_pixgrid, cmap='jet')
            axes[0].axis('image')
            axes[0].axis('off')
            axes[0].set_title('reconstruction_original')
            plt.colorbar(cax1, ax=axes[0])

            # 分割后的重建结果
            cax2 = axes[1].imshow(deltareco_pixgrid_segmented, cmap='gray')
            axes[1].axis('image')
            axes[1].axis('off')
            axes[1].set_title('reconstruction_segmented')
            plt.colorbar(cax2, ax=axes[1])
            
            print(f"File {objectno + 1}: No ground truth found, skipping scoring")

        imgpath = os.path.join(outputFolder, f"{objectno + 1}.png")
        plt.savefig(imgpath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved reconstruction image: {imgpath}")
    
    # ========== 第六部分：保存评分结果到CSV文件 ==========
    if scoring_results:
        csv_path = os.path.join(outputFolder, 'scoring_results.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['file_id', 'score', 'groundtruth_path', 'reconstruction_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in scoring_results:
                writer.writerow(result)
        
        # 计算统计信息
        scores = [result['score'] for result in scoring_results]
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        print(f"\n=== 评分统计结果 ===")
        print(f"总文件数: {len(scoring_results)}")
        print(f"平均评分: {avg_score:.4f}")
        print(f"评分标准差: {std_score:.4f}")
        print(f"评分范围: {min(scores):.4f} - {max(scores):.4f}")
        print(f"评分结果已保存至: {csv_path}")


if __name__ == '__main__':
    # repo_data = Path(__file__).resolve().parents[1] / 'EvaluationData' / 'evaluation_datasets' / 'level1'
    # main(str(repo_data), 'results/level1', 1)

    repo_root = Path(__file__).resolve().parents[1]
    for i in range(3, 8):
        repo_data = repo_root / 'EvaluationData' / 'evaluation_datasets' / f'level{i}'
        out_dir = repo_root / 'Codes_Python' / 'results' / f'level{i}'
        main(str(repo_data), str(out_dir), i)
