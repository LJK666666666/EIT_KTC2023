"""
可视化EIT重建网格与像素网格的区别
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import KTCMeshing
import KTCAux

# 加载有限元网格
mat_dict_mesh = sp.io.loadmat('Mesh_sparse.mat')
g = mat_dict_mesh['g']  # 节点坐标
H = mat_dict_mesh['H']  # 三角形单元节点索引
elfaces = mat_dict_mesh['elfaces'][0].tolist()

# 构建网格结构
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

Mesh = KTCMeshing.Mesh(H, g, elfaces, nodes, elements)

print(f"重建网格信息：")
print(f"  节点数量: {len(Mesh.g)}")
print(f"  三角形单元数量: {len(Mesh.H)}")
print(f"  节点坐标示例 (前5个):")
for i in range(min(5, len(Mesh.g))):
    print(f"    节点 {i}: ({Mesh.g[i][0]:.4f}, {Mesh.g[i][1]:.4f})")

# 创建一个示例重建结果（模拟一个简单的分布）
center_x, center_y = 0.0, 0.0
radius = 0.3
deltareco = np.zeros((len(Mesh.g), 1))
for i in range(len(Mesh.g)):
    dist = np.sqrt((Mesh.g[i][0] - center_x)**2 + (Mesh.g[i][1] - center_y)**2)
    if dist < radius:
        deltareco[i] = 1.0 - dist/radius  # 中心高，边缘低

# 插值到像素网格
deltareco_pixgrid = KTCAux.interpolateRecoToPixGrid(deltareco, Mesh)

print(f"\n像素网格信息：")
print(f"  像素网格形状: {deltareco_pixgrid.shape}")
print(f"  像素总数: {deltareco_pixgrid.size}")

# 创建可视化对比图
fig = plt.figure(figsize=(15, 5))

# 子图1: 有限元网格结构（显示三角形网格）
ax1 = fig.add_subplot(131)
ax1.triplot(Mesh.g[:, 0], Mesh.g[:, 1], Mesh.H, 'k-', linewidth=0.3)
ax1.plot(Mesh.g[:, 0], Mesh.g[:, 1], 'r.', markersize=2)
ax1.set_aspect('equal')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Reconstruction Mesh\n(Irregular FEM triangular mesh)')

# 子图2: 重建网格上的结果（使用三角剖分着色）
ax2 = fig.add_subplot(132)
tcf = ax2.tripcolor(Mesh.g[:, 0], Mesh.g[:, 1], Mesh.H,
                     deltareco.flatten(), shading='flat', cmap='jet')
ax2.set_aspect('equal')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Result on FEM Mesh\n(Triangular elements)')
plt.colorbar(tcf, ax=ax2)

# 子图3: 插值后的像素网格结果
ax3 = fig.add_subplot(133)
im = ax3.imshow(deltareco_pixgrid, cmap='jet', extent=[-1, 1, -1, 1], origin='lower')
ax3.set_aspect('equal')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Result on Pixel Grid\n(Regular rectangular pixels)')
plt.colorbar(im, ax=ax3)

plt.tight_layout()
plt.savefig('grid_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n可视化结果已保存到 grid_comparison.png")

# 额外展示网格细节对比
fig2, (ax4, ax5) = plt.subplots(1, 2, figsize=(12, 5))

# 局部放大：有限元网格
ax4.triplot(Mesh.g[:, 0], Mesh.g[:, 1], Mesh.H, 'k-', linewidth=0.5)
ax4.plot(Mesh.g[:, 0], Mesh.g[:, 1], 'r.', markersize=3)
ax4.set_xlim([-0.3, 0.3])
ax4.set_ylim([-0.3, 0.3])
ax4.set_aspect('equal')
ax4.grid(True, alpha=0.3)
ax4.set_title('FEM Mesh (zoomed)\nIrregular triangular elements')

# 像素网格的等效表示
y_coords = np.linspace(-1, 1, deltareco_pixgrid.shape[0])
x_coords = np.linspace(-1, 1, deltareco_pixgrid.shape[1])
pixel_size_y = y_coords[1] - y_coords[0]
pixel_size_x = x_coords[1] - x_coords[0]

# 只在局部区域绘制像素网格线
zoom_range = 10  # 中心区域的像素数
center_i = deltareco_pixgrid.shape[0] // 2
center_j = deltareco_pixgrid.shape[1] // 2

for i in range(center_i - zoom_range, center_i + zoom_range + 1):
    y = -1 + i * pixel_size_y
    ax5.axhline(y, color='k', linewidth=0.3)
for j in range(center_j - zoom_range, center_j + zoom_range + 1):
    x = -1 + j * pixel_size_x
    ax5.axvline(x, color='k', linewidth=0.3)

ax5.imshow(deltareco_pixgrid, cmap='jet', extent=[-1, 1, -1, 1], origin='lower', alpha=0.5)
ax5.set_xlim([-0.3, 0.3])
ax5.set_ylim([-0.3, 0.3])
ax5.set_aspect('equal')
ax5.set_title('Pixel Grid (zoomed)\nRegular rectangular pixels')

plt.tight_layout()
plt.savefig('grid_detail_comparison.png', dpi=150, bbox_inches='tight')
print(f"网格细节对比已保存到 grid_detail_comparison.png")

plt.close('all')
