"""
比较一阶网格和二阶网格的区别及作用
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import KTCMeshing

# 加载网格数据
mat_dict_mesh = sp.io.loadmat('Mesh_sparse.mat')

print("=" * 60)
print("一阶网格（粗网格）信息：")
print("=" * 60)

# 一阶网格
g = mat_dict_mesh['g']  # 节点坐标
H = mat_dict_mesh['H']  # 三角形单元
elfaces = mat_dict_mesh['elfaces'][0].tolist()

print(f"节点数量: {len(g)}")
print(f"三角形单元数量: {len(H)}")
print(f"每个三角形的节点数: {H.shape[1]} (3个顶点 - 一阶单元)")
print(f"总自由度: {len(g)}")

# 构建一阶网格结构
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

print("\n" + "=" * 60)
print("二阶网格（细网格）信息：")
print("=" * 60)

# 二阶网格
H2 = mat_dict_mesh['H2']
g2 = mat_dict_mesh['g2']
elfaces2 = mat_dict_mesh['elfaces2'][0].tolist()

print(f"节点数量: {len(g2)}")
print(f"三角形单元数量: {len(H2)}")
print(f"每个三角形的节点数: {H2.shape[1]} (3个顶点 + 3个边中点 = 6节点 - 二阶单元)")
print(f"总自由度: {len(g2)}")

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

Mesh2 = KTCMeshing.Mesh(H2, g2, elfaces2, nodes2, elements2)

print("\n" + "=" * 60)
print("网格对比：")
print("=" * 60)
print(f"节点数比值: {len(g2) / len(g):.2f}x")
print(f"单元数比值: {len(H2) / len(H):.2f}x (相同，只是每个单元的节点更多)")

print("\n" + "=" * 60)
print("在代码中的使用：")
print("=" * 60)
print("一阶网格 (Mesh)  -> 用于最终的图像重建和插值")
print("                   -> 重建结果 deltareco 定义在一阶网格的节点上")
print("                   -> 正则化先验 SMPrior 定义在一阶网格上")
print()
print("二阶网格 (Mesh2) -> 用于正向求解器 EITFEM")
print("                   -> 更精确地计算电场分布和雅可比矩阵")
print("                   -> 计算灵敏度矩阵 J (main.py:134)")

# 可视化对比
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 第一行：一阶网格
ax1 = axes[0, 0]
ax1.triplot(Mesh.g[:, 0], Mesh.g[:, 1], Mesh.H, 'k-', linewidth=0.3)
ax1.plot(Mesh.g[:, 0], Mesh.g[:, 1], 'r.', markersize=2)
ax1.set_aspect('equal')
ax1.set_title(f'1st Order Mesh - Full View\n{len(Mesh.g)} nodes, {len(Mesh.H)} elements')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# 一阶网格局部放大
ax2 = axes[0, 1]
ax2.triplot(Mesh.g[:, 0], Mesh.g[:, 1], Mesh.H, 'k-', linewidth=0.8)
ax2.plot(Mesh.g[:, 0], Mesh.g[:, 1], 'ro', markersize=5)
ax2.set_xlim([-0.15, 0.15])
ax2.set_ylim([-0.15, 0.15])
ax2.set_aspect('equal')
ax2.set_title('1st Order - Zoomed\n(3 nodes per triangle)')
ax2.grid(True, alpha=0.3)

# 一阶单元示意图
ax3 = axes[0, 2]
# 选择一个三角形单元展示
tri_idx = 500
tri_nodes = Mesh.H[tri_idx]
tri_coords = Mesh.g[tri_nodes]
# 绘制三角形
tri_plot = plt.Polygon(tri_coords, fill=False, edgecolor='k', linewidth=2)
ax3.add_patch(tri_plot)
ax3.plot(tri_coords[:, 0], tri_coords[:, 1], 'ro', markersize=15, label='Vertex nodes')
# 标注节点
for i, coord in enumerate(tri_coords):
    ax3.text(coord[0], coord[1], f'{i+1}', fontsize=12, ha='center', va='center')
ax3.set_xlim([tri_coords[:, 0].min()-0.01, tri_coords[:, 0].max()+0.01])
ax3.set_ylim([tri_coords[:, 1].min()-0.01, tri_coords[:, 1].max()+0.01])
ax3.set_aspect('equal')
ax3.set_title('1st Order Element\n(Linear interpolation)')
ax3.legend()

# 第二行：二阶网格（只使用前3个节点绘制三角形轮廓）
ax4 = axes[1, 0]
# 二阶网格的H2包含6个节点，但triplot只需要3个顶点，所以取前3列
H2_vertices = Mesh2.H[:, :3]
ax4.triplot(Mesh2.g[:, 0], Mesh2.g[:, 1], H2_vertices, 'b-', linewidth=0.3)
ax4.plot(Mesh2.g[:, 0], Mesh2.g[:, 1], 'g.', markersize=2)
ax4.set_aspect('equal')
ax4.set_title(f'2nd Order Mesh - Full View\n{len(Mesh2.g)} nodes, {len(Mesh2.H)} elements')
ax4.set_xlabel('x')
ax4.set_ylabel('y')

# 二阶网格局部放大
ax5 = axes[1, 1]
ax5.triplot(Mesh2.g[:, 0], Mesh2.g[:, 1], H2_vertices, 'b-', linewidth=0.8)
ax5.plot(Mesh2.g[:, 0], Mesh2.g[:, 1], 'go', markersize=5)
ax5.set_xlim([-0.15, 0.15])
ax5.set_ylim([-0.15, 0.15])
ax5.set_aspect('equal')
ax5.set_title('2nd Order - Zoomed\n(6 nodes per triangle)')
ax5.grid(True, alpha=0.3)

# 二阶单元示意图
ax6 = axes[1, 2]
# 选择同一个三角形单元
tri_nodes2 = Mesh2.H[tri_idx]
tri_coords2 = Mesh2.g[tri_nodes2]
# 绘制三角形（6个节点：3个顶点+3个边中点）
tri_plot2 = plt.Polygon(tri_coords2[:3], fill=False, edgecolor='b', linewidth=2)
ax6.add_patch(tri_plot2)
# 顶点节点
ax6.plot(tri_coords2[:3, 0], tri_coords2[:3, 1], 'ro', markersize=15, label='Vertex nodes')
# 边中点节点
ax6.plot(tri_coords2[3:6, 0], tri_coords2[3:6, 1], 'bs', markersize=12, label='Edge midpoint nodes')
# 标注节点
for i, coord in enumerate(tri_coords2):
    ax6.text(coord[0], coord[1], f'{i+1}', fontsize=10, ha='center', va='center', color='white')
ax6.set_xlim([tri_coords2[:3, 0].min()-0.01, tri_coords2[:3, 0].max()+0.01])
ax6.set_ylim([tri_coords2[:3, 1].min()-0.01, tri_coords2[:3, 1].max()+0.01])
ax6.set_aspect('equal')
ax6.set_title('2nd Order Element\n(Quadratic interpolation)')
ax6.legend()

plt.tight_layout()
plt.savefig('mesh_order_comparison.png', dpi=150, bbox_inches='tight')
print(f"\n可视化结果已保存到 mesh_order_comparison.png")

# 展示一个具体单元的节点分布
print("\n" + "=" * 60)
print(f"示例三角形单元 #{tri_idx}：")
print("=" * 60)
print("\n一阶单元节点坐标（3个顶点）：")
for i, coord in enumerate(tri_coords):
    print(f"  节点 {i+1}: ({coord[0]:.6f}, {coord[1]:.6f})")

print("\n二阶单元节点坐标（3个顶点 + 3个边中点）：")
for i, coord in enumerate(tri_coords2):
    node_type = "顶点" if i < 3 else "边中点"
    print(f"  节点 {i+1} ({node_type}): ({coord[0]:.6f}, {coord[1]:.6f})")

plt.close('all')
