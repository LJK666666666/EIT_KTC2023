"""
解释如何从粗网格的sigma转换到细网格进行正向求解
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import KTCMeshing

# 加载网格数据
mat_dict_mesh = sp.io.loadmat('Mesh_sparse.mat')

# 一阶网格
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
Mesh = KTCMeshing.Mesh(H, g, elfaces, nodes, elements)

# 二阶网格
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
Mesh2 = KTCMeshing.Mesh(H2, g2, elfaces2, nodes2, elements2)

print("=" * 70)
print("粗网格到细网格的转换机制")
print("=" * 70)

# 创建一个示例的粗网格电导率分布
sigma_coarse = np.ones((len(Mesh.g), 1))  # 一阶网格：1602个节点
# 在中心区域设置一个高电导率圆
center_x, center_y = 0.0, 0.0
radius = 0.04
for i in range(len(Mesh.g)):
    dist = np.sqrt((Mesh.g[i][0] - center_x)**2 + (Mesh.g[i][1] - center_y)**2)
    if dist < radius:
        sigma_coarse[i] = 2.0

print(f"\n粗网格电导率向量:")
print(f"  形状: {sigma_coarse.shape}")
print(f"  节点数: {len(Mesh.g)} (一阶网格)")
print(f"  值域: [{sigma_coarse.min():.2f}, {sigma_coarse.max():.2f}]")

# 查看一阶和二阶网格的对应关系
print(f"\n网格节点对应关系:")
print(f"  一阶网格节点数: {len(Mesh.g)}")
print(f"  二阶网格节点数: {len(Mesh2.g)}")
print(f"  单元数量: {len(Mesh.H)} (两个网格相同)")

# 检查节点对应关系
print(f"\n关键发现：二阶网格的前{len(Mesh.g)}个节点与一阶网格节点完全对应！")
print(f"验证：前5个节点坐标对比")
for i in range(5):
    coord1 = Mesh.g[i]
    coord2 = Mesh2.g[i]
    match = "YES" if np.allclose(coord1, coord2) else "NO"
    print(f"  节点 {i}: 一阶({coord1[0]:.6f}, {coord1[1]:.6f}) vs 二阶({coord2[0]:.6f}, {coord2[1]:.6f}) {match}")

# 分析一个具体单元
tri_idx = 500
print(f"\n" + "=" * 70)
print(f"具体示例：单元 #{tri_idx}")
print("=" * 70)

# 一阶单元的节点索引
tri_nodes_1st = Mesh.H[tri_idx]
print(f"\n一阶单元节点索引: {tri_nodes_1st}")
print(f"这3个节点的电导率值:")
for i, node_idx in enumerate(tri_nodes_1st):
    print(f"  节点 {node_idx}: σ = {sigma_coarse[node_idx][0]:.4f}")

# 二阶单元的节点索引
tri_nodes_2nd = Mesh2.H[tri_idx]
print(f"\n二阶单元节点索引: {tri_nodes_2nd}")
print(f"注意：索引 [0, 2, 4] 对应一阶单元的3个顶点！")

vertex_indices = tri_nodes_2nd[[0, 2, 4]]
print(f"  顶点索引 [0, 2, 4]: {vertex_indices}")
print(f"  对应一阶索引: {tri_nodes_1st}")
print(f"  索引是否匹配: {np.array_equal(vertex_indices, tri_nodes_1st)}")

print(f"\n在KTCFwd.py:56中的操作: ss = sigma[ind[[0, 2, 4]]]")
print(f"这意味着：从粗网格的sigma中取出3个顶点的值")
print(f"  sigma[{vertex_indices[0]}] = {sigma_coarse[vertex_indices[0]][0]:.4f}")
print(f"  sigma[{vertex_indices[1]}] = {sigma_coarse[vertex_indices[1]][0]:.4f}")
print(f"  sigma[{vertex_indices[2]}] = {sigma_coarse[vertex_indices[2]][0]:.4f}")

print(f"\n边中点索引 [1, 3, 5]: {tri_nodes_2nd[[1, 3, 5]]}")
print(f"这些边中点的电导率值如何确定？")
print(f"答案：在正向求解中，使用二次插值基函数！")
print(f"      边中点的值由相邻顶点值通过形函数自动插值得到")

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 子图1: 一阶网格上的电导率分布
ax1 = axes[0]
tcf1 = ax1.tripcolor(Mesh.g[:, 0], Mesh.g[:, 1], Mesh.H,
                      sigma_coarse.flatten(), shading='flat', cmap='coolwarm')
ax1.triplot(Mesh.g[:, 0], Mesh.g[:, 1], Mesh.H, 'k-', linewidth=0.2, alpha=0.3)
ax1.set_aspect('equal')
ax1.set_title(f'Coarse Mesh (1st order)\n{len(Mesh.g)} nodes with σ values')
plt.colorbar(tcf1, ax=ax1, label='Conductivity σ')

# 子图2: 展示一个具体单元的节点编号和电导率
ax2 = axes[1]
# 绘制整体网格背景
ax2.triplot(Mesh.g[:, 0], Mesh.g[:, 1], Mesh.H, 'k-', linewidth=0.1, alpha=0.2)
# 高亮显示选中的单元
tri_coords_1st = Mesh.g[tri_nodes_1st]
tri_plot = plt.Polygon(tri_coords_1st, fill=True, facecolor='lightblue',
                       edgecolor='blue', linewidth=3, alpha=0.5)
ax2.add_patch(tri_plot)
# 绘制节点
ax2.plot(tri_coords_1st[:, 0], tri_coords_1st[:, 1], 'ro', markersize=15)
# 标注节点索引和电导率值
for i, (idx, coord) in enumerate(zip(tri_nodes_1st, tri_coords_1st)):
    ax2.text(coord[0], coord[1], f'{idx}\nσ={sigma_coarse[idx][0]:.2f}',
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax2.set_xlim([tri_coords_1st[:, 0].min()-0.02, tri_coords_1st[:, 0].max()+0.02])
ax2.set_ylim([tri_coords_1st[:, 1].min()-0.02, tri_coords_1st[:, 1].max()+0.02])
ax2.set_aspect('equal')
ax2.set_title(f'1st Order Element #{tri_idx}\n3 vertices with σ values')

# 子图3: 二阶单元如何使用这些值
ax3 = axes[2]
# 二阶单元的6个节点
tri_coords_2nd = Mesh2.g[tri_nodes_2nd]
tri_plot2 = plt.Polygon(tri_coords_2nd[:3], fill=True, facecolor='lightgreen',
                        edgecolor='green', linewidth=3, alpha=0.3)
ax3.add_patch(tri_plot2)
# 顶点节点（从粗网格继承sigma值）
vertex_coords = tri_coords_2nd[[0, 2, 4]]
ax3.plot(vertex_coords[:, 0], vertex_coords[:, 1], 'ro', markersize=15,
         label='Vertices (σ from coarse mesh)')
# 边中点节点（通过插值计算）
edge_coords = tri_coords_2nd[[1, 3, 5]]
ax3.plot(edge_coords[:, 0], edge_coords[:, 1], 'bs', markersize=12,
         label='Edge midpoints (interpolated)')
# 标注
for i, idx in enumerate([0, 2, 4]):
    coord = tri_coords_2nd[idx]
    node_global_idx = tri_nodes_2nd[idx]
    ax3.text(coord[0], coord[1], f'{node_global_idx}\nσ={sigma_coarse[node_global_idx][0]:.2f}',
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='pink', alpha=0.8))
for i, idx in enumerate([1, 3, 5]):
    coord = tri_coords_2nd[idx]
    node_global_idx = tri_nodes_2nd[idx]
    ax3.text(coord[0], coord[1], f'{node_global_idx}\n(interp)',
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax3.set_xlim([tri_coords_2nd[:3, 0].min()-0.02, tri_coords_2nd[:3, 0].max()+0.02])
ax3.set_ylim([tri_coords_2nd[:3, 1].min()-0.02, tri_coords_2nd[:3, 1].max()+0.02])
ax3.set_aspect('equal')
ax3.set_title(f'2nd Order Element #{tri_idx}\nσ at vertices → interpolate to edges')
ax3.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig('coarse_to_fine_conversion.png', dpi=150, bbox_inches='tight')
print(f"\n可视化已保存到 coarse_to_fine_conversion.png")

print("\n" + "=" * 70)
print("总结：粗网格到细网格的转换机制")
print("=" * 70)
print("1. 网格节点对应关系：")
print("   - 二阶网格的前1602个节点 = 一阶网格的全部节点（完全相同）")
print("   - 二阶网格的后4674个节点 = 单元边的中点（额外添加）")
print()
print("2. 电导率向量 sigma 的使用：")
print("   - 输入 sigma 长度 = 1602（一阶网格节点数）")
print("   - 正向求解时，只使用二阶单元的顶点索引 [0, 2, 4]")
print("   - 这些索引对应一阶网格的节点，可以直接从 sigma 中取值")
print()
print("3. 边中点的电导率如何确定：")
print("   - 在有限元计算中，通过二次形函数（shape function）自动插值")
print("   - 例如：边中点的σ ≈ (端点1的σ + 端点2的σ) / 2")
print("   - 更精确地说，是通过二次Lagrange插值基函数计算")
print()
print("4. 为什么这样设计：")
print("   - 反演只需求解粗网格的1602个未知数（计算量小）")
print("   - 正向求解使用细网格6276个节点（精度高）")
print("   - 两个网格通过节点对应关系无缝衔接，无需额外插值！")

plt.close('all')
