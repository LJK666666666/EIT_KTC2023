"""
示例：如何使用SimulatedEITDataset

这个脚本展示了如何使用新创建的仿真数据集来训练模型或进行实验
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from sim_dataset import SimulatedEITDataset


def visualize_samples(dataset, num_samples=4):
    """可视化数据集中的多个样本"""
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 3))

    for i in range(num_samples):
        phantom, measurements = dataset[i]

        # 左侧：显示导电率分布
        axes[i, 0].imshow(phantom.numpy().T, origin='lower', cmap='viridis')
        axes[i, 0].set_title(f'Sample {i+1}: Phantom')
        axes[i, 0].axis('off')

        # 右侧：显示测量数据
        axes[i, 1].plot(measurements.numpy())
        axes[i, 1].set_title(f'Sample {i+1}: Measurements')
        axes[i, 1].set_xlabel('Measurement index')
        axes[i, 1].set_ylabel('Voltage difference')
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def test_dataloader(dataset, batch_size=4):
    """测试DataLoader功能"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"DataLoader created with batch_size={batch_size}")
    print(f"Number of batches: {len(dataloader)}")

    # 获取第一个batch
    phantoms, measurements = next(iter(dataloader))

    print(f"\nFirst batch:")
    print(f"  Phantoms shape: {phantoms.shape}")
    print(f"  Measurements shape: {measurements.shape}")
    print(f"  Phantoms dtype: {phantoms.dtype}")
    print(f"  Measurements dtype: {measurements.dtype}")

    return phantoms, measurements


def analyze_dataset_statistics(dataset, num_samples=100):
    """分析数据集的统计特性"""
    print(f"\nAnalyzing {num_samples} samples...")

    measurements_list = []
    phantom_stats = []

    for i in range(min(num_samples, len(dataset))):
        phantom, measurements = dataset[i]
        measurements_list.append(measurements.numpy())

        # 统计每个样本中的类别分布
        unique, counts = torch.unique(phantom, return_counts=True)
        phantom_stats.append({
            'unique_classes': unique.tolist(),
            'class_counts': counts.tolist()
        })

    measurements_array = np.array(measurements_list)

    print(f"\nMeasurements statistics:")
    print(f"  Mean: {measurements_array.mean():.6f}")
    print(f"  Std: {measurements_array.std():.6f}")
    print(f"  Min: {measurements_array.min():.6f}")
    print(f"  Max: {measurements_array.max():.6f}")

    print(f"\nPhantom class distribution (first 5 samples):")
    for i, stats in enumerate(phantom_stats[:5]):
        print(f"  Sample {i+1}: classes={stats['unique_classes']}, "
              f"counts={stats['class_counts']}")


def main():
    """主函数"""
    print("="*70)
    print("SimulatedEITDataset Example")
    print("="*70)

    # 1. 创建数据集
    print("\n1. Creating dataset...")
    dataset = SimulatedEITDataset(
        length=20,
        mesh_name="Mesh_dense.mat",
        noise_std1=0.1,  # 1% 噪声
        noise_std2=0,
        segments=3
    )
    print(f"   Dataset created with {len(dataset)} samples")
    print(f"   Mesh has {len(dataset.mesh.g)} nodes")
    print(f"   Number of electrodes: {dataset.Nel}")

    # 2. 可视化样本
    print("\n2. Visualizing samples...")
    fig = visualize_samples(dataset, num_samples=4)
    fig.savefig('example_samples.png', dpi=150, bbox_inches='tight')
    print("   Saved to 'example_samples.png'")

    # 3. 测试DataLoader
    print("\n3. Testing DataLoader...")
    phantoms_batch, measurements_batch = test_dataloader(dataset, batch_size=4)

    # 4. 分析数据集统计
    print("\n4. Analyzing dataset statistics...")
    analyze_dataset_statistics(dataset, num_samples=20)

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70)


if __name__ == "__main__":
    main()
