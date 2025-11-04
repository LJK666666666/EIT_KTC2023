"""
生成仿真数据并保存到SimData文件夹

生成10000条level1（不移除电极）的EIT仿真数据
"""

import os
import sys
import numpy as np
import scipy as sp
from tqdm import tqdm
import time
import multiprocessing as mp
from functools import partial

# 添加ktc_methods到系统路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ktc_methods'))

import KTCFwd
import KTCMeshing
import KTCAux

from sim_dataset import SimulatedEITDataset, load_mesh


def create_output_directories(base_path):
    """创建输出文件夹"""
    level1_path = os.path.join(base_path, "level_1")

    # 创建子文件夹
    gt_path = os.path.join(level1_path, "gt")
    measurements_path = os.path.join(level1_path, "measurements")

    os.makedirs(gt_path, exist_ok=True)
    os.makedirs(measurements_path, exist_ok=True)

    print(f"Created directories:")
    print(f"  {gt_path}")
    print(f"  {measurements_path}")

    return level1_path, gt_path, measurements_path


def get_existing_samples(gt_path, measurements_path):
    """获取已有样本的数量和最大索引"""
    if not os.path.exists(gt_path) or not os.path.exists(measurements_path):
        return 0, -1

    gt_files = [f for f in os.listdir(gt_path) if f.endswith('.npy')]
    meas_files = [f for f in os.listdir(measurements_path) if f.endswith('.npy')]

    if len(gt_files) == 0 or len(meas_files) == 0:
        return 0, -1

    # 提取索引
    gt_indices = [int(f.split('_')[1].split('.')[0]) for f in gt_files]
    meas_indices = [int(f.split('_')[1].split('.')[0]) for f in meas_files]

    # 检查一致性
    if set(gt_indices) != set(meas_indices):
        print("WARNING: gt and measurements files are not consistent!")
        print(f"  GT files: {len(gt_files)}, Measurements files: {len(meas_files)}")

    max_idx = max(max(gt_indices), max(meas_indices))
    num_samples = len(set(gt_indices) & set(meas_indices))

    return num_samples, max_idx


def generate_single_sample_worker(args):
    """
    生成单个样本的工作函数（用于多进程）
    每个进程独立创建dataset实例

    Args:
        args: (sample_idx, save_idx, mesh_name, noise_std1, noise_std2, segments, gt_path, meas_path)
    """
    sample_idx, save_idx, mesh_name, noise_std1, noise_std2, segments, gt_path, meas_path = args

    # 每个进程创建自己的dataset实例
    dataset = SimulatedEITDataset(
        length=sample_idx + 1,  # 只需要能访问到这个索引即可
        mesh_name=mesh_name,
        noise_std1=noise_std1,
        noise_std2=noise_std2,
        segments=segments,
        use_evaluation_pattern=True
    )

    # 生成样本
    phantom_pix, measurements = dataset[sample_idx]
    phantom_np = phantom_pix.numpy()
    measurements_np = measurements.numpy()

    # 保存文件
    gt_filename = os.path.join(gt_path, f"gt_{save_idx:05d}.npy")
    np.save(gt_filename, phantom_np)

    meas_filename = os.path.join(meas_path, f"u_{save_idx:05d}.npy")
    np.save(meas_filename, measurements_np)

    return save_idx


def generate_and_save_data(num_samples=10000,
                          mesh_name="Mesh_dense.mat",
                          noise_std1=0.1,
                          noise_std2=0,
                          segments=3,
                          clear_existing=False,
                          num_workers=None):
    """
    生成并保存仿真数据

    Args:
        num_samples: 生成的样本数量
        mesh_name: 使用的网格文件
        noise_std1: 噪声标准差（测量值的百分比）
        noise_std2: 第二噪声分量
        segments: 分割类别数量
        clear_existing: 是否清除已有数据重新开始
        num_workers: 并行进程数量（None表示自动，留3-4个CPU核心闲置）
    """

    # 设置基础路径
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SimData")

    print("="*70)
    print("Generating Simulated EIT Data")
    print("="*70)
    print(f"\nParameters:")
    print(f"  Number of samples: {num_samples}")
    print(f"  Mesh: {mesh_name}")
    print(f"  Noise std1: {noise_std1}")
    print(f"  Noise std2: {noise_std2}")
    print(f"  Segments: {segments}")
    print(f"  Level: 1 (no electrode removal)")

    # 创建输出文件夹
    level1_path, gt_path, measurements_path = create_output_directories(base_path)

    # 检查已有数据
    existing_samples, max_idx = get_existing_samples(gt_path, measurements_path)

    if clear_existing and existing_samples > 0:
        import shutil
        print(f"\nClearing existing data ({existing_samples} samples)...")
        shutil.rmtree(gt_path)
        shutil.rmtree(measurements_path)
        os.makedirs(gt_path, exist_ok=True)
        os.makedirs(measurements_path, exist_ok=True)
        existing_samples = 0
        max_idx = -1
        print("  ✓ Cleared")

    start_idx = max_idx + 1
    samples_to_generate = num_samples

    if existing_samples > 0:
        print(f"\nFound {existing_samples} existing samples (max index: {max_idx})")
        print(f"Will continue from index {start_idx}")
        print(f"Generating {samples_to_generate} new samples...")
    else:
        print(f"\nNo existing data found, starting from index 0")

    # 确定并行进程数
    total_cpus = mp.cpu_count()
    if num_workers is None:
        # 留3-4个CPU核心闲置
        num_workers = max(1, total_cpus - 4)
    num_workers = min(num_workers, total_cpus)

    print(f"\nParallel processing:")
    print(f"  Total CPUs: {total_cpus}")
    print(f"  Using workers: {num_workers}")
    print(f"  Reserved CPUs: {total_cpus - num_workers}")

    # 创建数据集实例
    print("\nInitializing dataset...")
    dataset = SimulatedEITDataset(
        length=samples_to_generate,
        mesh_name=mesh_name,
        noise_std1=noise_std1,
        noise_std2=noise_std2,
        segments=segments,
        use_evaluation_pattern=True  # 使用评估数据的测量模式
    )

    print(f"Dataset initialized:")
    print(f"  Mesh nodes: {len(dataset.mesh.g)}")
    print(f"  Number of electrodes: {dataset.Nel}")
    print(f"  Number of injection patterns: {dataset.Inj.shape[1]}")
    print(f"  Measurement dimensions: {dataset.Inj.shape[1] * (dataset.Nel - 1)}")

    # 生成并保存数据
    print(f"\nGenerating {samples_to_generate} samples...")
    print("-"*70)

    start_time = time.time()

    if num_workers == 1:
        # 单进程模式
        for i in tqdm(range(samples_to_generate), desc="Generating samples"):
            phantom_pix, measurements = dataset[i]

            save_idx = start_idx + i
            phantom_np = phantom_pix.numpy()
            measurements_np = measurements.numpy()

            gt_filename = os.path.join(gt_path, f"gt_{save_idx:05d}.npy")
            np.save(gt_filename, phantom_np)

            meas_filename = os.path.join(measurements_path, f"u_{save_idx:05d}.npy")
            np.save(meas_filename, measurements_np)
    else:
        # 多进程并行模式
        print(f"Using multiprocessing with {num_workers} workers...")

        # 准备参数列表
        args_list = [
            (i, start_idx + i, mesh_name, noise_std1, noise_std2, segments, gt_path, measurements_path)
            for i in range(samples_to_generate)
        ]

        # 使用进程池并行生成
        with mp.Pool(processes=num_workers) as pool:
            # 使用imap_unordered提高效率，配合tqdm显示进度
            list(tqdm(
                pool.imap_unordered(generate_single_sample_worker, args_list),
                total=samples_to_generate,
                desc="Generating samples"
            ))

    elapsed_time = time.time() - start_time

    print("-"*70)
    print(f"\nData generation completed!")
    print(f"  Total samples: {samples_to_generate}")
    print(f"  Index range: {start_idx} - {start_idx + samples_to_generate - 1}")
    print(f"  Total time: {elapsed_time:.2f} seconds")
    print(f"  Average time per sample: {elapsed_time/samples_to_generate:.4f} seconds")
    print(f"\nData saved to: {level1_path}")
    print(f"  Ground truth: {gt_path}")
    print(f"  Measurements: {measurements_path}")

    # 验证保存的数据
    print("\nVerifying saved data...")
    verify_saved_data(gt_path, measurements_path, start_idx, start_idx + samples_to_generate - 1)

    print("\n" + "="*70)
    print("All done!")
    print("="*70)


def verify_saved_data(gt_path, measurements_path, start_idx, end_idx):
    """验证保存的数据"""

    # 检查文件数量
    gt_files = [f for f in os.listdir(gt_path) if f.endswith('.npy')]
    meas_files = [f for f in os.listdir(measurements_path) if f.endswith('.npy')]

    print(f"  Ground truth files: {len(gt_files)}")
    print(f"  Measurement files: {len(meas_files)}")

    expected_samples = end_idx - start_idx + 1
    if len(gt_files) < expected_samples or len(meas_files) < expected_samples:
        print(f"  WARNING: Expected at least {expected_samples} new files!")
    else:
        print(f"  ✓ File count verified")

    # 加载并检查最后保存的样本
    if len(gt_files) > 0 and len(meas_files) > 0:
        last_idx = end_idx
        gt_sample = np.load(os.path.join(gt_path, f"gt_{last_idx:05d}.npy"))
        meas_sample = np.load(os.path.join(measurements_path, f"u_{last_idx:05d}.npy"))

        print(f"\nLast sample data shapes (index {last_idx}):")
        print(f"  Ground truth: {gt_sample.shape}")
        print(f"  Measurements: {meas_sample.shape}")

        print(f"\nGround truth statistics:")
        print(f"  Unique values: {np.unique(gt_sample)}")
        print(f"  Min: {gt_sample.min()}, Max: {gt_sample.max()}")

        print(f"\nMeasurements statistics:")
        print(f"  Mean: {meas_sample.mean():.6f}")
        print(f"  Std: {meas_sample.std():.6f}")
        print(f"  Min: {meas_sample.min():.6f}")
        print(f"  Max: {meas_sample.max():.6f}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate simulated EIT data')
    parser.add_argument('--num_samples', type=int, default=10000,
                       help='Number of samples to generate (default: 10000)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: generate only 10 samples')
    parser.add_argument('--clear', action='store_true',
                       help='Clear existing data and start from scratch')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto, leaves 4 CPUs free)')

    args = parser.parse_args()

    num_samples = 10 if args.test else args.num_samples

    # 生成level1数据
    generate_and_save_data(
        num_samples=num_samples,
        mesh_name="Mesh_dense.mat",
        noise_std1=0.1,  # 1% 噪声
        noise_std2=0,
        segments=3,
        clear_existing=args.clear,
        num_workers=args.workers
    )


if __name__ == "__main__":
    main()
