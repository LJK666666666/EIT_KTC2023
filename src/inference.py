"""
统一推理脚本
支持所有重建方法的推理
"""
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import numpy as np
import sys
from datetime import datetime
import re

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import ConfigManager, EITDataModule, EITEvaluator
from src.methods import create_method
from src.utils import get_logger, plot_reconstruction, save_mat


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='EIT Reconstruction Inference')

    # 方法相关
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['cnn', 'diffusion', 'traditional', 'deepdbar', 'fno', 'dbar'],
        help='Reconstruction method to use'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: src/configs/{method}_config.yaml)'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to model checkpoint (not required for traditional method)'
    )

    # 数据相关
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Path to data directory'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default='test',
        choices=['test', 'test2017', 'test2023'],
        help='Dataset to use for inference'
    )

    # 输出相关
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory name (or absolute base path) for results (saved as {base}_{num}). Default: if checkpoint provided, save under checkpoint folder/{dataset}; otherwise results/inference_{method}_{dataset}_{num}'
    )

    parser.add_argument(
        '--save_mat',
        action='store_true',
        help='Save reconstructions as .mat files'
    )

    parser.add_argument(
        '--save_images',
        action='store_true',
        default=True,
        help='Save reconstructions as .png images'
    )

    # 其他
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for inference'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda/cpu)'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 加载配置
    if args.config is None:
        args.config = f'src/configs/{args.method}_config.yaml'

    print(f"Loading config from: {args.config}")
    config = ConfigManager.load_config(args.config)

    # 更新配置
    config['data']['data_dir'] = args.data_dir
    config['data']['batch_size'] = args.batch_size
    config['training']['device'] = args.device
    config['method_name'] = args.method

    # ??????
    if args.output_dir is None:
        if args.checkpoint:
            base_dir = Path(args.checkpoint).parent / args.dataset
        else:
            base_dir = Path(f'results/inference_{args.method}_{args.dataset}')
    else:
        req_base = Path(args.output_dir)
        base_dir = req_base if req_base.is_absolute() else Path('results') / args.output_dir

    parent = base_dir.parent
    base_name = base_dir.name
    max_idx = -1
    if parent.exists():
        pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{2}})$")
        for item in parent.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    max_idx = max(max_idx, int(match.group(1)))
    output_dir = parent / f"{base_name}_{max_idx + 1:02d}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建日志记录器
    logger = get_logger('EIT_Inference', log_dir=str(output_dir))
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Inference method: {args.method}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Dataset: {args.dataset}")

    # 创建数据模块
    logger.info("Setting up data module...")
    # 为推理禁用 num_workers，确保样本按顺序处理，便于文件名匹配
    config['data']['num_workers'] = 0
    data_module = EITDataModule(config['data'])
    data_module.setup('test')

    # 选择数据集
    if args.dataset == 'test':
        dataloader = data_module.test_dataloader()
    elif args.dataset == 'test2017':
        dataloader = data_module.test2017_dataloader()
    elif args.dataset == 'test2023':
        dataloader = data_module.test2023_dataloader()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if dataloader is None:
        logger.error(f"Dataset {args.dataset} not found!")
        return

    logger.info(f"Dataset size: {len(dataloader.dataset)}")

    # 创建重建方法
    logger.info(f"Creating {args.method} method...")
    method = create_method(args.method, config)

    # 加载检查点
    if args.checkpoint and args.method.lower() not in ['traditional', 'dbar']:
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        method.load_checkpoint(args.checkpoint)
        method.model.eval() if method.model else None
    elif args.method.lower() == 'traditional':
        logger.info("Traditional method does not require checkpoint")
    elif args.method.lower() == 'dbar':
        logger.info("D-bar method does not require checkpoint (non-learning method)")
    else:
        logger.warning("No checkpoint provided, using randomly initialized model")

    # 创建评估器
    evaluator = EITEvaluator()

    # 推理
    logger.info("Starting inference...")

    # 检查是否是 Traditional 方法
    is_traditional = args.method.lower() == 'traditional'

    if is_traditional:
        # Traditional 方法特殊处理：直接调用一次 inference，它会处理所有文件
        logger.info("Traditional 方法：直接从 .mat 文件加载并重建")

        # 调用一次 inference（会忽略 measurements 参数）
        dummy_batch = next(iter(dataloader))
        dummy_measurements, _ = dummy_batch
        reconstructions = method.inference(dummy_measurements.to(args.device))

        # 获取 ground truth 文件夹
        category = config.get('ktc', {}).get('category', 1)
        gt_folder = Path(__file__).parent.parent / 'EvaluationData_full' / 'GroundTruths' / f'level_{category}'

        # 保存每个重建结果
        reconstructions_np = reconstructions.cpu().numpy()
        all_metrics = []

        for i in range(reconstructions_np.shape[0]):
            recon = reconstructions_np[i, 0]  # [H, W]
            file_id = i + 1  # 文件编号从 1 开始

            logger.info(f"Processing reconstruction {file_id}/{reconstructions_np.shape[0]}")

            # 保存 .mat 文件
            if args.save_mat:
                mat_path = output_dir / f'reconstruction_{file_id}.mat'
                save_mat({'reconstruction': recon}, str(mat_path))

            # 加载对应的 ground truth
            gt_path = gt_folder / f'{file_id}_true.mat'
            ground_truth = None

            if gt_path.exists():
                import scipy.io as sio
                try:
                    gt_data = sio.loadmat(str(gt_path))
                    # 尝试不同的键名
                    for key in ['truth', 'groundtruth', 'gt', 'reconstruction']:
                        if key in gt_data:
                            ground_truth = np.squeeze(gt_data[key])
                            break

                    if ground_truth is not None:
                        logger.info(f"Loaded ground truth from {gt_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load ground truth from {gt_path}: {e}")
            else:
                logger.warning(f"Ground truth not found: {gt_path}")

            # 保存图像
            if args.save_images:
                img_path = output_dir / f'reconstruction_{file_id}.png'
                if ground_truth is not None:
                    plot_reconstruction(recon, ground_truth, save_path=str(img_path))

                    # 计算评估指标
                    recon_tensor = torch.from_numpy(recon[np.newaxis, np.newaxis, :, :]).float().to(args.device)
                    gt_tensor = torch.from_numpy(ground_truth[np.newaxis, np.newaxis, :, :]).float().to(args.device)
                    metrics = evaluator.compute_all_metrics(recon_tensor, gt_tensor)
                    all_metrics.append(metrics)
                else:
                    plot_reconstruction(recon, save_path=str(img_path))
    else:
        # 标准推理流程（其他方法）
        all_metrics = []
        with torch.no_grad():
            for idx, batch in enumerate(tqdm(dataloader, desc="Inference")):
                measurements, target = batch
                measurements = measurements.to(args.device)

                # 推理
                reconstruction = method.inference(measurements)

                # 转换为 numpy
                reconstruction_np = reconstruction.cpu().numpy()
                measurements_np = measurements.cpu().numpy()

                # 保存结果
                for i in range(reconstruction.shape[0]):
                    sample_idx = idx * args.batch_size + i
                    recon = reconstruction_np[i, 0]  # [H, W]

                    # 如果重建结果尺寸与 ground truth 不同，调整尺寸
                    if target is not None:
                        target_size = target.shape[-1]  # 假设 target 是方形的
                        if recon.shape[0] != target_size:
                            from PIL import Image
                            recon = np.array(Image.fromarray(recon).resize(
                                (target_size, target_size), Image.BILINEAR
                            ))

                    # 获取原始文件名（不包括扩展名）
                    # 例如：从 "0_1.npz" 或 "0_1.mat" 提取 "0_1"
                    original_file_path = dataloader.dataset.data_files[sample_idx]
                    file_stem = original_file_path.stem  # 去掉扩展名
                    filename_without_ext = file_stem  # 例如 "0_1"

                    logger.info(f"Processing: {original_file_path.name} -> reconstruction_{filename_without_ext}")

                    # 保存 .mat 文件
                    if args.save_mat:
                        mat_path = output_dir / f'reconstruction_{filename_without_ext}.mat'
                        save_mat({'reconstruction': recon}, str(mat_path))

                    # 保存图像
                    if args.save_images:
                        # 如果有真实值，绘制对比图
                        if target is not None:
                            target_np = target.cpu().numpy()
                            ground_truth = target_np[i, 0]
                            img_path = output_dir / f'reconstruction_{filename_without_ext}.png'
                            plot_reconstruction(recon, ground_truth, save_path=str(img_path))

                            # 计算评估指标（使用调整尺寸后的 recon）
                            recon_tensor = torch.from_numpy(recon[np.newaxis, np.newaxis, :, :]).float().to(args.device)
                            metrics = evaluator.compute_all_metrics(
                                recon_tensor,
                                target[i:i+1].to(args.device)
                            )
                            all_metrics.append(metrics)
                        else:
                            img_path = output_dir / f'reconstruction_{filename_without_ext}.png'
                            plot_reconstruction(recon, save_path=str(img_path))

    # 汇总评估指标
    if all_metrics:
        avg_metrics = evaluator.aggregate_metrics(all_metrics)
        logger.info("Average metrics:")
        for name, value in avg_metrics.items():
            logger.info(f"  {name}: {value:.6f}")

        # 保存评估指标
        import json
        metrics_path = output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(avg_metrics, f, indent=2)

    logger.info(f"Inference completed! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
