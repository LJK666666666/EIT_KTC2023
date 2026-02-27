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
        choices=['test', 'test2017', 'test2023', 'ktc_full', 'ktc_eval'],
        help='Dataset to use for inference'
    )
    parser.add_argument(
        '--ktc_level',
        type=int,
        default=1,
        help='KTC challenge level for --dataset ktc_full/ktc_eval'
    )
    parser.add_argument(
        '--measurement_format',
        type=str,
        default=None,
        choices=['eim16', 'raw16x13', 'matrix32'],
        help='Measurement format override'
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

    parser.add_argument(
        '--sample_idx',
        type=int,
        default=None,
        help='Run inference on a single sample index in the selected dataset (supports fast debug)'
    )

    parser.add_argument(
        '--test_opt_physics',
        action='store_true',
        help='Enable test-time optimization with differentiable physics backend (learning methods)'
    )

    parser.add_argument(
        '--test_opt_backend',
        type=str,
        default='linearized_ktc',
        help='Physics backend type for test-time optimization'
    )
    parser.add_argument(
        '--test_opt_mode',
        type=str,
        default='pixel',
        choices=['pixel', 'contour_step'],
        help='Optimization mode: pixel (legacy) or contour_step (boundary-focused)'
    )

    parser.add_argument(
        '--test_opt_steps',
        type=int,
        default=20,
        help='Number of optimization steps per sample'
    )

    parser.add_argument(
        '--test_opt_lr',
        type=float,
        default=1e-2,
        help='Learning rate for test-time optimization'
    )

    parser.add_argument(
        '--test_opt_lambda_smooth',
        type=float,
        default=1e-4,
        help='Smoothness regularization weight for test-time optimization'
    )

    parser.add_argument(
        '--test_opt_lambda_anchor',
        type=float,
        default=5e-4,
        help='Anchor regularization weight to keep sigma close to each relinearization anchor'
    )

    parser.add_argument(
        '--test_opt_relinearize_every',
        type=int,
        default=20,
        help='Recompute linearized physics backend every N steps (0 to disable)'
    )

    parser.add_argument(
        '--test_opt_max_delta',
        type=float,
        default=0.25,
        help='Trust-region clamp radius around initial sigma (0 to disable)'
    )

    parser.add_argument(
        '--test_opt_lr_min_factor',
        type=float,
        default=0.1,
        help='Minimum LR factor for cosine annealing during optimization'
    )
    parser.add_argument(
        '--test_opt_seg_threshold',
        type=float,
        default=0.25,
        help='Threshold for ternary initialization from sigma prediction'
    )
    parser.add_argument(
        '--test_opt_bspline_grid',
        type=int,
        default=12,
        help='Control grid size for contour displacement field'
    )
    parser.add_argument(
        '--test_opt_tau_start',
        type=float,
        default=0.8,
        help='Start temperature for soft step'
    )
    parser.add_argument(
        '--test_opt_tau_end',
        type=float,
        default=0.35,
        help='End temperature for soft step'
    )
    parser.add_argument(
        '--test_opt_lambda_length',
        type=float,
        default=2e-3,
        help='Contour length regularization weight'
    )
    parser.add_argument(
        '--test_opt_lambda_area',
        type=float,
        default=5e-3,
        help='Area consistency regularization weight'
    )
    parser.add_argument(
        '--test_opt_lambda_anchor_shape',
        type=float,
        default=1e-3,
        help='Shape control-point anchor regularization weight'
    )
    parser.add_argument(
        '--test_opt_lambda_speckle',
        type=float,
        default=3e-3,
        help='Background speckle suppression weight'
    )
    parser.add_argument(
        '--test_opt_lambda_anchor_sigma',
        type=float,
        default=1e-3,
        help='Stage-B sigma anchor regularization weight'
    )
    parser.add_argument(
        '--test_opt_stage2_steps',
        type=int,
        default=60,
        help='Stage-B refinement steps for contour_step mode'
    )
    parser.add_argument(
        '--test_opt_min_component_ratio',
        type=float,
        default=0.0015,
        help='Minimum connected-component ratio for ternary cleanup'
    )
    parser.add_argument(
        '--test_opt_max_shift_px',
        type=float,
        default=8.0,
        help='Maximum contour displacement in pixels'
    )
    parser.add_argument(
        '--test_opt_value_bounds',
        type=str,
        default='-1.3,-0.2,-0.2,0.2,0.2,1.3',
        help='Class value bounds: neg_l,neg_u,bg_l,bg_u,pos_l,pos_u'
    )
    parser.add_argument(
        '--test_opt_save_ternary',
        action='store_true',
        default=True,
        help='Save ternary optimized map (.npy)'
    )
    parser.add_argument(
        '--test_opt_save_continuous',
        action='store_true',
        default=True,
        help='Save continuous optimized map (.npy)'
    )

    parser.add_argument(
        '--test_opt_save_curve',
        action='store_true',
        default=True,
        help='Save per-sample optimization loss curve'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    if args.sample_idx is not None and args.sample_idx < 0:
        raise ValueError(f"--sample_idx must be >= 0, got {args.sample_idx}")

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
    if args.measurement_format is not None:
        config['data']['measurement_format'] = args.measurement_format

    if args.test_opt_physics:
        if args.method.lower() in ['traditional', 'dbar']:
            raise ValueError("--test_opt_physics does not support non-learning methods: traditional/dbar")
        if args.checkpoint is None:
            default_checkpoint = Path('results') / f'{args.method}_01' / 'best_model.pth'
            if not default_checkpoint.exists():
                raise ValueError(
                    f"Default checkpoint not found: {default_checkpoint}. "
                    "Please pass --checkpoint explicitly."
                )
            args.checkpoint = str(default_checkpoint)

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
    logger.info(f"KTC level: {args.ktc_level}")
    logger.info(f"Measurement format: {config['data'].get('measurement_format', 'eim16')}")
    logger.info(f"Single sample index: {args.sample_idx}")
    logger.info(f"Test-time physics optimization: {args.test_opt_physics}")
    if args.test_opt_physics:
        logger.info(f"Physics backend: {args.test_opt_backend}")
        logger.info(
            f"Optimization params: steps={args.test_opt_steps}, lr={args.test_opt_lr}, "
            f"mode={args.test_opt_mode}, "
            f"lambda_smooth={args.test_opt_lambda_smooth}, "
            f"lambda_anchor={args.test_opt_lambda_anchor}, "
            f"relinearize_every={args.test_opt_relinearize_every}, "
            f"max_delta={args.test_opt_max_delta}, "
            f"lr_min_factor={args.test_opt_lr_min_factor}"
        )

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
    elif args.dataset == 'ktc_full':
        if config['data'].get('measurement_format') != 'matrix32':
            raise ValueError("--dataset ktc_full requires --measurement_format matrix32")
        dataloader = data_module.ktc_full_dataloader(level=args.ktc_level)
    elif args.dataset == 'ktc_eval':
        if config['data'].get('measurement_format') != 'matrix32':
            raise ValueError("--dataset ktc_eval requires --measurement_format matrix32")
        dataloader = data_module.ktc_eval_dataloader(level=args.ktc_level)
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

    if args.sample_idx is not None:
        if args.sample_idx >= len(dataloader.dataset):
            raise ValueError(
                f"--sample_idx out of range: {args.sample_idx}, dataset size: {len(dataloader.dataset)}"
            )
        if args.method.lower() == 'traditional':
            raise ValueError("--sample_idx is not supported for traditional method")

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
        if args.test_opt_physics:
            if config['data'].get('measurement_format', 'eim16') != 'eim16':
                raise ValueError("--test_opt_physics currently requires --measurement_format eim16")
            from src.methods.cnn.physics_backend import create_physics_backend
            from src.methods.cnn.test_time_opt import optimize_sigma_with_backend, optimize_sigma_contour_step, save_loss_curve

        if args.sample_idx is not None:
            sample_measurements, sample_target = dataloader.dataset[args.sample_idx]
            single_measurements = sample_measurements.unsqueeze(0).to(args.device)
            single_target = sample_target.unsqueeze(0) if sample_target is not None else None
            iter_batches = [(0, (single_measurements, single_target), args.sample_idx)]
        else:
            iter_batches = ((idx, batch, None) for idx, batch in enumerate(dataloader))

        for idx, batch, fixed_sample_idx in tqdm(iter_batches, desc="Inference"):
            measurements, target = batch
            measurements = measurements.to(args.device)

            # 初始推理（神经网络猜测）
            reconstruction_init = method.inference(measurements)
            if args.test_opt_physics:
                if reconstruction_init.dim() != 4 or reconstruction_init.shape[1] != 1:
                    raise ValueError(
                        f"--test_opt_physics expects method output shape [B,1,H,W], got {tuple(reconstruction_init.shape)}"
                    )

            # 保存结果
            for i in range(reconstruction_init.shape[0]):
                if fixed_sample_idx is not None:
                    sample_idx = fixed_sample_idx
                else:
                    sample_idx = idx * args.batch_size + i
                original_file_path = dataloader.dataset.data_files[sample_idx]
                file_stem = original_file_path.stem
                filename_without_ext = file_stem

                logger.info(f"Processing: {original_file_path.name} -> reconstruction_{filename_without_ext}")

                if args.test_opt_physics:
                    sigma_init = reconstruction_init[i:i+1]
                    meas_i = measurements[i:i+1]
                    try:
                        value_bounds = tuple(float(x.strip()) for x in str(args.test_opt_value_bounds).split(','))
                    except ValueError as e:
                        raise ValueError("--test_opt_value_bounds must be six comma-separated floats") from e
                    if len(value_bounds) != 6:
                        raise ValueError("--test_opt_value_bounds must contain 6 values")

                    backend = create_physics_backend(
                        args.test_opt_backend,
                        output_size=sigma_init.shape[-1],
                        device=str(sigma_init.device),
                        mean=dataloader.dataset.mean,
                        std=dataloader.dataset.std,
                        voltage=dataloader.dataset.voltage
                    )
                    if args.test_opt_mode == 'contour_step':
                        sigma_opt, sigma_ternary, loss_history = optimize_sigma_contour_step(
                            sigma_init=sigma_init,
                            measurements=meas_i,
                            backend=backend,
                            steps=args.test_opt_steps,
                            lr=args.test_opt_lr,
                            seg_threshold=args.test_opt_seg_threshold,
                            bspline_grid=args.test_opt_bspline_grid,
                            tau_start=args.test_opt_tau_start,
                            tau_end=args.test_opt_tau_end,
                            max_shift_px=args.test_opt_max_shift_px,
                            lambda_length=args.test_opt_lambda_length,
                            lambda_area=args.test_opt_lambda_area,
                            lambda_anchor_shape=args.test_opt_lambda_anchor_shape,
                            lambda_speckle=args.test_opt_lambda_speckle,
                            lambda_anchor_sigma=args.test_opt_lambda_anchor_sigma,
                            relinearize_every=args.test_opt_relinearize_every,
                            min_component_ratio=args.test_opt_min_component_ratio,
                            lr_min_factor=args.test_opt_lr_min_factor,
                            value_bounds=value_bounds,
                            stage2_steps=args.test_opt_stage2_steps
                        )
                    else:
                        sigma_opt, loss_history = optimize_sigma_with_backend(
                            sigma_init=sigma_init,
                            measurements=meas_i,
                            backend=backend,
                            steps=args.test_opt_steps,
                            lr=args.test_opt_lr,
                            lambda_smooth=args.test_opt_lambda_smooth,
                            lambda_anchor=args.test_opt_lambda_anchor,
                            relinearize_every=args.test_opt_relinearize_every,
                            max_delta=args.test_opt_max_delta,
                            lr_min_factor=args.test_opt_lr_min_factor
                        )
                        sigma_ternary = None
                    recon = sigma_opt[0, 0].detach().cpu().numpy()

                    if args.test_opt_save_curve:
                        curve_json = output_dir / f"loss_curve_{filename_without_ext}.json"
                        curve_png = output_dir / f"loss_curve_{filename_without_ext}.png"
                        save_loss_curve(loss_history, curve_json, curve_png)
                    if args.test_opt_save_continuous:
                        npy_path = output_dir / f"sigma_continuous_{filename_without_ext}.npy"
                        np.save(npy_path, recon.astype(np.float32))
                    if args.test_opt_save_ternary and sigma_ternary is not None:
                        ternary_path = output_dir / f"sigma_ternary_{filename_without_ext}.npy"
                        np.save(ternary_path, sigma_ternary[0, 0].detach().cpu().numpy().astype(np.float32))
                    nn_pred_for_plot = sigma_init[0, 0].detach().cpu().numpy()
                else:
                    recon = reconstruction_init[i, 0].detach().cpu().numpy()
                    nn_pred_for_plot = None

                # 如果重建结果尺寸与 ground truth 不同，调整尺寸
                if target is not None:
                    target_size = target.shape[-1]
                    if recon.shape[0] != target_size:
                        from PIL import Image
                        recon = np.array(Image.fromarray(recon).resize(
                            (target_size, target_size), Image.BILINEAR
                        ))
                    if nn_pred_for_plot is not None and nn_pred_for_plot.shape[0] != target_size:
                        from PIL import Image
                        nn_pred_for_plot = np.array(Image.fromarray(nn_pred_for_plot).resize(
                            (target_size, target_size), Image.BILINEAR
                        ))

                # 保存 .mat 文件
                if args.save_mat:
                    mat_path = output_dir / f'reconstruction_{filename_without_ext}.mat'
                    save_mat({'reconstruction': recon}, str(mat_path))

                # 保存图像
                if args.save_images:
                    if target is not None:
                        target_np = target.cpu().numpy()
                        ground_truth = target_np[i, 0]
                        img_path = output_dir / f'reconstruction_{filename_without_ext}.png'
                        if args.test_opt_physics:
                            plot_reconstruction(
                                recon,
                                ground_truth,
                                nn_prediction=nn_pred_for_plot,
                                save_path=str(img_path)
                            )
                        else:
                            plot_reconstruction(recon, ground_truth, save_path=str(img_path))

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
