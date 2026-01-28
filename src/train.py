"""
统一训练脚本
支持所有重建方法的训练
"""
import argparse
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core import ConfigManager, EITDataModule, UnifiedTrainer
from src.methods import create_method
from src.utils import get_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='EIT Reconstruction Training')

    # 方法相关
    parser.add_argument(
        '--method',
        type=str,
        required=True,
        choices=['cnn', 'diffusion', 'traditional', 'deepdbar', 'fno'],
        help='Reconstruction method to use'
    )

    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (default: src/configs/{method}_config.yaml)'
    )

    # 数据相关
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Path to data directory'
    )

    # 训练相关
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, overrides config)'
    )

    # 结果保存
    parser.add_argument(
        '--result_dir',
        type=str,
        default=None,
        help='Custom name for results directory (will be saved as results/{name}_{num})'
    )

    # 其他
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
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

    # 命令行参数覆盖配置
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir

    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs

    if args.batch_size:
        config['data']['batch_size'] = args.batch_size

    if args.lr:
        config['optimizer']['lr'] = args.lr

    if args.device:
        config['training']['device'] = args.device

    # 添加方法名称到训练配置
    config['method_name'] = args.method
    config['training']['method_name'] = args.method

    # Result directory
    if args.result_dir:
        # Custom base name
        import re

        def _next_indexed_dir(base_dir_str: str) -> str:
            base_dir = Path(base_dir_str)
            parent = base_dir.parent
            base_name = base_dir.name
            max_idx = -1
            if parent.exists():
                pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{3}})$")
                for item in parent.iterdir():
                    if item.is_dir():
                        match = pattern.match(item.name)
                        if match:
                            max_idx = max(max_idx, int(match.group(1)))
            return str(parent / f"{base_name}_{max_idx + 1:02d}")

        # Resume uses original directory when base name matches
        if args.resume:
            resume_path = Path(args.resume)
            resume_dir = resume_path.parent
            resume_dir_name = resume_dir.name
            resume_base_name = re.sub(r'_\d{3}$', '', resume_dir_name)

            if args.result_dir == resume_base_name or args.result_dir == resume_dir_name:
                result_dir = str(resume_dir)
                print(f"Resuming: use original dir {result_dir}")
            else:
                result_dir = _next_indexed_dir(f"results/{args.result_dir}")
                print(f"Resuming: create new dir {result_dir}")
        else:
            result_dir = _next_indexed_dir(f"results/{args.result_dir}")
    else:
        # Default to method name
        result_dir = None  # UnifiedTrainer generates

    logger = get_logger('EIT_Training', log_dir=result_dir)
    logger.info(f"Training method: {args.method}")
    logger.info(f"Config: {config}")

    # 创建数据模块
    logger.info("Setting up data module...")
    data_module = EITDataModule(config['data'])
    data_module.setup('fit')

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    logger.info(f"Train dataset size: {len(data_module.train_dataset)}")
    logger.info(f"Val dataset size: {len(data_module.val_dataset)}")

    # 创建重建方法
    logger.info(f"Creating {args.method} method...")
    method = create_method(args.method, config)

    # 创建训练器
    logger.info("Creating trainer...")
    trainer = UnifiedTrainer(
        method=method,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        result_dir=result_dir,
        full_config=config
    )

    # 如果需要恢复训练
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        trainer.resume_from_checkpoint(args.resume)

    # 开始训练
    logger.info("Starting training...")
    try:
        history = trainer.train()
        logger.info("Training completed successfully!")

        # 保存最终训练历史
        logger.info(f"Results saved to: {trainer.result_dir}")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == '__main__':
    main()
