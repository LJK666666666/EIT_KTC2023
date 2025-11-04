"""
训练EIT重建神经网络

使用仿真数据训练模型
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from datetime import datetime
import argparse
import sys

# 添加ktc_methods路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ktc_methods'))
from KTCScoring import scoringFunction

from model import get_model, SegmentationLoss


class SimDataLoader(Dataset):
    """从已保存的仿真数据加载"""

    def __init__(self, data_path='SimData/level_1', num_samples=None):
        self.data_path = data_path
        self.gt_path = os.path.join(data_path, 'gt')
        self.meas_path = os.path.join(data_path, 'measurements')

        # 统计可用样本数
        gt_files = sorted([f for f in os.listdir(self.gt_path) if f.endswith('.npy')])
        if num_samples is None:
            self.num_samples = len(gt_files)
        else:
            self.num_samples = min(num_samples, len(gt_files))

        print(f"Loaded {self.num_samples} samples from {data_path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        gt = np.load(os.path.join(self.gt_path, f'gt_{idx:05d}.npy'))
        measurements = np.load(os.path.join(self.meas_path, f'u_{idx:05d}.npy'))

        return torch.from_numpy(measurements).float(), torch.from_numpy(gt).float()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0

    pbar = tqdm(dataloader, desc='Training')
    for measurements, gt in pbar:
        measurements = measurements.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        # 前向传播
        output = model(measurements)

        # 计算损失
        loss, ce_loss, dice_loss = criterion(output, gt)

        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_dice_loss += dice_loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'dice': f'{dice_loss.item():.4f}'
        })

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_dice_loss = total_dice_loss / len(dataloader)

    return avg_loss, avg_ce_loss, avg_dice_loss


def validate_epoch(model, dataloader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_dice_loss = 0

    with torch.no_grad():
        for measurements, gt in tqdm(dataloader, desc='Validation'):
            measurements = measurements.to(device)
            gt = gt.to(device)

            output = model(measurements)
            loss, ce_loss, dice_loss = criterion(output, gt)

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_dice_loss += dice_loss.item()

    avg_loss = total_loss / len(dataloader)
    avg_ce_loss = total_ce_loss / len(dataloader)
    avg_dice_loss = total_dice_loss / len(dataloader)

    return avg_loss, avg_ce_loss, avg_dice_loss


def save_training_plots(history, save_dir):
    """保存训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Total loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # CE loss
    axes[0, 1].plot(history['train_ce_loss'], label='Train')
    axes[0, 1].plot(history['val_ce_loss'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Cross Entropy Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Dice loss
    axes[1, 0].plot(history['train_dice_loss'], label='Train')
    axes[1, 0].plot(history['val_dice_loss'], label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Dice Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Learning rate
    axes[1, 1].plot(history['learning_rate'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_yscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_prediction(gt, pred, score, save_path):
    """
    可视化预测结果和真实标签的对比

    Args:
        gt: Ground truth [256, 256]
        pred: Prediction [256, 256]
        score: Reconstruction score
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground truth
    im1 = axes[0].imshow(gt, cmap='gray', vmin=0, vmax=2)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    # Prediction
    im2 = axes[1].imshow(pred, cmap='gray', vmin=0, vmax=2)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Difference
    diff = np.abs(gt - pred)
    im3 = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=2)
    axes[2].set_title(f'Absolute Difference (Score: {score:.4f})')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_and_visualize(model, train_dataset, val_dataset, device, save_dir, num_samples=3):
    """
    从训练集和验证集中选取样本进行预测和可视化

    Args:
        model: 训练好的模型
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        device: 计算设备
        save_dir: 保存目录
        num_samples: 每个数据集选取的样本数
    """
    model.eval()

    # 选取样本索引
    train_indices = np.random.choice(len(train_dataset), min(num_samples, len(train_dataset)), replace=False)
    val_indices = np.random.choice(len(val_dataset), min(num_samples, len(val_dataset)), replace=False)

    # 用于存储评分结果
    scores_summary = {
        'train': [],
        'val': []
    }

    # 处理训练集样本
    print("\nEvaluating training samples...")
    for i, idx in enumerate(train_indices):
        measurements, gt = train_dataset[idx]
        measurements = measurements.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(measurements)

        # 转换为类别标签
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        gt = gt.cpu().numpy()

        # 计算评分
        score = scoringFunction(gt, pred)
        scores_summary['train'].append(score)

        # 可视化
        visualize_prediction(gt, pred, score,
                           os.path.join(save_dir, f'train_sample_{i+1}.png'))

        print(f"Train sample {i+1}: Score = {score:.4f}")

    # 处理验证集样本
    print("\nEvaluating validation samples...")
    for i, idx in enumerate(val_indices):
        measurements, gt = val_dataset[idx]
        measurements = measurements.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(measurements)

        # 转换为类别标签
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        gt = gt.cpu().numpy()

        # 计算评分
        score = scoringFunction(gt, pred)
        scores_summary['val'].append(score)

        # 可视化
        visualize_prediction(gt, pred, score,
                           os.path.join(save_dir, f'val_sample_{i+1}.png'))

        print(f"Val sample {i+1}: Score = {score:.4f}")

    # 保存评分摘要
    summary_text = [
        "="*70,
        "Reconstruction Scores Summary",
        "="*70,
        "",
        f"Training samples ({len(scores_summary['train'])} samples):",
    ]
    for i, score in enumerate(scores_summary['train']):
        summary_text.append(f"  Sample {i+1}: {score:.4f}")
    summary_text.append(f"  Mean: {np.mean(scores_summary['train']):.4f}")
    summary_text.append(f"  Std: {np.std(scores_summary['train']):.4f}")
    summary_text.append("")

    summary_text.append(f"Validation samples ({len(scores_summary['val'])} samples):")
    for i, score in enumerate(scores_summary['val']):
        summary_text.append(f"  Sample {i+1}: {score:.4f}")
    summary_text.append(f"  Mean: {np.mean(scores_summary['val']):.4f}")
    summary_text.append(f"  Std: {np.std(scores_summary['val']):.4f}")
    summary_text.append("")
    summary_text.append("="*70)

    # 保存到文件
    with open(os.path.join(save_dir, 'reconstruction_scores.txt'), 'w') as f:
        f.write('\n'.join(summary_text))

    # 打印摘要
    print("\n" + '\n'.join(summary_text))

    return scores_summary


def train_model(config):
    """
    训练模型主函数

    Args:
        config: 训练配置字典
    """
    # 创建结果文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get('exp_name', 'experiment')
    save_dir = os.path.join('results', f'{exp_name}_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print("="*70)
    print(f"Training EIT Reconstruction Network")
    print(f"Experiment: {exp_name}")
    print(f"Save directory: {save_dir}")
    print("="*70)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # 加载数据
    print("\nLoading data...")
    dataset = SimDataLoader(
        data_path=config['data_path'],
        num_samples=config.get('num_samples', None)
    )

    # 划分训练集和验证集
    train_size = int(config['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2)
    )

    # 创建模型
    print(f"\nCreating model: {config['model_name']}")
    model = get_model(config['model_name'], input_dim=config['input_dim'])
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # 损失函数和优化器
    criterion = SegmentationLoss(
        weight_ce=config.get('weight_ce', 1.0),
        weight_dice=config.get('weight_dice', 1.0)
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-5)
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=config.get('patience', 5)
    )

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_ce_loss': [],
        'val_ce_loss': [],
        'train_dice_loss': [],
        'val_dice_loss': [],
        'learning_rate': []
    }

    # 训练循环
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"\nStarting training for {config['epochs']} epochs...")
    print("-"*70)

    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")

        # 训练
        train_loss, train_ce, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证
        val_loss, val_ce, val_dice = validate_epoch(
            model, val_loader, criterion, device
        )

        # 记录历史
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_ce_loss'].append(train_ce)
        history['val_ce_loss'].append(val_ce)
        history['train_dice_loss'].append(train_dice)
        history['val_dice_loss'].append(val_dice)
        history['learning_rate'].append(current_lr)

        # 打印统计
        print(f"\nTrain Loss: {train_loss:.4f} (CE: {train_ce:.4f}, Dice: {train_dice:.4f})")
        print(f"Val Loss: {val_loss:.4f} (CE: {val_ce:.4f}, Dice: {val_dice:.4f})")
        print(f"Learning Rate: {current_lr:.2e}")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= config.get('early_stopping', 10):
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        # 定期保存
        if (epoch + 1) % config.get('save_interval', 10) == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

    # 保存最终模型和历史
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))

    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    # 保存训练曲线
    save_training_plots(history, save_dir)

    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Results saved to: {save_dir}")
    print("="*70)

    # 加载最佳模型进行评估和可视化
    print("\n" + "="*70)
    print("Loading best model for evaluation...")
    print("="*70)

    best_checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'))
    model.load_state_dict(best_checkpoint['model_state_dict'])

    # 评估和可视化
    scores_summary = evaluate_and_visualize(
        model, train_dataset, val_dataset, device, save_dir, num_samples=3
    )

    return model, history, save_dir, scores_summary


if __name__ == "__main__":
    # 命令行参数
    parser = argparse.ArgumentParser(description='Train EIT Reconstruction Network')

    # 实验设置
    parser.add_argument('--exp_name', type=str, default='eit_reconstruction',
                       help='Experiment name (default: eit_reconstruction)')
    parser.add_argument('--model_name', type=str, default='unet', choices=['simple', 'unet'],
                       help='Model architecture: simple or unet (default: simple)')

    # 数据设置
    parser.add_argument('--data_path', type=str, default='SimData/level_1',
                       help='Path to training data (default: SimData/level_1)')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to use (default: all available)')
    parser.add_argument('--train_split', type=float, default=0.9,
                       help='Training data split ratio (default: 0.8)')
    parser.add_argument('--input_dim', type=int, default=2356,
                       help='Input dimension (992 for training pattern, 2356 for evaluation pattern)')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-3, dest='learning_rate',
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')

    # 损失函数权重
    parser.add_argument('--weight_ce', type=float, default=1.0,
                       help='Cross entropy loss weight (default: 1.0)')
    parser.add_argument('--weight_dice', type=float, default=1.0,
                       help='Dice loss weight (default: 1.0)')

    # 训练控制
    parser.add_argument('--patience', type=int, default=3,
                       help='Learning rate scheduler patience (default: 5)')
    parser.add_argument('--early_stopping', type=int, default=20,
                       help='Early stopping patience (default: 15)')
    parser.add_argument('--save_interval', type=int, default=30,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers (default: 2)')

    args = parser.parse_args()

    # 构建训练配置
    config = {
        'exp_name': args.exp_name,
        'model_name': args.model_name,
        'input_dim': args.input_dim,
        'data_path': args.data_path,
        'num_samples': args.num_samples,
        'train_split': args.train_split,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'weight_ce': args.weight_ce,
        'weight_dice': args.weight_dice,
        'patience': args.patience,
        'early_stopping': args.early_stopping,
        'save_interval': args.save_interval,
        'num_workers': args.num_workers
    }

    # 训练模型
    model, history, save_dir, scores_summary = train_model(config)

    print(f"\nModel saved to: {save_dir}")
