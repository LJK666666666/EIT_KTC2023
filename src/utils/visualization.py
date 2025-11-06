"""
可视化工具模块
"""
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def plot_reconstruction(
    reconstruction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (10, 5)
):
    """
    绘制重建结果

    Args:
        reconstruction: 重建图像
        ground_truth: 真实图像（可选）
        title: 图像标题
        save_path: 保存路径
        cmap: 颜色映射
        figsize: 图像大小
    """
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 真实图像
        im1 = axes[0].imshow(ground_truth, cmap=cmap)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])

        # 重建图像
        im2 = axes[1].imshow(reconstruction, cmap=cmap)
        axes[1].set_title('Reconstruction')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])

        # 差异图
        diff = np.abs(ground_truth - reconstruction)
        im3 = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Absolute Error')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])

    else:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(reconstruction, cmap=cmap)
        ax.axis('off')
        plt.colorbar(im, ax=ax)

    if title and ground_truth is None:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    绘制训练曲线

    Args:
        history: 训练历史字典
        save_path: 保存路径
        figsize: 图像大小
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # 损失曲线
    if 'train_loss' in history and 'val_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # 学习率曲线
    if 'learning_rate' in history:
        axes[1].plot(history['learning_rate'])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics(
    metrics_list: List[dict],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    绘制评估指标曲线

    Args:
        metrics_list: 指标字典列表
        save_path: 保存路径
        figsize: 图像大小
    """
    if not metrics_list:
        return

    # 提取所有指标名称
    metric_names = list(metrics_list[0].keys())
    num_metrics = len(metric_names)

    # 创建子图
    rows = (num_metrics + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten() if num_metrics > 1 else [axes]

    # 绘制每个指标
    for idx, name in enumerate(metric_names):
        values = [m[name] for m in metrics_list]
        axes[idx].plot(values, marker='o')
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel(name.upper())
        axes[idx].set_title(f'{name.upper()} over Epochs')
        axes[idx].grid(True, alpha=0.3)

    # 隐藏多余的子图
    for idx in range(num_metrics, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    将 PyTorch tensor 转换为 numpy 数组

    Args:
        tensor: PyTorch tensor

    Returns:
        numpy 数组
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def plot_batch_samples(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_samples: int = 3,
    save_dir: Optional[str] = None,
    prefix: str = 'sample'
):
    """
    绘制批次中的多个样本

    Args:
        predictions: 预测结果 [B, C, H, W]
        targets: 真实值 [B, C, H, W]
        num_samples: 要绘制的样本数量
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    predictions = tensor_to_numpy(predictions)
    targets = tensor_to_numpy(targets)

    batch_size = min(predictions.shape[0], num_samples)

    for i in range(batch_size):
        pred = predictions[i, 0] if predictions.shape[1] == 1 else predictions[i]
        target = targets[i, 0] if targets.shape[1] == 1 else targets[i]

        save_path = None
        if save_dir:
            save_path = f"{save_dir}/{prefix}_{i + 1}.png"

        plot_reconstruction(pred, target, save_path=save_path)
