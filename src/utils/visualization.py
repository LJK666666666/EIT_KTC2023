"""
可视化工具模块
"""
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def _extract_training_series(history: dict):
    """
    从 history 中提取按 epoch 对齐的 train/val/lr 序列。
    优先使用 epoch_records；若不存在则回退旧格式。
    """
    epoch_records = history.get('epoch_records', [])
    if isinstance(epoch_records, list) and len(epoch_records) > 0:
        epochs = [int(item.get('epoch', idx + 1)) for idx, item in enumerate(epoch_records)]
        train_loss = [item.get('train_loss', np.nan) for item in epoch_records]
        val_loss = [item.get('val_loss', np.nan) for item in epoch_records]
        learning_rate = [item.get('learning_rate', np.nan) for item in epoch_records]
        return epochs, train_loss, val_loss, learning_rate

    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    learning_rate = history.get('learning_rate', [])
    max_len = max(len(train_loss), len(val_loss), len(learning_rate), 0)
    epochs = list(range(1, max_len + 1))
    return epochs, train_loss, val_loss, learning_rate


def _extract_metrics_series(
    metrics_list: Optional[List[dict]] = None,
    history: Optional[dict] = None
):
    """
    提取指标序列，返回 (epochs, metrics_seq)。
    优先从 history['epoch_records'][].val_metrics 提取；否则回退 metrics_list。
    """
    if history is not None:
        epoch_records = history.get('epoch_records', [])
        if isinstance(epoch_records, list) and len(epoch_records) > 0:
            epochs = []
            metrics_seq = []
            for idx, item in enumerate(epoch_records):
                metrics = item.get('val_metrics', {})
                if isinstance(metrics, dict) and len(metrics) > 0:
                    epochs.append(int(item.get('epoch', idx + 1)))
                    metrics_seq.append(metrics)
            if len(metrics_seq) > 0:
                return epochs, metrics_seq

        val_metrics = history.get('val_metrics', [])
        if isinstance(val_metrics, list) and len(val_metrics) > 0:
            epochs = list(range(1, len(val_metrics) + 1))
            return epochs, val_metrics

    if metrics_list is None or len(metrics_list) == 0:
        return [], []
    epochs = list(range(1, len(metrics_list) + 1))
    return epochs, metrics_list


def plot_reconstruction(
    reconstruction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    nn_prediction: Optional[np.ndarray] = None,
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
        nn_prediction: 神经网络原始预测图像（可选）
        title: 图像标题
        save_path: 保存路径
        cmap: 颜色映射
        figsize: 图像大小
    """
    if ground_truth is not None and nn_prediction is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        combined_min = float(min(np.min(ground_truth), np.min(nn_prediction), np.min(reconstruction)))
        combined_max = float(max(np.max(ground_truth), np.max(nn_prediction), np.max(reconstruction)))

        im1 = axes[0].imshow(ground_truth, cmap=cmap, vmin=combined_min, vmax=combined_max)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])

        im2 = axes[1].imshow(nn_prediction, cmap=cmap, vmin=combined_min, vmax=combined_max)
        axes[1].set_title('NN Prediction')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])

        im3 = axes[2].imshow(reconstruction, cmap=cmap, vmin=combined_min, vmax=combined_max)
        axes[2].set_title('Physics Optimized')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2])

        diff = np.abs(ground_truth - reconstruction)
        im4 = axes[3].imshow(diff, cmap='hot')
        axes[3].set_title('Absolute Error')
        axes[3].axis('off')
        plt.colorbar(im4, ax=axes[3])
    elif ground_truth is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        combined_min = float(min(np.min(ground_truth), np.min(reconstruction)))
        combined_max = float(max(np.max(ground_truth), np.max(reconstruction)))

        # 真实图像
        im1 = axes[0].imshow(ground_truth, cmap=cmap, vmin=combined_min, vmax=combined_max)
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])

        # 重建图像
        im2 = axes[1].imshow(reconstruction, cmap=cmap, vmin=combined_min, vmax=combined_max)
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
    epochs, train_loss, val_loss, learning_rate = _extract_training_series(history)

    # 损失曲线
    if len(train_loss) > 0 or len(val_loss) > 0:
        if len(train_loss) > 0:
            axes[0].plot(epochs[:len(train_loss)], train_loss, label='Train Loss')
        if len(val_loss) > 0:
            axes[0].plot(epochs[:len(val_loss)], val_loss, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # 学习率曲线
    if len(learning_rate) > 0:
        axes[1].plot(epochs[:len(learning_rate)], learning_rate)
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
    metrics_list: Optional[List[dict]] = None,
    history: Optional[dict] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    绘制评估指标曲线

    Args:
        metrics_list: 指标字典列表（旧接口）
        history: 训练历史字典（新接口，支持 epoch_records）
        save_path: 保存路径
        figsize: 图像大小
    """
    epochs, metrics_seq = _extract_metrics_series(metrics_list=metrics_list, history=history)
    if len(metrics_seq) == 0:
        return

    # 提取所有指标名称
    metric_names = list(metrics_seq[0].keys())
    num_metrics = len(metric_names)

    # 创建子图
    rows = (num_metrics + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=figsize)
    axes = axes.flatten() if num_metrics > 1 else [axes]

    # 绘制每个指标
    for idx, name in enumerate(metric_names):
        values = [m.get(name, np.nan) for m in metrics_seq]
        axes[idx].plot(epochs, values, marker='o')
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
