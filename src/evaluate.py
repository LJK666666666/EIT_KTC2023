"""
评估EIT重建模型并生成可视化对比图

在评估数据集上测试训练好的模型，计算KTC评分并生成对比图
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import scipy.io as sio
import json

from model import get_model
from eval_dataset import EvaluationDataset, load_all_evaluation_levels

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ktc_methods'))
import KTCScoring


def load_trained_model(checkpoint_path, device='cpu'):
    """
    加载训练好的模型

    Args:
        checkpoint_path: 模型检查点路径
        device: 运行设备

    Returns:
        model: 加载的模型
        config: 训练配置
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = {'model_name': 'simple', 'input_dim': 992}

    model = get_model(config['model_name'], input_dim=config['input_dim'])

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, config


def reconstruct_from_measurements(model, measurements, device='cpu'):
    """
    从测量数据重建导电率分布

    Args:
        model: 训练好的模型
        measurements: 测量数据 [N]
        device: 运行设备

    Returns:
        reconstruction: 重建图像 [256, 256]
    """
    model.eval()
    with torch.no_grad():
        measurements = measurements.unsqueeze(0).to(device)  # [1, N]
        output = model(measurements)  # [1, 3, 256, 256]

        # 转换为类别标签
        reconstruction = torch.argmax(output, dim=1).squeeze(0)  # [256, 256]

    return reconstruction.cpu().numpy()


def evaluate_single_sample(model, measurements, gt_image, device='cpu'):
    """
    评估单个样本

    Args:
        model: 训练好的模型
        measurements: 测量数据
        gt_image: Ground truth图像
        device: 运行设备

    Returns:
        reconstruction: 重建图像
        score: KTC评分
    """
    # 重建
    reconstruction = reconstruct_from_measurements(model, measurements, device)

    # 计算评分
    gt_np = gt_image.cpu().numpy()
    score = KTCScoring.scoringFunction(gt_np, reconstruction)

    return reconstruction, score


def plot_comparison(gt_image, reconstruction, measurements, data_idx, score, save_path):
    """
    绘制对比图：Ground Truth vs Reconstruction

    Args:
        gt_image: Ground truth图像 [256, 256]
        reconstruction: 重建图像 [256, 256]
        measurements: 测量数据
        data_idx: 数据索引
        score: KTC评分
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ground Truth
    im0 = axes[0].imshow(gt_image.T, origin='lower', cmap='gray', vmin=0, vmax=2)
    axes[0].set_title(f'Ground Truth')
    axes[0].axis('off')
    # 添加圆形边界
    circle = Circle((127.5, 127.5), 127.5, fill=False, edgecolor='red', linewidth=2)
    axes[0].add_patch(circle)

    # Reconstruction
    im1 = axes[1].imshow(reconstruction.T, origin='lower', cmap='gray', vmin=0, vmax=2)
    axes[1].set_title(f'Reconstruction (Score: {score:.4f})')
    axes[1].axis('off')
    circle = Circle((127.5, 127.5), 127.5, fill=False, edgecolor='red', linewidth=2)
    axes[1].add_patch(circle)

    # Measurements
    axes[2].plot(measurements.cpu().numpy())
    axes[2].set_title('Measurements')
    axes[2].set_xlabel('Measurement Index')
    axes[2].set_ylabel('Voltage Difference')
    axes[2].grid(True, alpha=0.3)

    # 添加colorbar
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    plt.suptitle(f'Data {data_idx}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_on_dataset(model, dataset, device, save_dir, level):
    """
    在整个数据集上评估模型

    Args:
        model: 训练好的模型
        dataset: 评估数据集
        device: 运行设备
        save_dir: 保存目录
        level: 评估级别

    Returns:
        results: 评估结果字典
    """
    results = {
        'level': level,
        'scores': [],
        'data_indices': []
    }

    level_dir = os.path.join(save_dir, f'level{level}')
    os.makedirs(level_dir, exist_ok=True)

    print(f"\nEvaluating Level {level}...")
    print("-"*70)

    for idx in range(len(dataset)):
        measurements, gt_image, data_idx = dataset[idx]

        # 评估
        reconstruction, score = evaluate_single_sample(
            model, measurements, gt_image, device
        )

        # 保存结果
        results['scores'].append(float(score))
        results['data_indices'].append(int(data_idx))

        # 生成对比图
        save_path = os.path.join(level_dir, f'data{data_idx}_comparison.png')
        plot_comparison(
            gt_image.numpy(),
            reconstruction,
            measurements,
            data_idx,
            score,
            save_path
        )

        # 保存重建结果
        np.save(os.path.join(level_dir, f'data{data_idx}_reconstruction.npy'), reconstruction)

        print(f"Data {data_idx}: Score = {score:.4f}")

    # 计算平均分
    avg_score = np.mean(results['scores'])
    results['average_score'] = float(avg_score)

    print(f"\nAverage Score: {avg_score:.4f}")
    print("-"*70)

    return results


def evaluate_all_levels(model_path, device='cpu', save_dir=None):
    """
    在所有评估级别上评估模型

    Args:
        model_path: 模型路径
        device: 运行设备
        save_dir: 保存目录

    Returns:
        all_results: 所有级别的评估结果
    """
    # 加载模型
    print("="*70)
    print("Loading model...")
    model, config = load_trained_model(model_path, device)
    print(f"Model loaded from: {model_path}")
    print(f"Model config: {config}")

    # 创建保存目录
    if save_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join('results', f'evaluation_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    print(f"Results will be saved to: {save_dir}")

    # 加载所有评估数据集
    print("\n" + "="*70)
    print("Loading evaluation datasets...")
    datasets = load_all_evaluation_levels()

    # 评估每个级别
    all_results = {}

    for level, dataset in datasets.items():
        results = evaluate_on_dataset(model, dataset, device, save_dir, level)
        all_results[f'level{level}'] = results

    # 保存总结果
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=4)

    # 生成总结报告
    generate_summary_report(all_results, save_dir)

    print("\n" + "="*70)
    print("Evaluation completed!")
    print(f"Results saved to: {save_dir}")
    print("="*70)

    return all_results


def generate_summary_report(all_results, save_dir):
    """
    生成总结报告

    Args:
        all_results: 所有级别的评估结果
        save_dir: 保存目录
    """
    # 创建总结图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    levels = []
    avg_scores = []
    num_samples = []

    for level_key in sorted(all_results.keys()):
        level = all_results[level_key]['level']
        levels.append(level)
        avg_scores.append(all_results[level_key]['average_score'])
        num_samples.append(len(all_results[level_key]['scores']))

    # 平均分柱状图
    axes[0].bar(levels, avg_scores, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Level')
    axes[0].set_ylabel('Average Score')
    axes[0].set_title('Average Score by Level')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_xticks(levels)

    # 在柱状图上显示数值
    for i, (level, score) in enumerate(zip(levels, avg_scores)):
        axes[0].text(level, score + 0.01, f'{score:.3f}',
                    ha='center', va='bottom', fontsize=10)

    # 样本数量
    axes[1].bar(levels, num_samples, color='coral', alpha=0.7)
    axes[1].set_xlabel('Level')
    axes[1].set_ylabel('Number of Samples')
    axes[1].set_title('Number of Samples by Level')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_xticks(levels)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'summary_report.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 生成文本报告
    report_path = os.path.join(save_dir, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EIT RECONSTRUCTION EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")

        total_score = 0
        total_samples = 0

        for level_key in sorted(all_results.keys()):
            level_data = all_results[level_key]
            level = level_data['level']
            avg_score = level_data['average_score']
            scores = level_data['scores']
            n_samples = len(scores)

            f.write(f"Level {level}:\n")
            f.write(f"  Average Score: {avg_score:.4f}\n")
            f.write(f"  Number of Samples: {n_samples}\n")
            f.write(f"  Individual Scores: {[f'{s:.4f}' for s in scores]}\n")
            f.write("\n")

            total_score += avg_score * n_samples
            total_samples += n_samples

        overall_avg = total_score / total_samples if total_samples > 0 else 0
        f.write("-"*70 + "\n")
        f.write(f"Overall Average Score: {overall_avg:.4f}\n")
        f.write(f"Total Samples: {total_samples}\n")
        f.write("="*70 + "\n")

    print(f"\nSummary report saved to: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate EIT reconstruction model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 评估
    results = evaluate_all_levels(
        model_path=args.model_path,
        device=device,
        save_dir=args.save_dir
    )
