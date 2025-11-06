"""
评估模块
提供 EIT 重建质量评估指标
"""
from typing import Dict, List, Optional
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim


class EITEvaluator:
    """EIT 重建评估器"""

    @staticmethod
    def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算均方误差"""
        return torch.mean((pred - target) ** 2).item()

    @staticmethod
    def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算平均绝对误差"""
        return torch.mean(torch.abs(pred - target)).item()

    @staticmethod
    def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
        """计算峰值信噪比"""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(max_val / torch.sqrt(mse)).item()

    @staticmethod
    def compute_ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算结构相似性指数"""
        # 转换为 numpy 数组并移到 CPU
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # 如果是批次数据，计算每个样本的 SSIM 并取平均
        if pred_np.ndim == 4:  # [B, C, H, W]
            ssim_values = []
            for i in range(pred_np.shape[0]):
                # 假设是单通道图像
                pred_img = pred_np[i, 0] if pred_np.shape[1] == 1 else pred_np[i]
                target_img = target_np[i, 0] if target_np.shape[1] == 1 else target_np[i]

                ssim_val = ssim(target_img, pred_img, data_range=target_img.max() - target_img.min())
                ssim_values.append(ssim_val)
            return np.mean(ssim_values)

        elif pred_np.ndim == 3:  # [C, H, W]
            pred_img = pred_np[0] if pred_np.shape[0] == 1 else pred_np
            target_img = target_np[0] if target_np.shape[0] == 1 else target_np
            return ssim(target_img, pred_img, data_range=target_img.max() - target_img.min())

        elif pred_np.ndim == 2:  # [H, W]
            return ssim(target_np, pred_np, data_range=target_np.max() - target_np.min())

        else:
            raise ValueError(f"Unsupported tensor dimension: {pred_np.ndim}")

    @staticmethod
    def compute_relative_error(pred: torch.Tensor, target: torch.Tensor) -> float:
        """计算相对误差"""
        return (torch.norm(pred - target) / torch.norm(target)).item()

    def compute_all_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        计算所有指标

        Args:
            pred: 预测结果
            target: 真实值
            metrics: 要计算的指标列表，默认为全部

        Returns:
            包含所有指标的字典
        """
        if metrics is None:
            metrics = ['mse', 'mae', 'psnr', 'ssim', 'relative_error']

        results = {}

        if 'mse' in metrics:
            results['mse'] = self.compute_mse(pred, target)

        if 'mae' in metrics:
            results['mae'] = self.compute_mae(pred, target)

        if 'psnr' in metrics:
            results['psnr'] = self.compute_psnr(pred, target)

        if 'ssim' in metrics:
            results['ssim'] = self.compute_ssim(pred, target)

        if 'relative_error' in metrics:
            results['relative_error'] = self.compute_relative_error(pred, target)

        return results

    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        聚合多个样本的指标

        Args:
            metrics_list: 指标字典列表

        Returns:
            平均指标字典
        """
        if not metrics_list:
            return {}

        # 获取所有指标名称
        metric_names = metrics_list[0].keys()

        # 计算每个指标的平均值
        aggregated = {}
        for name in metric_names:
            values = [m[name] for m in metrics_list if name in m]
            aggregated[name] = np.mean(values)

        return aggregated
