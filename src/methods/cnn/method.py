"""
CNN 重建方法实现
支持多种网络架构：UNet 或 论文的CNN-EIM
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.base import BaseReconstructionMethod
from .unet import create_unet


class CNNReconstruction(BaseReconstructionMethod):
    """CNN 重建方法（支持 UNet 或 论文的CNN-EIM架构）"""

    def __init__(self, config: Dict):
        # 确定模型类型和输出尺寸（必须在super().__init__()之前设置）
        model_config = config.get('model', {})
        self.model_type = model_config.get('type', 'unet')
        self.output_size = model_config.get('output_size', 128)

        super().__init__(config)
        self.loss_fn = nn.MSELoss()

    def _build_model(self) -> nn.Module:
        """构建模型"""
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'unet')

        if model_type == 'paper_cnn_eim':
            # 使用论文中的CNN-EIM架构
            from .paper_cnn_eim import create_cnn_eim
            model = create_cnn_eim(self.config)
            print(f"[CNN] 使用论文的CNN-EIM架构，输出尺寸: {self.output_size}×{self.output_size}")
        else:
            # 使用通用UNet架构
            model = create_unet(self.config)
            self.output_size = 128
            print(f"[CNN] 使用UNet架构，输出尺寸: {self.output_size}×{self.output_size}")

        return model

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        训练步骤

        Args:
            batch: (measurements, conductivity) 元组

        Returns:
            包含损失的字典
        """
        measurements, conductivity = batch
        measurements = measurements.to(self.device)
        conductivity = conductivity.to(self.device)

        # 前向传播
        pred = self.model(measurements)

        # 调整目标尺寸以匹配预测
        if conductivity.shape[-1] != pred.shape[-1]:
            # 使用插值调整目标尺寸
            target = torch.nn.functional.interpolate(
                conductivity,
                size=(pred.shape[-1], pred.shape[-2]),
                mode='bilinear',
                align_corners=False
            )
        else:
            target = conductivity

        # 计算损失
        loss = self.loss_fn(pred, target)

        return {'loss': loss}

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        验证步骤

        Args:
            batch: (measurements, conductivity) 元组

        Returns:
            包含损失的字典
        """
        measurements, conductivity = batch
        measurements = measurements.to(self.device)
        conductivity = conductivity.to(self.device)

        # 前向传播
        with torch.no_grad():
            pred = self.model(measurements)

        # 调整目标尺寸以匹配预测
        if conductivity.shape[-1] != pred.shape[-1]:
            target = torch.nn.functional.interpolate(
                conductivity,
                size=(pred.shape[-1], pred.shape[-2]),
                mode='bilinear',
                align_corners=False
            )
        else:
            target = conductivity

        # 计算损失
        loss = self.loss_fn(pred, target)

        return {'loss': loss}

    def inference(self, measurements: torch.Tensor) -> torch.Tensor:
        """
        推理

        Args:
            measurements: 测量数据

        Returns:
            重建的电导率图像
        """
        measurements = measurements.to(self.device)

        with torch.no_grad():
            self.model.eval()
            pred = self.model(measurements)

        return pred


def create_cnn_method(config: Dict) -> CNNReconstruction:
    """
    创建 CNN 重建方法

    Args:
        config: 配置字典

    Returns:
        CNN 重建方法实例
    """
    return CNNReconstruction(config)
