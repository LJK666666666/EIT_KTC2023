"""
DeepDbar 重建方法实现
基于 Hamilton & Hauptmann (2018)
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn

from ...core.base import BaseReconstructionMethod
from .model import create_deepdbar_model


class DeepDbarReconstruction(BaseReconstructionMethod):
    """DeepDbar U-Net 重建方法"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.loss_fn = nn.MSELoss()

        # 获取输出尺寸（DeepDbar固定为64x64）
        self.output_size = 64

    def _build_model(self) -> nn.Module:
        """构建 DeepDbar U-Net 模型"""
        model = create_deepdbar_model(self.config)
        print(f"[DeepDbar] 使用 DeepDbar U-Net 架构，输出尺寸: 64×64")
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

        # 确保输入是 [batch, 1, H, W] 格式
        if measurements.dim() == 3:
            measurements = measurements.unsqueeze(1)
        if conductivity.dim() == 3:
            conductivity = conductivity.unsqueeze(1)

        # 调整输入尺寸到64x64（如果需要）
        if measurements.shape[-1] != 64 or measurements.shape[-2] != 64:
            measurements = torch.nn.functional.interpolate(
                measurements,
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            )

        # 前向传播
        pred = self.model(measurements)

        # 调整目标尺寸到64x64（如果需要）
        if conductivity.shape[-1] != 64 or conductivity.shape[-2] != 64:
            target = torch.nn.functional.interpolate(
                conductivity,
                size=(64, 64),
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

        # 确保输入是 [batch, 1, H, W] 格式
        if measurements.dim() == 3:
            measurements = measurements.unsqueeze(1)
        if conductivity.dim() == 3:
            conductivity = conductivity.unsqueeze(1)

        # 调整输入尺寸到64x64（如果需要）
        if measurements.shape[-1] != 64 or measurements.shape[-2] != 64:
            measurements = torch.nn.functional.interpolate(
                measurements,
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            )

        # 前向传播
        with torch.no_grad():
            pred = self.model(measurements)

        # 调整目标尺寸到64x64（如果需要）
        if conductivity.shape[-1] != 64 or conductivity.shape[-2] != 64:
            target = torch.nn.functional.interpolate(
                conductivity,
                size=(64, 64),
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
            measurements: 测量数据 [batch, H, W] 或 [batch, 1, H, W]

        Returns:
            重建的电导率图像 [batch, 1, 64, 64]
        """
        measurements = measurements.to(self.device)

        # 确保输入是 [batch, 1, H, W] 格式
        if measurements.dim() == 3:
            measurements = measurements.unsqueeze(1)

        # 调整输入尺寸到64x64（如果需要）
        if measurements.shape[-1] != 64 or measurements.shape[-2] != 64:
            measurements = torch.nn.functional.interpolate(
                measurements,
                size=(64, 64),
                mode='bilinear',
                align_corners=False
            )

        with torch.no_grad():
            self.model.eval()
            pred = self.model(measurements)

        return pred


def create_deepdbar_method(config: Dict) -> DeepDbarReconstruction:
    """
    创建 DeepDbar 重建方法

    Args:
        config: 配置字典

    Returns:
        DeepDbar 重建方法实例
    """
    return DeepDbarReconstruction(config)
