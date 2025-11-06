"""
改进的CNN方法 - 支持论文架构和灵活的输出尺寸
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn
from ...core.base import BaseReconstructionMethod


class CNNReconstructionImproved(BaseReconstructionMethod):
    """改进的CNN重建方法，支持多种网络架构"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.loss_fn = nn.MSELoss()

    def _build_model(self) -> nn.Module:
        """构建模型"""
        from .paper_cnn_eim import create_cnn_eim

        # 从配置中获取模型类型
        model_type = self.config.get('model', {}).get('type', 'paper_cnn_eim')

        if model_type == 'paper_cnn_eim':
            # 使用论文中的CNN-EIM架构
            model = create_cnn_eim(self.config)
            self.output_size = self.config.get('model', {}).get('output_size', 64)
        else:
            # 保留原来的UNet支持
            from .unet import create_unet
            model = create_unet(self.config)
            self.output_size = 128

        print(f"[CNN] 使用模型类型: {model_type}")
        print(f"[CNN] 输出尺寸: {self.output_size}×{self.output_size}")

        return model

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        训练步骤

        Args:
            batch: (measurements, conductivity) 元组
                   measurements: [B, 1, 16, 16] (EIM格式)
                   conductivity: [B, 1, 128, 128]

        Returns:
            包含损失的字典
        """
        measurements, conductivity = batch
        measurements = measurements.to(self.device)
        conductivity = conductivity.to(self.device)

        # 前向传播
        pred = self.model(measurements)  # [B, 1, output_size, output_size]

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
        """
        measurements, conductivity = batch
        measurements = measurements.to(self.device)
        conductivity = conductivity.to(self.device)

        # 前向传播
        pred = self.model(measurements)

        # 调整目标尺寸
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
        推理步骤

        Args:
            measurements: [B, 1, 16, 16] EIM格式

        Returns:
            pred: [B, 1, output_size, output_size] 重建图像
        """
        measurements = measurements.to(self.device)

        with torch.no_grad():
            pred = self.model(measurements)

        # 如果需要输出128×128，则进行上采样
        if pred.shape[-1] < 128:
            pred = torch.nn.functional.interpolate(
                pred,
                size=(128, 128),
                mode='bilinear',
                align_corners=False
            )

        return pred

    def get_optimizer(self):
        """获取优化器"""
        lr = self.config.get('optimizer', {}).get('lr', 0.0001)
        weight_decay = self.config.get('optimizer', {}).get('weight_decay', 0.00001)

        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    def get_scheduler(self, optimizer):
        """获取学习率调度器"""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau')

        if scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=scheduler_config.get('mode', 'min'),
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 15),
                verbose=False
            )
        else:
            return None

    def save_checkpoint(self, path: str, epoch: int, optimizer_state=None, scheduler_state=None):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, optimizer=None, scheduler=None) -> Dict:
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return {'epoch': checkpoint.get('epoch', 0)}
