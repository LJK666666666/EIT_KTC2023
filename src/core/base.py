"""
EIT 重建项目 - 核心模块

所有重建方法的基类和通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn


class BaseReconstructionMethod(ABC):
    """
    所有重建方法的基类

    所有重建方法（CNN、扩散模型、传统方法等）都应该继承这个基类
    并实现所有抽象方法
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化重建方法

        Args:
            config: 配置字典，包含所有超参数
        """
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._build_model()

        if self.model is not None:
            self.model = self.model.to(self.device)

    @abstractmethod
    def _build_model(self) -> Optional[nn.Module]:
        """
        构建模型

        Returns:
            nn.Module: PyTorch 模型，如果不需要模型（如传统方法）则返回 None
        """
        pass

    @abstractmethod
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        单个训练步骤

        Args:
            batch: (measurements, ground_truth) 数据批次

        Returns:
            Dict[str, float]: 包含 loss 和其他指标的字典
                例如: {'loss': 0.5, 'ce_loss': 0.3, 'dice_loss': 0.2}
        """
        pass

    @abstractmethod
    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        单个验证步骤

        Args:
            batch: (measurements, ground_truth) 数据批次

        Returns:
            Dict[str, float]: 包含各种评估指标的字典
        """
        pass

    @abstractmethod
    def inference(self, measurements: torch.Tensor) -> torch.Tensor:
        """
        推理：从测量数据重建导电率分布

        Args:
            measurements: 测量数据 [batch_size, measurement_dim]

        Returns:
            torch.Tensor: 重建的导电率分布 [batch_size, height, width]
        """
        pass

    def save_checkpoint(self, path: str, epoch: int, optimizer_state: Optional[Dict] = None, scheduler_state: Optional[Dict] = None):
        """
        保存检查点

        Args:
            path: 保存路径
            epoch: 当前 epoch
            optimizer_state: 优化器状态（可选）
            scheduler_state: 调度器状态（可选）
        """
        if self.model is None:
            return

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }

        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state

        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """
        加载检查点（兼容多种格式）

        Args:
            path: 检查点路径
            optimizer: 优化器实例（如果需要恢复优化器状态）
            scheduler: 调度器实例（如果需要恢复调度器状态）

        Returns:
            Dict: 包含epoch和其他加载信息的字典
        """
        if self.model is None:
            return {'epoch': 0}

        checkpoint = torch.load(path, map_location=self.device)

        # 兼容不同的检查点格式
        # 格式1: {'model_state_dict': ..., 'optimizer_state_dict': ..., 'epoch': ...}  (当前项目格式)
        # 格式2: {'model': ..., 'epoch': ...}  (CDEIT原始格式)
        # 格式3: 直接是state_dict
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            model_state_dict = checkpoint['model']
        else:
            # 假设整个checkpoint就是state_dict
            model_state_dict = checkpoint

        # 加载模型权重
        self.model.load_state_dict(model_state_dict)

        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return {
            'epoch': checkpoint.get('epoch', 0),
            'has_optimizer': 'optimizer_state_dict' in checkpoint,
            'has_scheduler': 'scheduler_state_dict' in checkpoint
        }

    def get_optimizer(self) -> Optional[torch.optim.Optimizer]:
        """
        获取优化器（子类可以重写）

        Returns:
            torch.optim.Optimizer: 优化器
        """
        if self.model is None:
            return None

        # 从 optimizer 配置读取参数
        optimizer_config = self.config.get('optimizer', {})
        lr = optimizer_config.get('lr', 1e-3)
        weight_decay = optimizer_config.get('weight_decay', 1e-5)
        optimizer_type = optimizer_config.get('type', 'adam').lower()

        if optimizer_type == 'adam':
            return torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

    def get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        获取学习率调度器（子类可以重写）

        Args:
            optimizer: 优化器

        Returns:
            学习率调度器
        """
        # 从 scheduler 配置读取参数
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'reduce_on_plateau').lower()

        if scheduler_type == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10)
            )
        elif scheduler_type == 'steplr':
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config.get('step_size', 10),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        else:
            return None
