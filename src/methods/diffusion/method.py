"""
Diffusion 重建方法实现
基于 CDEIT 的 DiT 模型
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import sys
import os

# 添加CDEIT路径以导入其diffusion模块
cdeit_path = os.path.join(os.path.dirname(__file__), '../../../programs/CDEIT')
if cdeit_path not in sys.path:
    sys.path.insert(0, cdeit_path)

from ...core.base import BaseReconstructionMethod
from .models import DiT

# 使用CDEIT的diffusion模块
from diffusion import create_diffusion


class DiffusionReconstruction(BaseReconstructionMethod):
    """扩散模型重建方法（基于 DiT）"""

    def __init__(self, config: Dict):
        super().__init__(config)

        # 创建扩散模型
        model_config = config.get('model', {})
        self.num_timesteps = model_config.get('num_timesteps', 1000)
        self.beta_schedule = model_config.get('beta_schedule', 'linear')

        # 推理配置
        inference_config = config.get('inference', {})
        self.num_sampling_steps = inference_config.get('num_sampling_steps', 50)

        # 创建扩散过程（使用CDEIT的create_diffusion，参数更简单）
        self.diffusion = create_diffusion(timestep_respacing="")

    def _build_model(self) -> nn.Module:
        """构建 DiT 模型"""
        model_config = self.config.get('model', {})

        model = DiT(
            input_size=model_config.get('input_size', 128),
            patch_size=model_config.get('patch_size', 2),
            in_channels=model_config.get('in_channels', 2),  # 噪声图像(1) + 条件图像(1) = 2通道
            out_channels=1,
            hidden_size=model_config.get('hidden_size', 512),
            num_heads=model_config.get('num_heads', 8),
        )

        return model

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        训练步骤

        Args:
            batch: (measurements, conductivity) 元组
                   measurements: [B, 1, 16, 16] EIM格式

        Returns:
            包含损失的字典
        """
        measurements, conductivity = batch
        measurements = measurements.to(self.device)
        conductivity = conductivity.to(self.device)

        # 条件信息直接使用16x16的EIM（模型内部会上采样到128x128）
        condition = measurements  # [B, 1, 16, 16]

        # 采样随机时间步
        t = torch.randint(
            0, self.diffusion.num_timesteps,
            (conductivity.shape[0],),
            device=self.device
        )

        # 计算扩散损失
        model_kwargs = {"y": condition}
        loss_dict = self.diffusion.training_losses(
            self.model,
            conductivity,
            t,
            model_kwargs
        )

        loss = loss_dict["loss"].mean()

        return {'loss': loss}

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        验证步骤

        Args:
            batch: (measurements, conductivity) 元组
                   measurements: [B, 1, 16, 16] EIM格式

        Returns:
            包含损失的字典
        """
        measurements, conductivity = batch
        measurements = measurements.to(self.device)
        conductivity = conductivity.to(self.device)

        # 条件信息直接使用16x16的EIM
        condition = measurements  # [B, 1, 16, 16]

        # 采样随机时间步
        t = torch.randint(
            0, self.diffusion.num_timesteps,
            (conductivity.shape[0],),
            device=self.device
        )

        # 计算扩散损失
        with torch.no_grad():
            model_kwargs = {"y": condition}
            loss_dict = self.diffusion.training_losses(
                self.model,
                conductivity,
                t,
                model_kwargs
            )

        loss = loss_dict["loss"].mean()

        return {'loss': loss}

    def inference(self, measurements: torch.Tensor) -> torch.Tensor:
        """
        推理 - 使用 DDIM 采样（CDEIT的ddim_sampleEIT方法）

        Args:
            measurements: [B, 1, 16, 16] EIM格式的测量数据

        Returns:
            重建的电导率图像 [B, 1, 128, 128]
        """
        measurements = measurements.to(self.device)

        # 条件信息直接使用16x16的EIM
        condition = measurements  # [B, 1, 16, 16]

        # 获取图像大小
        batch_size = measurements.shape[0]
        img_size = self.config.get('model', {}).get('input_size', 128)

        # 目标shape
        shape = (batch_size, 1, img_size, img_size)

        # 使用CDEIT的ddim_sampleEIT方法进行采样
        with torch.no_grad():
            self.model.eval()
            model_kwargs = {"y": condition, "y_st": None}

            samples = self.diffusion.ddim_sampleEIT(
                self.model,
                shape,
                self.num_sampling_steps,
                model_kwargs,
                clip_denoised=True,
                eta=1
            )

        return samples


def create_diffusion_method(config: Dict) -> DiffusionReconstruction:
    """
    创建扩散重建方法

    Args:
        config: 配置字典

    Returns:
        扩散重建方法实例
    """
    return DiffusionReconstruction(config)
