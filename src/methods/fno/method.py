"""
FNO reconstruction method.
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.base import BaseReconstructionMethod
from .model import FNO2d


class FNOReconstruction(BaseReconstructionMethod):
    def __init__(self, config: Dict):
        model_config = config.get('model', {})
        self.output_size = model_config.get('output_size', 128)
        super().__init__(config)
        self.loss_fn = nn.MSELoss()

    def _build_model(self) -> nn.Module:
        model_config = self.config.get('model', {})
        return FNO2d(
            input_channels=model_config.get('input_channels', 1),
            output_channels=model_config.get('output_channels', 1),
            width=model_config.get('width', 64),
            modes1=model_config.get('modes1', 12),
            modes2=model_config.get('modes2', 12),
            depth=model_config.get('depth', 4),
            output_size=model_config.get('output_size', 128),
            add_coord=model_config.get('add_coord', True)
        )

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        measurements, conductivity = batch
        measurements = measurements.to(self.device)
        conductivity = conductivity.to(self.device)

        pred = self.model(measurements)

        if conductivity.shape[-1] != pred.shape[-1]:
            target = F.interpolate(
                conductivity,
                size=(pred.shape[-1], pred.shape[-2]),
                mode='bilinear',
                align_corners=False
            )
        else:
            target = conductivity

        loss = self.loss_fn(pred, target)
        return {'loss': loss}

    def val_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        measurements, conductivity = batch
        measurements = measurements.to(self.device)
        conductivity = conductivity.to(self.device)

        with torch.no_grad():
            pred = self.model(measurements)

        if conductivity.shape[-1] != pred.shape[-1]:
            target = F.interpolate(
                conductivity,
                size=(pred.shape[-1], pred.shape[-2]),
                mode='bilinear',
                align_corners=False
            )
        else:
            target = conductivity

        loss = self.loss_fn(pred, target)
        return {'loss': loss}

    def inference(self, measurements: torch.Tensor) -> torch.Tensor:
        measurements = measurements.to(self.device)

        with torch.no_grad():
            self.model.eval()
            pred = self.model(measurements)

        return pred


def create_fno_method(config: Dict) -> FNOReconstruction:
    return FNOReconstruction(config)
