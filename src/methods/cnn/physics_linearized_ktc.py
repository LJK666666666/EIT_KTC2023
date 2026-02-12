"""
Linearized physics backend for test-time optimization.

This implementation is intentionally lightweight for inference-time speed:
- linearized response around sigma0
- differentiable operator in EIM domain
"""
from typing import Optional
import torch
import torch.nn.functional as F

from .physics_backend import PhysicsBackend


def _build_eim_mask(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Build 16x16 EIM valid mask (3 excluded electrodes per injection row)."""
    num = 16
    rows = torch.arange(num, device=device).unsqueeze(1)
    cols = torch.arange(num, device=device).unsqueeze(0)
    valid = (
        (cols != rows)
        & (cols != (rows - 1) % num)
        & (cols != (rows + 1) % num)
    )
    return valid.to(dtype).unsqueeze(0).unsqueeze(0)


class LinearizedKTCBackend(PhysicsBackend):
    """
    Fast linearized backend in EIM space.

    y_lin(sigma) = y0 + P(sigma - sigma0), where P maps sigma image to EIM.
    """

    def __init__(self, output_size: int = 128, device: str = "cuda"):
        self.output_size = output_size
        self.device = torch.device(device)
        self._mask: Optional[torch.Tensor] = None
        self._sigma0: Optional[torch.Tensor] = None
        self._y0: Optional[torch.Tensor] = None

    def prepare(self, measurements, sigma_init) -> None:
        sigma0 = sigma_init.detach().to(self.device)
        if sigma0.dim() != 4:
            raise ValueError(f"sigma_init must be [B, C, H, W], got {sigma0.shape}")
        if sigma0.shape[0] != 1 or sigma0.shape[1] != 1:
            raise ValueError(f"Only single-sample sigma_init [1,1,H,W] is supported, got {sigma0.shape}")

        self._sigma0 = sigma0
        self._y0 = self._forward_operator(sigma0).detach()

        # Keep this check strict: measurement dimension must match EIM [1,1,16,16].
        if measurements.shape != self._y0.shape:
            raise ValueError(
                f"Measurement shape mismatch: expected {self._y0.shape}, got {measurements.shape}"
            )

    def predict(self, sigma):
        if self._sigma0 is None or self._y0 is None:
            raise ValueError("Backend is not prepared. Call prepare() before predict().")
        delta = sigma - self._sigma0
        return self._y0 + self._forward_operator(delta)

    def _forward_operator(self, sigma: torch.Tensor) -> torch.Tensor:
        # P: image -> 16x16 EIM-like response.
        pooled = F.interpolate(
            sigma,
            size=(16, 16),
            mode="area"
        )

        # Electrode-difference style local response on the 16-electrode ring axis.
        diff = pooled - torch.roll(pooled, shifts=-1, dims=-1)
        response = 0.7 * diff + 0.3 * pooled

        if self._mask is None or self._mask.device != sigma.device or self._mask.dtype != sigma.dtype:
            self._mask = _build_eim_mask(sigma.device, sigma.dtype)
        return response * self._mask

