"""
Contour-step parameterization helpers for test-time optimization.
"""
from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


def build_initial_labels(sigma: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """
    Convert continuous sigma to ternary labels: -1/0/1.
    """
    labels = np.zeros_like(sigma, dtype=np.int8)
    labels[sigma < -threshold] = -1
    labels[sigma > threshold] = 1
    return labels


def signed_distance(mask: np.ndarray) -> np.ndarray:
    """
    Signed distance map: inside negative, outside positive.
    """
    inside = mask.astype(bool)
    outside = ~inside
    d_out = distance_transform_edt(outside).astype(np.float32)
    d_in = distance_transform_edt(inside).astype(np.float32)
    return d_out - d_in


def build_distance_maps(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (d_neg, d_pos, fg_distance).
    """
    neg = labels == -1
    pos = labels == 1
    fg = np.logical_or(neg, pos)
    d_neg = signed_distance(neg)
    d_pos = signed_distance(pos)
    fg_distance = distance_transform_edt(~fg).astype(np.float32)
    return d_neg, d_pos, fg_distance


def upsample_displacement(theta: torch.Tensor, h: int, w: int, max_shift_px: float) -> torch.Tensor:
    """
    Upsample low-resolution control grid to full-resolution displacement field.
    """
    disp = F.interpolate(theta, size=(h, w), mode="bicubic", align_corners=False)
    return torch.tanh(disp) * max_shift_px


def synthesize_sigma(
    theta_neg: torch.Tensor,
    theta_pos: torch.Tensor,
    d_neg: torch.Tensor,
    d_pos: torch.Tensor,
    values: torch.Tensor,
    tau: float,
    max_shift_px: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build sigma from soft step functions and class values.
    Returns (sigma, m_neg, m_bg, m_pos).
    """
    h = int(d_neg.shape[-2])
    w = int(d_neg.shape[-1])
    delta_neg = upsample_displacement(theta_neg, h, w, max_shift_px=max_shift_px)
    delta_pos = upsample_displacement(theta_pos, h, w, max_shift_px=max_shift_px)
    m_neg = torch.sigmoid(-(d_neg + delta_neg) / max(tau, 1e-4))
    m_pos = torch.sigmoid(-(d_pos + delta_pos) / max(tau, 1e-4))
    m_bg = torch.clamp(1.0 - m_neg - m_pos, min=0.0, max=1.0)
    sigma = values[0] * m_neg + values[1] * m_bg + values[2] * m_pos
    return sigma, m_neg, m_bg, m_pos

