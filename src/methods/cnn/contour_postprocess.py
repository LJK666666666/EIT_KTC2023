"""
Post-processing helpers for ternary map cleanup.
"""
from typing import Tuple
import numpy as np
from scipy.ndimage import binary_fill_holes, label


def ternary_from_sigma(sigma: np.ndarray, threshold: float = 0.25) -> np.ndarray:
    """
    Convert continuous sigma into ternary {-1, 0, 1}.
    """
    out = np.zeros_like(sigma, dtype=np.float32)
    out[sigma < -threshold] = -1.0
    out[sigma > threshold] = 1.0
    return out


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    if min_pixels <= 0:
        return mask
    lbl, n = label(mask.astype(np.uint8))
    if n <= 0:
        return mask
    keep = np.zeros_like(mask, dtype=bool)
    for cid in range(1, n + 1):
        comp = lbl == cid
        if int(comp.sum()) >= min_pixels:
            keep |= comp
    return keep


def cleanup_ternary(ternary: np.ndarray, min_component_ratio: float = 0.0015) -> np.ndarray:
    """
    Remove tiny components and fill tiny holes for each class.
    """
    h, w = ternary.shape
    min_pixels = int(max(1, round(h * w * float(min_component_ratio))))
    neg = ternary < -0.5
    pos = ternary > 0.5
    neg = _remove_small_components(neg, min_pixels=min_pixels)
    pos = _remove_small_components(pos, min_pixels=min_pixels)
    neg = binary_fill_holes(neg)
    pos = binary_fill_holes(pos)
    out = np.zeros_like(ternary, dtype=np.float32)
    out[neg] = -1.0
    out[pos] = 1.0
    overlap = np.logical_and(neg, pos)
    out[overlap] = 0.0
    return out

