# splineops/src/splineops/utils/metrics.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from .image import crop_to_central_region

__all__ = ["compute_snr_and_mse_region"]

def _extract_region(
    a: np.ndarray,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,  # (row, col, h, w)
    mask: Optional[np.ndarray] = None,
    border_fraction: Optional[float] = None,
) -> np.ndarray:
    if mask is not None:
        if mask.shape != a.shape[:2]:
            raise ValueError("mask must match spatial shape of input")
        return a[mask]
    if roi is not None:
        r, c, h, w = roi
        return a[r:r + h, c:c + w]
    if border_fraction is not None:
        return crop_to_central_region(a, border_fraction)
    return a  # full image


def compute_snr_and_mse_region(
    original: np.ndarray,
    processed: np.ndarray,
    *,
    roi: Optional[Tuple[int, int, int, int]] = None,
    mask: Optional[np.ndarray] = None,
    border_fraction: Optional[float] = None,
) -> tuple[float, float]:
    """
    Return (SNR [dB], MSE) measured on a selected region.

    Priority: mask > roi > border_fraction > full image.
    `roi` is (row_top, col_left, height, width).
    """
    if original.shape != processed.shape:
        raise ValueError("original and processed must have identical shape")

    o = _extract_region(original, roi=roi, mask=mask, border_fraction=border_fraction)
    p = _extract_region(processed, roi=roi, mask=mask, border_fraction=border_fraction)

    o = o.astype(np.float64)
    p = p.astype(np.float64)

    signal = np.mean(o ** 2)
    noise = np.mean((o - p) ** 2)
    mse = noise
    snr = float("inf") if noise <= 1e-30 else 10.0 * np.log10(signal / noise)
    return snr, mse
