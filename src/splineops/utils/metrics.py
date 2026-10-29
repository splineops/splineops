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
        mask = np.asarray(mask)
        if mask.dtype != np.bool_:
            raise TypeError("mask must have a boolean dtype")
        if mask.shape != a.shape[:2]:
            raise ValueError("mask must match spatial shape of input")
        return a[mask]
    if roi is not None:
        r, c, h, w = roi
        return a[r : r + h, c : c + w]
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

    Inputs must be finite real arrays and the selected region must be non-empty.
    Exact matches have infinite SNR; a zero reference with nonzero error has
    negative infinite SNR. SNR is independent of the signal's physical units,
    even when the corresponding MSE exceeds float64's representable range.
    """
    original = np.asarray(original)
    processed = np.asarray(processed)
    if original.shape != processed.shape:
        raise ValueError("original and processed must have identical shape")

    o = _extract_region(original, roi=roi, mask=mask, border_fraction=border_fraction)
    p = _extract_region(processed, roi=roi, mask=mask, border_fraction=border_fraction)

    if o.size == 0:
        raise ValueError("The selected region must be non-empty.")
    for values in (o, p):
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise TypeError("Images must have real numeric dtypes.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Images must contain only finite values.")
    with np.errstate(over="ignore", under="ignore"):
        o = o.astype(np.float64, copy=False)
        p = p.astype(np.float64, copy=False)
    if not np.all(np.isfinite(o)) or not np.all(np.isfinite(p)):
        raise ValueError("Images must remain finite when converted to float64.")

    signal_scale = np.max(np.abs(o))
    # Subtract before scaling to preserve small differences between nearby values.
    # Opposite-sign extremes can exceed float64; scale first only in that case.
    with np.errstate(over="ignore", under="ignore"):
        difference = o - p
        error_scale = np.max(np.abs(difference))
        if error_scale == 0:
            return float("inf"), 0.0
        if np.isfinite(error_scale):
            normalized_error = difference / error_scale
        else:
            error_scale = max(signal_scale, np.max(np.abs(p)))
            normalized_error = o / error_scale - p / error_scale
        error_power = np.mean(normalized_error**2)
        mse = float((error_scale * error_power) * error_scale)
        if signal_scale == 0:
            return float("-inf"), mse
        signal_power = np.mean((o / signal_scale) ** 2)
    snr = 20.0 * (np.log10(signal_scale) - np.log10(error_scale))
    snr += 10.0 * (np.log10(signal_power) - np.log10(error_power))
    return float(snr), mse
