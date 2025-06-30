from __future__ import annotations
import numpy as np
from .image import crop_to_central_region


def compute_snr_and_mse_cropped(
    original: np.ndarray,
    processed: np.ndarray,
    border_fraction: float = 0.2,
) -> tuple[float, float]:
    """Return (SNR [dB], MSE) measured on a central region.

    Parameters
    ----------
    original, processed
        Images with the same shape.
    border_fraction
        Fraction to discard on each edge before evaluation.
    """
    o = crop_to_central_region(original, border_fraction)
    p = crop_to_central_region(processed, border_fraction)

    signal = np.mean(o ** 2)
    noise = np.mean((o - p) ** 2)
    mse = noise
    snr = float("inf") if noise <= 1e-30 else 10 * np.log10(signal / noise)
    return snr, mse
