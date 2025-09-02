# splineops/src/splineops/utils/resample.py

"""
splineops.utils.resample
========================
Resize helpers on top of ``splineops.resize.resize`` and SciPy’s
``ndimage.zoom``.  (Lazy‐importing avoids circular-import issues.)
"""

from __future__ import annotations
import time
from typing import Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import zoom as _scipy_zoom
from .metrics import compute_snr_and_mse_cropped

__all__ = [
    "resize_with_scipy_zoom",
    "resize_and_compute_metrics",
    "resize_multichannel",
]

_ZoomT = Union[Sequence[float], Tuple[float, float], float]

# -----------------------------------------------------------------------------#
# SciPy reference routine (unchanged)
# -----------------------------------------------------------------------------#
def resize_with_scipy_zoom(
    img: np.ndarray,
    zoom_factors: _ZoomT,
    *,
    scipy_order: int = 3,
    border_fraction: float = 0.2,
):
    t0 = time.perf_counter()
    out = _scipy_zoom(img, zoom_factors, order=scipy_order)
    elapsed = time.perf_counter() - t0

    recovered = _scipy_zoom(out, 1.0 / np.asarray(zoom_factors), order=scipy_order)
    snr, mse = compute_snr_and_mse_cropped(img, recovered, border_fraction)
    return out, recovered, snr, mse, elapsed


# -----------------------------------------------------------------------------#
# Lazy-import helper
# -----------------------------------------------------------------------------#
def _resize(*args, **kwargs):
    """Import `resize` only when actually called (breaks circular imports)."""
    from ..resize.resize import resize  # local import!
    return resize(*args, **kwargs)


# -----------------------------------------------------------------------------#
# Generic wrapper for any splineops preset
# -----------------------------------------------------------------------------#
def resize_and_compute_metrics(
    img: np.ndarray,
    *,
    method: str,
    zoom_factors: _ZoomT,
    border_fraction: float = 0.2,
    scipy_order: int = 3,
):
    if np.isscalar(zoom_factors):
        zoom_factors = (zoom_factors, zoom_factors)

    if method == "scipy":
        return resize_with_scipy_zoom(
            img, zoom_factors, scipy_order=scipy_order, border_fraction=border_fraction
        )

    t0 = time.perf_counter()
    resized = _resize(img, zoom_factors=zoom_factors, method=method)
    elapsed = time.perf_counter() - t0

    recovered = _resize(resized, output_size=img.shape, method=method)
    snr, mse = compute_snr_and_mse_cropped(img, recovered, border_fraction)
    return resized, recovered, snr, mse, elapsed


# -----------------------------------------------------------------------------#
# Channel-wise helper for RGB / N-channel data
# -----------------------------------------------------------------------------#
def resize_multichannel(
    img: np.ndarray,
    zoom: _ZoomT,
    *,
    method: str = "cubic",
    modes: str | Tuple[str, ...] = "mirror",
) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError("Expected an H×W×C array")

    if np.isscalar(zoom):
        zoom = (zoom, zoom)

    channels = [
        _resize(img[..., c], zoom_factors=zoom, method=method, modes=modes)
        for c in range(img.shape[2])
    ]
    out = np.stack(channels, axis=-1)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
