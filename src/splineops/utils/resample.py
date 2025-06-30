"""
splineops.utils.resample
========================

Convenience wrappers built on top of :pymeth:`splineops.resize.resize`
and SciPy’s ndimage.zoom, plus helpers that combine resizing and basic
error metrics.
"""

from __future__ import annotations

import time
from typing import Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import zoom as _scipy_zoom

from ..resize.resize import resize                 # core splineops routine
from .metrics import compute_snr_and_mse_cropped   # central-region metrics

__all__: list[str] = [
    "resize_with_scipy_zoom",
    "resize_and_compute_metrics",
    "resize_multichannel",
]

_ZoomT = Union[Sequence[float], Tuple[float, float], float]


# --------------------------------------------------------------------------- #
# SciPy-based reference implementation
# --------------------------------------------------------------------------- #

def resize_with_scipy_zoom(
    img: np.ndarray,
    zoom_factors: _ZoomT,
    *,
    degree: int = 3,
    border_fraction: float = 0.2,
):
    """Resize *img* with **SciPy** then undo the zoom and return metrics."""
    t0 = time.perf_counter()
    out = _scipy_zoom(img, zoom_factors, order=degree)
    elapsed = time.perf_counter() - t0

    recovered = _scipy_zoom(out, 1.0 / np.asarray(zoom_factors), order=degree)
    snr, mse = compute_snr_and_mse_cropped(img, recovered, border_fraction)
    return out, recovered, snr, mse, elapsed


# --------------------------------------------------------------------------- #
# One-shot helper that works with any splineops resize *method*
# --------------------------------------------------------------------------- #

def resize_and_compute_metrics(
    img: np.ndarray,
    *,
    method: str,
    degree: int,
    zoom_factors: _ZoomT,
    border_fraction: float = 0.2,
):
    """
    Resize *img* with the given splineops *method* (or SciPy for ``method="scipy"``),
    resize back to the original shape, and return

    ``(resized, recovered, snr, mse, elapsed_time)``
    """
    if np.isscalar(zoom_factors):
        zoom_factors = (zoom_factors, zoom_factors)

    if method == "scipy":
        return resize_with_scipy_zoom(img, zoom_factors,
                                      degree=degree, border_fraction=border_fraction)

    t0 = time.perf_counter()
    resized = resize(img, zoom_factors=zoom_factors, degree=degree, method=method)
    elapsed = time.perf_counter() - t0

    recovered = resize(resized, output_size=img.shape, degree=degree, method=method)
    snr, mse = compute_snr_and_mse_cropped(img, recovered, border_fraction)
    return resized, recovered, snr, mse, elapsed


# --------------------------------------------------------------------------- #
# Multi-channel RGB / N-channel convenience wrapper
# --------------------------------------------------------------------------- #

def resize_multichannel(
    img: np.ndarray,
    zoom: _ZoomT,
    *,
    degree: int = 3,
    method: str = "interpolation",
    modes: str | Tuple[str, ...] = "mirror",
) -> np.ndarray:
    """
    Resize an H×W×C image **channel-wise** with splineops and return uint8 output.

    Parameters
    ----------
    img
        Input array normalised to [0, 1] and shaped (H, W, C).
    zoom
        Scalar or pair of zoom factors (for height and width).
    degree, method, modes
        Passed straight to :pyfunc:`splineops.resize.resize`.

    Returns
    -------
    np.ndarray
        Resized image in the range [0, 255] with ``dtype=uint8``.
    """
    if img.ndim != 3:
        raise ValueError("Expected an H×W×C array")

    if np.isscalar(zoom):
        zoom = (zoom, zoom)

    channels = [
        resize(img[..., c], zoom_factors=zoom, degree=degree,
               method=method, modes=modes)
        for c in range(img.shape[2])
    ]
    out = np.stack(channels, axis=-1)
    return (np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)
