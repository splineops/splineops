from __future__ import annotations
import time
import numpy as np
from scipy.ndimage import zoom as scipy_zoom
from ..resize.resize import resize
from .metrics import compute_snr_and_mse_cropped

__all__ = [
    "resize_with_scipy_zoom",
    "resize_and_compute_metrics",
]

def resize_with_scipy_zoom(
    img: np.ndarray,
    zoom_factors: tuple[float, float] | float,
    degree: int = 3,
    border_fraction: float = 0.2,
):
    t0 = time.perf_counter()
    out = scipy_zoom(img, zoom_factors, order=degree)
    elapsed = time.perf_counter() - t0

    recovered = scipy_zoom(out, 1.0 / np.array(zoom_factors), order=degree)
    snr, mse = compute_snr_and_mse_cropped(img, recovered, border_fraction)
    return out, recovered, snr, mse, elapsed


def resize_and_compute_metrics(
    img: np.ndarray,
    method: str,
    degree: int,
    zoom_factors: tuple[float, float] | float,
    border_fraction: float = 0.2,
):
    if np.isscalar(zoom_factors):
        zoom_factors = (zoom_factors, zoom_factors)

    if method == "scipy":
        return resize_with_scipy_zoom(img, zoom_factors, degree, border_fraction)

    t0 = time.perf_counter()
    resized = resize(img, zoom_factors=zoom_factors, degree=degree, method=method)
    elapsed = time.perf_counter() - t0

    recovered = resize(resized, output_size=img.shape, degree=degree, method=method)
    snr, mse = compute_snr_and_mse_cropped(img, recovered, border_fraction)
    return resized, recovered, snr, mse, elapsed
