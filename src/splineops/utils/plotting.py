"""
splineops.utils.plotting
========================
Matplotlib helpers used by the example gallery and tutorials.

All functions accept NumPy arrays shaped (H, W) in *any* numeric range
(they handle normalisation internally) and never mutate their inputs.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Sequence, Tuple, Union

__all__ = [
    "plot_resized_image",
    "plot_recovered_image",
    "plot_difference_image",
]

_ZoomT = Union[Sequence[float], Tuple[float, float], float]


# -----------------------------------------------------------------------------#
# Internal utilities
# -----------------------------------------------------------------------------#

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Scale *arr* linearly to 0–255 and cast to uint8 (for display only)."""
    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        arr_norm = (arr - a_min) / (a_max - a_min)
    else:  # constant image – avoid divide-by-zero
        arr_norm = np.zeros_like(arr, dtype=np.float64)
    return (arr_norm * 255).astype(np.uint8)


# -----------------------------------------------------------------------------#
# Public plotting helpers
# -----------------------------------------------------------------------------#

def plot_resized_image(
    original: np.ndarray,
    resized: np.ndarray,
    method: str,
    zoom_factors: _ZoomT,
    time_elapsed: float,
) -> None:
    """Show *resized* on a padded canvas (if zoom < 1) with title & timing."""
    zs = (zoom_factors, zoom_factors) if np.isscalar(zoom_factors) else zoom_factors
    zoom_out = any(z < 1.0 for z in zs)

    orig_u8 = _to_uint8(original)
    resized_u8 = _to_uint8(resized)

    if zoom_out:
        canvas = np.full_like(orig_u8, 255, dtype=np.uint8)  # white background
        rh, rw = resized_u8.shape
        canvas[:rh, :rw] = resized_u8
        display = canvas
    else:
        display = resized_u8

    plt.figure(figsize=(5, 5))
    plt.imshow(display, cmap="gray", aspect="equal")
    plt.title(
        f"{method.capitalize()} Resized\n"
        f"Zoom: {zs}, Time: {time_elapsed:.4f}s"
    )
    plt.axis("off")
    plt.show()


def plot_recovered_image(recovered: np.ndarray) -> None:
    """Display the image obtained after resizing back to the original shape."""
    plt.figure(figsize=(6, 5))
    plt.imshow(recovered, cmap="gray", aspect="equal")
    plt.title("Recovered Image")
    plt.axis("off")
    plt.show()


def plot_difference_image(
    original: np.ndarray,
    recovered: np.ndarray,
    snr: float,
    mse: float,
    *,
    vmin: float = -0.8,
    vmax: float = 0.8,
) -> None:
    """Visualise *original – recovered* with a diverging colour map and colourbar.

    The fixed `vmin`/`vmax` keeps scales consistent across multiple plots;
    adjust them if you need a different dynamic range.
    """
    diff = original - recovered
    h, w = diff.shape
    aspect = h / float(w)

    fig_w = 6.0
    fig_h = fig_w * aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(diff, cmap="bwr", aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(f"Difference\nSNR: {snr:.2f} dB, MSE: {mse:.2e}")
    ax.axis("off")

    # Add a colourbar whose height matches the image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Difference (units)")

    plt.tight_layout()
    plt.show()
