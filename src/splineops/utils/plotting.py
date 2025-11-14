# splineops/src/splineops/utils/plotting.py
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
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Sequence, Tuple, Union, Optional
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from PIL import Image

__all__ = [
    "plot_resized_image",
    "plot_recovered_image",
    "plot_difference_image",
    "show_roi_zoom",
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
    roi: Optional[Tuple[int, int, int, int]] = None,
    mask: Optional[np.ndarray] = None,
    title_prefix: str = "Difference",
    cmap_mode: str = "bw",  # "bw" (default) or "bwr"
) -> None:
    """
    Visualise *original – recovered* with a diverging colour map and colourbar.

    If `roi` or `mask` is provided, the plot shows only that region.
    `roi` takes (row_top, col_left, height, width). `mask` must be H×W boolean.
    """
    diff_full = original - recovered

    if mask is not None:
        # Plot a compact view of just the masked pixels by cropping to mask bbox
        if mask.shape != diff_full.shape:
            # if original had channels already handled before this point
            raise ValueError("mask must match spatial shape of the images")
        rows, cols = np.where(mask)
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        diff = diff_full[r0:r1, c0:c1]
        # keep values outside mask out of view by zeroing them
        local_mask = mask[r0:r1, c0:c1]
        diff = np.where(local_mask, diff, 0.0)
        region_label = " (masked)"
    elif roi is not None:
        r, c, h, w = roi
        diff = diff_full[r:r + h, c:c + w]
        region_label = " (ROI)"
    else:
        diff = diff_full
        region_label = ""

    h, w = diff.shape
    aspect = h / float(w)

    fig_w = 6.0
    fig_h = fig_w * aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    if cmap_mode == "bw":
        # Black → mid-gray (0) → white, to keep sign in monochrome
        cmap_obj = LinearSegmentedColormap.from_list(
            "bw_div",
            [
                (0.0, "black"),
                (0.5, "0.5"),
                (1.0, "white"),
            ],
        )
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        im = ax.imshow(diff, cmap=cmap_obj, aspect="equal", norm=norm)
    else:
        im = ax.imshow(
            diff,
            cmap="bwr",
            aspect="equal",
            vmin=vmin,
            vmax=vmax,
        )

    ax.set_title(f"{title_prefix}{region_label}\nSNR: {snr:.2f} dB, MSE: {mse:.2e}")
    ax.axis("off")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Difference (units)")

    plt.tight_layout()
    plt.show()


def show_roi_zoom(
    img_source: Union[str, Path, np.ndarray],
    roi_height_frac: float = 1 / 3,
    grayscale: bool = True,
    roi_xy: Optional[Tuple[int, int]] = None,
    ax_titles: Optional[Tuple[str, str]] = None,
    fig_size: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """
    Display an image with a square ROI and a magnified (nearest-neighbour) view
    of that ROI.  Both panels share *identical display height*.

    Parameters
    ----------
    img_source : str | Path | ndarray
        URL, local path, or already-loaded NumPy array (H×W×C or H×W).
    roi_height_frac : float, default 1/3
        ROI side length as a fraction of the image height (0 < frac < 1).
    grayscale : bool, default True
        If True → convert colour images to grayscale and plot with `cmap="gray"`.
    roi_xy : (row, col) or None
        Top-left corner of the ROI.  If None → ROI is centred.
    ax_titles : (str, str) or None
        Titles for the (full image, magnified ROI) axes.  Defaults are used if
        None.
    fig_size : (width, height) or None
        Size in inches; if None → auto-scaled to keep reasonable aspect.

    Returns
    -------
    fig, (ax_left, ax_right)
        The Matplotlib figure and its two axes.
    """
    # ------------------------------------------------------------------ #
    # 1. Load image → NumPy array in [0, 1]                              #
    # ------------------------------------------------------------------ #
    if isinstance(img_source, np.ndarray):
        img = img_source.astype(np.float64)
    else:
        src = str(img_source)
        if src.startswith(("http://", "https://")):
            # Use stdlib urllib instead of requests
            with urlopen(src, timeout=10) as resp:
                data = resp.read()
            img = np.asarray(Image.open(BytesIO(data)), dtype=np.float64)
        else:
            img = np.asarray(Image.open(Path(src)), dtype=np.float64)

    if img.max() > 1.0:
        img /= 255.0

    # ------------------------------------------------------------------ #
    # 2. Optional grayscale conversion                                   #
    # ------------------------------------------------------------------ #
    if grayscale and img.ndim == 3 and img.shape[2] != 1:
        img = (
            0.2989 * img[..., 0] +
            0.5870 * img[..., 1] +
            0.1140 * img[..., 2]
        )
        img = img[..., None]  # keep channel dim for consistency

    # Drop trailing channel dim for plotting if grayscale
    plot_img = img.squeeze() if img.shape[-1] == 1 else img

    h_img, w_img = plot_img.shape[:2]

    # ------------------------------------------------------------------ #
    # 3. Choose square ROI                                               #
    # ------------------------------------------------------------------ #
    # Start from the requested fractional size; keep it as-is (up to
    # integer rounding), rather than shrinking it to divide h_img.
    roi_size = max(1, int(round(h_img * roi_height_frac)))
    roi_size = min(roi_size, h_img)  # clamp in case of tiny images / big frac

    if roi_xy is None:
        row0 = h_img // 2 - roi_size // 2
        col0 = w_img // 2 - roi_size // 2
    else:
        row0 = np.clip(roi_xy[0], 0, h_img - roi_size)
        col0 = np.clip(roi_xy[1], 0, w_img - roi_size)

    if plot_img.ndim == 2:
        roi = plot_img[row0:row0 + roi_size, col0:col0 + roi_size]
    else:
        roi = plot_img[row0:row0 + roi_size, col0:col0 + roi_size, :]

    # ------------------------------------------------------------------ #
    # 4. Magnify ROI with nearest-neighbour                              #
    # ------------------------------------------------------------------ #
    mag = max(1, h_img // roi_size)
    roi_big = np.repeat(np.repeat(roi, mag, axis=0), mag, axis=1)

    # ------------------------------------------------------------------ #
    # 5. Plot – widths proportional to pixel widths                      #
    # ------------------------------------------------------------------ #
    if fig_size is None:
        # Slightly taller figure so titles & images never get clipped,
        # even for images that are relatively tall.
        fig_size = (10.0, 5.5)

    fig, ax = plt.subplots(
        1, 2,
        figsize=fig_size,
        gridspec_kw={"width_ratios": [w_img, roi_big.shape[1]]},
        constrained_layout=True,
    )

    # left panel
    ax[0].imshow(plot_img, cmap="gray" if grayscale else None)
    ax[0].add_patch(
        patches.Rectangle(
            (col0, row0),
            roi_size,
            roi_size,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
    )
    ax[0].set_aspect("equal")
    ax[0].axis("off")
    ax[0].set_title(ax_titles[0] if ax_titles else "Image with ROI")

    # right panel
    ax[1].imshow(
        roi_big,
        cmap="gray" if grayscale else None,
        interpolation="nearest",
    )
    ax[1].set_aspect("equal")
    ax[1].axis("off")
    ax[1].set_title(ax_titles[1] if ax_titles else f"ROI x{mag} (nearest)")

    plt.show()
    return fig, ax
