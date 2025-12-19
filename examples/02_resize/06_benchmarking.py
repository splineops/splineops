# sphinx_gallery_start_ignore
# splineops/examples/02_resize/06_benchmarking.py
# sphinx_gallery_end_ignore

"""
Benchmarking
============

This example benchmarks several 2D downsampling methods over a *set* of test
images. For each image we:

1. Downsample by an image-specific zoom factor, then upsample back (round-trip).
2. Measure the runtime of the round-trip (forward + backward).
3. Compute SNR / MSE / SSIM on a local ROI.
4. Visualise the results with ROI-aware zooms: first a 2x2 figure showing the
   original image with its magnified ROI and the SplineOps Antialiasing
   first-pass image with the mapped ROI, then ROI montages of the original
   ROI and each method's first-pass ROI, all magnified with nearest-neighbour.

We compare:

- SciPy ndimage.zoom (cubic).
- SplineOps Standard cubic interpolation.
- SplineOps Cubic Antialiasing (oblique projection).
- OpenCV INTER_CUBIC.
- Pillow BICUBIC.
- scikit-image (resize, cubic).
- scikit-image (resize, cubic + anti_aliasing).
- PyTorch (F.interpolate bicubic, CPU).
- PyTorch (F.interpolate bicubic, antialias=True, CPU).

Notes
-----
- All ops run on grayscale images normalized to [0, 1] for metrics.
- Methods with missing deps are marked "unavailable" in the console and skipped
  from the ROI montages.
"""

# %%
# Imports and Configuration
# -------------------------

from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

# sphinx_gallery_thumbnail_number = 16  # Show the 15th figure as the gallery thumbnail
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from urllib.request import urlopen
from PIL import Image

# Default storage dtype for comparison (change to np.float64 if desired)
DTYPE = np.float32

# ROI / detail-window configuration
ROI_MAG_TARGET = 256           # target height for nearest-neighbour zoom tiles
ROI_TILE_TITLE_FONTSIZE = 12
ROI_SUPTITLE_FONTSIZE = 14  # currently unused, kept for consistency

# Plot appearance for slide-friendly export
PLOT_FIGSIZE = (14, 7)      # same 2:1 ratio as (10, 5), just larger
PLOT_TITLE_FONTSIZE = 18
PLOT_LABEL_FONTSIZE = 18
PLOT_TICK_FONTSIZE = 18
PLOT_LEGEND_FONTSIZE = 18

# Highlight styles (per method) used everywhere (ROI montages + plots)
HIGHLIGHT_STYLE = {
    "SplineOps Standard cubic": {
        "color": "#C2410C",
        "lw": 3.0,
    },
    "SplineOps Antialiasing cubic": {
        "color": "#BE185D",
        "lw": 3.0,
    },
}

AA_METHOD_LABEL = "SplineOps Antialiasing cubic"
AA_COLOR = HIGHLIGHT_STYLE[AA_METHOD_LABEL]["color"]

def fmt_ms(seconds: float) -> str:
    """Format seconds as a short 'X.X ms' string."""
    return f"{seconds * 1000.0:.1f} ms"

# Benchmark configuration
N_TRIALS = 10

# Optional deps
try:
    import cv2
    _HAS_CV2 = True
    # Undo OpenCV's Qt plugin path override to keep using the system/PyQt plugins
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
except Exception:
    _HAS_CV2 = False

try:
    from scipy.ndimage import zoom as _ndi_zoom
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from skimage.transform import resize as _sk_resize
    from skimage.metrics import structural_similarity as _ssim  # noqa: F401
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
    _ssim = None

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

# SplineOps
try:
    from splineops.resize import resize as sp_resize
    _HAS_SPLINEOPS = True
except Exception as e:
    _HAS_SPLINEOPS = False
    _SPLINEOPS_IMPORT_ERR = str(e)

try:
    from splineops.utils.specs import print_runtime_context
    _HAS_SPECS = True
except Exception:
    print_runtime_context = None
    _HAS_SPECS = False


# %%
# Kodak Test Set Configuration
# ----------------------------

KODAK_BASE = "https://r0k.us/graphics/kodak/kodak"
KODAK_IMAGES = [
    ("kodim05", f"{KODAK_BASE}/kodim05.png"),
    ("kodim07", f"{KODAK_BASE}/kodim07.png"),
    ("kodim14", f"{KODAK_BASE}/kodim14.png"),
    ("kodim15", f"{KODAK_BASE}/kodim15.png"),
    ("kodim19", f"{KODAK_BASE}/kodim19.png"),
    ("kodim22", f"{KODAK_BASE}/kodim22.png"),
    ("kodim23", f"{KODAK_BASE}/kodim23.png"),
]

# Per-image zoom + ROI config
IMAGE_CONFIG: Dict[str, Dict[str, object]] = {
    "kodim05": dict(
        zoom=0.15,
        roi_size_px=256,
        roi_center_frac=(0.75, 0.5),
    ),
    "kodim07": dict(
        zoom=0.15,
        roi_size_px=256,
        roi_center_frac=(0.40, 0.50),
    ),
    "kodim14": dict(
        zoom=0.3,
        roi_size_px=256,
        roi_center_frac=(0.75, 0.75),
    ),
    "kodim15": dict(
        zoom=0.3,
        roi_size_px=256,
        roi_center_frac=(0.30, 0.55),
    ),
    "kodim19": dict(
        zoom=0.2,
        roi_size_px=256,
        roi_center_frac=(0.65, 0.35),
    ),
    "kodim22": dict(
        zoom=0.2,
        roi_size_px=256,
        roi_center_frac=(0.50, 0.25),
    ),
    "kodim23": dict(
        zoom=0.15,
        roi_size_px=256,
        roi_center_frac=(0.40, 0.65),
    ),
}


def _load_kodak_gray(url: str) -> np.ndarray:
    """
    Download a Kodak image, convert to grayscale [0, 1] in DTYPE (float32).
    """
    with urlopen(url, timeout=10) as resp:
        img = Image.open(resp)
    arr = np.asarray(img, dtype=np.float64)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr01 = arr / 255.0
        gray = (
            0.2989 * arr01[..., 0]
            + 0.5870 * arr01[..., 1]
            + 0.1140 * arr01[..., 2]
        )
    else:
        vmax = float(arr.max()) or 1.0
        gray = arr / vmax

    return np.clip(gray, 0.0, 1.0).astype(DTYPE)


def _load_kodak_rgb(url: str) -> np.ndarray:
    """
    Download a Kodak image as RGB [0, 1] in DTYPE (float32).
    Used only for color visualizations; metrics remain on grayscale.
    """
    with urlopen(url, timeout=10) as resp:
        img = Image.open(resp).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0  # H×W×3 in [0,1]
    return np.clip(arr, 0.0, 1.0).astype(DTYPE, copy=False)


# %%
# Utilities (Metrics, ROI, Plotting)
# ----------------------------------

def _snr_db(x: np.ndarray, y: np.ndarray) -> float:
    num = float(np.sum(x * x, dtype=np.float64))
    den = float(np.sum((x - y) ** 2, dtype=np.float64))
    if den == 0.0:
        return float("inf")
    if num == 0.0:
        return -float("inf")
    return 10.0 * math.log10(num / den)


def _roi_rect_from_frac(
    shape: Tuple[int, int],
    roi_size_px: int,
    center_frac: Tuple[float, float],
) -> Tuple[int, int, int, int]:
    """Compute a square ROI inside an image, centred at fractional coordinates."""
    H, W = shape[:2]
    row_frac, col_frac = center_frac

    size = int(min(roi_size_px, H, W))
    if size < 1:
        size = min(H, W)

    center_r = int(round(row_frac * H))
    center_c = int(round(col_frac * W))

    row_top = int(np.clip(center_r - size // 2, 0, H - size))
    col_left = int(np.clip(center_c - size // 2, 0, W - size))

    return row_top, col_left, size, size


def _crop_roi(arr: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, h, w = rect
    return arr[r0 : r0 + h, c0 : c0 + w]


def _nearest_big(roi: np.ndarray, target_h: int = ROI_MAG_TARGET) -> np.ndarray:
    """Enlarge a small ROI with nearest-neighbour so its height is ~target_h."""
    h, w = roi.shape
    mag = max(1, int(round(target_h / max(h, 1))))
    return np.repeat(np.repeat(roi, mag, axis=0), mag, axis=1)


def _diff_normalized(
    orig: np.ndarray,
    rec: np.ndarray,
    *,
    max_abs: Optional[float] = None,
) -> np.ndarray:
    """
    Normalize signed difference (rec - orig) into [0,1] for display.

    0.5 = no difference, >0.5 positive, <0.5 negative.

    If max_abs is provided, it is used as a shared scale (same range across tiles).
    """
    diff = rec.astype(np.float64, copy=False) - orig.astype(np.float64, copy=False)

    if max_abs is None:
        max_abs = float(np.max(np.abs(diff)))
    else:
        max_abs = float(max_abs)

    if max_abs <= 0.0:
        return 0.5 * np.ones_like(diff, dtype=DTYPE)

    norm = 0.5 + 0.5 * diff / max_abs
    norm = np.clip(norm, 0.0, 1.0)
    return norm.astype(DTYPE, copy=False)

def _show_initial_original_vs_aa(
    gray: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    aa_first: np.ndarray,
    z: float,
    degree_label: str,
) -> None:
    """
    Plot a 2x2 figure:

    Row 1:
      - Original image with red ROI box
      - Magnified original ROI

    Row 2:
      - SplineOps Antialiasing first-pass resized image on white canvas with
        mapped ROI box
      - Magnified mapped ROI from the Antialiasing first-pass
    """
    H, W = gray.shape
    row0, col0, roi_h, roi_w = roi_rect

    roi_orig = _crop_roi(gray, roi_rect)
    roi_orig_big = _nearest_big(roi_orig, ROI_MAG_TARGET)

    H1, W1 = aa_first.shape
    center_r = row0 + roi_h / 2.0
    center_c = col0 + roi_w / 2.0

    roi_h_res = max(1, int(round(roi_h * z)))
    roi_w_res = max(1, int(round(roi_w * z)))

    if roi_h_res > H1 or roi_w_res > W1:
        aa_roi = aa_first
        row_top_res = 0
        col_left_res = 0
        roi_h_res = H1
        roi_w_res = W1
    else:
        center_r_res = int(round(center_r * z))
        center_c_res = int(round(center_c * z))
        row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, H1 - roi_h_res))
        col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, W1 - roi_w_res))
        aa_roi = aa_first[
            row_top_res : row_top_res + roi_h_res,
            col_left_res : col_left_res + roi_w_res,
        ]

    aa_roi_big = _nearest_big(aa_roi, ROI_MAG_TARGET)

    canvas_aa = np.ones_like(gray, dtype=aa_first.dtype)
    h_copy = min(H, H1)
    w_copy = min(W, W1)
    canvas_aa[:h_copy, :w_copy] = aa_first[:h_copy, :w_copy]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Row 1, left: original with ROI box
    ax = axes[0, 0]
    ax.imshow(gray, cmap="gray", interpolation="nearest", aspect="equal")
    rect = patches.Rectangle(
        (col0, row0),
        roi_w,
        roi_h,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title(
        f"Original image with ROI ({H}×{W} px)",
        fontsize=ROI_TILE_TITLE_FONTSIZE,
    )
    ax.axis("off")

    # Row 1, right: magnified original ROI
    ax = axes[0, 1]
    ax.imshow(roi_orig_big, cmap="gray", interpolation="nearest", aspect="equal")
    ax.set_title(
        f"Original ROI ({roi_h}×{roi_w} px, NN magnified)",
        fontsize=ROI_TILE_TITLE_FONTSIZE,
    )
    ax.axis("off")

    # Row 2, left: Antialiasing first-pass on canvas with mapped ROI box
    ax = axes[1, 0]
    ax.imshow(canvas_aa, cmap="gray", interpolation="nearest", aspect="equal")
    if row_top_res < h_copy and col_left_res < w_copy:
        box_h = min(roi_h_res, h_copy - row_top_res)
        box_w = min(roi_w_res, w_copy - col_left_res)
        rect_aa = patches.Rectangle(
            (col_left_res, row_top_res),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect_aa)

    ax.set_title(
        f"{AA_METHOD_LABEL}\n(zoom ×{z:g}, {H1}×{W1} px)",
        fontsize=ROI_TILE_TITLE_FONTSIZE,
        color=AA_COLOR,
        fontweight="bold",
        multialignment="center",
    )
    ax.axis("off")

    # Row 2, right: magnified Antialiasing ROI
    ax = axes[1, 1]
    ax.imshow(aa_roi_big, cmap="gray", interpolation="nearest", aspect="equal")
    ax.set_title(
        f"Antialiasing ROI ({roi_h_res}×{roi_w_res} px, NN magnified)",
        fontsize=ROI_TILE_TITLE_FONTSIZE,
    )
    ax.axis("off")

    fig.tight_layout()
    plt.show()


def _nearest_big_color(roi: np.ndarray, target_h: int = ROI_MAG_TARGET) -> np.ndarray:
    """
    Enlarge a small color ROI (H×W×3) with nearest-neighbour so its height
    is ~target_h pixels.
    """
    h, w, _ = roi.shape
    mag = max(1, int(round(target_h / max(h, 1))))
    return np.repeat(np.repeat(roi, mag, axis=0), mag, axis=1)


def show_intro_color(
    original_rgb: np.ndarray,
    shrunk_rgb: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    zoom: float,
    label: str,
    degree_label: str,
) -> None:
    """
    2×2 figure mirroring the benchmarking intro style, but in color.
    """
    H, W, _ = original_rgb.shape
    row0, col0, roi_h, roi_w = roi_rect

    roi_orig = original_rgb[row0:row0 + roi_h, col0:col0 + roi_w, :]
    roi_orig_big = _nearest_big_color(roi_orig, target_h=ROI_MAG_TARGET)

    Hs, Ws, _ = shrunk_rgb.shape
    center_r = row0 + roi_h / 2.0
    center_c = col0 + roi_w / 2.0

    roi_h_res = max(1, int(round(roi_h * zoom)))
    roi_w_res = max(1, int(round(roi_w * zoom)))

    if roi_h_res > Hs or roi_w_res > Ws:
        roi_shrunk = shrunk_rgb
        row_top_res = 0
        col_left_res = 0
        roi_h_res = Hs
        roi_w_res = Ws
    else:
        center_r_res = int(round(center_r * zoom))
        center_c_res = int(round(center_c * zoom))
        row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, Hs - roi_h_res))
        col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, Ws - roi_w_res))
        roi_shrunk = shrunk_rgb[
            row_top_res:row_top_res + roi_h_res,
            col_left_res:col_left_res + roi_w_res,
            :
        ]

    roi_shrunk_big = _nearest_big_color(roi_shrunk, target_h=ROI_MAG_TARGET)

    canvas = np.ones_like(original_rgb)
    h_copy = min(H, Hs)
    w_copy = min(W, Ws)
    canvas[:h_copy, :w_copy, :] = shrunk_rgb[:h_copy, :w_copy, :]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.imshow(np.clip(original_rgb, 0.0, 1.0))
    rect = patches.Rectangle(
        (col0, row0),
        roi_w,
        roi_h,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title(
        f"Original image with ROI ({H}×{W} px)",
        fontsize=ROI_TILE_TITLE_FONTSIZE,
    )
    ax.axis("off")

    ax = axes[0, 1]
    ax.imshow(np.clip(roi_orig_big, 0.0, 1.0))
    ax.set_title(
        f"Original ROI ({roi_h}×{roi_w} px, NN magnified)",
        fontsize=ROI_TILE_TITLE_FONTSIZE,
    )
    ax.axis("off")

    ax = axes[1, 0]
    ax.imshow(np.clip(canvas, 0.0, 1.0))
    if row_top_res < h_copy and col_left_res < w_copy:
        box_h = min(roi_h_res, h_copy - row_top_res)
        box_w = min(roi_w_res, w_copy - col_left_res)
        rect2 = patches.Rectangle(
            (col_left_res, row_top_res),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect2)
    # Bottom-left: resized image on canvas (change title only here)
    if label == "Antialiasing":
        title_kw = dict(
            fontsize=ROI_TILE_TITLE_FONTSIZE,
            fontweight="bold",
            color=AA_COLOR,
            multialignment="center",
        )
        ax.set_title(
            f"{AA_METHOD_LABEL}\n(zoom ×{zoom:g}, {Hs}×{Ws} px)",
            **title_kw,
        )
    ax.axis("off")

    ax = axes[1, 1]
    ax.imshow(np.clip(roi_shrunk_big, 0.0, 1.0))

    if label == "Antialiasing":
        title_kw = dict(fontsize=ROI_TILE_TITLE_FONTSIZE, fontweight="bold")
        if AA_COLOR is not None:
            title_kw["color"] = AA_COLOR
        ax.set_title(
            f"{label} ROI ({roi_h_res}×{roi_w_res} px, NN magnified)",
            **title_kw,
        )
    else:
        ax.set_title(
            f"{label} ROI ({roi_h_res}×{roi_w_res} px, NN magnified)",
            fontsize=ROI_TILE_TITLE_FONTSIZE,
        )

    ax.axis("off")

    fig.tight_layout()
    plt.show()

def _smart_ylim(
    values: np.ndarray,
    *,
    hi_cap: float | None = None,
    lo_cap: float | None = None,
    pad_frac: float = 0.06,
    iqr_k: float = 1.5,
    q_floor: float = 10.0,   # percentile used when min is an outlier
    min_span: float | None = None,
) -> tuple[float, float] | None:
    """
    Robust y-limits for plots.

    - If the minimum is a strong outlier (below Q1 - k*IQR), use q_floor percentile as the lower bound.
    - Otherwise use the true min.
    - Add a small padding.
    - Optionally clamp/cap.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None

    vmin = float(v.min())
    vmax = float(v.max())

    if vmin == vmax:
        span = float(min_span) if min_span is not None else (1e-3 if vmax <= 1.0 else 1.0)
        lo, hi = vmin - 0.5 * span, vmax + 0.5 * span
    else:
        q1, q3 = np.percentile(v, [25.0, 75.0])
        iqr = float(q3 - q1)

        lo0 = vmin
        if iqr > 0.0:
            low_outlier_thr = float(q1 - iqr_k * iqr)
            if vmin < low_outlier_thr:
                lo0 = float(np.percentile(v, q_floor))

        span = vmax - lo0
        pad = pad_frac * span
        lo, hi = lo0 - pad, vmax + pad

        if min_span is not None and (hi - lo) < float(min_span):
            mid = 0.5 * (hi + lo)
            lo = mid - 0.5 * float(min_span)
            hi = mid + 0.5 * float(min_span)

    if lo_cap is not None:
        lo = max(lo, float(lo_cap))
    if hi_cap is not None:
        hi = min(hi, float(hi_cap))

    if lo >= hi:
        hi = lo + (float(min_span) if min_span is not None else 1e-6)

    return lo, hi

def _highlight_tile(ax, *, color: str, lw: float = 3.0) -> None:
    # Full-axes border, works even with ax.axis("off")
    rect = patches.Rectangle(
        (0, 0), 1, 1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(rect)

# %%
# Round-Trip Backends and Time
# ----------------------------

def _rt_splineops(
    gray: np.ndarray, z: float, preset: str
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """SplineOps standard / antialiasing cubic."""
    if not _HAS_SPLINEOPS:
        return gray, gray, f"SplineOps unavailable: {_SPLINEOPS_IMPORT_ERR}"
    try:
        first = sp_resize(gray, zoom_factors=(z, z), method=preset)
        rec   = sp_resize(first, output_size=gray.shape, method=preset)
        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec   = np.clip(rec,   0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_scipy(
    gray: np.ndarray, z: float
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """SciPy ndimage.zoom, cubic, reflect boundary."""
    if not _HAS_SCIPY:
        return gray, gray, "SciPy not installed"
    try:
        order = 3
        need_prefilter = True

        first = _ndi_zoom(
            gray,
            (z, z),
            order=order,
            prefilter=need_prefilter,
            mode="reflect",
            grid_mode=False,
        )

        Hz, Wz = first.shape
        back = (gray.shape[0] / Hz, gray.shape[1] / Wz)
        rec = _ndi_zoom(
            first,
            back,
            order=order,
            prefilter=need_prefilter,
            mode="reflect",
            grid_mode=False,
        )

        first = np.clip(first, 0.0, 1.0)
        rec   = np.clip(rec,   0.0, 1.0)

        if rec.shape != gray.shape:
            h = min(rec.shape[0], gray.shape[0])
            w = min(rec.shape[1], gray.shape[1])
            r0 = (rec.shape[0] - h) // 2
            r1 = r0 + h
            c0 = (rec.shape[1] - w) // 2
            c1 = c0 + w
            rc = rec[r0:r1, c0:c1]
            g0 = (gray.shape[0] - h) // 2
            g1 = g0 + h
            g2 = (gray.shape[1] - w) // 2
            g3 = g2 + w
            tmp = np.zeros_like(gray)
            tmp[g0:g1, g2:g3] = rc
            rec = tmp

        return first.astype(gray.dtype, copy=False), rec.astype(gray.dtype, copy=False), None
    except Exception as e:
        return gray, gray, str(e)


def _rt_opencv(
    gray: np.ndarray, z: float
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """OpenCV INTER_CUBIC."""
    if not _HAS_CV2:
        return gray, gray, "OpenCV not installed"
    try:
        H, W = gray.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))

        first = cv2.resize(gray, (W1, H1), interpolation=cv2.INTER_CUBIC)
        rec   = cv2.resize(first, (W,  H),  interpolation=cv2.INTER_CUBIC)

        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec   = np.clip(rec,   0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_pillow(
    gray: np.ndarray, z: float
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Pillow BICUBIC on float32 images."""
    try:
        from PIL import Image as _Image

        H, W = gray.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))

        im = _Image.fromarray(gray.astype(np.float32, copy=False), mode="F")

        first_im = im.resize((W1, H1), resample=_Image.Resampling.BICUBIC)
        rec_im   = first_im.resize((W,  H),  resample=_Image.Resampling.BICUBIC)

        first = np.asarray(first_im, dtype=np.float32)
        rec   = np.asarray(rec_im,   dtype=np.float32)

        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec   = np.clip(rec,   0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_skimage(
    gray: np.ndarray, z: float
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """scikit-image resize, cubic, anti_aliasing=False."""
    if not _HAS_SKIMAGE:
        return gray, gray, "scikit-image not installed"
    try:
        H, W = gray.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))

        first = _sk_resize(
            gray,
            (H1, W1),
            order=3,
            anti_aliasing=False,
            preserve_range=True,
            mode="reflect",
        ).astype(np.float64)
        rec = _sk_resize(
            first,
            (H, W),
            order=3,
            anti_aliasing=False,
            preserve_range=True,
            mode="reflect",
        ).astype(np.float64)

        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec   = np.clip(rec,   0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_skimage_aa(
    gray: np.ndarray, z: float
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """scikit-image resize, cubic, anti_aliasing=True on shrink."""
    if not _HAS_SKIMAGE:
        return gray, gray, "scikit-image not installed"
    try:
        H, W = gray.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))

        first = _sk_resize(
            gray,
            (H1, W1),
            order=3,
            anti_aliasing=True,
            preserve_range=True,
            mode="reflect",
        ).astype(np.float64)
        rec = _sk_resize(
            first,
            (H, W),
            order=3,
            anti_aliasing=False,
            preserve_range=True,
            mode="reflect",
        ).astype(np.float64)

        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec   = np.clip(rec,   0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_torch(
    gray: np.ndarray, z: float
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """PyTorch F.interpolate bicubic (antialias=False)."""
    if not _HAS_TORCH:
        return gray, gray, "PyTorch not installed"
    try:
        arr = gray
        if arr.dtype == np.float32:
            t_dtype = torch.float32
        elif arr.dtype == np.float64:
            t_dtype = torch.float64
        else:
            t_dtype = torch.float32
            arr = arr.astype(np.float32, copy=False)

        H, W = arr.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))

        x = torch.from_numpy(arr).to(t_dtype).unsqueeze(0).unsqueeze(0)

        first_t = F.interpolate(
            x,
            size=(H1, W1),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        )
        rec_t = F.interpolate(
            first_t,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
            antialias=False,
        )

        first = first_t[0, 0].detach().cpu().numpy()
        rec   = rec_t[0, 0].detach().cpu().numpy()

        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec   = np.clip(rec,   0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_torch_aa(
    gray: np.ndarray, z: float
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """PyTorch F.interpolate bicubic (antialias=True)."""
    if not _HAS_TORCH:
        return gray, gray, "PyTorch not installed"
    try:
        arr = gray
        if arr.dtype == np.float32:
            t_dtype = torch.float32
        elif arr.dtype == np.float64:
            t_dtype = torch.float64
        else:
            t_dtype = torch.float32
            arr = arr.astype(np.float32, copy=False)

        H, W = arr.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))

        x = torch.from_numpy(arr).to(t_dtype).unsqueeze(0).unsqueeze(0)

        first_t = F.interpolate(
            x,
            size=(H1, W1),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        rec_t = F.interpolate(
            first_t,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        first = first_t[0, 0].detach().cpu().numpy()
        rec   = rec_t[0, 0].detach().cpu().numpy()

        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec   = np.clip(rec,   0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _avg_time(rt_fn, repeats: int = N_TRIALS, warmup: bool = True):
    """
    Run `rt_fn()` (which must return (first, rec, err)) `repeats` times.
    Returns (last_first, last_rec, mean_time, std_time, err).
    """
    if warmup:
        try:
            first, rec, err = rt_fn()
            if err is not None:
                return np.array([]), np.array([]), float("nan"), float("nan"), err
        except Exception as e:
            return np.array([]), np.array([]), float("nan"), float("nan"), str(e)

    times: List[float] = []
    last_first: Optional[np.ndarray] = None
    last_rec: Optional[np.ndarray] = None

    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        first, rec, err = rt_fn()
        if err is not None:
            return np.array([]), np.array([]), float("nan"), float("nan"), err
        dt = time.perf_counter() - t0
        times.append(dt)
        last_first = first
        last_rec = rec

    t_arr = np.asarray(times, dtype=np.float64)
    mean_t = float(t_arr.mean())
    sd_t = float(t_arr.std(ddof=1 if len(t_arr) > 1 else 0))
    assert last_first is not None and last_rec is not None
    return last_first, last_rec, mean_t, sd_t, None


# Methods and backend keys
BENCH_METHODS: List[Tuple[str, str]] = [
    ("SplineOps Standard cubic",       "spl_standard"),
    ("SplineOps Antialiasing cubic",   "spl_aa"),
    ("OpenCV INTER_CUBIC",             "opencv"),
    ("SciPy cubic",                    "scipy"),
    ("Pillow BICUBIC",                 "pillow"),
    ("scikit-image cubic",             "skimage"),
    ("scikit-image cubic (AA)",        "skimage_aa"),
    ("PyTorch bicubic (CPU)",          "torch"),
    ("PyTorch bicubic (AA, CPU)",      "torch_aa"),
]

# Subsets for ROI main vs AA
MAIN_METHOD_LABELS = [
    "SplineOps Standard cubic",
    "SplineOps Antialiasing cubic",
    "OpenCV INTER_CUBIC",
    "SciPy cubic",
    "scikit-image cubic",
    "PyTorch bicubic (CPU)",
]

AA_METHOD_LABELS = [
    "SplineOps Standard cubic",
    "SplineOps Antialiasing cubic",
    "Pillow BICUBIC",
    "scikit-image cubic (AA)",
    "PyTorch bicubic (AA, CPU)",
]


# %%
# Benchmarking Helpers
# --------------------

def benchmark_image(
    img_name: str,
    gray: np.ndarray,
    zoom: float,
    roi_size_px: int,
    roi_center_frac: Tuple[float, float],
    degree_label: str = "Cubic",
) -> Dict[str, object]:
    """
    Run the full round-trip benchmark on one image.

    Returns a dict with:
      - gray, z, roi_rect, roi, degree_label
      - aa_first (for intro plot)
      - roi_tiles (list of (name, tile) for first-pass ROI montages, grayscale)
      - diff_tiles (list of (name, tile) for ROI error montages, grayscale)
      - rows (per-method metrics)
      - diff_max_abs (shared max abs diff used for all diff tiles)
    """
    H, W = gray.shape
    z = zoom

    roi_rect = _roi_rect_from_frac(gray.shape, roi_size_px, roi_center_frac)
    roi = _crop_roi(gray, roi_rect)

    roi_h = roi_rect[2]
    roi_w = roi_rect[3]
    center_r = roi_rect[0] + roi_h / 2.0
    center_c = roi_rect[1] + roi_w / 2.0

    print(
        f"\n=== {img_name} | zoom={z:.3f} | shape={H}×{W} "
        f"| ROI size≈{roi_size_px} px at center_frac={roi_center_frac} ===\n"
    )

    rows: List[Dict[str, object]] = []
    roi_tiles: List[Tuple[str, np.ndarray]] = []
    diff_tiles: List[Tuple[str, np.ndarray]] = []

    # Original ROI tile
    orig_tile = _nearest_big(roi, ROI_MAG_TARGET)
    roi_tiles.append(("Original", orig_tile))

    # Zero-diff baseline (stays at mid-gray)
    diff_zero = 0.5 * np.ones_like(roi, dtype=DTYPE)
    diff_zero_big = _nearest_big(diff_zero, ROI_MAG_TARGET)
    diff_tiles.append(("Original (no diff)", diff_zero_big))

    # Collect per-method recovered ROI for a shared error scale (per image)
    rec_roi_store: List[Tuple[str, np.ndarray]] = []
    diff_max_abs = 0.0

    aa_first_for_plot: Optional[np.ndarray] = None

    header = (
        f"{'Method':<32} {'Time (mean)':>14} {'± SD':>10} "
        f"{'SNR (dB)':>10} {'MSE':>14} {'SSIM':>8}"
    )
    print(header)
    print("-" * len(header))

    for label, backend in BENCH_METHODS:
        if backend == "spl_standard":
            rt_fn = lambda gray=gray, z=z: _rt_splineops(gray, z, "cubic")
        elif backend == "spl_aa":
            rt_fn = lambda gray=gray, z=z: _rt_splineops(gray, z, "cubic-antialiasing")
        elif backend == "opencv":
            rt_fn = lambda gray=gray, z=z: _rt_opencv(gray, z)
        elif backend == "scipy":
            rt_fn = lambda gray=gray, z=z: _rt_scipy(gray, z)
        elif backend == "pillow":
            rt_fn = lambda gray=gray, z=z: _rt_pillow(gray, z)
        elif backend == "skimage":
            rt_fn = lambda gray=gray, z=z: _rt_skimage(gray, z)
        elif backend == "skimage_aa":
            rt_fn = lambda gray=gray, z=z: _rt_skimage_aa(gray, z)
        elif backend == "torch":
            rt_fn = lambda gray=gray, z=z: _rt_torch(gray, z)
        elif backend == "torch_aa":
            rt_fn = lambda gray=gray, z=z: _rt_torch_aa(gray, z)
        else:
            continue

        first, rec, t_mean, t_sd, err = _avg_time(rt_fn, repeats=N_TRIALS, warmup=True)

        if err is not None or first.size == 0 or rec.size == 0:
            print(
                f"{label:<32} {'unavailable':>14} {'':>10} "
                f"{'—':>10} {'—':>14} {'—':>8}"
            )
            rows.append(
                dict(
                    name=label,
                    time=np.nan,
                    sd=np.nan,
                    snr=np.nan,
                    mse=np.nan,
                    ssim=np.nan,
                    err=err,
                )
            )
            continue

        if backend == "spl_aa":
            aa_first_for_plot = first.copy()

        rec_roi = _crop_roi(rec, roi_rect)
        snr = _snr_db(roi, rec_roi)
        mse = float(np.mean((roi - rec_roi) ** 2, dtype=np.float64))

        if _HAS_SKIMAGE and _ssim is not None:
            try:
                dr = float(roi.max() - roi.min())
                if dr <= 0:
                    dr = 1.0
                ssim_val = float(_ssim(roi, rec_roi, data_range=dr))
            except Exception:
                ssim_val = float("nan")
        else:
            ssim_val = float("nan")

        print(
            f"{label:<32} {fmt_ms(t_mean):>14} {fmt_ms(t_sd):>10} "
            f"{snr:>10.2f} {mse:>14.3e} {ssim_val:>8.4f}"
        )

        rows.append(
            dict(
                name=label,
                time=t_mean,
                sd=t_sd,
                snr=snr,
                mse=mse,
                ssim=ssim_val,
                err=None,
            )
        )

        # First-pass ROI tile (in the resized domain)
        H1, W1 = first.shape
        roi_h_res = max(1, int(round(roi_h * z)))
        roi_w_res = max(1, int(round(roi_w * z)))

        if roi_h_res > H1 or roi_w_res > W1:
            first_roi = first
        else:
            center_r_res = int(round(center_r * z))
            center_c_res = int(round(center_c * z))
            row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, H1 - roi_h_res))
            col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, W1 - roi_w_res))
            first_roi = first[
                row_top_res : row_top_res + roi_h_res,
                col_left_res : col_left_res + roi_w_res,
            ]

        tile = _nearest_big(first_roi, ROI_MAG_TARGET)
        roi_tiles.append((label, tile))

        # Store recovered ROI (copy) for shared-scale diff montage
        rec_roi_store.append((label, rec_roi.astype(DTYPE, copy=True)))

        # Update shared max(|diff|) across all methods for this ROI
        d = rec_roi.astype(np.float64, copy=False) - roi.astype(np.float64, copy=False)
        diff_max_abs = max(diff_max_abs, float(np.max(np.abs(d))))

    # Build diff tiles with ONE shared scale (per image) so all tiles are comparable
    diff_max_abs = max(diff_max_abs, 1e-12)
    for label, rec_roi_m in rec_roi_store:
        diff_roi = _diff_normalized(roi, rec_roi_m, max_abs=diff_max_abs)
        diff_tile = _nearest_big(diff_roi, ROI_MAG_TARGET)
        diff_tiles.append((label, diff_tile))

    return dict(
        img_name=img_name,
        gray=gray,
        z=z,
        roi_rect=roi_rect,
        roi=roi,
        degree_label=degree_label,
        aa_first=aa_first_for_plot,
        roi_tiles=roi_tiles,
        diff_tiles=diff_tiles,
        diff_max_abs=diff_max_abs,
        rows=rows,
    )

def show_intro_from_bench(bench: Dict[str, object]) -> None:
    """2×2 introductory figure for SplineOps AA (grayscale)."""
    aa_first = bench["aa_first"]
    if aa_first is None:
        return
    _show_initial_original_vs_aa(
        gray=bench["gray"],              # type: ignore[arg-type]
        roi_rect=bench["roi_rect"],      # type: ignore[arg-type]
        aa_first=aa_first,               # type: ignore[arg-type]
        z=bench["z"],                    # type: ignore[arg-type]
        degree_label=bench["degree_label"],  # type: ignore[arg-type]
    )


def show_roi_montage_main_from_bench(bench: Dict[str, object]) -> None:
    """
    Grayscale ROI montage (main subset):

      Original +
      SplineOps Standard cubic
      SplineOps Antialiasing cubic
      OpenCV INTER_CUBIC
      SciPy cubic
      scikit-image cubic
      PyTorch bicubic (CPU)
    """
    roi_tiles: List[Tuple[str, np.ndarray]] = bench["roi_tiles"]  # type: ignore[assignment]
    if not roi_tiles:
        return

    tile_map = {name: tile for name, tile in roi_tiles}

    names = ["Original"]
    for lbl in MAIN_METHOD_LABELS:
        if lbl in tile_map:
            names.append(lbl)

    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, name in enumerate(names):
        if idx >= rows * cols:
            break

        tile = tile_map[name]
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        ax.imshow(tile, cmap="gray", interpolation="nearest")

        title_kw = dict(fontsize=ROI_TILE_TITLE_FONTSIZE)

        style = HIGHLIGHT_STYLE.get(name)
        if style is not None:
            c = style.get("color", "tab:blue")
            lw = float(style.get("lw", 3.0))
            title_kw.update(color=c, fontweight="bold")
            _highlight_tile(ax, color=c, lw=lw)

        ax.set_title(name, **title_kw)
        ax.axis("off")

    fig.tight_layout()
    plt.show()

def show_roi_montage_aa_from_bench(bench: Dict[str, object]) -> None:
    """
    Grayscale ROI montage (AA subset):

      Original +
      SplineOps Standard cubic
      SplineOps Antialiasing cubic
      Pillow BICUBIC
      scikit-image cubic (AA)
      PyTorch bicubic (AA, CPU)
    """
    roi_tiles: List[Tuple[str, np.ndarray]] = bench["roi_tiles"]  # type: ignore[assignment]
    if not roi_tiles:
        return

    tile_map = {name: tile for name, tile in roi_tiles}

    names = ["Original"]
    for lbl in AA_METHOD_LABELS:
        if lbl in tile_map:
            names.append(lbl)

    cols = 3
    n_tiles = len(names)
    rows = max(1, (n_tiles + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, name in enumerate(names):
        if idx >= rows * cols:
            break

        tile = tile_map[name]
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        ax.imshow(tile, cmap="gray", interpolation="nearest")

        title_kw = dict(fontsize=ROI_TILE_TITLE_FONTSIZE)

        style = HIGHLIGHT_STYLE.get(name)
        if style is not None:
            c = style.get("color", "tab:blue")
            lw = float(style.get("lw", 3.0))
            title_kw.update(color=c, fontweight="bold")
            _highlight_tile(ax, color=c, lw=lw)

        ax.set_title(name, **title_kw)
        ax.axis("off")

    fig.tight_layout()
    plt.show()

def show_error_montage_main_from_bench(bench: Dict[str, object]) -> None:
    """
    Grayscale error montage (main subset):

    (rec - orig) normalized:
      0.5 = 0, <0.5 = negative, >0.5 = positive.
    """
    diff_tiles: List[Tuple[str, np.ndarray]] = bench["diff_tiles"]  # type: ignore[assignment]
    if not diff_tiles:
        return

    tile_map = {name: tile for name, tile in diff_tiles}

    names = ["Original (no diff)"]
    for lbl in MAIN_METHOD_LABELS:
        if lbl in tile_map:
            names.append(lbl)

    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    max_tile_slots = rows * cols - 1  # reserve bottom-right for legend
    num_tiles = min(len(names), max_tile_slots)

    for idx in range(num_tiles):
        name = names[idx]
        tile = tile_map[name]
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        ax.imshow(tile, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)

        title_kw = dict(fontsize=ROI_TILE_TITLE_FONTSIZE)

        style = HIGHLIGHT_STYLE.get(name)
        if style is not None:
            c = style.get("color", "tab:blue")
            lw = float(style.get("lw", 3.0))
            title_kw.update(color=c, fontweight="bold")
            _highlight_tile(ax, color=c, lw=lw)

        ax.set_title(name, **title_kw)
        ax.axis("off")

    # Legend (unchanged)
    ax_leg = axes[-1, -1]
    ax_leg.axis("off")

    H_leg = ROI_MAG_TARGET
    W_leg = 32
    y = np.linspace(1.0, 0.0, H_leg, dtype=np.float32)
    legend_img = np.repeat(y[:, None], W_leg, axis=1)

    ax_leg.imshow(legend_img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
    ax_leg.set_title("Diff legend", fontsize=ROI_TILE_TITLE_FONTSIZE, pad=4)

    ax_leg.text(1.05, 0.05, "-1", transform=ax_leg.transAxes,
                fontsize=8, va="bottom", ha="left")
    ax_leg.text(1.05, 0.50, "0 (no diff)", transform=ax_leg.transAxes,
                fontsize=8, va="center", ha="left")
    ax_leg.text(1.05, 0.95, "+1", transform=ax_leg.transAxes,
                fontsize=8, va="top", ha="left")

    fig.suptitle("Normalized signed difference in ROI", fontsize=ROI_SUPTITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def show_error_montage_aa_from_bench(bench: Dict[str, object]) -> None:
    """
    Grayscale error montage (AA subset):

    (rec - orig) normalized:
      0.5 = 0, <0.5 = negative, >0.5 = positive.
    """
    diff_tiles: List[Tuple[str, np.ndarray]] = bench["diff_tiles"]  # type: ignore[assignment]
    if not diff_tiles:
        return

    tile_map = {name: tile for name, tile in diff_tiles}

    names = ["Original (no diff)"]
    for lbl in AA_METHOD_LABELS:
        if lbl in tile_map:
            names.append(lbl)

    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    max_tile_slots = rows * cols - 1
    num_tiles = min(len(names), max_tile_slots)

    for idx in range(num_tiles):
        name = names[idx]
        tile = tile_map[name]
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        ax.imshow(tile, cmap="gray", interpolation="nearest", vmin=0.0, vmax=1.0)

        title_kw = dict(fontsize=ROI_TILE_TITLE_FONTSIZE)

        style = HIGHLIGHT_STYLE.get(name)
        if style is not None:
            c = style.get("color", "tab:blue")
            lw = float(style.get("lw", 3.0))
            title_kw.update(color=c, fontweight="bold")
            _highlight_tile(ax, color=c, lw=lw)

        ax.set_title(name, **title_kw)
        ax.axis("off")

    # Legend (unchanged)
    ax_leg = axes[-1, -1]
    ax_leg.axis("off")

    H_leg = ROI_MAG_TARGET
    W_leg = 32
    y = np.linspace(1.0, 0.0, H_leg, dtype=np.float32)
    legend_img = np.repeat(y[:, None], W_leg, axis=1)

    ax_leg.imshow(legend_img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
    ax_leg.set_title("Diff legend", fontsize=ROI_TILE_TITLE_FONTSIZE, pad=4)

    ax_leg.text(1.05, 0.05, "-1", transform=ax_leg.transAxes,
                fontsize=8, va="bottom", ha="left")
    ax_leg.text(1.05, 0.50, "0 (no diff)", transform=ax_leg.transAxes,
                fontsize=8, va="center", ha="left")
    ax_leg.text(1.05, 0.95, "+1", transform=ax_leg.transAxes,
                fontsize=8, va="top", ha="left")

    fig.suptitle("Normalized signed difference in ROI", fontsize=ROI_SUPTITLE_FONTSIZE)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

def show_timing_plot_from_bench(bench: Dict[str, object]) -> None:
    """Horizontal bar chart of round-trip timing per method."""
    rows: List[Dict[str, object]] = bench["rows"]  # type: ignore[assignment]
    if not rows:
        return

    gray = bench["gray"]  # type: ignore[index]
    H, W = gray.shape
    z = float(bench["z"])
    degree_label = str(bench["degree_label"])

    valid = [r for r in rows if np.isfinite(r.get("time", np.nan))]
    if not valid:
        return

    names = [r["name"] for r in valid]
    times = np.array([r["time"] for r in valid], dtype=np.float64)
    sds   = np.array([r["sd"]   for r in valid], dtype=np.float64)

    order = np.argsort(times)
    names = [names[i] for i in order]
    times = times[order]
    sds   = sds[order]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    y = np.arange(len(names))

    bars = ax.barh(y, times, xerr=sds, alpha=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=PLOT_TICK_FONTSIZE)
    ax.tick_params(axis="x", labelsize=PLOT_TICK_FONTSIZE)

    ax.set_xlabel(
        f"Round-trip time (s) mean ± sd over {N_TRIALS} runs",
        fontsize=PLOT_LABEL_FONTSIZE,
    )
    ax.set_title(
        f"Timing vs Method (H×W = {H}×{W}, zoom ×{z:g}, degree={degree_label})",
        fontsize=PLOT_TITLE_FONTSIZE,
    )
    ax.grid(axis="x", alpha=0.3)

    # --- Color the METHOD NAMES (yticks) + outline highlighted bars ---
    for tick, name, bar in zip(ax.get_yticklabels(), names, bars.patches):
        style = HIGHLIGHT_STYLE.get(name)
        if style is not None:
            c = style.get("color", "tab:blue")
            lw = float(style.get("lw", 3.0))
            tick.set_color(c)
            tick.set_fontweight("bold")
            # Optional: outline the bar too (nice but not required)
            bar.set_edgecolor(c)
            bar.set_linewidth(lw)

    fig.tight_layout()
    plt.show()

def show_snr_ssim_plot_from_bench(bench: Dict[str, object]) -> None:
    """Combined SNR/SSIM bar chart per method."""
    if not (_HAS_SKIMAGE and _ssim is not None):
        return

    rows: List[Dict[str, object]] = bench["rows"]  # type: ignore[assignment]
    if not rows:
        return

    gray = bench["gray"]  # type: ignore[index]
    H, W = gray.shape
    z = float(bench["z"])
    degree_label = str(bench["degree_label"])

    valid = [
        r for r in rows
        if np.isfinite(r.get("snr", np.nan)) and np.isfinite(r.get("ssim", np.nan))
    ]
    if not valid:
        return

    names = [r["name"] for r in valid]
    snrs  = np.array([r["snr"]  for r in valid], dtype=np.float64)
    ssims = np.array([r["ssim"] for r in valid], dtype=np.float64)

    order = np.argsort(-snrs)
    names = [names[i] for i in order]
    snrs  = snrs[order]
    ssims = ssims[order]

    x = np.arange(len(names))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)

    snr_color  = "tab:blue"
    ssim_color = "tab:green"

    snr_bars = ax1.bar(
        x - width / 2,
        snrs,
        width,
        label="SNR (dB)",
        alpha=0.85,
        color=snr_color,
    )
    ax1.set_ylabel("SNR (dB)", color=snr_color, fontsize=PLOT_LABEL_FONTSIZE)
    ax1.tick_params(axis="y", labelcolor=snr_color, labelsize=PLOT_TICK_FONTSIZE)

    ax1.set_xticks(x)
    ax1.set_xticklabels(
        names,
        rotation=30,
        ha="right",
        fontsize=PLOT_TICK_FONTSIZE,
    )

    ax1.grid(axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ssim_bars = ax2.bar(
        x + width / 2,
        ssims,
        width,
        label="SSIM",
        alpha=0.6,
        color=ssim_color,
    )
    ax2.set_ylabel("SSIM", color=ssim_color, fontsize=PLOT_LABEL_FONTSIZE)
    ax2.tick_params(axis="y", labelcolor=ssim_color, labelsize=PLOT_TICK_FONTSIZE)

    roi_h, roi_w = bench["roi_rect"][2], bench["roi_rect"][3]  # type: ignore[index]

    ax1.set_title(
        f"SNR / SSIM vs Method (ROI = {roi_h}×{roi_w} px, zoom ×{z:g}, degree={degree_label})",
        fontsize=PLOT_TITLE_FONTSIZE,
    )

    handles = [snr_bars[0], ssim_bars[0]]
    labels  = ["SNR (dB)", "SSIM"]
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(1, 1),
        fontsize=PLOT_LEGEND_FONTSIZE,
    )

    # --- Highlight xtick labels (SplineOps methods) ---
    for tick, name in zip(ax1.get_xticklabels(), names):
        style = HIGHLIGHT_STYLE.get(name)
        if style is not None:
            col = style.get("color", "tab:blue")
            tick.set_color(col)
            tick.set_fontweight("bold")

    # --- Outline BOTH bars (now safe because ssim_bars exists) ---
    for i, name in enumerate(names):
        style = HIGHLIGHT_STYLE.get(name)
        if style is not None:
            col = style.get("color", "tab:blue")
            lw = float(style.get("lw", 3.0))
            snr_bars.patches[i].set_edgecolor(col)
            snr_bars.patches[i].set_linewidth(lw)
            ssim_bars.patches[i].set_edgecolor(col)
            ssim_bars.patches[i].set_linewidth(lw)

    # --- Smart truncated y-limits (robust) ---
    snr_lim = _smart_ylim(snrs, pad_frac=0.06, min_span=1.0)  # dB
    if snr_lim is not None:
        ax1.set_ylim(*snr_lim)

    ssim_lim = _smart_ylim(ssims, lo_cap=0.0, hi_cap=1.0, pad_frac=0.02, min_span=0.02)
    if ssim_lim is not None:
        ax2.set_ylim(*ssim_lim)

    fig.tight_layout()
    plt.show()

# %%
# Color ROI Montage Helpers
# -------------------------

def _first_pass_color_for_backend(
    backend: str,
    rgb: np.ndarray,
    z: float,
) -> Optional[np.ndarray]:
    """
    First-pass color resized image for a given backend, for color ROIs only.
    """
    H, W, C = rgb.shape
    H1 = int(round(H * z))
    W1 = int(round(W * z))
    if H1 < 1 or W1 < 1:
        return None

    try:
        if backend in ("spl_standard", "spl_aa"):
            if not _HAS_SPLINEOPS:
                return None
            method = "cubic" if backend == "spl_standard" else "cubic-antialiasing"
            zoom_hw = (z, z)
            channels = []
            for c in range(C):
                ch = sp_resize(
                    rgb[..., c],
                    zoom_factors=zoom_hw,
                    method=method,
                )
                channels.append(ch)
            first = np.stack(channels, axis=-1)

        elif backend == "scipy":
            if not _HAS_SCIPY:
                return None
            channels = []
            for c in range(C):
                ch = _ndi_zoom(
                    rgb[..., c],
                    (z, z),
                    order=3,
                    prefilter=True,
                    mode="reflect",
                    grid_mode=False,
                )
                channels.append(ch)
            first = np.stack(channels, axis=-1)

        elif backend == "opencv":
            if not _HAS_CV2:
                return None
            arr = rgb.astype(np.float32, copy=False)
            first = cv2.resize(arr, (W1, H1), interpolation=cv2.INTER_CUBIC)

        elif backend == "pillow":
            from PIL import Image as _Image
            arr_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
            im = _Image.fromarray(arr_uint8, mode="RGB")
            first_im = im.resize((W1, H1), resample=_Image.Resampling.BICUBIC)
            first = np.asarray(first_im, dtype=np.float32) / 255.0

        elif backend == "skimage":
            if not _HAS_SKIMAGE:
                return None
            first = _sk_resize(
                rgb,
                (H1, W1, C),
                order=3,
                anti_aliasing=False,
                preserve_range=True,
                mode="reflect",
            ).astype(np.float32)

        elif backend == "skimage_aa":
            if not _HAS_SKIMAGE:
                return None
            first = _sk_resize(
                rgb,
                (H1, W1, C),
                order=3,
                anti_aliasing=True,
                preserve_range=True,
                mode="reflect",
            ).astype(np.float32)

        elif backend == "torch":
            if not _HAS_TORCH:
                return None
            arr = rgb
            if arr.dtype == np.float32:
                t_dtype = torch.float32
            elif arr.dtype == np.float64:
                t_dtype = torch.float64
            else:
                t_dtype = torch.float32
                arr = arr.astype(np.float32, copy=False)
            x = torch.from_numpy(arr).to(t_dtype).permute(2, 0, 1).unsqueeze(0)
            first_t = F.interpolate(
                x,
                size=(H1, W1),
                mode="bicubic",
                align_corners=False,
                antialias=False,
            )
            first = first_t[0].permute(1, 2, 0).detach().cpu().numpy()

        elif backend == "torch_aa":
            if not _HAS_TORCH:
                return None
            arr = rgb
            if arr.dtype == np.float32:
                t_dtype = torch.float32
            elif arr.dtype == np.float64:
                t_dtype = torch.float64
            else:
                t_dtype = torch.float32
                arr = arr.astype(np.float32, copy=False)
            x = torch.from_numpy(arr).to(t_dtype).permute(2, 0, 1).unsqueeze(0)
            first_t = F.interpolate(
                x,
                size=(H1, W1),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            first = first_t[0].permute(1, 2, 0).detach().cpu().numpy()

        else:
            return None

        return np.clip(first, 0.0, 1.0).astype(DTYPE, copy=False)
    except Exception:
        return None


def show_roi_montage_color_main_from_bench(
    bench: Dict[str, object],
    orig_rgb: np.ndarray,
) -> None:
    """Color ROI montage for the main subset of methods."""
    roi_rect = bench["roi_rect"]
    z = float(bench["z"])
    row0, col0, roi_h, roi_w = roi_rect
    roi_orig = orig_rgb[row0:row0 + roi_h, col0:col0 + roi_w, :]
    orig_tile = _nearest_big_color(roi_orig, ROI_MAG_TARGET)

    tiles: List[Tuple[str, np.ndarray]] = [("Original (color)", orig_tile)]

    center_r = row0 + roi_h / 2.0
    center_c = col0 + roi_w / 2.0

    subset = [
        ("SplineOps Standard cubic",     "spl_standard"),
        ("SplineOps Antialiasing cubic", "spl_aa"),
        ("OpenCV INTER_CUBIC",           "opencv"),
        ("SciPy cubic",                  "scipy"),
        ("scikit-image cubic",           "skimage"),
        ("PyTorch bicubic (CPU)",        "torch"),
    ]

    for label, backend in subset:
        first_color = _first_pass_color_for_backend(backend, orig_rgb, z)
        if first_color is None:
            continue

        H1, W1, _ = first_color.shape
        roi_h_res = max(1, int(round(roi_h * z)))
        roi_w_res = max(1, int(round(roi_w * z)))

        if roi_h_res > H1 or roi_w_res > W1:
            roi_first = first_color
        else:
            center_r_res = int(round(center_r * z))
            center_c_res = int(round(center_c * z))
            row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, H1 - roi_h_res))
            col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, W1 - roi_w_res))
            roi_first = first_color[
                row_top_res : row_top_res + roi_h_res,
                col_left_res : col_left_res + roi_w_res,
                :
            ]

        tile = _nearest_big_color(roi_first, ROI_MAG_TARGET)
        tiles.append((label, tile))

    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, (name, tile) in enumerate(tiles):
        if idx >= rows * cols:
            break
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        ax.imshow(np.clip(tile, 0.0, 1.0))

        title_kw = dict(fontsize=ROI_TILE_TITLE_FONTSIZE)

        style = HIGHLIGHT_STYLE.get(name)
        if style is not None:
            c = style.get("color", "tab:blue")
            lw = float(style.get("lw", 3.0))
            title_kw.update(color=c, fontweight="bold")
            _highlight_tile(ax, color=c, lw=lw)

        ax.set_title(name, **title_kw)
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def show_roi_montage_color_aa_from_bench(
    bench: Dict[str, object],
    orig_rgb: np.ndarray,
) -> None:
    """Color ROI montage for AA / smoothing subset."""
    roi_rect = bench["roi_rect"]
    z = float(bench["z"])
    row0, col0, roi_h, roi_w = roi_rect
    roi_orig = orig_rgb[row0:row0 + roi_h, col0:col0 + roi_w, :]
    orig_tile = _nearest_big_color(roi_orig, ROI_MAG_TARGET)

    tiles: List[Tuple[str, np.ndarray]] = [("Original (color)", orig_tile)]

    center_r = row0 + roi_h / 2.0
    center_c = col0 + roi_w / 2.0

    subset = [
        ("SplineOps Standard cubic",     "spl_standard"),
        ("SplineOps Antialiasing cubic", "spl_aa"),
        ("Pillow BICUBIC",               "pillow"),
        ("scikit-image cubic (AA)",      "skimage_aa"),
        ("PyTorch bicubic (AA, CPU)",    "torch_aa"),
    ]

    for label, backend in subset:
        first_color = _first_pass_color_for_backend(backend, orig_rgb, z)
        if first_color is None:
            continue

        H1, W1, _ = first_color.shape
        roi_h_res = max(1, int(round(roi_h * z)))
        roi_w_res = max(1, int(round(roi_w * z)))

        if roi_h_res > H1 or roi_w_res > W1:
            roi_first = first_color
        else:
            center_r_res = int(round(center_r * z))
            center_c_res = int(round(center_c * z))
            row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, H1 - roi_h_res))
            col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, W1 - roi_w_res))
            roi_first = first_color[
                row_top_res : row_top_res + roi_h_res,
                col_left_res : col_left_res + roi_w_res,
                :
            ]

        tile = _nearest_big_color(roi_first, ROI_MAG_TARGET)
        tiles.append((label, tile))

    cols = 3
    n_tiles = len(tiles)
    rows = max(1, (n_tiles + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(rows, cols)

    for ax in axes.ravel():
        ax.axis("off")

    for idx, (name, tile) in enumerate(tiles):
        if idx >= rows * cols:
            break
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        ax.imshow(np.clip(tile, 0.0, 1.0))

        title_kw = dict(fontsize=ROI_TILE_TITLE_FONTSIZE)

        style = HIGHLIGHT_STYLE.get(name)
        if style is not None:
            c = style.get("color", "tab:blue")
            lw = float(style.get("lw", 3.0))
            title_kw.update(color=c, fontweight="bold")
            _highlight_tile(ax, color=c, lw=lw)

        ax.set_title(name, **title_kw)
        ax.axis("off")

    fig.tight_layout()
    plt.show()

# %%
# Load All Images
# ---------------

orig_images: Dict[str, np.ndarray] = {}
orig_images_rgb: Dict[str, np.ndarray] = {}

for name, url in KODAK_IMAGES:
    gray = _load_kodak_gray(url)
    orig_images[name] = gray

    rgb = _load_kodak_rgb(url)
    orig_images_rgb[name] = rgb
    print(f"Loaded {name} from {url}  |  gray shape={gray.shape}, rgb shape={rgb.shape}")

print("\nTimings averaged over "
      f"{N_TRIALS} runs per method (1 warm-up run not counted).\n")

# Small helper for color intro using SplineOps antialiasing
def _color_intro_for_image(
    img_name: str,
    bench: Dict[str, object],
) -> None:
    if not _HAS_SPLINEOPS:
        show_intro_from_bench(bench)
        return

    rgb = orig_images_rgb[img_name]
    roi_rect = bench["roi_rect"]
    z = float(bench["z"])

    zoom_hw = (z, z)
    channels = []
    for c in range(rgb.shape[2]):
        ch = sp_resize(
            rgb[..., c],
            zoom_factors=zoom_hw,
            method="cubic-antialiasing",
        )
        channels.append(ch)
    aa_rgb = np.stack(channels, axis=-1)

    show_intro_color(
        original_rgb=rgb,
        shrunk_rgb=aa_rgb,
        roi_rect=roi_rect,
        zoom=z,
        label="Antialiasing",
        degree_label=bench["degree_label"],  # type: ignore[arg-type]
    )


# %%
# Image: kodim05
# --------------

img_name = "kodim05"
img_orig = orig_images[img_name]
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = tuple(map(float, cfg["roi_center_frac"]))  # type: ignore[arg-type]

bench_kodim05 = benchmark_image(
    img_name=img_name,
    gray=img_orig,
    zoom=zoom,
    roi_size_px=roi_size_px,
    roi_center_frac=roi_center_frac,
    degree_label="Cubic",
)

# %%
# Original and Resized
# ~~~~~~~~~~~~~~~~~~~~

_color_intro_for_image(img_name, bench_kodim05)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

show_roi_montage_main_from_bench(bench_kodim05)

# %%
# ROI Comparison Error
# ~~~~~~~~~~~~~~~~~~~~

show_error_montage_main_from_bench(bench_kodim05)

# %%
# ROI Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_aa_from_bench(bench_kodim05)

# %%
# ROI Comparison Error (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_error_montage_aa_from_bench(bench_kodim05)

# %%
# ROI Color Comparison
# ~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_main_from_bench(bench_kodim05, orig_images_rgb[img_name])

# %%
# ROI Color Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_aa_from_bench(bench_kodim05, orig_images_rgb[img_name])

# %%
# Timing Comparison
# ~~~~~~~~~~~~~~~~~

show_timing_plot_from_bench(bench_kodim05)

# %%
# SNR/SSIM Comparison
# ~~~~~~~~~~~~~~~~~~~

show_snr_ssim_plot_from_bench(bench_kodim05)

# %%
# Image: kodim07
# --------------

img_name = "kodim07"
img_orig = orig_images[img_name]
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = tuple(map(float, cfg["roi_center_frac"]))  # type: ignore[arg-type]

bench_kodim07 = benchmark_image(
    img_name=img_name,
    gray=img_orig,
    zoom=zoom,
    roi_size_px=roi_size_px,
    roi_center_frac=roi_center_frac,
    degree_label="Cubic",
)

# %%
# Original and Resized
# ~~~~~~~~~~~~~~~~~~~~

_color_intro_for_image(img_name, bench_kodim07)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

show_roi_montage_main_from_bench(bench_kodim07)

# %%
# ROI Comparison Error
# ~~~~~~~~~~~~~~~~~~~~

show_error_montage_main_from_bench(bench_kodim07)

# %%
# ROI Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_aa_from_bench(bench_kodim07)

# %%
# ROI Comparison Error (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_error_montage_aa_from_bench(bench_kodim07)

# %%
# ROI Color Comparison
# ~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_main_from_bench(bench_kodim07, orig_images_rgb[img_name])

# %%
# ROI Color Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_aa_from_bench(bench_kodim07, orig_images_rgb[img_name])

# %%
# Timing Comparison
# ~~~~~~~~~~~~~~~~~

show_timing_plot_from_bench(bench_kodim07)

# %%
# SNR/SSIM Comparison
# ~~~~~~~~~~~~~~~~~~~

show_snr_ssim_plot_from_bench(bench_kodim07)

# %%
# Image: kodim14
# --------------

img_name = "kodim14"
img_orig = orig_images[img_name]
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = tuple(map(float, cfg["roi_center_frac"]))  # type: ignore[arg-type]

bench_kodim14 = benchmark_image(
    img_name=img_name,
    gray=img_orig,
    zoom=zoom,
    roi_size_px=roi_size_px,
    roi_center_frac=roi_center_frac,
    degree_label="Cubic",
)

# %%
# Original and Resized
# ~~~~~~~~~~~~~~~~~~~~

_color_intro_for_image(img_name, bench_kodim14)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

show_roi_montage_main_from_bench(bench_kodim14)

# %%
# ROI Comparison Error
# ~~~~~~~~~~~~~~~~~~~~

show_error_montage_main_from_bench(bench_kodim14)

# %%
# ROI Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_aa_from_bench(bench_kodim14)

# %%
# ROI Comparison Error (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_error_montage_aa_from_bench(bench_kodim14)

# %%
# ROI Color Comparison
# ~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_main_from_bench(bench_kodim14, orig_images_rgb[img_name])

# %%
# ROI Color Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_aa_from_bench(bench_kodim14, orig_images_rgb[img_name])

# %%
# Time Comparison
# ~~~~~~~~~~~~~~~

show_timing_plot_from_bench(bench_kodim14)

# %%
# SNR/SSIM Comparison
# ~~~~~~~~~~~~~~~~~~~

show_snr_ssim_plot_from_bench(bench_kodim14)

# %%
# Image: kodim15
# --------------

img_name = "kodim15"
img_orig = orig_images[img_name]
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = tuple(map(float, cfg["roi_center_frac"]))  # type: ignore[arg-type]

bench_kodim15 = benchmark_image(
    img_name=img_name,
    gray=img_orig,
    zoom=zoom,
    roi_size_px=roi_size_px,
    roi_center_frac=roi_center_frac,
    degree_label="Cubic",
)

# %%
# Original and Resized
# ~~~~~~~~~~~~~~~~~~~~

_color_intro_for_image(img_name, bench_kodim15)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

show_roi_montage_main_from_bench(bench_kodim15)

# %%
# ROI Comparison Error
# ~~~~~~~~~~~~~~~~~~~~

show_error_montage_main_from_bench(bench_kodim15)

# %%
# ROI Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_aa_from_bench(bench_kodim15)

# %%
# ROI Comparison Error (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_error_montage_aa_from_bench(bench_kodim15)

# %%
# ROI Color Comparison
# ~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_main_from_bench(bench_kodim15, orig_images_rgb[img_name])

# %%
# ROI Color Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_aa_from_bench(bench_kodim15, orig_images_rgb[img_name])

# %%
# Time Comparison
# ~~~~~~~~~~~~~~~

show_timing_plot_from_bench(bench_kodim15)

# %%
# SNR/SSIM Comparison
# ~~~~~~~~~~~~~~~~~~~

show_snr_ssim_plot_from_bench(bench_kodim15)

# %%
# Image: kodim19
# --------------

img_name = "kodim19"
img_orig = orig_images[img_name]
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = tuple(map(float, cfg["roi_center_frac"]))  # type: ignore[arg-type]

bench_kodim19 = benchmark_image(
    img_name=img_name,
    gray=img_orig,
    zoom=zoom,
    roi_size_px=roi_size_px,
    roi_center_frac=roi_center_frac,
    degree_label="Cubic",
)

# %%
# Original and Resized
# ~~~~~~~~~~~~~~~~~~~~

_color_intro_for_image(img_name, bench_kodim19)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

show_roi_montage_main_from_bench(bench_kodim19)

# %%
# ROI Comparison Error
# ~~~~~~~~~~~~~~~~~~~~

show_error_montage_main_from_bench(bench_kodim19)

# %%
# ROI Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_aa_from_bench(bench_kodim19)

# %%
# ROI Comparison Error (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_error_montage_aa_from_bench(bench_kodim19)

# %%
# ROI Color Comparison
# ~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_main_from_bench(bench_kodim19, orig_images_rgb[img_name])

# %%
# ROI Color Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_aa_from_bench(bench_kodim19, orig_images_rgb[img_name])

# %%
# Time Comparison
# ~~~~~~~~~~~~~~~

show_timing_plot_from_bench(bench_kodim19)

# %%
# SNR/SSIM Comparison
# ~~~~~~~~~~~~~~~~~~~

show_snr_ssim_plot_from_bench(bench_kodim19)

# %%
# Image: kodim22
# --------------

img_name = "kodim22"
img_orig = orig_images[img_name]
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = tuple(map(float, cfg["roi_center_frac"]))  # type: ignore[arg-type]

bench_kodim22 = benchmark_image(
    img_name=img_name,
    gray=img_orig,
    zoom=zoom,
    roi_size_px=roi_size_px,
    roi_center_frac=roi_center_frac,
    degree_label="Cubic",
)

# %%
# Original and Resized
# ~~~~~~~~~~~~~~~~~~~~

_color_intro_for_image(img_name, bench_kodim22)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

show_roi_montage_main_from_bench(bench_kodim22)

# %%
# ROI Comparison Error
# ~~~~~~~~~~~~~~~~~~~~

show_error_montage_main_from_bench(bench_kodim22)

# %%
# ROI Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_aa_from_bench(bench_kodim22)

# %%
# ROI Comparison Error (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_error_montage_aa_from_bench(bench_kodim22)

# %%
# ROI Color Comparison
# ~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_main_from_bench(bench_kodim22, orig_images_rgb[img_name])

# %%
# ROI Color Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_aa_from_bench(bench_kodim22, orig_images_rgb[img_name])

# %%
# Time Comparison
# ~~~~~~~~~~~~~~~

show_timing_plot_from_bench(bench_kodim22)

# %%
# SNR/SSIM Comparison
# ~~~~~~~~~~~~~~~~~~~

show_snr_ssim_plot_from_bench(bench_kodim22)


# %%
# Image: kodim23
# --------------

img_name = "kodim23"
img_orig = orig_images[img_name]
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = tuple(map(float, cfg["roi_center_frac"]))  # type: ignore[arg-type]

bench_kodim23 = benchmark_image(
    img_name=img_name,
    gray=img_orig,
    zoom=zoom,
    roi_size_px=roi_size_px,
    roi_center_frac=roi_center_frac,
    degree_label="Cubic",
)

# %%
# Original and Resized
# ~~~~~~~~~~~~~~~~~~~~

_color_intro_for_image(img_name, bench_kodim23)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

show_roi_montage_main_from_bench(bench_kodim23)

# %%
# ROI Comparison Error
# ~~~~~~~~~~~~~~~~~~~~

show_error_montage_main_from_bench(bench_kodim23)

# %%
# ROI Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_aa_from_bench(bench_kodim23)

# %%
# ROI Comparison Error (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_error_montage_aa_from_bench(bench_kodim23)

# %%
# ROI Color Comparison
# ~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_main_from_bench(bench_kodim23, orig_images_rgb[img_name])

# %%
# ROI Color Comparison (Antialiased)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

show_roi_montage_color_aa_from_bench(bench_kodim23, orig_images_rgb[img_name])

# %%
# Time Comparison
# ~~~~~~~~~~~~~~~

show_timing_plot_from_bench(bench_kodim23)

# %%
# SNR/SSIM Comparison
# ~~~~~~~~~~~~~~~~~~~

show_snr_ssim_plot_from_bench(bench_kodim23)

# %%
# Runtime Context
# ---------------
#
# Finally, we print a short summary of the runtime environment and the storage
# dtype used for the benchmark.

if _HAS_SPECS and print_runtime_context is not None:
    print_runtime_context(include_threadpools=True)
print(f"Benchmark storage dtype: {np.dtype(DTYPE).name}")