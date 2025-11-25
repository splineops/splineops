# splineops/scripts/script_resize_comparison.py
# -*- coding: utf-8 -*-
"""
script_resize_comparison.py

Compare splineops interpolation against common stacks at a chosen zoom:

- splineops: Standard (linear/cubic), Least-Squares (best AA), Oblique (fast AA)
- SciPy ndimage.zoom (linear/cubic)
- OpenCV (INTER_LINEAR / INTER_CUBIC)
- Pillow (BILINEAR / BICUBIC)
- scikit-image (resize, order=1 or 3, anti_aliasing=True)
- PyTorch (F.interpolate bilinear/bicubic, antialias=True, CPU)

Workflow
--------
1) Pick an image (dialog). If you cancel, you'll be asked for a URL.
2) Enter zoom factor z (>0), e.g. 0.3 for downscale or 1.7 for upscale.
3) Script runs round-trip per method (z, then 1/z), averages timing over N runs.
4) Computes SNR/MSE on the round-trip image over a fixed square ROI (≈256×256)
   and shows:
   - Initial 2×2 figure:
       row 1: original image + magnified ROI
       row 2: LS first-pass resized image on white canvas + magnified mapped ROI
   - ROI montage with ORIGINAL ROI + each method's *first-pass* ROI
     (nearest-neighbour magnified at the mapped location)
   - Bar charts for timing and SNR

Notes
-----
- All ops run on grayscale images normalized to [0, 1].
- Methods with missing deps are marked "Unavailable" and skipped.
- For fairness we use mirror/reflect-like boundaries when available.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Default storage dtype for comparison (change to np.float64 if desired)
DTYPE = np.float32

# ROI / detail-window configuration
ROI_SIZE_PX = 256              # approximate ROI size in original image
ROI_CENTER_FRAC = (0.40, 0.65)  # (row_frac, col_frac) in [0, 1]
ROI_MAG_TARGET = 256           # target height for nearest-neighbour zoom tiles

# Plot appearance for slide-friendly export
PLOT_FIGSIZE = (14, 7)      # same 2:1 ratio as (10, 5), just larger
PLOT_TITLE_FONTSIZE = 18
PLOT_LABEL_FONTSIZE = 18
PLOT_TICK_FONTSIZE = 18
PLOT_LEGEND_FONTSIZE = 18

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

# splineops
try:
    from splineops.resize.resize import resize as sp_resize

    _HAS_SPLINEOPS = True
except Exception as e:
    _HAS_SPLINEOPS = False
    _SPLINEOPS_IMPORT_ERR = str(e)

# Optional for URL loading
try:
    import requests

    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False

from PyQt5 import QtWidgets

# Optional: runtime specs (Python/OS/libs/etc.)
try:
    from splineops.utils.specs import print_runtime_context as _print_runtime_context
    _HAS_SPECS = True
except Exception:
    _print_runtime_context = None
    _HAS_SPECS = False

# ---------------------------
# Utilities
# ---------------------------

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


def _fmt_time(s: Optional[float]) -> str:
    if s is None or not np.isfinite(s):
        return "n/a"
    return f"{s*1e3:.1f} ms" if s < 1.0 else f"{s:.3f} s"


def _nearest_big(roi: np.ndarray, target_h: int) -> np.ndarray:
    h, w = roi.shape[:2]
    mag = max(1, int(round(target_h / max(h, 1))))
    out = np.repeat(np.repeat(roi, mag, axis=0), mag, axis=1)
    return out


def _load_image_any(path_or_url: str) -> Image.Image:
    if "://" in path_or_url:
        if not _HAS_REQUESTS:
            raise RuntimeError("requests not installed; cannot fetch URLs.")
        r = requests.get(path_or_url, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content))
    return Image.open(path_or_url)


def _choose_image_dialog() -> Optional[str]:
    """Use a Qt file dialog to pick an image. If cancelled, ask for a URL."""
    filters = (
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
        "All files (*)"
    )

    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select an image",
        "",
        filters,
    )

    if path:
        try:
            Image.open(path).close()
            return path
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                None,
                "Open failed",
                f"Could not open file:\n{e}",
            )
            return None

    url, ok = QtWidgets.QInputDialog.getText(
        None,
        "Image URL",
        "Paste an image URL (or Cancel):",
    )
    if ok:
        url_str = str(url).strip()
        if url_str:
            return url_str

    return None


def _ask_zoom_factor(default: float = 0.3) -> Optional[float]:
    """Ask for zoom factor via Qt input dialog."""
    text, ok = QtWidgets.QInputDialog.getText(
        None,
        "Zoom factor",
        "Enter zoom factor (>0):",
        text=str(default),
    )
    if not ok:
        return None

    s = str(text).strip()
    if not s:
        return None

    try:
        z = float(s)
    except Exception:
        return None

    if not np.isfinite(z) or z <= 0:
        return None

    return z


def _to_gray01(im: Image.Image) -> np.ndarray:
    # Drop alpha for simplicity
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGB")
    if im.mode == "L":
        arr = np.asarray(im, dtype=np.float64) / 255.0
    elif im.mode in ("I;16", "I"):
        arr = np.asarray(im, dtype=np.float64)
        amin, amax = float(arr.min()), float(arr.max())
        arr = (arr - amin) / (amax - amin + 1e-12)
    else:
        if im.mode != "RGB":
            im = im.convert("RGB")
        rgb = np.asarray(im, dtype=np.float64) / 255.0
        arr = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
    arr = np.clip(arr, 0.0, 1.0)
    return np.ascontiguousarray(arr, dtype=DTYPE)


# ---------------------------
# Backends: first-pass + round-trip
# ---------------------------

def _rt_splineops(
    gray: np.ndarray, z: float, preset: str
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """
    Return (first, rec, err):

    first = sp_resize(gray, z)    -- first-pass resized image
    rec   = sp_resize(first, 1/z) -- round-trip back to original size
    """
    if not _HAS_SPLINEOPS:
        return gray, gray, f"splineops unavailable: {_SPLINEOPS_IMPORT_ERR}"
    try:
        first = sp_resize(gray, zoom_factors=(z, z), method=preset)
        rec = sp_resize(first, output_size=gray.shape, method=preset)
        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec = np.clip(rec, 0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_scipy(
    gray: np.ndarray, z: float, degree: str
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Return (first, rec, err) using SciPy ndimage.zoom with order=1/3."""
    if not _HAS_SCIPY:
        return gray, gray, "SciPy not installed"
    try:
        order_map = {"linear": 1, "cubic": 3}
        order = order_map[degree]
        need_prefilter = order >= 3

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
        rec = np.clip(rec, 0.0, 1.0)
        first = np.clip(first, 0.0, 1.0)

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

        first = first.astype(gray.dtype, copy=False)
        rec = rec.astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_opencv(
    gray: np.ndarray, z: float, which: str
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Return (first, rec, err) using OpenCV INTER_LINEAR / INTER_CUBIC."""
    if not _HAS_CV2:
        return gray, gray, "OpenCV not installed"
    try:
        interp = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
        }[which]
        H, W = gray.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))

        first = cv2.resize(gray, (W1, H1), interpolation=interp)
        rec = cv2.resize(first, (W, H), interpolation=interp)

        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec = np.clip(rec, 0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_pillow(
    gray: np.ndarray, z: float, which: str
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Return (first, rec, err) using Pillow BILINEAR/BICUBIC."""
    try:
        from PIL import Image as _Image

        resample = {
            "linear": _Image.Resampling.BILINEAR,
            "cubic": _Image.Resampling.BICUBIC,
        }[which]
        H, W = gray.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))

        im = _Image.fromarray(
            np.rint(np.clip(gray, 0, 1) * 255.0).astype(np.uint8), mode="L"
        )
        first_im = im.resize((W1, H1), resample=resample)
        rec_im = first_im.resize((W, H), resample=resample)

        first = np.asarray(first_im, dtype=np.float64) / 255.0
        rec = np.asarray(rec_im, dtype=np.float64) / 255.0

        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec = np.clip(rec, 0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_skimage(
    gray: np.ndarray, z: float, degree: str
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Return (first, rec, err) using skimage.transform.resize with order=1/3."""
    if not _HAS_SKIMAGE:
        return gray, gray, "scikit-image not installed"
    try:
        order_map = {"linear": 1, "cubic": 3}
        order = order_map[degree]
        H, W = gray.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))

        first = _sk_resize(
            gray,
            (H1, W1),
            order=order,
            anti_aliasing=True,
            preserve_range=True,
            mode="reflect",
        ).astype(np.float64)
        rec = _sk_resize(
            first,
            (H, W),
            order=order,
            anti_aliasing=True,
            preserve_range=True,
            mode="reflect",
        ).astype(np.float64)

        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec = np.clip(rec, 0.0, 1.0).astype(gray.dtype, copy=False)
        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)


def _rt_torch(
    gray: np.ndarray, z: float, degree: str
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
    """Return (first, rec, err) using torch F.interpolate with bilinear/bicubic + AA."""
    if not _HAS_TORCH:
        return gray, gray, "PyTorch not installed"
    try:
        mode = "bilinear" if degree == "linear" else "bicubic"

        # Work with the existing dtype; only cast if needed
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

        # (1, 1, H, W) tensor
        x = torch.from_numpy(arr).to(t_dtype).unsqueeze(0).unsqueeze(0)

        first_t = F.interpolate(
            x,
            size=(H1, W1),
            mode=mode,
            align_corners=False,
            antialias=True,
        )
        rec_t = F.interpolate(
            first_t,
            size=(H, W),
            mode=mode,
            align_corners=False,
            antialias=True,
        )

        # Back to NumPy
        first = first_t[0, 0].detach().cpu().numpy()
        rec = rec_t[0, 0].detach().cpu().numpy()

        # Clip + cast to match other methods / DTYPE
        first = np.clip(first, 0.0, 1.0).astype(gray.dtype, copy=False)
        rec = np.clip(rec, 0.0, 1.0).astype(gray.dtype, copy=False)

        return first, rec, None
    except Exception as e:
        return gray, gray, str(e)

# ---------------------------
# Benchmark harness
# ---------------------------

def _avg_time(fn, repeats: int = 10, warmup: bool = True):
    """
    Run `fn()` (which must return (first, rec, err)) `repeats` times,
    measuring total time of first+rec per run.

    Returns (last_first, last_rec, mean_time, std_time, err).
    """
    if warmup:
        try:
            first, rec, err = fn()
            if err is not None:
                return np.array([]), np.array([]), float("nan"), float("nan"), err
        except Exception as e:
            return np.array([]), np.array([]), float("nan"), float("nan"), str(e)

    times: List[float] = []
    last_first: Optional[np.ndarray] = None
    last_rec: Optional[np.ndarray] = None

    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        first, rec, err = fn()
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


# ---------------------------
# Initial 2×2 figure helper
# ---------------------------

def _show_initial_original_vs_ls(
    gray: np.ndarray,
    roi_rect: Tuple[int, int, int, int],
    ls_first: np.ndarray,
    z: float,
    degree_label: str,
) -> None:
    """
    Plot a 2x2 figure:

    Row 1:
      - Original image with red ROI box
      - Magnified original ROI

    Row 2:
      - LS first-pass resized image on white canvas with mapped ROI box
      - Magnified mapped ROI from LS first-pass
    """
    H, W = gray.shape
    row0, col0, roi_h, roi_w = roi_rect

    # Original ROI
    roi_orig = _crop_roi(gray, roi_rect)
    roi_orig_big = _nearest_big(roi_orig, ROI_MAG_TARGET)

    # LS first-pass ROI mapping
    H1, W1 = ls_first.shape
    # Original ROI center
    center_r = row0 + roi_h / 2.0
    center_c = col0 + roi_w / 2.0

    roi_h_res = max(1, int(round(roi_h * z)))
    roi_w_res = max(1, int(round(roi_w * z)))

    if roi_h_res > H1 or roi_w_res > W1:
        # If mapped ROI is larger than LS image, just use full LS image
        ls_roi = ls_first
        row_top_res = 0
        col_left_res = 0
        roi_h_res = H1
        roi_w_res = W1
    else:
        center_r_res = int(round(center_r * z))
        center_c_res = int(round(center_c * z))
        row_top_res = int(
            np.clip(center_r_res - roi_h_res // 2, 0, H1 - roi_h_res)
        )
        col_left_res = int(
            np.clip(center_c_res - roi_w_res // 2, 0, W1 - roi_w_res)
        )
        ls_roi = ls_first[
            row_top_res : row_top_res + roi_h_res,
            col_left_res : col_left_res + roi_w_res,
        ]

    ls_roi_big = _nearest_big(ls_roi, ROI_MAG_TARGET)

    # LS canvas: white background with LS first in the top-left (possibly cropped)
    canvas_ls = np.ones_like(gray, dtype=ls_first.dtype)
    h_copy = min(H, H1)
    w_copy = min(W, W1)
    canvas_ls[:h_copy, :w_copy] = ls_first[:h_copy, :w_copy]

    # Build the figure
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
    ax.set_title(f"Original image with ROI ({H}×{W} px)")
    ax.axis("off")

    # Row 1, right: magnified original ROI
    ax = axes[0, 1]
    ax.imshow(roi_orig_big, cmap="gray", interpolation="nearest", aspect="equal")
    ax.set_title(f"Original ROI ({roi_h}×{roi_w} px, magnified)")
    ax.axis("off")

    # Row 2, left: LS first-pass on canvas with mapped ROI box
    ax = axes[1, 0]
    ax.imshow(canvas_ls, cmap="gray", interpolation="nearest", aspect="equal")
    # Only draw box if ROI fits inside LS extents we pasted
    if row_top_res < h_copy and col_left_res < w_copy:
        box_h = min(roi_h_res, h_copy - row_top_res)
        box_w = min(roi_w_res, w_copy - col_left_res)
        rect_ls = patches.Rectangle(
            (col_left_res, row_top_res),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect_ls)
    ax.set_title(
        f"Splineops LS first-pass ({degree_label}, zoom ×{z:g}, {H1}×{W1} px)"
    )
    ax.axis("off")

    # Row 2, right: magnified LS ROI
    ax = axes[1, 1]
    ax.imshow(ls_roi_big, cmap="gray", interpolation="nearest", aspect="equal")
    ax.set_title(
        f"LS first-pass ROI ({roi_h_res}×{roi_w_res} px, magnified)"
    )
    ax.axis("off")

    fig.tight_layout()
    plt.show()


# ---------------------------
# main
# ---------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare splineops vs common stacks at a chosen zoom (linear/cubic)."
    )
    ap.add_argument(
        "--degree",
        type=str,
        default="cubic",
        choices=("linear", "cubic"),
        help="Degree / interpolation mode (linear or cubic) for all methods.",
    )
    args = ap.parse_args(argv)
    degree = args.degree

    # Ensure a Qt application exists before degree dialog / file dialog
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # Degree selection dialog
    items = ["Linear", "Cubic"]
    default_idx = 0 if degree == "linear" else 1
    choice, ok = QtWidgets.QInputDialog.getItem(
        None,
        "Interpolation degree",
        "Choose interpolation degree:",
        items,
        default_idx,
        False,
    )
    if ok and choice:
        degree = choice.lower()

    degree_label = degree.title()

    # --- Select image ---
    path_or_url = _choose_image_dialog() or ""
    if not path_or_url:
        print("No image selected. Aborting.")
        return 0

    try:
        im = _load_image_any(path_or_url)
    except Exception as e:
        print(f"Open failed: {e}", file=sys.stderr)
        return 1
    gray = _to_gray01(im)
    H, W = gray.shape
    print(f"Loaded: {path_or_url} | shape={gray.shape}, dtype={gray.dtype}")

    # --- Ask zoom ---
    z = _ask_zoom_factor(0.3)
    if z is None or z <= 0:
        print("Invalid or cancelled zoom factor. Aborting.")
        return 0

    # --- Settings ---
    repeats = 10  # avg runs

    # ROI in the *original* image (for round-trip quality metrics)
    roi_rect = _roi_rect_from_frac(gray.shape, ROI_SIZE_PX, ROI_CENTER_FRAC)
    roi = _crop_roi(gray, roi_rect)

    # Precompute original ROI center/size to map into resized space
    roi_h = roi_rect[2]
    roi_w = roi_rect[3]
    center_r = roi_rect[0] + roi_h / 2.0
    center_c = roi_rect[1] + roi_w / 2.0

    # --- Methods to compare ---
    methods = [
        (
            f"Splineops — Standard {degree_label}",
            lambda: _rt_splineops(gray, z, degree),
        ),
        (
            f"Splineops — LS (best AA) {degree_label}",
            lambda: _rt_splineops(gray, z, f"{degree}-best_antialiasing"),
        ),
        (
            f"Splineops — Oblique (fast AA) {degree_label}",
            lambda: _rt_splineops(gray, z, f"{degree}-fast_antialiasing"),
        ),
        (
            f"SciPy {degree_label}",
            lambda: _rt_scipy(gray, z, degree),
        ),
        (
            f"OpenCV INTER_{degree_label.upper()}",
            lambda: _rt_opencv(gray, z, degree),
        ),
        (
            f"Pillow {degree_label.upper()}",
            lambda: _rt_pillow(gray, z, degree),
        ),
        (
            f"scikit-image ({degree_label}, AA)",
            lambda: _rt_skimage(gray, z, degree),
        ),
        (
            f"PyTorch {degree_label} (AA, CPU)",
            lambda: _rt_torch(gray, z, degree),
        ),
    ]

    rows: List[Dict] = []
    roi_tiles: List[Tuple[str, np.ndarray]] = []

    # Add ORIGINAL ROI tile as first comparison panel
    orig_tile = _nearest_big(roi, ROI_MAG_TARGET)
    roi_tiles.append(("Original", orig_tile))

    # We also want to remember LS first-pass image for the initial 2x2 figure
    ls_first_for_plot: Optional[np.ndarray] = None

    # Print runtime specs (if available)
    if _HAS_SPECS and _print_runtime_context is not None:
        print()
        _print_runtime_context(include_threadpools=True)
        print()

    print(f"\nBenchmarking round-trip @ zoom ×{z:.5g}  (repeats={repeats})\n")
    header = f"{'Method':<40} {'Time (mean)':>13} {'± SD':>10} {'SNR (dB)':>10} {'MSE':>14}"
    print(header)
    print("-" * len(header))

    for name, runner in methods:
        first, rec, t_mean, t_sd, err = _avg_time(runner, repeats=repeats, warmup=True)
        if err is not None or rec.size == 0 or first.size == 0:
            print(f"{name:<40} {'unavailable':>13} {'':>10} {'—':>10} {'—':>14}")
            rows.append(
                {
                    "name": name,
                    "time": np.nan,
                    "sd": np.nan,
                    "snr": np.nan,
                    "mse": np.nan,
                    "err": err,
                }
            )
            continue

        # Capture LS first-pass for the initial figure
        if name.startswith("Splineops — LS (best AA)"):
            ls_first_for_plot = first.copy()

        # Round-trip metrics on fixed ROI in original resolution
        rec_roi = _crop_roi(rec, roi_rect)
        snr = _snr_db(roi, rec_roi)
        mse = float(np.mean((roi - rec_roi) ** 2, dtype=np.float64))
        print(
            f"{name:<40} {_fmt_time(t_mean):>13} {_fmt_time(t_sd):>10} "
            f"{snr:>10.2f} {mse:>14.3e}"
        )

        rows.append(
            {
                "name": name,
                "time": t_mean,
                "sd": t_sd,
                "snr": snr,
                "mse": mse,
                "err": None,
            }
        )

        # ROI tile for montage: map original ROI center + size into resized space
        H1, W1 = first.shape
        roi_h_res = max(1, int(round(roi_h * z)))
        roi_w_res = max(1, int(round(roi_w * z)))

        if roi_h_res > H1 or roi_w_res > W1:
            # If the mapped ROI is larger than the resized image, just use full image
            first_roi = first
        else:
            center_r_res = int(round(center_r * z))
            center_c_res = int(round(center_c * z))
            row_top_res = int(
                np.clip(center_r_res - roi_h_res // 2, 0, H1 - roi_h_res)
            )
            col_left_res = int(
                np.clip(center_c_res - roi_w_res // 2, 0, W1 - roi_w_res)
            )
            first_roi = first[
                row_top_res : row_top_res + roi_h_res,
                col_left_res : col_left_res + roi_w_res,
            ]

        tile = _nearest_big(first_roi, ROI_MAG_TARGET)
        roi_tiles.append((name, tile))

    # --- Initial 2×2 figure: original vs LS first-pass ---
    if ls_first_for_plot is not None:
        _show_initial_original_vs_ls(
            gray=gray,
            roi_rect=roi_rect,
            ls_first=ls_first_for_plot,
            z=z,
            degree_label=degree_label,
        )

    # --- ROI montage (original + first-pass results) ---
    if roi_tiles:
        cols = min(3, len(roi_tiles))
        rows_n = int(np.ceil(len(roi_tiles) / cols))
        fig, axes = plt.subplots(
            rows_n, cols, figsize=(cols * 3.2, rows_n * 3.4)
        )
        if not isinstance(axes, np.ndarray):
            axes = np.array([[axes]])
        axes = axes.reshape(rows_n, cols)
        for ax in axes.ravel():
            ax.set_axis_off()
        for idx, (name, tile) in enumerate(roi_tiles):
            r, c = divmod(idx, cols)
            ax = axes[r, c]
            ax.imshow(tile, cmap="gray", interpolation="nearest")
            ax.set_title(name, fontsize=9)
            ax.set_axis_off()
        h_roi, w_roi = roi.shape
        fig.suptitle(
            f"ROI comparison (original + first-pass resized) — "
            f"{h_roi}×{w_roi} px, zoom ×{z:g}, degree={degree_label}",
            fontsize=12,
        )
        plt.tight_layout()
        plt.show()

        # --- Timing bar chart (round-trip) ---
        valid = [r for r in rows if np.isfinite(r["time"])]
        if valid:
            names = [r["name"] for r in valid]
            times = np.array([r["time"] for r in valid])
            sds = np.array([r["sd"] for r in valid])
            order = np.argsort(times)
            names = [names[i] for i in order]
            times = times[order]
            sds = sds[order]

            plt.figure(figsize=PLOT_FIGSIZE)
            y = np.arange(len(names))
            plt.barh(y, times, xerr=sds, alpha=0.8)
            plt.yticks(y, names, fontsize=PLOT_TICK_FONTSIZE)
            plt.xticks(fontsize=PLOT_TICK_FONTSIZE)  # ⬅️ make 0.000, 0.005, ... big too
            plt.xlabel(
                f"Round-trip time (s) — mean ± sd over {repeats} runs",
                fontsize=PLOT_LABEL_FONTSIZE,
            )
            plt.title(
                f"Timing vs Method (H×W = {H}×{W}, zoom ×{z:g}, degree={degree_label})",
                fontsize=PLOT_TITLE_FONTSIZE,
            )
            plt.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.show()

        # --- SNR + MSE bar chart (round-trip) ---
        valid = [r for r in rows if np.isfinite(r["snr"]) and np.isfinite(r["mse"])]
        if valid:
            names = [r["name"] for r in valid]
            snrs = np.array([r["snr"] for r in valid])
            mses = np.array([r["mse"] for r in valid])

            # Sort by SNR (higher is better) and keep same order for MSE
            order = np.argsort(-snrs)
            names = [names[i] for i in order]
            snrs = snrs[order]
            mses = mses[order]

            x = np.arange(len(names))
            width = 0.4  # bar width for each metric

            fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE)

            # Left y-axis: SNR
            snr_color = "tab:blue"
            mse_color = "tab:orange"

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

            # Right y-axis: MSE
            ax2 = ax1.twinx()
            mse_bars = ax2.bar(
                x + width / 2,
                mses,
                width,
                label="MSE",
                alpha=0.6,
                color=mse_color,
            )
            ax2.set_ylabel("MSE", color=mse_color, fontsize=PLOT_LABEL_FONTSIZE)
            ax2.tick_params(axis="y", labelcolor=mse_color, labelsize=PLOT_TICK_FONTSIZE)

            # Shared title / legend
            ax1.set_title(
                f"SNR / MSE vs Method (H×W = {H}×{W}, zoom ×{z:g}, degree={degree_label})",
                fontsize=PLOT_TITLE_FONTSIZE,
            )

            # Combine legends from both axes
            handles = snr_bars.patches[:1] + mse_bars.patches[:1]
            labels = ["SNR (dB)", "MSE"]
            fig.legend(
                handles,
                labels,
                loc="upper right",
                bbox_to_anchor=(1, 1),
                fontsize=PLOT_LEGEND_FONTSIZE,
            )

            fig.tight_layout()
            plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
