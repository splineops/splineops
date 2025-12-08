# splineops/scripts/script_resize_2d_plot.py
# -*- coding: utf-8 -*-
"""
Interactive timing, SNR & SSIM sweep (splineops vs SciPy vs others) over zoom factors.

Compares:

- SciPy ndimage.zoom (linear / cubic)
- splineops Standard (linear / cubic)
- splineops Antialiasing (linear / cubic)
- PyTorch bilinear/bicubic
- OpenCV INTER_LINEAR / INTER_CUBIC
- Pillow
- scikit-image (linear/cubic)

Zoom sweep:
  • 0 < z < 2, excluding 1.0
  • Only round-trip-size-preserving zooms are kept.

Plots:
  • Timing vs zoom
  • SNR vs zoom
  • SSIM vs zoom
"""

from __future__ import annotations

import argparse
import io
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# SciPy cubic baseline (we'll also use it for linear)
from scipy.ndimage import zoom as ndi_zoom

# Optional for URL
try:
    import requests
except Exception:
    requests = None

# Optional PyTorch (for comparison)
try:
    import torch
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    torch = None
    F = None

# Optional OpenCV (for comparison)
try:
    import cv2

    _HAS_CV2 = True
    # Undo OpenCV's Qt plugin path override to avoid conflicts with PyQt/Matplotlib
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
except Exception:
    _HAS_CV2 = False

# Optional scikit-image (for comparison + SSIM)
try:
    from skimage.transform import resize as sk_resize
    from skimage.metrics import structural_similarity as sk_ssim  # NEW: SSIM

    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False
    sk_ssim = None  # type: ignore[assignment]

# PyQt5 dialogs for interactive selection
from PyQt5 import QtWidgets

# splineops
from splineops.resize import resize as spl_resize

# Default storage dtype for the sweep (change to np.float64 if desired)
DTYPE = np.float32
DTYPE_NAME = np.dtype(DTYPE).name

# Plot appearance for slide-friendly export
PLOT_FIGSIZE = (14, 7)      # wider, 2:1-ish
PLOT_TITLE_FONTSIZE = 18
PLOT_LABEL_FONTSIZE = 18
PLOT_TICK_FONTSIZE = 18
PLOT_LEGEND_FONTSIZE = 18

MARKER_SIZE = 6             # bigger markers
LINEWIDTH = 2.0             # thicker lines

# Show markers only on every N-th point (sparser markers).
# All methods share the same stride but use different phase offsets
# so their markers don't sit on top of each other.
MARK_EVERY_BASE = 8

# ---------------- Method toggles ----------------
# Set any of these to False to skip computing/plotting that method.
ENABLE_SCIPY                   = True
ENABLE_SPLINEOPS_STANDARD      = True
ENABLE_SPLINEOPS_ANTIALIASING  = True
ENABLE_TORCH                   = True
ENABLE_OPENCV                  = True
ENABLE_PILLOW                  = True
ENABLE_SKIMAGE                 = True

# -------------------------- UI / I/O helpers --------------------------


def choose_image_dialog() -> str | None:
    """Open a file dialog; if canceled, prompt for URL; return a path/URL or None."""
    file_filter = (
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;"
        "PNG (*.png);;"
        "JPEG (*.jpg *.jpeg);;"
        "TIFF (*.tif *.tiff);;"
        "All files (*)"
    )

    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Select an image",
        "",
        file_filter,
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

    # No file selected: ask for URL
    url, ok = QtWidgets.QInputDialog.getText(
        None,
        "Image URL",
        "Paste an image URL (or Cancel):",
    )
    if ok:
        url = str(url).strip()
        if url:
            return url
    return None


def load_image_any(path_or_url: str, grayscale: bool = True) -> np.ndarray:
    """Load local path or URL into [0,1] as DTYPE. If RGB and grayscale=True, convert."""
    if "://" in path_or_url:
        if requests is None:
            raise RuntimeError("requests is not installed; cannot load from URL.")
        r = requests.get(path_or_url, timeout=15)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content))
    else:
        img = Image.open(path_or_url)

    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:  # already gray
        out = arr / 255.0
    else:
        out = arr / 255.0
        if grayscale:
            out = (
                0.2989 * out[..., 0]
                + 0.5870 * out[..., 1]
                + 0.1140 * out[..., 2]
            )
    out = np.clip(out, 0.0, 1.0)
    return np.ascontiguousarray(out, dtype=DTYPE)


def roundtrip_size_ok(shape: Tuple[int, ...], z: float) -> bool:
    """Accept z only if H,W -> round(H*z) then back with 1/z returns original."""
    if len(shape) < 2:
        return False
    H, W = int(shape[0]), int(shape[1])
    H1 = int(round(H * z))
    W1 = int(round(W * z))
    if H1 <= 0 or W1 <= 0:
        return False
    H2 = int(round(H1 * (1.0 / z)))
    W2 = int(round(W1 * (1.0 / z)))
    return (H2 == H) and (W2 == W)


def snr_db(x: np.ndarray, y: np.ndarray) -> float:
    """10*log10(sum(x^2)/sum((x-y)^2)). Returns +inf for perfect match."""
    num = float(np.sum(x * x, dtype=np.float64))
    den = float(np.sum((x - y) ** 2, dtype=np.float64))
    if den == 0.0:
        return float("inf")
    if num == 0.0:
        return -float("inf")
    return 10.0 * math.log10(num / den)


# ----------------------------- runners ------------------------------


def scipy_roundtrip(
    img: np.ndarray, z: float, degree: str
) -> Tuple[np.ndarray, float]:
    """
    Round-trip with SciPy ndimage.zoom using order=1 (linear) or 3 (cubic)
    and reflect boundary; prefilter is used only for cubic.
    """
    order_map = {"linear": 1, "cubic": 3}
    order = order_map[degree]
    need_prefilter = order >= 3

    zoom_fwd = (z, z) if img.ndim == 2 else (z, z, 1.0)
    zoom_bwd = (1.0 / z, 1.0 / z) if img.ndim == 2 else (1.0 / z, 1.0 / z, 1.0)

    t0 = time.perf_counter()
    out = ndi_zoom(
        img,
        zoom=zoom_fwd,
        order=order,
        prefilter=need_prefilter,
        mode="reflect",
        grid_mode=False,
    )
    rec = ndi_zoom(
        out,
        zoom=zoom_bwd,
        order=order,
        prefilter=need_prefilter,
        mode="reflect",
        grid_mode=False,
    )
    dt = time.perf_counter() - t0

    rec = np.clip(rec, 0.0, 1.0)
    return rec.astype(img.dtype, copy=False), dt


def spl_roundtrip(img: np.ndarray, z: float, method: str) -> Tuple[np.ndarray, float]:
    """
    splineops round-trip using a single preset string:

      - "linear", "cubic"            → Standard interpolation
      - "linear-antialiasing", ...   → Antialiasing (oblique projection)
    """
    zoom_fwd = (z, z) if img.ndim == 2 else (z, z, 1.0)
    zoom_bwd = (1.0 / z, 1.0 / z) if img.ndim == 2 else (1.0 / z, 1.0 / z, 1.0)
    t0 = time.perf_counter()
    out = spl_resize(img, zoom_factors=zoom_fwd, method=method)
    rec = spl_resize(out, zoom_factors=zoom_bwd, method=method)
    dt = time.perf_counter() - t0
    rec = np.clip(rec, 0.0, 1.0)
    return rec.astype(img.dtype, copy=False), dt


def torch_roundtrip(
    img: np.ndarray, z: float, degree: str
) -> Tuple[np.ndarray, float]:
    """
    Round-trip using torch.nn.functional.interpolate with bilinear (linear)
    or bicubic (cubic). Runs on CPU.

    Timing includes:
      - numpy -> torch conversion
      - forward + backward interpolate
      - torch -> numpy conversion
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available")

    t0 = time.perf_counter()  # start timing before conversions

    mode = "bilinear" if degree == "linear" else "bicubic"

    arr = img
    if arr.dtype == np.float32:
        t_dtype = torch.float32
    elif arr.dtype == np.float64:
        t_dtype = torch.float64
    else:
        t_dtype = torch.float32
        arr = arr.astype(np.float32, copy=False)

    if arr.ndim == 2:
        H, W = arr.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))
        x = torch.from_numpy(arr).to(t_dtype).unsqueeze(0).unsqueeze(0)
        y = F.interpolate(
            x,
            size=(H1, W1),
            mode=mode,
            align_corners=False,
            antialias=False,
        )
        y2 = F.interpolate(
            y,
            size=(H, W),
            mode=mode,
            align_corners=False,
            antialias=False,
        )
        rec = y2[0, 0].cpu().numpy().astype(arr.dtype, copy=False)

    elif arr.ndim == 3:
        H, W, C = arr.shape
        H1 = int(round(H * z))
        W1 = int(round(W * z))
        x = torch.from_numpy(arr).to(t_dtype).permute(2, 0, 1).unsqueeze(0)
        y = F.interpolate(
            x,
            size=(H1, W1),
            mode=mode,
            align_corners=False,
            antialias=False,
        )
        y2 = F.interpolate(
            y,
            size=(H, W),
            mode=mode,
            align_corners=False,
            antialias=False,
        )
        rec = (
            y2[0]
            .permute(1, 2, 0)
            .cpu()
            .numpy()
            .astype(arr.dtype, copy=False)
        )
    else:
        raise ValueError("Expected 2D (H×W) or 3D (H×W×C) image for PyTorch path.")

    rec = np.clip(rec, 0.0, 1.0).astype(img.dtype, copy=False)

    dt = time.perf_counter() - t0
    return rec, dt


def opencv_roundtrip(
    img: np.ndarray, z: float, which: str
) -> Tuple[np.ndarray, float]:
    """
    Round-trip with OpenCV resize using INTER_LINEAR or INTER_CUBIC.

    Supports 2D (H,W) and 3D (H,W,C) arrays.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV not available")

    interp = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }[which]

    H, W = img.shape[:2]
    W1 = int(round(W * z))
    H1 = int(round(H * z))

    t0 = time.perf_counter()
    out = cv2.resize(img, (W1, H1), interpolation=interp)
    rec = cv2.resize(out, (W, H), interpolation=interp)
    dt = time.perf_counter() - t0

    rec = np.clip(rec, 0.0, 1.0)
    return rec.astype(img.dtype, copy=False), dt


def pillow_roundtrip(
    img: np.ndarray, z: float, which: str
) -> Tuple[np.ndarray, float]:
    """
    Round-trip with Pillow's resize using BILINEAR/BICUBIC/LANCZOS.

    For 2D (grayscale) arrays, this uses a pure float32 ("F" mode) pipeline so
    there is no 8-bit quantization advantage. For RGB images we fall back to
    uint8, since Pillow doesn't support multi-channel float modes directly.
    """
    resample_map = {
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic":  Image.Resampling.BICUBIC,
        "lanczos":  Image.Resampling.LANCZOS,
    }
    if which not in resample_map:
        raise ValueError(f"Unsupported Pillow kernel: {which}")
    resample = resample_map[which]

    H, W = img.shape[:2]
    W1 = int(round(W * z))
    H1 = int(round(H * z))

    t0 = time.perf_counter()

    if img.ndim == 2:
        im = Image.fromarray(img.astype(np.float32, copy=False), mode="F")
        out = im.resize((W1, H1), resample=resample)
        rec_im = out.resize((W, H), resample=resample)
        rec_arr = np.asarray(rec_im, dtype=np.float32)
    elif img.ndim == 3 and img.shape[2] in (3, 4):
        arr01 = np.clip(img, 0.0, 1.0)
        u8 = np.rint(arr01 * 255.0).astype(np.uint8)
        mode = "RGB" if img.shape[2] == 3 else "RGBA"
        im = Image.fromarray(u8, mode=mode)
        out = im.resize((W1, H1), resample=resample)
        rec_im = out.resize((W, H), resample=resample)
        rec_arr = np.asarray(rec_im, dtype=np.float32) / 255.0
    else:
        raise ValueError("Pillow round-trip expects 2D or 3D (H×W×3/4) array")

    rec_arr = np.clip(rec_arr, 0.0, 1.0).astype(img.dtype, copy=False)
    dt = time.perf_counter() - t0
    return rec_arr, dt


def skimage_roundtrip(img: np.ndarray, z: float, degree: str) -> Tuple[np.ndarray, float]:
    """
    Round-trip with scikit-image.transform.resize using order=1 (linear) or
    order=3 (cubic). Supports 2D (H,W) and 3D (H,W,C) arrays.
    """
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image not available")

    order_map = {"linear": 1, "cubic": 3}
    order = order_map[degree]

    arr = np.asarray(img, dtype=np.float64)
    H, W = arr.shape[:2]
    H1 = int(round(H * z))
    W1 = int(round(W * z))

    t0 = time.perf_counter()

    if arr.ndim == 2:
        out = sk_resize(
            arr,
            (H1, W1),
            order=order,
            anti_aliasing=False,
            preserve_range=True,
            mode="reflect",
        )
        rec = sk_resize(
            out,
            (H, W),
            order=order,
            anti_aliasing=False,
            preserve_range=True,
            mode="reflect",
        )
    elif arr.ndim == 3:
        C = arr.shape[2]
        out = sk_resize(
            arr,
            (H1, W1, C),
            order=order,
            anti_aliasing=False,
            preserve_range=True,
            mode="reflect",
        )
        rec = sk_resize(
            out,
            (H, W, C),
            order=order,
            anti_aliasing=False,
            preserve_range=True,
            mode="reflect",
        )
    else:
        raise ValueError("scikit-image round-trip expects 2D or 3D (H×W×C) array")

    dt = time.perf_counter() - t0

    rec = np.clip(rec, 0.0, 1.0)
    return rec.astype(img.dtype, copy=False), dt


def average_time(run, repeats: int = 10, warmup: bool = True):
    """
    Return (last_rec, mean_time, std_time) over 'repeats' runs.

    If warmup=True, run one extra un-timed warmup call.
    """
    if warmup:
        # Warmup run (ignore timing + result)
        run()

    times: List[float] = []
    rec = None
    for _ in range(max(1, repeats)):
        rec, dt = run()
        times.append(dt)
    times_arr = np.asarray(times, dtype=np.float64)
    mean_t = float(times_arr.mean())
    sd_t = float(times_arr.std(ddof=1 if times_arr.size > 1 else 0))
    return rec, mean_t, sd_t

# ------------------------------ main -------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Timing, SNR & SSIM sweep with interactive image selection (averaged runs)."
    )
    ap.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path/URL; if omitted, a dialog opens.",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Base number of zoom samples per side if --samples-down/--samples-up are not given.",
    )
    ap.add_argument(
        "--samples-down",
        type=int,
        default=None,
        help="Number of zoom samples in the interval (0, 1). Overrides --samples if set.",
    )
    ap.add_argument(
        "--samples-up",
        type=int,
        default=None,
        help="Number of zoom samples in the interval (1, 2). Overrides --samples if set.",
    )
    ap.add_argument(
        "--which",
        type=str,
        default="down",
        choices=("both", "down", "up"),
        help="Which zoom regime to plot: 'down' (0<z<1), 'up' (1<z<2), or 'both'.",
    )
    ap.add_argument(
        "--grayscale",
        type=int,
        default=1,
        help="1=convert to grayscale, 0=keep RGB.",
    )
    ap.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Average this many runs per (method, z).",
    )
    ap.add_argument(
        "--degree",
        type=str,
        default="cubic",
        choices=("linear", "cubic"),
        help="Degree / interpolation mode (linear or cubic) for splineops/SciPy.",
    )
    args = brush_args(ap.parse_args())

    degree = args.degree

    # Ensure a Qt application exists before showing degree dialog / file dialog
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    # Small degree dialog (overrides CLI if confirmed)
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

    # Pick image (dialog if not provided)
    path_or_url = args.image
    if path_or_url is None:
        path_or_url = choose_image_dialog()
        if not path_or_url:
            print("No image selected. Aborting.")
            sys.exit(1)

    img = load_image_any(path_or_url, grayscale=bool(args.grayscale))
    H, W = int(img.shape[0]), int(img.shape[1])
    print(f"Loaded image: {path_or_url} | shape={img.shape}, dtype={img.dtype}")

    #
    # Build zoom candidates
    #
    n_down = args.samples_down if args.samples_down is not None else args.samples
    n_up = args.samples_up if args.samples_up is not None else args.samples

    eps = 1e-6  # margin to avoid hitting exactly 0, 1, or 2
    if n_down > 0:
        z_down = np.linspace(0.001, 1.0 - eps, n_down, endpoint=True, dtype=np.float64)
    else:
        z_down = np.array([], dtype=np.float64)

    if n_up > 0:
        z_up = np.linspace(1.0 + eps, 2.0 - eps, n_up, endpoint=True, dtype=np.float64)
    else:
        z_up = np.array([], dtype=np.float64)

    # Use only the requested regime(s)
    if args.which == "down":
        z_candidates = z_down
    elif args.which == "up":
        z_candidates = z_up
    else:  # "both"
        z_candidates = np.concatenate([z_down, z_up])

    # Guard against any accidental inclusion of 1.0 or 2.0
    z_candidates = z_candidates[(z_candidates > 0.0) & (z_candidates < 2.0)]

    # DROP zoom factors "too close" to 1.0 to avoid near-identity spikes
    NEAR_ONE_EPS = 1e-2  # e.g. exclude (0.99, 1.01); tune if you like
    z_candidates = z_candidates[np.abs(z_candidates - 1.0) > NEAR_ONE_EPS]

    # Keep only round-trip-preserving zooms
    z_list = [float(z) for z in z_candidates if roundtrip_size_ok(img.shape, float(z))]
    if not z_list:
        print(
            "No valid zoom factors after round-trip size check. "
            "Try increasing --samples-down/--samples-up or reducing NEAR_ONE_EPS."
        )
        sys.exit(1)

    print(
        f"Accepted {len(z_list)} / {len(z_candidates)} zooms "
        f"(down: {n_down}, up: {n_up}, |z-1|>{NEAR_ONE_EPS}, 2.0 excluded)."
    )

    #
    # Methods
    #
    METHODS: Dict[str, Tuple[str, str | None]] = {}

    # SciPy
    if ENABLE_SCIPY:
        METHODS[f"SciPy {degree_label}"] = ("scipy", degree)

    # splineops: Standard / Antialiasing
    if ENABLE_SPLINEOPS_STANDARD:
        METHODS[f"Standard {degree_label}"] = ("splineops", degree)

    if ENABLE_SPLINEOPS_ANTIALIASING:
        METHODS[f"Antialiasing {degree_label}"] = (
            "splineops",
            f"{degree}-antialiasing",
        )

    # PyTorch
    if ENABLE_TORCH:
        if _HAS_TORCH:
            METHODS[f"PyTorch {degree_label}"] = ("torch", degree)
        else:
            print("[info] PyTorch not found; 'PyTorch' curve will be omitted.")

    # OpenCV
    if ENABLE_OPENCV:
        if _HAS_CV2:
            METHODS[f"OpenCV INTER_{degree_label.upper()}"] = ("opencv", degree)
        else:
            print("[info] OpenCV not found; 'OpenCV' curve will be omitted.")

    # Pillow
    if ENABLE_PILLOW:
        if degree == "linear":
            METHODS["Pillow BILINEAR (float)"] = ("pillow", "bilinear")
        else:
            METHODS["Pillow BICUBIC (float)"] = ("pillow", "bicubic")

    # scikit-image
    if ENABLE_SKIMAGE:
        if _HAS_SKIMAGE:
            METHODS[f"scikit-image ({degree_label})"] = ("skimage", degree)
        else:
            print(
                "[info] scikit-image not found; 'scikit-image' curve will be omitted."
            )

    results: Dict[str, Dict[str, List[float]]] = {
        # NEW: add "ssim" channel
        name: {"z": [], "time": [], "time_sd": [], "snr": [], "ssim": []}
        for name in METHODS
    }

    #
    # Run sweep
    #
    for idx, z in enumerate(z_list, 1):
        print(f"[{idx:>3}/{len(z_list)}] z={z:.5f}", end="\r")
        for name, (kind, method) in METHODS.items():
            if kind == "scipy":
                runner = lambda z=z, deg=method: scipy_roundtrip(img, z, deg)  # type: ignore[arg-type]
            elif kind == "splineops":
                runner = lambda z=z, m=method: spl_roundtrip(img, z, m)        # type: ignore[arg-type]
            elif kind == "torch":
                runner = lambda z=z, deg=method: torch_roundtrip(img, z, deg)  # type: ignore[arg-type]
            elif kind == "opencv":
                runner = lambda z=z, w=method: opencv_roundtrip(img, z, w)     # type: ignore[arg-type]
            elif kind == "pillow":
                runner = lambda z=z, w=method: pillow_roundtrip(img, z, w)     # type: ignore[arg-type]
            elif kind == "skimage":
                runner = lambda z=z, deg=method: skimage_roundtrip(img, z, deg)  # type: ignore[arg-type]
            else:
                continue

            try:
                rec, t_mean, t_sd = average_time(runner, repeats=args.repeats, warmup=True)
            except Exception as e:
                # If any method fails at a particular zoom, skip that sample
                print(f"\n[warn] {name} failed at z={z:.5f}: {e}")
                continue

            s = snr_db(img, rec)

            # NEW: SSIM (global, full image)
            if _HAS_SKIMAGE and sk_ssim is not None:
                try:
                    # Work in grayscale for SSIM, like in the benchmark.
                    # If img is 2D, use it directly; if it's RGB, convert to luma.
                    if img.ndim == 2:
                        ref = img
                        rec_ = rec
                    elif img.ndim == 3 and img.shape[2] >= 3:
                        # Convert to luma (same weights as load_image_any)
                        ref = (
                            0.2989 * img[..., 0]
                            + 0.5870 * img[..., 1]
                            + 0.1140 * img[..., 2]
                        )
                        rec_ = (
                            0.2989 * rec[..., 0]
                            + 0.5870 * rec[..., 1]
                            + 0.1140 * rec[..., 2]
                        )
                    else:
                        # Fallback: treat as scalar field
                        ref = img
                        rec_ = rec

                    # Global data range, analogous to the ROI code.
                    dr = float(ref.max() - ref.min())
                    if dr <= 0.0:
                        dr = 1.0  # flat image; arbitrary but safe

                    ssim_val = float(sk_ssim(ref, rec_, data_range=dr))
                except Exception:
                    ssim_val = float("nan")
            else:
                ssim_val = float("nan")

            results[name]["z"].append(z)
            results[name]["time"].append(t_mean)
            results[name]["time_sd"].append(t_sd)
            results[name]["snr"].append(s)
            results[name]["ssim"].append(ssim_val)  # NEW
    print("\nDone. Plotting...")

    #
    # Plot helpers
    #
    def plot_region(region: str):
        if region == "down":
            title_suffix = " (downsampling, 0 < z < 1)"
            mask_fn = lambda z: z < 1.0
        elif region == "up":
            title_suffix = " (upsampling, 1 < z < 2)"
            mask_fn = lambda z: z > 1.0
        elif region == "both":
            title_suffix = " (0 < z < 2)"
            # z_list is already filtered to (0,2) and |z-1|>NEAR_ONE_EPS,
            # but this keeps the logic explicit:
            mask_fn = lambda z: (z > 0.0) & (z < 2.0)
        else:
            return  # no-op

        # Prepare per-method markers for accessibility (B/W friendly)
        # and stagger marker positions so they don't overlap too much.
        marker_cycle = ["o", "s", "^", "v", "D", "x", "+", "*", "P", "X"]
        marker_for: Dict[str, str] = {}
        markevery_for: Dict[str, Tuple[int, int]] = {}
        for idx_name, name in enumerate(results.keys()):
            marker_for[name] = marker_cycle[idx_name % len(marker_cycle)]
            # Each method uses the same stride but a different phase offset
            offset = idx_name % MARK_EVERY_BASE
            markevery_for[name] = (offset, MARK_EVERY_BASE)

        # ---------------- Timing plot ----------------
        plt.figure(figsize=PLOT_FIGSIZE)
        any_curve = False
        for name, data in results.items():
            if not data["z"]:
                continue
            z_arr = np.array(data["z"], dtype=float)
            t_arr = np.array(data["time"], dtype=float)
            mask = mask_fn(z_arr)
            if not mask.any():
                continue
            any_curve = True
            plt.plot(
                z_arr[mask],
                t_arr[mask],
                marker=marker_for.get(name, "o"),
                markevery=markevery_for.get(name, (0, MARK_EVERY_BASE)),
                markersize=MARKER_SIZE,
                linewidth=LINEWIDTH,
                label=name,
            )
        if any_curve:
            plt.xlabel("Zoom factor", fontsize=PLOT_LABEL_FONTSIZE)
            plt.ylabel(
                f"Time (s)  [avg of {args.repeats} runs, forward + backward]",
                fontsize=PLOT_LABEL_FONTSIZE,
            )
            plt.title(
                f"Round-Trip Timing vs Zoom{title_suffix}  "
                f"(H×W = {H}×{W}, dtype={DTYPE_NAME}, degree={degree_label})",
                fontsize=PLOT_TITLE_FONTSIZE,
            )
            plt.xticks(fontsize=PLOT_TICK_FONTSIZE)
            plt.yticks(fontsize=PLOT_TICK_FONTSIZE)

            plt.grid(True, alpha=0.35)
            plt.legend(fontsize=PLOT_LEGEND_FONTSIZE)
            plt.tight_layout()

        # ---------------- SNR plot ----------------
        plt.figure(figsize=PLOT_FIGSIZE)
        any_curve = False
        for name, data in results.items():
            if not data["z"]:
                continue
            z_arr = np.array(data["z"], dtype=float)
            s_arr = np.array(data["snr"], dtype=float)
            mask = mask_fn(z_arr)
            if not mask.any():
                continue
            any_curve = True
            s_plot = np.where(np.isfinite(s_arr[mask]), s_arr[mask], np.nan)
            plt.plot(
                z_arr[mask],
                s_plot,
                marker=marker_for.get(name, "o"),
                markevery=markevery_for.get(name, (0, MARK_EVERY_BASE)),
                markersize=MARKER_SIZE,
                linewidth=LINEWIDTH,
                label=name,
            )
        if any_curve:
            plt.xlabel("Zoom factor", fontsize=PLOT_LABEL_FONTSIZE)
            plt.ylabel("SNR (dB)  [original vs recovered]", fontsize=PLOT_LABEL_FONTSIZE)
            plt.title(
                f"Round-Trip SNR vs Zoom{title_suffix}  "
                f"(H×W = {H}×{W}, dtype={DTYPE_NAME}, degree={degree_label})",
                fontsize=PLOT_TITLE_FONTSIZE,
            )
            plt.xticks(fontsize=PLOT_TICK_FONTSIZE)
            plt.yticks(fontsize=PLOT_TICK_FONTSIZE)

            plt.grid(True, alpha=0.35)
            plt.legend(fontsize=PLOT_LEGEND_FONTSIZE)
            plt.tight_layout()

        # ---------------- SSIM plot (NEW) ----------------
        if _HAS_SKIMAGE and sk_ssim is not None:
            plt.figure(figsize=PLOT_FIGSIZE)
            any_curve = False
            for name, data in results.items():
                if not data["z"]:
                    continue
                z_arr = np.array(data["z"], dtype=float)
                q_arr = np.array(data["ssim"], dtype=float)
                mask = mask_fn(z_arr)
                if not mask.any():
                    continue
                any_curve = True
                q_plot = np.where(np.isfinite(q_arr[mask]), q_arr[mask], np.nan)
                plt.plot(
                    z_arr[mask],
                    q_plot,
                    marker=marker_for.get(name, "o"),
                    markevery=markevery_for.get(name, (0, MARK_EVERY_BASE)),
                    markersize=MARKER_SIZE,
                    linewidth=LINEWIDTH,
                    label=name,
                )
            if any_curve:
                plt.xlabel("Zoom factor", fontsize=PLOT_LABEL_FONTSIZE)
                plt.ylabel("SSIM  [original vs recovered]", fontsize=PLOT_LABEL_FONTSIZE)
                plt.title(
                    f"Round-Trip SSIM vs Zoom{title_suffix}  "
                    f"(H×W = {H}×{W}, dtype={DTYPE_NAME}, degree={degree_label})",
                    fontsize=PLOT_TITLE_FONTSIZE,
                )
                plt.xticks(fontsize=PLOT_TICK_FONTSIZE)
                plt.yticks(fontsize=PLOT_TICK_FONTSIZE)
                plt.grid(True, alpha=0.35)
                plt.legend(fontsize=PLOT_LEGEND_FONTSIZE)
                plt.tight_layout()

    #
    # Plot selected regions
    #
    if args.which == "down":
        plot_region("down")
    elif args.which == "up":
        plot_region("up")
    else:  # "both"
        plot_region("both")

    plt.show()


def brush_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Small helper to clamp/validate arguments if you ever want to add a --dtype flag, etc.
    For now it just returns args unchanged.
    """
    return args


if __name__ == "__main__":
    main()
