# splineops/scripts/script_resize_2d_plot_pytorch_opencv.py
"""
Sweep zoom factors in (0, 2) while *excluding 1.0*, and compare:

- Standard linear/cubic (splineops)
- Least-Squares linear/cubic (best AA, splineops)
- Oblique linear/cubic (fast AA, splineops)
- PyTorch bilinear/bicubic (antialiased, CPU)
- OpenCV INTER_LINEAR / INTER_CUBIC

If --image is not provided, a file dialog pops up; canceling it prompts for a URL.

This version averages timing over N runs per zoom (default: 10) and displays
timing and SNR vs zoom plots. You can plot downsampling (0<z<1), upsampling
(1<z<2), or both.

For each z, we:

  1. Resize to target size (H1, W1) ≈ (z·H, z·W)
  2. Resize back explicitly to (H, W)

so the recovered image always matches the original shape.
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

# PyQt5 dialogs for interactive selection
from PyQt5 import QtWidgets

# splineops
from splineops.resize import resize as spl_resize

# Default storage dtype for the sweep (change to np.float64 if desired)
DTYPE = np.float32
DTYPE_NAME = np.dtype(DTYPE).name

MARKER_SIZE = 3

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


def spl_roundtrip(img: np.ndarray, z: float, method: str) -> Tuple[np.ndarray, float]:
    """
    Round-trip with splineops.resize:

      forward:  img -> out,   output_size ≈ (z·H, z·W)
      backward: out -> rec,   output_size = (H, W)

    so rec has exactly the same shape as img.
    """
    H, W = img.shape[:2]
    H1 = max(1, int(round(H * z)))
    W1 = max(1, int(round(W * z)))

    t0 = time.perf_counter()

    if img.ndim == 2:
        out = spl_resize(img, output_size=(H1, W1), method=method)
        rec = spl_resize(out, output_size=(H, W), method=method)
    elif img.ndim == 3:
        C = img.shape[2]
        out = spl_resize(img, output_size=(H1, W1, C), method=method)
        rec = spl_resize(out, output_size=(H, W, C), method=method)
    else:
        raise ValueError("Expected 2D (H×W) or 3D (H×W×C) image for splineops path.")

    dt = time.perf_counter() - t0
    return np.asarray(rec, dtype=img.dtype), dt


def torch_roundtrip(img: np.ndarray, z: float, degree: str) -> Tuple[np.ndarray, float]:
    """
    Round-trip using torch.nn.functional.interpolate with bilinear (linear)
    or bicubic (cubic) + antialias=True. Runs on CPU.
    Works for 2D (H,W) and 3D (H,W,C) images.

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

    H, W = arr.shape[:2]
    H1 = max(1, int(round(H * z)))
    W1 = max(1, int(round(W * z)))

    if arr.ndim == 2:
        x = torch.from_numpy(arr).to(t_dtype).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        y = F.interpolate(
            x,
            size=(H1, W1),
            mode=mode,
            align_corners=False,
            antialias=True,
        )
        y2 = F.interpolate(
            y,
            size=(H, W),
            mode=mode,
            align_corners=False,
            antialias=True,
        )
        rec = y2[0, 0].cpu().numpy().astype(arr.dtype, copy=False)

    elif arr.ndim == 3:
        C = arr.shape[2]
        x = torch.from_numpy(arr).to(t_dtype).permute(2, 0, 1).unsqueeze(0)
        y = F.interpolate(
            x,
            size=(H1, W1),
            mode=mode,
            align_corners=False,
            antialias=True,
        )
        y2 = F.interpolate(
            y,
            size=(H, W),
            mode=mode,
            align_corners=False,
            antialias=True,
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
    img: np.ndarray, z: float, which: str = "cubic"
) -> Tuple[np.ndarray, float]:
    """
    Round-trip with OpenCV resize using the given interpolation.

      which in {"linear","cubic"}.

    Supports 2D (H,W) and 3D (H,W,C) arrays. Rec has exactly the same shape as img.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV not available")

    interp = {"linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC}[which]

    H, W = img.shape[:2]
    H1 = max(1, int(round(H * z)))
    W1 = max(1, int(round(W * z)))

    t0 = time.perf_counter()
    out = cv2.resize(img, (W1, H1), interpolation=interp)
    rec = cv2.resize(out, (W, H), interpolation=interp)
    dt = time.perf_counter() - t0

    rec = np.clip(rec, 0.0, 1.0)
    return rec.astype(img.dtype, copy=False), dt


def average_time(run, repeats: int = 10, warmup: bool = True):
    """
    Return (last_rec, mean_time, std_time) over 'repeats' runs.

    If warmup=True, run one extra un-timed warmup call (like script_resize_comparison._avg_time).
    """
    if warmup:
        run()  # warmup, ignore timing

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
        description="Timing & SNR sweep (splineops vs PyTorch vs OpenCV) with averaged runs."
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
        help="Degree / interpolation mode for splineops and other libs.",
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

    # Keep *all* remaining zooms (no roundtrip-size filtering)
    z_list = [float(z) for z in z_candidates]
    if not z_list:
        print("No zoom factors to test after basic range checks.")
        sys.exit(1)

    print(
        f"Using {len(z_list)} zoom factors "
        f"(down: {n_down}, up: {n_up}, 0<z<2, 1.0 excluded)."
    )

    #
    # Methods: splineops + PyTorch + OpenCV
    #
    METHODS: Dict[str, Tuple[str, str | None]] = {
        f"Standard {degree_label}": (
            "splineops",
            degree,
        ),
        f"Least-Squares (AA {degree_label})": (
            "splineops",
            f"{degree}-best_antialiasing",
        ),
        f"Oblique (fast AA {degree_label})": (
            "splineops",
            f"{degree}-fast_antialiasing",
        ),
    }
    if _HAS_TORCH:
        METHODS[f"PyTorch {degree_label} (AA)"] = ("torch", degree)
    else:
        print(
            "[info] PyTorch not found; 'PyTorch (AA)' curve will be omitted."
        )

    if _HAS_CV2:
        METHODS[f"OpenCV INTER_{degree_label.upper()}"] = ("opencv", degree)
    else:
        print(
            "[info] OpenCV not found; 'OpenCV' curve will be omitted."
        )

    results: Dict[str, Dict[str, List[float]]] = {
        name: {"z": [], "time": [], "time_sd": [], "snr": []} for name in METHODS
    }

    #
    # Run sweep
    #
    for idx, z in enumerate(z_list, 1):
        print(f"[{idx:>3}/{len(z_list)}] z={z:.5f}", end="\r")
        for name, (kind, method) in METHODS.items():
            if kind == "splineops":
                runner = lambda z=z, m=method: spl_roundtrip(img, z, m)
            elif kind == "torch":
                runner = lambda z=z, deg=method: torch_roundtrip(img, z, deg)
            elif kind == "opencv":
                runner = lambda z=z, w=method: opencv_roundtrip(img, z, w)
            else:
                continue

            try:
                rec, t_mean, t_sd = average_time(runner, repeats=args.repeats, warmup=True)
            except Exception as e:
                # If any method fails at a particular zoom, skip that sample for that method
                print(f"\n[warn] {name} failed at z={z:.5f}: {e}")
                continue

            s = snr_db(img, rec)

            results[name]["z"].append(z)
            results[name]["time"].append(t_mean)
            results[name]["time_sd"].append(t_sd)
            results[name]["snr"].append(s)
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
        else:
            return  # no-op

        # Assign distinct markers per method for B/W readability
        marker_cycle = ["o", "s", "^", "v", "D", "x", "+", "*", "P", "X"]
        marker_for: Dict[str, str] = {}
        for idx_name, name in enumerate(results.keys()):
            marker_for[name] = marker_cycle[idx_name % len(marker_cycle)]

        # Timing
        plt.figure(figsize=(9.5, 5.5))
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
                markersize=MARKER_SIZE,
                linewidth=1.5,
                label=name,
            )
        if any_curve:
            plt.xlabel("Zoom factor")
            plt.ylabel(
                f"Time (s)  [avg of {args.repeats} runs, forward + backward]"
            )
            plt.title(
                f"Round-Trip Timing vs Zoom{title_suffix}  "
                f"(H×W = {H}×{W}, dtype={DTYPE_NAME}, degree={degree_label})"
            )
            plt.grid(True, alpha=0.35)
            plt.legend()
            plt.tight_layout()

        # SNR
        plt.figure(figsize=(9.5, 5.5))
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
                markersize=MARKER_SIZE,
                linewidth=1.5,
                label=name,
            )
        if any_curve:
            plt.xlabel("Zoom factor")
            plt.ylabel("SNR (dB)  [original vs recovered]")
            plt.title(
                f"Round-Trip SNR vs Zoom{title_suffix}  "
                f"(H×W = {H}×{W}, dtype={DTYPE_NAME}, degree={degree_label})"
            )
            plt.grid(True, alpha=0.35)
            plt.legend()
            plt.tight_layout()

    #
    # Plot selected regions
    #
    if args.which in ("both", "down"):
        plot_region("down")
    if args.which in ("both", "up"):
        plot_region("up")

    plt.show()


def brush_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Small helper to clamp/validate arguments if you ever want to add a --dtype flag, etc.
    For now it just returns args unchanged.
    """
    return args


if __name__ == "__main__":
    main()
