# splineops/scripts/script_resize_2d_plot_linear_cubic.py
# -*- coding: utf-8 -*-
"""
Timing & SNR sweep over zoom factors, comparing splineops linear vs cubic:

- Standard Linear
- Standard Cubic
- Least-Squares Linear (best AA)
- Least-Squares Cubic (best AA)
- Oblique Linear (fast AA)
- Oblique Cubic (fast AA)

Zoom sweep:
  • 0 < z < 2, excluding 1.0
  • Only round-trip-size-preserving zooms are kept.
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

# Optional for URL loading
try:
    import requests
except Exception:
    requests = None

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


def spl_roundtrip(img: np.ndarray, z: float, method: str) -> Tuple[np.ndarray, float]:
    """
    Round-trip with splineops.resize using zoom_factors:

      forward:  img -> out,   zoom_factors = (z, z [,1])
      backward: out -> rec,   zoom_factors = (1/z, 1/z [,1])

    so rec has exactly the same shape as img (by construction).
    """
    zoom_fwd = (z, z) if img.ndim == 2 else (z, z, 1.0)
    zoom_bwd = (1.0 / z, 1.0 / z) if img.ndim == 2 else (1.0 / z, 1.0 / z, 1.0)

    t0 = time.perf_counter()
    out = spl_resize(img, zoom_factors=zoom_fwd, method=method)
    rec = spl_resize(out, zoom_factors=zoom_bwd, method=method)
    dt = time.perf_counter() - t0

    rec = np.clip(rec, 0.0, 1.0)
    return rec.astype(img.dtype, copy=False), dt


def average_time(run, repeats: int = 10, warmup: bool = True):
    """
    Return (last_rec, mean_time, std_time) over 'repeats' runs.

    If warmup=True, run one extra un-timed warmup call (like script_resize_comparison._avg_time).
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
        description="Timing & SNR sweep for splineops linear vs cubic (Standard/LS/Oblique)."
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

    args = brush_args(ap.parse_args())

    # Ensure a Qt application exists before showing dialogs
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

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
    # Methods: splineops Standard / LS / Oblique, linear + cubic
    #
    METHODS: Dict[str, str] = {
        "Standard Linear": "linear",
        "Standard Cubic": "cubic",
        "Least-Squares Linear (AA)": "linear-best_antialiasing",
        "Least-Squares Cubic (AA)": "cubic-best_antialiasing",
        "Oblique Linear (fast AA)": "linear-fast_antialiasing",
        "Oblique Cubic (fast AA)": "cubic-fast_antialiasing",
    }

    results: Dict[str, Dict[str, List[float]]] = {
        name: {"z": [], "time": [], "time_sd": [], "snr": []} for name in METHODS
    }

    #
    # Run sweep
    #
    for idx, z in enumerate(z_list, 1):
        print(f"[{idx:>3}/{len(z_list)}] z={z:.5f}", end="\r")
        for name, method in METHODS.items():
            runner = lambda z=z, m=method: spl_roundtrip(img, z, m)

            try:
                rec, t_mean, t_sd = average_time(
                    runner, repeats=args.repeats, warmup=True
                )
            except Exception as e:
                # If any method fails at a particular zoom, skip that sample
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
                f"(H×W = {H}×{W}, dtype={DTYPE_NAME})",
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
                markersize=MARKER_SIZE,
                linewidth=LINEWIDTH,
                label=name,
            )
        if any_curve:
            plt.xlabel("Zoom factor", fontsize=PLOT_LABEL_FONTSIZE)
            plt.ylabel("SNR (dB)  [original vs recovered]", fontsize=PLOT_LABEL_FONTSIZE)
            plt.title(
                f"Round-Trip SNR vs Zoom{title_suffix}  "
                f"(H×W = {H}×{W}, dtype={DTYPE_NAME})",
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
