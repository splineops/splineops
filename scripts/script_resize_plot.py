# splineops/scripts/script_resize_plot.py
"""
Sweep zoom factors in [0.01, 2.0], keep only those that round-trip sizes exactly,
and compare four methods:

- SciPy cubic                  (scipy.ndimage.zoom order=3)
- Standard cubic               (splineops.resize(..., method="cubic"))
- Least-Squares cubic (AA)     (method="cubic-best_antialiasing")
- Oblique cubic (fast AA)      (method="cubic-fast_antialiasing")

For each accepted zoom z:
  1) forward resize with z
  2) backward "revert" with 1/z
  3) record total time (forward+backward) and SNR(original, recovered)

Outputs two plots: timing vs zoom and SNR vs zoom.

Usage (defaults to a Kodak image URL and 80 zoom samples):
    python script_resize_plot.py
    python script_resize_plot.py --image PATH/OR/URL --samples 120 --grayscale 1

Tip: for maximum native speed in splineops, set env before running:
    set SPLINEOPS_ACCEL=always   (Windows cmd)
    export SPLINEOPS_ACCEL=always (bash/zsh)
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# SciPy cubic
from scipy.ndimage import zoom as ndi_zoom

# Image I/O
from PIL import Image
import io
import os
import sys

try:
    import requests
except Exception:
    requests = None  # allow using only local files

# Our resizer
from splineops.resize.resize import resize as spl_resize


def load_image_any(path_or_url: str, grayscale: bool = True) -> np.ndarray:
    """Load image from local path or URL -> float64 in [0,1]."""
    data: np.ndarray
    if "://" in path_or_url:
        if requests is None:
            raise RuntimeError("requests not available; use a local file or install requests.")
        resp = requests.get(path_or_url, timeout=10)
        resp.raise_for_status()
        img = Image.open(io.BytesIO(resp.content))
    else:
        img = Image.open(path_or_url)

    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 2:
        out = arr / 255.0
    else:
        out = arr / 255.0
        if grayscale:
            # luminance-ish weights
            out = 0.2989 * out[..., 0] + 0.5870 * out[..., 1] + 0.1140 * out[..., 2]
    return out


def roundtrip_size_ok(shape: Tuple[int, ...], z: float) -> bool:
    """Accept z only if rounding H,W -> round(H*z) and back with 1/z returns original."""
    if len(shape) < 2:
        return False
    H, W = int(shape[0]), int(shape[1])
    H1 = int(round(H * z)); W1 = int(round(W * z))
    if H1 <= 0 or W1 <= 0:
        return False
    H2 = int(round(H1 * (1.0 / z))); W2 = int(round(W1 * (1.0 / z)))
    return (H2 == H) and (W2 == W)


def snr_db(x: np.ndarray, y: np.ndarray) -> float:
    """10*log10( sum(x^2)/sum((x-y)^2) ). If identical, returns +inf."""
    num = np.sum(x * x, dtype=np.float64)
    den = np.sum((x - y) ** 2, dtype=np.float64)
    if den == 0.0:
        return float("inf")
    if num == 0.0:
        return -float("inf")
    return 10.0 * math.log10(num / den)


def scipy_cubic_roundtrip(img: np.ndarray, z: float) -> Tuple[np.ndarray, float]:
    """Forward then backward with SciPy cubic. Return (recovered, total_time_s)."""
    zoom_fwd = (z, z) if img.ndim == 2 else (z, z, 1.0)
    zoom_bwd = (1.0 / z, 1.0 / z) if img.ndim == 2 else (1.0 / z, 1.0 / z, 1.0)
    t0 = time.perf_counter()
    out = ndi_zoom(img, zoom=zoom_fwd, order=3, mode="reflect", prefilter=True)
    rec = ndi_zoom(out, zoom=zoom_bwd, order=3, mode="reflect", prefilter=True)
    dt = time.perf_counter() - t0
    return rec, dt


def spl_roundtrip(img: np.ndarray, z: float, method: str) -> Tuple[np.ndarray, float]:
    """Forward then backward with splineops.resize() for a given method."""
    zoom_fwd = (z, z) if img.ndim == 2 else (z, z, 1.0)
    zoom_bwd = (1.0 / z, 1.0 / z) if img.ndim == 2 else (1.0 / z, 1.0 / z, 1.0)
    t0 = time.perf_counter()
    out = spl_resize(img, zoom_factors=zoom_fwd, method=method)
    rec = spl_resize(out, zoom_factors=zoom_bwd, method=method)
    dt = time.perf_counter() - t0
    return rec, dt


def best_of(func, repeats: int = 2):
    """Run function several times, return best (recovered, min_time)."""
    best_t = float("inf")
    best_rec = None
    for _ in range(max(1, repeats)):
        rec, dt = func()
        if dt < best_t:
            best_t = dt
            best_rec = rec
    return best_rec, best_t


def main():
    parser = argparse.ArgumentParser(description="Compare resizers across zoom range.")
    parser.add_argument(
        "--image",
        type=str,
        default="https://r0k.us/graphics/kodak/kodak/kodim14.png",
        help="Local path or URL to an image (default: Kodak kodim14.png).",
    )
    parser.add_argument("--samples", type=int, default=80, help="Number of zoom samples in [0.01, 2.0].")
    parser.add_argument("--grayscale", type=int, default=1, help="1 to convert to grayscale, 0 to keep RGB.")
    parser.add_argument("--repeats", type=int, default=2, help="Best-of repeats per (method,zoom).")
    parser.add_argument("--save_prefix", type=str, default="resize", help="Prefix for saved plot files.")
    args = parser.parse_args()

    # Load input
    img = load_image_any(args.image, grayscale=bool(args.grayscale))
    img = np.ascontiguousarray(img, dtype=np.float64)  # make sure C-contig

    H, W = int(img.shape[0]), int(img.shape[1])
    print(f"Loaded image: shape={img.shape}, dtype={img.dtype}")

    # Zoom candidates
    z_candidates = np.linspace(0.01, 2.0, args.samples, dtype=np.float64)
    # Ensure we include exactly 1.0
    z_candidates = np.unique(np.append(z_candidates, 1.0))
    # Filter by round-trip shape condition
    z_list = [float(z) for z in z_candidates if roundtrip_size_ok(img.shape, float(z))]
    if not z_list:
        print("No valid zoom factors after round-trip size filtering; try --samples larger.", file=sys.stderr)
        sys.exit(1)

    print(f"Accepted {len(z_list)} / {len(z_candidates)} zooms after round-trip size check.")

    # Methods to test
    METHODS = {
        "SciPy cubic": ("scipy", None),
        "Standard cubic": ("splineops", "cubic"),
        "Least-Squares (AA cubic)": ("splineops", "cubic-best_antialiasing"),
        "Oblique (fast AA cubic)": ("splineops", "cubic-fast_antialiasing"),
    }

    results: Dict[str, Dict[str, List[float]]] = {
        name: {"z": [], "time": [], "snr": []} for name in METHODS
    }

    # Sweep
    for zi, z in enumerate(z_list, 1):
        print(f"[{zi:>3}/{len(z_list)}] z = {z:.4f}", end="\r")
        for name, (kind, method) in METHODS.items():
            if kind == "scipy":
                fn = lambda z=z: scipy_cubic_roundtrip(img, z)
            else:
                fn = lambda z=z, method=method: spl_roundtrip(img, z, method)

            rec, tsec = best_of(fn, repeats=args.repeats)
            s = snr_db(img, rec)
            results[name]["z"].append(z)
            results[name]["time"].append(tsec)
            results[name]["snr"].append(s)

    print("\nDone. Plotting...")

    # ---- Plot: timing vs zoom ----
    plt.figure(figsize=(9.5, 5.5))
    for name, data in results.items():
        z = np.array(data["z"], dtype=float)
        t = np.array(data["time"], dtype=float)
        plt.plot(z, t, marker="o", markersize=3, linewidth=1.5, label=name)
    plt.xlabel("Zoom factor")
    plt.ylabel("Time (s)  [forward + backward]")
    plt.title(f"Resize Round-Trip Timing vs Zoom  (H×W = {H}×{W}, repeats={args.repeats})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    timing_path = f"{args.save_prefix}_timing_vs_zoom.png"
    plt.savefig(timing_path, dpi=140)
    print(f"Saved: {timing_path}")

    # ---- Plot: SNR vs zoom ----
    plt.figure(figsize=(9.5, 5.5))
    for name, data in results.items():
        z = np.array(data["z"], dtype=float)
        s = np.array(data["snr"], dtype=float)
        # Clip inf for plotting
        s_plot = np.where(np.isfinite(s), s, np.nan)
        plt.plot(z, s_plot, marker="o", markersize=3, linewidth=1.5, label=name)
    plt.xlabel("Zoom factor")
    plt.ylabel("SNR (dB)  [original vs recovered]")
    plt.title(f"Resize Round-Trip SNR vs Zoom  (H×W = {H}×{W})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    snr_path = f"{args.save_prefix}_snr_vs_zoom.png"
    plt.savefig(snr_path, dpi=140)
    print(f"Saved: {snr_path}")

    # Show interactive
    plt.show()


if __name__ == "__main__":
    main()
