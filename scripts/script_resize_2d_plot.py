# splineops/scripts/script_resize_2d_plot.py
"""
Sweep zoom factors in [0.01, 2.0) (2.0 excluded) while *excluding 1.0*, keep only those that
round-trip image size exactly, and compare four methods:

- SciPy cubic
- Standard cubic
- Least-Squares cubic (best AA)
- Oblique cubic (fast AA)

If --image is not provided, a file dialog pops up; canceling it prompts for a URL.

This version averages timing over N runs per zoom (default: 10) and displays
two plots: timing vs zoom and SNR vs zoom (no files are written to disk).
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

# SciPy cubic baseline
from scipy.ndimage import zoom as ndi_zoom

# Optional for URL
try:
    import requests
except Exception:
    requests = None

# TK dialogs for interactive selection
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# splineops
from splineops.resize.resize import resize as spl_resize


# -------------------------- UI / I/O helpers --------------------------

def choose_image_dialog() -> str | None:
    """Open a file dialog; if canceled, prompt for URL; return a path/URL or None."""
    root = tk.Tk()
    root.withdraw()
    root.update()

    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Images", ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")),
            ("PNG", "*.png"),
            ("JPEG", ("*.jpg", "*.jpeg")),
            ("TIFF", ("*.tif", "*.tiff")),
            ("All files", "*.*"),
        ],
        parent=root,
    )
    root.update()

    if path:
        try:
            Image.open(path).close()
            root.destroy()
            return path
        except Exception as e:
            messagebox.showerror("Open failed", f"Could not open file:\n{e}")
            root.destroy()
            return None

    # No file selected: ask for URL
    url = simpledialog.askstring("Image URL", "Paste an image URL (or Cancel):", parent=root)
    root.destroy()
    if url and url.strip():
        return url.strip()
    return None


def load_image_any(path_or_url: str, grayscale: bool = True) -> np.ndarray:
    """Load local path or URL into float64 [0,1]. If RGB and grayscale=True, convert."""
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
            out = 0.2989 * out[..., 0] + 0.5870 * out[..., 1] + 0.1140 * out[..., 2]
    return np.ascontiguousarray(out, dtype=np.float64)


def roundtrip_size_ok(shape: Tuple[int, ...], z: float) -> bool:
    """Accept z only if H,W -> round(H*z) then back with 1/z returns original."""
    if len(shape) < 2:
        return False
    H, W = int(shape[0]), int(shape[1])
    H1 = int(round(H * z)); W1 = int(round(W * z))
    if H1 <= 0 or W1 <= 0:
        return False
    H2 = int(round(H1 * (1.0 / z))); W2 = int(round(W1 * (1.0 / z)))
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

def scipy_cubic_roundtrip(img: np.ndarray, z: float) -> Tuple[np.ndarray, float]:
    zoom_fwd = (z, z) if img.ndim == 2 else (z, z, 1.0)
    zoom_bwd = (1.0 / z, 1.0 / z) if img.ndim == 2 else (1.0 / z, 1.0 / z, 1.0)
    t0 = time.perf_counter()
    out = ndi_zoom(img, zoom=zoom_fwd, order=3, mode="reflect", prefilter=True)
    rec = ndi_zoom(out, zoom=zoom_bwd, order=3, mode="reflect", prefilter=True)
    dt = time.perf_counter() - t0
    return rec, dt


def spl_roundtrip(img: np.ndarray, z: float, method: str) -> Tuple[np.ndarray, float]:
    zoom_fwd = (z, z) if img.ndim == 2 else (z, z, 1.0)
    zoom_bwd = (1.0 / z, 1.0 / z) if img.ndim == 2 else (1.0 / z, 1.0 / z, 1.0)
    t0 = time.perf_counter()
    out = spl_resize(img, zoom_factors=zoom_fwd, method=method)
    rec = spl_resize(out, zoom_factors=zoom_bwd, method=method)
    dt = time.perf_counter() - t0
    return rec, dt


def average_time(run, repeats: int = 10):
    """Return (last_rec, mean_time, std_time) over 'repeats' runs."""
    times = []
    rec = None
    for _ in range(max(1, repeats)):
        rec, dt = run()
        times.append(dt)
    times = np.asarray(times, dtype=np.float64)
    return rec, float(times.mean()), float(times.std(ddof=1 if len(times) > 1 else 0))


# ------------------------------ main -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Timing & SNR sweep with interactive image selection (averaged runs).")
    ap.add_argument("--image", type=str, default=None, help="Optional path/URL; if omitted, a dialog opens.")
    ap.add_argument("--samples", type=int, default=80, help="Number of zoom samples in [0.01, 2.0) (2.0 excluded).")
    ap.add_argument("--grayscale", type=int, default=1, help="1=convert to grayscale, 0=keep RGB.")
    ap.add_argument("--repeats", type=int, default=10, help="Average this many runs per (method, z).")
    args = ap.parse_args()

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

    # Zooms in [0.01, 2.0) (2.0 excluded), EXCLUDING 1.0
    z_candidates = np.linspace(0.01, 2.0, args.samples, endpoint=False, dtype=np.float64)
    # Guard against any accidental inclusion of 1.0
    z_candidates = z_candidates[np.abs(z_candidates - 1.0) > 1e-12]

    # Keep only round-trip-preserving zooms
    z_list = [float(z) for z in z_candidates if roundtrip_size_ok(img.shape, float(z))]
    if not z_list:
        print("No valid zoom factors after round-trip size check. Try increasing --samples.")
        sys.exit(1)
    print(f"Accepted {len(z_list)} / {len(z_candidates)} zooms (1.0 excluded).")

    METHODS = {
        "SciPy cubic": ("scipy", None),
        "Standard cubic": ("splineops", "cubic"),
        "Least-Squares (AA cubic)": ("splineops", "cubic-best_antialiasing"),
        "Oblique (fast AA cubic)": ("splineops", "cubic-fast_antialiasing"),
    }

    results: Dict[str, Dict[str, List[float]]] = {
        name: {"z": [], "time": [], "time_sd": [], "snr": []} for name in METHODS
    }

    for idx, z in enumerate(z_list, 1):
        print(f"[{idx:>3}/{len(z_list)}] z={z:.5f}", end="\r")
        for name, (kind, method) in METHODS.items():
            if kind == "scipy":
                fn = lambda z=z: scipy_cubic_roundtrip(img, z)
            else:
                fn = lambda z=z, m=method: spl_roundtrip(img, z, m)

            # Average timing over N runs; use last rec for SNR (deterministic).
            rec, t_mean, t_sd = average_time(fn, repeats=args.repeats)
            s = snr_db(img, rec)

            results[name]["z"].append(z)
            results[name]["time"].append(t_mean)
            results[name]["time_sd"].append(t_sd)
            results[name]["snr"].append(s)
    print("\nDone. Plotting...")

    # Timing vs zoom (mean only; uncomment errorbar section for SD bars)
    plt.figure(figsize=(9.5, 5.5))
    for name, data in results.items():
        z = np.array(data["z"], dtype=float)
        t = np.array(data["time"], dtype=float)
        plt.plot(z, t, marker="o", markersize=3, linewidth=1.5, label=name)
        # t_sd = np.array(data["time_sd"], dtype=float)
        # plt.errorbar(z, t, yerr=t_sd, fmt='none', ecolor='gray', alpha=0.2)
    plt.xlabel("Zoom factor (1.0 excluded)")
    plt.ylabel(f"Time (s)  [avg of {args.repeats} runs, forward + backward]")
    plt.title(f"Round-Trip Timing vs Zoom  (H×W = {H}×{W})")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    # SNR vs zoom
    plt.figure(figsize=(9.5, 5.5))
    for name, data in results.items():
        z = np.array(data["z"], dtype=float)
        s = np.array(data["snr"], dtype=float)
        s_plot = np.where(np.isfinite(s), s, np.nan)  # hide +inf for plotting
        plt.plot(z, s_plot, marker="o", markersize=3, linewidth=1.5, label=name)
    plt.xlabel("Zoom factor (1.0 excluded)")
    plt.ylabel("SNR (dB)  [original vs recovered]")
    plt.title(f"Round-Trip SNR vs Zoom  (H×W = {H}×{W})")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
