# splineops/scripts/script_resize_2d_plot.py
"""
Sweep zoom factors in (0, 2) while *excluding 1.0*, keep only those that
round-trip image size exactly, and compare multiple methods:

- SciPy cubic
- Standard cubic (splineops)
- Least-Squares cubic (best AA, splineops)
- Oblique cubic (fast AA, splineops)
- PyTorch bicubic (antialiased, CPU)
- OpenCV INTER_AREA
- Pillow LANCZOS
- scikit-image (cubic, anti-aliased)

If --image is not provided, a file dialog pops up; canceling it prompts for a URL.

This version averages timing over N runs per zoom (default: 10) and displays
timing and SNR vs zoom plots. You can plot downsampling (0<z<1), upsampling
(1<z<2), or both.

By default the sweep runs in float32 for performance. You can change the
global DTYPE constant to np.float64 if you want full double precision.
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
except Exception:
    _HAS_CV2 = False

# Optional scikit-image (for comparison)
try:
    from skimage.transform import resize as sk_resize
    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

# TK dialogs for interactive selection
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# splineops
from splineops.resize.resize import resize as spl_resize

# Default storage dtype for the sweep (change to np.float64 if desired)
DTYPE = np.float32


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
                0.2989 * out[..., 0] +
                0.5870 * out[..., 1] +
                0.1140 * out[..., 2]
            )
    out = np.clip(out, 0.0, 1.0)
    return np.ascontiguousarray(out, dtype=DTYPE)


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
    return rec.astype(img.dtype, copy=False), dt


def spl_roundtrip(img: np.ndarray, z: float, method: str) -> Tuple[np.ndarray, float]:
    zoom_fwd = (z, z) if img.ndim == 2 else (z, z, 1.0)
    zoom_bwd = (1.0 / z, 1.0 / z) if img.ndim == 2 else (1.0 / z, 1.0 / z, 1.0)
    t0 = time.perf_counter()
    out = spl_resize(img, zoom_factors=zoom_fwd, method=method)
    rec = spl_resize(out, zoom_factors=zoom_bwd, method=method)
    dt = time.perf_counter() - t0
    return rec.astype(img.dtype, copy=False), dt


def torch_cubic_roundtrip(img: np.ndarray, z: float) -> Tuple[np.ndarray, float]:
    """
    Round-trip using torch.nn.functional.interpolate with bicubic + antialias=True.
    Runs on CPU. Works for 2D (H,W) and 3D (H,W,C) images.
    """
    if not _HAS_TORCH:
        raise RuntimeError("PyTorch not available")

    arr = img
    # Map numpy dtype to torch dtype
    if arr.dtype == np.float32:
        t_dtype = torch.float32
    elif arr.dtype == np.float64:
        t_dtype = torch.float64
    else:
        # upcast other types to float32
        t_dtype = torch.float32
        arr = arr.astype(np.float32, copy=False)

    if arr.ndim == 2:
        H, W = arr.shape
        x = torch.from_numpy(arr).to(t_dtype).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        H1 = int(round(H * z))
        W1 = int(round(W * z))
        t0 = time.perf_counter()
        y = F.interpolate(
            x,
            size=(H1, W1),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        y2 = F.interpolate(
            y,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        dt = time.perf_counter() - t0
        rec = y2[0, 0].cpu().numpy().astype(arr.dtype, copy=False)
        return rec, dt

    elif arr.ndim == 3:
        H, W, C = arr.shape
        # Convert H×W×C -> 1×C×H×W
        x = torch.from_numpy(arr).to(t_dtype).permute(2, 0, 1).unsqueeze(0)
        H1 = int(round(H * z))
        W1 = int(round(W * z))
        t0 = time.perf_counter()
        y = F.interpolate(
            x,
            size=(H1, W1),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        y2 = F.interpolate(
            y,
            size=(H, W),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        dt = time.perf_counter() - t0
        rec = y2[0].permute(1, 2, 0).cpu().numpy().astype(arr.dtype, copy=False)
        return rec, dt

    else:
        raise ValueError("Expected 2D (H×W) or 3D (H×W×C) image for PyTorch path.")


def opencv_roundtrip(img: np.ndarray, z: float, which: str) -> Tuple[np.ndarray, float]:
    """
    Round-trip with OpenCV resize using the given interpolation:
      which in {"area", "cubic", "lanczos"}.
    Supports 2D (H,W) and 3D (H,W,C) arrays.
    """
    if not _HAS_CV2:
        raise RuntimeError("OpenCV not available")

    interp = {
        "area":   cv2.INTER_AREA,
        "cubic":  cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
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


def pillow_roundtrip(img: np.ndarray, z: float, which: str) -> Tuple[np.ndarray, float]:
    """
    Round-trip with Pillow's resize using LANCZOS or BICUBIC.
    Supports 2D (H,W) and 3D (H,W,3) arrays.
    """
    resample_map = {
        "lanczos": Image.Resampling.LANCZOS,
        "bicubic": Image.Resampling.BICUBIC,
    }
    resample = resample_map[which]

    H, W = img.shape[:2]
    W1 = int(round(W * z))
    H1 = int(round(H * z))

    # Convert to uint8 for Pillow
    arr01 = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        u8 = np.rint(arr01 * 255.0).astype(np.uint8)
        im = Image.fromarray(u8, mode="L")
    elif img.ndim == 3 and img.shape[2] == 3:
        u8 = np.rint(arr01 * 255.0).astype(np.uint8)
        im = Image.fromarray(u8, mode="RGB")
    else:
        raise ValueError("Pillow round-trip expects 2D or 3D (H×W×3) array")

    t0 = time.perf_counter()
    out = im.resize((W1, H1), resample=resample)
    rec_im = out.resize((W, H), resample=resample)
    dt = time.perf_counter() - t0

    rec_arr = np.asarray(rec_im, dtype=np.float64) / 255.0
    rec_arr = np.clip(rec_arr, 0.0, 1.0)

    # If grayscale, rec_arr is (H,W); if RGB, (H,W,3)
    return rec_arr.astype(img.dtype, copy=False), dt


def skimage_roundtrip(img: np.ndarray, z: float) -> Tuple[np.ndarray, float]:
    """
    Round-trip with scikit-image.transform.resize using cubic (order=3) + anti_aliasing=True.
    Supports 2D (H,W) and 3D (H,W,C) arrays.
    """
    if not _HAS_SKIMAGE:
        raise RuntimeError("scikit-image not available")

    arr = np.asarray(img, dtype=np.float64)
    H, W = arr.shape[:2]
    H1 = int(round(H * z))
    W1 = int(round(W * z))

    t0 = time.perf_counter()

    if arr.ndim == 2:
        out = sk_resize(
            arr,
            (H1, W1),
            order=3,
            anti_aliasing=True,
            preserve_range=True,
            mode="reflect",
        )
        rec = sk_resize(
            out,
            (H, W),
            order=3,
            anti_aliasing=True,
            preserve_range=True,
            mode="reflect",
        )
    elif arr.ndim == 3:
        C = arr.shape[2]
        out = sk_resize(
            arr,
            (H1, W1, C),
            order=3,
            anti_aliasing=True,
            preserve_range=True,
            mode="reflect",
        )
        rec = sk_resize(
            out,
            (H, W, C),
            order=3,
            anti_aliasing=True,
            preserve_range=True,
            mode="reflect",
        )
    else:
        raise ValueError("scikit-image round-trip expects 2D or 3D (H×W×C) array")

    dt = time.perf_counter() - t0

    rec = np.clip(rec, 0.0, 1.0)
    return rec.astype(img.dtype, copy=False), dt


def average_time(run, repeats: int = 10):
    """Return (last_rec, mean_time, std_time) over 'repeats' runs."""
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
        description="Timing & SNR sweep with interactive image selection (averaged runs)."
    )
    ap.add_argument("--image", type=str, default=None, help="Optional path/URL; if omitted, a dialog opens.")
    ap.add_argument("--samples", type=int, default=200,
                    help="Base number of zoom samples per side if --samples-down/--samples-up are not given.")
    ap.add_argument("--samples-down", type=int, default=None,
                    help="Number of zoom samples in the interval (0, 1). Overrides --samples if set.")
    ap.add_argument("--samples-up", type=int, default=None,
                    help="Number of zoom samples in the interval (1, 2). Overrides --samples if set.")
    ap.add_argument(
        "--which",
        type=str,
        default="down",
        choices=("both", "down", "up"),
        help="Which zoom regime to plot: 'down' (0<z<1), 'up' (1<z<2), or 'both'.",
    )
    ap.add_argument("--grayscale", type=int, default=1, help="1=convert to grayscale, 0=keep RGB.")
    ap.add_argument("--repeats", type=int, default=10, help="Average this many runs per (method, z).")
    args = brush_args(ap.parse_args())

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
    n_up   = args.samples_up   if args.samples_up   is not None else args.samples

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
        print("No valid zoom factors after round-trip size check. "
              "Try increasing --samples-down/--samples-up or reducing NEAR_ONE_EPS.")
        sys.exit(1)

    print(
        f"Accepted {len(z_list)} / {len(z_candidates)} zooms "
        f"(down: {n_down}, up: {n_up}, |z-1|>{NEAR_ONE_EPS}, 2.0 excluded)."
    )

    #
    # Methods
    #
    METHODS: Dict[str, Tuple[str, str | None]] = {
        "SciPy cubic":               ("scipy",     None),
        "Standard cubic":            ("splineops", "cubic"),
        "Least-Squares (AA cubic)":  ("splineops", "cubic-best_antialiasing"),
        "Oblique (fast AA cubic)":   ("splineops", "cubic-fast_antialiasing"),
    }
    if _HAS_TORCH:
        METHODS["PyTorch bicubic (AA)"] = ("torch", None)
    else:
        print("[info] PyTorch not found; 'PyTorch bicubic (AA)' curve will be omitted.")

    if _HAS_CV2:
        METHODS["OpenCV INTER_AREA"] = ("opencv", "area")
    else:
        print("[info] OpenCV not found; 'OpenCV INTER_AREA' curve will be omitted.")

    # Pillow is always available (we already import PIL.Image above)
    METHODS["Pillow LANCZOS"] = ("pillow", "lanczos")

    if _HAS_SKIMAGE:
        METHODS["scikit-image (cubic, AA)"] = ("skimage", None)
    else:
        print("[info] scikit-image not found; 'scikit-image (cubic, AA)' curve will be omitted.")

    results: Dict[str, Dict[str, List[float]]] = {
        name: {"z": [], "time": [], "time_sd": [], "snr": []} for name in METHODS
    }

    #
    # Run sweep
    #
    for idx, z in enumerate(z_list, 1):
        print(f"[{idx:>3}/{len(z_list)}] z={z:.5f}", end="\r")
        for name, (kind, method) in METHODS.items():
            if kind == "scipy":
                runner = lambda z=z: scipy_cubic_roundtrip(img, z)
            elif kind == "splineops":
                runner = lambda z=z, m=method: spl_roundtrip(img, z, m)
            elif kind == "torch":
                runner = lambda z=z: torch_cubic_roundtrip(img, z)
            elif kind == "opencv":
                runner = lambda z=z, w=method: opencv_roundtrip(img, z, w)
            elif kind == "pillow":
                runner = lambda z=z, w=method: pillow_roundtrip(img, z, w)
            elif kind == "skimage":
                runner = lambda z=z: skimage_roundtrip(img, z)
            else:
                continue

            try:
                rec, t_mean, t_sd = average_time(runner, repeats=args.repeats)
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

        # Timing
        plt.figure(figsize=(9.5, 5.5))
        any_curve = False
        for name, data in results.items():
            if not data["z"]:
                continue
            z = np.array(data["z"], dtype=float)
            t = np.array(data["time"], dtype=float)
            mask = mask_fn(z)
            if not mask.any():
                continue
            any_curve = True
            plt.plot(z[mask], t[mask], marker="o", markersize=3, linewidth=1.5, label=name)
        if any_curve:
            plt.xlabel("Zoom factor")
            plt.ylabel(f"Time (s)  [avg of {args.repeats} runs, forward + backward]")
            plt.title(f"Round-Trip Timing vs Zoom{title_suffix}  (H×W = {H}×{W}, dtype={DTYPE})")
            plt.grid(True, alpha=0.35)
            plt.legend()
            plt.tight_layout()

        # SNR
        plt.figure(figsize=(9.5, 5.5))
        any_curve = False
        for name, data in results.items():
            if not data["z"]:
                continue
            z = np.array(data["z"], dtype=float)
            s = np.array(data["snr"], dtype=float)
            mask = mask_fn(z)
            if not mask.any():
                continue
            any_curve = True
            s_plot = np.where(np.isfinite(s[mask]), s[mask], np.nan)
            plt.plot(z[mask], s_plot, marker="o", markersize=3, linewidth=1.5, label=name)
        if any_curve:
            plt.xlabel("Zoom factor")
            plt.ylabel("SNR (dB)  [original vs recovered]")
            plt.title(f"Round-Trip SNR vs Zoom{title_suffix}  (H×W = {H}×{W}, dtype={DTYPE})")
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
