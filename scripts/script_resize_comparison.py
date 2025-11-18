# -*- coding: utf-8 -*-
"""
script_resize_comparison.py

Compare splineops interpolation against common stacks at a chosen zoom:

- splineops: Standard (cubic), Least-Squares (best AA cubic), Oblique (fast AA cubic)
- SciPy ndimage.zoom (cubic)
- OpenCV (INTER_AREA, INTER_CUBIC, INTER_LANCZOS4)
- Pillow (LANCZOS, BICUBIC)
- scikit-image (resize, order=3, anti_aliasing=True)
- PyTorch (F.interpolate bicubic, antialias=True, CPU)

pip install opencv-python scikit-image torch torchvision

Workflow
--------
1) Pick an image (dialog). If you cancel, you'll be asked for a URL.
2) Enter zoom factor z (>0), e.g. 0.3 for downscale or 1.7 for upscale.
3) Script runs round-trip per method (z, then 1/z), averages timing over N runs.
4) Computes SNR/MSE over a central crop and shows:
   - ROI montage (nearest-neighbor magnified)
   - Bar charts for timing and SNR

Notes
-----
- All ops run on grayscale float64 images normalized to [0, 1].
- Methods with missing deps are marked "Unavailable" and skipped.
- For fairness we use mirror/reflect-like boundaries when available.
"""

from __future__ import annotations

import io
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Default storage dtype for comparison (change to np.float64 if desired)
DTYPE = np.float32

# Optional deps (gracefully skipped when missing)
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    from scipy.ndimage import zoom as _ndi_zoom
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    from skimage.transform import resize as _sk_resize
    from skimage.metrics import structural_similarity as _ssim
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

# ---- Tk file/URL dialogs ----
try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog, messagebox
    _HAS_TK = True
except Exception:
    _HAS_TK = False
    filedialog = None
    simpledialog = None
    messagebox = None


# ---------------------------
# Utilities
# ---------------------------
def nearest_roundtrip_zoom(shape: Tuple[int, int], z: float,
                           max_delta: float = 0.02) -> float:
    """
    Find the closest zoom z' near z such that:
      round(round(H*z') / z') == H and round(round(W*z') / z') == W

    This makes naive round-trip with 1/z' shape-exact for libs that use 'round'
    in their sizing. Searches in a small neighborhood around z.
    """
    H, W = int(shape[0]), int(shape[1])

    def ok(zz: float) -> bool:
        H1 = int(round(H * zz)); W1 = int(round(W * zz))
        if H1 < 1 or W1 < 1:
            return False
        H2 = int(round(H1 * (1.0 / zz))); W2 = int(round(W1 * (1.0 / zz)))
        return (H2 == H) and (W2 == W)

    if z > 0 and ok(z):
        return z

    # The smallest zoom tweak that flips a rounded size is ~1/H or ~1/W.
    base_step = 0.25 * min(1.0 / max(H, 1), 1.0 / max(W, 1))
    steps = max(1, int(max_delta / base_step))

    best_z = z
    best_d = float("inf")

    for k in range(1, steps + 1):
        for sgn in (-1, +1):
            zz = z + sgn * k * base_step
            if zz <= 0:
                continue
            if ok(zz):
                return zz
            d = abs(zz - z)
            if d < best_d:
                best_z, best_d = zz, d

    # If none satisfies the strict round/round condition, return the closest tried
    return best_z

def _snr_db(x: np.ndarray, y: np.ndarray) -> float:
    num = float(np.sum(x * x, dtype=np.float64))
    den = float(np.sum((x - y) ** 2, dtype=np.float64))
    if den == 0.0:
        return float("inf")
    if num == 0.0:
        return -float("inf")
    return 10.0 * math.log10(num / den)

def _central_crop(arr: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """Take a central crop with height/width = (1 - 2*frac) of the image."""
    h, w = arr.shape[:2]
    dh = int(round(h * frac))
    dw = int(round(w * frac))
    return arr[dh:h - dh if dh else h, dw:w - dw if dw else w]

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
    if not _HAS_TK:
        return None
    root = tk.Tk()
    root.withdraw()
    root.update()
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[
            ("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
            ("All files", "*.*"),
        ],
    )
    root.update()
    if path:
        try:
            Image.open(path).close()
            root.destroy()
            return path
        except Exception as e:
            messagebox.showerror("Open failed", f"Could not open file:\n{e}", parent=root)
            root.destroy()
            return None

    # Fallback: ask for URL
    url = simpledialog.askstring("Image URL", "Paste an image URL (or Cancel):", parent=root)
    root.destroy()
    return url.strip() if url else None

def _ask_zoom_factor(default: float = 0.3) -> Optional[float]:
    if not _HAS_TK:
        return default
    root = tk.Tk(); root.withdraw(); root.update()
    s = simpledialog.askstring("Zoom factor", "Enter zoom factor (>0):", initialvalue=str(default), parent=root)
    root.destroy()
    if s is None:
        return None
    try:
        z = float(s)
        if not np.isfinite(z) or z <= 0:
            return None
        return z
    except Exception:
        return None


# ---------------------------
# Normalization to grayscale [0,1]
# ---------------------------
def _to_gray01(im: Image.Image) -> np.ndarray:
    # Drop alpha for simplicity
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGB")
    if im.mode == "L":
        arr = np.asarray(im, dtype=np.float64) / 255.0
    elif im.mode in ("I;16", "I"):
        arr = np.asarray(im, dtype=np.float64)
        # Normalize robustly to [0,1]
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
# Backends (round-trip z → 1/z)
# ---------------------------
def _rt_splineops(gray: np.ndarray, z: float, preset: str) -> Tuple[np.ndarray, Optional[str]]:
    if not _HAS_SPLINEOPS:
        return gray, f"splineops unavailable: {_SPLINEOPS_IMPORT_ERR}"
    try:
        out = sp_resize(gray, zoom_factors=(z, z), method=preset)
        rec = sp_resize(out, output_size=gray.shape, method=preset)
        return np.clip(rec, 0.0, 1.0), None
    except Exception as e:
        return gray, str(e)

def _rt_scipy(gray: np.ndarray, z: float) -> Tuple[np.ndarray, Optional[str]]:
    if not _HAS_SCIPY:
        return gray, "SciPy not installed"
    try:
        out = _ndi_zoom(
            gray, (z, z), order=3, prefilter=True, mode="reflect", grid_mode=False
        )
        # Backward factors chosen to return EXACT original shape:
        Hz, Wz = out.shape
        back = (gray.shape[0] / Hz, gray.shape[1] / Wz)
        rec = _ndi_zoom(
            out, back, order=3, prefilter=True, mode="reflect", grid_mode=False
        )
        # Clip & ensure exact target shape by central trim/pad if any 1-px mismatch remains
        rec = np.clip(rec, 0.0, 1.0)
        if rec.shape != gray.shape:
            h = min(rec.shape[0], gray.shape[0])
            w = min(rec.shape[1], gray.shape[1])
            # central crop to common size, then pad back if needed (rare)
            r0 = (rec.shape[0] - h) // 2; r1 = r0 + h
            c0 = (rec.shape[1] - w) // 2; c1 = c0 + w
            rc = rec[r0:r1, c0:c1]
            g0 = (gray.shape[0] - h) // 2; g1 = g0 + h
            g2 = (gray.shape[1] - w) // 2; g3 = g2 + w
            # place into a new array matching gray.shape
            tmp = np.zeros_like(gray)
            tmp[g0:g1, g2:g3] = rc
            rec = tmp
        return rec, None
    except Exception as e:
        return gray, str(e)

def _rt_opencv(gray: np.ndarray, z: float, which: str) -> Tuple[np.ndarray, Optional[str]]:
    if not _HAS_CV2:
        return gray, "OpenCV not installed"
    try:
        interp = {"area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}[which]
        H, W = gray.shape
        out = cv2.resize(gray, (int(round(W * z)), int(round(H * z))), interpolation=interp)
        rec = cv2.resize(out, (W, H), interpolation=interp)
        return np.clip(rec, 0.0, 1.0), None
    except Exception as e:
        return gray, str(e)

def _rt_pillow(gray: np.ndarray, z: float, which: str) -> Tuple[np.ndarray, Optional[str]]:
    try:
        from PIL import Image
        resample = {"bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}[which]
        H, W = gray.shape
        im = Image.fromarray(np.rint(np.clip(gray, 0, 1) * 255.0).astype(np.uint8), mode="L")
        out = im.resize((int(round(W * z)), int(round(H * z))), resample=resample)
        rec = out.resize((W, H), resample=resample)
        arr = np.asarray(rec, dtype=np.float64) / 255.0
        return np.clip(arr, 0.0, 1.0), None
    except Exception as e:
        return gray, str(e)

def _rt_skimage(gray: np.ndarray, z: float) -> Tuple[np.ndarray, Optional[str]]:
    if not _HAS_SKIMAGE:
        return gray, "scikit-image not installed"
    try:
        H, W = gray.shape
        out = _sk_resize(gray, (int(round(H * z)), int(round(W * z))),
                         order=3, anti_aliasing=True, preserve_range=True, mode="reflect").astype(np.float64)
        rec = _sk_resize(out, (H, W),
                         order=3, anti_aliasing=True, preserve_range=True, mode="reflect").astype(np.float64)
        return np.clip(rec, 0.0, 1.0), None
    except Exception as e:
        return gray, str(e)

def _rt_torch(gray: np.ndarray, z: float) -> Tuple[np.ndarray, Optional[str]]:
    if not _HAS_TORCH:
        return gray, "PyTorch not installed"
    try:
        x = torch.from_numpy(gray[None, None].astype(np.float32))
        H, W = gray.shape
        y = F.interpolate(x, size=(int(round(H * z)), int(round(W * z))),
                          mode="bicubic", align_corners=False, antialias=True)
        y2 = F.interpolate(y, size=(H, W),
                           mode="bicubic", align_corners=False, antialias=True)
        arr = y2[0, 0].detach().cpu().numpy().astype(np.float64)
        return np.clip(arr, 0.0, 1.0), None
    except Exception as e:
        return gray, str(e)


# ---------------------------
# Benchmark harness
# ---------------------------
def _avg_time(fn, repeats: int = 10, warmup: bool = True) -> Tuple[np.ndarray, float, float, Optional[str]]:
    """Return (last_rec, mean_time, std_time, error)."""
    if warmup:
        try:
            _ = fn()
        except Exception as e:
            return np.array([]), float("nan"), float("nan"), str(e)

    times = []
    rec = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        rec, err = fn()
        if err is not None:
            return np.array([]), float("nan"), float("nan"), err
        times.append(time.perf_counter() - t0)
    t = np.asarray(times, dtype=np.float64)
    return rec, float(t.mean()), float(t.std(ddof=1 if len(t) > 1 else 0)), None


def main():
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
    repeats = 10          # avg runs
    border_frac = 0.2     # central crop for metrics
    roi = _central_crop(gray, border_frac)

    # --- Methods to compare ---
    methods = [
        ("Splineops — Standard cubic",      lambda: _rt_splineops(gray, z, "cubic")),
        ("Splineops — LS (best AA) cubic",  lambda: _rt_splineops(gray, z, "cubic-best_antialiasing")),
        ("Splineops — Oblique (fast AA)",   lambda: _rt_splineops(gray, z, "cubic-fast_antialiasing")),
        ("SciPy cubic",                     lambda: _rt_scipy(gray, z)),
        # Single OpenCV baseline (INTER_AREA is a good downsampling reference)
        ("OpenCV INTER_AREA",               lambda: _rt_opencv(gray, z, "area")),
        # Single Pillow baseline (LANCZOS as “high-quality” reference)
        ("Pillow LANCZOS",                  lambda: _rt_pillow(gray, z, "lanczos")),
        ("scikit-image (cubic, AA)",        lambda: _rt_skimage(gray, z)),
        ("PyTorch bicubic (AA, CPU)",       lambda: _rt_torch(gray, z)),
    ]

    rows: List[Dict] = []
    roi_tiles = []

    print(f"\nBenchmarking round-trip @ zoom ×{z:.5g}  (repeats={repeats})\n")
    header = f"{'Method':<34} {'Time (mean)':>13} {'± SD':>10} {'SNR (dB)':>10} {'MSE':>14}"
    print(header)
    print("-" * len(header))

    for name, runner in methods:
        rec, t_mean, t_sd, err = _avg_time(runner, repeats=repeats, warmup=True)
        if err is not None or rec.size == 0:
            print(f"{name:<34} {'unavailable':>13} {'':>10} {'—':>10} {'—':>14}")
            rows.append({"name": name, "time": np.nan, "sd": np.nan, "snr": np.nan, "mse": np.nan, "rec": None, "err": err})
            continue

        # Metrics on central crop
        rec_roi = _central_crop(rec, border_frac)
        snr = _snr_db(roi, rec_roi)
        mse = float(np.mean((roi - rec_roi) ** 2, dtype=np.float64))
        print(f"{name:<34} {_fmt_time(t_mean):>13} {_fmt_time(t_sd):>10} {snr:>10.2f} {mse:>14.3e}")

        rows.append({"name": name, "time": t_mean, "sd": t_sd, "snr": snr, "mse": mse, "rec": rec, "err": None})
        # ROI tile for montage
        tile = _nearest_big(rec_roi, 240)
        roi_tiles.append((name, tile))

    # --- ROI montage ---
    if roi_tiles:
        cols = min(3, len(roi_tiles))
        rows_n = int(np.ceil(len(roi_tiles) / cols))
        fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 3.2, rows_n * 3.4))
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
        fig.suptitle(f"Recovered ROI (central {int((1-2*border_frac)*100)}% area) — zoom ×{z:g}", fontsize=12)
        plt.tight_layout()
        plt.show()

    # --- Timing bar chart ---
    valid = [r for r in rows if np.isfinite(r["time"])]
    if valid:
        names = [r["name"] for r in valid]
        times = np.array([r["time"] for r in valid])
        sds   = np.array([r["sd"]   for r in valid])
        order = np.argsort(times)
        names = [names[i] for i in order]
        times = times[order]; sds = sds[order]

        plt.figure(figsize=(10, 5))
        y = np.arange(len(names))
        plt.barh(y, times, xerr=sds, color="tab:blue", alpha=0.8)
        plt.yticks(y, names, fontsize=9)
        plt.xlabel(f"Round-trip time (s) — mean ± sd over {repeats} runs")
        plt.title(f"Timing vs Method (H×W = {H}×{W}, zoom ×{z:g})")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()

    # --- SNR bar chart ---
    valid = [r for r in rows if np.isfinite(r["snr"])]
    if valid:
        names = [r["name"] for r in valid]
        snrs  = np.array([r["snr"] for r in valid])
        order = np.argsort(-snrs)  # higher is better
        names = [names[i] for i in order]
        snrs = snrs[order]

        plt.figure(figsize=(10, 5))
        x = np.arange(len(names))
        plt.bar(x, snrs, color="tab:green", alpha=0.85)
        plt.xticks(x, names, rotation=30, ha="right", fontsize=9)
        plt.ylabel("SNR (dB) on central crop")
        plt.title(f"SNR vs Method (H×W = {H}×{W}, zoom ×{z:g})")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
