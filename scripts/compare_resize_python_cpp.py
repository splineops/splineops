# sphinx_gallery_start_ignore
# splineops/scripts/compare_resize_python_cpp.py
# sphinx_gallery_end_ignore

"""
Compare Python vs C++ Implementations (image-based)
===================================================

Measure the performance of **Least-Squares (best AA)** and **Oblique (fast AA)**
using the **pure-Python** fallback versus the **C++-accelerated** path on one
or more input images.

Usage
-----
    python compare_python_cpp_resize.py [img1 ... imgN]

If no images are given and a native file dialog is not possible, the script
falls back to synthetic images.

Notes
-----
- We pin OpenMP/BLAS threads for reproducible wall-clock timings.
- We toggle ``SPLINEOPS_ACCEL`` and **reload** ``splineops.resize.resize`` so
  the implementation re-reads the policy for each timing.
"""

from __future__ import annotations

import os
import sys
import time
import importlib
import importlib.util as _util
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ------------------------------------------------------------------ #
# Environment for stable timings                                     #
# ------------------------------------------------------------------ #
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("SPLINEOPS_ACCEL", "auto")


def _has_cpp() -> bool:
    """Is the native module importable in this environment?"""
    return _util.find_spec("splineops._lsresize") is not None


def _load_resize_module(*, force_reload: bool = False):
    """Load/reload the resize implementation so it re-reads SPLINEOPS_ACCEL."""
    name = "splineops.resize.resize"
    if force_reload and name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


HAS_CPP = _has_cpp()
print(f"[splineops] C++ acceleration available: {HAS_CPP}")
print(f"[splineops] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','<unset>')}\n")


# ------------------------------------------------------------------ #
# Image loading & grayscale conversion                               #
# ------------------------------------------------------------------ #
def to_gray01_from_any(a: np.ndarray) -> np.ndarray:
    """
    Convert H×W or H×W×C array to grayscale float64 in [0, 1].
    Keeps intensity if already H×W; converts integer types via their full range.
    """
    a = np.asarray(a)
    orig_dtype = a.dtype

    if a.ndim == 3:
        if a.shape[2] == 1:
            a = a[..., 0]
        else:
            r = a[..., 0].astype(np.float64)
            g = a[..., 1].astype(np.float64)
            b = a[..., 2].astype(np.float64)
            if np.issubdtype(orig_dtype, np.integer):
                maxv = np.iinfo(orig_dtype).max
                r, g, b = r / maxv, g / maxv, b / maxv
            else:
                lo = min(r.min(), g.min(), b.min())
                hi = max(r.max(), g.max(), b.max())
                if hi > 1.0 or lo < 0.0:
                    r = (r - lo) / (hi - lo + 1e-12)
                    g = (g - lo) / (hi - lo + 1e-12)
                    b = (b - lo) / (hi - lo + 1e-12)
            a = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        a = a.astype(np.float64)
        if np.issubdtype(orig_dtype, np.integer):
            a /= np.iinfo(orig_dtype).max
        else:
            amin, amax = a.min(), a.max()
            if amax > 1.0 or amin < 0.0:
                a = (a - amin) / (amax - amin + 1e-12)

    return np.clip(a.astype(np.float64), 0.0, 1.0)


def _missing_or_not_file(p: Optional[Path]) -> bool:
    if p is None:
        return True
    s = str(p).strip()
    if not s:
        return True
    return not p.exists() or not p.is_file()


def _try_open_file_dialog(multiple: bool = True) -> list[Path]:
    """Open native file dialog; return selected files or [] if cancelled/unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        if multiple:
            paths = filedialog.askopenfilenames(
                title="Select image(s)",
                filetypes=[
                    ("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                    ("All files", "*.*"),
                ],
            )
        else:
            single = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[
                    ("Images", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
                    ("All files", "*.*"),
                ],
            )
            paths = (single,) if single else ()
        root.destroy()
        return [Path(p).expanduser() for p in paths]
    except Exception:
        return []


def _load_image_as_gray(path: Path) -> np.ndarray:
    im = Image.open(str(path))
    if getattr(im, "is_animated", False):
        try:
            im.seek(0)  # first frame if multi-page
        except Exception:
            pass
    arr = np.asarray(im)
    im.close()
    return to_gray01_from_any(arr)


# ------------------------------------------------------------------ #
# Timing helpers                                                     #
# ------------------------------------------------------------------ #
def _time_resize(mode: str, img: np.ndarray, zoom: tuple[float, float], preset: str, repeats: int = 2):
    """
    Return (best_time_sec, output_array) for one policy ('always' C++, 'never' Python).
    Reloads resize module so it re-reads SPLINEOPS_ACCEL.
    """
    if mode == "always" and not HAS_CPP:
        return float("nan"), None
    os.environ["SPLINEOPS_ACCEL"] = mode
    rz = _load_resize_module(force_reload=True)
    # warmup
    out = rz.resize(img, zoom_factors=zoom, method=preset)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out_tmp = rz.resize(img, zoom_factors=zoom, method=preset)
        best = min(best, time.perf_counter() - t0)
        out = out_tmp
    return best, out


# ------------------------------------------------------------------ #
# Main: collect images or synthesize                                 #
# ------------------------------------------------------------------ #
cli_paths = [Path(p).expanduser() for p in sys.argv[1:] if not p.startswith("-")]
paths: list[Path] = [p for p in cli_paths if p.exists() and p.is_file()]

if not paths:
    picked = _try_open_file_dialog(multiple=True)
    paths.extend(picked)

# If still empty, synthesize a couple of images
SYNTHETIC = False
if not paths:
    SYNTHETIC = True
    print("[info] No images selected; using synthetic test arrays.")
    rng = np.random.default_rng(0)
    # Two synthetic images with simple texture
    paths = []
    synth_bank = [
        ("synthetic_512x512", rng.random((512, 512), dtype=np.float64)),
        ("synthetic_1024x1024", rng.random((1024, 1024), dtype=np.float64)),
    ]

# Presets & zoom scenarios
methods = [
    ("Least-Squares (best AA)", "cubic-best_antialiasing"),
    ("Oblique (fast AA)",       "cubic-fast_antialiasing"),
]
zooms = [
    ("↓0.5×", (0.5, 0.5)),
    ("↑1.7×", (1.7, 1.7)),
]

# ---------------------------------------------------------------------------- #
# Run comparisons                                                              #
# ---------------------------------------------------------------------------- #
all_rows = []  # (img_name, method_label, zoom_label, t_cpp, t_py, speedup, max_abs_diff)
for entry in paths:
    if SYNTHETIC:
        # Already have (name, array) in synth_bank; entry is a fake path
        continue

# Build per-image arrays (either real images or synthetic)
image_items: list[tuple[str, np.ndarray]] = []
if SYNTHETIC:
    image_items.extend(synth_bank)
else:
    for p in paths:
        try:
            img = _load_image_as_gray(p)
        except Exception as e:
            print(f"[warn] Skipping {p} (load error: {e})")
            continue
        image_items.append((p.name, img))

for img_name, img in image_items:
    print(f"\n=== {img_name}  shape={img.shape} ===")
    for meth_label, preset in methods:
        for zoom_label, zoom in zooms:
            # C++
            t_cpp, y_cpp = _time_resize("always", img, zoom, preset, repeats=2)
            # Python
            t_py, y_py = _time_resize("never",  img, zoom, preset, repeats=2)

            # numeric sanity (if C++ ran)
            if HAS_CPP and y_cpp is not None:
                maxdiff = float(np.max(np.abs(y_cpp - y_py)))
            else:
                maxdiff = float("nan")

            # speedup
            speed = (t_py / t_cpp) if (HAS_CPP and t_cpp > 0) else float("nan")

            cxx_str = "n/a" if not HAS_CPP else f"{t_cpp*1000:7.1f} ms"
            py_str  = f"{t_py*1000:7.1f} ms"
            sp_str  = "n/a" if not HAS_CPP else f"×{speed:4.1f}"
            diff_str = "n/a" if not HAS_CPP else f"{maxdiff:.2e}"

            print(f"  {meth_label:24s} {zoom_label:>5s}  C++ {cxx_str}  Py {py_str}  {sp_str}  max|Δ|={diff_str}")

            all_rows.append((img_name, meth_label, zoom_label, t_cpp, t_py, speed, maxdiff))


# ------------------------------------------------------------------ #
# Optional: plot speedup summary                                     #
# ------------------------------------------------------------------ #
SHOW_PLOT = os.environ.get("SPLINEOPS_SCRIPTS_PLOT", "1") != "0"
if SHOW_PLOT and len(all_rows) > 0:
    # Build labels & speedups; skip NaNs if C++ not available
    labels = []
    speedups = []
    for (img_name, meth_label, zoom_label, t_cpp, t_py, speed, _) in all_rows:
        if not HAS_CPP or not np.isfinite(speed):
            continue
        labels.append(f"{img_name}\n{meth_label}\n{zoom_label}")
        speedups.append(speed)

    if speedups:
        fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(speedups)), 5))
        x = np.arange(len(speedups))
        bars = ax.bar(x, speedups)
        ax.set_xticks(x, labels, rotation=35, ha="right")
        ax.set_ylabel("Speedup (Python time / C++ time)")
        ax.set_title("C++ vs Python – LS/Oblique (best-of-2)")
        for i, s in enumerate(speedups):
            ax.text(i, bars[i].get_height(), f"×{s:.1f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        plt.show()
    else:
        print("\n[info] No C++ timings to plot (native extension unavailable).")
