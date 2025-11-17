# sphinx_gallery_start_ignore
# splineops/scripts/script_compare_resize_python_cpp.py
# sphinx_gallery_end_ignore

"""
Compare Python vs C++ Implementations (synthetic images)
=======================================================

Measure the performance of **Least-Squares (best AA)** and **Oblique (fast AA)**
using the **pure-Python** fallback versus the **C++-accelerated** path on two
synthetic test images.

This version is tuned for **maximum throughput** by default:
- OpenMP (C++ path) uses **all available CPU cores**.
- BLAS stacks are capped to 1 thread to avoid oversubscription.
- We reload the resize module on every timing to honor SPLINEOPS_ACCEL policy.
"""

from __future__ import annotations

import os
import sys
import time
import importlib
import importlib.util as _util

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
# Max-throughput environment                                         #
# ------------------------------------------------------------------ #
# Let OpenMP use all cores for the C++ path:
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
# Keep BLAS stacks single-threaded to avoid oversubscription with OpenMP:
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Fix thread count instead of allowing OpenMP to shrink/expand:
os.environ.setdefault("OMP_DYNAMIC", "false")
# Prefer binding threads near each other for cache locality:
os.environ.setdefault("OMP_PROC_BIND", "close")
# Default policy (we’ll override per-call in timing):
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
print(f"[splineops] OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS','<unset>')}")
print(f"[splineops] OMP_DYNAMIC={os.environ.get('OMP_DYNAMIC','<unset>')}")
print(f"[splineops] OMP_PROC_BIND={os.environ.get('OMP_PROC_BIND','<unset>')}\n")


# ------------------------------------------------------------------ #
# Timing helpers                                                     #
# ------------------------------------------------------------------ #
def _time_resize(mode: str, img: np.ndarray, zoom: tuple[float, float], preset: str, repeats: int = 5):
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
# Synthetic images                                                   #
# ------------------------------------------------------------------ #
rng = np.random.default_rng(0)

DTYPES = (np.float32, np.float64)

images: list[tuple[str, np.ndarray]] = []
for dtype in DTYPES:
    suffix = "f32" if dtype == np.float32 else "f64"
    images.append((f"synthetic_1024x1024_{suffix}", rng.random((1024, 1024), dtype=dtype)))
    images.append((f"synthetic_640x480_{suffix}",   rng.random((480, 640),  dtype=dtype)))

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

for img_name, img in images:
    print(f"\n=== {img_name}  shape={img.shape}, dtype={img.dtype} ===")
    for meth_label, preset in methods:
        for zoom_label, zoom in zooms:
            # C++
            t_cpp, y_cpp = _time_resize("always", img, zoom, preset, repeats=5)
            # Python
            t_py, y_py = _time_resize("never",  img, zoom, preset, repeats=5)

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
        ax.set_title("C++ vs Python – LS/Oblique (best-of-5)")
        for i, s in enumerate(speedups):
            ax.text(i, bars[i].get_height(), f"×{s:.1f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        plt.show()
    else:
        print("\n[info] No C++ timings to plot (native extension unavailable).")
