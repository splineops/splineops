# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_07_compare_different_implementations.py
# sphinx_gallery_end_ignore

"""
Compare Python vs C++ Implementations
=====================================

Measure the performance of Least-Squares (best AA) and Oblique (fast AA)
using the **pure-Python** fallback versus the **C++-accelerated** path.

Notes
-----
- We pin OpenMP threads for reproducible wall-clock timings in the gallery.
- To toggle modes we set ``SPLINEOPS_ACCEL`` and **reload** the module
  ``splineops.resize.resize`` each time so it re-reads the policy.
"""

# %%
# Imports & Setup
# ---------------
import os
import sys
import time
import importlib
import importlib.util as _util

import numpy as np
import matplotlib.pyplot as plt

# Prefer C++ when present; keep timings stable in the gallery
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("SPLINEOPS_ACCEL", "auto")


def _load_resize_module(*, force_reload: bool = False):
    """
    Load/reload the submodule so internal policy (SPLINEOPS_ACCEL) is re-read.
    Avoids name-shadowing and ensures the module state reflects the env var.
    """
    name = "splineops.resize.resize"
    if force_reload and name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def _has_cpp() -> bool:
    return _util.find_spec("splineops._lsresize") is not None


print("[splineops] C++ acceleration available:", _has_cpp())
print("[splineops] OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS", "<unset>"))

# %%
# Helper: time a single resize call
# ---------------------------------
def time_one(
    mode: str,
    arr: np.ndarray,
    zoom: tuple[float, float],
    method: str,
    repeats: int = 3,
) -> float:
    """
    Parameters
    ----------
    mode : {'always','never'}
        'always' = force C++ path; 'never' = force Python fallback.
    arr : ndarray
        Input image (H×W).
    zoom : (zh, zw)
        Per-axis scale factors.
    method : str
        'cubic-best_antialiasing' (LS) or 'cubic-fast_antialiasing' (Oblique).
    repeats : int
        Number of timed runs; we return best-of-N.

    Returns
    -------
    float
        Best runtime (seconds).
    """
    os.environ["SPLINEOPS_ACCEL"] = mode
    rz = _load_resize_module(force_reload=True)
    # warmup (import caches, kernel pages, etc.)
    rz.resize(arr, zoom_factors=zoom, method=method)

    times: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        rz.resize(arr, zoom_factors=zoom, method=method)
        times.append(time.perf_counter() - t0)
    return min(times)


# %%
# Benchmark scenarios
# -------------------
rng = np.random.default_rng(0)
cases = [
    ("Downsample 512×512 → ×0.5",   (512,  512),  (0.5, 0.5)),
    ("Downsample 1024×1024 → ×0.5", (1024, 1024), (0.5, 0.5)),
    ("Upsample   512×512 → ×1.7",   (512,  512),  (1.7, 1.7)),
    ("Upsample  1024×1024 → ×1.7",  (1024, 1024), (1.7, 1.7)),
]
methods = [
    ("Least-Squares (best AA)", "cubic-best_antialiasing"),
    ("Oblique (fast AA)",       "cubic-fast_antialiasing"),
]

results = []  # (label, method_label, t_cpp, t_py, speedup, max_abs_diff)

for label, shape, zoom in cases:
    arr = rng.random(shape, dtype=np.float64)
    for meth_label, preset in methods:
        # timings
        t_cpp = time_one("always", arr, zoom, preset)
        t_py  = time_one("never",  arr, zoom, preset)
        speed = (t_py / t_cpp) if t_cpp > 0 else np.inf

        # numeric sanity (max abs diff)
        os.environ["SPLINEOPS_ACCEL"] = "always"
        rz = _load_resize_module(force_reload=True)
        y_cpp = rz.resize(arr, zoom_factors=zoom, method=preset)

        os.environ["SPLINEOPS_ACCEL"] = "never"
        rz = _load_resize_module(force_reload=True)
        y_py = rz.resize(arr, zoom_factors=zoom, method=preset)

        maxdiff = float(np.max(np.abs(y_cpp - y_py)))

        print(
            f"{label:28s} | {meth_label:24s} | "
            f"C++ {t_cpp:.4f}s | Py {t_py:.4f}s | ×{speed:.1f} | max|Δ|={maxdiff:.2e}"
        )
        results.append((label, meth_label, t_cpp, t_py, speed, maxdiff))

# %%
# Plot: speedup summary
# ---------------------
labels = [f"{L}\n{M}" for (L, M, _, _, _, _) in results]
speedups = [s for (_, _, _, _, s, _) in results]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(speedups)), speedups)
ax.set_xticks(range(len(speedups)), labels, rotation=35, ha="right")
ax.set_ylabel("Speedup (Python time / C++ time)")
ax.set_title("C++ vs Python – LS/Oblique (best of 3)")
for i, s in enumerate(speedups):
    ax.text(i, bars[i].get_height(), f"×{s:.1f}", ha="center", va="bottom", fontsize=9)
fig.tight_layout()
plt.show()
