# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_07_benchmarking.py
# sphinx_gallery_end_ignore

"""
Batch Benchmark of 2D Downsampling Methods
==========================================

This example benchmarks several 2D downsampling methods over a *set* of test
images. For each image we:

1. Downsample by a fixed zoom factor
2. Measure the runtime of the **forward pass** only

We compare:

- SciPy cubic interpolation (:func:`scipy.ndimage.zoom`)
- Standard cubic interpolation (:func:`splineops.resize.resize`, ``method="cubic"``)
- Least-Squares cubic anti-aliasing (``"cubic-best_antialiasing"``)
- Oblique cubic fast anti-aliasing (``"cubic-fast_antialiasing"``)

The goal is to inspect both the speed and the visual appearance of the
downsampled results.
"""

# %%
# Imports
# -------

from __future__ import annotations

import time
from io import BytesIO
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import requests
from PIL import Image

from scipy.ndimage import zoom as _scipy_zoom

from splineops.resize import resize
from splineops.utils.specs import print_runtime_context


# %%
# Test Images
# -----------
#
# We use a small subset of the Kodak image set. All are downloaded as RGB,
# converted to grayscale, and normalized to [0, 1].
#
# We start with ``kodim05.png`` as requested.

KODAK_BASE = "https://r0k.us/graphics/kodak/kodak"
KODAK_IMAGES = [
    ("kodim05", f"{KODAK_BASE}/kodim05.png"),
    ("kodim07", f"{KODAK_BASE}/kodim07.png"),
    ("kodim14", f"{KODAK_BASE}/kodim14.png"),
    ("kodim23", f"{KODAK_BASE}/kodim23.png"),
]


def _load_kodak_gray(url: str) -> np.ndarray:
    """
    Download a Kodak image, convert to grayscale [0, 1] float64.

    Returns
    -------
    img_gray : ndarray, shape (H, W)
    """
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    img = Image.open(BytesIO(r.content))
    arr = np.asarray(img, dtype=np.float64)

    # RGB → grayscale; if already single-channel, assume 8-bit-ish
    if arr.ndim == 3 and arr.shape[2] >= 3:
        arr01 = arr / 255.0
        gray = (
            0.2989 * arr01[..., 0] +
            0.5870 * arr01[..., 1] +
            0.1140 * arr01[..., 2]
        )
    else:
        # Fallback: scale to [0, 1] by max value
        vmax = float(arr.max()) or 1.0
        gray = arr / vmax

    return np.clip(gray, 0.0, 1.0)


# %%
# Benchmark Configuration
# -----------------------
#
# We use a single down-sampling factor. The value 0.25 matches the earlier
# comparison example :mod:`03_06_compare_different_methods`.

ZOOM = 0.134
ZOOM_FACTORS_2D = (ZOOM, ZOOM)

# Number of timing runs per (image, method). One warm-up run is not counted.
N_TRIALS = 7

# Methods to benchmark:
# - label shown in the table/plot
# - kind: "scipy" or "splineops"
# - splineops `method` string when kind == "splineops"
BENCH_METHODS: List[Tuple[str, str, str | None]] = [
    ("SciPy cubic",              "scipy",     None),
    ("Standard cubic",           "splineops", "cubic"),
    ("Least-Squares (AA cubic)", "splineops", "cubic-best_antialiasing"),
    ("Oblique (fast AA cubic)",  "splineops", "cubic-fast_antialiasing"),
]


# %%
# Core timing helper
# ------------------
#
# For each method we only time the **forward** (downsampling) pass. We still
# keep the downsampled image so we can display it later.

def _run_once_forward(img: np.ndarray, *, kind: str, method: str | None) -> Tuple[np.ndarray, float]:
    """
    One forward run, returning (downsampled_image, elapsed_sec).
    """
    zoom_factors = ZOOM_FACTORS_2D

    if kind == "scipy":
        # SciPy baseline using ndimage.zoom with cubic interpolation
        t0 = time.perf_counter()
        down = _scipy_zoom(img, zoom_factors, order=3, mode="reflect", prefilter=True)
        elapsed = time.perf_counter() - t0
    else:
        # splineops.resize path
        assert method is not None
        t0 = time.perf_counter()
        down = resize(img, zoom_factors=zoom_factors, method=method)
        elapsed = time.perf_counter() - t0

    return np.asarray(down, dtype=np.float64), elapsed


def run_with_repeats(
    img: np.ndarray,
    *,
    kind: str,
    method: str | None,
    trials: int = N_TRIALS,
    warmup: int = 1,
) -> Tuple[np.ndarray, float, float]:
    """
    Run a given method multiple times on `img`.

    Returns
    -------
    downsampled : ndarray
        Downsampled image from the last run (deterministic).
    time_mean : float
        Mean forward runtime over all trials (seconds).
    time_sd : float
        Sample standard deviation of forward runtime (seconds).
    """
    # warm-up (not counted)
    for _ in range(warmup):
        _run_once_forward(img, kind=kind, method=method)

    # first measured run – keep result + time
    downsampled, t = _run_once_forward(img, kind=kind, method=method)
    times = [t]

    # additional runs – time only
    for _ in range(trials - 1):
        _, t = _run_once_forward(img, kind=kind, method=method)
        times.append(t)

    times = np.asarray(times, dtype=np.float64)
    time_mean = float(times.mean())
    time_sd   = float(times.std(ddof=1)) if times.size > 1 else 0.0
    return downsampled, time_mean, time_sd


# %%
# Run Benchmark
# -------------
#
# For each image and each method, we run the benchmark and store:
#
# - Average runtime and standard deviation
# - One downsampled image (for visual inspection)

results: List[Dict[str, object]] = []
orig_images: Dict[str, np.ndarray] = {}

for name, url in KODAK_IMAGES:
    img = _load_kodak_gray(url)
    h, w = img.shape
    orig_images[name] = img
    print(f"Loaded {name}  shape={h}×{w}")

    for label, kind, method in BENCH_METHODS:
        down, t_mean, t_sd = run_with_repeats(img, kind=kind, method=method)
        results.append(
            dict(
                image=name,
                shape=(h, w),
                method_label=label,
                kind=kind,
                downsampled=down,
                t_mean=t_mean,
                t_sd=t_sd,
            )
        )

print("\n=== Timing summary over all images (forward pass only) ===\n")

# Print a simple text table (grouped by image)
for name, url in KODAK_IMAGES:
    print(f"Image: {name}")
    print(f"  URL: {url}")
    rows = [r for r in results if r["image"] == name]
    header = f"{'Method':<28} {'Time (s, avg±sd)':>20}"
    print("  " + header)
    print("  " + "-" * len(header))
    for r in rows:
        label   = str(r["method_label"])
        t_mean  = float(r["t_mean"])
        t_sd    = float(r["t_sd"])
        time_str = f"{t_mean:.4f} ± {t_sd:.4f}"
        print(f"  {label:<28} {time_str:>20}")
    print()

print(f"Timings averaged over {N_TRIALS} runs (1 warm-up run not counted).\n")
print_runtime_context()


# %%
# Visual Comparison for All Images
# --------------------------------
#
# For each image we display:
#
# - Original
# - Downsampled by each method (with timing in the title)

for img_name, _ in KODAK_IMAGES:
    img_orig = orig_images[img_name]

    # Collect downsampled results + timings for this image
    rows_this = [r for r in results if r["image"] == img_name]
    down_by_label = {r["method_label"]: r["downsampled"] for r in rows_this}
    time_by_label = {
        r["method_label"]: (float(r["t_mean"]), float(r["t_sd"]))
        for r in rows_this
    }
    labels_order = [m[0] for m in BENCH_METHODS]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(10.5, 7.0),
        constrained_layout=True,
    )
    axes = axes.ravel()

    # Original in the first slot
    axes[0].imshow(img_orig, cmap="gray", interpolation="nearest", aspect="equal")
    h0, w0 = img_orig.shape
    axes[0].set_title(f"{img_name}: original\nshape={h0}×{w0}")
    axes[0].axis("off")

    # Downsampled variants in the remaining slots
    for ax, label in zip(axes[1:], labels_order):
        down = down_by_label[label]
        h_d, w_d = down.shape
        t_mean, t_sd = time_by_label[label]
        ax.imshow(down, cmap="gray", interpolation="nearest", aspect="equal")
        ax.set_title(
            f"{label}\n"
            f"shape={h_d}×{w_d}\n"
            f"{t_mean*1000:.1f} ms ± {t_sd*1000:.1f} ms",
            fontsize=9,
        )
        ax.axis("off")

    # If any subplot is left unused (2×3 grid), blank it
    for ax in axes[1 + len(labels_order):]:
        ax.axis("off")

    plt.show()