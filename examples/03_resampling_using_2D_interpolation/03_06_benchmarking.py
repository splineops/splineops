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
3. Visualise the results with ROI-aware zooms, one method per figure

We compare:

- SciPy cubic interpolation (:func:`scipy.ndimage.zoom`)
- Standard cubic interpolation (:func:`splineops.resize.resize`, ``method="cubic"``)
- Least-Squares cubic anti-aliasing (``"cubic-best_antialiasing"``)
- Oblique cubic fast anti-aliasing (``"cubic-fast_antialiasing"``)

The goal is to inspect both the speed and the visual appearance of the
downsampled results, especially on a small detail region (ROI).
"""

# %%
# Imports
# -------

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from PIL import Image

from scipy.ndimage import zoom as _scipy_zoom

from splineops.resize import resize
from splineops.utils.specs import print_runtime_context
from splineops.utils.plotting import show_roi_zoom


def fmt_ms(seconds: float) -> str:
    """Format seconds as a short 'X.X ms' string."""
    return f"{seconds * 1000.0:.1f} ms"


# %%
# Test Images
# -----------
#
# We use a small subset of the Kodak image set. All are downloaded as RGB,
# converted to grayscale, and normalized to [0, 1].
#
# We start with ``kodim05.png``.

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
    with urlopen(url, timeout=10) as resp:
        img = Image.open(resp)
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
# We use a single down-sampling factor (same for all images and methods).

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

def _run_once_forward(
    img: np.ndarray, *, kind: str, method: str | None
) -> Tuple[np.ndarray, float]:
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
# ROI helpers
# -----------
#
# We will focus on a small square ROI (centered by default) to inspect
# aliasing in detail.

ROI_SIZE_PX = 64  # side length in original image pixels


def _nearest_big(roi: np.ndarray, target_h: int = 256) -> np.ndarray:
    """
    Enlarge a small ROI with nearest-neighbour so that its height is ~target_h.
    """
    h, w = roi.shape
    mag = max(1, int(round(target_h / h)))
    return np.repeat(np.repeat(roi, mag, axis=0), mag, axis=1)


def _build_canvas_and_roi(
    down: np.ndarray,
    *,
    h_img: int,
    w_img: int,
    center_r: int,
    center_c: int,
    roi_size_px: int,
    zoom_factors: Tuple[float, float],
) -> Tuple[np.ndarray, dict, np.ndarray]:
    """
    Place the downsampled image on a white canvas of original size and compute
    matching ROI parameters and a small ROI patch in downsampled space.

    Returns
    -------
    canvas : ndarray, shape (h_img, w_img)
        White canvas with the downsampled image pasted at (0, 0).
    roi_kwargs_on_canvas : dict
        kwargs to pass into `show_roi_zoom` to produce the ROI inset.
    roi_patch : ndarray
        Small ROI patch cropped from the downsampled image.
    """
    zoom_r, zoom_c = zoom_factors
    h_res, w_res = down.shape

    # ROI size in the downsampled image
    roi_h_res = max(1, int(round(roi_size_px * zoom_r)))
    roi_w_res = max(1, int(round(roi_size_px * zoom_c)))

    # Same *relative* center as in original
    center_r_res = int(round(center_r * zoom_r))
    center_c_res = int(round(center_c * zoom_c))

    # Top-left of the ROI in downsampled coords, clipped to bounds
    row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res - roi_h_res))
    col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res - roi_w_res))

    # Build original-size white canvas and paste downsampled at (0, 0)
    canvas = np.ones((h_img, w_img), dtype=down.dtype)
    canvas[:h_res, :w_res] = down

    roi_kwargs_on_canvas = dict(
        roi_height_frac=roi_h_res / h_img,
        grayscale=True,
        roi_xy=(row_top_res, col_left_res),
    )

    roi_patch = down[
        row_top_res : row_top_res + roi_h_res,
        col_left_res : col_left_res + roi_w_res,
    ]

    return canvas, roi_kwargs_on_canvas, roi_patch


# %%
# Visual Comparison for Each Image
# --------------------------------
#
# For each Kodak image we show:
#
# 1. Original image with an ROI.
# 2. SciPy downsampled image on an original-size canvas + ROI inset.
# 3. Standard cubic downsampled + ROI inset.
# 4. Least-Squares (AA cubic) downsampled + ROI inset.
# 5. Oblique (fast AA cubic) downsampled + ROI inset.
# 6. A 4-way ROI comparison (SciPy vs Standard vs LS vs Oblique), enlarged.

for img_name, _ in KODAK_IMAGES:
    img_orig = orig_images[img_name]
    h_img, w_img = img_orig.shape

    # Centered ROI in the original image
    center_r = h_img // 2
    center_c = w_img // 2
    row_top = int(np.clip(center_r - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
    col_left = int(np.clip(center_c - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))
    roi_rect = (row_top, col_left, ROI_SIZE_PX, ROI_SIZE_PX)

    roi_kwargs_orig = dict(
        roi_height_frac=ROI_SIZE_PX / h_img,
        grayscale=True,
        roi_xy=(row_top, col_left),
    )

    # Collect downsampled results + timings for this image
    rows_this = [r for r in results if r["image"] == img_name]
    down_by_label = {r["method_label"]: r["downsampled"] for r in rows_this}
    time_by_label = {
        r["method_label"]: (float(r["t_mean"]), float(r["t_sd"]))
        for r in rows_this
    }

    print(f"\n=== Visual inspection for {img_name} ===\n")

    # 1) Original image with ROI
    _ = show_roi_zoom(
        img_orig,
        ax_titles=(f"{img_name}: Original (with ROI)", None),
        **roi_kwargs_orig,
    )

    # Common method labels in the order we want to show them
    ordered_methods = [
        ("SciPy cubic",              "SciPy cubic"),
        ("Standard cubic",           "Standard cubic"),
        ("Least-Squares (AA cubic)", "Least-Squares (AA)"),
        ("Oblique (fast AA cubic)",  "Oblique (fast AA)"),
    ]

    # Store ROI patches for the final side-by-side comparison
    roi_patches = []
    roi_titles  = []

    for label, short_name in ordered_methods:
        down = down_by_label[label]
        t_mean, t_sd = time_by_label[label]

        canvas, roi_kwargs_canvas, roi_patch = _build_canvas_and_roi(
            down,
            h_img=h_img,
            w_img=w_img,
            center_r=center_r,
            center_c=center_c,
            roi_size_px=ROI_SIZE_PX,
            zoom_factors=ZOOM_FACTORS_2D,
        )

        roi_patches.append(roi_patch)
        roi_titles.append(
            f"{short_name}\n{fmt_ms(t_mean)} ± {fmt_ms(t_sd)}"
        )

        _ = show_roi_zoom(
            canvas,
            ax_titles=(
                f"{img_name}: {short_name}\n"
                f"{fmt_ms(t_mean)} ± {fmt_ms(t_sd)}",
                None,
            ),
            **roi_kwargs_canvas,
        )

    # 6) Side-by-side ROI comparison (SciPy vs Standard vs LS vs Oblique)
    DISPLAY_H = 256
    roi_big_list = [_nearest_big(r, DISPLAY_H) for r in roi_patches]

    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8))
    for ax, im, title in zip(axes, roi_big_list, roi_titles):
        ax.imshow(im, cmap="gray", interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    fig.suptitle(f"{img_name}: Downsampled ROI comparison", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()
