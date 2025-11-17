# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_06_benchmarking.py
# sphinx_gallery_end_ignore

"""
Benchmarking
============

This example benchmarks several 2D downsampling methods over a *set* of test
images. For each image we:

1. Downsample by an image-specific zoom factor.
2. Measure the runtime of the forward pass only.
3. Visualise the results with ROI-aware zooms, one method per figure.

We compare:

- SciPy cubic interpolation.
- Standard cubic interpolation.
- Least-Squares cubic anti-aliasing.
- Oblique cubic fast anti-aliasing.
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

# Use float32 for storage / IO (resize still computes internally in float64).
DTYPE = np.float32

# %%
# Test Image Configuration
# ------------------------
#
# All images are downloaded as RGB, converted to grayscale, and normalized
# to [0, 1]. For each image we also define:
#
# - a specific zoom factor,
# - an ROI size (square),
# - an ROI center given as (row_frac, col_frac) in [0, 1].

KODAK_BASE = "https://r0k.us/graphics/kodak/kodak"
KODAK_IMAGES = [
    ("kodim05", f"{KODAK_BASE}/kodim05.png"),
    ("kodim07", f"{KODAK_BASE}/kodim07.png"),
    ("kodim14", f"{KODAK_BASE}/kodim14.png"),
    ("kodim15", f"{KODAK_BASE}/kodim15.png"),
    ("kodim19", f"{KODAK_BASE}/kodim19.png"),
    ("kodim23", f"{KODAK_BASE}/kodim23.png"),
]

IMAGE_CONFIG: Dict[str, Dict[str, object]] = {
    "kodim05": dict(
        zoom=0.15,
        roi_size_px=256,
        roi_center_frac=(0.75, 0.5),
    ),
    "kodim07": dict(
        zoom=0.15,
        roi_size_px=256,
        roi_center_frac=(0.40, 0.50),
    ),
    "kodim14": dict(
        zoom=0.3,
        roi_size_px=256,
        roi_center_frac=(0.75, 0.75),
    ),
    "kodim15": dict(
        zoom=0.3,
        roi_size_px=256,
        roi_center_frac=(0.30, 0.55),
    ),
    "kodim19": dict(
        zoom=0.2,
        roi_size_px=256,
        roi_center_frac=(0.65, 0.35),
    ),
    "kodim23": dict(
        zoom=0.15,
        roi_size_px=256,
        roi_center_frac=(0.40, 0.65),
    ),
}

def _load_kodak_gray(url: str) -> np.ndarray:
    """
    Download a Kodak image, convert to grayscale [0, 1] in DTYPE
    (float32 by default).

    Returns
    -------
    img_gray : ndarray, shape (H, W), dtype=DTYPE
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

    return np.clip(gray, 0.0, 1.0).astype(DTYPE)

# %%
# Benchmark Configuration
# -----------------------
#
# Number of timing runs per (image, method). One warm-up run is not counted.

N_TRIALS = 10

# Methods to benchmark:
# - label shown in tables/plots
# - kind: "scipy" or "splineops"
# - splineops `method` string when kind == "splineops"
BENCH_METHODS: List[Tuple[str, str, str | None]] = [
    ("SciPy",              "scipy",     None),
    ("Standard",           "splineops", "cubic"),
    ("Least-Squares", "splineops", "cubic-best_antialiasing"),
    ("Oblique",  "splineops", "cubic-fast_antialiasing"),
]


# %%
# Timing Helper
# -------------
#
# For each method we only time the **forward** (downsampling) pass. We still
# keep the downsampled image so we can display it later.

def _run_once_forward(
    img: np.ndarray,
    *,
    zoom_factors: Tuple[float, float],
    kind: str,
    method: str | None,
) -> Tuple[np.ndarray, float]:
    """
    One forward run, returning (downsampled_image, elapsed_sec).
    """
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

    return np.asarray(down, dtype=img.dtype), elapsed


def run_with_repeats(
    img: np.ndarray,
    *,
    zoom_factors: Tuple[float, float],
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
        _run_once_forward(img, zoom_factors=zoom_factors, kind=kind, method=method)

    # first measured run – keep result + time
    downsampled, t = _run_once_forward(img, zoom_factors=zoom_factors, kind=kind, method=method)
    times = [t]

    # additional runs – time only
    for _ in range(trials - 1):
        _, t = _run_once_forward(img, zoom_factors=zoom_factors, kind=kind, method=method)
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
    cfg = IMAGE_CONFIG[name]
    zoom = float(cfg["zoom"])
    zoom_factors_2d = (zoom, zoom)

    print(f"Loaded {name}  shape={h}×{w}  zoom={zoom}")

    for label, kind, method in BENCH_METHODS:
        down, t_mean, t_sd = run_with_repeats(
            img,
            zoom_factors=zoom_factors_2d,
            kind=kind,
            method=method,
        )
        results.append(
            dict(
                image=name,
                shape=(h, w),
                zoom=zoom,
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
    cfg = IMAGE_CONFIG[name]
    zoom = float(cfg["zoom"])
    print(f"Image: {name}  (zoom={zoom})")
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
# ROI Helpers
# -----------
#
# We will focus on image-specific square ROIs to inspect aliasing in detail.

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

    # ROI size in the downsampled image: direct scaled footprint of the
    # original ROI. We keep it as-is (up to integer rounding).
    roi_h_res = max(1, int(round(roi_size_px * zoom_r)))
    roi_w_res = max(1, int(round(roi_size_px * zoom_c)))

    # Make the ROI square by taking the smaller side
    roi_side = int(max(1, min(roi_h_res, roi_w_res)))
    roi_h_res = roi_side
    roi_w_res = roi_side

    # Same *relative* center as in original
    center_r_res = int(round(center_r * zoom_r))
    center_c_res = int(round(center_c * zoom_c))

    # Top-left of the ROI in downsampled coords, clipped to bounds
    row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res - roi_h_res))
    col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res - roi_w_res))

    # Build original-size white canvas and paste downsampled at (0, 0)
    canvas = np.ones((h_img, w_img), dtype=down.dtype)
    canvas[:h_res, :w_res] = down

    # Tell show_roi_zoom to use exactly roi_h_res×roi_h_res on the canvas,
    # starting at (row_top_res, col_left_res).
    roi_kwargs_on_canvas = dict(
        roi_height_frac=roi_h_res / h_img,
        grayscale=True,
        roi_xy=(row_top_res, col_left_res),
    )

    # Crop the same region for the 4-way ROI comparison
    roi_patch = down[
        row_top_res : row_top_res + roi_h_res,
        col_left_res : col_left_res + roi_w_res,
    ]

    return canvas, roi_kwargs_on_canvas, roi_patch

# %%
# Image
# -----

img_name = "kodim05"
img_orig = orig_images[img_name]
h_img, w_img = img_orig.shape
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
zoom_factors_2d = (zoom, zoom)
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = cfg["roi_center_frac"]  # (row_frac, col_frac)
row_frac, col_frac = float(roi_center_frac[0]), float(roi_center_frac[1])

# Image-specific ROI center in absolute pixels
center_r = int(round(row_frac * h_img))
center_c = int(round(col_frac * w_img))
row_top = int(np.clip(center_r - roi_size_px // 2, 0, h_img - roi_size_px))
col_left = int(np.clip(center_c - roi_size_px // 2, 0, w_img - roi_size_px))
roi_rect = (row_top, col_left, roi_size_px, roi_size_px)

roi_kwargs_orig = dict(
    roi_height_frac=roi_size_px / h_img,
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

# %%
# Original with ROI
# ~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    img_orig,
    ax_titles=("Original with ROI", None),
    **roi_kwargs_orig,
)

# Prepare storage for ROI comparison
roi_patches = []
roi_titles  = []

# %%
# SciPy Interpolation
# ~~~~~~~~~~~~~~~~~~~

down_scipy = down_by_label["SciPy"]
t_mean_scipy, t_sd_scipy = time_by_label["SciPy"]

canvas_scipy, roi_kwargs_canvas_scipy, roi_patch_scipy = _build_canvas_and_roi(
    down_scipy,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_scipy)
roi_titles.append(
    f"SciPy\n{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}"
)

_ = show_roi_zoom(
    canvas_scipy,
    ax_titles=(
        f"SciPy (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}",
        None,
    ),
    **roi_kwargs_canvas_scipy,
)

# %%
# Standard Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~

down_std = down_by_label["Standard"]
t_mean_std, t_sd_std = time_by_label["Standard"]

canvas_std, roi_kwargs_canvas_std, roi_patch_std = _build_canvas_and_roi(
    down_std,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_std)
roi_titles.append(
    f"Standard\n{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}"
)

_ = show_roi_zoom(
    canvas_std,
    ax_titles=(
        f"Standard (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}",
        None,
    ),
    **roi_kwargs_canvas_std,
)

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~

down_ls = down_by_label["Least-Squares"]
t_mean_ls, t_sd_ls = time_by_label["Least-Squares"]

canvas_ls, roi_kwargs_canvas_ls, roi_patch_ls = _build_canvas_and_roi(
    down_ls,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ls)
roi_titles.append(
    f"Least-Squares\n{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}"
)

_ = show_roi_zoom(
    canvas_ls,
    ax_titles=(
        f"Least-Squares (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}",
        None,
    ),
    **roi_kwargs_canvas_ls,
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

down_ob = down_by_label["Oblique"]
t_mean_ob, t_sd_ob = time_by_label["Oblique"]

canvas_ob, roi_kwargs_canvas_ob, roi_patch_ob = _build_canvas_and_roi(
    down_ob,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ob)
roi_titles.append(
    f"Oblique\n{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}"
)

_ = show_roi_zoom(
    canvas_ob,
    ax_titles=(
        f"Oblique (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}",
        None,
    ),
    **roi_kwargs_canvas_ob,
)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

roi_big_list = [_nearest_big(r, 256) for r in roi_patches]

# Make the figure a bit taller so multi-line titles and the suptitle
# have enough vertical room, even when the ROIs are square.
fig_width = 12.5
fig_height = 5.0  # was 3.8; bump to 5.0 to avoid clipping

fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height))
for ax, im, title in zip(axes, roi_big_list, roi_titles):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig.suptitle("Downsampled ROI comparison", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])  # leave a touch more room for the suptitle
plt.show()

# %%
# Image
# -----

img_name = "kodim07"
img_orig = orig_images[img_name]
h_img, w_img = img_orig.shape
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
zoom_factors_2d = (zoom, zoom)
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = cfg["roi_center_frac"]  # (row_frac, col_frac)
row_frac, col_frac = float(roi_center_frac[0]), float(roi_center_frac[1])

# Image-specific ROI center in absolute pixels
center_r = int(round(row_frac * h_img))
center_c = int(round(col_frac * w_img))
row_top = int(np.clip(center_r - roi_size_px // 2, 0, h_img - roi_size_px))
col_left = int(np.clip(center_c - roi_size_px // 2, 0, w_img - roi_size_px))
roi_rect = (row_top, col_left, roi_size_px, roi_size_px)

roi_kwargs_orig = dict(
    roi_height_frac=roi_size_px / h_img,
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

# %%
# Original with ROI
# ~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    img_orig,
    ax_titles=("Original with ROI", None),
    **roi_kwargs_orig,
)

# Prepare storage for ROI comparison
roi_patches = []
roi_titles  = []

# %%
# SciPy Interpolation
# ~~~~~~~~~~~~~~~~~~~

down_scipy = down_by_label["SciPy"]
t_mean_scipy, t_sd_scipy = time_by_label["SciPy"]

canvas_scipy, roi_kwargs_canvas_scipy, roi_patch_scipy = _build_canvas_and_roi(
    down_scipy,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_scipy)
roi_titles.append(
    f"SciPy\n{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}"
)

_ = show_roi_zoom(
    canvas_scipy,
    ax_titles=(
        f"SciPy (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}",
        None,
    ),
    **roi_kwargs_canvas_scipy,
)

# %%
# Standard Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~

down_std = down_by_label["Standard"]
t_mean_std, t_sd_std = time_by_label["Standard"]

canvas_std, roi_kwargs_canvas_std, roi_patch_std = _build_canvas_and_roi(
    down_std,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_std)
roi_titles.append(
    f"Standard\n{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}"
)

_ = show_roi_zoom(
    canvas_std,
    ax_titles=(
        f"Standard (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}",
        None,
    ),
    **roi_kwargs_canvas_std,
)

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~

down_ls = down_by_label["Least-Squares"]
t_mean_ls, t_sd_ls = time_by_label["Least-Squares"]

canvas_ls, roi_kwargs_canvas_ls, roi_patch_ls = _build_canvas_and_roi(
    down_ls,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ls)
roi_titles.append(
    f"Least-Squares\n{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}"
)

_ = show_roi_zoom(
    canvas_ls,
    ax_titles=(
        f"Least-Squares (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}",
        None,
    ),
    **roi_kwargs_canvas_ls,
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

down_ob = down_by_label["Oblique"]
t_mean_ob, t_sd_ob = time_by_label["Oblique"]

canvas_ob, roi_kwargs_canvas_ob, roi_patch_ob = _build_canvas_and_roi(
    down_ob,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ob)
roi_titles.append(
    f"Oblique\n{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}"
)

_ = show_roi_zoom(
    canvas_ob,
    ax_titles=(
        f"Oblique (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}",
        None,
    ),
    **roi_kwargs_canvas_ob,
)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

roi_big_list = [_nearest_big(r, 256) for r in roi_patches]

# Make the figure a bit taller so multi-line titles and the suptitle
# have enough vertical room, even when the ROIs are square.
fig_width = 12.5
fig_height = 5.0  # was 3.8; bump to 5.0 to avoid clipping

fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height))
for ax, im, title in zip(axes, roi_big_list, roi_titles):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig.suptitle("Downsampled ROI comparison", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])  # leave a touch more room for the suptitle
plt.show()

# %%
# Image
# -----

img_name = "kodim14"
img_orig = orig_images[img_name]
h_img, w_img = img_orig.shape
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
zoom_factors_2d = (zoom, zoom)
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = cfg["roi_center_frac"]  # (row_frac, col_frac)
row_frac, col_frac = float(roi_center_frac[0]), float(roi_center_frac[1])

# Image-specific ROI center in absolute pixels
center_r = int(round(row_frac * h_img))
center_c = int(round(col_frac * w_img))
row_top = int(np.clip(center_r - roi_size_px // 2, 0, h_img - roi_size_px))
col_left = int(np.clip(center_c - roi_size_px // 2, 0, w_img - roi_size_px))
roi_rect = (row_top, col_left, roi_size_px, roi_size_px)

roi_kwargs_orig = dict(
    roi_height_frac=roi_size_px / h_img,
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

# %%
# Original with ROI
# ~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    img_orig,
    ax_titles=("Original with ROI", None),
    **roi_kwargs_orig,
)

# Prepare storage for ROI comparison
roi_patches = []
roi_titles  = []

# %%
# SciPy Interpolation
# ~~~~~~~~~~~~~~~~~~~

down_scipy = down_by_label["SciPy"]
t_mean_scipy, t_sd_scipy = time_by_label["SciPy"]

canvas_scipy, roi_kwargs_canvas_scipy, roi_patch_scipy = _build_canvas_and_roi(
    down_scipy,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_scipy)
roi_titles.append(
    f"SciPy\n{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}"
)

_ = show_roi_zoom(
    canvas_scipy,
    ax_titles=(
        f"SciPy (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}",
        None,
    ),
    **roi_kwargs_canvas_scipy,
)

# %%
# Standard Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~

down_std = down_by_label["Standard"]
t_mean_std, t_sd_std = time_by_label["Standard"]

canvas_std, roi_kwargs_canvas_std, roi_patch_std = _build_canvas_and_roi(
    down_std,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_std)
roi_titles.append(
    f"Standard\n{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}"
)

_ = show_roi_zoom(
    canvas_std,
    ax_titles=(
        f"Standard (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}",
        None,
    ),
    **roi_kwargs_canvas_std,
)

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~

down_ls = down_by_label["Least-Squares"]
t_mean_ls, t_sd_ls = time_by_label["Least-Squares"]

canvas_ls, roi_kwargs_canvas_ls, roi_patch_ls = _build_canvas_and_roi(
    down_ls,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ls)
roi_titles.append(
    f"Least-Squares\n{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}"
)

_ = show_roi_zoom(
    canvas_ls,
    ax_titles=(
        f"Least-Squares (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}",
        None,
    ),
    **roi_kwargs_canvas_ls,
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

down_ob = down_by_label["Oblique"]
t_mean_ob, t_sd_ob = time_by_label["Oblique"]

canvas_ob, roi_kwargs_canvas_ob, roi_patch_ob = _build_canvas_and_roi(
    down_ob,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ob)
roi_titles.append(
    f"Oblique\n{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}"
)

_ = show_roi_zoom(
    canvas_ob,
    ax_titles=(
        f"Oblique (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}",
        None,
    ),
    **roi_kwargs_canvas_ob,
)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

roi_big_list = [_nearest_big(r, 256) for r in roi_patches]

# Make the figure a bit taller so multi-line titles and the suptitle
# have enough vertical room, even when the ROIs are square.
fig_width = 12.5
fig_height = 5.0  # was 3.8; bump to 5.0 to avoid clipping

fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height))
for ax, im, title in zip(axes, roi_big_list, roi_titles):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig.suptitle("Downsampled ROI comparison", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])  # leave a touch more room for the suptitle
plt.show()

# %%
# Image
# -----

img_name = "kodim15"
img_orig = orig_images[img_name]
h_img, w_img = img_orig.shape
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
zoom_factors_2d = (zoom, zoom)
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = cfg["roi_center_frac"]  # (row_frac, col_frac)
row_frac, col_frac = float(roi_center_frac[0]), float(roi_center_frac[1])

# Image-specific ROI center in absolute pixels
center_r = int(round(row_frac * h_img))
center_c = int(round(col_frac * w_img))
row_top = int(np.clip(center_r - roi_size_px // 2, 0, h_img - roi_size_px))
col_left = int(np.clip(center_c - roi_size_px // 2, 0, w_img - roi_size_px))
roi_rect = (row_top, col_left, roi_size_px, roi_size_px)

roi_kwargs_orig = dict(
    roi_height_frac=roi_size_px / h_img,
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

# %%
# Original with ROI
# ~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    img_orig,
    ax_titles=("Original with ROI", None),
    **roi_kwargs_orig,
)

# Prepare storage for ROI comparison
roi_patches = []
roi_titles  = []

# %%
# SciPy Interpolation
# ~~~~~~~~~~~~~~~~~~~

down_scipy = down_by_label["SciPy"]
t_mean_scipy, t_sd_scipy = time_by_label["SciPy"]

canvas_scipy, roi_kwargs_canvas_scipy, roi_patch_scipy = _build_canvas_and_roi(
    down_scipy,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_scipy)
roi_titles.append(
    f"SciPy\n{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}"
)

_ = show_roi_zoom(
    canvas_scipy,
    ax_titles=(
        f"SciPy (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}",
        None,
    ),
    **roi_kwargs_canvas_scipy,
)

# %%
# Standard Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~

down_std = down_by_label["Standard"]
t_mean_std, t_sd_std = time_by_label["Standard"]

canvas_std, roi_kwargs_canvas_std, roi_patch_std = _build_canvas_and_roi(
    down_std,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_std)
roi_titles.append(
    f"Standard\n{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}"
)

_ = show_roi_zoom(
    canvas_std,
    ax_titles=(
        f"Standard (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}",
        None,
    ),
    **roi_kwargs_canvas_std,
)

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~

down_ls = down_by_label["Least-Squares"]
t_mean_ls, t_sd_ls = time_by_label["Least-Squares"]

canvas_ls, roi_kwargs_canvas_ls, roi_patch_ls = _build_canvas_and_roi(
    down_ls,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ls)
roi_titles.append(
    f"Least-Squares\n{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}"
)

_ = show_roi_zoom(
    canvas_ls,
    ax_titles=(
        f"Least-Squares (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}",
        None,
    ),
    **roi_kwargs_canvas_ls,
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

down_ob = down_by_label["Oblique"]
t_mean_ob, t_sd_ob = time_by_label["Oblique"]

canvas_ob, roi_kwargs_canvas_ob, roi_patch_ob = _build_canvas_and_roi(
    down_ob,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ob)
roi_titles.append(
    f"Oblique\n{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}"
)

_ = show_roi_zoom(
    canvas_ob,
    ax_titles=(
        f"Oblique (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}",
        None,
    ),
    **roi_kwargs_canvas_ob,
)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

roi_big_list = [_nearest_big(r, 256) for r in roi_patches]

# Make the figure a bit taller so multi-line titles and the suptitle
# have enough vertical room, even when the ROIs are square.
fig_width = 12.5
fig_height = 5.0  # was 3.8; bump to 5.0 to avoid clipping

fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height))
for ax, im, title in zip(axes, roi_big_list, roi_titles):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig.suptitle("Downsampled ROI comparison", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])  # leave a touch more room for the suptitle
plt.show()

# %%
# Image
# -----

img_name = "kodim19"
img_orig = orig_images[img_name]
h_img, w_img = img_orig.shape
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
zoom_factors_2d = (zoom, zoom)
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = cfg["roi_center_frac"]  # (row_frac, col_frac)
row_frac, col_frac = float(roi_center_frac[0]), float(roi_center_frac[1])

# Image-specific ROI center in absolute pixels
center_r = int(round(row_frac * h_img))
center_c = int(round(col_frac * w_img))
row_top = int(np.clip(center_r - roi_size_px // 2, 0, h_img - roi_size_px))
col_left = int(np.clip(center_c - roi_size_px // 2, 0, w_img - roi_size_px))
roi_rect = (row_top, col_left, roi_size_px, roi_size_px)

roi_kwargs_orig = dict(
    roi_height_frac=roi_size_px / h_img,
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

# %%
# Original with ROI
# ~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    img_orig,
    ax_titles=("Original with ROI", None),
    **roi_kwargs_orig,
)

# Prepare storage for ROI comparison
roi_patches = []
roi_titles  = []

# %%
# SciPy Interpolation
# ~~~~~~~~~~~~~~~~~~~

down_scipy = down_by_label["SciPy"]
t_mean_scipy, t_sd_scipy = time_by_label["SciPy"]

canvas_scipy, roi_kwargs_canvas_scipy, roi_patch_scipy = _build_canvas_and_roi(
    down_scipy,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_scipy)
roi_titles.append(
    f"SciPy\n{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}"
)

_ = show_roi_zoom(
    canvas_scipy,
    ax_titles=(
        f"SciPy (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}",
        None,
    ),
    **roi_kwargs_canvas_scipy,
)

# %%
# Standard Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~

down_std = down_by_label["Standard"]
t_mean_std, t_sd_std = time_by_label["Standard"]

canvas_std, roi_kwargs_canvas_std, roi_patch_std = _build_canvas_and_roi(
    down_std,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_std)
roi_titles.append(
    f"Standard\n{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}"
)

_ = show_roi_zoom(
    canvas_std,
    ax_titles=(
        f"Standard (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}",
        None,
    ),
    **roi_kwargs_canvas_std,
)

# %%
# Least-Squares Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~

down_ls = down_by_label["Least-Squares"]
t_mean_ls, t_sd_ls = time_by_label["Least-Squares"]

canvas_ls, roi_kwargs_canvas_ls, roi_patch_ls = _build_canvas_and_roi(
    down_ls,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ls)
roi_titles.append(
    f"Least-Squares\n{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}"
)

_ = show_roi_zoom(
    canvas_ls,
    ax_titles=(
        f"Least-Squares (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}",
        None,
    ),
    **roi_kwargs_canvas_ls,
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

down_ob = down_by_label["Oblique"]
t_mean_ob, t_sd_ob = time_by_label["Oblique"]

canvas_ob, roi_kwargs_canvas_ob, roi_patch_ob = _build_canvas_and_roi(
    down_ob,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ob)
roi_titles.append(
    f"Oblique\n{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}"
)

_ = show_roi_zoom(
    canvas_ob,
    ax_titles=(
        f"Oblique (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}",
        None,
    ),
    **roi_kwargs_canvas_ob,
)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

roi_big_list = [_nearest_big(r, 256) for r in roi_patches]

# Make the figure a bit taller so multi-line titles and the suptitle
# have enough vertical room, even when the ROIs are square.
fig_width = 12.5
fig_height = 5.0  # was 3.8; bump to 5.0 to avoid clipping

fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height))
for ax, im, title in zip(axes, roi_big_list, roi_titles):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig.suptitle("Downsampled ROI comparison", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])  # leave a touch more room for the suptitle
plt.show()

# %%
# Image
# -----

img_name = "kodim23"
img_orig = orig_images[img_name]
h_img, w_img = img_orig.shape
cfg = IMAGE_CONFIG[img_name]
zoom = float(cfg["zoom"])
zoom_factors_2d = (zoom, zoom)
roi_size_px = int(cfg["roi_size_px"])
roi_center_frac = cfg["roi_center_frac"]  # (row_frac, col_frac)
row_frac, col_frac = float(roi_center_frac[0]), float(roi_center_frac[1])

# Image-specific ROI center in absolute pixels
center_r = int(round(row_frac * h_img))
center_c = int(round(col_frac * w_img))
row_top = int(np.clip(center_r - roi_size_px // 2, 0, h_img - roi_size_px))
col_left = int(np.clip(center_c - roi_size_px // 2, 0, w_img - roi_size_px))
roi_rect = (row_top, col_left, roi_size_px, roi_size_px)

roi_kwargs_orig = dict(
    roi_height_frac=roi_size_px / h_img,
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

# %%
# Original with ROI
# ~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    img_orig,
    ax_titles=("Original with ROI", None),
    **roi_kwargs_orig,
)

# Prepare storage for ROI comparison
roi_patches = []
roi_titles  = []

# %%
# SciPy Interpolation
# ~~~~~~~~~~~~~~~~~~~

down_scipy = down_by_label["SciPy"]
t_mean_scipy, t_sd_scipy = time_by_label["SciPy"]

canvas_scipy, roi_kwargs_canvas_scipy, roi_patch_scipy = _build_canvas_and_roi(
    down_scipy,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_scipy)
roi_titles.append(
    f"SciPy\n{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}"
)

_ = show_roi_zoom(
    canvas_scipy,
    ax_titles=(
        f"SciPy (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_scipy)} ± {fmt_ms(t_sd_scipy)}",
        None,
    ),
    **roi_kwargs_canvas_scipy,
)

# %%
# Standard Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~

down_std = down_by_label["Standard"]
t_mean_std, t_sd_std = time_by_label["Standard"]

canvas_std, roi_kwargs_canvas_std, roi_patch_std = _build_canvas_and_roi(
    down_std,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_std)
roi_titles.append(
    f"Standard\n{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}"
)

_ = show_roi_zoom(
    canvas_std,
    ax_titles=(
        f"Standard (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_std)} ± {fmt_ms(t_sd_std)}",
        None,
    ),
    **roi_kwargs_canvas_std,
)

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~

down_ls = down_by_label["Least-Squares"]
t_mean_ls, t_sd_ls = time_by_label["Least-Squares"]

canvas_ls, roi_kwargs_canvas_ls, roi_patch_ls = _build_canvas_and_roi(
    down_ls,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ls)
roi_titles.append(
    f"Least-Squares\n{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}"
)

_ = show_roi_zoom(
    canvas_ls,
    ax_titles=(
        f"Least-Squares (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ls)} ± {fmt_ms(t_sd_ls)}",
        None,
    ),
    **roi_kwargs_canvas_ls,
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

down_ob = down_by_label["Oblique"]
t_mean_ob, t_sd_ob = time_by_label["Oblique"]

canvas_ob, roi_kwargs_canvas_ob, roi_patch_ob = _build_canvas_and_roi(
    down_ob,
    h_img=h_img,
    w_img=w_img,
    center_r=center_r,
    center_c=center_c,
    roi_size_px=roi_size_px,
    zoom_factors=zoom_factors_2d,
)
roi_patches.append(roi_patch_ob)
roi_titles.append(
    f"Oblique\n{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}"
)

_ = show_roi_zoom(
    canvas_ob,
    ax_titles=(
        f"Oblique (zoom={zoom:.3f}, orig={h_img}×{w_img})\n"
        f"{fmt_ms(t_mean_ob)} ± {fmt_ms(t_sd_ob)}",
        None,
    ),
    **roi_kwargs_canvas_ob,
)

# %%
# ROI Comparison
# ~~~~~~~~~~~~~~

roi_big_list = [_nearest_big(r, 256) for r in roi_patches]

# Make the figure a bit taller so multi-line titles and the suptitle
# have enough vertical room, even when the ROIs are square.
fig_width = 12.5
fig_height = 5.0  # was 3.8; bump to 5.0 to avoid clipping

fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height))
for ax, im, title in zip(axes, roi_big_list, roi_titles):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

fig.suptitle("Downsampled ROI comparison", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.93])  # leave a touch more room for the suptitle
plt.show()