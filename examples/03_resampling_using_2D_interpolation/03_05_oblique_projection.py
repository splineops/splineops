# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_05_oblique_projection.py
# sphinx_gallery_end_ignore

"""
Oblique Projection
==================

Interpolate 2D images with an oblique projection and compare against
least-squares projection. SNR/MSE are computed on a central region
(via border_fraction) to reduce boundary artifacts, while visual
comparisons use a face ROI.
"""

# %%
# Imports
# -------

import numpy as np
import time

# sphinx_gallery_thumbnail_number = 2  # show second figure as thumbnail
from urllib.request import urlopen
from PIL import Image
import matplotlib.pyplot as plt

from splineops.resize import resize
from splineops.utils.metrics import compute_snr_and_mse_region
from splineops.utils.plotting import plot_difference_image, show_roi_zoom
from splineops.utils.diagram import draw_leastsq_vs_oblique_pipeline
from splineops.utils.specs import print_runtime_context

def fmt_ms(seconds: float) -> str:
    """Format seconds as a short 'X.X ms' string."""
    return f"{seconds * 1000.0:.1f} ms"

# Use float32 for storage / IO (resize still computes internally in float64).
DTYPE = np.float32

# Small helper: run one resize pipeline for a given method
def _run_pipeline(
    img: np.ndarray,
    *,
    method: str,
    zoom_factors: tuple[float, float],
    border_fraction: float,
    roi=None,
):
    """
    Forward + backward resize with timing and SNR/MSE.

    Returns
    -------
    resized, recovered, snr, mse,
    elapsed_forward_s, elapsed_backward_s, elapsed_total_s
    """
    t0 = time.perf_counter()
    resized = resize(img, zoom_factors=zoom_factors, method=method)
    t1 = time.perf_counter()

    recovered = resize(resized, output_size=img.shape, method=method)
    t2 = time.perf_counter()

    elapsed_forward = t1 - t0
    elapsed_backward = t2 - t1
    elapsed_total = t2 - t0

    snr, mse = compute_snr_and_mse_region(
        img,
        recovered,
        roi=roi,
        border_fraction=border_fraction,
    )
    return resized, recovered, snr, mse, elapsed_forward, elapsed_backward, elapsed_total


# %%
# Pipeline Diagram
# ----------------

_ = draw_leastsq_vs_oblique_pipeline(
    include_upsample_labels=True,
    width=12.0
)

# %%
# Load and Normalize an Image
# ---------------------------

url = 'https://r0k.us/graphics/kodak/kodak/kodim14.png'
with urlopen(url, timeout=10) as resp:
    img = Image.open(resp)
data = np.array(img, dtype=np.float64)

# Convert to [0..1] + grayscale
input_image_normalized = data / 255.0
input_image_normalized = (
    input_image_normalized[:, :, 0] * 0.2989 +  # Red
    input_image_normalized[:, :, 1] * 0.5870 +  # Green
    input_image_normalized[:, :, 2] * 0.1140    # Blue
)

input_image_normalized = input_image_normalized.astype(DTYPE, copy=False)

h_img, w_img = input_image_normalized.shape

# Shared constants
zoom = np.e / 9          # ≈ 0.3020313142732272
zoom_factors_2d = (zoom, zoom)
border_fraction = 0.3
ROI_SIZE_PX = 64

# ROI center (face-ish area)
FACE_ROW, FACE_COL = 400, 600

# Top-left of the 64×64 box, clipped to stay inside the image
row_top = int(np.clip(FACE_ROW - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col_left = int(np.clip(FACE_COL - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))
roi_rect = (row_top, col_left, ROI_SIZE_PX, ROI_SIZE_PX)  # (r, c, h, w)

roi_kwargs = dict(
    roi_height_frac=ROI_SIZE_PX / h_img,  # keeps height at 64 px (square ROI)
    grayscale=True,
    roi_xy=(row_top, col_left),           # top-left of the ROI
)

# Mapping for ROI in resized images
zoom_r, zoom_c = zoom_factors_2d
center_r_res = int(round(FACE_ROW * zoom_r))
center_c_res = int(round(FACE_COL * zoom_c))
roi_h_res = max(1, int(round(ROI_SIZE_PX * zoom_r)))
roi_w_res = max(1, int(round(ROI_SIZE_PX * zoom_c)))

# %%
# Least-Squares vs Oblique
# ------------------------

# Standard interpolation: cubic (baseline)
(resized_2d_std, recovered_2d_std,
 snr_2d_std, mse_2d_std,
 time_2d_std_fwd, time_2d_std_back, time_2d_std) = _run_pipeline(
    input_image_normalized,
    method="cubic",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=None,  # metrics on central region
)

# Least-squares projection: cubic-best_antialiasing
(resized_2d_ls, recovered_2d_ls,
 snr_2d_ls, mse_2d_ls,
 time_2d_ls_fwd, time_2d_ls_back, time_2d_ls) = _run_pipeline(
    input_image_normalized,
    method="cubic-best_antialiasing",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=None,  # metrics on central region
)

# Oblique projection: cubic-fast_antialiasing
(resized_2d_ob, recovered_2d_ob,
 snr_2d_ob, mse_2d_ob,
 time_2d_ob_fwd, time_2d_ob_back, time_2d_ob) = _run_pipeline(
    input_image_normalized,
    method="cubic-fast_antialiasing",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=None,  # metrics on central region
)

# %%
# ROI Comparison
# --------------
#
# Build a 1×3 ROI triptych (nearest-neighbour magnification).

def _nearest_big(roi: np.ndarray, target_h: int) -> np.ndarray:
    h, w = roi.shape
    mag = max(1, int(round(target_h / h)))
    return np.repeat(np.repeat(roi, mag, axis=0), mag, axis=1)

row0 = row_top
col0 = col_left

roi_orig = input_image_normalized[row0:row0+ROI_SIZE_PX, col0:col0+ROI_SIZE_PX]
roi_ls   = recovered_2d_ls[  row0:row0+ROI_SIZE_PX, col0:col0+ROI_SIZE_PX]
roi_ob   = recovered_2d_ob[  row0:row0+ROI_SIZE_PX, col0:col0+ROI_SIZE_PX]

DISPLAY_H = 256
roi_big_orig = _nearest_big(roi_orig, DISPLAY_H)
roi_big_ls   = _nearest_big(roi_ls,   DISPLAY_H)
roi_big_ob   = _nearest_big(roi_ob,   DISPLAY_H)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))

titles = [
    "Original ROI",
    f"Recovered (Least-Squares, {fmt_ms(time_2d_ls_back)})",
    f"Recovered (Oblique, {fmt_ms(time_2d_ob_back)})",
]

for ax, im, title in zip(
    axes,
    [roi_big_orig, roi_big_ls, roi_big_ob],
    titles,
):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")
fig.tight_layout()
plt.show()

# %%
# Original (with ROI)
# -------------------

_ = show_roi_zoom(
    input_image_normalized,
    ax_titles=("Original Image", None),
    **roi_kwargs
)

# %%
# Resized Images
# --------------

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~

h_res_ls, w_res_ls = resized_2d_ls.shape

row_top_res_ls = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res_ls - roi_h_res))
col_left_res_ls = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res_ls - roi_w_res))

canvas_ls = np.ones((h_img, w_img), dtype=resized_2d_ls.dtype)  # white background in [0,1]
canvas_ls[:h_res_ls, :w_res_ls] = resized_2d_ls

roi_kwargs_on_canvas_ls = dict(
    roi_height_frac=roi_h_res / h_img,
    grayscale=True,
    roi_xy=(row_top_res_ls, col_left_res_ls),
)

_ = show_roi_zoom(
    canvas_ls,
    ax_titles=(f"Resized Image (least-squares, {fmt_ms(time_2d_ls_fwd)})", None),
    **roi_kwargs_on_canvas_ls
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

h_res_ob, w_res_ob = resized_2d_ob.shape

row_top_res_ob = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res_ob - roi_h_res))
col_left_res_ob = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res_ob - roi_w_res))

canvas_ob = np.ones((h_img, w_img), dtype=resized_2d_ob.dtype)
canvas_ob[:h_res_ob, :w_res_ob] = resized_2d_ob

roi_kwargs_on_canvas_ob = dict(
    roi_height_frac=roi_h_res / h_img,
    grayscale=True,
    roi_xy=(row_top_res_ob, col_left_res_ob),
)

_ = show_roi_zoom(
    canvas_ob,
    ax_titles=(f"Resized Image (oblique, {fmt_ms(time_2d_ob_fwd)})", None),
    **roi_kwargs_on_canvas_ob
)

# %%
# Recovered Images
# ----------------

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    recovered_2d_ls,
    ax_titles=(f"Recovered Image (least-squares projection, {fmt_ms(time_2d_ls_back)})", None),
    **roi_kwargs
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    recovered_2d_ob,
    ax_titles=(f"Recovered Image (oblique projection, {fmt_ms(time_2d_ob_back)})", None),
    **roi_kwargs
)

# %%
# Difference Images
# -----------------

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Difference with original image on the face ROI (SNR/MSE shown come from the
# central-region metrics, not strictly ROI-only).

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ls,
    snr=snr_2d_ls,
    mse=mse_2d_ls,
    roi=roi_rect,
    title_prefix="Difference (least-squares)",
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~
#
# Difference with original image on the face ROI.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ob,
    snr=snr_2d_ob,
    mse=mse_2d_ob,
    roi=roi_rect,
    title_prefix="Difference (oblique)",
)

# %%
# Performance Comparison
# ----------------------

N_TRIALS = 10
WARMUP   = 1

def _avg_time_over_runs(
    img,
    *,
    method: str,
    zoom_factors: tuple[float, float],
    border_fraction: float,
    roi=None,
    trials: int = N_TRIALS,
    warmup: int = WARMUP,
):
    """Return (mean_s, sd_s) total pipeline timing over multiple runs; warm-up not counted."""
    # Warm-up (not timed)
    for _ in range(warmup):
        _run_pipeline(
            img,
            method=method,
            zoom_factors=zoom_factors,
            border_fraction=border_fraction,
            roi=roi,
        )

    times = []
    for _ in range(trials):
        _, _, _, _, _, _, t_total = _run_pipeline(
            img,
            method=method,
            zoom_factors=zoom_factors,
            border_fraction=border_fraction,
            roi=roi,
        )
        times.append(t_total)

    times = np.asarray(times, dtype=np.float64)
    mean_s = float(times.mean())
    sd_s   = float(times.std(ddof=1)) if times.size > 1 else 0.0
    return mean_s, sd_s

# Measure averages for Standard, LS, and Oblique (total pipeline time)
mean_std, sd_std = _avg_time_over_runs(
    input_image_normalized,
    method="cubic",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=None,
)
mean_ls, sd_ls = _avg_time_over_runs(
    input_image_normalized,
    method="cubic-best_antialiasing",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=None,
)
mean_ob, sd_ob = _avg_time_over_runs(
    input_image_normalized,
    method="cubic-fast_antialiasing",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=None,
)

speedup_mean = (mean_ls / mean_ob) if mean_ob > 0 else np.inf
impr_pct_mean = max(0.0, (1.0 - mean_ob / max(mean_ls, 1e-12)) * 100.0)

print(f"[Timing averages over {N_TRIALS} runs] Standard     : {mean_std*1000:.1f} ± {sd_std*1000:.1f} ms")
print(f"[Timing averages over {N_TRIALS} runs] Least-Squares: {mean_ls*1000:.1f} ± {sd_ls*1000:.1f} ms")
print(f"[Timing averages over {N_TRIALS} runs] Oblique      : {mean_ob*1000:.1f} ± {sd_ob*1000:.1f} ms")
print(f"[Timing] Speedup (LS/OB): {speedup_mean:.2f}×  (~{impr_pct_mean:.1f}% less time)\n")

fig, ax = plt.subplots(figsize=(7.0, 3.8))
methods   = ["Standard", "Least-Squares", "Oblique"]
means_s   = [mean_std,  mean_ls,         mean_ob]
errs_s    = [sd_std,    sd_ls,          sd_ob]

bars = ax.bar(methods, means_s, yerr=errs_s, capsize=6)
ax.set_ylabel("Time (s)")
ax.set_title(
    f"Standard vs LS vs Oblique "
    f"(Oblique is ≈ {speedup_mean:.2f}× faster than LS)"
)

for rect, m, sd in zip(bars, means_s, errs_s):
    h = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        h,
        f"{m*1000:.1f} ± {sd*1000:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
plt.show()

# %%
# Summary Table
# -------------
#
# Central-region SNR/MSE (via border_fraction) plus averaged total
# (forward + backward) timings for all three methods in this example.

methods_summary = [
    ("Standard (cubic)",          snr_2d_std, mse_2d_std, mean_std, sd_std),
    ("Least-Squares (best AA)",   snr_2d_ls,  mse_2d_ls,  mean_ls,  sd_ls),
    ("Oblique (fast AA)",         snr_2d_ob,  mse_2d_ob,  mean_ob,  sd_ob),
]

header_line = f"{'Method':<28} {'SNR (dB)':>10} {'MSE':>16} {'Time (s, avg±sd)':>22}"
print(header_line)
print("-" * len(header_line))

for name, snr_val, mse_val, t_mean, t_sd in methods_summary:
    time_str = f"{t_mean:.4f} ± {t_sd:.4f}"
    print(
        f"{name:<28} "
        f"{snr_val:>10.2f} "
        f"{mse_val:>16.2e} "
        f"{time_str:>22}"
    )

print_runtime_context()
