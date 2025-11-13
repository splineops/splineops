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
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

from splineops.resize import resize
from splineops.utils.metrics import compute_snr_and_mse_region
from splineops.utils.plotting import plot_difference_image, show_roi_zoom
from splineops.utils.diagram import draw_leastsq_vs_oblique_pipeline
from splineops.utils.specs import print_runtime_context

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
    resized, recovered, snr, mse, elapsed_s
    """
    t0 = time.perf_counter()
    resized = resize(img, zoom_factors=zoom_factors, method=method)
    elapsed = time.perf_counter() - t0

    recovered = resize(resized, output_size=img.shape, method=method)

    snr, mse = compute_snr_and_mse_region(
        img,
        recovered,
        roi=roi,
        border_fraction=border_fraction,
    )
    return resized, recovered, snr, mse, elapsed

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
response = requests.get(url, timeout=10)
img = Image.open(BytesIO(response.content))
data = np.array(img, dtype=np.float64)

# Convert to [0..1] + grayscale
input_image_normalized = data / 255.0
input_image_normalized = (
    input_image_normalized[:, :, 0] * 0.2989 +  # Red
    input_image_normalized[:, :, 1] * 0.5870 +  # Green
    input_image_normalized[:, :, 2] * 0.1140    # Blue
)

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

# Least-squares projection: cubic-best_antialiasing
resized_2d_ls, recovered_2d_ls, snr_2d_ls, mse_2d_ls, time_2d_ls = _run_pipeline(
    input_image_normalized,
    method="cubic-best_antialiasing",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=None,  # metrics on central region
)

# Oblique projection: cubic-fast_antialiasing
resized_2d_ob, recovered_2d_ob, snr_2d_ob, mse_2d_ob, time_2d_ob = _run_pipeline(
    input_image_normalized,
    method="cubic-fast_antialiasing",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=None,  # metrics on central region
)

# %%
# Highlights: ROI comparison
# --------------------------
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
for ax, im, title in zip(
    axes,
    [roi_big_orig, roi_big_ls, roi_big_ob],
    ["Original ROI", "Recovered (Least-Squares)", "Recovered (Oblique)"]
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

# Least-Squares Projection (resized)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    ax_titles=(f"Resized Image (least-squares; t={time_2d_ls*1000:.1f} ms)", None),
    **roi_kwargs_on_canvas_ls
)

# Oblique Projection (resized)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    ax_titles=(f"Resized Image (oblique; t={time_2d_ob*1000:.1f} ms)", None),
    **roi_kwargs_on_canvas_ob
)

# %%
# Recovered Images
# ----------------

# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    recovered_2d_ls,
    ax_titles=("Recovered Image (least-squares projection)", None),
    **roi_kwargs
)

# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    recovered_2d_ob,
    ax_titles=("Recovered Image (oblique projection)", None),
    **roi_kwargs
)

# %%
# Difference Images
# -----------------

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
# Performance: Time Comparison
# ----------------------------

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
    """Return (mean_s, sd_s) timing over multiple runs; warm-up not counted."""
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
        _, _, _, _, t = _run_pipeline(
            img,
            method=method,
            zoom_factors=zoom_factors,
            border_fraction=border_fraction,
            roi=roi,
        )
        times.append(t)

    times = np.asarray(times, dtype=np.float64)
    mean_s = float(times.mean())
    sd_s   = float(times.std(ddof=1)) if times.size > 1 else 0.0
    return mean_s, sd_s

# Measure LS and Oblique averages
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

print(f"[Timing averages over {N_TRIALS} runs] Least-Squares: {mean_ls*1000:.1f} ± {sd_ls*1000:.1f} ms")
print(f"[Timing averages over {N_TRIALS} runs] Oblique      : {mean_ob*1000:.1f} ± {sd_ob*1000:.1f} ms")
print(f"[Timing] Speedup (LS/OB): {speedup_mean:.2f}×  (~{impr_pct_mean:.1f}% less time)\n")

fig, ax = plt.subplots(figsize=(7.0, 3.8))
methods   = ["Least-Squares", "Oblique"]
means_s   = [mean_ls, mean_ob]
errs_s    = [sd_ls, sd_ob]

bars = ax.bar(methods, means_s, yerr=errs_s, capsize=6)
ax.set_ylabel("Time (s)")
ax.set_title(
    f"Oblique is ≈ {speedup_mean:.2f}× faster on average "
    f"({impr_pct_mean:.1f}% less time over {N_TRIALS} runs)"
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

print_runtime_context()
