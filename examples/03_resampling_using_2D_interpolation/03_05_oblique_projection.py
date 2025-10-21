# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_05_oblique_projection.py
# sphinx_gallery_end_ignore

"""
Oblique Projection
==================

Interpolate 2D images with and oblique projection.
"""

# %%
# Imports
# -------

import numpy as np

# sphinx_gallery_thumbnail_number = 2  # show second figure as thumbnail
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

from splineops.utils import (
    resize_and_compute_metrics,          # resampling + metrics
    plot_difference_image,
    show_roi_zoom,
    draw_leastsq_vs_oblique_pipeline,    # reused diagram helper (for layout consistency)
)

# %%
# Pipeline Diagram
# ----------------

_ = draw_leastsq_vs_oblique_pipeline(
    include_upsample_labels=True,
    width=12.0
)

# %%
# Highlights: ROI comparison
# --------------------------
# Load once, compute BOTH methods (keeping recovered + metrics) *if missing*,
# then show a 1×3 ROI triptych.

# --- Load (only if not already available) ---
if "input_image_normalized" not in locals():
    url = 'https://r0k.us/graphics/kodak/kodak/kodim14.png'
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content))
    data = np.array(img, dtype=np.float64)
    input_image_normalized = data / 255.0
    input_image_normalized = (
        input_image_normalized[:, :, 0] * 0.2989 +
        input_image_normalized[:, :, 1] * 0.5870 +
        input_image_normalized[:, :, 2] * 0.1140
    )

# Reuse / set shared constants
zoom_factors_2d = locals().get("zoom_factors_2d", (0.25, 0.25))
border_fraction = locals().get("border_fraction", 0.3)
ROI_SIZE_PX = locals().get("ROI_SIZE_PX", 64)

# ROI
FACE_ROW    = locals().get("FACE_ROW", 400)
FACE_COL    = locals().get("FACE_COL", 600)

# --- Compute both pipelines ONCE and keep recovered+metrics (reused later) ---
if not all(v in locals() for v in ("resized_2d_ls","recovered_2d_ls","snr_2d_ls","mse_2d_ls","time_2d_ls")):
    (resized_2d_ls, recovered_2d_ls, snr_2d_ls, mse_2d_ls, time_2d_ls) = resize_and_compute_metrics(
        input_image_normalized, method="cubic-best_antialiasing",
        zoom_factors=zoom_factors_2d, border_fraction=border_fraction
    )
if not all(v in locals() for v in ("resized_2d_ob","recovered_2d_ob","snr_2d_ob","mse_2d_ob","time_2d_ob")):
    (resized_2d_ob, recovered_2d_ob, snr_2d_ob, mse_2d_ob, time_2d_ob) = resize_and_compute_metrics(
        input_image_normalized, method="cubic-fast_antialiasing",
        zoom_factors=zoom_factors_2d, border_fraction=border_fraction
    )

# --- Build a quick ROI triptych (nearest-neighbour magnification) ---
def _nearest_big(roi: np.ndarray, target_h: int) -> np.ndarray:
    h, w = roi.shape
    mag = max(1, int(round(target_h / h)))
    return np.repeat(np.repeat(roi, mag, axis=0), mag, axis=1)

# Same ROI coords for all three since recovered images are original-sized
h_img, w_img = input_image_normalized.shape
row0 = int(np.clip(FACE_ROW - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col0 = int(np.clip(FACE_COL - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))

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
    ax.set_title(title); ax.axis("off"); ax.set_aspect("equal")
fig.tight_layout()
plt.show()

# %%
# Load and Normalize an Image
# ---------------------------

if "input_image_normalized" not in locals():
    url = 'https://r0k.us/graphics/kodak/kodak/kodim14.png'
    response = requests.get(url, timeout=10)
    img = Image.open(BytesIO(response.content))
    data = np.array(img, dtype=np.float64)

    # Convert to [0..1] + grayscale
    input_image_normalized = data / 255.0
    input_image_normalized = (
        input_image_normalized[:, :, 0] * 0.2989,  # Red
    ) + (
        input_image_normalized[:, :, 1] * 0.5870   # Green
    ) + (
        input_image_normalized[:, :, 2] * 0.1140   # Blue
    )

# Reuse constants if present; otherwise set them here.
zoom_factors_2d = locals().get("zoom_factors_2d", (0.25, 0.25))
border_fraction = locals().get("border_fraction", 0.3)

# ROI
ROI_SIZE_PX = locals().get("ROI_SIZE_PX", 64)
FACE_ROW    = locals().get("FACE_ROW", 250)
FACE_COL    = locals().get("FACE_COL", 445)

h_img, w_img = input_image_normalized.shape

# Top-left of the 64×64 box, clipped to stay inside the image
row_top = int(np.clip(FACE_ROW - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col_left = int(np.clip(FACE_COL - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))
roi_rect = (row_top, col_left, ROI_SIZE_PX, ROI_SIZE_PX)  # (r, c, h, w)

roi_kwargs = dict(
    roi_height_frac=ROI_SIZE_PX / h_img,  # keeps height at 64 px (square ROI)
    grayscale=True,
    roi_xy=(row_top, col_left),           # top-left of the ROI
)

# Shared mapping for resized-space ROI (used by both resized displays)
zoom_r, zoom_c = zoom_factors_2d
center_r_res = int(round(FACE_ROW * zoom_r))
center_c_res = int(round(FACE_COL * zoom_c))
roi_h_res = max(1, int(round(ROI_SIZE_PX * zoom_r)))
roi_w_res = max(1, int(round(ROI_SIZE_PX * zoom_c)))

# Original (shifted ROI)
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

need_ls = not all(
    v in locals()
    for v in ("resized_2d_ls", "recovered_2d_ls", "snr_2d_ls", "mse_2d_ls", "time_2d_ls")
)
if need_ls:
    (resized_2d_ls, recovered_2d_ls, snr_2d_ls, mse_2d_ls, time_2d_ls) = resize_and_compute_metrics(
        input_image_normalized,
        method="cubic-best_antialiasing",
        zoom_factors=zoom_factors_2d,
        border_fraction=border_fraction,
        roi=roi_rect
    )

h_res_ls, w_res_ls = resized_2d_ls.shape

row_top_res_ls = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res_ls - roi_h_res))
col_left_res_ls = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res_ls - roi_w_res))

# Build original-size white canvas and paste the small resized LS image at top-left (0,0)
canvas_ls = np.ones((h_img, w_img), dtype=resized_2d_ls.dtype)  # white background in [0,1]
canvas_ls[:h_res_ls, :w_res_ls] = resized_2d_ls

roi_kwargs_on_canvas_ls = dict(
    roi_height_frac=roi_h_res / h_img,     # << was roi_h_res
    grayscale=True,
    roi_xy=(row_top_res_ls, col_left_res_ls),
)

_ = show_roi_zoom(
    canvas_ls,
    ax_titles=(f"Resized Image (least-squares; t={time_2d_ls*1000:.1f} ms)", None),
    **roi_kwargs_on_canvas_ls
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~

need_ob = not all(
    v in locals()
    for v in ("resized_2d_ob", "recovered_2d_ob", "snr_2d_ob", "mse_2d_ob", "time_2d_ob")
)
if need_ob:
    (resized_2d_ob, recovered_2d_ob, snr_2d_ob, mse_2d_ob, time_2d_ob) = resize_and_compute_metrics(
        input_image_normalized,
        method="cubic-fast_antialiasing",
        zoom_factors=zoom_factors_2d,
        border_fraction=border_fraction,
        roi=roi_rect
    )

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

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    recovered_2d_ls,
    ax_titles=("Recovered Image (least-squares projection)", None),
    **roi_kwargs
)

# %%
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

# %%
# Least-Squares Projection
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Difference with original image on ROI.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ls,
    snr=snr_2d_ls,
    mse=mse_2d_ls,
    roi=roi_rect,
    title_prefix="Difference (least-squares)"
)

# %%
# Oblique Projection
# ~~~~~~~~~~~~~~~~~~
#
# Difference with original image on ROI.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ob,
    snr=snr_2d_ob,
    mse=mse_2d_ob,
    roi=roi_rect,
    title_prefix="Difference (oblique)"
)

# %%
# Performance: Time Comparison
# ----------------------------

speedup = (time_2d_ls / time_2d_ob) if time_2d_ob > 0 else np.inf
impr_pct = max(0.0, (1.0 - time_2d_ob / max(time_2d_ls, 1e-12)) * 100.0)

print(f"[Timing] Least-Squares: {time_2d_ls*1000:.1f} ms")
print(f"[Timing] Oblique      : {time_2d_ob*1000:.1f} ms")
print(f"[Timing] Speedup (LS/OB): {speedup:.2f}×  (~{impr_pct:.1f}% less time)")

fig, ax = plt.subplots(figsize=(6.5, 3.6))
methods = ["Least-Squares", "Oblique"]
times_s = [time_2d_ls, time_2d_ob]
bars = ax.bar(methods, times_s)
ax.set_ylabel("Time (s)")
ax.set_title(f"Oblique is ≈ {speedup:.2f}× faster ({impr_pct:.1f}% less time)")
for rect, t in zip(bars, times_s):
    h = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, h, f"{t*1000:.1f} ms",
            ha="center", va="bottom", fontsize=9)
fig.tight_layout()
plt.show()
