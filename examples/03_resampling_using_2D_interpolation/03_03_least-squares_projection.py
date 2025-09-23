# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_03_least-squares_projection.py
# sphinx_gallery_end_ignore

"""
Least-Squares Projection
========================

Interpolate 2D images with least-squares projection.
Compare them to *standard interpolation*. We compute SNR and MSE only on a
central region to exclude boundary artifacts.
"""

# %%
# Imports
# -------

import numpy as np

# sphinx_gallery_thumbnail_number = 4  # show fourth figure as thumbnail
import requests
from io import BytesIO
from PIL import Image

from splineops.utils import (
    resize_and_compute_metrics,      # resampling + metrics
    plot_difference_image,
    show_roi_zoom,
    draw_standard_vs_leastsq_pipeline, # reused diagram helper (for layout consistency)
)

# %%
# Pipeline Diagram
# ----------------
#
# These experiments validate least-squares projection against *standard interpolation*
# by showing how close their results are (and where they differ).

_ = draw_standard_vs_leastsq_pipeline(
    include_upsample_labels=True, # show '↑ 4' inside both boxes
    width=12.0                    # figure width in inches (height auto)
)

# %%
# Load and Normalize an Image
# ---------------------------
#
# Here, we load an example image from an online repository.
# We convert it to grayscale in [0, 1].

url = 'https://r0k.us/graphics/kodak/kodak/kodim14.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
data = np.array(img, dtype=np.float64)

# Convert to [0..1]
input_image_normalized = data / 255.0

# Convert to grayscale via simple weighting
input_image_normalized = (
    input_image_normalized[:, :, 0] * 0.2989 +  # Red channel
    input_image_normalized[:, :, 1] * 0.5870 +  # Green channel
    input_image_normalized[:, :, 2] * 0.1140    # Blue channel
)

zoom_factors_2d = (0.25, 0.25)
border_fraction = 0.3

# Face-centered 64×64 ROI
ROI_SIZE_PX = 64
FACE_ROW, FACE_COL = 250, 445  # (row, col)

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

# Original (shifted ROI)
_ = show_roi_zoom(
    input_image_normalized,
    ax_titles=("Original Image", None),
    **roi_kwargs
)

# %%
# Least-Squares Projection
# ------------------------
#
# We use the least-squares projection method (cubic with best anti-aliasing).

(
    resized_2d_ls,
    recovered_2d_ls,
    snr_2d_ls,
    mse_2d_ls,
    time_2d_ls
) = resize_and_compute_metrics(
    input_image_normalized,
    method="cubic-best_antialiasing",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=roi_rect
)

# %%
# Resized Image (least-squares)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We plot the resized image. Note that this is the same for all methods in this
# gallery; we just compute it using the chosen method for convenience.

# Zoomed face detail for the resized (least-squares) image — pasted onto original-size canvas ===
h_res_ls, w_res_ls = resized_2d_ls.shape
zoom_r, zoom_c = zoom_factors_2d

# ROI size in the resized image (e.g., 64 -> 16 px when zoom=0.25)
roi_h_res_ls = max(1, int(round(ROI_SIZE_PX * zoom_r)))
roi_w_res_ls = max(1, int(round(ROI_SIZE_PX * zoom_c)))

# ROI center mapped into the resized image
center_r_res = int(round(FACE_ROW * zoom_r))
center_c_res = int(round(FACE_COL * zoom_c))

# Top-left of the ROI in the resized (least-squares) image, clipped to bounds
row_top_res_ls = int(np.clip(center_r_res - roi_h_res_ls // 2, 0, h_res_ls - roi_h_res_ls))
col_left_res_ls = int(np.clip(center_c_res - roi_w_res_ls // 2, 0, w_res_ls - roi_w_res_ls))

# --- Build original-size white canvas and paste the small resized LS image at top-left (0,0) ---
canvas_ls = np.ones((h_img, w_img), dtype=resized_2d_ls.dtype)  # white background in [0,1]
canvas_ls[:h_res_ls, :w_res_ls] = resized_2d_ls

# IMPORTANT: roi_height_frac must be relative to the canvas height (original size),
# but the ROI dimensions are those of the resized image region.
roi_kwargs_on_canvas_ls = dict(
    roi_height_frac=roi_h_res_ls / h_img,   # keeps the inset square at roi_h_res_ls pixels high
    grayscale=True,
    roi_xy=(row_top_res_ls, col_left_res_ls),  # same coords since pasted at (0,0)
)

_ = show_roi_zoom(
    canvas_ls,
    ax_titles=("Resized Image (least-squares)", None),
    **roi_kwargs_on_canvas_ls
)

# %%
# Standard Interpolation
# ----------------------
#
# For comparison purposes, we also use the *standard interpolation* (cubic).

(
    resized_2d_std,
    recovered_2d_std,
    snr_2d_std,
    mse_2d_std,
    time_2d_std
) = resize_and_compute_metrics(
    input_image_normalized,
    method="cubic",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction,
    roi=roi_rect
)

# %%
# Resized Image (standard)
# ~~~~~~~~~~~~~~~~~~~~~~~~
#
# Mirror the least-squares resized view for the standard interpolation.

h_res_std, w_res_std = resized_2d_std.shape

roi_h_res_std = max(1, int(round(ROI_SIZE_PX * zoom_r)))
roi_w_res_std = max(1, int(round(ROI_SIZE_PX * zoom_c)))

row_top_res_std = int(np.clip(center_r_res - roi_h_res_std // 2, 0, h_res_std - roi_h_res_std))
col_left_res_std = int(np.clip(center_c_res - roi_w_res_std // 2, 0, w_res_std - roi_w_res_std))

canvas_std = np.ones((h_img, w_img), dtype=resized_2d_std.dtype)
canvas_std[:h_res_std, :w_res_std] = resized_2d_std

roi_kwargs_on_canvas_std = dict(
    roi_height_frac=roi_h_res_std / h_img,
    grayscale=True,
    roi_xy=(row_top_res_std, col_left_res_std),
)

_ = show_roi_zoom(
    canvas_std,
    ax_titles=("Resized Image (standard)", None),
    **roi_kwargs_on_canvas_std
)

# %%
# Recovered Image (least-squares projection)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We plot the recovered images after reversing the zoom factors.

_ = show_roi_zoom(
    recovered_2d_ls,
    ax_titles=("Recovered Image (least-squares projection)", None),
    **roi_kwargs
)

# %%
# Recovered Image (standard interpolation)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We plot the recovered images after reversing the zoom factors.

_ = show_roi_zoom(
    recovered_2d_std,
    ax_titles=("Recovered Image (standard interpolation)", None),
    **roi_kwargs
)

# %%
# Difference with original image (least-squares)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered with least-squares) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ls,
    snr=snr_2d_ls,
    mse=mse_2d_ls,
    roi=roi_rect,
    title_prefix="Difference (least-squares)"
)

# %%
# Difference with original image (standard)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered with standard interpolation) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_std,
    snr=snr_2d_std,
    mse=mse_2d_std,
    roi=roi_rect,
    title_prefix="Difference (standard)"
)