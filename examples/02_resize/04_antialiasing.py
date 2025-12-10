# sphinx_gallery_start_ignore
# splineops/examples/02_resize/04_antialiasing.py
# sphinx_gallery_end_ignore

"""
Antialiasing
============

Interpolate 2D images with an **antialiased** down-sampling step and compare
the result to standard interpolation.

We shrink the image with either:

- plain cubic interpolation (no explicit low-pass),
- cubic antialiasing (oblique projection low-pass) via ``"cubic-antialiasing"``,

then up-sample both back to the original size using standard cubic
interpolation. SNR and MSE are computed only on a central region to
exclude boundary artifacts.
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

from scipy.ndimage import zoom as _scipy_zoom  # only if you want extra comparisons
from splineops.resize import resize, resize_degrees
from splineops.utils.metrics import compute_snr_and_mse_region
from splineops.utils.plotting import plot_difference_image, show_roi_zoom
from splineops.utils.diagram import draw_two_method_comparisons


def fmt_ms(seconds: float) -> str:
    """Format seconds as a short 'X.X ms' string."""
    return f"{seconds * 1000.0:.1f} ms"


# Use float32 for storage / IO (resize still computes internally in float64)
DTYPE = np.float32

# %%
# Pipeline Diagram
# ----------------

_ = draw_two_method_comparisons(
    "Standard Interpolation",
    "Antialiasing",
    include_downsample_labels=True,
    include_upsample_labels=True,
    scale_factor=4,
    width=12.0,
)

# %%
# Load and Normalize an Image
# ---------------------------

url = "https://r0k.us/graphics/kodak/kodak/kodim14.png"
with urlopen(url, timeout=10) as resp:
    img = Image.open(resp)
data = np.array(img, dtype=np.float64)

# Convert to [0..1] + grayscale
input_image_normalized = data / 255.0
input_image_normalized = (
    input_image_normalized[:, :, 0] * 0.2989  # Red channel
    + input_image_normalized[:, :, 1] * 0.5870  # Green channel
    + input_image_normalized[:, :, 2] * 0.1140  # Blue channel
)

# Run the spline backend in float32 for performance
# (it still computes internally in float64).
input_image_normalized = input_image_normalized.astype(DTYPE, copy=False)

h_img, w_img = input_image_normalized.shape

# Shared parameters
zoom = np.e / 9          # ≈ 0.3020313142732272
zoom_factors_2d = (zoom, zoom)
border_fraction = 0.3    # central crop for SNR/MSE
ROI_SIZE_PX = 64

# Face-centered 64×64 ROI (for visual comparisons)
FACE_ROW, FACE_COL = 400, 600  # (row, col) approx center of the detail

# Top-left of the 64×64 box, clipped to stay inside the image
row_top = int(np.clip(FACE_ROW - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col_left = int(np.clip(FACE_COL - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))
roi_rect = (row_top, col_left, ROI_SIZE_PX, ROI_SIZE_PX)  # (r, c, h, w)

roi_kwargs = dict(
    roi_height_frac=ROI_SIZE_PX / h_img,  # keeps height at 64 px (square ROI)
    grayscale=True,
    roi_xy=(row_top, col_left),           # top-left of the ROI
)

# Mapping for resized-space ROI (used by both resized displays)
zoom_r, zoom_c = zoom_factors_2d
center_r_res = int(round(FACE_ROW * zoom_r))
center_c_res = int(round(FACE_COL * zoom_c))
roi_h_res = max(1, int(round(ROI_SIZE_PX * zoom_r)))
roi_w_res = max(1, int(round(ROI_SIZE_PX * zoom_c)))

# %%
# Standard Interpolation
# ----------------------

t0 = time.perf_counter()
resized_2d_std = resize(
    input_image_normalized,
    zoom_factors=zoom_factors_2d,
    method="cubic",
)
t1 = time.perf_counter()
recovered_2d_std = resize(
    resized_2d_std,
    output_size=input_image_normalized.shape,
    method="cubic",
)
t2 = time.perf_counter()

time_2d_std_fwd = t1 - t0         # forward resize (down/up)
time_2d_std_back = t2 - t1        # backward resize (return to original size)
time_2d_std = t2 - t0             # total pipeline time

# SNR/MSE on central region (no ROI cropping here)
snr_2d_std, mse_2d_std = compute_snr_and_mse_region(
    input_image_normalized,
    recovered_2d_std,
    border_fraction=border_fraction,
)

# %%
# Antialiasing
# ------------

t0 = time.perf_counter()
resized_2d_aa = resize(
    input_image_normalized,
    zoom_factors=zoom_factors_2d,
    method="cubic-antialiasing",  # antialiased shrink
)
t1 = time.perf_counter()
recovered_2d_aa = resize(
    resized_2d_aa,
    output_size=input_image_normalized.shape,
    method="cubic-antialiasing",  # (you could also use "cubic" here)
)
t2 = time.perf_counter()

time_2d_aa_fwd = t1 - t0
time_2d_aa_back = t2 - t1
time_2d_aa = t2 - t0

snr_2d_aa, mse_2d_aa = compute_snr_and_mse_region(
    input_image_normalized,
    recovered_2d_aa,
    border_fraction=border_fraction,
)

# %%
# ROI Comparison
# --------------
#
# Build a quick ROI triptych (nearest-neighbour magnification) from the
# recovered images for visual comparison.

def _nearest_big(roi: np.ndarray, target_h: int) -> np.ndarray:
    h, w = roi.shape
    mag = max(1, int(round(target_h / h)))
    return np.repeat(np.repeat(roi, mag, axis=0), mag, axis=1)

roi_orig = input_image_normalized[row_top:row_top+ROI_SIZE_PX, col_left:col_left+ROI_SIZE_PX]
roi_std  = recovered_2d_std[row_top:row_top+ROI_SIZE_PX, col_left:col_left+ROI_SIZE_PX]
roi_aa   = recovered_2d_aa[row_top:row_top+ROI_SIZE_PX, col_left:col_left+ROI_SIZE_PX]

DISPLAY_H = 256
roi_big_orig = _nearest_big(roi_orig, DISPLAY_H)
roi_big_std  = _nearest_big(roi_std,  DISPLAY_H)
roi_big_aa   = _nearest_big(roi_aa,   DISPLAY_H)

fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))

titles = [
    "Original ROI",
    f"Recovered (Standard, {fmt_ms(time_2d_std_back)})",
    f"Recovered (Antialiasing, {fmt_ms(time_2d_aa_back)})",
]

for ax, im, title in zip(
    axes,
    [roi_big_orig, roi_big_std, roi_big_aa],
    titles,
):
    ax.imshow(im, cmap="gray", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")

fig.tight_layout()
plt.show()

# %%
# Original with ROI
# -----------------

_ = show_roi_zoom(
    input_image_normalized,
    ax_titles=("Original Image", None),
    **roi_kwargs
)

# %%
# Resized Images
# --------------

# %%
# Antialiasing
# ~~~~~~~~~~~~

h_res_aa, w_res_aa = resized_2d_aa.shape

row_top_res_aa = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res_aa - roi_h_res))
col_left_res_aa = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res_aa - roi_w_res))

canvas_aa = np.ones((h_img, w_img), dtype=resized_2d_aa.dtype)  # white background in [0,1]
canvas_aa[:h_res_aa, :w_res_aa] = resized_2d_aa

roi_kwargs_on_canvas_aa = dict(
    roi_height_frac=roi_h_res / h_img,
    grayscale=True,
    roi_xy=(row_top_res_aa, col_left_res_aa),
)

_ = show_roi_zoom(
    canvas_aa,
    ax_titles=(
        f"Resized Image (antialiasing, {fmt_ms(time_2d_aa_fwd)})",
        None,
    ),
    **roi_kwargs_on_canvas_aa
)

# %%
# Standard Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~

h_res_std, w_res_std = resized_2d_std.shape

row_top_res_std = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res_std - roi_h_res))
col_left_res_std = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res_std - roi_w_res))

canvas_std = np.ones((h_img, w_img), dtype=resized_2d_std.dtype)
canvas_std[:h_res_std, :w_res_std] = resized_2d_std

roi_kwargs_on_canvas_std = dict(
    roi_height_frac=roi_h_res / h_img,
    grayscale=True,
    roi_xy=(row_top_res_std, col_left_res_std),
)

_ = show_roi_zoom(
    canvas_std,
    ax_titles=(
        f"Resized Image (standard, {fmt_ms(time_2d_std_fwd)})",
        None,
    ),
    **roi_kwargs_on_canvas_std
)

# %%
# Recovered Images
# ----------------

# %%
# Antialiasing Pipeline
# ~~~~~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    recovered_2d_aa,
    ax_titles=(
        f"Recovered Image (antialiased, {fmt_ms(time_2d_aa_back)})",
        None,
    ),
    **roi_kwargs
)

# %%
# Standard Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~

_ = show_roi_zoom(
    recovered_2d_std,
    ax_titles=(
        f"Recovered Image (standard interpolation, {fmt_ms(time_2d_std_back)})",
        None,
    ),
    **roi_kwargs
)

# %%
# Difference Images
# -----------------

# %%
# Antialiasing
# ~~~~~~~~~~~~
#
# Difference with original image on ROI (SNR/MSE numbers are from the
# central-region metrics computed earlier).

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_aa,
    snr=snr_2d_aa,
    mse=mse_2d_aa,
    roi=roi_rect,
    title_prefix="Difference (antialiasing)",
)

# %%
# Standard Interpolation
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Difference with original image on ROI.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_std,
    snr=snr_2d_std,
    mse=mse_2d_std,
    roi=roi_rect,
    title_prefix="Difference (standard)",
)

# %%
# Performance Comparison
# ----------------------
#
# As a compact summary, we print a table with:
#
# - SNR / MSE on the central region (via border_fraction),
# - total (forward + backward) timing of the interpolation pipeline.
#
# This lets you see the cost/benefit trade-off between
# standard interpolation and antialiased shrink/expand.

methods = [
    ("Standard Interpolation (cubic)",          snr_2d_std, mse_2d_std, time_2d_std),
    ("Antialiasing (cubic shrink, cubic up)",   snr_2d_aa,  mse_2d_aa,  time_2d_aa),
]

header_line = f"{'Method':<40} {'SNR (dB)':>10} {'MSE':>16} {'Time (s)':>12}"
print(header_line)
print("-" * len(header_line))

for name, snr_val, mse_val, t in methods:
    print(
        f"{name:<40} "
        f"{snr_val:>10.2f} "
        f"{mse_val:>16.2e} "
        f"{t:>12.4f}"
    )

# %%
# Least-Squares vs Antialiasing
# -----------------------------
#
# We can also compare the antialiasing pipeline
# against a full **Least-Squares** projection of degree 3 using the low-level
# :func:`resize_degrees` API. We don't change any of the figures above; we only
# print SNR / MSE / time numbers here.
#
# In this particular example, the Least-Squares variant often achieves very
# good metrics (SNR/MSE) and can even look slightly “cleaner” numerically.
# However, in practice we generally recommend the **Antialiasing** preset:
#
# - it is extremely stable and robust across a wide range of zooms and images,
# - it is faster than full Least-Squares,
# - and the visual quality is usually very close.

t0 = time.perf_counter()
resized_2d_ls = resize_degrees(
    input_image_normalized,
    zoom_factors=zoom_factors_2d,
    interp_degree=3,
    analy_degree=3,
    synthe_degree=3,
    inversable=False,
)
t1 = time.perf_counter()
recovered_2d_ls = resize_degrees(
    resized_2d_ls,
    output_size=input_image_normalized.shape,
    interp_degree=3,
    analy_degree=3,
    synthe_degree=3,
    inversable=False,
)
t2 = time.perf_counter()

time_2d_ls_fwd = t1 - t0
time_2d_ls_back = t2 - t1
time_2d_ls = t2 - t0

snr_2d_ls, mse_2d_ls = compute_snr_and_mse_region(
    input_image_normalized,
    recovered_2d_ls,
    border_fraction=border_fraction,
)

methods_ls_vs_aa = [
    ("Antialiasing (cubic shrink, cubic up)", snr_2d_aa, mse_2d_aa, time_2d_aa),
    ("Least-Squares (cubic) shrink+up",       snr_2d_ls, mse_2d_ls, time_2d_ls),
]

header_line_ls = f"{'Method':<40} {'SNR (dB)':>10} {'MSE':>16} {'Time (s)':>12}"
print()
print(header_line_ls)
print("-" * len(header_line_ls))

for name, snr_val, mse_val, t in methods_ls_vs_aa:
    print(
        f"{name:<40} "
        f"{snr_val:>10.2f} "
        f"{mse_val:>16.2e} "
        f"{t:>12.4f}"
    )
