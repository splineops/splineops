"""
Standard Interpolation
======================

Interpolate 2D images with standard interpolation. Compare them to SciPy zoom. We compute SNR and MSE only on a central region 
to exclude boundary artifacts.
"""

# %%
# Imports
# -------

import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

from splineops.utils import (
    resize_and_compute_metrics,      # resampling + metrics
    compute_snr_and_mse_cropped,     # used once later
    plot_resized_image,              # visual helpers
    plot_difference_image,
    show_roi_zoom,
)

# %%
# Pipeline Diagram — TikZ-faithful re-creation in Matplotlib
def draw_pipeline_diagram():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Circle

    # data extents WITHOUT the far-right plus
    XMIN, XMAX = -2.5, 34.8
    YMIN, YMAX = -2.5, 16.0
    ratio = (XMAX - XMIN) / (YMAX - YMIN)  # ~2.027

    width = 12.0
    height = width / ratio

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # ---------- helpers ----------
    def rect(x1, y1, x2, y2, label, fs=12):
        # TikZ gave two opposite corners; convert to lower-left + size
        x_lo, x_hi = min(x1, x2), max(x1, x2)
        y_lo, y_hi = min(y1, y2), max(y1, y2)
        w, h = x_hi - x_lo, y_hi - y_lo
        r = FancyBboxPatch(
            (x_lo, y_lo), w, h,
            boxstyle="round,pad=0.18,rounding_size=0.18",
            linewidth=1.6, edgecolor="black", facecolor="white"
        )
        ax.add_patch(r)
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, ha="center", va="center", fontsize=fs)
        return r

    def circ(x, y, r, label=None, fs=16):
        c = Circle((x, y), r, fill=False, linewidth=1.6, edgecolor="black")
        ax.add_patch(c)
        if label is not None:
            ax.text(x, y, label, ha="center", va="center", fontsize=fs)
        return c

    def dot(x, y, s=4.5):
        ax.plot([x], [y], marker="o", markersize=s, color="black")

    def line(x1, y1, x2, y2, style="solid", lw=1.6, z=2):
        ax.plot([x1, x2], [y1, y2], linestyle=style, linewidth=lw, color="black", zorder=z)

    def arrow(x1, y1, x2, y2, lw=1.6):
        # Use annotate for data→data arrows
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", linewidth=lw, shrinkA=0, shrinkB=0)
        )

    # ---------- nodes (matching your TikZ coordinates) ----------
    # Left: Original
    rect(-2, 14.25, 4.25, 12.5, "Original Image", fs=12)
    arrow(4.25, 13.25, 8, 13.25)

    # Downsample circle
    circ(9, 13.25, 1.0, r"$\downarrow 4$", fs=18)
    arrow(10, 13.25, 14.5, 13.25)

    # Junctions and split to two branches
    dot(12, 13.25)
    line(12, 13.25, 12, 8.25)
    arrow(12, 8.25, 14.5, 8.25)

    # Method boxes
    rect(14.5, 14, 20.75, 12.25, "Standard Interpolation", fs=12)
    rect(14.5, 9, 20.75, 7.25, "SciPy Interpolation", fs=12)

    # Top branch to the right
    line(20.75, 13, 32.25, 13)
    dot(27.25, 13)
    dot(30, 13)
    arrow(30, 13, 30, 11.75)

    # Middle sum node
    circ(30, 10.75, 1.0, r"$\sum$", fs=18)
    line(31, 10.75, 33.5, 10.75)
    arrow(33.5, 10.75, 33.5, 0)

    # Bottom right plumbing
    circ(25, 5.5, 1.0, r"$\sum$", fs=18)
    circ(27.25, 2, 1.0, r"$\sum$", fs=18)
    arrow(25, 4.5, 25, 0)
    arrow(27.25, 1, 27.25, 0)

    # Lower minus node and feed
    circ(27.25, 5.5, 1.0, r"$-$", fs=20)
    arrow(27.25, 13, 27.25, 6.5)
    arrow(27.25, 4.5, 27.25, 3)

    # Middle horizontal from SciPy, with minus and junctions
    line(23.75, 8.25, 32.25, 8.25)
    circ(22.75, 8.25, 1.0, r"$-$", fs=20)
    arrow(20.75, 8.25, 21.75, 8.25)
    dot(25, 8.25)
    dot(30, 8.25)
    arrow(25, 8.25, 25, 6.5)
    arrow(30, 8.25, 30, 9.75)

    # Left vertical trunk feeding two lower rows
    line(5.5, 13.25, 5.5, 2)
    dot(5.5, 13.25)
    dot(5.5, 5.5)
    arrow(5.5, 5.5, 24, 5.5)
    arrow(5.5, 2, 26.25, 2)

    # Dashed separator
    line(10.75, 15.5, 10.75, 0.25, style="dashed", lw=1.2, z=1)

    # Collector box at the bottom
    rect(23.75, -0.25, 34.25, -2, "Difference Images", fs=12)

    fig.tight_layout(pad=0.4)
    plt.show()

draw_pipeline_diagram()


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

# --- Reduce resolution by half on each axis (anti-aliased) ---
#h, w = input_image_normalized.shape
# If the image has odd dims, crop 1 pixel so reshape works cleanly
#h2, w2 = h - (h % 2), w - (w % 2)
#img_cropped = input_image_normalized[:h2, :w2]

# Average 2×2 blocks → halves both H and W
#input_image_normalized = img_cropped.reshape(h2//2, 2, w2//2, 2).mean(axis=(1, 3))
# ----------------------------------------------------------------

zoom_factors_2d = (0.25, 0.25)
border_fraction = 0.3

# We plot the original grayscale image.

#plt.figure(figsize=(6, 5))
#plt.imshow(input_image_normalized, cmap='gray', aspect='equal')
#plt.title("Original Image")
#plt.axis("off")
#plt.show()

# Face-centered 64×64 ROI
ROI_SIZE_PX = 64
FACE_ROW, FACE_COL = 250, 445  # (row, col)

h_img, w_img = input_image_normalized.shape

# Top-left of the 64×64 box, clipped to stay inside the image
row_top = int(np.clip(FACE_ROW - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col_left = int(np.clip(FACE_COL - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))

roi_kwargs = dict(
    roi_height_frac=ROI_SIZE_PX / h_img,  # keeps height at 64 px (square ROI)
    grayscale=True,
    roi_xy=(row_top, col_left),           # top-left of the ROI
)

# Original (shifted ROI)
show_roi_zoom(
    input_image_normalized,
    ax_titles=("Original Image", None),
    **roi_kwargs
)

# %%
# Standard Interpolation
# ----------------------
#
# We use our standard interpolation method.

(
    resized_2d_interp, 
    recovered_2d_interp, 
    snr_2d_interp, 
    mse_2d_interp, 
    time_2d_interp
) = resize_and_compute_metrics(
    input_image_normalized,
    method="cubic",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

# %%
# Recovered Image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after a reversing of the zoom factors.

#plot_recovered_image(recovered_2d_interp)

# Recovered (standard interpolation) – same ROI
show_roi_zoom(
    recovered_2d_interp,
    ax_titles=("Recovered Image (cubic)", None),
    **roi_kwargs
)

# %%
# Resized Image
# ~~~~~~~~~~~~~
#
# We plot the resized image with standard interpolation.

plot_resized_image(
    original=input_image_normalized,
    resized=resized_2d_interp,
    method="cubic",
    zoom_factors=zoom_factors_2d,
    time_elapsed=time_2d_interp
)

# %%
# Difference image
# ~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_interp,
    snr=snr_2d_interp,
    mse=mse_2d_interp
)

# %%
# SciPy Interpolation
# -------------------
#
# For comparison purposes, we also use the SciPy zoom method for resizing.

(
    resized_2d_scipy,
    recovered_2d_scipy,
    snr_2d_scipy,
    mse_2d_scipy,
    time_2d_scipy
) = resize_and_compute_metrics(
    input_image_normalized,
    method="scipy",
    scipy_order=3,
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

# %%
# Recovered Image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after a reversing of the zoom factors.

#plot_recovered_image(recovered_2d_scipy)

# Recovered (SciPy) – same ROI
show_roi_zoom(
    recovered_2d_scipy,
    ax_titles=("Recovered Image (SciPy)", None),
    **roi_kwargs
)

# %%
# Resized Image
# ~~~~~~~~~~~~~
#
# We plot the resized image with SciPy interpolation.

plot_resized_image(
    original=input_image_normalized,
    resized=resized_2d_scipy,
    method="scipy",
    zoom_factors=zoom_factors_2d,
    time_elapsed=time_2d_scipy
)

# %%
# Difference Image
# ~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_scipy,
    snr=snr_2d_scipy,
    mse=mse_2d_scipy
)

# %%
# Difference with SciPy
# ---------------------
#
# Now we compute the difference between the recovered image from the
# standard interpolation and the SciPy interpolation. We also compute
# SNR and MSE on the central region and display them.
# Because they are nearly identical, we conclude that the two interpolation 
# methods produce the same results.

snr_scipy_vs_interp, mse_scipy_vs_interp = compute_snr_and_mse_cropped(
    recovered_2d_scipy, recovered_2d_interp, border_fraction
)

plot_difference_image(
    original=recovered_2d_scipy,
    recovered=recovered_2d_interp,
    snr=snr_scipy_vs_interp,
    mse=mse_scipy_vs_interp
)

# %%
# Alternative using TensorSpline
# ------------------------------
#
# As an alternative, we can replicate the same interpolation manually using the 
# ``TensorSpline`` class, which underpins the `resize()` function behind the scene.

from splineops.interpolate.tensorspline import TensorSpline

# 1) Build uniform coordinate arrays that match the shape of 'input_image_normalized'

height, width = input_image_normalized.shape
x_coords = np.linspace(0, height - 1, height)
y_coords = np.linspace(0, width - 1, width)
coordinates_2d = (x_coords, y_coords)

# 2) For "cubic interpolation", pick "bspline3".
#    For boundary handling, we can pick "mirror", "zero", etc.

ts = TensorSpline(
    data=input_image_normalized,
    coordinates=coordinates_2d,
    bases="bspline3",  # cubic B-splines
    modes="mirror"     # handles boundaries with mirroring
)

# 3) Define new coordinate grids for the "zoomed" shape. 

zoomed_height = int(height * zoom_factors_2d[0])
zoomed_width = int(width * zoom_factors_2d[1])

x_coords_zoomed = np.linspace(0, height - 1, zoomed_height)
y_coords_zoomed = np.linspace(0, width - 1, zoomed_width)
coords_zoomed_2d = (x_coords_zoomed, y_coords_zoomed)

# Evaluate (forward pass): zoom in or out

resized_direct_ts = ts(coordinates=coords_zoomed_2d)

# 4) Define coordinate grids for returning to the original shape
x_coords_orig = np.linspace(0, height - 1, height)
y_coords_orig = np.linspace(0, width - 1, width)
coords_orig_2d = (x_coords_orig, y_coords_orig)

# Evaluate (backward pass): from zoomed shape back to original

ts_zoomed = TensorSpline(
    data=resized_direct_ts,
    coordinates=coords_zoomed_2d,
    bases="bspline3",
    modes="mirror"
)
recovered_direct_ts = ts_zoomed(coordinates=coords_orig_2d)

# Now, resized_direct_ts / recovered_direct_ts should be very similar 
# to 'resized_2d_interp' / 'recovered_2d_interp' from the high-level "resize()" approach.
# Let's compute MSE to confirm:

mse_forward = np.mean((resized_direct_ts - resized_2d_interp) ** 2)
mse_backward = np.mean((recovered_direct_ts - recovered_2d_interp) ** 2)
print(f"MSE (TensorSpline vs. resize()) resized:  {mse_forward:.6e}")
print(f"MSE (TensorSpline vs. resize()) recovered: {mse_backward:.6e}")