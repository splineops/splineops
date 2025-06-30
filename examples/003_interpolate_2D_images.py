"""
Interpolate 2D Images
=====================

Interpolate 2D images with standard interpolation, least-squares, and oblique projection.
Compare them to SciPy zoom. We compute SNR and MSE only on a central region 
to exclude boundary artifacts. A summary of the cost/benefit tradeoff of the three methods
is provided at the bottom of this page.

You can download this example at the tab at right (Python script or Jupyter notebook.
"""

# Required Libraries
# ------------------

import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

from splineops.utils import (
    resize_and_compute_metrics,      # resampling + metrics
    compute_snr_and_mse_cropped,     # used once later
    plot_resized_image,              # visual helpers
    plot_recovered_image,
    plot_difference_image,
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

# We plot the original grayscale image.

plt.figure(figsize=(6, 5))
plt.imshow(input_image_normalized, cmap='gray', aspect='equal')
plt.title("Original Image")
plt.axis("off")
plt.show()

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
# Recovered Image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after a reversing of the zoom factors.

plot_recovered_image(recovered_2d_scipy)

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
# Trivial Interpolation
# ---------------------
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
# Recovered Image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after a reversing of the zoom factors.

plot_recovered_image(recovered_2d_interp)

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
# Difference with SciPy
# ~~~~~~~~~~~~~~~~~~~~~
#
# Now we compute the difference between the recovered image from the
# trivial interpolation and the SciPy interpolation. We also compute
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
# Alternative with TensorSpline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

# %%
# Least-Squares Projection
# ------------------------
#
# We use the least-squares projection method.

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
    border_fraction=border_fraction
)

# %%
# Resized Image
# ~~~~~~~~~~~~~
#
# We plot the resized image with least-squares projection method.

plot_resized_image(
    original=input_image_normalized,
    resized=resized_2d_ls,
    method="cubic-best_antialiasing",
    zoom_factors=zoom_factors_2d,
    time_elapsed=time_2d_ls
)

# %%
# Recovered Image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after reversing zoom factors.

plot_recovered_image(recovered_2d_ls)

# %%
# Difference Image
# ~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ls,
    snr=snr_2d_ls,
    mse=mse_2d_ls
)

# %%
# Oblique Projection
# ------------------
#
# We use the oblique-projection method.

(
    resized_2d_ob,
    recovered_2d_ob,
    snr_2d_ob,
    mse_2d_ob,
    time_2d_ob
) = resize_and_compute_metrics(
    input_image_normalized,
    method="cubic-fast_antialiasing",
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

# %%
# Resized Image
# ~~~~~~~~~~~~~
#
# We plot the resized image with the oblique-projection method.

plot_resized_image(
    original=input_image_normalized,
    resized=resized_2d_ob,
    method="cubic-fast_antialiasing",
    zoom_factors=zoom_factors_2d,
    time_elapsed=time_2d_ob
)

# %%
# Recovered Image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after a reversing of the zoom factors.

plot_recovered_image(recovered_2d_ob)

# %%
# Difference Image
# ~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ob,
    snr=snr_2d_ob,
    mse=mse_2d_ob
)

# %%
# Comparison
# ----------
#
# We compare the performance of the different methods being analyzed.

# %%
# Comparison Table
# ~~~~~~~~~~~~~~~~
#
# We print the SNR, MSE, and timing data for each method.

methods = [
    ("SciPy Interpolation", snr_2d_scipy, mse_2d_scipy, time_2d_scipy),
    ("Trivial Interpolation", snr_2d_interp, mse_2d_interp, time_2d_interp),
    ("Least-Squares Projection", snr_2d_ls, mse_2d_ls, time_2d_ls),
    ("Oblique Projection", snr_2d_ob, mse_2d_ob, time_2d_ob),
]

# Print the table header using the same widths as we'll use for data
header_line = f"{'Method':<25} {'SNR (dB)':>10} {'MSE':>16} {'Time (s)':>12}"
print(header_line)
print("-" * len(header_line))  # or manually set a dash length, e.g. 67

# Now print each row with the matching format
for method_name, snr_val, mse_val, time_val in methods:
    row_line = f"{method_name:<25} {snr_val:>10.2f} {mse_val:>16.2e} {time_val:>12.4f}"
    print(row_line)

# %%
# Side-by-side (2×2) view of the recovered images
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

recovered_stack = [
    ("SciPy",                recovered_2d_scipy,  snr_2d_scipy,  mse_2d_scipy),
    ("Cubic (interp)",       recovered_2d_interp, snr_2d_interp, mse_2d_interp),
    ("Least-Squares (best)", recovered_2d_ls,     snr_2d_ls,     mse_2d_ls),
    ("Oblique (fast AA)",    recovered_2d_ob,     snr_2d_ob,     mse_2d_ob),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.ravel()   # flatten to 1-D iterator

for ax, (label, img, snr_val, mse_val) in zip(axes, recovered_stack):
    ax.imshow(img, cmap="gray", aspect="equal")
    ax.set_title(f"{label}\nSNR {snr_val:.1f} dB  ·  MSE {mse_val:.2e}")
    ax.axis("off")

plt.tight_layout()
plt.show()
