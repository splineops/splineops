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

plot_recovered_image(recovered_2d_interp)

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

plot_recovered_image(recovered_2d_scipy)

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

"""
Zoomed Region Inspection (grayscale)
------------------------------------

* Load Kodak “kodim14” → grayscale float • [0,1]
* Down-sample with cubic B-spline (splineops.resize standard mode)
* Draw a red square ROI on the low-res image
* Magnify that ROI with nearest-neighbor so its display height matches the low-res image
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests, io
from PIL import Image
from splineops.utils import resize_multichannel   # cubic interpolation helper

# ── parameters ────────────────────────────────────────────────────────────────
IMG_URL          = "https://r0k.us/graphics/kodak/kodak/kodim14.png"
DOWNSAMPLE       = 0.25      # keep 25 %
ROI_FRAC_HEIGHT  = 1 / 3     # ROI height as fraction of low-res height
# ──────────────────────────────────────────────────────────────────────────────

# 1) fetch image → grayscale float64
rgb = Image.open(io.BytesIO(requests.get(IMG_URL, timeout=10).content))
rgb_np = np.asarray(rgb, dtype=np.float64) / 255.0            # H×W×3
gray  = (0.2989*rgb_np[...,0] + 0.5870*rgb_np[...,1] + 0.1140*rgb_np[...,2])
gray  = gray[..., None]                                        # H×W×1 (for resize)

# 2) cubic down-sample (mirror boundaries)
low_res = resize_multichannel(gray, DOWNSAMPLE,
                              method="cubic", modes="mirror").squeeze(-1)  # H×W uint8
h_lr, w_lr = low_res.shape

# 3) pick square ROI whose side divides h_lr  → integer magnification
roi_size = max(1, int(h_lr * ROI_FRAC_HEIGHT))
while h_lr % roi_size:
    roi_size -= 1
row0 = h_lr // 2 - roi_size // 2
col0 = w_lr // 2 - roi_size // 2
roi  = low_res[row0:row0+roi_size, col0:col0+roi_size]

# 4) enlarge ROI with nearest-neighbor so height == h_lr
mag     = h_lr // roi_size
roi_big = np.repeat(np.repeat(roi, mag, 0), mag, 1)

# 5) plot – width ratios proportional to pixel widths → equal heights
fig, ax = plt.subplots(
    1, 2,
    figsize=(10, h_lr / 20),
    gridspec_kw={"width_ratios": [w_lr, roi_big.shape[1]]}
)

# left: low-res image with ROI
ax[0].imshow(low_res, cmap="gray")
ax[0].add_patch(patches.Rectangle((col0, row0), roi_size, roi_size,
                                  linewidth=2, edgecolor="red", facecolor="none"))
ax[0].set_title("Down-sampled (cubic)")
ax[0].axis("off"); ax[0].set_aspect("equal")

# right: magnified ROI
ax[1].imshow(roi_big, cmap="gray", interpolation="nearest")
ax[1].set_title(f"ROI ×{mag} (nearest)")
ax[1].axis("off"); ax[1].set_aspect("equal")

plt.tight_layout()
plt.show()
