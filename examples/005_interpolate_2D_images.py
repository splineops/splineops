"""
Interpolate 2D images
=====================

Interpolate 2D images with standard interpolation, least-squares and oblique projection,
comparing them to SciPy's zoom. We compute SNR and MSE only on a central region 
to exclude boundary artifacts.

You can download this example at the tab at right, as both a Python script
and as a Jupyter notebook.
"""

# %%
# Import required libraries
# -------------------------
#
# We import the required libraries, including NumPy for numerical computations,
# Matplotlib for plotting, and the custom `resize` function from the `splineops` package.

import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from scipy.ndimage import zoom  # For SciPy's zoom comparison
from splineops.interpolate.resize import resize  # Unified resize function
import time

# %%
# Helper functions
# ----------------
#
# We define:
#   - a utility to crop out ~20% borders around the image
#   - SNR and MSE on that central cropped area
#   - resizing functions

def crop_to_central_region(image, border_fraction):
    """
    Return a central sub-region of 'image', skipping 'border_fraction'
    of the width/height on all sides.
    """
    H, W = image.shape
    top = int(H * border_fraction)
    bottom = int(H * (1 - border_fraction))
    left = int(W * border_fraction)
    right = int(W * (1 - border_fraction))
    # Guard against degenerate cases
    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, H)
    right = min(right, W)
    return image[top:bottom, left:right]

def compute_snr_and_mse_cropped(original, processed, border_fraction):
    """
    Compute SNR and MSE on the 'central' cropped area, ignoring border_fraction
    of the image on each side.
    """
    # Crop both images consistently
    orig_cropped = crop_to_central_region(original, border_fraction)
    proc_cropped = crop_to_central_region(processed, border_fraction)

    # Now compute SNR and MSE on that region
    signal_power = np.mean(orig_cropped**2)
    noise_power = np.mean((orig_cropped - proc_cropped)**2)
    mse_val = noise_power

    if noise_power <= 1e-30:  # near-zero difference
        snr_val = float('inf')
    else:
        snr_val = 10 * np.log10(signal_power / noise_power)

    return snr_val, mse_val

def resize_with_scipy_zoom(input_image, zoom_factors, degree, border_fraction):
    """
    Resize using SciPy's zoom, then resize back and compute SNR/MSE
    *only on a central region* to avoid boundary artifacts.
    """
    start_time = time.perf_counter()
    resized_image = zoom(input_image, zoom_factors, order=degree)
    time_elapsed = time.perf_counter() - start_time

    reverse_zoom_factors = 1.0 / np.array(zoom_factors)
    resized_back_image = zoom(resized_image, reverse_zoom_factors, order=degree)

    snr = 0.0
    mse = 0.0
    # Compute SNR/MSE on central region
    snr, mse = compute_snr_and_mse_cropped(input_image, resized_back_image, border_fraction)

    return resized_image, resized_back_image, snr, mse, time_elapsed

def resize_and_compute_metrics(input_image, method, degree, zoom_factors, border_fraction):
    """
    Resize a 2D image using the specified method, then resize back
    to original size and compute SNR, MSE, and timing *only on a central region*.
    """
    if np.isscalar(zoom_factors):
        zoom_factors = (zoom_factors, zoom_factors)

    if method == "scipy":
        return resize_with_scipy_zoom(
            input_image, zoom_factors, degree, border_fraction
        )
    else:
        start_time = time.perf_counter()
        resized_image = resize(
            data=input_image,
            zoom_factors=zoom_factors,
            degree=degree,
            method=method
        )
        time_elapsed = time.perf_counter() - start_time

        # Resize back to original shape:
        original_shape = input_image.shape
        resized_back_image = resize(
            data=resized_image,
            output_size=original_shape,
            degree=degree,
            method=method
        )

        # Compute SNR/MSE on central region
        snr, mse = compute_snr_and_mse_cropped(input_image, resized_back_image, border_fraction)

        return resized_image, resized_back_image, snr, mse, time_elapsed

# %%
# Plotting function (vertical subplots)
# -------------------------------------
#
# We display three images in one column:
# (1) Original, (2) Resized, (3) Difference (Original - ResizedBack).
# If zoom < 1, we embed the resized image on a white canvas matching original's shape.

def plot_2d_results(original, resized, resized_back, method, zoom_factors, snr, mse, time_elapsed):
    """
    Display three vertical 2D images: original, resized, and difference (original - resized_back).
    If any zoom factor < 1, we place the resized image on a white canvas matching the original shape.

    SNR/MSE are computed only on the central region (by the prior functions).
    """
    difference = original - resized_back
    zoom_out = any(zf < 1.0 for zf in zoom_factors)

    def to_uint8(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr_scaled = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr_scaled = arr * 0.0
        return (arr_scaled * 255).astype(np.uint8)

    orig_8 = to_uint8(original)
    resized_8 = to_uint8(resized)
    diff_8 = to_uint8(difference)

    # White canvas if zoomed out
    if zoom_out:
        canvas_8 = np.full_like(orig_8, 255, dtype=np.uint8)  # white
        rh, rw = resized_8.shape
        canvas_8[:rh, :rw] = resized_8
        resized_display = canvas_8
    else:
        resized_display = resized_8

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 14))

    # Original
    axes[0].imshow(orig_8, cmap='gray', aspect='equal')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Resized
    axes[1].imshow(resized_display, cmap='gray', aspect='equal')
    axes[1].set_title(
        f"{method.capitalize()} Resized\n"
        f"Zoom: {zoom_factors}, Time: {time_elapsed:.4f}s"
    )
    axes[1].axis("off")

    # Difference
    axes[2].imshow(diff_8, cmap='gray', aspect='equal')
    axes[2].set_title(
        f"Difference\n"
        f"SNR: {snr:.2f} dB, MSE: {mse:.2e}"
    )
    axes[2].axis("off")

    plt.tight_layout(pad=3.0)  # Increase padding between subplots
    plt.show()

# %%
# Load and normalize a 2D image
# -----------------------------
#
# Here, we load an example Kodak image from an online repository.
# We convert it to grayscale in [0..1].

url = 'https://r0k.us/graphics/kodak/kodak/kodim23.png'
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

degree = 3
zoom_factors_2d = (0.25, 0.25)
border_fraction = 0.3

# %%
# 2D resizing: interpolation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

(
    resized_2d_interp, 
    resized_back_2d_interp, 
    snr_2d_interp, 
    mse_2d_interp, 
    time_2d_interp
) = resize_and_compute_metrics(
    input_image_normalized,
    method="interpolation",
    degree=degree,
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

plot_2d_results(
    original=input_image_normalized,
    resized=resized_2d_interp,
    resized_back=resized_back_2d_interp,
    method="interpolation",
    zoom_factors=zoom_factors_2d,
    snr=snr_2d_interp,
    mse=mse_2d_interp,
    time_elapsed=time_2d_interp
)

# %%
# 2D resizing: least-squares
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

(
    resized_2d_ls,
    resized_back_2d_ls,
    snr_2d_ls,
    mse_2d_ls,
    time_2d_ls
) = resize_and_compute_metrics(
    input_image_normalized,
    method="least-squares",
    degree=degree,
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

plot_2d_results(
    original=input_image_normalized,
    resized=resized_2d_ls,
    resized_back=resized_back_2d_ls,
    method="least-squares",
    zoom_factors=zoom_factors_2d,
    snr=snr_2d_ls,
    mse=mse_2d_ls,
    time_elapsed=time_2d_ls
)

# %%
# 2D resizing: oblique
# ~~~~~~~~~~~~~~~~~~~~

(
    resized_2d_ob,
    resized_back_2d_ob,
    snr_2d_ob,
    mse_2d_ob,
    time_2d_ob
) = resize_and_compute_metrics(
    input_image_normalized,
    method="oblique",
    degree=degree,
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

plot_2d_results(
    original=input_image_normalized,
    resized=resized_2d_ob,
    resized_back=resized_back_2d_ob,
    method="oblique",
    zoom_factors=zoom_factors_2d,
    snr=snr_2d_ob,
    mse=mse_2d_ob,
    time_elapsed=time_2d_ob
)

# %%
# 2D resizing: SciPy interpolation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(
    resized_2d_scipy,
    resized_back_2d_scipy,
    snr_2d_scipy,
    mse_2d_scipy,
    time_2d_scipy
) = resize_and_compute_metrics(
    input_image_normalized,
    method="scipy",
    degree=degree,
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

plot_2d_results(
    original=input_image_normalized,
    resized=resized_2d_scipy,
    resized_back=resized_back_2d_scipy,
    method="scipy",
    zoom_factors=zoom_factors_2d,
    snr=snr_2d_scipy,
    mse=mse_2d_scipy,
    time_elapsed=time_2d_scipy
)
