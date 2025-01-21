"""
Interpolate 2D images
=====================

Interpolate 2D images with standard interpolation, least-squares and oblique projection,
comparing them to SciPy's zoom.

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
# Compute Signal-to-Noise Ratio (SNR), Mean Squared Error (MSE),
# and perform resizing with various methods.

def compute_snr(original, processed):
    """Compute Signal-to-Noise Ratio (dB) between two 2D signals."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - processed) ** 2)
    if noise_power == 0:
        return np.inf  # Perfect match
    return 10 * np.log10(signal_power / noise_power)

def compute_mse(original, processed):
    """Compute Mean Squared Error between two 2D signals."""
    return np.mean((original - processed) ** 2)

def resize_with_scipy_zoom(input_image, zoom_factors, degree):
    """
    Resize using SciPy's zoom, then resize back and compute SNR/MSE.
    Returns (resized_image, resized_back_image, snr, mse, time_elapsed).
    """
    start_time = time.perf_counter()
    resized_image = zoom(input_image, zoom_factors, order=degree)
    time_elapsed = time.perf_counter() - start_time

    reverse_zoom_factors = 1.0 / np.array(zoom_factors)
    resized_back_image = zoom(resized_image, reverse_zoom_factors, order=degree)

    snr = compute_snr(input_image, resized_back_image)
    mse = compute_mse(input_image, resized_back_image)
    return resized_image, resized_back_image, snr, mse, time_elapsed

def resize_and_compute_metrics(input_image, method, degree, zoom_factors):
    """
    Resize a 2D image using the specified method, then resize back
    to original size and compute SNR, MSE, and timing.

    Returns (resized_image, resized_back_image, snr, mse, time_elapsed).
    """
    if np.isscalar(zoom_factors):
        zoom_factors = (zoom_factors, zoom_factors)

    if method == "scipy":
        return resize_with_scipy_zoom(
            input_image, zoom_factors, degree
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

        snr = compute_snr(input_image, resized_back_image)
        mse = compute_mse(input_image, resized_back_image)
        return resized_image, resized_back_image, snr, mse, time_elapsed

# %%
# Plotting function (vertical subplots)
# -------------------------------------
#
# We define a simpler plotting function that displays three images in one column:
# (1) Original, (2) Resized, (3) Difference (Original - ResizedBack).

def plot_2d_results(original, resized, resized_back, method, zoom_factors, snr, mse, time_elapsed):
    """
    Display three vertical 2D images: original, resized, and difference (original - resized_back).
    If any zoom factor < 1, we place the resized image on a white canvas matching the original shape.

    Parameters
    ----------
    original : np.ndarray
        2D array of the original image data (e.g., floats in [0..1]).
    resized : np.ndarray
        2D array of the resized image (floats in [0..1]).
    resized_back : np.ndarray
        2D array of the resized-back image, same shape as `original`.
    method : str
        Resizing method ("interpolation", "least-squares", "oblique", or "scipy").
    zoom_factors : tuple
        The (zoom_y, zoom_x) factors used.
    snr : float
        Computed SNR in dB.
    mse : float
        Computed mean squared error.
    time_elapsed : float
        Elapsed time for the forward resizing step.
    """
    # 1) Compute the difference for the bottom subplot
    difference = original - resized_back

    # 2) Decide if we need a white-canvas approach for the "resized" image
    #    That is, if either dimension was zoomed out (< 1), we'll embed it in a white background
    zoom_out = any(zf < 1.0 for zf in zoom_factors)

    # 3) Convert arrays to [0..255] for display
    def to_uint8(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr_scaled = (arr - arr_min) / (arr_max - arr_min)
        else:
            # constant image
            arr_scaled = arr * 0.0
        return (arr_scaled * 255).astype(np.uint8)

    orig_8 = to_uint8(original)
    resized_8 = to_uint8(resized)
    diff_8 = to_uint8(difference)

    # 4) If zoom_out is True, place `resized_8` in a white canvas
    if zoom_out:
        canvas_8 = np.ones_like(orig_8, dtype=np.uint8) * 255  # white background
        rh, rw = resized_8.shape
        # Place top-left corner at (0,0):
        canvas_8[:rh, :rw] = resized_8
        resized_display = canvas_8
    else:
        resized_display = resized_8

    # 5) Make vertical subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6, 14))

    # --- (a) Original ---
    axes[0].imshow(orig_8, cmap='gray', aspect='equal')
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # --- (b) Resized (possibly on white canvas) ---
    axes[1].imshow(resized_display, cmap='gray', aspect='equal')
    axes[1].set_title(
        f"{method.capitalize()} Resized\n"
        f"Zoom: {zoom_factors}\n"
        f"Time: {time_elapsed:.4f}s"
    )
    axes[1].axis("off")

    # --- (c) Difference (original - resizedBack) ---
    axes[2].imshow(diff_8, cmap='gray', aspect='equal')
    axes[2].set_title(
        f"Difference (Original - ResizedBack)\n"
        f"SNR: {snr:.2f} dB, MSE: {mse:.2e}"
    )
    axes[2].axis("off")

    plt.tight_layout()
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

plt.imshow(input_image_normalized, cmap='gray')
plt.title("Original 2D Image (Grayscale)")
plt.axis('off')
plt.show()

# %%
# 2D resizing: interpolation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

degree = 3
zoom_factors_2d = (0.25, 0.25)

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
    zoom_factors=zoom_factors_2d
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
    zoom_factors=zoom_factors_2d
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
    zoom_factors=zoom_factors_2d
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
    zoom_factors=zoom_factors_2d
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
