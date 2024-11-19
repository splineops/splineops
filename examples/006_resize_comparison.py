"""
Comparison of Resizing Methods with TensorSpline and Advanced Techniques
=======================================================================

This example compares TensorSpline resizing with advanced interpolation methods:
Least-Squares, Oblique Projection, and SciPy's built-in zoom.
"""

# %%
# Load MRI Image and Normalize
# ----------------------------
#
# We load the MRI head image and normalize it to the range [0, 1].

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from splineops.interpolate.resize import resize
from splineops.utils.image_loader import load_head_mri_image  # Import the MRI loader

# Load the MRI image
input_image = load_head_mri_image()

# Helper Functions for Metric Calculation
# ----------------------------------------
def compute_snr(original, processed):
    signal_power = 1.0 ** 2
    noise_power = np.mean((original - processed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def compute_mse(original, processed):
    return np.mean((original - processed) ** 2)

def create_black_background(image, original_shape):
    black_background = np.zeros(original_shape)
    black_background[:image.shape[0], :image.shape[1]] = image
    return black_background

def resize_with_scipy_zoom(input_signal, zoom_factors, degree):
    """Resize using SciPy's zoom, then resize back and compute metrics."""
    resized_signal = zoom(input_signal, zoom_factors, order=degree)
    reverse_zoom_factors = 1.0 / np.array(zoom_factors)
    resized_back_signal = zoom(resized_signal, reverse_zoom_factors, order=degree)

    snr = compute_snr(input_signal, resized_back_signal)
    mse = compute_mse(input_signal, resized_back_signal)

    return resized_signal, resized_back_signal, snr, mse

def resize_and_compute_metrics(input_image, method, degree, zoom_factor):
    """Resize an image using a given method and compute metrics."""
    # Ensure zoom_factors is an array
    if np.isscalar(zoom_factor):
        zoom_factors = [zoom_factor] * len(input_image.shape)

    if method == "scipy":
        resized_signal, resized_back_signal, snr, mse = resize_with_scipy_zoom(
            input_signal=input_image,
            zoom_factors=zoom_factors,
            degree=degree
        )
    else:
        resized_signal = resize(
            data=input_image,
            zoom_factors=zoom_factors,
            degree=degree,
            method=method
        )
        resized_back_signal = resize(
            data=resized_signal,
            output_size=input_image.shape,
            degree=degree,
            method=method
        )
        snr = compute_snr(input_image, resized_back_signal)
        mse = compute_mse(input_image, resized_back_signal)

    return resized_signal, resized_back_signal, snr, mse

def plot_2d_results(input_image, resized_signal_display, resized_back_signal, method, zoom_factor, snr, mse, normalized_input_image):
    """
    Display the original image, resized image, and difference map.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    ax[0].imshow(input_image, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Resized image with optional black background for zoom out
    if zoom_factor < 1:
        resized_display_with_background = create_black_background(resized_signal_display, input_image.shape)
    else:
        resized_display_with_background = resized_signal_display

    ax[1].imshow(resized_display_with_background, cmap="gray")
    ax[1].set_title(f"{method.capitalize()} Resized (Zoom: {zoom_factor}x)")
    ax[1].axis("off")

    # Difference map
    diff_map = np.clip(np.abs(normalized_input_image - resized_back_signal), 0, 1)
    ax[2].imshow(diff_map, cmap="gray")
    ax[2].set_title(f"Difference (SNR: {snr:.2f} dB, MSE: {mse:.2e})")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

# Normalize the image
input_image_normalized = (input_image / 255.0).astype(np.float64)

# Define parameters
zoom_factor = 0.5  # Resize factor
degree = 3
methods = ["interpolation", "least-squares", "oblique", "scipy"]

# %%
# TensorSpline: Interpolation Method
# -----------------------------------
method = "interpolation"

resized_signal, resized_back_signal, snr, mse = resize_and_compute_metrics(
    input_image_normalized, method, degree, zoom_factor
)

# Convert results back to [0, 255] for visualization
resized_signal_display = np.clip(resized_signal * 255.0, 0, 255).astype(np.uint8)

# Display results
plot_2d_results(
    input_image=input_image,
    resized_signal_display=resized_signal_display,
    resized_back_signal=resized_back_signal,
    method=method,
    zoom_factor=zoom_factor,
    snr=snr,
    mse=mse,
    normalized_input_image=input_image_normalized
)

# %%
# TensorSpline: Least-Squares Method
# -----------------------------------
method = "least-squares"

resized_signal, resized_back_signal, snr, mse = resize_and_compute_metrics(
    input_image_normalized, method, degree, zoom_factor
)

# Convert results back to [0, 255] for visualization
resized_signal_display = np.clip(resized_signal * 255.0, 0, 255).astype(np.uint8)

# Display results
plot_2d_results(
    input_image=input_image,
    resized_signal_display=resized_signal_display,
    resized_back_signal=resized_back_signal,
    method=method,
    zoom_factor=zoom_factor,
    snr=snr,
    mse=mse,
    normalized_input_image=input_image_normalized
)

# %%
# TensorSpline: Oblique Projection Method
# ---------------------------------------
method = "oblique"

resized_signal, resized_back_signal, snr, mse = resize_and_compute_metrics(
    input_image_normalized, method, degree, zoom_factor
)

# Convert results back to [0, 255] for visualization
resized_signal_display = np.clip(resized_signal * 255.0, 0, 255).astype(np.uint8)

# Display results
plot_2d_results(
    input_image=input_image,
    resized_signal_display=resized_signal_display,
    resized_back_signal=resized_back_signal,
    method=method,
    zoom_factor=zoom_factor,
    snr=snr,
    mse=mse,
    normalized_input_image=input_image_normalized
)

# %%
# SciPy Zoom Resizing
# --------------------
resized_signal, resized_back_signal, snr, mse = resize_with_scipy_zoom(
    input_signal=input_image_normalized, 
    zoom_factors=(zoom_factor, zoom_factor), 
    degree=degree
)

# Convert results back to [0, 255] for visualization
resized_signal_display = np.clip(resized_signal * 255.0, 0, 255).astype(np.uint8)

# Display results
plot_2d_results(
    input_image=input_image,
    resized_signal_display=resized_signal_display,
    resized_back_signal=resized_back_signal,
    method="scipy",
    zoom_factor=zoom_factor,
    snr=snr,
    mse=mse,
    normalized_input_image=input_image_normalized
)
