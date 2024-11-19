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

# Normalize the image
input_image_normalized = (input_image / 255.0).astype(np.float64)

# Display the original image
plt.imshow(input_image_normalized, cmap='gray')
plt.title("Original MRI Image")
plt.axis('off')
plt.show()

# %%
# Helper Functions for Metric Calculation
# ---------------------------------------
#
# Define functions to compute Signal-to-Noise Ratio (SNR) and Mean Squared Error (MSE).

def compute_snr(original, processed):
    signal_power = 1.0 ** 2  # Since the image is normalized to [0, 1]
    noise_power = np.mean((original - processed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def compute_mse(original, processed):
    return np.mean((original - processed) ** 2)

def resize_with_scipy_zoom(input_signal, zoom_factors, degree):
    """Resize using SciPy's zoom, then resize back and compute metrics."""
    resized_signal = zoom(input_signal, zoom_factors, order=degree)
    reverse_zoom_factors = 1.0 / np.array(zoom_factors)
    resized_back_signal = zoom(resized_signal, reverse_zoom_factors, order=degree)

    snr = compute_snr(input_signal, resized_back_signal)
    mse = compute_mse(input_signal, resized_back_signal)

    return resized_signal, resized_back_signal, snr, mse

def resize_and_compute_metrics(input_image, method, degree, zoom_factors):
    """Resize an image using a given method and compute metrics."""
    if np.isscalar(zoom_factors):
        zoom_factors = [zoom_factors] * len(input_image.shape)

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

def plot_results(
    original_slice, 
    resized_slice, 
    resized_back_slice, 
    method, 
    zoom_factors, 
    snr, 
    mse
):
    """
    Generalized plot function for 2D and 3D data.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Check if all zoom factors are less than 1 (zooming out in all dimensions)
    zoom_out = all(zf < 1 for zf in zoom_factors)

    # Prepare resized display array
    if zoom_out:
        # Create a black background of the original slice's shape
        resized_display = np.zeros_like(original_slice)
        # Place the resized slice in the top-left corner of the black background
        resized_display[
            tuple(slice(0, r_dim) for r_dim in resized_slice.shape)
        ] = resized_slice
    else:
        # Just use the resized slice as is
        resized_display = resized_slice

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    # Original slice
    ax[0].imshow(original_slice, cmap="gray", aspect='auto')
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Resized slice
    ax[1].imshow(resized_display, cmap="gray", aspect='auto')
    ax[1].set_title(f"{method.capitalize()} Resized (Zoom: {zoom_factors})")
    ax[1].axis("off")

    # Difference map
    difference = original_slice - resized_back_slice
    ax[2].imshow(difference, cmap="gray", aspect='auto')
    ax[2].set_title(f"Difference (SNR: {snr:.2f} dB, MSE: {mse:.2e})")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

# %%
# Define Parameters for Resizing
# ------------------------------
#
# Set the zoom factors, degree of interpolation, and the methods to be compared.

degree = 3
zoom_factors = (0.5, 0.5)
methods = ["interpolation", "least-squares", "oblique", "scipy"]

# %%
# Resizing and Comparing Methods
# ------------------------------
#
# Iterate over the different methods and perform resizing, compute metrics, and plot results.

for method in methods:
    if method == "scipy":
        resized_signal, resized_back_signal, snr, mse = resize_with_scipy_zoom(
            input_signal=input_image_normalized, 
            zoom_factors=zoom_factors, 
            degree=degree
        )
    else:
        resized_signal, resized_back_signal, snr, mse = resize_and_compute_metrics(
            input_image_normalized, method, degree, zoom_factors
        )

    # Plot results
    plot_results(
        original_slice=input_image_normalized,
        resized_slice=resized_signal,
        resized_back_slice=resized_back_signal,
        method=method,
        zoom_factors=zoom_factors,
        snr=snr,
        mse=mse
    )
