"""
Comparison of Resizing Methods with TensorSpline and Advanced Techniques
=======================================================================

This example compares TensorSpline resizing with advanced interpolation methods:
Least-Squares and Oblique Projection, as well as SciPy's built-in zoom.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.datasets import ascent
from scipy.ndimage import zoom
from splineops.interpolate.ls_oblique.ls_oblique_resize import ls_oblique_resize

# %%
# Helper Functions
# ----------------
#
# Define utility functions to load, process, and evaluate images.

def create_square_image():
    """Create a simple 10x10 image with a white square at the center."""
    img = np.zeros((10, 10))
    img[3:7, 3:7] = 255.0
    return img

def load_ascent_image():
    """Load and resize the 'ascent' image from SciPy datasets."""
    img = ascent()
    img_resized = zoom(img, (256 / img.shape[0], 256 / img.shape[1]), order=3)
    return img_resized

def compute_snr(original, processed):
    """Compute Signal-to-Noise Ratio between two images."""
    signal_power = 255.0 ** 2
    noise_power = np.mean((original - processed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_mse(original, processed):
    """Compute Mean Squared Error between two images."""
    mse = np.mean((original - processed) ** 2)
    return mse

def resize_and_compute_metrics(input_image, method, interpolation, zoom_factor):
    """Resize an image using a specified method, compute SNR and MSE after resizing back."""
    # Normalize input image to [0, 1] range for resizing
    input_image_normalized = (input_image / 255.0).astype(np.float64)

    # Resize (shrink) the image
    shrunken_image = ls_oblique_resize(
        input_img_normalized=input_image_normalized,
        zoom_factors=(zoom_factor, zoom_factor),
        method=method,
        interpolation=interpolation,
        inversable=False
    )

    # Resize (expand) back to original size
    expanded_image = ls_oblique_resize(
        input_img_normalized=shrunken_image,
        output_size=input_image_normalized.shape,
        method=method,
        interpolation=interpolation,
        inversable=False
    )

    # Calculate SNR and MSE
    snr = compute_snr(input_image_normalized, expanded_image)
    mse = compute_mse(input_image_normalized, expanded_image)

    # Convert images back to [0, 255] range for display
    shrunken_image_display = np.clip(shrunken_image * 255.0, 0, 255)
    expanded_image_display = np.clip(expanded_image * 255.0, 0, 255)

    return shrunken_image_display, expanded_image_display, snr, mse

def resize_with_scipy_zoom(input_image_normalized, zoom_factor, interpolation):
    """Resize an image with SciPy's zoom function, then resize back and compute metrics."""
    degree = {'linear': 1, 'quadratic': 2, 'cubic': 3}[interpolation]
    resized_image = zoom(input_image_normalized, (zoom_factor, zoom_factor), order=degree)
    reverse_zoom_factor = 1.0 / zoom_factor
    resized_reverse_image = zoom(resized_image, (reverse_zoom_factor, reverse_zoom_factor), order=degree)

    zoom_factors = (input_image_normalized.shape[0] / resized_reverse_image.shape[0],
                    input_image_normalized.shape[1] / resized_reverse_image.shape[1])
    resized_reverse_output = zoom(resized_reverse_image, zoom_factors, order=degree)

    # Calculate metrics
    snr = compute_snr(input_image_normalized, resized_reverse_output)
    mse = compute_mse(input_image_normalized, resized_reverse_output)

    # Convert images to [0, 255] range for display
    resized_image_display = np.clip(resized_image * 255.0, 0, 255)
    resized_reverse_output_display = np.clip(resized_reverse_output * 255.0, 0, 255)

    return resized_image_display, resized_reverse_output_display, snr, mse

def create_black_background(image, original_shape):
    """Embed resized image on a black background of the original image size."""
    black_background = np.zeros(original_shape)
    black_background[:image.shape[0], :image.shape[1]] = image
    return black_background

# %%
# Load Image and Set Parameters
# -----------------------------
# Load the 'ascent' image and set parameters for the resizing methods.

input_image = load_ascent_image()  # Load and resize ascent image
input_image_normalized = (input_image / 255.0).astype(np.float64)  # Normalize to [0, 1]

zoom_factor = 1 / 3.14
methods = ["least-Squares", "oblique"]
interpolation_type = "cubic"

# %%
# Run Resizing and Compare Results
# --------------------------------
# For each method, resize the image using TensorSpline and SciPy, then compute and display metrics.

for method in methods:
    # Measure and process with advanced method
    start_time = time.time()
    ls_output, ls_resized_reverse, ls_snr, ls_mse = resize_and_compute_metrics(
        input_image, method, interpolation_type, zoom_factor
    )
    ls_time = time.time() - start_time

    # Process with SciPy zoom
    start_time = time.time()
    scipy_output, scipy_resized_reverse, scipy_snr, scipy_mse = resize_with_scipy_zoom(
        input_image_normalized, zoom_factor, interpolation_type
    )
    scipy_time = time.time() - start_time

    # Display and analyze results
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    ax[0, 0].imshow(input_image, cmap="gray")
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")

    interp_degree = {'linear': 1, 'quadratic': 2, 'cubic': 3}[interpolation_type]

    ax[0, 1].imshow(create_black_background(ls_output, input_image.shape), cmap="gray")
    ax[0, 1].set_title(f"{method} Resized (Zoom: {zoom_factor}x, Degree: {interp_degree}, Time: {ls_time:.2f}s)")
    ax[0, 1].axis("off")

    ls_diff = np.clip(np.abs(input_image_normalized - ls_resized_reverse / 255.0), 0, 1)
    ax[0, 2].imshow(ls_diff, cmap="gray")
    ax[0, 2].set_title(f"{method} Difference (SNR: {ls_snr:.2f} dB, MSE: {ls_mse:.2e})")
    ax[0, 2].axis("off")

    ax[1, 0].imshow(input_image, cmap="gray")
    ax[1, 0].set_title("Original Image")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(create_black_background(scipy_output, input_image.shape), cmap="gray")
    ax[1, 1].set_title(f"SciPy Zoom (Zoom: {zoom_factor}x, Degree: {interp_degree}, Time: {scipy_time:.2f}s)")
    ax[1, 1].axis("off")

    scipy_diff = np.clip(np.abs(input_image_normalized - scipy_resized_reverse / 255.0), 0, 1)
    ax[1, 2].imshow(scipy_diff, cmap="gray")
    ax[1, 2].set_title(f"SciPy Difference (SNR: {scipy_snr:.2f} dB, MSE: {scipy_mse:.2e})")
    ax[1, 2].axis("off")

    plt.tight_layout()
    plt.show()
