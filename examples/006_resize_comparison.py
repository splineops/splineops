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
input_image_normalized = (input_image / 255.0).astype(np.float64)  # Normalize to [0, 1]

zoom_factor = 0.5  # Resize factor
interpolation_type = "cubic"
interp_degree = {'linear': 1, 'quadratic': 2, 'cubic': 3}[interpolation_type]

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

def resize_and_compute_metrics(input_image, method, degree, zoom_factor):
    """Resize an image using the `resize` function, compute SNR and MSE after resizing back."""
    input_image_normalized = (input_image / 255.0).astype(np.float64)

    # Resize (shrink) the image
    shrunken_image = resize(
        data=input_image_normalized,
        zoom_factors=(zoom_factor, zoom_factor),
        degree=degree,
        method=method
    )

    # Resize (expand) back to original size
    expanded_image = resize(
        data=shrunken_image,
        output_size=input_image_normalized.shape,
        degree=degree,
        method=method
    )

    # Calculate SNR and MSE
    snr = compute_snr(input_image_normalized, expanded_image)
    mse = compute_mse(input_image_normalized, expanded_image)

    # Convert images back to [0, 255] range for display
    shrunken_image_display = np.clip(shrunken_image * 255.0, 0, 255)
    expanded_image_display = np.clip(expanded_image * 255.0, 0, 255)

    return shrunken_image_display, expanded_image_display, snr, mse

# %%
# TensorSpline: Interpolation Method
# -----------------------------------
#
# Resizing using TensorSpline with the "interpolation" method.

method = "interpolation"

ts_output, ts_resized_reverse, ts_snr, ts_mse = resize_and_compute_metrics(
    input_image, method, interp_degree, zoom_factor
)

# Display results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(input_image, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

if zoom_factor < 1:
    ts_display = create_black_background(ts_output, input_image.shape)
else:
    ts_display = ts_output

ax[1].imshow(ts_display, cmap="gray")
ax[1].set_title(f"{method.capitalize()} Resized (Zoom: {zoom_factor}x)")
ax[1].axis("off")

ts_diff = np.clip(np.abs(input_image_normalized - ts_resized_reverse / 255.0), 0, 1)
ax[2].imshow(ts_diff, cmap="gray")
ax[2].set_title(f"Difference (SNR: {ts_snr:.2f} dB, MSE: {ts_mse:.2e})")
ax[2].axis("off")
plt.tight_layout()
plt.show()

# %%
# TensorSpline: Least-Squares Method
# -----------------------------------
#
# Resizing using TensorSpline with the "least-squares" method.

method = "least-squares"

ts_output, ts_resized_reverse, ts_snr, ts_mse = resize_and_compute_metrics(
    input_image, method, interp_degree, zoom_factor
)

# Display results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(input_image, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

if zoom_factor < 1:
    ts_display = create_black_background(ts_output, input_image.shape)
else:
    ts_display = ts_output

ax[1].imshow(ts_display, cmap="gray")
ax[1].set_title(f"{method.capitalize()} Resized (Zoom: {zoom_factor}x)")
ax[1].axis("off")

ts_diff = np.clip(np.abs(input_image_normalized - ts_resized_reverse / 255.0), 0, 1)
ax[2].imshow(ts_diff, cmap="gray")
ax[2].set_title(f"Difference (SNR: {ts_snr:.2f} dB, MSE: {ts_mse:.2e})")
ax[2].axis("off")
plt.tight_layout()
plt.show()

# %%
# TensorSpline: Oblique Projection Method
# ---------------------------------------
#
# Resizing using TensorSpline with the "oblique" method.

method = "oblique"

ts_output, ts_resized_reverse, ts_snr, ts_mse = resize_and_compute_metrics(
    input_image, method, interp_degree, zoom_factor
)

# Display results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(input_image, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

if zoom_factor < 1:
    ts_display = create_black_background(ts_output, input_image.shape)
else:
    ts_display = ts_output

ax[1].imshow(ts_display, cmap="gray")
ax[1].set_title(f"{method.capitalize()} Resized (Zoom: {zoom_factor}x)")
ax[1].axis("off")

ts_diff = np.clip(np.abs(input_image_normalized - ts_resized_reverse / 255.0), 0, 1)
ax[2].imshow(ts_diff, cmap="gray")
ax[2].set_title(f"Difference (SNR: {ts_snr:.2f} dB, MSE: {ts_mse:.2e})")
ax[2].axis("off")
plt.tight_layout()
plt.show()

# %%
# SciPy Zoom Resizing
# --------------------
#
# Resizing using SciPy's built-in zoom method for comparison.

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

scipy_output, scipy_resized_reverse, scipy_snr, scipy_mse = resize_with_scipy_zoom(
    input_image_normalized, zoom_factor, interpolation_type
)

# Display results
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(input_image, cmap="gray")
ax[0].set_title("Original Image")
ax[0].axis("off")

if zoom_factor < 1:
    scipy_display = create_black_background(scipy_output, input_image.shape)
else:
    scipy_display = scipy_output

ax[1].imshow(scipy_display, cmap="gray")
ax[1].set_title(f"SciPy Zoom (Zoom: {zoom_factor}x)")
ax[1].axis("off")

scipy_diff = np.clip(np.abs(input_image_normalized - scipy_resized_reverse / 255.0), 0, 1)
ax[2].imshow(scipy_diff, cmap="gray")
ax[2].set_title(f"Difference (SNR: {scipy_snr:.2f} dB, MSE: {scipy_mse:.2e})")
ax[2].axis("off")
plt.tight_layout()
plt.show()
