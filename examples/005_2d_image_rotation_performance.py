"""
2D Image Rotation Performance
=============================

This example demonstrates how to create a basic rotation using the TensorSpline API and compare it against SciPy's.
"""

# %%
# Imports
# -------
#
# Import necessary libraries.

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import ndimage, datasets

from splineops.interpolate.rotate import rotate  # Import the rotate function

# %%
# Calculate inscribed rectangle bounds from image
# -----------------------------------------------
#
# Calculate the bounds for the largest rectangle that can be inscribed within a circle, which itself is inscribed within the original image.


def calculate_inscribed_rectangle_bounds_from_image(image):
    height, width = image.shape[:2]
    radius = min(width, height) / 2
    side_length = radius * np.sqrt(2)
    cx, cy = width / 2, height / 2
    x_min = int(cx - side_length / 2)
    y_min = int(cy - side_length / 2)
    x_max = int(cx + side_length / 2)
    y_max = int(cy + side_length / 2)
    return np.array([x_min, y_min, x_max, y_max])


# %%
# Crop image to bounds
# --------------------
#
# Crop an image to the specified bounds.

def crop_image_to_bounds(image, bounds):
    x_min, y_min, x_max, y_max = bounds
    return image[y_min:y_max, x_min:x_max]


# %%
# Calculate signal-to-noise ratio (SNR)
# -------------------------------------
#
# Compute the SNR between the original and modified images.

def calculate_snr(original, modified):
    original_normalized = original / 255.0 if original.max() > 1 else original
    processed_normalized = modified / 255.0 if modified.max() > 1 else modified
    noise = original_normalized - processed_normalized
    mean_signal = np.mean(original_normalized)
    variance_noise = np.var(noise)
    epsilon = 1e-3
    snr = 10 * np.log10((mean_signal**2) / (variance_noise + epsilon))
    return snr


# %%
# Calculate mean squared error (MSE)
# ----------------------------------
#
# Compute the MSE between the original and modified images.

def calculate_mse(original, modified):
    mse = np.mean((original - modified) ** 2)
    return mse


# %%
# Rotate image and crop using SplineOps
# -------------------------------------
#
# Rotate an image by a specified angle using the splineops library's `rotate` function and crop the result.

def rotate_image_and_crop_splineops(image, angle, degree=3, mode="zero", iterations=1):
    rotated_image = image
    for _ in range(iterations):
        rotated_image = rotate(rotated_image, angle, degree=degree, mode=mode)  # Use rotate from rotate.py
    return rotated_image


# %%
# Rotate image and crop using SciPy
# ---------------------------------
#
# Rotate an image by a specified angle using SciPy's ndimage.rotate function and crop the result.

def rotate_image_and_crop_scipy(image, angle, order=3, iterations=5):
    rotated_image = image.copy()
    for _ in range(iterations):
        rotated_image = ndimage.rotate(
            rotated_image, angle, reshape=False, order=order, mode="constant", cval=0
        )
    return rotated_image


# %%
# Benchmark and display rotation
# ------------------------------
#
# Perform a benchmark of the rotation operation for both SplineOps and SciPy libraries and display images.

def benchmark_and_display_rotation(image, angle, degree, iterations):
    start_time_custom = time.time()
    custom_rotated_and_cropped_splineops = rotate_image_and_crop_splineops(
        image, angle, degree=degree, mode="zero", iterations=iterations
    )
    time_custom = time.time() - start_time_custom

    start_time_scipy = time.time()
    scipy_rotated_and_cropped = rotate_image_and_crop_scipy(
        image, angle, order=degree, iterations=iterations
    )
    time_scipy = time.time() - start_time_scipy

    bounds = calculate_inscribed_rectangle_bounds_from_image(image)
    image_cropped = crop_image_to_bounds(image, bounds)
    custom_rotated_and_cropped_splineops = crop_image_to_bounds(
        custom_rotated_and_cropped_splineops, bounds
    )
    scipy_rotated_and_cropped = crop_image_to_bounds(scipy_rotated_and_cropped, bounds)

    snr_splineops = calculate_snr(image_cropped, custom_rotated_and_cropped_splineops)
    snr_scipy = calculate_snr(image_cropped, scipy_rotated_and_cropped)
    mse_splineops = calculate_mse(image_cropped, custom_rotated_and_cropped_splineops)
    mse_scipy = calculate_mse(image_cropped, scipy_rotated_and_cropped)

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))
    axes[0].imshow(image_cropped, cmap="gray")
    axes[0].set_title("Original Image")
    axes[1].imshow(custom_rotated_and_cropped_splineops, cmap="gray")
    axes[1].set_title(
        f"SplineOps Rotated\nSNR: {snr_splineops:.2f}dB, MSE: {mse_splineops:.2e}\nAngle: {angle}°, Iter: {iterations}\nDegree: {degree}, Time: {time_custom:.2f}s"
    )
    axes[2].imshow(scipy_rotated_and_cropped, cmap="gray")
    axes[2].set_title(
        f"SciPy Rotated\nSNR: {snr_scipy:.2f}dB, MSE: {mse_scipy:.2e}\nAngle: {angle}°, Iter: {iterations}\nDegree: {degree}, Time: {time_scipy:.2f}s"
    )

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.05)
    plt.show()


# %%
# Load image and perform rotations
# --------------------------------
#
# Load the image, perform rotations using both SplineOps and SciPy methods, and display the results.

# Image size, Rotation angle and iterations and degree of spline interpolation
size = 1000
angle = 72
iterations = 5
degree = 3

# Load and resize the ascent image
image = datasets.ascent()
image_resized = ndimage.zoom(
    image, (size / image.shape[0], size / image.shape[1]), order=degree
)

# Convert to float32
image_resized = image_resized.astype(np.float32)

# Benchmark and display rotation results
benchmark_and_display_rotation(image_resized, angle, degree, iterations)
