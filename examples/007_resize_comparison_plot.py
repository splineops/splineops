"""
Comparison of Resizing Methods with TensorSpline and Advanced Techniques
=======================================================================

This example compares TensorSpline resizing with advanced interpolation methods:
Least-Squares, Oblique Projection, and SciPy's built-in zoom.

For various zoom factors between 0.1 and 2.0, this script computes the Signal-to-Noise Ratio (SNR) and Mean Squared Error (MSE) for each resizing method and plots them for comparison.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.datasets import ascent
from scipy.ndimage import zoom
from splineops.interpolate.resize import resize
from splineops.utils.image_loader import load_head_mri_image  # Import the MRI loader

# Helper Functions
# ----------------
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

def resize_and_compute_metrics(input_image, method, degree, zoom_factor):
    """Resize an image using the resize function, compute SNR and MSE after resizing back."""
    input_image_normalized = (input_image / 255.0).astype(np.float64)

    # Resize (shrink or enlarge) the image
    resized_image = resize(
        data=input_image_normalized,
        zoom_factors=(zoom_factor, zoom_factor),
        degree=degree,
        method=method
    )

    # Resize back to original size
    resized_back_image = resize(
        data=resized_image,
        output_size=input_image_normalized.shape,
        degree=degree,
        method=method
    )

    # Calculate SNR and MSE
    snr = compute_snr(input_image_normalized * 255.0, resized_back_image * 255.0)
    mse = compute_mse(input_image_normalized * 255.0, resized_back_image * 255.0)

    return snr, mse

def resize_with_scipy_zoom(input_image_normalized, zoom_factor, degree):
    """Resize an image with SciPy's zoom function, then resize back and compute metrics."""
    # Resize (shrink or enlarge) the image
    resized_image = zoom(input_image_normalized, (zoom_factor, zoom_factor), order=degree)

    # Resize back to original size
    reverse_zoom_factor = 1.0 / zoom_factor
    resized_back_image = zoom(resized_image, (reverse_zoom_factor, reverse_zoom_factor), order=degree)

    # Adjust the size if necessary
    if resized_back_image.shape != input_image_normalized.shape:
        resized_back_image = zoom(resized_back_image, (
            input_image_normalized.shape[0] / resized_back_image.shape[0],
            input_image_normalized.shape[1] / resized_back_image.shape[1]
        ), order=degree)

    # Calculate SNR and MSE
    snr = compute_snr(input_image_normalized * 255.0, resized_back_image * 255.0)
    mse = compute_mse(input_image_normalized * 255.0, resized_back_image * 255.0)

    return snr, mse

# Load MRI Image and Set Parameters
# ---------------------------------
input_image = load_head_mri_image()  # Load the head MRI image

# Uncomment the following line to use the Ascent image instead
# input_image = ascent()

input_image_normalized = (input_image / 255.0).astype(np.float64)  # Normalize to [0, 1]

methods = ["interpolation", "least-squares", "oblique"]
interpolation_type = "cubic"
interp_degree = {'linear': 1, 'quadratic': 2, 'cubic': 3}[interpolation_type]

zoom_factors = np.linspace(0.1, 2.0, num=30)  # Zoom factors between 0.1 and 2.0

# Initialize dictionaries to store SNR and MSE values
snr_results = {method: [] for method in methods + ['scipy']}
mse_results = {method: [] for method in methods + ['scipy']}

# Run Resizing and Compute Metrics
# --------------------------------
for zoom_factor in zoom_factors:
    print(f"Processing zoom factor: {zoom_factor:.2f}")
    for method in methods:
        # Measure and process with each method
        start_time = time.time()
        snr, mse = resize_and_compute_metrics(
            input_image, method, interp_degree, zoom_factor
        )
        elapsed_time = time.time() - start_time

        # Store the results
        snr_results[method].append(snr)
        mse_results[method].append(mse)

    # Process with SciPy zoom
    start_time = time.time()
    snr, mse = resize_with_scipy_zoom(
        input_image_normalized, zoom_factor, interp_degree
    )
    elapsed_time = time.time() - start_time

    # Store the results
    snr_results['scipy'].append(snr)
    mse_results['scipy'].append(mse)

# Plotting the Results
# --------------------
plt.figure(figsize=(12, 6))
for method in methods + ['scipy']:
    plt.plot(zoom_factors, snr_results[method], label=method.capitalize())

plt.title('SNR vs. Zoom Factor')
plt.xlabel('Zoom Factor')
plt.ylabel('SNR (dB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for method in methods + ['scipy']:
    plt.plot(zoom_factors, mse_results[method], label=method.capitalize())

plt.title('MSE vs. Zoom Factor')
plt.xlabel('Zoom Factor')
plt.ylabel('MSE')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
