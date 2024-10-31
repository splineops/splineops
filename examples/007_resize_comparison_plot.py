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

    return snr, mse

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

    return snr, mse

# Load MRI Image and Set Parameters
# ---------------------------------
#input_image = load_head_mri_image()  # Load the head MRI image

# Uncomment the following line to use the Ascent image instead
input_image = ascent()

input_image_normalized = (input_image / 255.0).astype(np.float64)  # Normalize to [0, 1]

methods = ["interpolation", "least-squares", "oblique"]
interpolation_type = "linear"
interp_degree = {'linear': 1, 'quadratic': 2, 'cubic': 3}[interpolation_type]

zoom_factors = np.linspace(0.25, 1.99, num=20)  # Zoom factors between 0.1 and 2.0

# Initialize dictionaries to store SNR and MSE values
snr_results = {method: [] for method in methods + ['scipy']}
mse_results = {method: [] for method in methods + ['scipy']}

# Run Resizing and Compute Metrics
# --------------------------------
for zoom_factor in zoom_factors:
    print(f"\nZoom Factor: {zoom_factor:.2f}")
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

        # Print SNR and MSE for each method
        print(f"  Method: {method.capitalize()}, SNR: {snr:.2f} dB, MSE: {mse:.6f}, Time: {elapsed_time:.3f} s")

    # Process with SciPy zoom
    start_time = time.time()
    snr, mse = resize_with_scipy_zoom(
        input_image_normalized, zoom_factor, interpolation_type
    )
    elapsed_time = time.time() - start_time

    # Store the results
    snr_results['scipy'].append(snr)
    mse_results['scipy'].append(mse)

    # Print SNR and MSE for SciPy zoom
    print(f"  Method: SciPy, SNR: {snr:.2f} dB, MSE: {mse:.6f}, Time: {elapsed_time:.3f} s")

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
plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))  # Use scientific notation for y-axis
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
