"""
Comparison of resizing methods using SplineOps
==============================================

This example compares SplineOps resizing with advanced interpolation methods:
Least-Squares, Oblique Projection, and SciPy's built-in zoom, on 2D, 1D, and 3D signals.

We demonstrate the performance of these methods on a 2D MRI image, a 1D signal, and a 3D volume,
with step-by-step computations and visualizations.
"""

# %%
# Import Necessary Libraries
# --------------------------
#
# We import the required libraries, including NumPy for numerical computations,
# Matplotlib for plotting, and the custom `resize` function from the `splineops` package.

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom  # For SciPy's zoom comparison
from splineops.interpolate.resize import resize  # Unified resize function
from splineops.utils.image_loader import load_head_mri_image  # Import the MRI loader

# %%
# Helper Functions for Metric Calculation
# ---------------------------------------
#
# We define functions to compute Signal-to-Noise Ratio (SNR) and Mean Squared Error (MSE).
# Additionally, we include functions to perform resizing with SciPy's zoom and to compute
# metrics for the resized signals.

def compute_snr(original, processed):
    """Compute Signal-to-Noise Ratio between two signals."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - processed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def compute_mse(original, processed):
    """Compute Mean Squared Error between two signals."""
    return np.mean((original - processed) ** 2)

def resize_with_scipy_zoom(input_signal, zoom_factors, degree):
    """Resize using SciPy's zoom, then resize back and compute metrics."""
    import time

    start_time = time.perf_counter()
    resized_signal = zoom(input_signal, zoom_factors, order=degree)
    time_elapsed = time.perf_counter() - start_time

    reverse_zoom_factors = 1.0 / np.array(zoom_factors)
    resized_back_signal = zoom(resized_signal, reverse_zoom_factors, order=degree)

    snr = compute_snr(input_signal, resized_back_signal)
    mse = compute_mse(input_signal, resized_back_signal)

    return resized_signal, resized_back_signal, snr, mse, time_elapsed

def resize_and_compute_metrics(input_signal, method, degree, zoom_factors):
    """Resize a signal using a given method and compute metrics."""
    import time

    if np.isscalar(zoom_factors):
        zoom_factors = [zoom_factors] * len(input_signal.shape)

    if method == "scipy":
        resized_signal, resized_back_signal, snr, mse, time_elapsed = resize_with_scipy_zoom(
            input_signal=input_signal,
            zoom_factors=zoom_factors,
            degree=degree
        )
    else:
        start_time = time.perf_counter()
        resized_signal = resize(
            data=input_signal,
            zoom_factors=zoom_factors,
            degree=degree,
            method=method
        )
        time_elapsed = time.perf_counter() - start_time
        resized_back_signal = resize(
            data=resized_signal,
            output_size=input_signal.shape,
            degree=degree,
            method=method
        )
        snr = compute_snr(input_signal, resized_back_signal)
        mse = compute_mse(input_signal, resized_back_signal)

    return resized_signal, resized_back_signal, snr, mse, time_elapsed

def plot_results(
    original_slice, 
    resized_slice, 
    resized_back_slice, 
    method, 
    zoom_factors, 
    snr, 
    mse,
    time_elapsed
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
        start_indices = [0] * len(original_slice.shape)
        slices = tuple(slice(start, start + res_dim) for start, res_dim in zip(start_indices, resized_slice.shape))
        resized_display[slices] = resized_slice
    else:
        # Just use the resized slice as is
        resized_display = resized_slice

    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Original slice
    ax[0].imshow(original_slice, cmap="gray", aspect='auto')
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Resized slice
    ax[1].imshow(resized_display, cmap="gray", aspect='auto')
    ax[1].set_title(f"{method.capitalize()} Resized (Zoom: {zoom_factors})\nTime: {time_elapsed:.4f}s")
    ax[1].axis("off")

    # Difference map
    difference = original_slice - resized_back_slice
    ax[2].imshow(difference, cmap="gray", aspect='auto')
    ax[2].set_title(f"Difference (SNR: {snr:.2f} dB, MSE: {mse:.2e})")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

def plot_1d_results(original, resized, resized_back, method, x, zoom_factor, snr, mse, time_elapsed):
    """Plot results for 1D signals."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Original signal
    ax[0].plot(x, original, label="Original", color="blue")
    ax[0].set_title("Original Signal")
    ax[0].legend()
    ax[0].grid(True)

    # Resized signal
    resized_x = np.linspace(x[0], x[-1], resized.shape[0])
    ax[1].plot(
        resized_x,
        resized,
        label=f"Resized ({method})",
        color="orange"
    )
    ax[1].set_title(f"Resized Signal ({method})\nTime: {time_elapsed:.4f}s")
    ax[1].legend()
    ax[1].grid(True)

    # Difference
    difference = original - resized_back
    ax[2].plot(x, difference, label="Difference", color="red")
    ax[2].set_title(f"Difference (SNR: {snr:.2f} dB, MSE: {mse:.2e})")
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

# %%
# Example of resizing
# -------------------
#
# Create a simple 2D image as a sample data (e.g., a gradient or a checkerboard pattern).

# Create a simple 2D gradient image
nx, ny = 50, 50  # Original image dimensions
data = np.linspace(0, 1, nx * ny).reshape((nx, ny))

# Visualize the original data
plt.figure(figsize=(5, 5))
plt.imshow(data, cmap='gray', aspect='equal')
plt.title("Original data")
plt.show()

# %%
# Resizing with degree 1 and fixed output size
# --------------------------------------------
#
# Apply the resize function with degree=1 for linear B-spline interpolation and a fixed output size.

output_size = (100, 100)  # Target output size (doubling the size)

resized_degree_1 = resize(data, output_size=output_size, degree=1, modes="mirror", method="interpolation")

# Visualize the resized data
plt.figure(figsize=(5, 5))
plt.imshow(resized_degree_1, cmap='gray', aspect='equal')
plt.title("Degree 1, output size 100x100")
plt.show()

# %%
# Resizing with degree 2 and zoom factor 0.3
# ------------------------------------------
#
# Apply the resize function with degree=2 for quadratic B-spline interpolation and a zoom factor of 0.3.

zoom_factor_2 = 0.3  # Downscale by a factor of 0.3

resized_degree_2 = resize(data, zoom_factors=zoom_factor_2, degree=2, modes="mirror", method="interpolation")

# Visualize the resized data
plt.figure(figsize=(5, 5))
plt.imshow(resized_degree_2, cmap='gray', aspect='equal')
plt.title("Degree 2, zoom factor 0.3")
plt.show()

# %%
# Resizing with degree 3 and zoom factor 2.5
# ------------------------------------------
#
# Apply the resize function with degree=3 for cubic B-spline interpolation and a zoom factor of 2.5.

zoom_factor_3 = 2.5  # Upscale by a factor of 2.5

resized_degree_3 = resize(data, zoom_factors=zoom_factor_3, degree=3, modes="mirror", method="interpolation")

# Visualize the resized data
plt.figure(figsize=(5, 5))
plt.imshow(resized_degree_3, cmap='gray', aspect='equal')
plt.title("Degree 3, zoom factor 2.5")
plt.show()

# %%
# Set common parameters for comparison of resizing methods
# --------------------------------------------------------
#
# We define the degree of interpolation and the methods to be compared.

degree = 3
methods = ["interpolation", "least-squares", "oblique", "scipy"]

# %%
# Load MRI Image and Normalize
# ----------------------------
#
# We load the MRI head image and normalize it to the range [0, 1].

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
# Define Parameters for Resizing the MRI Image
# --------------------------------------------
#
# Set the zoom factors for the MRI image.

zoom_factors_2d = (0.5, 0.5)

# %%
# Resizing and Comparing Methods for MRI Image
# --------------------------------------------
#
# Iterate over the different methods, perform resizing, compute metrics, and plot results for the MRI image.

for method in methods:
    resized_signal, resized_back_signal, snr, mse, time_elapsed = resize_and_compute_metrics(
        input_image_normalized, method, degree, zoom_factors_2d
    )

    # Plot results
    plot_results(
        original_slice=input_image_normalized,
        resized_slice=resized_signal,
        resized_back_slice=resized_back_signal,
        method=method,
        zoom_factors=zoom_factors_2d,
        snr=snr,
        mse=mse,
        time_elapsed=time_elapsed
    )

# %%
# Generate and Process 1D Signal
# ------------------------------
#
# We generate a noisy sine wave signal and define the parameters for resizing.

# Generate the original 1D signal
x = np.linspace(0, 4 * np.pi, 100)
original_signal = np.sin(x) + 0.1 * np.random.randn(100)

# Define parameters for 1D signal
zoom_factor_1d = 0.5

# Display the original signal
plt.figure(figsize=(10, 4))
plt.plot(x, original_signal, label="Original Signal", color="blue")
plt.title("Original 1D Signal")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Resizing and Comparing Methods for 1D Signal
# --------------------------------------------
#
# Iterate over the different methods, perform resizing, compute metrics, and plot the results for the 1D signal.

for method in methods:
    resized_signal, resized_back_signal, snr, mse, time_elapsed = resize_and_compute_metrics(
        original_signal, method, degree, zoom_factor_1d
    )

    # Plot results
    plot_1d_results(
        original=original_signal,
        resized=resized_signal,
        resized_back=resized_back_signal,
        method=method,
        x=x,
        zoom_factor=zoom_factor_1d,
        snr=snr,
        mse=mse,
        time_elapsed=time_elapsed
    )

# %%
# Generate and Process 3D Signal
# ------------------------------
#
# We generate a 3D sine wave volume and define the parameters for resizing.

# Generate the original 3D volume
z, y, x_grid = np.meshgrid(
    np.linspace(0, 4 * np.pi, 50),
    np.linspace(0, 4 * np.pi, 50),
    np.linspace(0, 4 * np.pi, 50)
)
original_volume = np.sin(x_grid) * np.sin(y) * np.sin(z)

# Define parameters for 3D signal
zoom_factors_3d = (0.5, 0.5, 0.5)

# Display a middle slice of the original volume
middle_slice = original_volume.shape[0] // 2
plt.imshow(original_volume[middle_slice, :, :], cmap="gray")
plt.title("Original Volume Slice (Middle)")
plt.colorbar()
plt.axis('off')
plt.show()

# %%
# Resizing and Comparing Methods for 3D Signal
# --------------------------------------------
#
# Iterate over the different methods, perform resizing, compute metrics, and plot the results for the 3D signal.

for method in methods:
    resized_volume, resized_back_volume, snr, mse, time_elapsed = resize_and_compute_metrics(
        original_volume, method, degree, zoom_factors_3d
    )

    # Extract slices for visualization
    original_slice = original_volume[middle_slice, :, :]
    resized_slice = resized_volume[resized_volume.shape[0] // 2, :, :]
    resized_back_slice = resized_back_volume[middle_slice, :, :]

    # Plot results
    plot_results(
        original_slice=original_slice,
        resized_slice=resized_slice,
        resized_back_slice=resized_back_slice,
        method=method,
        zoom_factors=zoom_factors_3d,
        snr=snr,
        mse=mse,
        time_elapsed=time_elapsed
    )
