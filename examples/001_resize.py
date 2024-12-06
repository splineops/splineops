"""
Resizing signals and images
===========================

This example compares SplineOps resizing with advanced interpolation methods:
Least-Squares, Oblique Projection, and SciPy's built-in zoom, on 2D, 1D, and 3D signals.

We demonstrate the performance of these methods on a 2D MRI image, a 1D signal, and a 3D volume,
with step-by-step computations and visualizations.
"""

# %%
# Import required libraries
# -------------------------
#
# We import the required libraries, including NumPy for numerical computations,
# Matplotlib for plotting, and the custom `resize` function from the `splineops` package.

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom  # For SciPy's zoom comparison
from splineops.interpolate.resize import resize  # Unified resize function
from splineops.utils.image_loader import load_head_mri_image  # Import the MRI loader

# %%
# Helper functions
# ----------------
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

def plot_universal_results(
    original,
    resized,
    resized_back,
    method,
    zoom_factors,
    snr,
    mse,
    time_elapsed
):
    """
    A universal plotting function that handles both 1D and 2D data without using if statements.
    - For 1D data, line plots are used.
    - For 2D and 3D slices, imshow is used.
    
    When zoom factors are < 1, resized data is shown in a smaller area over a white background.
    Parameters
    ----------
    original : np.ndarray
        The original data (1D, 2D, or 3D slice).
    resized : np.ndarray
        The resized data (same dimensionality as original).
    resized_back : np.ndarray
        The data after resizing back to the original shape.
    method : str
        The resizing method name.
    zoom_factors : float or tuple
        The zoom factor(s) used for resizing.
    snr : float
        The Signal-to-Noise Ratio in dB.
    mse : float
        The Mean Squared Error.
    time_elapsed : float
        The time taken for the resizing operation.
    """

    # Compute difference
    difference = original - resized_back

    # Ensure original image is in the range [0, 255]
    if original.ndim > 1:  # Only for 2D or 3D slices
        original_scaled = (original - original.min()) / (original.max() - original.min()) * 255.0
        original_scaled = original_scaled.astype(np.uint8)
    else:
        original_scaled = original  # Keep 1D data unchanged

    # Check if zoom factors are < 1 in any direction
    zoom_factors = [zoom_factors] if isinstance(zoom_factors, (int, float)) else zoom_factors
    zoom_out = any(zf < 1 for zf in zoom_factors)

    # Adjust resized data to overlay on white background for 2D or 3D slices
    if original.ndim > 1 and zoom_out:
        # Normalize resized to [0, 255] for better visibility
        resized_normalized = (resized - resized.min()) / (resized.max() - resized.min()) * 255.0
        resized_normalized = resized_normalized.astype(np.uint8)

        # Create a white background of the original's shape
        resized_display = np.ones_like(original_scaled) * 255  # White background
        # Place resized data in the top-left corner
        start_indices = [0] * len(original_scaled.shape)
        slices = tuple(slice(start, start + res_dim) for start, res_dim in zip(start_indices, resized.shape))
        resized_display[slices] = resized_normalized
    else:
        resized_display = resized

    # Normalize difference to [0, 255] for better visualization
    if original.ndim > 1:
        difference_normalized = (difference - difference.min()) / (difference.max() - difference.min()) * 255.0
        difference_normalized = difference_normalized.astype(np.uint8)
    else:
        difference_normalized = difference

    # Titles for 1D and 2D/3D cases
    titles = {
        1: [
            "Original Signal",
            f"Resized Signal ({method})\nTime: {time_elapsed:.4f}s",
            f"Difference (SNR: {snr:.2f} dB, MSE: {mse:.2e})"
        ],
        2: [
            "Original Image",
            f"{method.capitalize()} Resized\nZoom: {zoom_factors} Time: {time_elapsed:.4f}s",
            f"Difference (SNR: {snr:.2f} dB, MSE: {mse:.2e})"
        ]
    }

    # Plotting functions for 1D and 2D/3D slices
    plotters = {
        1: lambda a, d, t, xv: (
            a.plot(
                np.linspace(0, 1, len(d)),  # Automatically generate x-axis based on data length
                d
            ),
            a.set_title(t),
            a.set_xlabel("X-axis"),
            a.set_ylabel("Amplitude"),
            a.grid(True)
        ),
        2: lambda a, d, t, xv: (
            a.imshow(d, cmap="gray", aspect='auto', vmin=0, vmax=255),
            a.set_title(t),
            a.axis("off")
        )
    }

    dim = original.ndim
    plot_func = plotters[dim]
    chosen_titles = titles[dim]

    # Create figure and subplots
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Plot original (use scaled values for 2D/3D, raw for 1D)
    plot_func(ax[0], original_scaled, chosen_titles[0], None)
    # Plot resized (with adjustment for zoom-out)
    plot_func(ax[1], resized_display, chosen_titles[1], None)
    # Plot difference
    plot_func(ax[2], difference_normalized, chosen_titles[2], None)

    plt.tight_layout()
    plt.show()




# %%
# Basic resizing example
# ----------------------
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
# Resizing with fixed output size
# -------------------------------
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
# Resizing with zoom factor 0.3
# -----------------------------
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
# Resizing with zoom factor 2.5
# -----------------------------
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
# Process 2D signal
# -----------------
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
# 2D resizing: interpolation
# --------------------------
#
# Perform resizing using interpolation method, compute metrics, and plot results for the MRI image.

degree = 3
zoom_factors_2d = (0.5, 0.5)

(
    resized_signal_MRI_interpolation, 
    resized_back_signal_MRI_interpolation, 
    snr_MRI_interpolation, mse_MRI_interpolation, 
    time_elapsed_MRI_interpolation
) = resize_and_compute_metrics(
    input_image_normalized, 
    "interpolation", 
    degree, 
    zoom_factors_2d
)

# Plot results
plot_universal_results(
    original=input_image_normalized,
    resized=resized_signal_MRI_interpolation,
    resized_back=resized_back_signal_MRI_interpolation,
    method="interpolation",
    zoom_factors=zoom_factors_2d,
    snr=snr_MRI_interpolation,
    mse=mse_MRI_interpolation,
    time_elapsed=time_elapsed_MRI_interpolation
)

# %%
# 2D resizing: least-squares
# --------------------------
#
# Perform resizing using least-squares projection method, compute metrics, and plot results for the MRI image.

(
    resized_signal_MRI_least_squares, 
    resized_back_signal_MRI_least_squares, 
    snr_MRI_least_squares, 
    mse_MRI_least_squares, 
    time_elapsed_MRI_least_squares
) = resize_and_compute_metrics(
    input_image_normalized, 
    "least_squares", 
    degree, 
    zoom_factors_2d
)

# Plot results
plot_universal_results(
    original=input_image_normalized,
    resized=resized_signal_MRI_least_squares,
    resized_back=resized_back_signal_MRI_least_squares,
    method="least-squares",
    zoom_factors=zoom_factors_2d,
    snr=snr_MRI_least_squares,
    mse=mse_MRI_least_squares,
    time_elapsed=time_elapsed_MRI_least_squares
)

# %%
# 2D resizing: oblique
# --------------------
#
# Perform resizing using oblique projection method, compute metrics, and plot results for the MRI image.

(
    resized_signal_MRI_oblique, 
    resized_back_signal_MRI_oblique, 
    snr_MRI_oblique, 
    mse_MRI_oblique, 
    time_elapsed_MRI_oblique
) = resize_and_compute_metrics(
    input_image_normalized, 
    "oblique", 
    degree, 
    zoom_factors_2d
)

# Plot results
plot_universal_results(
    original=input_image_normalized,
    resized=resized_signal_MRI_oblique,
    resized_back=resized_back_signal_MRI_oblique,
    method="oblique",
    zoom_factors=zoom_factors_2d,
    snr=snr_MRI_oblique,
    mse=mse_MRI_oblique,
    time_elapsed=time_elapsed_MRI_oblique
)

# %%
# 2D resizing: scipy
# ------------------
#
# Perform resizing using scipy ndimage zoom for benchmarking, compute metrics, and plot results for the MRI image.

(
    resized_signal_MRI_scipy, 
    resized_back_signal_MRI_scipy, 
    snr_MRI_scipy, mse_MRI_scipy, 
    time_elapsed_MRI_scipy
) = resize_and_compute_metrics(
    input_image_normalized, 
    "scipy", 
    degree, 
    zoom_factors_2d
)

# Plot results
plot_universal_results(
    original=input_image_normalized,
    resized=resized_signal_MRI_scipy,
    resized_back=resized_back_signal_MRI_scipy,
    method="scipy",
    zoom_factors=zoom_factors_2d,
    snr=snr_MRI_scipy,
    mse=mse_MRI_scipy,
    time_elapsed=time_elapsed_MRI_scipy
)

# %%
# Process 1D signal
# -----------------
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
# 1D resizing: interpolation
# --------------------------
#
# Perform resizing with interpolation method, compute metrics, and plot the results for the 1D signal.

(
    resized_signal_1d_interpolation, 
    resized_back_signal_1d_interpolation, 
    snr_1d_interpolation, 
    mse_1d_interpolation, 
    time_elapsed_1d_interpolation
) = resize_and_compute_metrics(
    original_signal, 
    "interpolation", 
    degree, 
    zoom_factor_1d
)

# Plot results
plot_universal_results(
    original=original_signal,
    resized=resized_signal_1d_interpolation,
    resized_back=resized_back_signal_1d_interpolation,
    method="interpolation",
    zoom_factors=zoom_factor_1d,
    snr=snr_1d_interpolation,
    mse=mse_1d_interpolation,
    time_elapsed=time_elapsed_1d_interpolation
)

# %%
# 1D resizing: least-squares
# --------------------------
#
# Perform resizing with least-squares projection method, compute metrics, and plot the results for the 1D signal.

(
    resized_signal_1d_least_squares, 
    resized_back_signal_1d_least_squares, 
    snr_1d_least_squares, 
    mse_1d_least_squares, 
    time_elapsed_1d_least_squares
) = resize_and_compute_metrics(
    original_signal, 
    "least-squares", 
    degree, 
    zoom_factor_1d
)

# Plot results
plot_universal_results(
    original=original_signal,
    resized=resized_signal_1d_least_squares,
    resized_back=resized_back_signal_1d_least_squares,
    method="least-squares",
    zoom_factors=zoom_factor_1d,
    snr=snr_1d_least_squares,
    mse=mse_1d_least_squares,
    time_elapsed=time_elapsed_1d_least_squares
)

# %%
# 1D resizing: oblique
# --------------------
#
# Perform resizing with oblique projection method, compute metrics, and plot the results for the 1D signal.

(
    resized_signal_1d_oblique, 
    resized_back_signal_1d_oblique, 
    snr_1d_oblique, 
    mse_1d_oblique, 
    time_elapsed_1d_oblique
) = resize_and_compute_metrics(
    original_signal, 
    "oblique", 
    degree, 
    zoom_factor_1d
)

# Plot results
plot_universal_results(
    original=original_signal,
    resized=resized_signal_1d_oblique,
    resized_back=resized_back_signal_1d_oblique,
    method="oblique",
    zoom_factors=zoom_factor_1d,
    snr=snr_1d_oblique,
    mse=mse_1d_oblique,
    time_elapsed=time_elapsed_1d_oblique
)

# %%
# 1D resizing: scipy
# ------------------
#
# Perform resizing with scipy ndimage zoom for benchmarking, compute metrics, and plot the results for the 1D signal.

(
    resized_signal_1d_scipy, 
    resized_back_signal_1d_scipy, 
    snr_1d_scipy, 
    mse_1d_scipy, 
    time_elapsed_1d_scipy
) = resize_and_compute_metrics(
    original_signal, "scipy", degree, zoom_factor_1d
)

# Plot results
plot_universal_results(
    original=original_signal,
    resized=resized_signal_1d_scipy,
    resized_back=resized_back_signal_1d_scipy,
    method="scipy",
    zoom_factors=zoom_factor_1d,
    snr=snr_1d_scipy,
    mse=mse_1d_scipy,
    time_elapsed=time_elapsed_1d_scipy
)

# %%
# Process 3D signal
# -----------------
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

# Extract slices for visualization
original_slice = original_volume[middle_slice, :, :]

# %%
# 3D resizing: interpolation
# --------------------------
#
# Perform resizing with interpolation method, compute metrics, and plot the results for the 3D signal.

(
    resized_volume_interpolation, 
    resized_back_volume_interpolation, 
    snr_interpolation, 
    mse_interpolation, 
    time_elapsed_interpolation
) = resize_and_compute_metrics(
    original_volume, 
    "interpolation",
      degree, 
      zoom_factors_3d
)

resized_slice_interpolation = resized_volume_interpolation[resized_volume_interpolation.shape[0] // 2, :, :]
resized_back_slice_interpolation = resized_back_volume_interpolation[middle_slice, :, :]

# Plot results
plot_universal_results(
    original=original_slice,
    resized=resized_slice_interpolation,
    resized_back=resized_back_slice_interpolation,
    method="interpolation",
    zoom_factors=zoom_factors_3d,
    snr=snr_interpolation,
    mse=mse_interpolation,
    time_elapsed=time_elapsed_interpolation
)

# %%
# 3D resizing: least-squares
# --------------------------
#
# Perform resizing with least-squares projection method, compute metrics, and plot the results for the 3D signal.

(
    resized_volume_least_squares, 
    resized_back_volume_least_squares, 
    snr_least_squares, 
    mse_least_squares, 
    time_elapsed_least_squares
) = resize_and_compute_metrics(
    original_volume, 
    "least_squares", 
    degree, 
    zoom_factors_3d
)

resized_slice_least_squares = resized_volume_least_squares[resized_volume_least_squares.shape[0] // 2, :, :]
resized_back_slice_least_squares = resized_back_volume_least_squares[middle_slice, :, :]

# Plot results
plot_universal_results(
    original=original_slice,
    resized=resized_slice_least_squares,
    resized_back=resized_back_slice_least_squares,
    method="least_squares",
    zoom_factors=zoom_factors_3d,
    snr=snr_least_squares,
    mse=mse_least_squares,
    time_elapsed=time_elapsed_least_squares
)

# %%
# 3D resizing: oblique
# --------------------
#
# Perform resizing with oblique projection method, compute metrics, and plot the results for the 3D signal.

(
    resized_volume_oblique, 
    resized_back_volume_oblique, 
    snr_oblique, 
    mse_oblique, 
    time_elapsed_oblique
) = resize_and_compute_metrics(
    original_volume, 
    "oblique", 
    degree, 
    zoom_factors_3d
)

resized_slice_oblique = resized_volume_oblique[resized_volume_oblique.shape[0] // 2, :, :]
resized_back_slice_oblique = resized_back_volume_oblique[middle_slice, :, :]

# Plot results
plot_universal_results(
    original=original_slice,
    resized=resized_slice_oblique,
    resized_back=resized_back_slice_oblique,
    method="oblique",
    zoom_factors=zoom_factors_3d,
    snr=snr_oblique,
    mse=mse_oblique,
    time_elapsed=time_elapsed_oblique
)

# %%
# 3D resizing: scipy
# ------------------
#
# Perform resizing with oblique projection method, compute metrics, and plot the results for the 3D signal.

(
    resized_volume_scipy, 
    resized_back_volume_scipy, 
    snr_scipy, 
    mse_scipy, 
    time_elapsed_scipy
) = resize_and_compute_metrics(
    original_volume, 
    "scipy", 
    degree, 
    zoom_factors_3d
)

resized_slice_scipy = resized_volume_scipy[resized_volume_scipy.shape[0] // 2, :, :]
resized_back_slice_scipy = resized_back_volume_scipy[middle_slice, :, :]

# Plot results
plot_universal_results(
    original=original_slice,
    resized=resized_slice_scipy,
    resized_back=resized_back_slice_scipy,
    method="scipy",
    zoom_factors=zoom_factors_3d,
    snr=snr_scipy,
    mse=mse_scipy,
    time_elapsed=time_elapsed_scipy
)