"""
Resizing 1D samples
===================

This example compares SplineOps resizing with advanced interpolation methods:
Least-Squares, Oblique Projection, and SciPy's built-in zoom, on 1D samples.

You can download this example as both a Python script and as a Jupyter notebook.
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
import requests
from io import BytesIO
from PIL import Image

# %%
# Helper functions
# ----------------
#
# We define functions to compute Signal-to-Noise Ratio (SNR) and Mean Squared Error (MSE).
# Additionally, we include functions to perform resizing with SciPy's zoom and to compute
# metrics for the resized samples.

def compute_snr(original, processed):
    """Compute Signal-to-Noise Ratio between two samples."""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - processed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def compute_mse(original, processed):
    """Compute Mean Squared Error between two samples."""
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
    """Resize Samples using a given method and compute metrics."""
    import time

    if np.isscalar(zoom_factors):
        zoom_factors = [zoom_factors] * len(input_signal.shape)

    if method == "scipy":
        (
            resized_signal, 
            resized_back_signal, 
            snr, 
            mse, 
            time_elapsed,
        ) = resize_with_scipy_zoom(
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
    # Set global font size
    plt.rcParams.update({
        'font.size': 14,  # Base font size
        'axes.titlesize': 18,  # Title font size
        'axes.labelsize': 16,  # Label font size
        'xtick.labelsize': 14,  # X-axis tick font size
        'ytick.labelsize': 14   # Y-axis tick font size
    })

    # Compute difference
    difference = original - resized_back

    # Ensure original image is in the range [0, 255]
    if original.ndim > 1:  # Only for 2D or 3D slices
        original_scaled = (
            (original - original.min()) 
            / (original.max() - original.min()) 
            * 255.0
        )
        original_scaled = original_scaled.astype(np.uint8)
    else:
        original_scaled = original  # Keep 1D data unchanged

    # Check if zoom factors are < 1 in any direction
    zoom_factors = (
        [zoom_factors] 
        if isinstance(zoom_factors, (int, float)) 
        else zoom_factors
    )
    zoom_out = any(zf < 1 for zf in zoom_factors)

    # Adjust resized data to overlay on white background for 2D or 3D slices
    if original.ndim > 1 and zoom_out:
        # Normalize resized to [0, 255] for better visibility
        resized_normalized = (
            (resized - resized.min()) 
            / (resized.max() - resized.min()) 
            * 255.0
        )
        resized_normalized = resized_normalized.astype(np.uint8)

        # Create a white background of the original's shape
        resized_display = np.ones_like(original_scaled) * 255  # White background
        # Place resized data in the top-left corner
        start_indices = [0] * len(original_scaled.shape)
        slices = tuple(
            slice(start, start + res_dim) 
            for start, res_dim in zip(start_indices, resized.shape)
        )
        resized_display[slices] = resized_normalized
    else:
        resized_display = resized

    # Normalize difference to [0, 255] for better visualization
    if original.ndim > 1:
        difference_normalized = (
            (difference - difference.min()) 
            / (difference.max() - difference.min()) 
            * 255.0
        )
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
                np.linspace(0, 1, len(d)),  # Generate x-axis based on data length
                d
            ),
            a.set_title(t),
            a.set_xlabel("X-axis"),
            a.set_ylabel("Amplitude"),
            a.grid(True)
        ),
        2: lambda a, d, t, xv: (
            a.imshow(d, cmap="gray", aspect='equal', vmin=0, vmax=255),
            a.set_title(t),
            a.axis("off")
        )
    }

    dim = original.ndim
    plot_func = plotters[dim]
    chosen_titles = titles[dim]

    # Create figure and subplots arranged vertically
    fig, ax = plt.subplots(3, 1, figsize=(12, 24))

    # Plot original (use scaled values for 2D/3D, raw for 1D)
    plot_func(ax[0], original_scaled, chosen_titles[0], None)
    # Plot resized (with adjustment for zoom-out)
    plot_func(ax[1], resized_display, chosen_titles[1], None)
    # Plot difference
    plot_func(ax[2], difference_normalized, chosen_titles[2], None)

    # Adjust spacing between plots
    fig.subplots_adjust(hspace=0.5)  # Increase spacing between rows

    plt.tight_layout()
    plt.show()

# %%
# Initial 1D samples
# ------------------
#
# We generate 1D samples and treat them as discrete signal points.
# 
# Let :math:`\mathbf{x} = [x_1, x_2, \dots, x_N]` be a set of 1D sampled points, and let the discrete signal
# :math:`f_{\text{samples}}(x)` be defined as random values within a specified range:
#
# .. math::
#
#    f_{\text{samples}}(x_i) \sim \text{Uniform}(-1, 1), \quad i = 1, \dots, N.
#
# These are the input samples that we will interpolate.

# Create a small array of about 10 samples
x = np.linspace(0, 4 * np.pi, 10)  # Only 10 samples

# Generate random samples between -1 and 1
np.random.seed(42)  # Set seed for reproducibility
original_samples = np.random.uniform(-1, 1, len(x))  # Random values in range [-1, 1]

plt.figure(figsize=(10, 4))
plt.title("Original Sparse Samples")
plt.stem(x, original_samples, basefmt=" ")
plt.xlabel("X-axis")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# %%
# Interpolate the samples with spline f
# -------------------------------------
#
# We interpolate the 1D samples with a spline to obtain a continuous function f.
#
# Given the discrete samples :math:`f_{\text{samples}}(x_i)`, the spline interpolation :math:`f(x)` can be expressed as:
#
# .. math::
#
#    f(x) = \sum_{k} c_k \beta_n(x - k),
#
# where:
# - :math:`\beta_n` is the B-spline of degree :math:`n`.
# - :math:`c_k` are the spline coefficients determined from the input samples.
#
# By choosing a sufficiently fine grid, we approximate a continuous function :math:`f` from the discrete samples.

degree = 3
high_res_factor = 10  # Upsample by a factor of 10 for smooth interpolation
new_length = len(original_samples) * high_res_factor

# Interpolated signal
resized_signal = resize(
    data=original_samples,
    output_size=(new_length,),
    degree=degree,
    method="interpolation"
)

x_high_res = np.linspace(x[0], x[-1], new_length)

plt.figure(figsize=(10, 4))
plt.title("Original Samples with Interpolated Spline (f)")
plt.stem(x, original_samples, basefmt=" ", label="Original Samples")
plt.plot(x_high_res, resized_signal, color="green", linewidth=2, label="Spline Interpolation (f)")
plt.xlabel("X-axis")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Rescaled spline g
# -----------------
#
# We now create a new function g by first extracting samples of f at a coarser resolution
# and then using these samples to construct a new continuous interpolation function g.
#
# Unlike a simple scaled version of f, g is defined independently from f's spline coefficients,
# using its own spline interpolation based on the downsampled samples of f.
#
# Specifically, let us take the high-resolution function :math:`f(x)` that was previously computed.
# We downsample :math:`f(x)` by a known factor to obtain fewer samples. From these fewer samples,
# we construct a new spline interpolation function :math:`g(x)` as:
#
# .. math::
#
#    g(x) = \sum_{k} d_k \beta_n(x - k),
#
# where :math:`d_k` are the spline coefficients computed from the downsampled samples of f,
# and :math:`\beta_n` is the same B-spline basis function of degree n used in f.
#
# To compare f and g properly, we must realign their domains. Let :math:`\lambda` be the inverse
# of the scaling factor used to define g's resolution relative to f. Then we consider :math:`g(\lambda x)`
# when comparing against :math:`f(x)`. To measure how closely g approximates f, we define:
#
# .. math::
#
#    h(x) = f(x) - g(\lambda x).
#
# By sampling h(x) on the same grid as f(x), we can compute the Mean Squared Error (MSE):
#
# .. math::
#
#    \text{MSE} = \frac{1}{N} \sum_{i=1}^{N} [f(x_i) - g(\lambda x_i)]^2.
#
# The MSE provides a quantitative metric of the approximation quality of g relative to f,
# taking into account the scaling and ensuring a fair comparison.

inverse_resample_factor = 1 / high_res_factor
resampled_length = int(len(resized_signal) * inverse_resample_factor)

# Resample f to obtain g
resampled_signal = resize(
    data=resized_signal,
    output_size=(resampled_length,),
    degree=degree,
    method="interpolation"
)

x_resampled = np.linspace(x[0], x[-1], resampled_length)

# Compute MSE between f and g(lambda x)
resampled_back_signal = resize(
    data=resampled_signal,
    output_size=(len(resized_signal),),
    degree=degree,
    method="interpolation"
)
mse_f_g = compute_mse(resized_signal, resampled_back_signal)

print(f"Mean Squared Error (MSE) between f(x) and g(λx): {mse_f_g:.2e}")

plt.figure(figsize=(10, 4))
plt.title("Original Samples, Interpolated Spline (f), and Resampled Spline (g)")
plt.stem(x, original_samples, basefmt=" ", linefmt='grey', markerfmt='o', label="Original Samples")
plt.plot(x_high_res, resized_signal, color="green", linewidth=2, label="Spline Interpolation (f)")
plt.plot(x_resampled, resampled_signal, color="red", linewidth=2, linestyle="--", label="Resampled Spline (g)")
plt.xlabel("X-axis")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
