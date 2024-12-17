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
# Process 1D samples
# ------------------
#
# We generate 1D samples and resize them using interpolation, least-squares and oblique projections.

# Generate the original 1D samples
x = np.linspace(0, 4 * np.pi, 100)
original_samples = np.sin(x) + 0.1 * np.random.randn(100)

# Define parameters for 1D samples
zoom_factor_1d = 0.5

# Display the original samples
plt.figure(figsize=(10, 4))
plt.plot(x, original_samples, label="Original Samples", color="blue")
plt.title("Original 1D Samples")
plt.legend()
plt.grid(True)
plt.show()

# %%
# 1D resizing: interpolation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

degree = 3

(
    resized_signal_1d_interp, 
    resized_back_signal_1d_interp, 
    snr_1d_interp, 
    mse_1d_interp, 
    time_elapsed_1d_interp
) = resize_and_compute_metrics(
    original_samples, 
    "interpolation", 
    degree, 
    zoom_factor_1d
)

# Plot results
plot_universal_results(
    original=original_samples,
    resized=resized_signal_1d_interp,
    resized_back=resized_back_signal_1d_interp,
    method="interpolation",
    zoom_factors=zoom_factor_1d,
    snr=snr_1d_interp,
    mse=mse_1d_interp,
    time_elapsed=time_elapsed_1d_interp
)

# %%
# 1D resizing: least-squares projection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(
    resized_signal_1d_ls, 
    resized_back_signal_1d_ls, 
    snr_1d_ls, 
    mse_1d_ls, 
    time_elapsed_1d_ls
) = resize_and_compute_metrics(
    original_samples, 
    "least-squares", 
    degree, 
    zoom_factor_1d
)

# Plot results
plot_universal_results(
    original=original_samples,
    resized=resized_signal_1d_ls,
    resized_back=resized_back_signal_1d_ls,
    method="least-squares",
    zoom_factors=zoom_factor_1d,
    snr=snr_1d_ls,
    mse=mse_1d_ls,
    time_elapsed=time_elapsed_1d_ls
)

# %%
# 1D resizing: oblique projection
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(
    resized_signal_1d_ob, 
    resized_back_signal_1d_ob, 
    snr_1d_ob, 
    mse_1d_ob, 
    time_elapsed_1d_ob
) = resize_and_compute_metrics(
    original_samples, 
    "oblique", 
    degree, 
    zoom_factor_1d
)

# Plot results
plot_universal_results(
    original=original_samples,
    resized=resized_signal_1d_ob,
    resized_back=resized_back_signal_1d_ob,
    method="oblique",
    zoom_factors=zoom_factor_1d,
    snr=snr_1d_ob,
    mse=mse_1d_ob,
    time_elapsed=time_elapsed_1d_ob
)

# %%
# 1D resizing: scipy interpolation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(
    resized_signal_1d_scipy, 
    resized_back_signal_1d_scipy, 
    snr_1d_scipy, 
    mse_1d_scipy, 
    time_elapsed_1d_scipy
) = resize_and_compute_metrics(
    original_samples, "scipy", degree, zoom_factor_1d
)

# Plot results
plot_universal_results(
    original=original_samples,
    resized=resized_signal_1d_scipy,
    resized_back=resized_back_signal_1d_scipy,
    method="scipy",
    zoom_factors=zoom_factor_1d,
    snr=snr_1d_scipy,
    mse=mse_1d_scipy,
    time_elapsed=time_elapsed_1d_scipy
)