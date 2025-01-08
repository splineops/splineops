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
from matplotlib.gridspec import GridSpec
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

x = np.arange(27)               # integer coordinates [0, 1, 2, ...]
np.random.seed(42)              # for reproducibility
original_samples = np.random.uniform(-1, 1, len(x))  # random in [-1, 1]

plt.figure(figsize=(10, 4))
plt.title("Original Samples on Integer Grid")
plt.stem(x, original_samples, basefmt=" ")
plt.xlabel("X-axis (integer grid)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
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
high_res_factor = 3
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
plt.tight_layout()
plt.show()

# %%
# Sampling the Spline at Every "high_res_factor" Sample
# -----------------------------------------------------
#
# We re-plot the original samples along with the interpolated spline. 
# Then we define a natural number λ = high_res_factor and sample the spline at 
# x = λk, i.e., at every "high_res_factor" point in the original domain. 
# These newly sampled points, g[k] = f(λk), are plotted as red squares 
# with no fill to highlight them.

lambda_val = high_res_factor  # sample every 'high_res_factor' points
x_lambda = np.arange(x[0], x[-1] + 1, lambda_val)
g_lambda = np.interp(x_lambda, x_high_res, resized_signal)  # sample from the high-resolution spline

plt.figure(figsize=(10, 4))
plt.title("Original Samples with Interpolated Spline (f) and λ = high_res_factor Samples")

# Plot original discrete samples
plt.stem(x, original_samples, basefmt=" ", label="Original Samples")

# Plot spline interpolation
plt.plot(x_high_res, resized_signal, color="green", linewidth=2, label="Spline Interpolation (f)")

# Plot sampled points g[k] = f(λk) as red squares
plt.plot(
    x_lambda, 
    g_lambda, 
    'rs',            # 'r' for red, 's' for square
    mfc='none',      # Marker face color = 'none' (hollow square)
    markersize=12,   
    markeredgewidth=2,
    label="g[k] = f(λk)"
)

plt.xlabel("X-axis")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Side-by-Side Comparison with Shrunken Bottom Plot
# -------------------------------------------------
#
# Finally, we create a figure with a 2×2 GridSpec:
#   - The top row spans both columns, showing the entire domain x=0..(len(x)-1)
#   - The bottom row is split: left subplot for the "shrunken" domain of g[k] vs. k,
#     and right subplot left blank (white space).
#
# The ratio of widths is chosen dynamically so that the bottom-left subplot domain 
# visually matches g[k]’s domain length compared to the total domain length above.
#
# This ensures no hard-coded references (like "3", "27", or "8"); 
# everything is derived from the data itself.

fig = plt.figure(figsize=(12, 6))

domain_length = x[-1] - x[0]  # e.g. 26 if x is 0..26
num_g_points = len(g_lambda)  # e.g. 10 if x_lambda has 10 points
g_domain_length = num_g_points - 1  # e.g. 9 => last index is 9

# e.g., if domain_length=26, g_domain_length=9 => width_ratios=[9, 17]
width_ratios = [g_domain_length, domain_length - g_domain_length]

gs = GridSpec(nrows=2, ncols=2, width_ratios=width_ratios, height_ratios=[1, 1])

#####################################
# Top subplot (ax1) - Spans 2 columns
#####################################
ax1 = fig.add_subplot(gs[0, :])  # spans both columns

ax1.set_title("Original Samples, Interpolated f, and Sampled g[k]")

ax1.stem(x, original_samples, basefmt=" ", label="Original Samples")
ax1.plot(x_high_res, resized_signal, color="green", linewidth=2, label="Spline Interpolation (f)")

# Plot sampled points g[k] in red hollow squares
ax1.plot(
    x_lambda, 
    g_lambda, 
    'rs',
    mfc='none',
    markersize=12,
    markeredgewidth=2,
    label="Sampled g[k]"
)

# Set domain to [x[0], x[-1]] with integer ticks
ax1.set_xlim(x[0], x[-1])
ax1.set_xticks(np.arange(x[0], x[-1] + 1, 1)) 
ax1.set_xlabel("Domain (x)")
ax1.set_ylabel("Amplitude")
ax1.legend()
ax1.grid(True)

###############################################
# Bottom-left subplot (ax2) - "Shrunken" domain
###############################################
ax2 = fig.add_subplot(gs[1, 0])
ax2.set_title("Shrunken View of g[k] vs. k")

k_values = np.arange(num_g_points)

# Vertical red lines for each sample
ax2.vlines(
    k_values,
    ymin=0,
    ymax=g_lambda,
    color='red',
    linestyle='-',
    linewidth=1
)
# Red hollow squares at each sample
ax2.plot(
    k_values,
    g_lambda,
    'rs',
    mfc='none',
    markersize=12,
    markeredgewidth=2
)

# x-limits for the "g" domain => 0..(num_g_points-1)
ax2.set_xlim(0, num_g_points - 1)
ax2.set_xticks(np.arange(0, num_g_points, 1))
ax2.set_xlabel("Index (k)")
ax2.set_ylabel("Amplitude")
ax2.grid(True)

# Match the y-scale to the top subplot for direct amplitude comparison
ax2.set_ylim(ax1.get_ylim())

#####################################################
# Bottom-right subplot (ax_blank) - left blank/white
#####################################################
ax_blank = fig.add_subplot(gs[1, 1])
ax_blank.axis("off")  # Hide everything

fig.tight_layout()
plt.show()
