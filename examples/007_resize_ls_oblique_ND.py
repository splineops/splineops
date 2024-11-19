"""
Resizing 1D and 3D Signals: Step-by-Step Comparison
===================================================

This script demonstrates the performance of different resizing methods: `interpolation`,
`least-squares`, `oblique`, and SciPy's `zoom` for both 1D and 3D signals, with step-by-step
computations and visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.resize import resize  # Unified resize function
from scipy.ndimage import zoom  # For SciPy's zoom comparison

def compute_snr(original, processed):
    """Compute Signal-to-Noise Ratio between two signals."""
    signal_power = np.max(original) ** 2
    noise_power = np.mean((original - processed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def compute_mse(original, processed):
    """Compute Mean Squared Error between two signals."""
    return np.mean((original - processed) ** 2)

def resize_with_scipy_zoom(input_signal, zoom_factors, degree):
    """Resize using SciPy's zoom, then resize back and compute metrics."""
    resized_signal = zoom(input_signal, zoom_factors, order=degree)
    reverse_zoom_factors = 1.0 / np.array(zoom_factors)
    resized_back_signal = zoom(resized_signal, reverse_zoom_factors, order=degree)

    snr = compute_snr(input_signal, resized_back_signal)
    mse = compute_mse(input_signal, resized_back_signal)

    return resized_signal, resized_back_signal, snr, mse

def resize_and_compute_metrics(input_image, method, degree, zoom_factors):
    """Resize an image using a given method and compute metrics."""
    # Ensure zoom_factors is an array
    if np.isscalar(zoom_factors):
        zoom_factors = [zoom_factors] * len(input_image.shape)

    if method == "scipy":
        resized_signal, resized_back_signal, snr, mse = resize_with_scipy_zoom(
            input_signal=input_image,
            zoom_factors=zoom_factors,
            degree=degree
        )
    else:
        resized_signal = resize(
            data=input_image,
            zoom_factors=zoom_factors,
            degree=degree,
            method=method
        )
        resized_back_signal = resize(
            data=resized_signal,
            output_size=input_image.shape,
            degree=degree,
            method=method
        )
        snr = compute_snr(input_image, resized_back_signal)
        mse = compute_mse(input_image, resized_back_signal)

    return resized_signal, resized_back_signal, snr, mse

def plot_1d_results(original, resized, resized_back, method, x, zoom_factor, snr, mse):
    """Plot results for 1D signals."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Original signal
    ax[0].plot(x, original, label="Original", color="blue")
    ax[0].set_title("Original Signal")
    ax[0].legend()
    ax[0].grid(True)

    # Resized signal
    ax[1].plot(
        np.linspace(0, x[-1], int(len(x) * zoom_factor)),
        resized,
        label=f"Resized ({method})",
        color="orange"
    )
    ax[1].set_title(f"Resized Signal ({method})")
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

def plot_3d_results(original, resized, resized_back, method, middle_slice, snr, mse):
    """Plot results for 3D volumes."""
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Original slice
    ax[0].imshow(original[middle_slice, :, :], cmap="gray")
    ax[0].set_title("Original Volume Slice")
    ax[0].axis("off")

    # Resized slice
    resized_middle = resized.shape[0] // 2
    ax[1].imshow(resized[resized_middle, :, :], cmap="gray")
    ax[1].set_title(f"Resized Volume Slice ({method})")
    ax[1].axis("off")

    # Difference slice
    difference = original - resized_back
    ax[2].imshow(difference[middle_slice, :, :], cmap="gray")
    ax[2].set_title(f"Difference Slice (SNR: {snr:.2f} dB, MSE: {mse:.2e})")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()

# %%
# Generate a 1D signal
# --------------------
x = np.linspace(0, 4 * np.pi, 100)
original_signal = np.sin(x) + 0.1 * np.random.randn(100)

# Define parameters
zoom_factor_1d = 0.5
methods = ["interpolation", "least-squares", "oblique", "scipy"]
degree = 3

# Display the original signal
plt.plot(x, original_signal, label="Original Signal", color="blue")
plt.title("Original 1D Signal")
plt.legend()
plt.grid(True)
plt.show()

# %%
# Method 1: Interpolation
# ------------------------
method = "interpolation"
resized_signal, resized_back_signal, snr, mse = resize_and_compute_metrics(
    original_signal, method, degree, zoom_factor_1d
)

plot_1d_results(original_signal, resized_signal, resized_back_signal, method, x, zoom_factor_1d, snr, mse)

# %%
# Method 2: Least-Squares
# ------------------------
method = "least-squares"
resized_signal, resized_back_signal, snr, mse = resize_and_compute_metrics(
    original_signal, method, degree, zoom_factor_1d
)

plot_1d_results(original_signal, resized_signal, resized_back_signal, method, x, zoom_factor_1d, snr, mse)

# %%
# Method 3: Oblique
# ------------------
method = "oblique"
resized_signal, resized_back_signal, snr, mse = resize_and_compute_metrics(
    original_signal, method, degree, zoom_factor_1d
)

plot_1d_results(original_signal, resized_signal, resized_back_signal, method, x, zoom_factor_1d, snr, mse)

# %%
# Method 4: SciPy
# ---------------
method = "scipy"
resized_signal, resized_back_signal, snr, mse = resize_and_compute_metrics(
    original_signal, method, degree, zoom_factor_1d
)

plot_1d_results(original_signal, resized_signal, resized_back_signal, method, x, zoom_factor_1d, snr, mse)

# %%
# Generate a 3D signal
# --------------------
z, y, x = np.meshgrid(
    np.linspace(0, 4 * np.pi, 50),
    np.linspace(0, 4 * np.pi, 50),
    np.linspace(0, 4 * np.pi, 50)
)
original_volume = np.sin(x) * np.sin(y) * np.sin(z)

# Define parameters
zoom_factors_3d = (0.5, 0.5, 0.5)
methods = ["interpolation", "least-squares", "oblique", "scipy"]
degree = 3

# Display a middle slice of the original volume
middle_slice = original_volume.shape[0] // 2
plt.imshow(original_volume[middle_slice, :, :], cmap="gray")
plt.title("Original Volume Slice (Middle)")
plt.colorbar()
plt.show()

# %%
# Method 1: Interpolation
# ------------------------
method = "interpolation"
resized_volume, resized_back_volume, snr, mse = resize_and_compute_metrics(
    original_volume, method, degree, zoom_factors_3d
)

plot_3d_results(original_volume, resized_volume, resized_back_volume, method, middle_slice, snr, mse)

# %%
# Method 2: Least-Squares
# ------------------------
method = "least-squares"
resized_volume, resized_back_volume, snr, mse = resize_and_compute_metrics(
    original_volume, method, degree, zoom_factors_3d
)

plot_3d_results(original_volume, resized_volume, resized_back_volume, method, middle_slice, snr, mse)

# %%
# Method 3: Oblique
# ------------------
method = "oblique"
resized_volume, resized_back_volume, snr, mse = resize_and_compute_metrics(
    original_volume, method, degree, zoom_factors_3d
)

plot_3d_results(original_volume, resized_volume, resized_back_volume, method, middle_slice, snr, mse)

# %%
# Method 4: SciPy
# ---------------
method = "scipy"
resized_volume, resized_back_volume, snr, mse = resize_and_compute_metrics(
    original_volume, method, degree, zoom_factors_3d
)

plot_3d_results(original_volume, resized_volume, resized_back_volume, method, middle_slice, snr, mse)