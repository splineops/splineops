"""
Resizing 1D and 3D Signals: Comparison of Methods
=================================================

This script compares the performance of different resizing methods: `interpolation`,
`least-squares`, `oblique`, and SciPy's `zoom` for both 1D and 3D signals.
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

def resize_and_compare_methods(input_signal, zoom_factors, degree, methods):
    """Compare resizing methods and compute metrics."""
    results = {}
    for method in methods:
        if method == "scipy":
            resized_signal, expanded_signal, snr, mse = resize_with_scipy_zoom(input_signal, zoom_factors, degree)
        else:
            resized_signal = resize(
                data=input_signal,
                zoom_factors=zoom_factors,
                degree=degree,
                method=method
            )
            expanded_signal = resize(
                data=resized_signal,
                output_size=input_signal.shape,
                degree=degree,
                method=method
            )
            snr = compute_snr(input_signal, expanded_signal)
            mse = compute_mse(input_signal, expanded_signal)

        results[method] = (resized_signal, expanded_signal, snr, mse)
    return results

# %%
# 1D Signal: Comparison of Methods
# ---------------------------------
# Generate a synthetic 1D signal and compare resizing methods.

# Generate a 1D signal (sine wave with added noise)
x = np.linspace(0, 4 * np.pi, 100)
original_signal = np.sin(x) + 0.1 * np.random.randn(100)

# Set parameters
zoom_factor_1d = 0.5
methods = ["interpolation", "least-squares", "oblique", "scipy"]
degree = 3

# Compare methods
results_1d = resize_and_compare_methods(original_signal, (zoom_factor_1d,), degree, methods)

# %%
# Visualization of 1D Signal Results
# -----------------------------------
# Plot results for each resizing method.

for method, (resized, expanded, snr, mse) in results_1d.items():
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # Plot original signal
    ax[0].plot(x, original_signal, label="Original", color="blue")
    ax[0].set_title("Original Signal")
    ax[0].legend()
    ax[0].grid(True)

    # Plot resized signal
    ax[1].plot(
        np.linspace(0, 4 * np.pi, int(100 * zoom_factor_1d)),
        resized,
        label=f"Resized ({method})",
        color="orange"
    )
    ax[1].set_title(f"Resized Signal ({method})")
    ax[1].legend()
    ax[1].grid(True)

    # Plot difference
    difference = original_signal - expanded
    ax[2].plot(x, difference, label="Difference", color="red")
    ax[2].set_title(f"Difference (SNR: {snr:.2f} dB, MSE: {mse:.2e})")
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    plt.show()

# %%
# 3D Volume: Comparison of Methods
# ---------------------------------
# Generate a synthetic 3D sine wave volume and compare resizing methods.

# Generate a synthetic 3D sine wave volume
z, y, x = np.meshgrid(
    np.linspace(0, 4 * np.pi, 50),
    np.linspace(0, 4 * np.pi, 50),
    np.linspace(0, 4 * np.pi, 50)
)
original_volume = np.sin(x) * np.sin(y) * np.sin(z)

# Set parameters
zoom_factors_3d = (0.5, 0.5, 0.5)

# Compare methods
results_3d = resize_and_compare_methods(original_volume, zoom_factors_3d, degree, methods)

# %%
# Visualization of 3D Volume Results
# -----------------------------------
# Plot results for each resizing method.

for method, (resized, expanded, snr, mse) in results_3d.items():
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Get the middle slice index for each volume
    original_middle_index = original_volume.shape[0] // 2
    resized_middle_index = resized.shape[0] // 2
    expanded_middle_index = expanded.shape[0] // 2

    # Plot original volume slice
    ax[0].imshow(original_volume[original_middle_index, :, :], cmap="gray")
    ax[0].set_title("Original Volume Slice")
    
    # Plot resized volume slice
    ax[1].imshow(resized[resized_middle_index, :, :], cmap="gray")
    ax[1].set_title(f"Resized Volume Slice ({method})")

    # Plot difference slice
    difference = original_volume - expanded
    ax[2].imshow(difference[original_middle_index, :, :], cmap="gray")
    ax[2].set_title(f"Difference (SNR: {snr:.2f} dB, MSE: {mse:.2e})")

    plt.tight_layout()
    plt.show()
