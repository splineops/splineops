"""
Resizing 1D and 3D Signals with TensorSpline and Advanced Techniques
====================================================================

This example demonstrates using the LS_Oblique_Resize model for resizing 1D and 3D signals.
We will apply it to a synthetic 1D signal and a 3D volume to explore how it behaves in different dimensions.
"""

# %%
# Import necessary libraries and define helper functions
# ------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.ls_oblique.ls_oblique_resize import ls_oblique_resize

def compute_snr(original, processed):
    """Compute Signal-to-Noise Ratio between two signals."""
    signal_power = np.max(original) ** 2
    noise_power = np.mean((original - processed) ** 2)
    return 10 * np.log10(signal_power / noise_power)

def compute_mse(original, processed):
    """Compute Mean Squared Error between two signals."""
    return np.mean((original - processed) ** 2)

def resize_and_compute_metrics(input_signal, method, interpolation, zoom_factors):
    """Resize a signal, compute SNR and MSE, and return the resized and reconstructed signals for comparison."""
    input_signal_normalized = (input_signal / np.max(input_signal)).astype(np.float64)

    # Resize (shrink) the signal
    resized_signal = ls_oblique_resize(
        input_img_normalized=input_signal_normalized,
        zoom_factors=zoom_factors,
        method=method,
        interpolation=interpolation,
        inversable=False
    )

    # Resize (expand) back to original size
    expanded_signal = ls_oblique_resize(
        input_img_normalized=resized_signal,
        output_size=input_signal_normalized.shape,
        method=method,
        interpolation=interpolation,
        inversable=False
    )

    # Calculate SNR and MSE
    snr = compute_snr(input_signal_normalized, expanded_signal)
    mse = compute_mse(input_signal_normalized, expanded_signal)

    return resized_signal, expanded_signal, snr, mse

# %%
# 1D Signal Example
# -----------------
# Generate a synthetic 1D signal, resize it with LS_Oblique_Resize, and visualize the results.

# Generate a 1D signal (sine wave with added noise)
x = np.linspace(0, 4 * np.pi, 100)
original_signal = np.sin(x) + 0.1 * np.random.randn(100)

# Set parameters
zoom_factor_1d = 0.5
method = "least-squares"
interpolation_type = "cubic"

# Apply LS_Oblique_Resize to the 1D signal
resized_signal_1d, expanded_signal_1d, snr_1d, mse_1d = resize_and_compute_metrics(
    original_signal, method, interpolation_type, (zoom_factor_1d,)
)

# Plot the original, reconstructed, and difference signals
fig, ax = plt.subplots(1, 3, figsize=(18, 5))

# Plot original signal
ax[0].plot(x, original_signal, label="Original Signal", color="blue")
ax[0].set_title("Original Signal")
ax[0].legend()
ax[0].grid(True)

# Plot resized signal
ax[1].plot(np.linspace(0, 4 * np.pi, int(100 * zoom_factor_1d)), resized_signal_1d, label="Resized Signal", color="orange")
ax[1].set_title("Resized Signal")
ax[1].legend()
ax[1].grid(True)

# Plot difference
difference_1d = original_signal - expanded_signal_1d
ax[2].plot(x, difference_1d, label="Difference", color="red")
ax[2].set_title(f"Difference (SNR: {snr_1d:.2f} dB, MSE: {mse_1d:.2e})")
ax[2].legend()
ax[2].grid(True)

plt.tight_layout()
plt.show()

# %%
# 3D Volume Example
# -----------------
# Generate a synthetic 3D volume (a 3D sine wave pattern) and resize it with LS_Oblique_Resize.

# Generate a 3D sine wave volume
z, y, x = np.meshgrid(
    np.linspace(0, 4 * np.pi, 50),
    np.linspace(0, 4 * np.pi, 50),
    np.linspace(0, 4 * np.pi, 50)
)
original_volume = np.sin(x) * np.sin(y) * np.sin(z)

# Set parameters
zoom_factors_3d = (0.5, 0.5, 0.5)

# Apply LS_Oblique_Resize to the 3D volume
resized_volume_3d, expanded_volume_3d, snr_3d, mse_3d = resize_and_compute_metrics(
    original_volume, method, interpolation_type, zoom_factors_3d
)

# Visualize a slice of the original, resized, and difference volumes
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Get the middle slice index for each volume
original_middle_index = original_volume.shape[0] // 2
resized_middle_index = resized_volume_3d.shape[0] // 2
expanded_middle_index = expanded_volume_3d.shape[0] // 2

# Plot original volume slice
ax[0].imshow(original_volume[original_middle_index, :, :], cmap="gray")
ax[0].set_title("Original Volume Slice")

# Plot resized volume slice
ax[1].imshow(resized_volume_3d[resized_middle_index, :, :], cmap="gray")
ax[1].set_title("Resized Volume Slice")

# Plot difference slice
difference_3d = original_volume - expanded_volume_3d
ax[2].imshow(difference_3d[original_middle_index, :, :], cmap="gray")
ax[2].set_title(f"Difference (SNR: {snr_3d:.2f} dB, MSE: {mse_3d:.2e})")

plt.tight_layout()
plt.show()
