"""
Resizing 1D and 3D Signals with TensorSpline and Advanced Techniques
====================================================================

This example demonstrates using the LS_Oblique_Resize model for resizing 1D and 3D signals.
We will apply it to a synthetic 1D signal and a 3D volume to explore how it behaves in different dimensions.
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.ls_oblique.ls_oblique_resize import ls_oblique_resize
from scipy.ndimage import zoom

# %%
# Helper Functions
# ----------------
# Define functions for loading, processing, and evaluating signals in 1D and 3D.

def compute_snr(original, processed):
    """Compute Signal-to-Noise Ratio between two signals."""
    signal_power = np.max(original) ** 2
    noise_power = np.mean((original - processed) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def compute_mse(original, processed):
    """Compute Mean Squared Error between two signals."""
    mse = np.mean((original - processed) ** 2)
    return mse

def resize_and_compute_metrics(input_signal, method, interpolation, zoom_factors, signal_dim="1D"):
    """Resize a signal, compute SNR and MSE, and return the resized and reconstructed signals for comparison."""
    input_signal_normalized = (input_signal / np.max(input_signal)).astype(np.float64)

    # Resize (shrink) the signal
    shrunken_signal = ls_oblique_resize(
        input_img_normalized=input_signal_normalized,
        zoom_factors=zoom_factors,
        method=method,
        interpolation=interpolation,
        inversable=False
    )

    # Resize (expand) back to original size
    expanded_signal = ls_oblique_resize(
        input_img_normalized=shrunken_signal,
        output_size=input_signal_normalized.shape,
        method=method,
        interpolation=interpolation,
        inversable=False
    )

    # Calculate SNR and MSE
    snr = compute_snr(input_signal_normalized, expanded_signal)
    mse = compute_mse(input_signal_normalized, expanded_signal)

    # Convert signals back to original range for display
    shrunken_signal_display = shrunken_signal * np.max(input_signal)
    expanded_signal_display = expanded_signal * np.max(input_signal)

    return shrunken_signal_display, expanded_signal_display, snr, mse

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
shrunken_signal_1d, expanded_signal_1d, snr_1d, mse_1d = resize_and_compute_metrics(
    original_signal, method, interpolation_type, (zoom_factor_1d,), signal_dim="1D"
)

# Plot the original, shrunken, and reconstructed signals
plt.figure(figsize=(12, 6))
plt.plot(x, original_signal, label="Original Signal", color="blue")
plt.plot(np.linspace(0, 4 * np.pi, int(100 * zoom_factor_1d)), shrunken_signal_1d, label="Shrunken Signal", color="orange")
plt.plot(x, expanded_signal_1d, label="Reconstructed Signal", color="green")
plt.title(f"1D Signal Resizing - SNR: {snr_1d:.2f} dB, MSE: {mse_1d:.2e}")
plt.legend()
plt.show()

# %%
# 3D Volume Example
# -----------------
# Generate a synthetic 3D volume (a 3D sine wave pattern) and resize it with LS_Oblique_Resize.

# Generate a 3D sine wave volume
z, y, x = np.meshgrid(np.linspace(0, 4 * np.pi, 50), np.linspace(0, 4 * np.pi, 50), np.linspace(0, 4 * np.pi, 50))
original_volume = np.sin(x) * np.sin(y) * np.sin(z)

# Set parameters
zoom_factors_3d = (0.5, 0.5, 0.5)

# Apply LS_Oblique_Resize to the 3D volume
shrunken_volume_3d, expanded_volume_3d, snr_3d, mse_3d = resize_and_compute_metrics(
    original_volume, method, interpolation_type, zoom_factors_3d, signal_dim="3D"
)

# Visualize a slice of the original, shrunken, and reconstructed 3D volumes
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Get the middle slice index for each volume
original_middle_index = original_volume.shape[0] // 2
shrunken_middle_index = shrunken_volume_3d.shape[0] // 2
expanded_middle_index = expanded_volume_3d.shape[0] // 2

ax[0].imshow(original_volume[original_middle_index, :, :], cmap="gray")
ax[0].set_title("Original Volume Slice")
ax[1].imshow(shrunken_volume_3d[shrunken_middle_index, :, :], cmap="gray")
ax[1].set_title("Shrunken Volume Slice")
ax[2].imshow(expanded_volume_3d[expanded_middle_index, :, :], cmap="gray")
ax[2].set_title(f"Reconstructed Volume Slice - SNR: {snr_3d:.2f} dB, MSE: {mse_3d:.2e}")

plt.show()
