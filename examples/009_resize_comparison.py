import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import ascent
from scipy.ndimage import zoom
from splineops.interpolate.resize import resize

# Helper functions for metrics
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

# Load the ascent image and prepare parameters
image = ascent()
image_normalized = (image / 255.0).astype(np.float64)  # Normalize to [0, 1]

zoom_factor = 0.5
degrees = [1, 3]
methods = ["interpolation", "least-squares", "oblique", "scipy"]

# Set up subplots
fig, axes = plt.subplots(len(methods), len(degrees), figsize=(15, 10))
fig.suptitle("Comparison of Resizing Methods with SNR and MSE on Ascent Image")

for i, method in enumerate(methods):
    for j, degree in enumerate(degrees):
        # Perform resizing based on the chosen method
        if method == "scipy":
            # Use SciPy's zoom function for interpolation
            resized_down = zoom(image_normalized, zoom_factor, order=degree)
            resized_up = zoom(resized_down, (1 / zoom_factor), order=degree)
        else:
            # Use our custom resize function for TensorSpline, LS, and Oblique
            resized_down = resize(image_normalized, zoom_factors=(zoom_factor, zoom_factor), degree=degree, method=method)
            resized_up = resize(resized_down, output_size=image.shape, degree=degree, method=method)

        # Compute SNR and MSE
        snr = compute_snr(image_normalized, resized_up)
        mse = compute_mse(image_normalized, resized_up)
        
        # Display the resized image and metrics
        axes[i, j].imshow(np.clip(resized_up * 255.0, 0, 255).astype(np.uint8), cmap="gray")
        title_method = "SciPy Zoom" if method == "scipy" else method.capitalize()
        axes[i, j].set_title(f"{title_method} (Degree {degree})\nSNR: {snr:.2f} dB, MSE: {mse:.2e}")
        axes[i, j].axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()
