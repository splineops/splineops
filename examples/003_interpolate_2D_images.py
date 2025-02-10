"""
Interpolate 2D images
=====================

Interpolate 2D images with standard interpolation, least-squares and oblique projection,
comparing them to SciPy's zoom. We compute SNR and MSE only on a central region 
to exclude boundary artifacts.

You can download this example at the tab at right, as both a Python script
and as a Jupyter notebook.
"""

# %%
# Import required libraries
# -------------------------
#
# We import the required libraries, including NumPy for numerical computations,
# Matplotlib for plotting, and the custom `resize` function from the `splineops` package.

import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image
from scipy.ndimage import zoom  # For SciPy's zoom comparison
from splineops.resize.resize import resize  # Unified resize function
import time


# %%
# Helper functions
# ----------------
#
# We define:
#   - a utility to crop out ~20% borders around the image
#   - SNR and MSE on that central cropped area
#   - resizing functions

def crop_to_central_region(image, border_fraction):
    """
    Return a central sub-region of 'image', skipping 'border_fraction'
    of the width/height on all sides.
    """
    H, W = image.shape
    top = int(H * border_fraction)
    bottom = int(H * (1 - border_fraction))
    left = int(W * border_fraction)
    right = int(W * (1 - border_fraction))
    # Guard against degenerate cases
    top = max(top, 0)
    left = max(left, 0)
    bottom = min(bottom, H)
    right = min(right, W)
    return image[top:bottom, left:right]


def compute_snr_and_mse_cropped(original, processed, border_fraction):
    """
    Compute SNR and MSE on the 'central' cropped area, ignoring border_fraction
    of the image on each side.
    """
    # Crop both images consistently
    orig_cropped = crop_to_central_region(original, border_fraction)
    proc_cropped = crop_to_central_region(processed, border_fraction)

    # Now compute SNR/MSE on that region
    signal_power = np.mean(orig_cropped**2)
    noise_power = np.mean((orig_cropped - proc_cropped)**2)
    mse_val = noise_power

    if noise_power <= 1e-30:  # near-zero difference
        snr_val = float('inf')
    else:
        snr_val = 10 * np.log10(signal_power / noise_power)

    return snr_val, mse_val


def resize_with_scipy_zoom(input_image, zoom_factors, degree, border_fraction):
    """
    Resize using SciPy's zoom, then resize back and compute SNR/MSE
    *only on a central region* to avoid boundary artifacts.
    """
    start_time = time.perf_counter()
    resized_image = zoom(input_image, zoom_factors, order=degree)
    time_elapsed = time.perf_counter() - start_time

    reverse_zoom_factors = 1.0 / np.array(zoom_factors)
    recovered_image = zoom(resized_image, reverse_zoom_factors, order=degree)

    # Compute SNR/MSE on central region
    snr, mse = compute_snr_and_mse_cropped(input_image, recovered_image, border_fraction)

    return resized_image, recovered_image, snr, mse, time_elapsed


def resize_and_compute_metrics(input_image, method, degree, zoom_factors, border_fraction):
    """
    Resize a 2D image using the specified method, then resize back
    to original size and compute SNR, MSE, and timing *only on a central region*.
    """
    if np.isscalar(zoom_factors):
        zoom_factors = (zoom_factors, zoom_factors)

    if method == "scipy":
        return resize_with_scipy_zoom(
            input_image, zoom_factors, degree, border_fraction
        )
    else:
        start_time = time.perf_counter()
        resized_image = resize(
            data=input_image,
            zoom_factors=zoom_factors,
            degree=degree,
            method=method
        )
        time_elapsed = time.perf_counter() - start_time

        # Resize back to original shape:
        original_shape = input_image.shape
        recovered_image = resize(
            data=resized_image,
            output_size=original_shape,
            degree=degree,
            method=method
        )

        # Compute SNR/MSE on central region
        snr, mse = compute_snr_and_mse_cropped(input_image, recovered_image, border_fraction)

        return resized_image, recovered_image, snr, mse, time_elapsed


# %%
# Plotting helpers
# ----------------
#
# We define three plotting helpers now:
#   1) plot_resized_image(): Show the resized 2D image
#   2) plot_recovered_image(): Show the image after resizing back
#   3) plot_difference_image(): Show the difference (original - recovered)

def plot_resized_image(original, resized, method, zoom_factors, time_elapsed):
    """
    Display the resized 2D image. If any zoom factor < 1, we place
    the resized image on a white canvas matching the original shape.
    """
    zoom_out = any(zf < 1.0 for zf in zoom_factors)

    # Convert images to 0..255 for visualization
    def to_uint8(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr_scaled = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr_scaled = arr * 0.0
        return (arr_scaled * 255).astype(np.uint8)

    orig_8 = to_uint8(original)
    resized_8 = to_uint8(resized)

    if zoom_out:
        canvas_8 = np.full_like(orig_8, 255, dtype=np.uint8)  # white canvas
        rh, rw = resized_8.shape
        canvas_8[:rh, :rw] = resized_8
        resized_display = canvas_8
    else:
        resized_display = resized_8

    plt.figure(figsize=(5, 5))
    plt.imshow(resized_display, cmap='gray', aspect='equal')
    plt.title(
        f"{method.capitalize()} Resized\n"
        f"Zoom: {zoom_factors}, Time: {time_elapsed:.4f}s"
    )
    plt.axis('off')
    plt.show()

def plot_recovered_image(recovered):
    """
    Display the recovered image after resizing back to the original shape.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(recovered, cmap='gray', aspect='equal')
    plt.title("Recovered Image")
    plt.axis('off')
    plt.show()

def plot_difference_image(original, recovered, snr, mse):
    """
    Display the difference (original - recovered) with a colorbar.
    The difference is shown in the *original numeric range*, not uint8, 
    so the colorbar reflects the actual difference scale.

    We fix the color scale to [-0.8, +0.8] for consistency across plots.

    The `fraction` parameter to colorbar determines the fraction of
    the axes area occupied by the colorbar. A common default is ~0.15,
    but we reduce it to 0.046 so the colorbar is narrower, and we set
    `pad=0.04` to leave a bit of padding between the main image and
    the colorbar.
    """
    difference = original - recovered

    plt.figure(figsize=(6, 5))
    im = plt.imshow(
        difference,
        cmap='bwr',
        aspect='equal',
        vmin=-0.8,    # lower bound of color scale
        vmax=0.8      # upper bound of color scale
    )
    # We choose fraction=0.046 to make the colorbar relatively thin, 
    # and pad=0.04 to add spacing from the main image.
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Difference (units)')
    plt.title(f"Difference\nSNR: {snr:.2f} dB, MSE: {mse:.2e}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# %%
# Load and normalize a 2D image
# -----------------------------
#
# Here, we load an example image from an online repository.
# We convert it to grayscale in [0, 1].

url = 'https://people.math.sc.edu/Burkardt/data/tif/columns.tif'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
data = np.array(img, dtype=np.float64)

# Convert to [0..1]
input_image_normalized = data / 255.0

# Convert to grayscale via simple weighting
input_image_normalized = (
    input_image_normalized[:, :, 0] * 0.2989 +  # Red channel
    input_image_normalized[:, :, 1] * 0.5870 +  # Green channel
    input_image_normalized[:, :, 2] * 0.1140    # Blue channel
)

degree = 3
zoom_factors_2d = (0.25, 0.25)
border_fraction = 0.3

# We plot the original grayscale image.

plt.figure(figsize=(6, 5))
plt.imshow(input_image_normalized, cmap='gray', aspect='equal')
plt.title("Original Image")
plt.axis("off")
plt.show()

# %%
# SciPy interpolation
# -------------------
#
# For comparison purposes, we also use SciPy's zoom method for resizing.

(
    resized_2d_scipy,
    recovered_2d_scipy,
    snr_2d_scipy,
    mse_2d_scipy,
    time_2d_scipy
) = resize_and_compute_metrics(
    input_image_normalized,
    method="scipy",
    degree=degree,
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

# %%
# Resized image
# ~~~~~~~~~~~~~
#
# We plot the resized image with SciPy interpolation.

plot_resized_image(
    original=input_image_normalized,
    resized=resized_2d_scipy,
    method="scipy",
    zoom_factors=zoom_factors_2d,
    time_elapsed=time_2d_scipy
)

# %%
# Recovered image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after reversing zoom factors.

plot_recovered_image(recovered_2d_scipy)

# %%
# Difference image
# ~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_scipy,
    snr=snr_2d_scipy,
    mse=mse_2d_scipy
)

# %%
# Trivial interpolation
# ---------------------
#
# We our standard interpolation method.

(
    resized_2d_interp, 
    recovered_2d_interp, 
    snr_2d_interp, 
    mse_2d_interp, 
    time_2d_interp
) = resize_and_compute_metrics(
    input_image_normalized,
    method="interpolation",
    degree=degree,
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

# %%
# Resized image
# ~~~~~~~~~~~~~
#
# We plot the resized image with standard interpolation.

plot_resized_image(
    original=input_image_normalized,
    resized=resized_2d_interp,
    method="interpolation",
    zoom_factors=zoom_factors_2d,
    time_elapsed=time_2d_interp
)

# %%
# Recovered image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after reversing zoom factors.

plot_recovered_image(recovered_2d_interp)

# %%
# Difference image
# ~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_interp,
    snr=snr_2d_interp,
    mse=mse_2d_interp
)

# %%
# Difference with SciPy
# ~~~~~~~~~~~~~~~~~~~~~
#
# Now we compute the difference between the recovered image from the
# trivial interpolation and SciPy interpolation. We also compute
# SNR and MSE on the central region and display the difference.
# We observe that the difference is extremely small, hence the two interpolation 
# methods are identical.

snr_scipy_vs_interp, mse_scipy_vs_interp = compute_snr_and_mse_cropped(
    recovered_2d_scipy, recovered_2d_interp, border_fraction
)

plot_difference_image(
    original=recovered_2d_scipy,
    recovered=recovered_2d_interp,
    snr=snr_scipy_vs_interp,
    mse=mse_scipy_vs_interp
)

# %%
# Alternative using TensorSpline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# As an alternative, we can replicate the same interpolation manually using the 
# ``TensorSpline`` class, which underpins the `resize()` function behind the scenes.

from splineops.interpolate.tensorspline import TensorSpline

# 1) Build uniform coordinate arrays that match the shape of 'input_image_normalized'

height, width = input_image_normalized.shape
x_coords = np.linspace(0, height - 1, height)
y_coords = np.linspace(0, width - 1, width)
coordinates_2d = (x_coords, y_coords)

# 2) For "cubic interpolation", pick "bspline3".
#    For boundary handling, we can pick "mirror", "zero", etc.

ts = TensorSpline(
    data=input_image_normalized,
    coordinates=coordinates_2d,
    bases="bspline3",  # cubic B-splines
    modes="mirror"     # handles boundaries with mirroring
)

# 3) Define new coordinate grids for the "zoomed" shape. 

zoomed_height = int(height * zoom_factors_2d[0])
zoomed_width = int(width * zoom_factors_2d[1])

x_coords_zoomed = np.linspace(0, height - 1, zoomed_height)
y_coords_zoomed = np.linspace(0, width - 1, zoomed_width)
coords_zoomed_2d = (x_coords_zoomed, y_coords_zoomed)

# Evaluate (forward pass): zoom in or out

resized_direct_ts = ts(coordinates=coords_zoomed_2d)

# 4) Define coordinate grids for returning to the original shape
x_coords_orig = np.linspace(0, height - 1, height)
y_coords_orig = np.linspace(0, width - 1, width)
coords_orig_2d = (x_coords_orig, y_coords_orig)

# Evaluate (backward pass): from zoomed shape back to original

ts_zoomed = TensorSpline(
    data=resized_direct_ts,
    coordinates=coords_zoomed_2d,
    bases="bspline3",
    modes="mirror"
)
recovered_direct_ts = ts_zoomed(coordinates=coords_orig_2d)

# Now, resized_direct_ts / recovered_direct_ts should be very similar 
# to 'resized_2d_interp' / 'recovered_2d_interp' from the high-level "resize()" approach.
# Let's compute MSE to confirm:

mse_forward = np.mean((resized_direct_ts - resized_2d_interp) ** 2)
mse_backward = np.mean((recovered_direct_ts - recovered_2d_interp) ** 2)
print(f"MSE (TensorSpline vs. resize()) resized:  {mse_forward:.6e}")
print(f"MSE (TensorSpline vs. resize()) recovered: {mse_backward:.6e}")

# %%
# Least-squares projection
# ------------------------
#
# We use the least-squares projection method.

(
    resized_2d_ls,
    recovered_2d_ls,
    snr_2d_ls,
    mse_2d_ls,
    time_2d_ls
) = resize_and_compute_metrics(
    input_image_normalized,
    method="least-squares",
    degree=degree,
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

# %%
# Resized image
# ~~~~~~~~~~~~~
#
# We plot the resized image with least-squares projection method.

plot_resized_image(
    original=input_image_normalized,
    resized=resized_2d_ls,
    method="least-squares",
    zoom_factors=zoom_factors_2d,
    time_elapsed=time_2d_ls
)

# %%
# Recovered image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after reversing zoom factors.

plot_recovered_image(recovered_2d_ls)

# %%
# Difference image
# ~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ls,
    snr=snr_2d_ls,
    mse=mse_2d_ls
)

# %%
# Oblique projection
# ------------------
#
# We use the oblique projection method.

(
    resized_2d_ob,
    recovered_2d_ob,
    snr_2d_ob,
    mse_2d_ob,
    time_2d_ob
) = resize_and_compute_metrics(
    input_image_normalized,
    method="oblique",
    degree=degree,
    zoom_factors=zoom_factors_2d,
    border_fraction=border_fraction
)

# %%
# Resized image
# ~~~~~~~~~~~~~
#
# We plot the resized image with oblique projection method.

plot_resized_image(
    original=input_image_normalized,
    resized=resized_2d_ob,
    method="oblique",
    zoom_factors=zoom_factors_2d,
    time_elapsed=time_2d_ob
)

# %%
# Recovered image
# ~~~~~~~~~~~~~~~
#
# We plot the recovered image after reversing zoom factors.

plot_recovered_image(recovered_2d_ob)

# %%
# Difference image
# ~~~~~~~~~~~~~~~~
#
# Display the difference image (original - recovered) with colorbar.

plot_difference_image(
    original=input_image_normalized,
    recovered=recovered_2d_ob,
    snr=snr_2d_ob,
    mse=mse_2d_ob
)

# %%
# Comparison
# ----------
#
# We compare the performance of the different methods analysed.

# %%
# Comparison table
# ~~~~~~~~~~~~~~~~
#
# We print the SNR, MSE, and timing data for each method.

methods = [
    ("SciPy Interpolation", snr_2d_scipy, mse_2d_scipy, time_2d_scipy),
    ("Trivial Interpolation", snr_2d_interp, mse_2d_interp, time_2d_interp),
    ("Least-Squares Projection", snr_2d_ls, mse_2d_ls, time_2d_ls),
    ("Oblique Projection", snr_2d_ob, mse_2d_ob, time_2d_ob),
]

# Print the table header
print(f"{'Method':<24} {'SNR (dB)':>12} {'MSE':>15} {'Time (s)':>11}")
print("-" * 67)

# Print each row with some spacing/formatting
for method_name, snr_val, mse_val, time_val in methods:
    print(f"{method_name:<25} {snr_val:>10.2f} {mse_val:>16.2e} {time_val:>12.4f}")

# %%
# Comparison plot
# ~~~~~~~~~~~~~~~
#
# Here, we compare how SciPy, Least-Squares, and Oblique projection perform
# (in terms of SNR and MSE) across multiple zoom factors in [0.1, 0.9].
# As before, we compute SNR/MSE between the recovered image and the
# original for each method at each zoom factor, and plot the results on
# dual y-axes (SNR on the left, MSE on the right). The x-axis is in log scale.
# Note that we don't compare with trivial interpolation as it virtually gives
# the same values as the SciPy interpolation.

zoom_values = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9])

snr_scipy_list = []
mse_scipy_list = []
snr_ls_list = []
mse_ls_list = []
snr_ob_list = []
mse_ob_list = []

for z in zoom_values:
    # SciPy
    _, recovered_scipy, snr_scipy_z, mse_scipy_z, _ = resize_and_compute_metrics(
        input_image_normalized,
        method="scipy",
        degree=degree,
        zoom_factors=z,
        border_fraction=border_fraction
    )
    snr_scipy_list.append(snr_scipy_z)
    mse_scipy_list.append(mse_scipy_z)

    # Least-Squares
    _, recovered_ls, snr_ls_z, mse_ls_z, _ = resize_and_compute_metrics(
        input_image_normalized,
        method="least-squares",
        degree=degree,
        zoom_factors=z,
        border_fraction=border_fraction
    )
    snr_ls_list.append(snr_ls_z)
    mse_ls_list.append(mse_ls_z)

    # Oblique
    _, recovered_ob, snr_ob_z, mse_ob_z, _ = resize_and_compute_metrics(
        input_image_normalized,
        method="oblique",
        degree=degree,
        zoom_factors=z,
        border_fraction=border_fraction
    )
    snr_ob_list.append(snr_ob_z)
    mse_ob_list.append(mse_ob_z)

# Now we plot both SNR and MSE in the same figure using dual y-axes
fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

# Plot SNR (dB) on ax1
p1 = ax1.plot(zoom_values, snr_scipy_list, 'bo-', label='SciPy SNR (dB)')
p2 = ax1.plot(zoom_values, snr_ls_list,    'go-', label='LS SNR (dB)')
p3 = ax1.plot(zoom_values, snr_ob_list,    'ro-', label='Oblique SNR (dB)')

# Plot MSE on ax2
p4 = ax2.plot(zoom_values, mse_scipy_list, 'b^--', label='SciPy MSE')
p5 = ax2.plot(zoom_values, mse_ls_list,    'g^--', label='LS MSE')
p6 = ax2.plot(zoom_values, mse_ob_list,    'r^--', label='Oblique MSE')

# Set the zoom factor axis to log scale
ax1.set_xscale('log')
ax1.set_xlabel('Zoom factor (log scale)')
ax1.set_ylabel('SNR (dB)')
ax2.set_ylabel('MSE')

ax1.set_title('SciPy, Least-Squares, and Oblique\nSNR and MSE vs. Zoom Factor')

# Combine all line references to show a single legend
lines = p1 + p2 + p3 + p4 + p5 + p6
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='best')

plt.tight_layout()
plt.show()
