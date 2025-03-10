"""
Using decompose module
======================

This example demonstrates how to use the 'decompose' module for:

- Pyramid decomposition (reduce & expand) in 1D and 2D
- Wavelet decomposition (analysis & synthesis) in 2D:
  * Haar wavelets (2D)
  * Spline wavelets (2D) of orders 1,3,5

We replicate some small tests from reference C implementations, illustrating:
- How to create small 1D/2D signals
- Perform pyramid reduce/expand
- Perform wavelet analysis/synthesis
- Measure reconstruction errors
- Visualize intermediate and final results

You can download this example as both a Python script and a Jupyter notebook.
"""

# %%
# Imports
# -------
#
# We import the necessary libraries and modules for demonstrating
# pyramid decomposition and wavelet analysis/synthesis.

import numpy as np
import matplotlib.pyplot as plt

# Pyramid decomposition utilities
from splineops.decompose.pyramid import (
    get_pyramid_filter,
    reduce_1d, expand_1d,
    reduce_2d, expand_2d
)

# Wavelet classes for 2D
from splineops.decompose.wavelets.haar import HaarWavelets
from splineops.decompose.wavelets.splinewavelets import (
    Spline1Wavelets,
    Spline3Wavelets,
    Spline5Wavelets
)

# %%
# 1D Pyramid Decomposition
# ------------------------
#
# In this section, we build a 1D signal (length=10) and do a pyramid
# reduce-then-expand. This replicates the logic of a simpler reference test.

x = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -2.0, -4.0, -6.0], 
                dtype=np.float64)

filter_name = "Centered Spline"
order = 3
g, h, is_centered = get_pyramid_filter(filter_name, order)

reduced = reduce_1d(x, g, is_centered)
expanded = expand_1d(reduced, h, is_centered)
error = expanded - x

print("[1D Pyramid Test]")
print(f"Filter: '{filter_name}' (order={order}), is_centered={is_centered}")
print("Input   x:", x)
print("Reduced   :", reduced)
print("Expanded  :", expanded)
print("Error     :", error)

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 6))
axs[0].plot(x, 'o-', label='Input')
axs[0].set_title("1D Input Signal")
axs[0].legend()

axs[1].plot(reduced, 'o--', color='r', label='Reduced')
axs[1].set_title("Reduced (Half-Size)")
axs[1].legend()

axs[2].plot(expanded, 'o--', color='g', label='Expanded')
axs[2].plot(x, 'o-', color='k', alpha=0.3, label='Original')
axs[2].set_title(f"Expanded vs Original (Error max={np.abs(error).max():.3g})")
axs[2].legend()

plt.tight_layout()
plt.show()

# %%
# 2D Pyramid Decomposition
# ------------------------
#
# This section demonstrates a small 4x4 "image" being reduced and then expanded
# using the same pyramid filter logic.

arr = np.array([
    [0.0, 1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 3.0, 2.0]
], dtype=np.float32)

filter_name = "Spline"
order = 3
g, h, is_centered = get_pyramid_filter(filter_name, order)

reduced_2d = reduce_2d(arr, g, is_centered)
expanded_2d = expand_2d(reduced_2d, h, is_centered)

error_2d = expanded_2d - arr
max_err = np.abs(error_2d).max()

print("[2D Pyramid Test]")
print(f"Filter: '{filter_name}' (order={order}), is_centered={is_centered}")
print("Input 4x4 :\n", arr)
print("Reduced 2x2:\n", reduced_2d)
print("Expanded 4x4:\n", expanded_2d)
print("Error:\n", error_2d)
print(f"Max error: {max_err}")

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
ax[0].imshow(arr, cmap='viridis', vmin=arr.min(), vmax=arr.max())
ax[0].set_title("Original 4x4")

ax[1].imshow(expanded_2d, cmap='viridis', vmin=arr.min(), vmax=arr.max())
ax[1].set_title("Expanded from Reduced")

im2 = ax[2].imshow(error_2d, cmap='bwr')
ax[2].set_title(f"Error (max={max_err:.2g})")
plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %%
# Haar Wavelets (2D)
# ------------------
#
# Next, we demonstrate wavelet decomposition (analysis) and reconstruction (synthesis)
# using 2D Haar wavelets. For 2D Haar wavelets, we require `ny >= 2` and `nx >= 2`.

ny, nx = 32, 32
image = np.random.rand(ny, nx).astype(np.float32)*2 - 1.0  # negative & positive

haar2d = HaarWavelets(scales=3)
coeffs = haar2d.analysis(image)
recon = haar2d.synthesis(coeffs)
err = recon - image
max_err = np.abs(err).max()

print("[Wavelets 2D Haar Test]")
print(f"Max error after 3-scale decomposition: {max_err}")

fig, ax = plt.subplots(1, 3, figsize=(9, 3))
ax[0].imshow(image, cmap='gray')
ax[0].set_title("Original 32x32")

ax[1].imshow(recon, cmap='gray')
ax[1].set_title("Reconstructed from Haar Wavelets")

diffim = ax[2].imshow(err, cmap='bwr')
ax[2].set_title(f"Error (max={max_err:.3g})")
plt.colorbar(diffim, ax=ax[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

# %%
# Spline Wavelets (2D)
# --------------------
#
# Finally, we demonstrate 2D wavelet analysis/synthesis using spline wavelets of
# orders 1, 3, and 5. Each decomposition pass does row->column analysis, then
# column->row synthesis.

ny, nx = 32, 32
image = np.random.rand(ny, nx).astype(np.float32)*2 - 1.0

# We'll test 3 different spline wavelets: Spline1, Spline3, Spline5
wavelets_dict = {
    "Spline1": Spline1Wavelets(scales=3),
    "Spline3": Spline3Wavelets(scales=3),
    "Spline5": Spline5Wavelets(scales=3),
}

fig, axarr = plt.subplots(3, 3, figsize=(9, 9))

for idx, (name, wavelet) in enumerate(wavelets_dict.items()):
    coeffs = wavelet.analysis(image)
    recon = wavelet.synthesis(coeffs)
    err = recon - image
    max_err = np.abs(err).max()

    print(f"[Wavelets 2D {name} Test]")
    print(f"Max error after 3-scale decomposition: {max_err}")

    # Show original only once at the top-left
    if idx == 0:
        axarr[0,0].imshow(image, cmap='gray')
        axarr[0,0].set_title("Original 32x32")

    # Recon in row=idx, col=1
    axarr[idx,1].imshow(recon, cmap='gray')
    axarr[idx,1].set_title(f"{name} Reconstructed\nErr={max_err:.3g}")

    # Diff in row=idx, col=2
    im2 = axarr[idx,2].imshow(err, cmap='bwr')
    axarr[idx,2].set_title("Difference")
    plt.colorbar(im2, ax=axarr[idx,2], fraction=0.046, pad=0.04)

# Hide empty subplots in the first column (rows 1 and 2) since we only show original once
axarr[1,0].axis('off')
axarr[2,0].axis('off')

plt.tight_layout()
plt.show()