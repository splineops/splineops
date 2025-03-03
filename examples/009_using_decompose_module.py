#!/usr/bin/env python3
"""
009_using_decompose_module.py

Demonstrates how to use the 'decompose' module for:
- Pyramid decomposition (reduce & expand) in 1D and 2D
- Wavelet decomposition (analysis & synthesis), specifically:
  * Spline wavelets in 1D
  * Haar wavelets in 2D

We replicate and visualize the test logic from:
  - test_1d.c (creating a 1D signal, reducing, expanding, checking error)
  - test_2d.c (creating a 2D signal, reducing, expanding, checking error)
"""

import numpy as np
import matplotlib.pyplot as plt

# Pyramid functionality
from splineops.decompose.pyramid import (
    get_pyramid_filter,
    reduce_1d, expand_1d,
    reduce_2d, expand_2d
)

# Wavelets:
#  - HaarWavelets is your 2D Haar wavelet class that requires shape (ny,nx) with ny>=2,nx>=2
#  - SplineWavelets is your Spline wavelet code, which can handle shape (1,N) but 
#    also is typically 2D. We'll demonstrate Spline in 1D for convenience.
from splineops.decompose.wavelets.haar import HaarWavelets  # the 2D version
from splineops.decompose.wavelets.splinewavelets import SplineWavelets


##############################################################################
# 1. Test 1D Pyramid (replicates logic of test_1d.c)
##############################################################################

def test_1d_pyramid():
    """
    Build a 1D signal, reduce & expand it using a pyramid filter, then
    plot the results and print the error.
    """

    # 1) Create an input 1D signal (length=10, as in test_1d.c)
    x = np.array([0.0, 1.0, 2.0, 3.0, 2.0, 1.0, 0.0, -2.0, -4.0, -6.0], dtype=np.float64)

    # 2) Get a filter (example: "Centered Spline" of order 3)
    filter_name = "Centered Spline"
    order = 3
    g, h, is_centered = get_pyramid_filter(filter_name, order)

    # 3) Reduce and expand
    reduced = reduce_1d(x, g, is_centered)
    expanded = expand_1d(reduced, h, is_centered)

    # 4) Compute error
    error = expanded - x

    # 5) Print results
    print("[1D Pyramid Test]")
    print(f"Filter: '{filter_name}' (order={order}), is_centered={is_centered}")
    print("Input   x:", x)
    print("Reduced   :", reduced)
    print("Expanded  :", expanded)
    print("Error     :", error)

    # 6) Plot
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8,6))
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


##############################################################################
# 2. Test 2D Pyramid (replicates logic of test_2d.c)
##############################################################################

def test_2d_pyramid():
    """
    Build a small 4x4 image, reduce & expand it using a pyramid filter, 
    then display the results and print the error.
    """
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

    fig, ax = plt.subplots(1, 3, figsize=(10,3))
    ax[0].imshow(arr, cmap='viridis', vmin=arr.min(), vmax=arr.max())
    ax[0].set_title("Original 4x4")

    ax[1].imshow(expanded_2d, cmap='viridis', vmin=arr.min(), vmax=arr.max())
    ax[1].set_title("Expanded from Reduced")

    im2 = ax[2].imshow(error_2d, cmap='bwr')
    ax[2].set_title(f"Error (max={max_err:.2g})")
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


##############################################################################
# 3. Demonstrate Wavelets usage
##############################################################################

def test_wavelets_1d_spline():
    """
    Simple test of wavelet analysis/synthesis in 1D using only Spline wavelets.
    (We skip Haar in 1D because HaarWavelets is purely 2D.)
    """
    x = np.linspace(0, 10, 16, dtype=np.float32)
    x[8:] -= 5.0  # partial negative

    # Use Spline wavelet
    spline3 = SplineWavelets(scales=2, order=3)
    # shape (1,16) so the wavelet code sees (ny=1, nx=16)
    w_spl3 = spline3.analysis(x.reshape(1,-1))
    x_spl3_rec = spline3.synthesis(w_spl3)[0,:]
    err_spl3 = x_spl3_rec - x

    print("[Wavelets 1D Spline Test]")
    print("Spline3 reconstruction error (max):", np.abs(err_spl3).max())

    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(7,5))
    axs[0].plot(x, 'o-', label='Original 1D Signal')
    axs[0].set_title("Original 1D Signal")
    axs[0].legend()

    axs[1].plot(x_spl3_rec, 'o--', label='Reconstructed (Spline3)')
    axs[1].plot(x, color='k', alpha=0.3)
    axs[1].set_title(f"Spline3: max error={np.abs(err_spl3).max():.3g}")
    axs[1].legend()

    plt.tight_layout()
    plt.show()


def test_wavelets_2d_haar():
    """
    Simple test of wavelet analysis/synthesis in 2D using Haar wavelets
    (the 2D version that requires ny>=2, nx>=2).
    """
    # Create a random 2D signal of shape (32,32), both dims >=2
    image = np.random.rand(32,32).astype(np.float32)*2 - 1.0  # negative & positive

    haar2d = HaarWavelets(scales=3)
    coeffs = haar2d.analysis(image)
    recon = haar2d.synthesis(coeffs)
    err = recon - image
    max_err = np.abs(err).max()

    print("[Wavelets 2D Haar Test]")
    print(f"Max error after 3-scale decomposition: {max_err}")

    fig, ax = plt.subplots(1,3, figsize=(9,3))
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title("Original 32x32")

    ax[1].imshow(recon, cmap='gray')
    ax[1].set_title("Reconstructed from Wavelets")

    diffim = ax[2].imshow(err, cmap='bwr')
    ax[2].set_title(f"Error (max={max_err:.3g})")
    plt.colorbar(diffim, ax=ax[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


##############################################################################
# Main driver
##############################################################################

if __name__ == "__main__":
    # 1) Pyramid tests
    test_1d_pyramid()
    test_2d_pyramid()

    # 2) Wavelets tests
    #    We do a 1D Spline test, and a 2D Haar test
    test_wavelets_1d_spline()
    test_wavelets_2d_haar()
