# sphinx_gallery_start_ignore
# splineops/examples/02_resampling_using_1d_interpolation/02_01_interpolate_1d_samples.py
# sphinx_gallery_end_ignore

"""
Interpolate 1D samples
======================

Interpolate 1D samples with standard interpolation.

1. Assume that a user-provided 1D list of samples :math:`f[k]` has been obtained by sampling a spline on a unit grid. 

2. From the samples, recover the continuously defined spline :math:`f(x)`.
"""

# %%
# Imports
# -------

import numpy as np
import matplotlib.pyplot as plt
from splineops.spline_interpolation.tensorspline import TensorSpline
from splineops.spline_interpolation.bases.utils import create_basis

plt.rcParams.update({
    "font.size": 14,       # Base font size
    "axes.titlesize": 18,  # Title font size
    "axes.labelsize": 16,  # Label font size
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# %%
# Helper for Basis Decomposition
# ------------------------------
#
# Reusable function to visualize a weighted sum of shifted spline bases
# for different spline families (e.g., cubic, linear).
#
# The grid is constructed with a rational step 1 / samples_per_unit so
# that all integer positions k are *exactly* included, i.e. we always
# sample each basis function at its center x = k.

def plot_basis_decomposition(
    f_support,
    f_samples,
    basis_name: str,
    title: str,
    samples_per_unit: int = 32,
    x_margin: float = 2.0,
):
    """
    Plot f[k] together with c_k * β(x - k) for a given spline basis.

    Parameters
    ----------
    f_support : array-like
        Integer sample positions k.
    f_samples : array-like
        Sample values f[k], reused here as illustrative weights c_k.
    basis_name : str
        Name of the spline basis (e.g., "bspline3", "bspline1").
    title : str
        Matplotlib figure title.
    samples_per_unit : int, optional
        Number of evaluation points per unit interval for the dense grid.
    x_margin : float, optional
        Extra domain added on both sides of the sample support.
    """
    basis = create_basis(basis_name)
    f_support_length = len(f_support)

    # Domain slightly beyond the data support: [0, K-1] expanded by x_margin
    x_min = -x_margin
    x_max = (f_support_length - 1) + x_margin

    # Dense grid with rational step so that all integers are hit exactly
    dx = 1.0 / samples_per_unit
    x_dense = np.arange(x_min, x_max + 0.5 * dx, dx, dtype=np.float64)

    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")
    plt.axhline(y=0, color="black", linewidth=1, zorder=0)

    # Plot each weighted, shifted basis function c_k * β(x - k)
    for k, c_k in enumerate(f_samples):
        y_basis = c_k * basis.eval(x_dense - k)
        plt.plot(x_dense, y_basis, linewidth=2, alpha=0.7)

    plt.xlabel("x")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# %%
# Initial 1D Samples
# ------------------
#
# We generate 1D samples and treat them as discrete signal points.
# 
# Let :math:`\mathbf{f} = (f[0], f[1], f[2], \dots, f[K-1])` be a 1D array of data.
#
# These are the input samples that we are going to interpolate.

number_of_samples = 27

f_support = np.arange(number_of_samples, dtype=np.float64)
f_support_length = len(f_support)  # It's equal to number_of_samples

f_samples = np.array([
    -0.657391, -0.641319, -0.613081, -0.518523, -0.453829, -0.385138,
    -0.270688, -0.179849, -0.11805, -0.0243016, 0.0130667, 0.0355389,
    0.0901577, 0.219599, 0.374669, 0.384896, 0.301386, 0.128646,
    -0.00811776, 0.0153119, 0.106126, 0.21688, 0.347629, 0.419532,
    0.50695, 0.544767, 0.555373
], dtype=np.float64)

plt.figure(figsize=(10, 4))
plt.title("f[k] samples")
plt.stem(f_support, f_samples, basefmt=" ")
# Add a black horizontal line at y=0:
plt.axhline(
    y=0,
    color="black",
    linewidth=1,
    zorder=0,
)
plt.xlabel("k")
plt.ylabel("f[k]")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Decomposition in Cubic Spline Basis
# -----------------------------------
#
# Visualize the shifted cubic B-spline basis functions β(x - k) that serve
# as the building blocks of the interpolant.
#
# In a true spline interpolation scheme, the spline coefficients c[k] are
# computed from the samples f[k] by a filtering operation, so in general
# c[k] ≠ f[k]. Here, we simply reuse f[k] as illustrative weights to show
# how a *linear combination* of shifted basis functions can form a smooth
# curve.

plot_basis_decomposition(
    f_support=f_support,
    f_samples=f_samples,
    basis_name="bspline3",
    title="f[k] samples with shifted cubic spline basis functions",
    samples_per_unit=32,
    x_margin=2.0,
)

# %%
# Interpolate the Samples with a Spline
# -------------------------------------
#
# We interpolate the 1D samples with a spline to obtain the continuously defined function
#
# .. math::
#
#    f(x) = \sum_{k\in{\mathbb{Z}}}\,c[k]\,\beta^{n}(x-k),
#
# where
#
# - the B-spline of degree :math:`n` is :math:`\beta^n`;
#
# - the spline coefficients :math:`c[k]` are determined from the input samples, such that :math:`f(k) = f[k]`.
#
# Let us now plot :math:`f`.

# Plot points
plot_points_per_unit = 12

# Interpolated signal
base = "bspline3"
mode = "mirror"

# %%
# TensorSpline
# ~~~~~~~~~~~~
#
# Here is one way to perform the standard interpolation.

f = TensorSpline(data=f_samples, coordinates=f_support, bases=base, modes=mode)

f_coords = np.array([
    q / plot_points_per_unit
    for q in range(plot_points_per_unit * f_support_length)
])

# Syntax hint: pass (plot_coords,) not plot_coords
f_data = f(coordinates=(f_coords,), grid=False)

# %%
# Resize Method
# ~~~~~~~~~~~~~
#
# The resize method with standard interpolation yields the same result.

from splineops.resize import resize

# We'll produce the same number of output samples as in f_coords
desired_length = plot_points_per_unit * f_support_length

# IMPORTANT: We explicitly define a coordinate array from 0..(f_support_length - 1)
# with `desired_length` points. This matches the domain and size that the `resize`
# function will produce below, ensuring the two outputs are sampled at the exact
# same x-positions, and thus comparable point-by-point.
f_coords_resize = np.linspace(0, f_support_length - 1, desired_length, dtype=np.float64)

f_data_resize = resize(
    data=f_samples,             # 1D input
    output_size=(desired_length,),
    method="cubic",             # ensures TensorSpline standard interpolation, not least-squares or oblique
)

# Ensure both arrays have identical shapes
f_data_spline = f(coordinates=(f_coords_resize,), grid=False)
assert f_data_spline.shape == f_data_resize.shape, "Arrays must match in shape."
mse_diff = np.mean((f_data_spline - f_data_resize)**2)
print(f"MSE between TensorSpline result and resize result = {mse_diff:.6e}")

# %%
# Plot of the Spline f
# ~~~~~~~~~~~~~~~~~~~~

plt.figure(figsize=(10, 4))
plt.title("f[k] samples with interpolated f spline")
plt.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")
# Add a black horizontal line at y=0:
plt.axhline(
    y=0,
    color="black",
    linewidth=1,
    zorder=0,  # draw behind other plot elements
)
plt.plot(f_coords_resize, f_data_resize, color="green", linewidth=2, label="f spline")
plt.xlabel("k")
plt.ylabel("f")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Linear Spline Basis Decomposition
# ---------------------------------
#
# Repeat the same visualization, now using the linear B-spline basis
# ("bspline1") instead of the cubic one. This highlights how a different
# choice of basis changes the local shape of the building blocks.
#
# As before, the true spline coefficients c[k] are obtained from f[k] by
# a filtering operation, so c[k] ≠ f[k] in general. We again reuse f[k]
# as illustrative weights.

plot_basis_decomposition(
    f_support=f_support,
    f_samples=f_samples,
    basis_name="bspline1",
    title="f[k] samples with shifted linear spline basis functions",
    samples_per_unit=32,
    x_margin=2.0,
)

# %%
# Interpolate the Samples with a Linear Spline
# --------------------------------------------
#
# We now build the interpolant using the linear B-spline basis ("bspline1")
# instead of the cubic one, and plot it together with the samples.

base_lin = "bspline1"
mode_lin = "mirror"

f_lin = TensorSpline(data=f_samples, coordinates=f_support, bases=base_lin, modes=mode_lin)

# Reuse the same dense coordinate grid as for the cubic case so that
# the two interpolants can be visually compared if desired.
f_data_lin = f_lin(coordinates=(f_coords_resize,), grid=False)

plt.figure(figsize=(10, 4))
plt.title("f[k] samples with interpolated f spline (linear basis)")
plt.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")
plt.axhline(
    y=0,
    color="black",
    linewidth=1,
    zorder=0,
)
plt.plot(f_coords_resize, f_data_lin, linewidth=2, label="f spline (linear)")
plt.xlabel("k")
plt.ylabel("f")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
