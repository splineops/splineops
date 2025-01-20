"""
Resizing 1D samples
===================

This example demonstrates how to perform 1D spline interpolation using the
`splineops` library, and how to downsample and re-expand a spline to measure
approximation quality. 

Specifically, we:

1. Interpolate an initial set of 1D samples with a B-spline to form a continuous function f(x).

2. Downsample f(x) by extracting fewer samples g[k] = f(λk).

3. Create a new spline g(x) from the discrete g[k].

4. Re-expand g(x) to match f's domain via h(x) = g(x / λ).

5. Compute the Mean Squared Error (MSE) between f and h to quantify the downsampling and re-expansion accuracy.

By the end, we visualize how closely h approximates f and see the 
effect of downsampling followed by spline-based reconstruction.

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
from splineops.interpolate.resize import resize
from splineops.interpolate.tensorspline import TensorSpline

plt.rcParams.update({
    "font.size": 14,     # Base font size
    "axes.titlesize": 18,  # Title font size
    "axes.labelsize": 16,  # Label font size
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# %%
# Initial 1D samples
# ------------------
#
# We generate 1D samples and treat them as discrete signal points.
# 
# Let :math:`\mathbf{f} = (f[0], f[1], f[2], \dots, f[K-1])` be a 1D array of data that are uniformly i.i.d. in (-1, 1).
#
# These are the input samples that we will interpolate.

f_support = np.arange(27)               # integer coordinates [0, 1, 2, ...]
np.random.seed(42)              # for reproducibility
f_samples = np.random.uniform(-1, 1, len(f_support))  # random in [-1, 1]

plt.figure(figsize=(10, 4))
plt.title("f[k] samples")
plt.stem(f_support, f_samples, basefmt=" ")
# Add a black horizontal line at y=0:
plt.axhline(
    y=0,
    color="black",
    linewidth=1,  # make it thicker if you like
    zorder=0      # draw behind other plot elements
)
plt.xlabel("k")
plt.ylabel("f[]")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Interpolate samples with spline f
# ---------------------------------
#
# We interpolate the 1D samples with a spline to obtain a continuously defined function f.
#
# Given the discrete samples :math:`f_{\text{samples}}(x_i)`, the spline interpolation :math:`f(x)` can be expressed as:
#
# .. math::
#
#    f(x) = \sum_{k\in{\mathbb{Z}}}\,c[k]\,\beta^{n}(x-k),
#
# where:
#
# - :math:`\beta^n` is the B-spline of degree :math:`n`.
#
# - :math:`c[k]` are the spline coefficients determined from the input samples.
#
# Let us now plot the continuously defined :math:`f(x)`.

# Plot points
plot_points_per_unit = 12

# Interpolated signal
bases = "bspline3"  # Linear interpolation
modes = "mirror"  # Mirror extension mode
f = TensorSpline(data=f_samples, coordinates=f_support, bases=bases, modes=modes)

plot_coords = np.array([q / plot_points_per_unit 
                        for q in range(plot_points_per_unit * len(f_support))])

# The key: pass (plot_coords,) not plot_coords
plot_data = f(coordinates=(plot_coords,), grid=False)

plt.figure(figsize=(10, 4))
plt.title("f[k] samples with interpolated f spline")
plt.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")
# Add a black horizontal line at y=0:
plt.axhline(
    y=0,
    color="black",
    linewidth=1,  # make it thicker if you like
    zorder=0      # draw behind other plot elements
)
plt.plot(plot_coords, plot_data, color="green", linewidth=2, label="f spline")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Coarsening of f
# ---------------

val_lambda = np.pi

g_support_length = round(len(f_support) // val_lambda)
f_resampled_coords = np.array([q * val_lambda for q in range(g_support_length)])
samples_of_g = f(coordinates=(f_resampled_coords,), grid=False)
support_of_g = np.arange(g_support_length)               # integer coordinates [0, 1, 2, ...]
g = TensorSpline(data=samples_of_g, coordinates=support_of_g, bases=bases, modes=modes)

plot_coords = np.array([q/plot_points_per_unit for q in range(plot_points_per_unit * len(support_of_g))])
plot_data = g(coordinates=(plot_coords,), grid=False)

fig = plt.figure(figsize=(12, 8))

gs = GridSpec(
    nrows=2, 
    ncols=2,
    # Match widths: first column = g_support_length, second column = leftover
    width_ratios=[g_support_length, len(f_support) - g_support_length],
    height_ratios=[1, 1]
)

# -- Top row: entire row (two columns combined)
ax_top = fig.add_subplot(gs[0, :])

# -- Bottom row: left side for g, right side blank
ax_bottom_left = fig.add_subplot(gs[1, 0])
ax_bottom_right = fig.add_subplot(gs[1, 1])
ax_bottom_right.axis("off")  # leave right side blank

#
# 1) TOP ROW: f[k] + f spline + discrete g[k]
#
ax_top.set_title("f[k] samples, interpolated f spline, and g[k] samples")

# Plot discrete f[k] as stems
ax_top.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")

# Plot continuous spline f(x) over x=0..(len(f_support)-1)
fine_x = np.linspace(0, len(f_support) - 1, 300)
fine_f = f(coordinates=(fine_x,), grid=False)
ax_top.plot(fine_x, fine_f, color="green", linewidth=2, label="f spline")

# Overplot discrete g[k] as unfilled red squares at x = k * val_lambda
x_g = np.arange(g_support_length) * val_lambda
ax_top.plot(
    x_g, 
    samples_of_g,
    "rs",              # red squares
    mfc='none',        # unfilled
    markersize=12,
    markeredgewidth=2, 
    label="g[k] samples"
)

# Horizontal line at 0 for reference
ax_top.axhline(0, color='black', linewidth=1, zorder=0)

# Make sure the top axis goes from 0..(len(f_support)-1)
ax_top.set_xlim(0, len(f_support) - 1)
ax_top.set_xticks(np.arange(0, len(f_support), 1))
ax_top.set_xlabel("x")
ax_top.set_ylabel("Amplitude")
ax_top.grid(True)
ax_top.legend()

#
# 2) BOTTOM LEFT: discrete g[k] + g spline
#
ax_bottom_left.set_title("g[k] samples and g spline")

# Plot discrete g[k] with red vertical lines and unfilled red squares
ax_bottom_left.vlines(
    x=support_of_g,
    ymin=0,
    ymax=samples_of_g,
    color='red',
    linestyle='-',
    linewidth=1
)
ax_bottom_left.plot(
    support_of_g,
    samples_of_g,
    "rs",              # red squares
    mfc='none',        # unfilled
    markersize=12,
    markeredgewidth=2,
    label="g[k] samples"
)

# Plot continuous g spline in purple over the same domain
plot_coords_g = np.linspace(0, g_support_length - 1, 200)
plot_data_g = g(coordinates=(plot_coords_g,), grid=False)
ax_bottom_left.plot(
    plot_coords_g, 
    plot_data_g,
    color="purple", 
    linewidth=2,
    label="g spline"
)

# Horizontal line at 0
ax_bottom_left.axhline(0, color='black', linewidth=1, zorder=0)

ax_bottom_left.set_xlim(0, g_support_length - 1)
ax_bottom_left.set_xticks(np.arange(0, g_support_length, 1))
ax_bottom_left.set_xlabel("x")
ax_bottom_left.set_ylabel("Amplitude")
ax_bottom_left.grid(True)
ax_bottom_left.legend()

# (Optional) match vertical scale with the top axis
# ax_bottom_left.set_ylim(ax_top.get_ylim())

fig.tight_layout()
plt.show()

# # %%
# # Coarsening of f
# # ---------------
# #
# #
# # We define :math:`\lambda` as a natural number and sample :math:`f(x)` 
# # at :math:`x = \lambda k`. Mathematically:
# #
# # .. math::
# #    g[k] = f(\lambda k).
# #
# # These :math:`g[k]` points form a new discrete set, which we will then treat 
# # as a separate signal to build another spline, :math:`g(x)`. Finally, to 
# # compare :math:`g` on the same domain as :math:`f`, we expand g by defining 
# # a new function :math:`h(x)`.
# #
# # .. math::
# #
# #    h(x) = g\bigl(\tfrac{x}{\lambda}\bigr),
# #
# # where :math:`g(\cdot)` is the continuous spline built from the :math:`g[k]` 
# # discrete points. Hence, :math:`h(x)` and :math:`f(x)` share the same domain 
# # and can be directly compared (e.g., by computing an MSE).

# lambda_val = high_res_factor  # sample every 'high_res_factor' points
# x_lambda = np.arange(x[0], x[-1] + 1, lambda_val)
# g_lambda = np.interp(x_lambda, x_high_res, resized_signal)  # sample from the high-resolution spline

# fig = plt.figure(figsize=(12, 9))  # increased height to accommodate 3 rows

# domain_length = x[-1] - x[0]        # e.g. 26 if x goes 0..26
# num_g_points = len(g_lambda)        # number of discrete g samples
# g_domain_length = num_g_points - 1  # e.g. 9 if len(g_lambda)=10

# # We'll have 3 rows × 2 columns:
# #   Row 0: entire top for f + g
# #   Row 1: left subplot for "shrunken" g, right is blank
# #   Row 2: entire bottom for h
# gs = GridSpec(nrows=3,
#     ncols=2,
#     width_ratios=[g_domain_length, domain_length - g_domain_length],
#     height_ratios=[1, 1, 1]  # three equal rows
# )

# # (1) TOP ROW: Original f + discrete g
# ax1 = fig.add_subplot(gs[0, :])  # spans both columns
# ax1.set_title("Original f[k] samples, f spline and g[k] samples")

# ax1.stem(x, original_samples, basefmt=" ", label="f[k] samples")
# ax1.plot(x_high_res, resized_signal, color="green", linewidth=2, label="f spline")
# ax1.plot(
#     x_lambda, g_lambda,
#     'rs', mfc='none', markersize=12, markeredgewidth=2,
#     label="g[k] samples"
# )

# ax1.set_xlim(x[0], x[-1])
# ax1.set_xticks(np.arange(x[0], x[-1] + 1, 1)) 
# ax1.set_xlabel("x")
# ax1.set_ylabel("Amplitude")
# ax1.grid(True)
# ax1.legend()

# # (2) MIDDLE ROW: Shrunken view of g

# ax2 = fig.add_subplot(gs[1, 0])  # bottom-left
# ax2.set_title("g[k] samples and interpolated g spline")

# k_values = np.arange(num_g_points)

# # Red vertical lines for each g[k]
# ax2.vlines(
#     k_values,
#     ymin=0,
#     ymax=g_lambda,
#     color='red',
#     linestyle='-',
#     linewidth=1
# )
# # Red hollow squares at each g[k]
# ax2.plot(
#     k_values, g_lambda,
#     'rs', mfc='none',
#     markersize=12, markeredgewidth=2,
#     label="g[k] samples"
# )

# # (Optional) A spline of g in purple
# new_length_for_g = (num_g_points - 1) * high_res_factor + 1
# g_spline = resize(
#     data=g_lambda,
#     output_size=(new_length_for_g,),
#     degree=degree,
#     method="interpolation"
# )
# k_high_res = np.linspace(0, num_g_points - 1, new_length_for_g)
# ax2.plot(
#     k_high_res, g_spline,
#     color='purple',
#     linewidth=2,
#     label="g spline"
# )

# ax2.set_xlim(0, num_g_points - 1)
# ax2.set_xticks(np.arange(0, num_g_points, 1))
# ax2.set_xlabel("x")
# ax2.set_ylabel("Amplitude")
# ax2.grid(True)
# ax2.legend()
# ax2.set_ylim(ax1.get_ylim())  # match top plot's amplitude range

# # Blank right cell
# ax_blank = fig.add_subplot(gs[1, 1])
# ax_blank.axis("off")

# ax3 = fig.add_subplot(gs[2, :])  # spans both columns
# ax3.set_title("Interpolated h spline over f spline domain, h(x) = g(x / λ)")

# # We'll sample k in [0..domain_length]
# k_full = np.arange(domain_length + 1)
# k_in_g = k_full / float(lambda_val)    # fractional index in g's domain

# # h[k] = g( k / λ )
# h_expanded = np.interp(k_in_g, k_high_res, g_spline)

# # Plot h in blue
# ax3.plot(
#     k_full, h_expanded,
#     color='blue',
#     linewidth=2,
#     label="h spline"
# )

# # Compute which g samples fit in [0..26]
# x_lambda_in_bounds = x_lambda[x_lambda <= domain_length]
# g_lambda_in_bounds = g_lambda[:len(x_lambda_in_bounds)]

# # Overlay vertical red lines, matching the squares
# ax3.vlines(
#     x_lambda_in_bounds,
#     ymin=0,
#     ymax=g_lambda_in_bounds,
#     color='red',
#     linestyle='-',
#     linewidth=1
# )

# # Also overlay the discrete g squares
# ax3.plot(
#     x_lambda_in_bounds,
#     g_lambda_in_bounds,
#     'rs', mfc='none',
#     markersize=12, markeredgewidth=2,
#     label="g[k] samples"
# )

# ax3.set_xlim(0, domain_length)  # 0..26
# ax3.set_xticks(np.arange(0, domain_length + 1, 1))
# ax3.set_xlabel("x")
# ax3.set_ylabel("Amplitude")
# ax3.grid(True)
# ax3.legend()

# # Keep amplitude scale consistent with top row
# ax3.set_ylim(ax1.get_ylim())

# for ax in [ax1, ax2, ax3]:
#     ax.axhline(0, color='black', linewidth=1, zorder=0)

# fig.tight_layout()
# plt.show()

# # %%
# # MSE Between f and h
# # -------------------
# #
# # To quantify how well :math:`h(x)` approximates :math:`f(x)`, we compute the 
# # Mean Squared Error (MSE) over the domain :math:`[a,b] = [0, 26]`:
# #
# # .. math::
# #    \text{MSE} = \frac{1}{b - a} \int_{a}^{b} [f(x) - h(x)]^2 \, dx.
# #
# # **Riemann Sum Approximation**:
# #
# # Instead of computing this integral analytically, we discretize the interval
# # :math:`[a,b]` into :math:`N` points. At each point :math:`x_i`, we evaluate
# # :math:`(f(x_i) - h(x_i))^2` and multiply by the small width :math:`\Delta x`.
# # Summing across all points approximates the integral:
# #
# # .. math::
# #    \int_{a}^{b} [f(x) - h(x)]^2 \, dx 
# #    \;\approx\; \Delta x \sum_{i=1}^{N} [f(x_i) - h(x_i)]^2.
# #
# # Dividing by :math:`b-a` yields the MSE.

# # 1) Define a fine sampling domain
# sample_count = 1000  # number of points for Riemann sum
# a, b = x[0], x[-1]   # 0..26
# fine_x = np.linspace(a, b, sample_count)

# # 2) Evaluate f(x) at this fine grid
# #    We already have f on (x_high_res, resized_signal), so we just interpolate:
# f_fine = np.interp(fine_x, x_high_res, resized_signal)

# # 3) Evaluate h(x) at this same fine grid
# #    h(x) = g_spline(x / lambda_val).
# #    Recall we have k_high_res, g_spline from the bottom subplot.
# h_fine = np.interp(fine_x / lambda_val, k_high_res, g_spline)

# # 4) Compute the Riemann sum for ∫(f(x)-h(x))^2 dx over [a, b]
# #    Here we use a simple rectangular rule with spacing dx:
# dx = (b - a) / (sample_count - 1)
# integral_value = np.sum((f_fine - h_fine)**2) * dx

# # 5) Divide by (b - a) to get the MSE
# mse_riemann = integral_value / (b - a)

# print(f"MSE between f and h (via Riemann sum) = {mse_riemann:.6e}")

# # (Optional) Quick check with a simple discrete mean of squared errors
# # over the same 1D samples (not an integral, but a discrete approximation):
# mse_check = np.mean((f_fine - h_fine)**2)
# print(f"MSE check (discrete mean over {sample_count} samples) = {mse_check:.6e}")
