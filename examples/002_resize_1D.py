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

# %%
# Initial 1D samples
# ------------------
#
# We generate 1D samples and treat them as discrete signal points.
# 
# Let :math:`\mathbf{x} = [x_1, x_2, \dots, x_N]` be a set of 1D sampled points, and let the discrete signal
# :math:`f_{\text{samples}}(x)` be defined as random values within a specified range:
#
# .. math::
#
#    f_{\text{samples}}(x_i) \sim \text{Uniform}(-1, 1), \quad i = 1, \dots, N.
#
# These are the input samples that we will interpolate.

x = np.arange(27)               # integer coordinates [0, 1, 2, ...]
np.random.seed(42)              # for reproducibility
original_samples = np.random.uniform(-1, 1, len(x))  # random in [-1, 1]

plt.figure(figsize=(10, 4))
plt.title("Original f[k] samples")
plt.stem(x, original_samples, basefmt=" ")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Interpolate samples with spline f
# ---------------------------------
#
# We interpolate the 1D samples with a spline to obtain a continuous function f.
#
# Given the discrete samples :math:`f_{\text{samples}}(x_i)`, the spline interpolation :math:`f(x)` can be expressed as:
#
# .. math::
#
#    f(x) = \sum_{k} c_k \beta_n(x - k),
#
# where:
# - :math:`\beta_n` is the B-spline of degree :math:`n`.
# - :math:`c_k` are the spline coefficients determined from the input samples.
#
# By choosing a sufficiently fine grid, we approximate a continuous function :math:`f` from the discrete samples.

degree = 3
high_res_factor = 3
new_length = len(original_samples) * high_res_factor

# Interpolated signal
resized_signal = resize(
    data=original_samples,
    output_size=(new_length,),
    degree=degree,
    method="interpolation"
)

x_high_res = np.linspace(x[0], x[-1], new_length)

plt.figure(figsize=(10, 4))
plt.title("Original f[k] samples with interpolated f spline")
plt.stem(x, original_samples, basefmt=" ", label="f[k] samples")
plt.plot(x_high_res, resized_signal, color="green", linewidth=2, label="f spline")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Sampling f to get g and h
# -------------------------
#
#
# We define :math:`\lambda` as a natural number and sample :math:`f(x)` 
# at :math:`x = \lambda k`. Mathematically:
#
# .. math::
#    g[k] = f(\lambda k).
#
# These :math:`g[k]` points form a new discrete set, which we will then treat 
# as a separate signal to build another spline, :math:`g(x)`. Finally, to 
# compare :math:`g` on the same domain as :math:`f`, we expand g by defining 
# a new function :math:`h(x)`.
#
# .. math::
#
#    h(x) = g\bigl(\tfrac{x}{\lambda}\bigr),
#
# where :math:`g(\cdot)` is the continuous spline built from the :math:`g[k]` 
# discrete points. Hence, :math:`h(x)` and :math:`f(x)` share the same domain 
# and can be directly compared (e.g., by computing an MSE).

lambda_val = high_res_factor  # sample every 'high_res_factor' points
x_lambda = np.arange(x[0], x[-1] + 1, lambda_val)
g_lambda = np.interp(x_lambda, x_high_res, resized_signal)  # sample from the high-resolution spline

fig = plt.figure(figsize=(12, 9))  # increased height to accommodate 3 rows

domain_length = x[-1] - x[0]        # e.g. 26 if x goes 0..26
num_g_points = len(g_lambda)        # number of discrete g samples
g_domain_length = num_g_points - 1  # e.g. 9 if len(g_lambda)=10

# We'll have 3 rows × 2 columns:
#   Row 0: entire top for f + g
#   Row 1: left subplot for "shrunken" g, right is blank
#   Row 2: entire bottom for h
gs = GridSpec(nrows=3,
    ncols=2,
    width_ratios=[g_domain_length, domain_length - g_domain_length],
    height_ratios=[1, 1, 1]  # three equal rows
)

# (1) TOP ROW: Original f + discrete g
ax1 = fig.add_subplot(gs[0, :])  # spans both columns
ax1.set_title("Original f[k] samples, f spline and g[k] samples")

ax1.stem(x, original_samples, basefmt=" ", label="f[k] samples")
ax1.plot(x_high_res, resized_signal, color="green", linewidth=2, label="f spline")
ax1.plot(
    x_lambda, g_lambda,
    'rs', mfc='none', markersize=12, markeredgewidth=2,
    label="g[k] samples"
)

ax1.set_xlim(x[0], x[-1])
ax1.set_xticks(np.arange(x[0], x[-1] + 1, 1)) 
ax1.set_xlabel("x")
ax1.set_ylabel("Amplitude")
ax1.grid(True)
ax1.legend()

# (2) MIDDLE ROW: Shrunken view of g

ax2 = fig.add_subplot(gs[1, 0])  # bottom-left
ax2.set_title("g[k] samples and interpolated g spline")

k_values = np.arange(num_g_points)

# Red vertical lines for each g[k]
ax2.vlines(
    k_values,
    ymin=0,
    ymax=g_lambda,
    color='red',
    linestyle='-',
    linewidth=1
)
# Red hollow squares at each g[k]
ax2.plot(
    k_values, g_lambda,
    'rs', mfc='none',
    markersize=12, markeredgewidth=2,
    label="g[k] samples"
)

# (Optional) A spline of g in purple
new_length_for_g = (num_g_points - 1) * high_res_factor + 1
g_spline = resize(
    data=g_lambda,
    output_size=(new_length_for_g,),
    degree=degree,
    method="interpolation"
)
k_high_res = np.linspace(0, num_g_points - 1, new_length_for_g)
ax2.plot(
    k_high_res, g_spline,
    color='purple',
    linewidth=2,
    label="g spline"
)

ax2.set_xlim(0, num_g_points - 1)
ax2.set_xticks(np.arange(0, num_g_points, 1))
ax2.set_xlabel("x")
ax2.set_ylabel("Amplitude")
ax2.grid(True)
ax2.legend()
ax2.set_ylim(ax1.get_ylim())  # match top plot's amplitude range

# Blank right cell
ax_blank = fig.add_subplot(gs[1, 1])
ax_blank.axis("off")

ax3 = fig.add_subplot(gs[2, :])  # spans both columns
ax3.set_title("Interpolated h spline over f spline domain, h(x) = g(x / λ)")

# We'll sample k in [0..domain_length]
k_full = np.arange(domain_length + 1)
k_in_g = k_full / float(lambda_val)    # fractional index in g's domain

# h[k] = g( k / λ )
h_expanded = np.interp(k_in_g, k_high_res, g_spline)

# Plot h in blue
ax3.plot(
    k_full, h_expanded,
    color='blue',
    linewidth=2,
    label="h spline"
)

# Compute which g samples fit in [0..26]
x_lambda_in_bounds = x_lambda[x_lambda <= domain_length]
g_lambda_in_bounds = g_lambda[:len(x_lambda_in_bounds)]

# Overlay vertical red lines, matching the squares
ax3.vlines(
    x_lambda_in_bounds,
    ymin=0,
    ymax=g_lambda_in_bounds,
    color='red',
    linestyle='-',
    linewidth=1
)

# Also overlay the discrete g squares
ax3.plot(
    x_lambda_in_bounds,
    g_lambda_in_bounds,
    'rs', mfc='none',
    markersize=12, markeredgewidth=2,
    label="g[k] samples"
)

ax3.set_xlim(0, domain_length)  # 0..26
ax3.set_xticks(np.arange(0, domain_length + 1, 1))
ax3.set_xlabel("x")
ax3.set_ylabel("Amplitude")
ax3.grid(True)
ax3.legend()

# Keep amplitude scale consistent with top row
ax3.set_ylim(ax1.get_ylim())

fig.tight_layout()
plt.show()

# %%
# MSE Between f and h
# -------------------
#
# To quantify how well :math:`h(x)` approximates :math:`f(x)`, we compute the 
# Mean Squared Error (MSE) over the domain :math:`[a,b] = [0, 26]`:
#
# .. math::
#    \text{MSE} = \frac{1}{b - a} \int_{a}^{b} [f(x) - h(x)]^2 \, dx.
#
# **Riemann Sum Approximation**:
#
# Instead of computing this integral analytically, we discretize the interval
# :math:`[a,b]` into :math:`N` points. At each point :math:`x_i`, we evaluate
# :math:`(f(x_i) - h(x_i))^2` and multiply by the small width :math:`\Delta x`.
# Summing across all points approximates the integral:
#
# .. math::
#    \int_{a}^{b} [f(x) - h(x)]^2 \, dx 
#    \;\approx\; \Delta x \sum_{i=1}^{N} [f(x_i) - h(x_i)]^2.
#
# Dividing by :math:`b-a` yields the MSE.

# 1) Define a fine sampling domain
sample_count = 1000  # number of points for Riemann sum
a, b = x[0], x[-1]   # 0..26
fine_x = np.linspace(a, b, sample_count)

# 2) Evaluate f(x) at this fine grid
#    We already have f on (x_high_res, resized_signal), so we just interpolate:
f_fine = np.interp(fine_x, x_high_res, resized_signal)

# 3) Evaluate h(x) at this same fine grid
#    h(x) = g_spline(x / lambda_val).
#    Recall we have k_high_res, g_spline from the bottom subplot.
h_fine = np.interp(fine_x / lambda_val, k_high_res, g_spline)

# 4) Compute the Riemann sum for ∫(f(x)-h(x))^2 dx over [a, b]
#    Here we use a simple rectangular rule with spacing dx:
dx = (b - a) / (sample_count - 1)
integral_value = np.sum((f_fine - h_fine)**2) * dx

# 5) Divide by (b - a) to get the MSE
mse_riemann = integral_value / (b - a)

print(f"MSE between f and h (via Riemann sum) = {mse_riemann:.6e}")

# (Optional) Quick check with a simple discrete mean of squared errors
# over the same 1D samples (not an integral, but a discrete approximation):
mse_check = np.mean((f_fine - h_fine)**2)
print(f"MSE check (discrete mean over {sample_count} samples) = {mse_check:.6e}")
