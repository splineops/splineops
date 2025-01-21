r"""
Interpolating 1D samples
========================

Interpolate 1D samples with standard interpolation.

Specifically, we:

1. Interpolate an initial set of 1D samples :math:`f[k]`, placed on a unit grid with a B-spline to form a continuously defined function :math:`f(x)`.

2. Resample :math:`f(x)` to get :math:`g[k] = f(\lambda k)`, with :math:`|\lambda| > 1`.

3. Create a new spline :math:`g(x)`.

4. We define :math:`h(x) = g(x / \lambda)`.

5. Compute the Mean Squared Error (MSE) between :math:`f` and :math:`h`.

You can download this example at the tab at right, as both a Python script and as a Jupyter notebook.
"""

# %%
# Import required libraries
# -------------------------
#
# We import the required libraries, including numpy for numerical computations,
# Matplotlib for plotting, and the `splineops` package.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
# Let :math:`\mathbf{f} = (f[0], f[1], f[2], \dots, f[K-1])` be a 1D array of data that are uniformly i.i.d. in :math:`(-1, 1)`.
#
# These are the input samples that we will interpolate.

number_of_samples = 27

f_support = np.arange(number_of_samples)
f_support_length = len(f_support) # It's equal to number_of_samples
np.random.seed(42)
f_samples = np.random.uniform(-1, 1, f_support_length)

plt.figure(figsize=(10, 4))
plt.title("f[k] samples")
plt.stem(f_support, f_samples, basefmt=" ")
# Add a black horizontal line at y=0:
plt.axhline(
    y=0,
    color="black",
    linewidth=1,
    zorder=0
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
# We interpolate the 1D samples with a spline to obtain the continuously defined function
#
# .. math::
#
#    f(x) = \sum_{k\in{\mathbb{Z}}}\,c[k]\,\beta^{n}(x-k),
#
# where:
#
# - :math:`\beta^n` is the B-spline of degree :math:`n`;
#
# - :math:`c[k]` are the spline coefficients determined from the input samples, such that :math:`f(k) = f[k]`.
#
# Let us now plot :math:`f`.

# Plot points
plot_points_per_unit = 12

# Interpolated signal
base = "bspline3"
mode = "mirror"

f = TensorSpline(data=f_samples, coordinates=f_support, bases=base, modes=mode)

f_coords = np.array([q / plot_points_per_unit 
                        for q in range(plot_points_per_unit * f_support_length)])

# Syntax hint: pass (plot_coords,) not plot_coords
f_data = f(coordinates=(f_coords,), grid=False)

plt.figure(figsize=(10, 4))
plt.title("f[k] samples with interpolated f spline")
plt.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")
# Add a black horizontal line at y=0:
plt.axhline(
    y=0,
    color="black",
    linewidth=1,
    zorder=0 # draw behind other plot elements
)
plt.plot(f_coords, f_data, color="green", linewidth=2, label="f spline")
plt.xlabel("x")
plt.ylabel("f")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
# Coarsening of f
# ---------------
# We define :math:`\lambda` with :math:`|\lambda| > 1` and sample :math:`f(x)` 
# at :math:`x = \lambda k`:
#
# .. math::
#    g[k] = f(\lambda k).
#
# These :math:`g[k]` points form a new discrete set, which we will then treat 
# as a separate signal to build another spline, :math:`g`.

val_lambda = np.pi

g_support_length = round(f_support_length // val_lambda)
g_support = np.arange(g_support_length)  
f_resampled_coords = np.array([q * val_lambda for q in range(g_support_length)])
g_samples = f(coordinates=(f_resampled_coords,), grid=False)
g = TensorSpline(data=g_samples, coordinates=g_support, bases=base, modes=mode)

g_coords = np.array([q/plot_points_per_unit 
                     for q in range(plot_points_per_unit * len(g_support))])
g_data = g(coordinates=(g_coords,), grid=False)

fig = plt.figure(figsize=(12, 8))

gs = GridSpec(
    nrows=2, 
    ncols=2,
    # Match widths: first column = g_support_length, second column = leftover
    width_ratios=[g_support_length, f_support_length - g_support_length],
    height_ratios=[1, 1]
)

# Top row: entire row (two columns combined)
ax_top = fig.add_subplot(gs[0, :])

# Bottom row: left side for g, right side blank
ax_bottom_left = fig.add_subplot(gs[1, 0])
ax_bottom_right = fig.add_subplot(gs[1, 1])
ax_bottom_right.axis("off")  # leave right side blank

# 1) TOP ROW: f[k] + f spline + discrete g[k]
ax_top.set_title("Interpolated f spline")

# Plot discrete f[k] as stems
ax_top.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")

# Plot spline f(x)
ax_top.plot(f_coords, f_data, color="green", linewidth=2, label="f spline")

# Overplot discrete g[k] as unfilled red squares at x = k * val_lambda
x_g = np.arange(g_support_length) * val_lambda
ax_top.plot(
    x_g, 
    g_samples,
    "rs",              # red squares
    mfc='none',        # unfilled
    markersize=12,
    markeredgewidth=2, 
    label="g[k] samples"
)

# Horizontal line at 0 for reference
ax_top.axhline(0, color='black', linewidth=1, zorder=0)

# Make sure the top axis goes from 0..(f_support_length-1)
ax_top.set_xlim(0, f_support_length - 1)
ax_top.set_xticks(np.arange(0, f_support_length, 1))
ax_top.set_xlabel("x")
ax_top.set_ylabel("f")
ax_top.grid(True)
ax_top.legend()

# 2) BOTTOM LEFT: discrete g[k] + g spline
ax_bottom_left.set_title("Interpolated g spline")

# Plot discrete g[k] with red vertical lines and unfilled red squares
ax_bottom_left.vlines(
    x=g_support,
    ymin=0,
    ymax=g_samples,
    color='red',
    linestyle='-',
    linewidth=1
)
ax_bottom_left.plot(
    g_support,
    g_samples,
    "rs",              # red squares
    mfc='none',        # unfilled
    markersize=12,
    markeredgewidth=2,
    label="g[k] samples"
)

# Plot g spline in purple over the same domain
ax_bottom_left.plot(
    g_coords, 
    g_data,
    color="purple", 
    linewidth=2,
    label="g spline"
)

# Horizontal line at 0
ax_bottom_left.axhline(0, color='black', linewidth=1, zorder=0)

ax_bottom_left.set_xlim(0, g_support_length - 1)
ax_bottom_left.set_xticks(np.arange(0, g_support_length, 1))
ax_bottom_left.set_xlabel("x")
ax_bottom_left.set_ylabel("g")
ax_bottom_left.grid(True)
ax_bottom_left.legend()

# Match vertical scale with the top axis
ax_bottom_left.set_ylim(ax_top.get_ylim())

fig.tight_layout()
plt.show()

# %%
# Expanding g to obtain h
# -----------------------
#
# To compare :math:`g` on the same domain as :math:`f`, we expand :math:`g` by defining 
# a new function :math:`h`:
#
# .. math::
#
#    h(x) = g\bigl(\tfrac{x}{\lambda}\bigr),
#
# where :math:`g` is the continuously defined spline built from the :math:`g[k]` 
# discrete points. Hence, :math:`h` and :math:`f` have the same support 
# and can be directly compared (e.g., by computing an MSE).

fig2 = plt.figure(figsize=(12, 12))

gs2 = GridSpec(
    nrows=3,
    ncols=2,
    width_ratios=[g_support_length, f_support_length - g_support_length],
    height_ratios=[1, 1, 1]  # three equal rows
)

# TOP ROW: f + f spline + discrete g[k]
ax_top = fig2.add_subplot(gs2[0, :])  # spans both columns
ax_top.set_title("Interpolated f spline")

# Replot discrete f[k] as stems
ax_top.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")

# Replot f spline
ax_top.plot(f_coords, f_data, color="green", linewidth=2, label="f spline")

# Overplot discrete g[k] in red squares at x = k*val_lambda
x_g = np.arange(g_support_length) * val_lambda
ax_top.plot(
    x_g, g_samples,
    "rs", mfc='none', markersize=12, markeredgewidth=2,
    label="g[k] samples"
)

# Horizontal line at 0
ax_top.axhline(0, color="black", linewidth=1, zorder=0)

ax_top.set_xlim(0, f_support_length - 1)
ax_top.set_xticks(np.arange(0, f_support_length, 1))
ax_top.set_xlabel("x")
ax_top.set_ylabel("f")
ax_top.legend()
ax_top.grid(True)

# MIDDLE ROW: discrete g + g spline
ax_mid_left = fig2.add_subplot(gs2[1, 0])  # left cell
ax_mid_right = fig2.add_subplot(gs2[1, 1]) # right cell
ax_mid_right.axis("off")                  # keep it blank

ax_mid_left.set_title("Interpolated g spline")

# Plot discrete g[k] with red stems, unfilled squares
ax_mid_left.vlines(
    x=g_support,
    ymin=0,
    ymax=g_samples,
    color='red',
    linestyle='-',
    linewidth=1
)
ax_mid_left.plot(
    g_support,
    g_samples,
    "rs", mfc='none', markersize=12, markeredgewidth=2,
    label="g[k] samples"
)

# Plot g spline
ax_mid_left.plot(
    g_coords, g_data,
    color="purple", linewidth=2,
    label="g spline"
)

ax_mid_left.axhline(0, color='black', linewidth=1, zorder=0)
ax_mid_left.set_xlim(0, g_support_length - 1)
ax_mid_left.set_xticks(np.arange(0, g_support_length, 1))
ax_mid_left.set_xlabel("x")
ax_mid_left.set_ylabel("g")
ax_mid_left.legend()
ax_mid_left.grid(True)

# Match y-limits with top row:
ax_mid_left.set_ylim(ax_top.get_ylim())

# BOTTOM ROW: expanded h(x) = g(x / λ)
ax_bottom = fig2.add_subplot(gs2[2, :])  # spans both columns
ax_bottom.set_title("h spline, h(x) = g(x / λ)")

# We'll sample h over 0..(f_support_length-1)
h_coords = f_coords
# Evaluate h(x) = g(x/val_lambda)
h_data = g(coordinates=(h_coords / val_lambda,), grid=False)

# Plot h in blue
ax_bottom.plot(h_coords, h_data, color="blue", linewidth=2, label="h(x)")

# Horizontal line at 0
ax_bottom.axhline(0, color='black', linewidth=1, zorder=0)

# The domain is the same as f
ax_bottom.set_xlim(0, f_support_length - 1)
ax_bottom.set_xticks(np.arange(0, f_support_length, 1))
ax_bottom.set_xlabel("x")
ax_bottom.set_ylabel("h")
ax_bottom.grid(True)
ax_bottom.legend()

# Match y axis with top row
ax_bottom.set_ylim(ax_top.get_ylim())

fig2.tight_layout()
plt.show()

# %%
# MSE between f and h
# -------------------
#
# We compute the Mean Squared Error (MSE) between :math:`h(x)` and :math:`f(x)`:
#
# .. math::
#    \text{MSE} = \frac{1}{b - a} \int_{a}^{b} (f(x) - h(x))^2 \, \mathrm{d}x.
#
# **Riemann Rule Approximation**:
#
# Instead of computing this integral analytically, we discretize the interval
# :math:`[a,b]` into :math:`K` points. At each point :math:`x_k`, we evaluate
# :math:`(f(x_k) - h(x_k))^2` and multiply by the small width :math:`\Delta x`.
# Summing across all points approximates the integral:
#
# .. math::
#    \int_{a}^{b} (f(x) - h(x))^2 \, \mathrm{d}x 
#    \;\approx\; \Delta x \sum_{k=1}^{K} (f(x_k) - h(x_k))^2.
#
# Dividing by :math:`b-a` yields the MSE.

# 1) Define a midpoint sampling domain for [a, b]
N = 1000
padding_fraction = 0.2 # We avoid artifacts near the edges by excluding part of the domain from each side
a = (f_support_length - 1) * padding_fraction
b = (f_support_length - 1) * (1 - padding_fraction)
dx = (b - a) / N
mid_x = np.linspace(a + dx/2, b - dx/2, N)  # midpoints

# 2) Evaluate f(x) and h(x) at those midpoints
f_mid = f(coordinates=(mid_x,), grid=False)
h_mid = g(coordinates=(mid_x / val_lambda,), grid=False)

# 3) Compute the midpoint Riemann sum for ∫(f(x)-h(x))^2 dx
squared_diff = (f_mid - h_mid) ** 2
integral_value = np.sum(squared_diff) * dx

# 4) Divide by (b - a) to get the MSE
mse_midpoint = integral_value / (b - a)
print(f"MSE between f and h = {mse_midpoint:.6e}")

# %%
# Variation using linear splines
# ------------------------------
#
# We repeat exactly everything but using linear splines.

base = "bspline1"
mode = "mirror"

# 1) Rebuild linear-spline version of f, then g
f_lin = TensorSpline(data=f_samples, coordinates=f_support, bases=base, modes=mode)
g_lin_samps = f_lin(coordinates=(f_resampled_coords,), grid=False)
g_lin = TensorSpline(data=g_lin_samps, coordinates=g_support, bases=base, modes=mode)

# 2) Evaluate them at the same plotting coordinates
f_lin_f = f_lin(coordinates=(f_coords,), grid=False)               # f_lin over domain 0..(K-1)
g_lin_g = g_lin(coordinates=(g_coords,), grid=False)               # g_lin over domain 0..(g_support_length-1)
h_lin_h = g_lin(coordinates=(f_coords / val_lambda,), grid=False)  # h_lin(x)=g_lin(x/λ) over 0..(K-1)

# 3) Create the 3×2 figure layout
fig3 = plt.figure(figsize=(12, 12))
gs3 = GridSpec(
    nrows=3,
    ncols=2,
    width_ratios=[g_support_length, f_support_length - g_support_length],
    height_ratios=[1, 1, 1]
)

# TOP ROW: entire row (two columns combined) for f
ax_top = fig3.add_subplot(gs3[0, :])
ax_top.set_title("Linear f spline")

ax_top.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")
ax_top.plot(f_coords, f_lin_f, color="green", linewidth=2, label="f")
ax_top.axhline(0, color='black', linewidth=1, zorder=0)
ax_top.set_xlim(0, f_support_length - 1)
ax_top.set_xticks(np.arange(0, f_support_length, 1))
ax_top.set_xlabel("x")
ax_top.set_ylabel("f")
ax_top.grid(True)
ax_top.legend()

# MIDDLE ROW: left subplot shows g domain, right subplot is blank
ax_mid_left = fig3.add_subplot(gs3[1, 0])
ax_mid_right = fig3.add_subplot(gs3[1, 1])
ax_mid_right.axis("off")  # keep right side blank

ax_mid_left.set_title("Linear g spline")

# Discrete g[k] with stems
ax_mid_left.vlines(
    x=g_support,
    ymin=0,
    ymax=g_lin_samps,
    color='red',
    linewidth=1
)
ax_mid_left.plot(
    g_support,
    g_lin_samps,
    "rs", mfc='none', markersize=8, markeredgewidth=2, 
    label="g[k]"
)
# g spline
ax_mid_left.plot(
    g_coords,
    g_lin_g,
    color="purple", linewidth=2,
    label="g"
)
ax_mid_left.axhline(0, color='black', linewidth=1, zorder=0)
ax_mid_left.set_xlim(0, g_support_length - 1)
ax_mid_left.set_xticks(np.arange(0, g_support_length, 1))
ax_mid_left.set_xlabel("x")
ax_mid_left.set_ylabel("g")
ax_mid_left.grid(True)
ax_mid_left.legend()
# Match y-range with top
ax_mid_left.set_ylim(ax_top.get_ylim())

# BOTTOM ROW: entire row for h
ax_bottom = fig3.add_subplot(gs3[2, :])
ax_bottom.set_title("Linear h spline, h(x)=g(x/λ)")

ax_bottom.plot(f_coords, h_lin_h, color="blue", linewidth=2, label="h")
ax_bottom.axhline(0, color='black', linewidth=1, zorder=0)
ax_bottom.set_xlim(0, f_support_length - 1)
ax_bottom.set_xticks(np.arange(0, f_support_length, 1))
ax_bottom.set_xlabel("x")
ax_bottom.set_ylabel("h")
ax_bottom.grid(True)
ax_bottom.legend()
# Match y-range
ax_bottom.set_ylim(ax_top.get_ylim())

fig3.tight_layout()
plt.show()

# 4) Recompute MSE with linear splines using midpoint rule
N = 1000
padding_fraction = 0.2 # We avoid artifacts near the edges by excluding part of the domain from each side
a = (f_support_length - 1) * padding_fraction
b = (f_support_length - 1) * (1 - padding_fraction)
dx = (b - a) / N

# mid_x are the midpoints of each subinterval
mid_x = np.linspace(a + dx/2, b - dx/2, N)

# Evaluate f_lin and h_lin at midpoints
f_lin_mid = f_lin(coordinates=(mid_x,), grid=False)
h_lin_mid = g_lin(coordinates=(mid_x / val_lambda,), grid=False)

# Midpoint Riemann sum for ∫(f_lin - h_lin)²
squared_diff_lin = (f_lin_mid - h_lin_mid) ** 2
integral_value_lin = np.sum(squared_diff_lin) * dx

# Divide by (b - a) to get MSE
mse_lin_midpoint = integral_value_lin / (b - a)
print(f"MSE with linear splines = {mse_lin_midpoint:.6e}")
