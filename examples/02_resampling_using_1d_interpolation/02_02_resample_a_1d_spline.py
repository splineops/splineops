# sphinx_gallery_start_ignore
# splineops/examples/02_resampling_using_1d_interpolation/02_02_resample_a_1d_spline.py
# sphinx_gallery_end_ignore

"""
Resample a 1D spline
====================

Resample a 1D spline with different sampling rate.

1. Assume that a user-provided 1D list of samples :math:`f[k]` has been obtained by sampling a spline on a unit grid. 

2. From the samples, recover the continuously defined spline :math:`f(x)`.

3. Resample :math:`f(x)` to get :math:`g[k] = f(Tk)`, with :math:`|T| > 1`.

4. Create a new spline :math:`g(x)` from the samples :math:`g[k]`.
"""

# %%
# Imports
# -------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from splineops.spline_interpolation.tensorspline import TensorSpline
from splineops.spline_interpolation.bases.utils import create_basis

plt.rcParams.update({
    "font.size": 14,     # Base font size
    "axes.titlesize": 18,  # Title font size
    "axes.labelsize": 16,  # Label font size
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# %%
# Initial 1D Samples
# ------------------
#
# Define a 1D discrete signal :math:`f[k]` on a unit grid, which we will treat as
# samples of an underlying spline.

number_of_samples = 27

f_support = np.arange(number_of_samples, dtype=np.float64)
f_support_length = len(f_support)  # == number_of_samples

f_samples = np.array([
    -0.657391, -0.641319, -0.613081, -0.518523, -0.453829, -0.385138,
    -0.270688, -0.179849, -0.11805, -0.0243016, 0.0130667, 0.0355389,
    0.0901577, 0.219599, 0.374669, 0.384896, 0.301386, 0.128646,
    -0.00811776, 0.0153119, 0.106126, 0.21688, 0.347629, 0.419532,
    0.50695, 0.544767, 0.555373
], dtype=np.float64)

plot_points_per_unit = 12

# Interpolated signal
base = "bspline3"
mode = "mirror"

f = TensorSpline(data=f_samples, coordinates=f_support, bases=base, modes=mode)

f_coords = np.array([q / plot_points_per_unit
                     for q in range(plot_points_per_unit * f_support_length)])
f_data = f(coordinates=(f_coords,), grid=False)

# %%
# Coarsening of f
# ---------------
#
# Sample the fine spline :math:`f(x)` on a coarser grid to obtain the new
# discrete sequence :math:`g[k] = f(Tk)`.

val_T = np.pi

# Number of g samples (e.g., 8 for 27 // pi)
g_support_length = round(f_support_length // val_T)
k = np.arange(g_support_length, dtype=np.float64)

# Physical positions where g is sampled from f: x = T * k
x_g = k * val_T
g_samples = f(coordinates=(x_g,), grid=False)

# Build g as a spline over PHYSICAL x (so markers align across plots)
g = TensorSpline(data=g_samples, coordinates=x_g, bases=base, modes=mode)

# Evaluate g across the full width of f (mirror padding extends toward the right)
g_coords_full = f_coords
g_data_full = g(coordinates=(g_coords_full,), grid=False)

# Preview: same content as the top row of the final 2-row plot
plt.figure(figsize=(12, 4))
ax = plt.gca()
ax.set_title("Interpolated f spline with coarse samples g[k]")
ax.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")
ax.plot(f_coords, f_data, color="green", linewidth=2, label="f spline")

# Vertical red lines from 0 to g[k] at x = T*k (now thick)
ax.vlines(x=x_g, ymin=0, ymax=g_samples, color="red", linewidth=2.0)

# g[k] markers at x = T*k
ax.plot(
    x_g, g_samples, "rs",
    mfc="none", markersize=12, markeredgewidth=2, label="g[k] samples"
)

ax.axhline(0, color="black", linewidth=1, zorder=0)
ax.set_xlim(0, f_support_length - 1)
ax.set_xticks(np.arange(0, f_support_length, 1))  # show 0..26 on the axis
ax.set_xlabel("x")
ax.set_ylabel("f")
ax.grid(True)
ax.legend()

# --- Annotate one interval of length T between two g[k] samples ---
if g_support_length >= 2:
    # Prefer a later interval if possible:
    #  - third interval: between g[2] and g[3] if g_support_length >= 4
    #  - else second: between g[1] and g[2] if g_support_length >= 3
    #  - else first: between g[0] and g[1]
    if g_support_length >= 4:
        start_idx = 2  # interval between k=2 and k=3
    elif g_support_length >= 3:
        start_idx = 1  # interval between k=1 and k=2
    else:
        start_idx = 0  # only one interval available

    x_T_start = x_g[start_idx]
    x_T_end = x_g[start_idx + 1]

    # Place the annotation slightly above the x-axis
    ymin, ymax = ax.get_ylim()
    y_T = ymin + 0.1 * (ymax - ymin)

    # Red double arrow between the chosen g[k] samples
    ax.annotate(
        "",
        xy=(x_T_start, y_T),
        xytext=(x_T_end, y_T),
        arrowprops=dict(arrowstyle="<->", color="red", linewidth=1.5),
    )

    # Label "T" at the midpoint of the interval, also in red
    ax.text(
        0.5 * (x_T_start + x_T_end),
        y_T,
        "T",
        ha="center",
        va="bottom",
        fontsize=14,
        color="red",
    )

    # Emphasize that these two vertical lines are the boundaries of the T interval
    # by extending them across the full vertical range.
    ax.vlines(
        [x_T_start, x_T_end],
        ymin,
        ymax,
        color="red",
        linewidth=2.0,
        zorder=2,
    )

plt.tight_layout()
plt.show()

# %%
# Coarse-Grid Basis Functions
# ---------------------------
#
# We now visualize the resized shifted basis functions on a coarser grid.
# Each coarse sample :math:`g[k]` weights a shifted basis function
# :math:`\varphi(x/T - k)` on the coarse grid. In this example,
# :math:`\varphi = \beta^{3}` is the cubic B-spline.

# Retrieve the true spline coefficients for g (on the coarse grid)
g_coeffs = g.coefficients

# Basis function corresponding to `base` (e.g. "bspline3")
basis = create_basis(base)

# Dense x-grid over the same physical domain as the final g-plot (second row)
x_dense = g_coords_full  # same as used for g_data_full

plt.figure(figsize=(12, 4))
ax = plt.gca()
ax.set_title("g[k] samples with resized shifted basis functions")

# Coarse samples g[k] at x = T*k
ax.vlines(x=x_g, ymin=0, ymax=g_samples, color="red", linewidth=2.0)
ax.plot(
    x_g, g_samples, "rs",
    mfc="none", markersize=12, markeredgewidth=2, label="g[k] samples"
)

# Overlay the coarse-grid basis functions: c_T[k] · ϕ(x/T − k)
for k_idx, c_k in enumerate(g_coeffs):
    y_basis = c_k * basis.eval(x_dense / val_T - k_idx)
    ax.plot(x_dense, y_basis, linewidth=2, alpha=0.7)

ax.axhline(0, color="black", linewidth=1, zorder=0)
ax.set_xlim(0, f_support_length - 1)
ax.set_ylabel("Amplitude")
ax.grid(True)

# Use the same coarse-grid ticks as in the final plot (second row)
max_k_tick = int(np.floor((f_support_length - 1) / val_T))
tick_ks = np.arange(max_k_tick + 1)           # coarse indices k = 0,1,...
tick_positions = tick_ks * val_T              # physical positions x = kT
ax.set_xticks(tick_positions)
ax.set_xticklabels([str(k) for k in tick_ks])
ax.set_xlabel("x")

ax.legend()
plt.tight_layout()
plt.show()

# %%
# Spline g
# --------
#
# Compare the original spline :math:`f` and the coarse spline g on the same
# physical domain, using the coarse grid on the x-axis.

fig = plt.figure(figsize=(12, 8))
gs = GridSpec(nrows=2, ncols=1, height_ratios=[1, 1])

# Top: full width with x in 0..26
ax_top = fig.add_subplot(gs[0, 0])

# Bottom: full width with x in 0..26 (own ticks so we can label at multiples of T)
ax_bottom = fig.add_subplot(gs[1, 0])

# --- TOP: f[k] + f(x) + g[k] markers at x = T*k ---
ax_top.set_title("Interpolated f spline")
ax_top.stem(f_support, f_samples, basefmt=" ", label="f[k] samples")
ax_top.plot(f_coords, f_data, color="green", linewidth=2, label="f spline")

# Red lines from 0 to g[k] at x = T*k (match thickness used above)
ax_top.vlines(x=x_g, ymin=0, ymax=g_samples, color="red", linewidth=2.0)

# g[k] markers at x = T*k
ax_top.plot(
    x_g, g_samples, "rs",
    mfc="none", markersize=12, markeredgewidth=2, label="g[k] samples"
)

ax_top.axhline(0, color="black", linewidth=1, zorder=0)
ax_top.set_xlim(0, f_support_length - 1)
ax_top.set_xticks(np.arange(0, f_support_length, 1))  # show 0..26 on the top axis
ax_top.set_xlabel("x")
ax_top.set_ylabel("f")
ax_top.grid(True)
ax_top.legend()

# --- BOTTOM: g[k] + g(x) across full width; x-axis is uniform in k at multiples of T ---
ax_bottom.set_title("Interpolated g spline")
ax_bottom.vlines(x=x_g, ymin=0, ymax=g_samples, color="red", linewidth=2.0)
ax_bottom.plot(
    x_g, g_samples, "rs",
    mfc="none", markersize=12, markeredgewidth=2, label="g[k] samples"
)
ax_bottom.plot(g_coords_full, g_data_full, color="purple", linewidth=2, label="g spline")

ax_bottom.axhline(0, color="black", linewidth=1, zorder=0)
ax_bottom.set_xlim(0, f_support_length - 1)
ax_bottom.set_ylabel("g")
ax_bottom.grid(True)
ax_bottom.legend()
ax_bottom.set_ylim(ax_top.get_ylim())  # optional: match vertical scale

# Bottom ticks at every multiple of T that fits (0..8 for T=pi with width 0..26)
max_k_tick = int(np.floor((f_support_length - 1) / val_T))
tick_ks = np.arange(max_k_tick + 1)  # e.g., 0..8
tick_positions = tick_ks * val_T
ax_bottom.set_xticks(tick_positions)
ax_bottom.set_xticklabels([str(k) for k in tick_ks])
ax_bottom.set_xlabel("x")

fig.tight_layout()
plt.show()
