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

# %%
# Plotting
# --------

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

# NEW: thin red lines from 0 to g[k] at x = T*k
ax_top.vlines(x=x_g, ymin=0, ymax=g_samples, color='red', linewidth=1)

# g[k] markers at x = T*k
ax_top.plot(
    x_g, g_samples, "rs",
    mfc='none', markersize=12, markeredgewidth=2, label="g[k] samples"
)

ax_top.axhline(0, color='black', linewidth=1, zorder=0)
ax_top.set_xlim(0, f_support_length - 1)
ax_top.set_xticks(np.arange(0, f_support_length, 1))  # show 0..26 on the top axis
ax_top.set_xlabel("x")
ax_top.set_ylabel("f")
ax_top.grid(True)
ax_top.legend()

# --- BOTTOM: g[k] + g(x) across full width; x-axis is uniform in k at multiples of T ---
ax_bottom.set_title("Interpolated g spline")
ax_bottom.vlines(x=x_g, ymin=0, ymax=g_samples, color='red', linewidth=1)
ax_bottom.plot(
    x_g, g_samples, "rs",
    mfc='none', markersize=12, markeredgewidth=2, label="g[k] samples"
)
ax_bottom.plot(g_coords_full, g_data_full, color="purple", linewidth=2, label="g spline")

ax_bottom.axhline(0, color='black', linewidth=1, zorder=0)
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
