#!/usr/bin/env python3
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

from bssp.interpolate.tensorspline import TensorSpline

# Data type (need to provide floating numbers, "float64" and "float32" are typical)
dtype = "float32"

# Create random data samples and corresponding coordinates
nx, ny = 2, 5
xmin, xmax = -3.1, +1
ymin, ymax = 2, 6.5
xx = np.linspace(xmin, xmax, nx, dtype=dtype)
yy = np.linspace(ymin, ymax, ny, dtype=dtype)
coordinates = xx, yy
prng = np.random.default_rng(seed=5250)
data = prng.standard_normal(size=tuple(c.size for c in coordinates))
data = np.ascontiguousarray(data, dtype=dtype)

# Tensor spline bases and modes
bases = "bspline3"  # same basis applied to all dimensions
modes = "mirror"  # same mode applied to all dimensions

# Create tensor spline from NumPy data
data_np = data
coordinates_np = coordinates
tensor_spline_np = TensorSpline(
    data=data_np, coordinates=coordinates_np, bases=bases, modes=modes
)

# Create tensor spline from CuPy data for GPU computations
#   Note: we first need to convert the NumPy data to CuPy
data_cp = cp.asarray(data)
coordinates_cp = cp.asarray(xx), cp.asarray(yy)
tensor_spline_cp = TensorSpline(
    data=data_cp, coordinates=coordinates_cp, bases=bases, modes=modes
)

# Create evaluation coordinates (extended and oversampled in this case)
dx = (xx[-1] - xx[0]) / (nx - 1)
dy = (yy[-1] - yy[0]) / (ny - 1)
pad_fct = 1.1
px = pad_fct * nx * dx
py = pad_fct * ny * dy
eval_xx = np.linspace(xx[0] - px, xx[-1] + px, 100 * nx)
eval_yy = np.linspace(yy[0] - py, yy[-1] + py, 100 * ny)

# Evaluate using NumPy
eval_coords_np = eval_xx, eval_yy
data_eval_np = tensor_spline_np(coordinates=eval_coords_np)

# Evaluate using CuPy
#  Note: we first need to convert the evalution coordinates to CuPy
eval_coords_cp = cp.asarray(eval_xx), cp.asarray(eval_yy)
data_eval_cp = tensor_spline_cp(coordinates=eval_coords_cp)

# Compute difference
abs_diff = np.abs(data_eval_cp.get() - data_eval_np)
print(f"Maximum absolute difference: {np.max(abs_diff)}")

# Figures
ax: plt.Axes
fig: plt.Figure
fig, axes = plt.subplots(
    nrows=1, ncols=3, sharex="all", sharey="all", layout="constrained"
)
ax = axes[0]
ax.imshow(data_eval_np.T)
ax.set_title("NumPy")
ax = axes[1]
ax.set_title("CuPy")
ax.imshow(data_eval_cp.get().T)
ax = axes[2]
ax.set_title("Absolute difference")
ax.imshow(abs_diff.T)

plt.show()
