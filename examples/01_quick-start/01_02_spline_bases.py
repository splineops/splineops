# sphinx_gallery_start_ignore
# splineops/examples/01_quick-start/01_02_spline_bases.py
# sphinx_gallery_end_ignore

"""
Spline bases
============

Plotting the spline bases of the library.
"""

# %%
# Imports and Utilities
# ---------------------
#
# Define a helper function to visualize the spline bases.

import numpy as np
import matplotlib.pyplot as plt
from splineops.spline_interpolation.bases.utils import create_basis

x_values = np.linspace(-3, 3, 1000)

def plot_bases(names, x_values, title):
    plt.figure(figsize=(12, 6))
    for name in names:
        if name == "keys":
            readable_name = "Keys Spline"
        else:
            name_parts = name.split("-")
            readable_name = f"{name_parts[0][:-1]} degree {name_parts[0][-1]}"
        y_values = create_basis(name).eval(x_values)
        plt.plot(x_values, y_values, label=readable_name)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.show()

# %%
# Plot B-Spline Bases
# -------------------
#
# Plot B-spline basis functions for degree 0 to 9.

plot_bases(
    names=[f"bspline{i}" for i in range(10)],
    x_values=x_values,
    title="B-Spline Basis Functions: Degrees 0 to 9",
)

# %%
# Plot OMOMS Bases
# ----------------
#
# Plot OMOMS basis functions for degree 0 to 5.

plot_bases(
    names=[f"omoms{i}" for i in range(6)],
    x_values=x_values,
    title="OMOMS Basis Functions: Degrees 0 to 5",
)

# %%
# Plot Keys Basis
# ---------------
#
# Plot the Keys basis function.

plot_bases(
    names=["keys"],
    x_values=x_values,
    title="Keys Basis Function",
)

beta3_basis = create_basis("bspline3")

def beta3(x):
    """Centered cubic B-spline β(x, 3)."""
    return beta3_basis.eval(x)

# %%
# Figure 1 – β(x, 3)

plt.figure(figsize=(8, 4))
plt.plot(x_values, beta3(x_values), label=r"$\beta(x, 3)$")
plt.xlim(-3, 3)
plt.ylim(-0.2, 0.7)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("β(x, 3)")
plt.title("Cubic B-spline β(x, 3)")
plt.legend()
plt.show()


# %%
# Figure 2 – Shifted β(x - 1/3, 3)

shift = 1.0 / 3.0

plt.figure(figsize=(8, 4))
plt.plot(
    x_values,
    beta3(x_values - shift),
    label=rf"$\beta(x - {shift:.3f}, 3)$",
)
plt.xlim(-3, 3)
plt.ylim(-0.2, 0.7)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("β shifted")
plt.title("Shifted cubic B-spline β(x - 1/3, 3)")
plt.legend()
plt.show()

# %%
# Figure 3 – Horizontally scaled β((x - 1/3)/0.5, 3)

shift = 1.0 / 3.0
scale = 0.5   # same as "/ 0.5" in Mathematica

plt.figure(figsize=(8, 4))
plt.plot(
    x_values,
    beta3((x_values - shift) / scale),
    label=r"$\beta((x - 1/3)/0.5, 3)$",
)
plt.xlim(-3, 3)
plt.ylim(-0.2, 0.7)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("β scaled")
plt.title("Horizontally scaled β((x - 1/3)/0.5, 3)")
plt.legend()
plt.show()

# %%
# Figure 4 – Vertically scaled 0.25 * β((x - 1/3)/0.5, 3)

shift = 1.0 / 3.0
scale = 0.5
amp = 0.25

plt.figure(figsize=(8, 4))
plt.plot(
    x_values,
    amp * beta3((x_values - shift) / scale),
    label=r"$0.25 \,\beta((x - 1/3)/0.5, 3)$",
)
plt.xlim(-3, 3)
plt.ylim(-0.2, 0.7)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("Scaled β")
plt.title("Vertically scaled 0.25 · β((x - 1/3)/0.5, 3)")
plt.legend()
plt.show()

# %%
# Figure 5 – Combination of four weighted cubic B-splines

def beta3_shift_scale(x, shift, scale, amp=1.0):
    return amp * beta3((x + shift) / scale)

# Note: Mathematica uses (x - 1/3)/0.5 etc.
# I keep the same algebra but write shifts explicitly:
curves = [
    ("0.25 β((x - 1/3)/0.5, 3)",
     lambda x: 0.25 * beta3((x - 1.0/3.0) / 0.5)),
    ("0.8 β((x + 5/3)/1.2, 3)",
     lambda x: 0.8  * beta3((x + 5.0/3.0) / 1.2)),
    ("-0.25 β((x + 4/5)/0.2, 3)",
     lambda x: -0.25 * beta3((x + 4.0/5.0) / 0.2)),
    ("-0.2 β((x - 1)/1, 3)",
     lambda x: -0.2  * beta3((x - 1.0) / 1.0)),
]

plt.figure(figsize=(10, 5))
for label, f in curves:
    plt.plot(x_values, f(x_values), label=label)

plt.xlim(-3, 3)
plt.ylim(-0.2, 0.7)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("Weighted β")
plt.title("Four weighted & shifted cubic B-splines")
plt.legend()
plt.show()

# %%
# Figure 6 – Sum of the four weighted cubic B-splines

def combined_spline(x):
    return sum(f(x) for _, f in curves)

plt.figure(figsize=(8, 4))
plt.plot(x_values, combined_spline(x_values), label="Sum of four β’s")
plt.xlim(-3, 3)
plt.ylim(-0.2, 0.7)
plt.grid(True)
plt.xlabel("x")
plt.ylabel("Sum")
plt.title("Sum of weighted cubic B-splines (Mathematica Total[…])")
plt.legend()
plt.show()

