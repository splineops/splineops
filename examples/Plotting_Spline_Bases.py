"""
Plotting Spline Bases
=====================

This example demonstrates how to plot the spline bases of the library.
"""

# %%
# Imports
# -------
#
# Import the necessary libraries and utility functions to define spline bases.

import numpy as np
import matplotlib.pyplot as plt
from splineops.bases.utils import basis_map, create_basis

# %%
# Function to Plot Bases
# ----------------------
#
# Define a helper function to plot spline bases.

def plot_bases(names, x_values, title):
    plt.figure(figsize=(12, 6))
    for name in names:
        # Convert the base name to a more readable format
        name_parts = name.split('-')
        readable_name = f"{name_parts[0][:-1]} degree {name_parts[0][-1]}"
        if len(name_parts) > 1:
            readable_name += f" {name_parts[1].capitalize()}"
        
        y_values = create_basis(name).eval(x_values)
        plt.plot(x_values, y_values, label=readable_name)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.legend()
    plt.show()

# %%
# Spline Bases Evaluation and Plotting
# ------------------------------------
#
# Define the x range and plot the spline bases in figures.

x_values = np.linspace(-3, 3, 1000)

# %%
# Plot for B-Spline degree 0 (Nearest Neighbor)
# ---------------------------------------------
plot_bases(
    names=["bspline0", "bspline0-sym"],
    x_values=x_values,
    title="B-Spline Basis Functions: B-Spline degree 0 (Nearest Neighbor) and B-Spline degree 0 Sym",
)

# %%
# Plot for B-Spline degree 1 (Linear), 2 and 3
# --------------------------------------------
plot_bases(
    names=["bspline1", "bspline2", "bspline3"],
    x_values=x_values,
    title="B-Spline Basis Functions: B-Spline degree 1 (Linear), B-Spline degree 2, B-Spline degree 3",
)

# %%
# Plot for B-Spline degree 4 to 9
# -------------------------------
plot_bases(
    names=["bspline4", "bspline5", "bspline6", "bspline7", "bspline8", "bspline9"],
    x_values=x_values,
    title="B-Spline Basis Functions: B-Spline degree 4 to B-Spline degree 9",
)

# %%
# Plot for OMOMS degree 0
# -----------------------
plot_bases(
    names=["omoms0", "omoms0-sym"],
    x_values=x_values,
    title="OMOMS Basis Functions: OMOMS degree 0 and OMOMS degree 0 Sym",
)

# %%
# Plot for OMOMS degree 1, 2 and 3
# --------------------------------
plot_bases(
    names=["omoms1", "omoms2", "omoms2-sym", "omoms3"],
    x_values=x_values,
    title="OMOMS Basis Functions: OMOMS degree 1, OMOMS degree 2, OMOMS degree 2 Sym, OMOMS degree 3",
)

# %%
# Plot for OMOMS degree 4 and 5
# -----------------------------
plot_bases(
    names=["omoms4", "omoms4-sym", "omoms5"],
    x_values=x_values,
    title="OMOMS Basis Functions: OMOMS degree 4, OMOMS degree 4 Sym, OMOMS degree 5",
)

# %%
# Plot for Keys Basis Function
# ----------------------------
plot_bases(
    names=["keys"],
    x_values=x_values,
    title="Keys Basis Function",
)
