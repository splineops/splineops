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
        if name == "keys":
            # Special case for Keys spline: no degree in the label
            readable_name = "Keys Spline"
        else:
            name_parts = name.split('-')
            readable_name = f"{name_parts[0][:-1]} degree {name_parts[0][-1]}"
        
        # Evaluate the basis function
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
# Combined Plot for B-Spline degrees 0 to 9 (excluding Sym bases)
# ---------------------------------------------------------------
plot_bases(
    names=["bspline0", "bspline1", "bspline2", "bspline3", "bspline4", "bspline5", "bspline6", "bspline7", "bspline8", "bspline9"],
    x_values=x_values,
    title="B-Spline Basis Functions: B-Spline degree 0 to B-Spline degree 9",
)

# %%
# Combined Plot for OMOMS degrees 0 to 5 (excluding Sym bases)
# ------------------------------------------------------------
plot_bases(
    names=["omoms0", "omoms1", "omoms2", "omoms3", "omoms4", "omoms5"],
    x_values=x_values,
    title="OMOMS Basis Functions: OMOMS degree 0 to OMOMS degree 5",
)

# %%
# Plot for Keys Basis Function (unchanged)
# ----------------------------------------
plot_bases(
    names=["keys"],
    x_values=x_values,
    title="Keys Basis Function",
)