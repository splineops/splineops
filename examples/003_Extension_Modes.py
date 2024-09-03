"""
Displaying Extension Modes
==========================

This example shows current extension modes Finite Support Coefficients and Narrow Mirroring.
"""

# %%
# Imports
# -------
#
# Import the necessary libraries and utility functions.

import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.tensorspline import TensorSpline
from splineops.bases.utils import create_basis

# %%
# Function to Sample Spline Basis
# -------------------------------
#
# Sample the spline basis function at specified x values.

def sample_spline_basis(basis_name, x_values):
    basis = create_basis(basis_name)
    return basis.eval(x_values)

# %%
# Function to Plot Extension Modes for Multiple Degrees
# -----------------------------------------------------
#
# Define a helper function to plot extension modes for B-splines of degrees 0 to 3 using TensorSpline.

def plot_extension_modes_for_degrees(degrees, mode_name, x_values, title):
    plt.figure(figsize=(12, 6))
    
    for degree in degrees:
        basis_name = f"bspline{degree}"
        
        # Sample the spline basis
        data = sample_spline_basis(basis_name, x_values)
        
        # Ensure that the data is scaled correctly for interpolation (degree 1 should interpolate exactly)
        if degree == 1:
            data = np.interp(x_values, x_values, data)
        
        # Create TensorSpline instance
        tensor_spline = TensorSpline(data=data, coordinates=(x_values,), bases=basis_name, modes=mode_name)
        
        # Define the extended evaluation grid (from -10 to 10)
        eval_x_values = np.linspace(-10, 10, 2000)
        eval_coords = (eval_x_values,)
        
        # Evaluate the tensor spline
        extended_data = tensor_spline.eval(coordinates=eval_coords)
        
        # Plot the results
        plt.plot(eval_x_values, extended_data, label=f"B-Spline Degree {degree}")
    
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('Interpolated Value')
    plt.grid(True)
    plt.legend()
    plt.show()

# %%
# Define x Range
# --------------
x_values = np.linspace(-3, 3, 101)  # Use 101 points to ensure 0 (middle) is included

# Define degrees to plot
degrees = [0, 1, 2, 3]

# %%
# Plot for Finite Support Coefficients
# ------------------------------------
plot_extension_modes_for_degrees(
    degrees=degrees,
    mode_name="zero",
    x_values=x_values,
    title="Extension Mode: Finite Support Coefficients for B-Spline Degrees 0 to 3",
)

# %%
# Plot for Narrow Mirroring
# -------------------------
plot_extension_modes_for_degrees(
    degrees=degrees,
    mode_name="mirror",
    x_values=x_values,
    title="Extension Mode: Narrow Mirroring for B-Spline Degrees 0 to 3",
)
