"""
Displaying Extension Modes
==========================

This example shows current extension modes Finite Support Coefficients, Narrow Mirroring, and Periodic Padding.
"""

# %%
# Imports
# -------
#
# Import the necessary libraries and utility functions.

import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.tensorspline import TensorSpline

# %%
# Function to Create Linear Signal
# --------------------------------
#
# Create a simple linear signal for the specified x values.

def create_linear_signal(x_values):
    return x_values  # Linear function: f(x) = x

# %%
# Function to Plot Extension Modes for Linear Function
# ----------------------------------------------------
#
# Define a helper function to plot extension modes using a linear function and adding boundary markers.

def plot_extension_modes_for_linear_function(mode_name, x_values, title):
    plt.figure(figsize=(12, 6))

    # Create the linear signal
    data = create_linear_signal(x_values)
    
    # Create TensorSpline instance
    tensor_spline = TensorSpline(data=data, coordinates=(x_values,), bases="linear", modes=mode_name)
    
    # Define the extended evaluation grid (from -10 to 10)
    eval_x_values = np.linspace(-10, 10, 2000)
    eval_coords = (eval_x_values,)
    
    # Evaluate the tensor spline
    extended_data = tensor_spline.eval(coordinates=eval_coords)
    
    # Plot the results
    plt.plot(eval_x_values, extended_data, label="Linear Function f(x) = x")
    
    # Add vertical lines at the boundaries of the original signal
    plt.axvline(x=x_values[0], color='red', linestyle='--', label='Original Signal Start')
    plt.axvline(x=x_values[-1], color='blue', linestyle='--', label='Original Signal End')

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

# %%
# Plot for Finite Support Coefficients
# ------------------------------------
plot_extension_modes_for_linear_function(
    mode_name="zero",  # Finite Support Coefficients is represented by "zero"
    x_values=x_values,
    title="Extension Mode: Finite Support Coefficients for Linear Function",
)

# %%
# Plot for Narrow Mirroring
# -------------------------
plot_extension_modes_for_linear_function(
    mode_name="mirror",  # Narrow Mirroring is represented by "mirror"
    x_values=x_values,
    title="Extension Mode: Narrow Mirroring for Linear Function",
)

# %%
# Plot for Periodic Extension Mode
# --------------------------------
plot_extension_modes_for_linear_function(
    mode_name="periodic",  # New Periodic Mode
    x_values=x_values,
    title="Extension Mode: Periodic Padding for Linear Function",
)
