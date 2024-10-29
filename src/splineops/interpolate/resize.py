# splineops/interpolate/resize.py

import numpy as np
from splineops.interpolate.tensorspline import TensorSpline

def resize(data, output=None, output_size=None, zoom_factors=None, bases="linear", modes="mirror", degree=3):
    """
    Resize an N-dimensional image using TensorSpline for interpolation.
    
    Parameters:
        data (ndarray): The input data to resize.
        output (ndarray, optional): Array to store the resized output.
        output_size (tuple, optional): Desired output shape. If provided, zoom_factors is ignored.
        zoom_factors (float or sequence, optional): Scaling factors for each axis. Ignored if output_size is provided.
        bases (str or sequence of str): Spline basis or list of bases for each dimension.
        modes (str or sequence of str): Extension modes or list of modes for each dimension.
        degree (int): Degree of the spline interpolation.

    Returns:
        ndarray: Resized data, either in `output` or a new array.
    """
    if output_size is not None:
        # Calculate zoom factors based on output size
        zoom_factors = [new / old for new, old in zip(output_size, data.shape)]
    elif zoom_factors is None:
        raise ValueError("Either output_size or zoom_factors must be provided.")
    
    # If zoom_factors is a scalar, apply it uniformly across all dimensions
    if isinstance(zoom_factors, (int, float)):
        zoom_factors = [zoom_factors] * data.ndim

    # Define a consistent dtype based on the input data
    dtype = data.dtype

    output_data = data
    for axis, zoom_factor in enumerate(zoom_factors):
        if zoom_factor != 1.0:  # Skip axis if no resizing needed
            # Adjust coordinates to map the entire original field of view
            original_coords = np.linspace(0, output_data.shape[axis] - 1, output_data.shape[axis], dtype=dtype)
            new_coords_len = output_size[axis] if output_size else int(output_data.shape[axis] * zoom_factor)
            new_coords = np.linspace(original_coords[0], original_coords[-1], new_coords_len, dtype=dtype)

            # Create a TensorSpline instance for this axis
            tensor_spline = TensorSpline(
                data=output_data,
                coordinates=[original_coords if i == axis else np.arange(output_data.shape[i], dtype=dtype)
                             for i in range(output_data.ndim)],
                bases=bases,
                modes=modes
            )

            # Interpolate along the current axis
            coords = [new_coords if i == axis else np.arange(output_data.shape[i], dtype=dtype)
                      for i in range(output_data.ndim)]
            output_data = tensor_spline.eval(coordinates=coords, grid=True)

    # Assign to output array if specified
    if output is not None:
        np.copyto(output, output_data)
        return output
    return output_data
