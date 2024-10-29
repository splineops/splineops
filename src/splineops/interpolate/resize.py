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

    # Original coordinates for each dimension
    original_coords = [np.linspace(0, dim - 1, dim, dtype=dtype) for dim in data.shape]

    # New coordinates based on zoom_factors or output_size
    new_coords = [
        np.linspace(0, dim - 1, int(dim * zoom), dtype=dtype)
        for dim, zoom in zip(data.shape, zoom_factors)
    ]

    # Create a single TensorSpline instance
    tensor_spline = TensorSpline(
        data=data,
        coordinates=original_coords,
        bases=bases,
        modes=modes
    )

    # Evaluate the TensorSpline at the new coordinates grid
    output_data = tensor_spline.eval(coordinates=new_coords, grid=True)

    # Assign to output array if specified
    if output is not None:
        np.copyto(output, output_data)
        return output
    return output_data
