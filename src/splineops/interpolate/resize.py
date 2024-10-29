# splineops/interpolate/resize.py

import numpy as np
from splineops.interpolate.tensorspline import TensorSpline
from splineops.bases.utils import asbasis

def resize(data, zoom_factors=None, output=None, output_size=None, degree=3, modes="mirror"):
    """
    Resize an N-dimensional image using TensorSpline for interpolation.

    Parameters:
        data (ndarray): The input data to resize.
        zoom_factors (float or sequence, optional): Scaling factors for each axis. Ignored if output_size is provided.
        output (ndarray or dtype, optional): Array in which to place the output, or the dtype of the returned array.
        output_size (tuple, optional): Desired output shape. If provided, zoom_factors is ignored.
        degree (int): Degree of the B-spline interpolation (0 to 9).
        modes (str or sequence of str): Extension modes or list of modes for each dimension.

    Returns:
        ndarray: Resized data in `output` if specified, otherwise a new array.
    """
    if not (0 <= degree <= 9):
        raise ValueError("degree must be an integer between 0 and 9 for B-spline interpolation.")

    if output_size is not None:
        # Calculate zoom factors based on output size
        zoom_factors = [new / old for new, old in zip(output_size, data.shape)]
    elif zoom_factors is None:
        raise ValueError("Either output_size or zoom_factors must be provided.")
    
    # If zoom_factors is a scalar, apply it uniformly across all dimensions
    if isinstance(zoom_factors, (int, float)):
        zoom_factors = [zoom_factors] * data.ndim

    # Define a consistent dtype based on the input data
    dtype = data.dtype if output is None else output.dtype

    # Choose B-spline basis string based on degree
    basis_str = f"bspline{degree}"
    basis = asbasis(basis_str)

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
        bases=basis,
        modes=modes
    )

    # Evaluate the TensorSpline at the new coordinates grid
    output_data = tensor_spline.eval(coordinates=new_coords, grid=True)

    # Assign to output array if specified
    if output is not None:
        if isinstance(output, np.ndarray):
            np.copyto(output, output_data)
            return output
        else:
            # Create an array with the specified dtype
            output = np.empty(output_data.shape, dtype=output)
            np.copyto(output, output_data)
            return output
    
    return output_data
