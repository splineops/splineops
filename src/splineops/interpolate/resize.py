import numpy as np
from splineops.interpolate.tensorspline import TensorSpline
from splineops.bases.utils import asbasis
from splineops.interpolate.ls_oblique.ls_oblique_resize import ls_oblique_resize

def resize(data, zoom_factors=None, output=None, output_size=None, degree=3, modes="mirror", method="interpolation"):
    """
    Resize an N-dimensional image using TensorSpline for interpolation or LS/oblique projection methods.

    Parameters:
        data (ndarray): The input data to resize.
        zoom_factors (float or sequence, optional): Scaling factors for each axis. Ignored if output_size is provided.
        output (ndarray or dtype, optional): Array in which to place the output, or the dtype of the returned array.
        output_size (tuple, optional): Desired output shape. If provided, zoom_factors is ignored.
        degree (int): Degree of the B-spline interpolation (0 to 9).
        modes (str or sequence of str): Extension modes or list of modes for each dimension.
        method (str): Interpolation method, "interpolation" (default), "least-squares", or "oblique".

    Returns:
        ndarray: Resized data in `output` if specified, otherwise a new array.
    """
    if not (0 <= degree <= 9):
        raise ValueError("degree must be an integer between 0 and 9 for B-spline interpolation.")

    if output_size is not None:
        zoom_factors = [new / old for new, old in zip(output_size, data.shape)]
    elif zoom_factors is None:
        raise ValueError("Either output_size or zoom_factors must be provided.")
    
    if isinstance(zoom_factors, (int, float)):
        zoom_factors = [zoom_factors] * data.ndim

    dtype = data.dtype if output is None else output.dtype

    # Call LS/oblique resize if conditions are met, else use TensorSpline
    if method in {"least-squares", "oblique"} and degree in {1, 2, 3}:
        print(f"Using {method} method with mirror boundary conditions.")
        output_data = ls_oblique_resize(
            input_img_normalized=data,
            output_size=output_size,
            zoom_factors=zoom_factors,
            method=method,
            interpolation={1: "linear", 2: "quadratic", 3: "cubic"}[degree]
        )
    else:
        # Use TensorSpline for standard interpolation
        if method in {"least-squares", "oblique"}:
            print("Standard interpolation is used because the degree is not 1, 2, or 3.")
        basis_str = f"bspline{degree}"
        basis = asbasis(basis_str)
        original_coords = [np.linspace(0, dim - 1, dim, dtype=dtype) for dim in data.shape]
        new_coords = [
            np.linspace(0, dim - 1, int(dim * zoom), dtype=dtype)
            for dim, zoom in zip(data.shape, zoom_factors)
        ]
        tensor_spline = TensorSpline(data=data, coordinates=original_coords, bases=basis, modes=modes)
        output_data = tensor_spline.eval(coordinates=new_coords, grid=True)

    # Assign to output array if specified
    if output is not None:
        if isinstance(output, np.ndarray):
            np.copyto(output, output_data)
            return output
        else:
            output = np.empty(output_data.shape, dtype=output)
            np.copyto(output, output_data)
            return output

    return output_data
