import numpy as np
import numpy.typing as npt
from typing import Optional, Union, Sequence, Tuple
from splineops.interpolate.tensorspline import TensorSpline
from splineops.bases.utils import asbasis
from splineops.resize.ls_oblique_resize import ls_oblique_resize


def resize(
    data: npt.NDArray,
    zoom_factors: Optional[Union[float, Sequence[float]]] = None,
    output: Optional[Union[npt.NDArray, np.dtype]] = None,
    output_size: Optional[Tuple[int, ...]] = None,
    degree: int = 3,
    modes: Union[str, Sequence[str]] = "mirror",
    method: str = "interpolation"
) -> npt.NDArray:
    """
    Resize an N-dimensional image using TensorSpline for interpolation or LS/oblique projection methods.

    Parameters
    ----------
    data : ndarray
        The input data to resize.
    zoom_factors : float or sequence of float, optional
        Scaling factors for each axis. Ignored if `output_size` is provided.
    output : ndarray or numpy.dtype, optional
        If an ndarray, the result is copied into it. If a dtype, a new array
        of that dtype is returned. Default is None.
    output_size : tuple of int, optional
        Desired output shape. If provided, `zoom_factors` is ignored.
    degree : int, optional
        Degree of the B-spline interpolation (0 to 9). Default is 3.
    modes : str or sequence of str, optional
        Extension mode(s) for each dimension. Default is "mirror".
    method : {'interpolation', 'least-squares', 'oblique'}, optional
        Resizing method. Default is "interpolation".

    Returns
    -------
    resized_data : ndarray
        Resized data. If `output` is an ndarray, the function writes the
        result in-place and returns `output`.

    Examples
    --------
    Resize a 2D array using interpolation:
    
    >>> import numpy as np
    >>> from splineops.interpolate.resize import resize
    >>> data = np.array([[1, 2], [3, 4]])
    >>> resized_data = resize(data, zoom_factors=2)
    >>> resized_data.shape
    (4, 4)

    Resize a 3D array using LS projection:
    
    >>> data_3d = np.random.rand(4, 4, 4)
    >>> resized_data_3d = resize(data_3d, output_size=(8, 8, 8), degree=3, method="least-squares")
    >>> resized_data_3d.shape
    (8, 8, 8)
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
        # Use LS/oblique resize
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
            np.linspace(0, dim - 1, round(dim * zoom), dtype=dtype)
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
