# splineops/src/splineops/affine/affine.py

import numbers
import operator
from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from splineops.spline_interpolation.tensor_spline import TensorSpline

_AFFINE_TILE_SIZE = 65_536


def rotate(
    data: npt.NDArray,
    angle: float,
    axis: Optional[Tuple[float, float, float]] = None,
    center: Optional[Tuple[float, float, float]] = None,
    degree: int = 3,
    mode: str = "zero",
) -> npt.NDArray:
    """
    Rotate 2D or 3D data around a specified center using spline interpolation.

    Parameters
    ----------
    data : ndarray
        2D or 3D input data array to rotate.
    angle : float
        Rotation angle in degrees.
    axis : tuple of float, optional
        The axis of rotation for 3D data. Defaults to (0, 0, 1).
    center : tuple of float, optional
        The center of rotation. Defaults to the array center.
    degree : int, optional
        B-spline degree (0 to 7). Default is 3.
    mode : str, optional
        Boundary handling mode (e.g., "zero", "mirror"). Default is "zero".

    Returns
    -------
    rotated_data : ndarray
        The data array after rotation.

    Examples
    --------
    Rotate a 2D array by 45 degrees:

    >>> import numpy as np
    >>> from splineops.affine import rotate
    >>> data = np.array([[1, 2], [3, 4]])
    >>> rotated_data = rotate(data, angle=45)
    >>> rotated_data.shape
    (2, 2)

    Rotate a 3D array around a custom axis:

    >>> data_3d = np.random.rand(4, 4, 4)
    >>> rotated_data_3d = rotate(data_3d, angle=30, axis=(1, 0, 0))
    >>> rotated_data_3d.shape
    (4, 4, 4)
    """
    if not isinstance(data, np.ndarray):
        raise TypeError("'data' must be a NumPy array.")
    ndim = data.ndim
    if ndim not in (2, 3):
        raise ValueError("rotate: only 2D or 3D data are supported.")
    if any(length == 0 for length in data.shape):
        raise ValueError("'data' dimensions must be non-empty.")
    if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
        raise TypeError("'data' must have a real numeric dtype.")
    if not isinstance(angle, numbers.Real) or isinstance(angle, (bool, np.bool_)):
        raise TypeError("'angle' must be a real number.")
    if not np.isfinite(angle):
        raise ValueError("'angle' must be finite.")

    try:
        degree = operator.index(degree)
    except TypeError as exc:
        raise TypeError("'degree' must be an integer from 0 through 7.") from exc
    if isinstance(degree, (bool, np.bool_)) or not 0 <= degree <= 7:
        raise ValueError("'degree' must be an integer from 0 through 7.")
    basis = f"bspline{degree}"

    # Integer samples describe values, not an integer interpolation space.
    # Promote them predictably; preserve supported floating precision.
    work_dtype = (
        data.dtype if np.issubdtype(data.dtype, np.floating) else np.dtype(np.float64)
    )
    data = data.astype(work_dtype, copy=False)

    # Setup tensor spline on N-dimensional data
    coordinates = [np.linspace(0, dim - 1, dim, dtype=data.dtype) for dim in data.shape]
    tensor_spline = TensorSpline(
        data=data,
        coordinates=coordinates,
        bases=basis,
        modes=mode,
    )

    # Use specified center or default to array center
    if center is None:
        center_coords = [(dim - 1) / 2.0 for dim in data.shape]
    else:
        if len(center) != ndim:
            raise ValueError("center must have same length as data.ndim")
        center_coords = [float(c) for c in center]
        if not all(np.isfinite(c) for c in center_coords):
            raise ValueError("'center' entries must be finite.")

    # Convert angle to radians
    angle_rad = np.radians(angle)

    if ndim == 2:
        # 2D rotation matrix (pull-back, so angle is negated)
        cos_angle = np.cos(-angle_rad)
        sin_angle = np.sin(-angle_rad)
        R = np.array(
            [[cos_angle, -sin_angle], [sin_angle, cos_angle]],
            dtype=data.dtype,
        )
    else:  # ndim == 3
        # Default axis of rotation (z-axis) if not provided
        if axis is None:
            axis = (0.0, 0.0, 1.0)
        if len(axis) != 3:
            raise ValueError("'axis' must contain three values for 3D rotation.")
        axis_vec = np.array(axis, dtype=data.dtype)
        if not np.all(np.isfinite(axis_vec)):
            raise ValueError("'axis' entries must be finite.")
        norm = np.linalg.norm(axis_vec)
        if norm == 0:
            raise ValueError("axis must be non-zero for 3D rotation")
        axis_vec /= norm  # normalize

        ux, uy, uz = axis_vec
        cos_angle = np.cos(-angle_rad)
        sin_angle = np.sin(-angle_rad)
        one_minus_cos = 1.0 - cos_angle

        R = np.array(
            [
                [
                    cos_angle + ux**2 * one_minus_cos,
                    ux * uy * one_minus_cos - uz * sin_angle,
                    ux * uz * one_minus_cos + uy * sin_angle,
                ],
                [
                    uy * ux * one_minus_cos + uz * sin_angle,
                    cos_angle + uy**2 * one_minus_cos,
                    uy * uz * one_minus_cos - ux * sin_angle,
                ],
                [
                    uz * ux * one_minus_cos - uy * sin_angle,
                    uz * uy * one_minus_cos + ux * sin_angle,
                    cos_angle + uz**2 * one_minus_cos,
                ],
            ],
            dtype=data.dtype,
        )

    # Generate, transform, and evaluate output coordinates in bounded tiles.
    # This avoids allocating ``ndim`` full-size meshgrid arrays and another two
    # full coordinate stacks for large images or volumes.
    center_array = np.asarray(center_coords, dtype=data.dtype)[:, None]
    interpolated_values = np.empty(data.size, dtype=data.dtype)
    for start in range(0, data.size, _AFFINE_TILE_SIZE):
        stop = min(start + _AFFINE_TILE_SIZE, data.size)
        flat_indices = np.arange(start, stop)
        coords = np.asarray(
            np.unravel_index(flat_indices, data.shape), dtype=data.dtype
        )
        rotated_coords = R @ (coords - center_array) + center_array
        interpolated_values[start:stop] = tensor_spline(
            coordinates=tuple(rotated_coords),
            grid=False,
        )

    rotated_data = interpolated_values.reshape(data.shape)

    return rotated_data
