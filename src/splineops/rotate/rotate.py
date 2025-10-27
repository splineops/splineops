# splineops/src/splineops/rotate/rotate.py

import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple
from splineops.spline_interpolation.tensorspline import TensorSpline

def rotate(
    data: npt.NDArray,
    angle: float,
    axis: Optional[Tuple[float, float, float]] = None,
    center: Optional[Tuple[float, float, float]] = None,
    degree: int = 3,
    mode: str = "zero"
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
    >>> from splineops.interpolate.rotate import rotate
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
    ndim = data.ndim
    if ndim not in (2, 3):
        raise ValueError("This function supports only 2D or 3D data.")

    # Ensure the degree is within valid bounds
    degree = max(0, min(degree, 7))
    basis = f"bspline{degree}"

    # Setup tensor spline on N-dimensional data
    coordinates = [np.linspace(0, dim - 1, dim, dtype=data.dtype) for dim in data.shape]
    tensor_spline = TensorSpline(data=data, coordinates=coordinates, bases=basis, modes=mode)

    # Create meshgrid for all dimensions to get coordinates with the correct shape
    grid = np.meshgrid(*[np.arange(dim) for dim in data.shape], indexing="ij")

    # Use specified center or default to array center
    if center is None:
        center_coords = [(dim - 1) / 2.0 for dim in data.shape]
    else:
        center_coords = center

    # Center the coordinates
    coords = np.stack([g - c for g, c in zip(grid, center_coords)], axis=0)
    coords_flat = coords.reshape(ndim, -1)

    # Convert angle to radians
    angle_rad = np.radians(angle)

    if ndim == 2:
        # 2D rotation matrix
        cos_angle = np.cos(-angle_rad)
        sin_angle = np.sin(-angle_rad)
        R = np.array([
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle]
        ])

        # Apply rotation
        rotated_coords_flat = R @ coords_flat

    elif ndim == 3:
        # Default axis of rotation (z-axis) if not provided
        if axis is None:
            axis = (0, 0, 1)
        axis = np.array(axis, dtype=data.dtype)
        axis /= np.linalg.norm(axis)  # Normalize to make it a unit vector

        # 3D rotation matrix using the axis-angle formula
        cos_angle = np.cos(-angle_rad)
        sin_angle = np.sin(-angle_rad)
        one_minus_cos = 1 - cos_angle
        ux, uy, uz = axis

        R = np.array([
            [cos_angle + ux**2 * one_minus_cos,
             ux * uy * one_minus_cos - uz * sin_angle,
             ux * uz * one_minus_cos + uy * sin_angle],
            [uy * ux * one_minus_cos + uz * sin_angle,
             cos_angle + uy**2 * one_minus_cos,
             uy * uz * one_minus_cos - ux * sin_angle],
            [uz * ux * one_minus_cos - uy * sin_angle,
             uz * uy * one_minus_cos + ux * sin_angle,
             cos_angle + uz**2 * one_minus_cos]
        ], dtype=data.dtype)

        # Apply rotation
        rotated_coords_flat = R @ coords_flat

    # Translate coordinates back to the original space
    rotated_coords_flat += np.array(center_coords)[:, None]

    # Interpolate using TensorSpline
    interpolated_values = tensor_spline(coordinates=tuple(rotated_coords_flat), grid=False)

    # Reshape back to the original data shape
    rotated_data = interpolated_values.reshape(data.shape)

    return rotated_data
