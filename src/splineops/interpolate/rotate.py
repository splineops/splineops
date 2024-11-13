import numpy as np
from splineops.interpolate.tensorspline import TensorSpline

def rotate(data, angle, degree=3, mode="zero"):
    """
    Rotate a 2D data by a specified angle using TensorSpline interpolation.

    Parameters:
        data (ndarray): The input 2D data array.
        angle (float): The rotation angle in degrees.
        degree (int): The degree of the spline (0 to 7). Default is 3.
        mode (str): The mode for handling boundaries. Default is "zero".

    Returns:
        ndarray: The rotated data as a 2D numpy array.
    """
    dtype = data.dtype
    ny, nx = data.shape
    xx = np.linspace(0, nx - 1, nx, dtype=dtype)
    yy = np.linspace(0, ny - 1, ny, dtype=dtype)
    data = np.ascontiguousarray(data, dtype=dtype)

    degree = max(0, min(degree, 7))
    basis = f"bspline{degree}"

    tensor_spline = TensorSpline(
        data=data, coordinates=(yy, xx), bases=basis, modes=mode
    )
    angle_rad = np.radians(-angle)
    cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
    original_center_x, original_center_y = (nx - 1) / 2.0, (ny - 1) / 2.0
    oy, ox = np.ogrid[0:ny, 0:nx]
    ox = ox - original_center_x
    oy = oy - original_center_y

    nx_coords = cos_angle * ox + sin_angle * oy + original_center_x
    ny_coords = -sin_angle * ox + cos_angle * oy + original_center_y

    eval_coords = ny_coords.flatten(), nx_coords.flatten()
    interpolated_values = tensor_spline(coordinates=eval_coords, grid=False)
    rotated_image = interpolated_values.reshape(ny, nx)

    return rotated_image