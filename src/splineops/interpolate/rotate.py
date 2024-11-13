import numpy as np
from splineops.interpolate.tensorspline import TensorSpline

def rotate(data, angle, degree=3, mode="zero", axes=(0, 1)):
    """
    Rotate N-dimensional data around a specified plane using TensorSpline interpolation.

    Parameters:
        data (ndarray): The input N-dimensional data array.
        angle (float): The rotation angle in degrees.
        degree (int): The degree of the spline (0 to 7). Default is 3.
        mode (str): The mode for handling boundaries. Default is "zero".
        axes (tuple of int): The two axes that define the plane of rotation. Default is (0, 1) for 2D.

    Returns:
        ndarray: The rotated data as an N-dimensional numpy array.
    """
    if len(axes) != 2:
        raise ValueError("The 'axes' parameter must be a tuple of two axis indices for the rotation plane.")

    # Ensure the degree is within valid bounds
    degree = max(0, min(degree, 7))
    basis = f"bspline{degree}"

    # Setup tensor spline on N-dimensional data
    coordinates = [np.linspace(0, dim - 1, dim, dtype=data.dtype) for dim in data.shape]
    tensor_spline = TensorSpline(data=data, coordinates=coordinates, bases=basis, modes=mode)

    # Convert angle to radians and calculate rotation matrix
    angle_rad = np.radians(angle)
    cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)

    # Create meshgrid for all dimensions to get coordinates with the correct shape
    grid = np.meshgrid(*[np.arange(dim) for dim in data.shape], indexing="ij")
    
    # Center the coordinates on the rotation plane
    center_coords = [(dim - 1) / 2.0 for dim in data.shape]
    coords = [grid[i] - center_coords[i] if i in axes else grid[i] for i in range(data.ndim)]
    
    # Apply the rotation matrix to the specified axes
    i, j = axes
    rotated_coords = [None] * data.ndim
    rotated_coords[i] = cos_angle * coords[i] + sin_angle * coords[j] + center_coords[i]
    rotated_coords[j] = -sin_angle * coords[i] + cos_angle * coords[j] + center_coords[j]
    
    # Non-rotated axes remain unchanged
    for k in range(data.ndim):
        if k not in axes:
            rotated_coords[k] = grid[k]

    # Flatten coordinates and evaluate using TensorSpline
    eval_coords = tuple(coord.flatten() for coord in rotated_coords)
    interpolated_values = tensor_spline(coordinates=eval_coords, grid=False)
    
    # Reshape back to the original data shape
    rotated_data = interpolated_values.reshape(data.shape)

    return rotated_data
