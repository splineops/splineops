Rotate Module
=============

Overview
--------

The `rotate` function in the `splineops` library allows you to rotate 2D or 3D data arrays around a specified axis and center using spline interpolation. This function is particularly useful for geometric transformations in image processing, computer graphics, and scientific computing.

Mathematical Background
-----------------------

**2D Rotation**

In 2D space, rotating a point :math:`(x, y)` around a center point :math:`(x_c, y_c)` by an angle :math:`\theta` (in radians) is achieved using the rotation matrix:

.. math::

   \begin{pmatrix}
   x' \\
   y'
   \end{pmatrix}
   =
   \begin{pmatrix}
   \cos\theta & -\sin\theta \\
   \sin\theta & \cos\theta \\
   \end{pmatrix}
   \begin{pmatrix}
   x - x_c \\
   y - y_c \\
   \end{pmatrix}
   +
   \begin{pmatrix}
   x_c \\
   y_c \\
   \end{pmatrix}

Here, :math:`(x', y')` are the coordinates of the rotated point.

**3D Rotation**

In 3D space, rotating a point :math:`\mathbf{v} = (x, y, z)` around an arbitrary axis defined by a unit vector :math:`\mathbf{u} = (u_x, u_y, u_z)` by an angle :math:`\theta` can be performed using Rodrigues' rotation formula:

.. math::

   \mathbf{v}' = \mathbf{v} \cos\theta + (\mathbf{u} \times \mathbf{v}) \sin\theta + \mathbf{u} \left( \mathbf{u} \cdot \mathbf{v} \right) (1 - \cos\theta)

Alternatively, the rotation can be represented using the rotation matrix :math:`\mathbf{R}`:

.. math::

   \mathbf{R} =
   \begin{pmatrix}
   \cos\theta + u_x^2 (1 - \cos\theta) & u_x u_y (1 - \cos\theta) - u_z \sin\theta & u_x u_z (1 - \cos\theta) + u_y \sin\theta \\
   u_y u_x (1 - \cos\theta) + u_z \sin\theta & \cos\theta + u_y^2 (1 - \cos\theta) & u_y u_z (1 - \cos\theta) - u_x \sin\theta \\
   u_z u_x (1 - \cos\theta) - u_y \sin\theta & u_z u_y (1 - \cos\theta) + u_x \sin\theta & \cos\theta + u_z^2 (1 - \cos\theta)
   \end{pmatrix}

Then the rotated point is calculated as:

.. math::

   \begin{pmatrix}
   x' \\
   y' \\
   z'
   \end{pmatrix}
   =
   \mathbf{R}
   \begin{pmatrix}
   x - x_c \\
   y - y_c \\
   z - z_c
   \end{pmatrix}
   +
   \begin{pmatrix}
   x_c \\
   y_c \\
   z_c
   \end{pmatrix}

where :math:`(x_c, y_c, z_c)` is the center of rotation.

**Interpolation**

After rotation, the points may not align with the discrete grid of the original data array. To obtain the values at these new coordinates, spline interpolation is used:

- **TensorSpline Interpolation**: This method performs multidimensional spline interpolation, providing smooth and accurate values for the rotated data.

Implementation Details
----------------------

1. **Centering Coordinates**: The data coordinates are shifted so that the center of rotation aligns with the origin.

2. **Applying Rotation Matrix**: The appropriate rotation matrix is applied to the centered coordinates.

3. **Translating Back**: The rotated coordinates are translated back by adding the center coordinates.

4. **Interpolation**: Spline interpolation is used to compute the data values at the new rotated coordinates.

Examples
--------

**Rotate a 2D Array by 45 Degrees**

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from splineops import rotate

   # Create a sample 2D data array (e.g., a Gaussian blob)
   x = np.linspace(-5, 5, 100)
   y = np.linspace(-5, 5, 100)
   X, Y = np.meshgrid(x, y)
   Z = np.exp(-0.1 * (X**2 + Y**2))

   # Rotate the data by 45 degrees
   rotated_Z = rotate(Z, angle=45)

   # Plot the original and rotated data
   fig, axes = plt.subplots(1, 2, figsize=(10, 5))
   axes[0].imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
   axes[0].set_title('Original Data')
   axes[1].imshow(rotated_Z, extent=[-5, 5, -5, 5], origin='lower', cmap='viridis')
   axes[1].set_title('Rotated Data (45°)')
   plt.show()

**Rotate a 3D Volume Around an Arbitrary Axis**

.. code-block:: python

   import numpy as np
   from splineops import rotate
   # Visualization libraries like PyVista or Mayavi can be used for 3D data

   # Create a sample 3D data array (e.g., a 3D Gaussian blob)
   x = np.linspace(-5, 5, 50)
   y = np.linspace(-5, 5, 50)
   z = np.linspace(-5, 5, 50)
   X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
   V = np.exp(-0.1 * (X**2 + Y**2 + Z**2))

   # Define the rotation axis and angle
   axis = (1, 1, 0)  # Arbitrary axis
   angle = 30        # Degrees

   # Rotate the data
   rotated_V = rotate(V, angle=angle, axis=axis)

   # Visualization of 3D data would require appropriate tools
   # For example, using PyVista or Mayavi to render volumetric data

API Reference
-------------

For more details, see the :ref:`Rotate API documentation <api-rotate>`.

