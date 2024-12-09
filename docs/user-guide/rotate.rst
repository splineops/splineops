Rotate
======

.. currentmodule:: splineops

Overview
--------

The `rotate` function in the `splineops` library enables rotation of 2D or 3D data arrays around a specified axis and center using spline interpolation. This function is widely used in image processing, computer graphics, and scientific computing.

This module supports:

- rotation of both 2D and 3D data arrays;
- specification of arbitrary rotation axes and centers;
- high-quality spline interpolation for accurate transformations.

Mathematical Details
--------------------

2D Rotation
~~~~~~~~~~~

In 2D space, a point :math:`(x, y)` can be rotated around a center point :math:`(x_c, y_c)` by an angle :math:`\theta` (in radians) using the rotation matrix:

.. math::

   \begin{pmatrix}
   x' \\
   y'
   \end{pmatrix}
   =
   \begin{pmatrix}
   \cos\theta & -\sin\theta \\
   \sin\theta & \cos\theta
   \end{pmatrix}
   \begin{pmatrix}
   x - x_c \\
   y - y_c
   \end{pmatrix}
   +
   \begin{pmatrix}
   x_c \\
   y_c
   \end{pmatrix}

Here, :math:`(x', y')` are the coordinates of the rotated point.

3D Rotation
~~~~~~~~~~~

For 3D data, a point :math:`\mathbf{v} = (x, y, z)` can be rotated around an arbitrary axis defined by a unit vector :math:`\mathbf{u} = (u_x, u_y, u_z)` by an angle :math:`\theta` using Rodrigues' rotation formula:

.. math::

   \mathbf{v}' = \mathbf{v} \cos\theta + (\mathbf{u} \times \mathbf{v}) \sin\theta + \mathbf{u} \left( \mathbf{u} \cdot \mathbf{v} \right) (1 - \cos\theta)

Alternatively, the rotation can be expressed with a rotation matrix :math:`\mathbf{R}`:

.. math::

   \mathbf{R} =
   \begin{pmatrix}
   \cos\theta + u_x^2 (1 - \cos\theta) & u_x u_y (1 - \cos\theta) - u_z \sin\theta & u_x u_z (1 - \cos\theta) + u_y \sin\theta \\
   u_y u_x (1 - \cos\theta) + u_z \sin\theta & \cos\theta + u_y^2 (1 - \cos\theta) & u_y u_z (1 - \cos\theta) - u_x \sin\theta \\
   u_z u_x (1 - \cos\theta) - u_y \sin\theta & u_z u_y (1 - \cos\theta) + u_x \sin\theta & \cos\theta + u_z^2 (1 - \cos\theta)
   \end{pmatrix}

The rotated point is calculated as:

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

where :math:`(x_c, y_c, z_c)` represents the center of rotation.

Interpolation
~~~~~~~~~~~~~

The rotation often results in coordinates that do not align with the original data grid. To compute the data values at these rotated coordinates, spline interpolation is used:

- **TensorSpline Interpolation**: This method uses tensor-product B-splines for smooth and accurate interpolation in multiple dimensions, minimizing artifacts such as aliasing and ensuring high-quality results.

Implementation Details
----------------------

1. **Centering Coordinates**: The coordinates are shifted such that the center of rotation aligns with the origin.
   
2. **Applying Rotation Matrix**: The appropriate rotation matrix (2D or 3D) is applied to the centered coordinates.

3. **Translating Back**: The rotated coordinates are shifted back to their original location by adding the center coordinates.

4. **Interpolation**: Spline interpolation is applied to evaluate the rotated data values at the new coordinates.

Rotation Example
----------------

* :ref:`sphx_glr_auto_examples_002_rotate.py`

References
----------

- Unser, M. (1999). `Splines: A perfect fit for signal/image processing <https://ieeexplore.ieee.org/document/7075842>`_. IEEE Signal Processing Magazine.
