Rotate
======

.. currentmodule:: splineops

Overview
--------

The `rotate` function in the `splineops` library enables rotation of 2D or 3D data arrays around a specified axis and center using spline interpolation. 
This function is widely used in image processing, computer graphics, and scientific computing [1]_.

2D Rotation
-----------

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
-----------

For 3D data, a point :math:`\mathbf{v} = (x, y, z)` can be rotated around an arbitrary axis defined by a unit vector :math:`\mathbf{u} = (u_x, u_y, u_z)` 
by an angle :math:`\theta` using Rodrigues' rotation formula:

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
-------------

During rotation, the new coordinates may not align with the original data grid, so spline interpolation is employed to compute the corresponding 
data values. This approach uses standard interpolation, which leverages tensor-product B-splines for smooth, accurate results across multiple 
dimensions while minimizing artifacts like aliasing. The process involves first recentering the coordinates so that the rotation center coincides 
with the origin, then applying the appropriate 2D or 3D rotation matrix to the centered coordinates, followed by translating the rotated coordinates 
back to their original reference frame, and finally using spline interpolation to determine the data values at these new positions.

Rotate Example
--------------

* :ref:`sphx_glr_auto_examples_005_rotate_module.py`

References
----------

.. [1] Unser, M. (1999),
   `Splines: A perfect fit for signal/image processing <https://ieeexplore.ieee.org/document/7075842>`_,
   IEEE Signal Processing Magazine.
