.. splineops/docs/user-guide/03_rotate.rst

Rotate
======

.. currentmodule:: splineops

Overview
--------

The *rotate* function in the *splineops* library allows for the rotation of 2D or 3D data arrays around a specified axis and center using spline interpolation. 
This function is widely used in image processing, computer graphics, and scientific computing [1]_.

2D Rotation
-----------

In 2D space, a point of coordinates :math:`(x, y)` can be rotated around a center point :math:`(x_\mathrm{c}, y_\mathrm{c})` by an angle :math:`\theta` 
(in radians) using the rotation matrix.

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
   x - x_\mathrm{c} \\
   y - y_\mathrm{c}
   \end{pmatrix}
   +
   \begin{pmatrix}
   x_\mathrm{c} \\
   y_\mathrm{c}
   \end{pmatrix}

Here, :math:`(x', y')` are the coordinates of the rotated point.

3D Rotation
-----------

For 3D data, a point :math:`\mathbf{v} = (x, y, z)` can be rotated around an arbitrary axis defined by a 
unit vector :math:`\mathbf{u} = (u_\mathrm{x}, u_\mathrm{y}, u_\mathrm{z})` 
by an angle :math:`\theta` using Rodrigues' rotation formula

.. math::

   \mathbf{v}' = \mathbf{v} \cos\theta + (\mathbf{u} \times \mathbf{v}) \sin\theta + \mathbf{u} \left( \mathbf{u} \cdot \mathbf{v} \right) (1 - \cos\theta)

Alternatively, the rotation can be expressed with the rotation matrix

.. math::

   \mathbf{R} =
   \begin{pmatrix}
     \cos\theta + u_{\mathrm{x}}^2\,(1 - \cos\theta)
     & u_{\mathrm{x}}\,u_{\mathrm{y}}\,(1 - \cos\theta) - u_{\mathrm{z}}\,\sin\theta
     & u_{\mathrm{x}}\,u_{\mathrm{z}}\,(1 - \cos\theta) + u_{\mathrm{y}}\,\sin\theta \\[6pt]
     u_{\mathrm{y}}\,u_{\mathrm{x}}\,(1 - \cos\theta) + u_{\mathrm{z}}\,\sin\theta
     & \cos\theta + u_{\mathrm{y}}^2\,(1 - \cos\theta)
     & u_{\mathrm{y}}\,u_{\mathrm{z}}\,(1 - \cos\theta) - u_{\mathrm{x}}\,\sin\theta \\[6pt]
     u_{\mathrm{z}}\,u_{\mathrm{x}}\,(1 - \cos\theta) - u_{\mathrm{y}}\,\sin\theta
     & u_{\mathrm{z}}\,u_{\mathrm{y}}\,(1 - \cos\theta) + u_{\mathrm{x}}\,\sin\theta
     & \cos\theta + u_{\mathrm{z}}^2\,(1 - \cos\theta)
   \end{pmatrix}

The rotated point is calculated as

.. math::

   \begin{pmatrix}
   x' \\
   y' \\
   z'
   \end{pmatrix}
   =
   \mathbf{R}
   \begin{pmatrix}
   x - x_\mathrm{c} \\
   y - y_\mathrm{c} \\
   z - z_\mathrm{c}
   \end{pmatrix}
   +
   \begin{pmatrix}
   x_\mathrm{c} \\
   y_\mathrm{c} \\
   z_\mathrm{c}
   \end{pmatrix}

Interpolation
-------------

The rotated coordinates may not coincide with the original data grid, so spline interpolation is employed for the resampling of the rotated data. 
The approach documented here uses standard interpolation, which leverages tensor-product B-splines for smooth, accurate results across multiple 
dimensions while minimizing artifacts like aliasing. The process asks one to first recenter the coordinates so that the rotation center coincides 
with the origin, then to apply the appropriate 2D or 3D rotation matrix to the recentered coordinates, followed by a translation of the rotated recentered coordinates 
back to their original reference frame, to compensate for the recentering step, and finally to use spline interpolation to determine the data values at these new positions.

.. note::
   The geometry of the transform (center, axis, angle) is identical across methods; what changes is the spline used for resampling. 
   Different spline degrees trade sharpness for smoothness (e.g., degree 0/nearest → blocky but fast; degree 1/linear → slight blur; degree 3/cubic → smoother, higher-quality edges).

Rotate Examples
---------------

* :ref:`sphx_glr_auto_examples_04_rotate_04_01_rotate_image.py`
* :ref:`sphx_glr_auto_examples_04_rotate_04_02_rotation_animation.py`

References
----------

.. [1] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22-38, November 1999.
