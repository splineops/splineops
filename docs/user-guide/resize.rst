Resize Module
=============

.. currentmodule:: splineops

Overview
--------

The `resize` function in the `splineops` library enables resizing (scaling) of N-dimensional data arrays using advanced spline-based methods. It
- supports arbitrary scaling factors and output sizes;

- handles multi-dimensional data (2D, 3D, and higher);

- has customizable spline degree and boundary extension modes.


Three resizing methods can be called:

- **Standard Interpolation**: Smooth, continuous interpolation.
- **Least-Squares Projection**: Optimal resizing with minimal approximation error. Slower than standard interpolation but with better interpolation quality.
- **Oblique Projection**: Similar to Least-Squares projection, but faster and lower quality.

The `resize` module is suitable for a wide range of applications, including Image processing, Scientific visualization and Medical imaging.

Examples
--------

* :ref:`sphx_glr_auto_examples_001_resize.py`

Mathematical Details
--------------------

**B-Splines and Interpolation**

B-splines are piecewise polynomial functions with compact support, commonly used as basis functions for interpolation. The B-spline of degree :math:`n` is defined recursively as:

.. math::

   \beta_0(x) =
   \begin{cases}
   1, & 0 \leq x < 1, \\
   0, & \text{otherwise},
   \end{cases}

and for :math:`n > 0`,

.. math::

   \beta_n(x) = \frac{x}{n} \beta_{n-1}(x) + \frac{n+1-x}{n} \beta_{n-1}(x-1).

Alternatively, B-splines can be expressed in terms of truncated power functions and binomial coefficients:

.. math::

   \beta_n(x) = \frac{1}{n!} \sum_{k=0}^{n+1} \binom{n+1}{k} (-1)^k (x-k)_+^n,

where :math:`(x-k)_+^n` is the truncated power function:

.. math::

   (x-k)_+^n =
   \begin{cases}
   (x-k)^n, & x \geq k, \\
   0, & \text{otherwise}.
   \end{cases}

The spline interpolation process represents the input data :math:`f(\mathbf{x})` as a weighted sum of B-splines:

.. math::

   s(\mathbf{x}) = \sum_{\mathbf{k}} c_{\mathbf{k}} \beta_n(\mathbf{x} - \mathbf{k}),

where :math:`c_{\mathbf{k}}` are the spline coefficients.

The resized data is then obtained by evaluating :math:`s(\mathbf{x})` at the transformed coordinates.

**Least-Squares Projection**

In least-squares resizing, the goal is to minimize the error between the original and resized data in the :math:`L_2` sense:

.. math::

   \min_{\tilde{f}} \int_{\Omega} \|f(\mathbf{x}) - \tilde{f}(\mathbf{T}^{-1} \mathbf{x})\|^2 \, d\mathbf{x},

where:
- :math:`f` is the original data.
- :math:`\tilde{f}` is the resized data.
- :math:`\mathbf{T}` is the transformation matrix representing scaling.

The least-squares method uses finite differences to compute the required inner products, ensuring optimal approximation.

**Oblique Projection**

Oblique projection involves projecting the data onto the space spanned by the scaling functions using biorthogonal basis functions. Unlike least-squares, this projection is not orthogonal, making it suitable for preserving specific directional properties of the data.

References
----------

- Unser, M. (1999). `Splines: A perfect fit for signal/image processing <https://ieeexplore.ieee.org/document/7075842>`_. IEEE Signal Processing Magazine.
- Muñoz, A., Blu, T., & Unser, M. (2001). `Least-squares image resizing using finite differences <https://ieeexplore.ieee.org/document/941860>`_. IEEE Transactions on Image Processing, 10(9), 1365–1378.



