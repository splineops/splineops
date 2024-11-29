Resize Module
=============

.. currentmodule:: splineops

Overview
--------

The `resize` function in the `splineops` library enables resizing (scaling) of N-dimensional data arrays using advanced spline-based methods. These methods are designed to provide high-quality transformations with reduced artifacts such as aliasing, blocking, and blurring. The function supports three interpolation techniques:

- **Standard Interpolation**: Uses B-spline interpolation for smooth scaling.
- **Least-Squares Projection**: Minimizes the approximation error in the least-squares sense.
- **Oblique Projection**: Employs biorthogonal basis functions for specialized resizing needs.

These approaches make the `resize` function suitable for a wide range of applications, including image processing, scientific visualization, and medical imaging.

Key Features:
- Supports arbitrary scaling factors and output sizes.
- Offers multiple interpolation methods:
  - **Standard Interpolation**: Smooth, continuous interpolation.
  - **Least-Squares Projection**: Optimal resizing with minimal approximation error.
  - **Oblique Projection**: Alternative projection for biorthogonal transformations.
- Handles multi-dimensional data (2D, 3D, and higher).
- Customizable spline degree and boundary extension modes.
- Reduces artifacts such as aliasing and blocking.

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_001_resize.py`

Mathematical Details
--------------------

.. dropdown:: B-Splines and Interpolation

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

.. dropdown:: Least-Squares Projection

    In least-squares resizing, the goal is to minimize the error between the original and resized data in the :math:`L_2` sense:

    .. math::

       \min_{\tilde{f}} \int_{\Omega} \|f(\mathbf{x}) - \tilde{f}(\mathbf{T}^{-1} \mathbf{x})\|^2 \, d\mathbf{x},

    where:
    - :math:`f` is the original data.
    - :math:`\tilde{f}` is the resized data.
    - :math:`\mathbf{T}` is the transformation matrix representing scaling.

    The least-squares method uses finite differences to compute the required inner products, ensuring optimal approximation.

.. dropdown:: Oblique Projection

    Oblique projection involves projecting the data onto the space spanned by the scaling functions using biorthogonal basis functions. Unlike least-squares, this projection is not orthogonal, making it suitable for preserving specific directional properties of the data.

References
----------

.. dropdown:: Unser, M. (1999). Splines: A Perfect Fit for Signal/Image Processing.

    *IEEE Signal Processing Magazine*. Available at: http://bigwww.epfl.ch/publications/unser9902p/

.. rubric:: Examples

* :ref:`sphx_glr_auto_examples_001_resize.py`
