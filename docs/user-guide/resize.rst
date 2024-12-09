Resize
======

.. currentmodule:: splineops

Overview
--------

The `resize` function in the `splineops` library enables resizing (scaling) of N-dimensional data arrays using advanced spline-based methods. It
- supports arbitrary scaling factors and output sizes;

- handles multi-dimensional data (2D, 3D, and higher);

- has customizable spline degree and boundary extension modes.


Three resizing methods can be called:

- **Standard Interpolation**: Smooth, continuous interpolation.
- **Least-Squares Projection**: Optimal resizing with minimal approximation error. Slower than standard interpolation but with better interpolation quality. It requires float64 precision.
- **Oblique Projection**: Similar to Least-Squares projection, but faster and lower quality. It works well with with float32 precision

In general, it is recommended to use standard interpolation for most cases; in case higher accuracy is required and float64 precision is used, least-squares projection is recommended.
Oblique projection provides better balance of performance, speed and accuracy.

The `resize` module is suitable for a wide range of applications, including Image processing, Scientific visualization and Medical imaging.

Resizing Example
----------------

* :ref:`sphx_glr_auto_examples_001_resize.py`

Mathematical Details
--------------------

B-Splines and Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

B-splines are piecewise polynomial functions with compact support, often used for interpolation. A B-spline of degree :math:`n`, denoted as :math:`\beta_n(x)`, is defined recursively:

.. math::

   \beta_0(x) =
   \begin{cases}
   1, & 0 \leq x < 1, \\
   0, & \text{otherwise},
   \end{cases}

and for :math:`n > 0`,

.. math::

   \beta_n(x) = \frac{x}{n} \beta_{n-1}(x) + \frac{n+1-x}{n} \beta_{n-1}(x-1).

Alternatively, the :math:`n`-degree B-spline can be expressed using truncated power functions and binomial coefficients:

.. math::

   \beta_n(x) = \frac{1}{n!} \sum_{k=0}^{n+1} \binom{n+1}{k} (-1)^k (x-k)_+^n,

where :math:`(x-k)_+^n` is the truncated power function defined as:

.. math::

   (x-k)_+^n =
   \begin{cases}
   (x-k)^n, & x \geq k, \\
   0, & \text{otherwise}.
   \end{cases}

The spline interpolation process approximates a function :math:`f(\mathbf{x})` using a linear combination of shifted B-splines:

.. math::

   s(\mathbf{x}) = \sum_{\mathbf{k}} c_{\mathbf{k}} \beta_n(\mathbf{x} - \mathbf{k}),

where :math:`c_{\mathbf{k}}` are the spline coefficients determined from the input data.

Least-Squares Projection
~~~~~~~~~~~~~~~~~~~~~~~~

Least-squares resizing minimizes the squared error between the original data :math:`f(\mathbf{x})` and the resized data :math:`\tilde{f}(\mathbf{T}^{-1}\mathbf{x})`, where :math:`\mathbf{T}` is the transformation matrix defining the scaling:

.. math::

   \min_{\tilde{f}} \int_{\Omega} \|f(\mathbf{x}) - \tilde{f}(\mathbf{T}^{-1} \mathbf{x})\|^2 \, d\mathbf{x}.

Key Insights:

- The projection is computed using finite differences to solve for the spline coefficients in a way that preserves the :math:`L_2`-norm.

- Pre-filtering is necessary to compute coefficients :math:`c_{\mathbf{k}}` accurately, which involves solving a linear system or applying recursive filtering. These filters minimize aliasing and improve stability.

- This approach is slower than direct interpolation but provides superior quality by minimizing artifacts and preserving fine details.

Oblique Projection
~~~~~~~~~~~~~~~~~~

The oblique projection generalizes the least-squares method by relaxing orthogonality constraints. It maps the input data onto a biorthogonal basis formed by the scaling and wavelet functions:

.. math::

   c_{\mathbf{k}} = \int_{\Omega} f(\mathbf{x}) \phi(\mathbf{x} - \mathbf{k}) \, d\mathbf{x},

where :math:`\phi` represents the biorthogonal dual basis. Unlike least-squares, the oblique projection:

- Reduces computational cost by approximating the exact solution.

- Results in slightly lower interpolation quality compared to least-squares.

This method is particularly effective for scenarios requiring fast computation with acceptable trade-offs in accuracy.

References
----------

- Unser, M. (1999). `Splines: A perfect fit for signal/image processing <https://ieeexplore.ieee.org/document/7075842>`_. IEEE Signal Processing Magazine.
- Muñoz, A., Blu, T., & Unser, M. (2001). `Least-squares image resizing using finite differences <https://ieeexplore.ieee.org/document/941860>`_. IEEE Transactions on Image Processing, 10(9), 1365–1378.
- Thévenaz, P., Blu, T., & Unser, M. (2000). `Interpolation revisited <https://ieeexplore.ieee.org/document/875199>`_. IEEE Transactions on Medical Imaging, 19(7), 739–758.
