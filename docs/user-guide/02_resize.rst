Resize
======

.. currentmodule:: splineops

Overview
--------

The `resize` function in the `splineops` library enables resizing (scaling) of N-dimensional data arrays using advanced spline-based methods [1]_, [2]_, [3]_.

Three resizing methods are available:

- Standard Interpolation: Smooth, continuous interpolation.
- Least-Squares Projection: Optimal resizing with minimal approximation error. Slower than standard interpolation but with better interpolation quality. It requires 64-bit float precision.
- Oblique Projection: Similar to Least-Squares projection, but faster and lower quality. It works well with with 32-bit float precision.

It is recommended to use standard interpolation for most cases; in case higher accuracy is required and float64 precision is used, least-squares projection is recommended.
Oblique projection provides better balance of performance, speed and accuracy.

Standard Interpolation
----------------------

B-spline interpolation is a method for reconstructing a smooth function from discrete data points using B-splines as basis functions. Given a discrete sequence :math:`\{f_k\}`, 
the interpolation function is defined as:

.. math::

    s(x) = \sum_k c_k \beta_n(x - k),

where:

- :math:`\beta_n(x)` is the B-spline of degree :math:`n`,
- :math:`c_k` are the interpolation coefficients obtained by applying a prefilter to the input samples.

The key property of B-splines is their compact support, which ensures efficient computation while maintaining high smoothness. The interpolation requirement,

.. math::

    s(k) = f_k,

is satisfied by computing the coefficients :math:`c_k` through a digital prefiltering step using a recursive IIR implementation.

Least-Squares Projection
------------------------

Least-squares projection provides an optimal approximation of a function in a given space by minimizing the squared error. Instead of direct interpolation, 
the least-squares approach seeks to find the function :math:`s(x)` in a spline space :math:`V_n` that best approximates a given function :math:`f(x)` in the sense of:

.. math::

    \min_{s \in V_n} \int |f(x) - s(x)|^2 dx.

The least-squares approximation is obtained by projecting :math:`f(x)` onto the space spanned by the basis functions. This projection is given by:

.. math::

    s(x) = \sum_k \langle f, \varphi_k \rangle \tilde{\varphi}_k(x),

where:

- :math:`\varphi_k(x)` are the basis functions (typically B-splines),
- :math:`\tilde{\varphi}_k(x)` are their duals, ensuring biorthogonality.

This method effectively reduces aliasing and blocking artifacts, improving image quality, especially for downsampling.

Oblique Projection
------------------

Oblique projection is a generalization of least-squares projection where the approximation space and the analysis space are different. Instead of computing an orthogonal 
projection, we use an auxiliary analysis function :math:`\psi(x)`, leading to an approximation:

.. math::

    s(x) = \sum_k \langle f, \psi_k \rangle \tilde{\varphi}_k(x).

If :math:`\psi_k = \tilde{\varphi}_k`, we obtain the orthogonal projection (least-squares solution). Otherwise, when :math:`\psi_k` differs from :math:`\tilde{\varphi}_k`, 
the projection is oblique.

The oblique projection has lower computational complexity than the least-squares projection, as it avoids explicit computation of the optimal prefilter. 
However, it introduces a slight approximation error depending on the angle between the analysis and synthesis spaces.


Resize Example
--------------

* :ref:`sphx_glr_auto_examples_004_using_resize_module.py`

References
----------

.. [1] Unser, M. (1999),
    `Splines: A perfect fit for signal/image processing <https://ieeexplore.ieee.org/document/7075842>`_,
    IEEE Signal Processing Magazine.

.. [2] Muñoz, A., Blu, T., & Unser, M. (2001), 
    `Least-squares image resizing using finite differences <https://ieeexplore.ieee.org/document/941860>`_,
    IEEE Transactions on Image Processing, 10(9), 1365–1378.

.. [3] Thévenaz, P., Blu, T., & Unser, M. (2000), 
    `Interpolation revisited <https://ieeexplore.ieee.org/document/875199>`_,
    IEEE Transactions on Medical Imaging, 19(7), 739–758.
