Resize
======

.. currentmodule:: splineops

Overview
--------

The `resize` function in the `splineops` library enables the resizing (scaling) of N-dimensional data arrays through advanced spline-based methods [1]_, [2]_, [3]_.

Three resizing methods are available.

- Standard Interpolation: Smooth, continuous interpolation.
- Least-Squares Projection: Optimal resizing with minimal approximation error. Slower than standard interpolation but with better interpolation quality. It requires 64-bit float precision.
- Oblique Projection: Similar to least-squares projection, with tunable speed at the expense of quality. It works well with with 32-bit float precision.

It is recommended to use standard interpolation for most cases; in case higher accuracy is required and float64 precision is used, least-squares projection is recommended.
Oblique projection provides a finer balance of performance, speed, and accuracy.

Standard Interpolation
----------------------

B-spline interpolation reconstructs a smooth function from samples using a B-spline as basis function.  
The interpolation function is defined as

.. math::

    s(x) = \sum_k c_k \beta^{n}(x - k),

where:

- the B-spline of degree :math:`n` is :math:`\beta^{n}(x)`;
- the interpolation coefficients :math:`c_k` are obtained by the application of a prefilter to the samples.

The key property of B-splines is their compact support, which ensures efficient computation while maintaining high smoothness. 
Given a discrete sequence :math:`\{f_k\}` of samples, the interpolation requirement

.. math::

    s(k) = f_k

is satisfied by a proper choice of the coefficients :math:`c_k`. We establish them through a digital prefiltering step that involves the application 
of a recursive IIR filter to the sequence :math:`\{f_k\}`.

Least-Squares Projection
------------------------

Least-squares projection provides an optimal approximation of a function. Instead of direct interpolation, 
the least-squares approach seeks to find the function :math:`s(x)` in a spline space :math:`V_n` that best approximates a given function :math:`f(x)` in the sense of

.. math::

    \min_{s \in V_n} \int |f(x) - s(x)|^2 \mathrm{d}x.

The least-squares approximation is obtained by projecting :math:`f(x)` onto the space spanned by the basis functions. This projection is given by

.. math::

    s(x) = \sum_k \langle f, \varphi_k \rangle \tilde{\varphi}_k(x),

where

- the basis functions are :math:`\varphi_k(x)` (typically, B-splines);
- the duals of the basis functions are :math:`\tilde{\varphi}_k(x)`, which ensures biorthonormality.

This method effectively reduces aliasing and blocking artifacts. It improves image quality, especially for downsampling.

Oblique Projection
------------------

Oblique projection is a generalization of least-squares projection where the approximation space and the analysis space are allowed to differ. 
Instead of computing an orthonormal projection, we use an auxiliary analysis function :math:`\psi(x)`, which leads to the approximation

.. math::

    s(x) = \sum_k \langle f, \psi_k \rangle \tilde{\varphi}_k(x).

If :math:`\psi_k = \tilde{\varphi}_k`, we obtain the orthonormal projection (least-squares solution). Otherwise, when :math:`\psi_k` differs from :math:`\tilde{\varphi}_k`, 
the projection is oblique.

The computational complexity of an oblique projection is lower than that of a least-squares projection, as the former avoids the explicit computation of the optimal prefilter. 
However, some approximation error arises, depending on the angle between the analysis and synthesis spaces.


Resize Example
--------------

* :ref:`sphx_glr_auto_examples_004_resize_module.py`

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
