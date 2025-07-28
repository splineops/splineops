Resize
======

.. currentmodule:: splineops

Overview
--------

The `resize` function in the `splineops` library enables the resizing (scaling) of N-dimensional data arrays through advanced spline-based methods [1]_, [2]_, [3]_.

Three resizing methods are available:

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

Oblique projection is a generalization of least-squares projection where the synthesis and analysis spline spaces are allowed to differ. 
Instead of computing an orthogonal projection (where the same basis is used for both approximation and analysis), the method employs an auxiliary 
analysis function :math:`\psi(x)` distinct from the synthesis function :math:`\tilde{\varphi}(x)`. The resulting approximation is given by:

.. math::

    s(x) = \sum_k \langle f, \psi_k \rangle \tilde{\varphi}_k(x),

where:

- :math:`\tilde{\varphi}_k(x)` are the synthesis basis functions (typically B-splines of degree :math:`n`);
- :math:`\psi_k(x)` are the translated analysis functions, often chosen to be simpler or more localized.

This formulation leads to an *oblique* rather than orthogonal projection. It trades off a small loss in optimality for improved speed and numerical stability. 
Empirical results show that the signal-to-noise ratio (SNR) degrades only slightly (e.g., 0.1–0.4 dB) compared to the exact least-squares projection [2]_.

Spline Degrees
~~~~~~~~~~~~~~

In the `splineops.resize` implementation, the oblique projection is configured to use:

- **Interpolation degree**: Determines the input model and spline interpolation order.
- **Synthesis spline**: Matches the interpolation degree (used to reconstruct the resized image).
- **Analysis spline**: Set to a lower degree, typically `interpolation degree - 1`.

For example:

.. list-table:: Spline degree configuration in oblique projection
   :header-rows: 1

   * - Method
     - Interpolation Degree
     - Synthesis Spline Degree
     - Analysis Spline Degree
   * - ``linear-fast_antialiasing``
     - 1
     - 1
     - 0
   * - ``quadratic-fast_antialiasing``
     - 2
     - 2
     - 1
   * - ``cubic-fast_antialiasing``
     - 3
     - 3
     - 1

The synthesis spline determines the space onto which the image is projected. The analysis spline is used to compute inner products with the scaled signal, 
effectively acting as a prefilter. Choosing a lower-degree analysis spline (e.g., linear) simplifies the filter computation and enables efficient recursive implementations 
using finite differences.

**Computational and Practical Implications**

- The oblique method retains much of the quality of least-squares projection, especially for moderate downsampling factors.
- It operates correctly in float32 precision and requires less memory and fewer computations than its least-squares counterpart.
- Because the projection is not orthogonal, there may be small residual aliasing or reconstruction errors, especially for high-frequency content or very aggressive downsampling.

This makes oblique projection a compelling compromise: faster and more stable than least-squares, but still significantly more accurate than naive interpolation.

Resize Example
--------------

* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_01_resize_module.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_02_standard_interpolation.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_03_least-squares_projection.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_04_oblique_projection.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_05_compare_different_methods.py`

References
----------

.. [1] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22–38, November 1999.

.. [2] A. Muñoz Barrutia, T. Blu, M. Unser, 
   `Least-Squares Image Resizing Using Finite Differences <https://doi.org/10.1109/83.941860>`_,
   IEEE Transactions on Image Processing, vol. 10, no. 9, pp. 1365–1378,
   September 2001.

.. [3] P. Thévenaz, T. Blu, M. Unser,
   `Interpolation Revisited <https://doi.org/10.1109/42.875199>`_,
   IEEE Transactions on Medical Imaging, vol. 19, no. 7, pp. 739–758,
   July 2000.
