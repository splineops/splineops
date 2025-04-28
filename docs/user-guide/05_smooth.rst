Smooth
======

.. currentmodule:: splineops

Overview
--------

The `smooth` module in `splineops` implements fractional smoothing splines, which are optimal estimators for smooth function approximation and interpolation [1]_, [2]_, [3]_.
Unlike standard polynomial splines, these splines are derived from fractional differential operators, making them highly adaptable for self-similar and fractal-like signals.

Key features.

- It supports 1D and N-dimensional smoothing splines.
- It provides a recursive-filtering implementation for fast computation.
- It uses fractional-order derivatives for the fine control of smoothness.
- It implements fast Fourier transform (FFT)-based methods for large-scale smoothing.
- It provides direct interpolation and denoising functionality.

These methods are particularly useful in signal processing, image reconstruction, and time-series modeling, especially for noisy or fractal-like data.

Problem Formulation
-------------------

Smoothing splines solve a regularized variational problem where the objective is to fit a function :math:`f(x)` to given data points :math:`(x_m, y_m)`, 
while penalizing roughness. The problem is formalized as

.. math::

   \arg\min_{f}
   \Biggl(
    \sum_{m=1}^{M} E\bigl(f(x_m),y_m\bigr)
    + \,\lambda\,\bigl\|\mathrm{D}^\gamma f\bigr\|_M\
   \Biggr),

where:

- The data-fidelity term is :math:`E(f(x_m), y_m)`. It is typically quadratic, with :math:`E(f(x_m), y_m) = (f(x_m) - y_m)^2`.
- The regularization parameter is :math:`\lambda`, which offer control over the smoothness.
- The fractional derivative of order :math:`\gamma = H + 0.5` is :math:`\mathrm{D}^\gamma f`.
- The norm :math:`\| \cdot \|_M` represents the total-variation norm and enforces smoothness.

This formulation ensures that the smoothing-spline solution is a fractional B-spline.

Fractional B-Splines
--------------------

Fractional splines generalize classical polynomial splines by allowing non-integer derivatives. The smoothing spline minimizes an energy functional of the form

.. math::

    \| \mathrm{D}^\gamma f \|^2_M,

which is equivalent to the application of a Butterworth-like low-pass filter.

For a discrete signal :math:`y[n]`, the solution is given by

.. math::

    y_{\text{smooth}} = \mathcal{F}^{-1} \left( H(\omega) \mathcal{F}(y) \right),

where the smoothing filter is

.. math::

    H(\omega) = \frac{1}{1 + \lambda |\omega|^{2\gamma}}.

This filter attenuates high frequencies and leads to optimal smoothing.

Regularization Parameter
------------------------

The regularization parameter :math:`\lambda` balances data fidelity and smoothness.

- Small :math:`\lambda`: Preserves details but may fail to attenuate noise.
- Large :math:`\lambda`: Produces a smooth function but may also oversmooth.

For images and high-dimensional data, a typical choice is :math:`\lambda \in [0.05, 0.2]`.

Smooth Example
--------------

* :ref:`sphx_glr_auto_examples_007_smooth_module.py`

References
----------

.. [1] Unser, M., & Blu, T,
    Self-Similarity: Part I — Splines and Operators,
    IEEE Transactions on Signal Processing, vol. 55, no. 4, pp. 1352-1363. 2007.

.. [2] Blu, T. & M. Unser,
    Self-Similarity: Part II — Optimal Estimation of Fractal Processes,
    IEEE Transactions on Signal Processing, vol. 55, no. 4, pp. 1364-1378, 2007.

.. [3] Unser, M., & Blu, T,
    Fractional Splines and Wavelets,
    SIAM Review, vol. 42, no. 1, pp. 43-67, 2000.