Smooth
======

.. currentmodule:: splineops

Overview
--------

The `smooth` module in `splineops` implements fractional smoothing splines, which are optimal estimators for smooth function approximation and interpolation [1]_, [2]_, [3]_.
Unlike standard polynomial splines, these splines are derived from fractional differential operators, making them highly adaptable for self-similar and fractal-like signals.

Key features:

- It supports 1D and N-dimensional smoothing splines.
- It provides a recursive filtering implementation for fast computation.
- It uses fractional-order derivatives for enhanced smoothness control.
- It implements fast Fourier transform (FFT)-based methods for large-scale smoothing.
- It provides direct interpolation and denoising functionality.

These methods are particularly useful in signal processing, image reconstruction, and time-series modeling, especially for noisy or fractal-like data.

Problem Formulation
-------------------

Smoothing splines solve a regularized variational problem where the objective is to fit a function :math:`f(x)` to given data points :math:`(x_m, y_m)`, 
while penalizing roughness:

.. math::

    \arg\min_{f} \sum_{m=1}^{M} E(f(x_m), y_m) + \lambda \| D^\gamma f \|_M,

where:

- :math:`E(f(x_m), y_m)` is a data-fidelity term, typically quadratic: :math:`(f(x_m) - y_m)^2/2`.
- :math:`\lambda` is the regularization parameter, controlling the smoothness.
- :math:`D^\gamma f` is the fractional derivative of order :math:`\gamma = H + 0.5`.
- :math:`\| \cdot \|_M` represents the total-variation norm, enforcing smoothness.

This formulation ensures that the smoothing spline solution is a fractional B-spline.

Fractional B-Splines
--------------------

Fractional splines generalize classical polynomial splines by allowing non-integer derivatives. The smoothing spline minimizes an energy functional of the form:

.. math::

    \| D^\gamma f \|^2_M,

which is equivalent to applying a Butterworth-like low-pass filter.

For a discrete signal :math:`y[n]`, the solution is given by:

.. math::

    y_{\text{smooth}} = \mathcal{F}^{-1} \left( H(\omega) \mathcal{F}(y) \right),

where the smoothing filter is:

.. math::

    H(\omega) = \frac{1}{1 + \lambda |\omega|^{2\gamma}}.

This filter attenuates high frequencies, leading to optimal smoothing.

Regularization Parameter
------------------------

The regularization parameter :math:`\lambda` balances data fidelity vs. smoothness:

- Small :math:`\lambda` → Preserves more details but may leave noise.
- Large :math:`\lambda` → Produces a smoother function but may oversmooth.

For images and high-dimensional data, a typical choice is :math:`\lambda \approx 0.05 - 0.2`.

Smooth Example
--------------

* :ref:`sphx_glr_auto_examples_007_using_smooth_module.py`

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