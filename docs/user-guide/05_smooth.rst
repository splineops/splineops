Smooth
======

.. currentmodule:: splineops

Overview
--------

The `smooth` module in `splineops` implements fractional smoothing splines, which are optimal estimators for smooth function approximation and interpolation [1]_, [2]_, [3]_.
Unlike standard polynomial splines, these splines are derived from fractional differential operators, making them highly adaptable for self-similar and fractal-like signals.

This module

- supports 1D and N-dimensional smoothing splines;
- provides a recursive-filtering implementation for fast computation;
- uses fractional-order derivatives for the fine control of smoothness;
- implements fast Fourier transform (FFT)-based methods for large-scale smoothing
- and provides direct interpolation and denoising functionality.

These methods are particularly useful in signal processing, image reconstruction, and time-series modeling, especially for noisy or fractal-like data.

Problem Minimization
--------------------

Smoothing splines solve a regularized variational problem where the objective is to fit a function :math:`f(x)` to given data points :math:`(x_m, y_m)`, 
while penalizing roughness. The problem is formalized as

.. math::

   \arg\min_{f}
   \Biggl(
    \sum_{m=1}^{M} E\bigl(f(x_m),y_m\bigr)
    + \,\lambda\,\bigl\|\mathrm{D}^\gamma f\bigr\|_{L_{2}}\
   \Biggr),

where

- the data-fidelity term is :math:`E(f(x_m), y_m)`. It is quadratic, with :math:`E(f(x_m), y_m) = (f(x_m) - y_m)^2`;
- the regularization parameter is :math:`\lambda`, which offer control over the smoothness;
- the fractional derivative of order :math:`\gamma` is :math:`\mathrm{D}^\gamma f`;
- the norm :math:`\| \cdot \|_{L_{2}}` represents the total-variation norm and enforces smoothness.

This formulation ensures that the smoothing-spline solution is a fractional B-spline.

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

.. [1] M. Unser, T. Blu, `Self-Similarity: Part I—Splines and Operators <https://doi.org/10.1109/TSP.2006.890843>`_, 
   IEEE Transactions on Signal Processing, vol. 55, no. 4, pp. 1352–1363,
   April 2007.

.. [2] T. Blu, M. Unser, `Self-Similarity: Part II—Optimal Estimation of
   Fractal Processes <https://doi.org/10.1109/TSP.2006.890845>`_, 
   IEEE Transactions on Signal Processing, vol. 55, no. 4, pp. 1364–1378,
   April 2007.

.. [3] M. Unser, T. Blu, `Fractional Splines and Wavelets <https://doi.org/10.1137/S0036144598349435>`_, 
   SIAM Review, vol. 42, no. 1, pp. 43–67, March 2000.
