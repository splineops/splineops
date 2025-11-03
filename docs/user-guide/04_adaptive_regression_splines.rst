.. splineops/docs/user-guide/04_adaptive_regression_splines.rst

Adaptive Regression Splines
===========================

.. currentmodule:: splineops

Overview
--------

This module performs one-dimensional regression with
total-variation (TV) regularisation on the second derivative [1]_.  
Because TV is measured with the measure norm (denoted
:math:`\|\cdot\|_{\mathcal{M}}`), solutions are piecewise-linear splines that
use few knots, giving very compact models.

Key features
~~~~~~~~~~~~

* Guarantees piecewise-linear solutions with few knots.
* Provides a fast two-step algorithm that returns the sparsest solution found.
* Works for both interpolation (exact fit) and regression (noisy data).

These properties are valuable in machine learning (where sparsity improves
generalisation) and in signal processing (where interpretability matters).
In 1D, the method is closely related to rectified linear unit (ReLU) neural networks, which also
create piecewise-linar functions, but here we obtain the sparsest possible representation
directly.

Mathematical background
-----------------------

Problem formulation
~~~~~~~~~~~~~~~~~~~

.. math::
   :label: eq:gblasso

   f^\star = \operatorname*{arg\,min}_{f}\;
   \Biggl(
     \sum_{m=1}^{M} E\bigl(f(x_m),y_m\bigr)
     \;+\; \lambda\,\|\mathrm{D}^2 f\|_{\mathcal{M}}
   \Biggr).

where

* the term :math:`E` is a data-fidelity term (e.g., squared loss
  :math:`(f(x_m)-y_m)^2`);
* the regularization parameter :math:`\lambda>0` balances fidelity and sparsity;
* the operator :math:`\mathrm{D}^2 f` is the second derivative;
* the norm :math:`\|\cdot\|_{\mathcal{M}}` is the total variation (TV) norm on measures, promoting sparse
  second derivatives.

This is the generalised Beurling LASSO (g-BLASSO):  
it extends the classical LASSO (:math:`L^1` regularisation on vectors)  
and the Beurling LASSO (BLASSO, :math:`L^1` on measures) by inserting the linear
operator :math:`\mathrm{D}^2`.

Representer theorem
~~~~~~~~~~~~~~~~~~~

A solution of the g-BLASSO has the form

.. math::

   f_\text{opt}(x)\;=\;b_0 + b_1 x \;+\;
   \sum_{k=1}^{K} a_k\bigl(x-\tau_k\bigr)_+,

where

* :math:`b_0,b_1\in\mathbb{R}` describe the global trend;
* :math:`(x-\tau_k)_+` is a shifted ReLU function;
* the number :math:`K` satisfies :math:`K\le M-2`, so only *few* knots appear.

Uniqueness and sparsity
~~~~~~~~~~~~~~~~~~~~~~~

The g-BLASSO may admit multiple solutions, but the algorithm implemented here
leverages the theoretical analysis of [1]_ to always return the sparsest
one (minimum K).

Algorithm
---------

The solver uses two stages\*:

1. Data fitting: solve a discrete :math:`L^1`-regularised problem to obtain
   :math:`y_\lambda` (each ADMM iteration costs :math:`\mathcal{O}(M)` and the
   residual decreases like :math:`\mathcal{O}(1/n)`).
2. Sparsification: in exactly :math:`\mathcal{O}(M)` time, extract the
   spline with the fewest knots.

\*Stage 2 is linear-time; stage 1 is linear per iteration.

Advantages and applications
---------------------------

* Few-knot guarantee: the returned spline is the sparsest among all
  feasible solutions.
* Exact interpolation: with :math:`\lambda=0`, the method finds the least
  angular spline through every point.
* Segmented regression: ideal for interpretable fits in finance,
  epidemiology, etc.
* ReLU connection: in 1-D this outperforms naïve ReLU networks in terms of
  parameter count.

Regularisation parameter
------------------------

Choosing :math:`\lambda`:

* Small :math:`\lambda` → exact or near-exact interpolation (risk of over-fit).  
* Large :math:`\lambda` → smoother, eventually linear.

Practical tip: run the solver on a grid of :math:`\lambda` values and
*plot sparsity vs. data-fidelity* (e.g., root-MSE) to pick a balanced point.

Example
-------

* :ref:`sphx_glr_auto_examples_05_adaptive_regression_splines_05_01_adaptive_regression_splines_module.py`

References
----------

.. [1] T. Debarre, Q. Denoyelle, M. Unser, J. Fageot,
   `Sparsest Piecewise-Linear Regression of One-Dimensional Data <https://doi.org/10.1016/j.cam.2021.114044>`_, 
   Journal of Computational and Applied Mathematics, vol. 406,
   paper no. 114044, 30 p., May 1, 2022.
