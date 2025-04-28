Regress
=======

.. currentmodule:: splineops

Overview
--------

The `regress` module in `splineops` provides a method to perform one-dimensional regression with total-variation (TV) regularization on the second derivative [1]_. 
This approach promotes solutions that are piecewise-linear with a minimal number of knots, which makes them ideal for applications that require sparse representations.

Key features of this method:

- It guarantees piecewise-linear solutions with the fewest knots.
- It provides a systematic analysis of unique vs. non-unique solutions.
- It introduces a fast two-step algorithm to efficiently determine the sparsest solutions.
- It supports both interpolation (exact fit) and regression (data fitting with noise).

This method is particularly relevant in machine learning, where sparsity improves generalization, and in signal processing, where fewer knots lead to simpler 
models. It is closely related to rectified linear unit (ReLU) neural networks, which also produce piecewise-linear functions. It can serve as a direct way to achieve a 
fewest-knot solution in 1D, although caution should be exercised in equating fewer parameters with superior overall performance in high-dimensional machine-learning tasks.

Mathematical Background
-----------------------

Problem Formulation
~~~~~~~~~~~~~~~~~~~

The sparsest linear regression problem is formulated as the inverse problem

.. math::

   \arg\min_{f}
   \Biggl(
     \sum_{m=1}^{M} E\bigl(f(x_m),y_m\bigr)
     + \lambda\,\bigl\|\mathrm{D}^2 f\bigr\|_M
   \Biggr),

where

- The term :math:`E(f(x_m), y_m)` measures the fidelity to the data (e.g., quadratic loss :math:`(f(x_m) - y_m)^2`).
- The regularization parameter :math:`\lambda` controls the sparsity.
- The operator :math:`\mathrm{D}^2 f` is a second derivative, which ensures the solution is piecewise-linear.
- The norm :math:`\| \cdot \|_M` is total variation (TV) and promotes sparsity.

This is called the generalized Beurling LASSO (g-BLASSO). It extends the classic LASSO regression to continuous functions.

Representer Theorem
~~~~~~~~~~~~~~~~~~~

A key result in this framework is that the solution to the optimization problem always takes the form

.. math::

    f_{\text{opt}}(x) = b_0 + b_1 x + \sum_{k=1}^{K} a_k (x - \tau_k)_+,

where

- the global linear trend is defined by :math:`b_0, b_1 \in \mathbb{R}`;
- the ReLU function is :math:`(x - \tau_k)_+`;
- the number :math:`K` of knots satisfies :math:`K \leq (M - 2)`, meaning that the model is sparse.

This theorem guarantees that the solutions are adaptive splines with the fewest knots.

Uniqueness and Sparsity
~~~~~~~~~~~~~~~~~~~~~~~

While the g-BLASSO problem always has solutions, it is generally non-unique. 
This module provides a full characterization of the solution set and identifies several cases.

- Cases where the solution is unique (e.g., when certain convexity conditions hold).
- Cases where multiple solutions exist, and how to select the sparsest one.
- The minimal number of knots required for a valid solution.

A dual certificate approach is used to analyze the support of the second derivative, which determines the solution space.

Algorithm
---------

A two-step algorithm is introduced to compute the sparsest solution efficiently.

1. Compute the optimal data points :math:`y_\lambda` as the solution of a discrete :math:`L^{1}`-regularized problem.  
   Techniques such as ADMM can be employed here for efficient optimization in large-scale settings.
2. Apply a sparsification step to obtain the final solution with the fewest knots.

This algorithm is agnostic to uniqueness and runs in linear time :math:`O(M)`, which makes it significantly faster than traditional total-variation denoising methods.

Advantages and Applications
---------------------------

- Fewest-Knots Guarantee: Among all solutions satisfying a given TV2-regularized objective, this algorithm recovers a piecewise-linear spline with the fewest knots. Unlike traditional TV-based methods, it explicitly finds the sparsest solution.
- Exact Interpolation: If you need a piecewise-linear spline that exactly interpolates your data (no noise term), this method can find the lowest-complexity fit. In other words, it will produce the least angular spline that still goes through every data point.
- Time-Series and Segmented Regression: Whenever a piecewise-linear model is suitable (e.g., finance, economics, or epidemiological data such as daily infection rates), this algorithm helps create interpretable "segmented regression" fits with a small number of segments.
- Connection to ReLU Networks: In 1D, neural networks with ReLU activations yield continuous piecewise-linear (CPWL) functions. This algorithm can be viewed as a more direct way to learn a CPWL function with the fewest parameters (knots). Although modern deep networks often over-parameterize for performance reasons, in strictly 1D scenarios, the method here outperforms a naive ReLU network in terms of parameter sparsity.

Limitations and Directions
--------------------------

- Higher Dimensions: The extension of this approach to dimensions :math:`d > 1` is non-trivial. The definition of analogous TV-based regularizations on higher-order derivatives (e.g., Hessian-based penalties) is an active area of research, but a direct multi-dimensional analogue of the fewest-knot guarantee remains challenging.
- Over-Parameterization vs. Minimal Parameters: In practice, neural networks often benefit from having more parameters than necessary. While our method finds the minimal number of parameters for a 1D CPWL model, the *optimal* size of a model in broader machine-learning contexts may not always be about a strict minimization of the number of parameters.

Regularization Parameter
------------------------

The regularization parameter :math:`\lambda` controls the tradeoff between data fidelity and sparsity.

- Small :math:`\lambda` → Interpolates the data, but may overfit.
- Large :math:`\lambda` → Produces smoother results, eventually converging to a linear fit.

A practical way to tune :math:`\lambda` is to plot sparsity vs. data fidelity and to select a balanced value.

Regression Example
------------------

* :ref:`sphx_glr_auto_examples_006_regress_module.py`

References
----------

.. [1] Debarre, T., Denoyelle, Q., Unser, M., & Fageot, J. (2022).
   Sparsest Piecewise-Linear Regression of One-Dimensional Data.  
   Journal of Computational and Applied Mathematics, 406, 114044.  
   `DOI: 10.1016/j.cam.2021.114044 <https://doi.org/10.1016/j.cam.2021.114044>`_.