Regress
=======

.. currentmodule:: splineops

Overview
--------

The `regress` module in `splineops` provides a method for performing one-dimensional regression using total-variation (TV) regularization on the second derivative [1]_. 
This approach promotes solutions that are piecewise-linear with the minimum number of knots, making it ideal for applications requiring sparse representations.

Key features of this method:

- It guarantees piecewise-linear solutions with minimal knots.
- It provides a systematic analysis of unique vs. non-unique solutions.
- It introduces a fast two-step algorithm to compute sparsest solutions efficiently.
- It supports both interpolation (exact fit) and regression (data fitting with noise).

This technique is particularly relevant in machine learning, where sparsity improves generalization, and in signal processing, where minimal knots lead to simpler 
models. It is closely related to ReLU neural networks, which also produce piecewise-linear functions. It can serve as a more direct way to achieve a 
minimal-knot solution in 1D, although caution should be exercised in equating fewer parameters with superior overall performance in high-dimensional machine-learning tasks [2]_.

Mathematical Background
-----------------------

Problem Formulation
~~~~~~~~~~~~~~~~~~~

The Sparsest Linear Regression problem is formulated as an inverse problem:

.. math::

    \arg\min_{f} \sum_{m=1}^{M} E(f(x_m), y_m) + \lambda \|D^2 f\|_M,

where:

- :math:`E(f(x_m), y_m)` is a data-fidelity term (e.g., quadratic loss :math:`(f(x_m) - y_m)^2/2`).
- :math:`\lambda` is a regularization parameter that controls the sparsity.
- :math:`D^2 f` is the second derivative, ensuring the solution is piecewise-linear.
- :math:`\| \cdot \|_M` is the total-variation norm, which promotes sparsity.

This is called the generalized Beurling LASSO (g-BLASSO), extending classical LASSO regression to continuous functions.

Representer Theorem
~~~~~~~~~~~~~~~~~~~

A key result in this framework is that the solution to the optimization problem always takes the form:

.. math::

    f_{\text{opt}}(x) = b_0 + b_1 x + \sum_{k=1}^{K} a_k (x - \tau_k)_+,

where:

- :math:`b_0, b_1 \in \mathbb{R}` define the global linear trend.
- :math:`(x - \tau_k)_+` is the ReLU function (rectified linear unit).
- The number of knots :math:`K` satisfies :math:`K \leq M - 2`, meaning the model is sparse.

This theorem guarantees that the solutions are adaptive splines with the fewest possible knots.

Uniqueness and Sparsity
~~~~~~~~~~~~~~~~~~~~~~~

While the g-BLASSO problem always has solutions, it is generally non-unique. This module provides a full characterization of the solution set, identifying:

- Cases where the solution is unique (e.g., when certain convexity conditions hold).
- Cases where multiple solutions exist, and how to select the sparsest one.
- The minimum number of knots required for a valid solution.

A dual certificate approach is used to analyze the support of the second derivative, which determines the solution space.

Algorithm
---------

A two-step algorithm is introduced to compute the sparsest solution efficiently:

1. Compute the optimal data points :math:`y_\lambda` by solving a discrete ℓ1-regularized problem.  
   (Techniques such as ADMM [3]_ can be employed here for efficient optimization in large-scale settings.)
2. Apply a sparsification step to obtain the final solution with the minimum number of knots.

This algorithm is agnostic to uniqueness and runs in linear time :math:`O(M)`, making it significantly faster than traditional total-variation denoising methods.

Advantages and Applications
---------------------------

- Fewest-Knots Guarantee: Among all solutions satisfying a given TV2-regularized objective, this algorithm recovers a piecewise-linear spline with the minimal number of knots. Unlike traditional TV-based methods, it explicitly finds the sparsest solution.
- Exact Interpolation: If you need a piecewise-linear spline that exactly interpolates your data (no noise term), this method can find the lowest-complexity fit. In other words, it will produce a spline with the fewest possible breakpoints that still goes through every data point.
- Time-Series and Segmented Regression: Whenever a piecewise-linear model is suitable (e.g., finance, economics, or epidemiological data such as daily infection rates), this algorithm helps create interpretable "segmented regression" fits with a small number of segments.
- Connection to ReLU Networks: In 1D, neural networks with ReLU activations yield continuous piecewise-linear (CPWL) functions. This algorithm can be viewed as a more direct way to learn a CPWL function with the fewest parameters (knots). Although modern deep networks often over-parameterize for performance reasons, in strictly 1D scenarios, the method here outperforms a naive ReLU network in terms of parameter sparsity.

Limitations and Directions
--------------------------

- Higher Dimensions: Extending this approach to dimensions :math:`d > 1` is non-trivial. Defining analogous TV-based regularizations on higher-order derivatives (e.g., Hessian-based penalties) is an active area of research, but a direct multi-dimensional analogue of the fewest-knot guarantee remains challenging.
- Over-Parameterization vs. Minimal Parameters: In practice, neural networks often benefit from having more parameters than necessary. While our method finds the minimal number of parameters for a 1D CPWL model, the *optimal* size of a model in broader machine-learning contexts may not always be about strictly minimizing parameters.

Regularization Parameter
------------------------

The regularization parameter :math:`\lambda` controls the trade-off between data fidelity and sparsity:

- Small :math:`\lambda` → Interpolates the data, but may overfit.
- Large :math:`\lambda` → Produces smoother results, eventually converging to a linear fit.

A practical way to tune :math:`\lambda` is by plotting sparsity vs. data fidelity and selecting a balanced value.

Regression Example
------------------

* :ref:`sphx_glr_auto_examples_006_using_regress_module.py`

References
----------

.. [1] Debarre, T., Denoyelle, Q., Unser, M., & Fageot, J. (2022).
   Sparsest Piecewise-Linear Regression of One-Dimensional Data.  
   Journal of Computational and Applied Mathematics, 406, 114044.  
   `DOI: 10.1016/j.cam.2021.114044 <https://doi.org/10.1016/j.cam.2021.114044>`_.

.. [2] Abbott, S., Hellewell, J., Thompson, R. N., Sherratt, K., Gibbs, H. P., 
   Bosse, N. I., ... & Funk, S. (2020).
   Estimating the time-varying reproduction number of SARS-CoV-2 using national
   and subnational case counts.
   PLoS ONE, 15(7), e0237901.
   `DOI: 10.1371/journal.pone.0237901 <https://doi.org/10.1371/journal.pone.0237901>`_.

.. [3] Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011).  
   Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers.  
   Foundations and Trends in Machine Learning, 3(1), 1-122.  
   `DOI: 10.1561/2200000016 <https://doi.org/10.1561/2200000016>`_.
