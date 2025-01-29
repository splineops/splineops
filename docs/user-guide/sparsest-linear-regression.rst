Sparsest Linear Regression
==========================

.. currentmodule:: splineops

Overview
--------

The `sparsest linear regression` module in `splineops` provides a method for performing one-dimensional regression using **total-variation (TV) regularization** on the second derivative. This approach promotes solutions that are **piecewise-linear with the minimum number of knots**, making it ideal for applications requiring **sparse representations**.

Key features of this method:

- Guarantees **piecewise-linear** solutions with minimal knots.
- Provides a systematic analysis of **unique vs. non-unique solutions**.
- Introduces a fast **two-step algorithm** to compute sparsest solutions efficiently.
- Supports both **interpolation (exact fit)** and **regression (data fitting with noise).**

This technique is particularly relevant in **machine learning**, where sparsity improves **generalization**, and in **signal processing**, where minimal knots lead to **simpler models**.

Mathematical Background
------------------------

Problem Formulation
~~~~~~~~~~~~~~~~~~~

The **Sparsest Linear Regression** problem is formulated as an **inverse problem**:

.. math::

    \arg\min_{f} \sum_{m=1}^{M} E(f(x_m), y_m) + \lambda \|D^2 f\|_M,

where:

- :math:`E(f(x_m), y_m)` is a **data-fidelity** term (e.g., quadratic loss :math:`(f(x_m) - y_m)^2/2`).
- :math:`\lambda` is a **regularization parameter** that controls the sparsity.
- :math:`D^2 f` is the **second derivative**, ensuring the solution is **piecewise-linear**.
- :math:`\| \cdot \|_M` is the **total-variation norm**, which promotes sparsity.

This is called the **generalized Beurling LASSO (g-BLASSO)**, extending classical LASSO regression to continuous functions.

Representer Theorem
~~~~~~~~~~~~~~~~~~~

A key result in this framework is that the solution to the optimization problem always takes the form:

.. math::

    f_{\text{opt}}(x) = b_0 + b_1 x + \sum_{k=1}^{K} a_k (x - \tau_k)_+,

where:

- :math:`b_0, b_1 \in \mathbb{R}` define the **global linear trend**.
- :math:`(x - \tau_k)_+` is the **ReLU function** (rectified linear unit).
- The number of **knots** :math:`K` satisfies :math:`K \leq M - 2`, meaning the model is **sparse**.

This theorem guarantees that the solutions are **adaptive splines** with the fewest possible knots.

Uniqueness and Sparsity
~~~~~~~~~~~~~~~~~~~~~~~

While the **g-BLASSO** problem always has solutions, it is generally **non-unique**. This module provides a full **characterization of the solution set**, identifying:

- **Cases where the solution is unique** (e.g., when certain convexity conditions hold).
- **Cases where multiple solutions exist**, and how to select the **sparsest** one.
- The minimum number of **knots required** for a valid solution.

A **dual certificate approach** is used to analyze the **support of the second derivative**, which determines the **solution space**.

Algorithm
---------

A **two-step algorithm** is introduced to compute the **sparsest solution** efficiently:

1. **Compute the optimal data points** :math:`y_\lambda` by solving a **discrete ℓ1-regularized problem**.
2. **Apply a sparsification step** to obtain the final solution with the **minimum number of knots**.

This algorithm is **agnostic to uniqueness** and runs in **linear time** :math:`O(M)`, making it significantly faster than traditional **total-variation denoising** methods.

Example Usage
-------------

Here’s how to use the `sparsest linear regression` module in `splineops`:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from splineops.interpolate.sparsest_linear.denoising import denoise_y
    from splineops.interpolate.sparsest_linear.sparsification import sparsest_interpolant, linear_spline

    # Sample data (noisy)
    x = np.linspace(0, 1, 50)
    y = np.sin(2 * np.pi * x) + 0.1 * np.random.randn(len(x))

    # Regularization parameter
    lamb = 0.01

    # Compute denoised y
    y_denoised = denoise_y(x, y, lamb, rho=lamb)

    # Compute sparsest spline
    knots, amplitudes, polynomial = sparsest_interpolant(x, y_denoised)

    # Plot results
    plt.figure()
    plt.plot(x, y, 'x', label='Noisy Data')
    plt.plot(x, y_denoised, 'o', label='Denoised Data')
    plt.plot(x, linear_spline(x, knots, amplitudes, polynomial), label='Sparsest Regression')
    plt.legend()
    plt.show()

This example demonstrates:

- **Denoising** noisy data using **total-variation regularization**.
- **Finding the sparsest linear spline** with minimal knots.
- **Visualizing the results** to compare noisy, denoised, and sparse regression outputs.

Regularization Parameter
------------------------

The regularization parameter :math:`\lambda` controls the trade-off between **data fidelity** and **sparsity**:

- **Small** :math:`\lambda` → Interpolates the data, but may overfit.
- **Large** :math:`\lambda` → Produces smoother results, eventually converging to a linear fit.

A practical way to **tune** :math:`\lambda` is by plotting **sparsity vs. data fidelity** and selecting a balanced value.

References
----------

The method is based on the following paper:

- **Debarre, T., Denoyelle, Q., Unser, M., & Fageot, J.** (2022). *Sparsest Piecewise-Linear Regression of One-Dimensional Data*.  
  Journal of Computational and Applied Mathematics, 406, 114044.  
  `DOI: 10.1016/j.cam.2021.114044 <https://doi.org/10.1016/j.cam.2021.114044>`_.

.. note::
    This method is closely related to **ReLU neural networks**, which also produce **piecewise-linear functions**.
