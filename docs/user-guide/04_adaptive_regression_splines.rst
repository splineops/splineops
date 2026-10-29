.. splineops/docs/user-guide/04_adaptive_regression_splines.rst

Adaptive Regression Splines
===========================

.. currentmodule:: splineops

Overview
--------

The :ref:`adaptive regression splines <api-adaptive_regression_splines>` module in :ref:`SplineOps <api-index>` performs 
one-dimensional regression with total-variation (TV) regularisation on the second derivative [1]_.  
Because TV is measured with the measure norm (denoted
:math:`\|\cdot\|_{\mathcal{M}}`), solutions are piecewise-linear splines that
use few knots, giving very compact models.

Key Features
~~~~~~~~~~~~

* Produces piecewise-linear solutions with few knots.
* Provides a two-step denoising and sparsification implementation.
* Works for both interpolation (exact fit) and regression (noisy data).

These properties are valuable in machine learning (where sparsity improves
generalisation) and in signal processing (where interpretability matters).
In 1D, the method is closely related to rectified linear unit (ReLU) neural networks, which also
create piecewise-linear functions, but here we obtain the sparsest possible representation
directly.

Mathematical Background
-----------------------

Problem Formulation
~~~~~~~~~~~~~~~~~~~

.. math::
   :label: eq:gblasso

   f^\star = \operatorname*{arg\,min}_{f}\;
   \Biggl(
     \frac{1}{2}\sum_{m=1}^{M} (f(x_m)-y_m)^2
     \;+\; \lambda\,\|\mathrm{D}^2 f\|_{\mathcal{M}}
   \Biggr).

where

* the regularization parameter :math:`\lambda>0` balances fidelity and sparsity;
* the operator :math:`\mathrm{D}^2` is the second derivative;
* the norm :math:`\|\cdot\|_{\mathcal{M}}` is the total variation (TV) norm on measures, promoting sparse
  second derivatives.

This is the generalised Beurling LASSO (g-BLASSO):  
it extends the classical LASSO (:math:`L^1` regularisation on vectors)  
and the Beurling LASSO (BLASSO, :math:`L^1` on measures) by inserting the linear
operator :math:`\mathrm{D}^2`.

Representer Theorem
~~~~~~~~~~~~~~~~~~~

A solution of the g-BLASSO has the form

.. math::

   f_\text{opt}(x)\;=\;b_0 + b_1 x \;+\;
   \sum_{k=1}^{K} a_k\bigl(x-\tau_k\bigr)_+,

where

* :math:`b_0,b_1\in\mathbb{R}` describe the global trend;
* :math:`(x-\tau_k)_+` is a shifted ReLU function;
* the number :math:`K` satisfies :math:`K\le M-2`, so only *few* knots appear.

Uniqueness and Sparsity
~~~~~~~~~~~~~~~~~~~~~~~

The g-BLASSO may admit multiple solutions.  The method in [1]_ characterizes a
sparsest solution. The stable CPU implementation is checked against sparse
hinge models and independent primal and dual convex optimizers. The finite
ADMM iteration budget still needs a convergence check for each fitted signal.

Algorithm
---------

The solver uses two stages\*:

1. Data fitting: solve a discrete :math:`L^1`-regularised problem to obtain
   :math:`y_\lambda`. A fixed banded system is factorized once; each ADMM
   iteration uses sparse matrix products and a solve.
2. Sparsification: combine compatible slope changes into a sparse
   piecewise-linear representation. ``sparsity_tol`` controls numerical
   pruning; nonzero tolerance can change the reconstructed samples slightly.

\*Stage 2 is linear-time; stage 1 is linear per iteration.

Advantages and Applications
---------------------------

* Compact representation: redundant numerical knots are pruned while the
  reconstructed piecewise-linear function is checked at the samples.
* Interpolation: :math:`\lambda=0` preserves the observations before optional
  sparsification; use zero pruning tolerance when numerical knots must stay.
* Segmented regression: slope changes give an interpretable fit.
* ReLU connection: the hinge representation is a one-dimensional ReLU model
  with explicitly fitted knot locations and amplitudes.

Regularisation Parameter
------------------------

Choosing :math:`\lambda`:

* Small :math:`\lambda` → exact or near-exact interpolation (risk of over-fit).  
* Large :math:`\lambda` → smoother, eventually linear.

Practical tip: run the solver on a grid of :math:`\lambda` values and
*plot sparsity vs. data-fidelity* (e.g., root-MSE) to pick a balanced point.

For such a sweep, :class:`splineops.adaptive_regression_splines.DenoisingPlan`
factorizes the fixed ``x``/``rho`` system once.  Observations and regularization
strength may change between solves:

.. code-block:: python

   from splineops.adaptive_regression_splines import DenoisingPlan

   plan = DenoisingPlan(x, rho=1e-3)
   solutions = [plan.solve(y, lamb=value) for value in lambda_values]

This removes repeated sparse factorization but does not make the iterative ADMM
work disappear.  ``x`` must stay strictly increasing and ``rho`` is fixed by
the plan; construct another plan when either changes.  Zero regularization and
the linear-regression limit still use their direct branches.

For an ordered sweep on one signal, ``solve_path`` uses controlled warm starts:

.. code-block:: python

   solutions, diagnostics = plan.solve_path(
       y,
       lambda_values,
       return_diagnostics=True,
   )

Iteration state is local to that call.  Repeated calls therefore remain
deterministic and the plan stays safe from hidden cross-call state.  Nearby
penalties often require fewer iterations, but this is not a guaranteed speedup;
the diagnostics expose the actual result.  ``retained_array_bytes`` is a lower
bound covering known NumPy/SciPy sparse arrays, excluding opaque solver-factor
storage.

Convergence diagnostics
-----------------------

``denoise_y`` keeps its historical array return by default.  Set
``return_diagnostics=True`` to receive a ``DenoisingDiagnostics`` record with
the iteration count, convergence flag, and final primal and dual residuals:

.. code-block:: python

   from splineops.adaptive_regression_splines import denoise_y

   denoised, diagnostics = denoise_y(
       x,
       y,
       lamb=1e-3,
       rho=1e-3,
       return_diagnostics=True,
   )

The record makes an exhausted iteration budget visible instead of implying
convergence.  Zero regularization and the linear-regression limit are solved
without ADMM and report zero iterations.  ``rho`` affects convergence speed;
for difficult cases it should be tuned together with :math:`\lambda`.

Treat ``diagnostics.converged=False`` as an unfinished optimization. Increase
``max_iter`` or adjust ``rho`` and solve again before accepting the fit.
Changing ``rho`` changes convergence, not the target objective; construct a
new plan when changing it. A practical check is that two converged solves
with different positive ``rho`` values agree within the requested tolerance.

Coordinate units matter for regularization. Translating ``x`` does not change
the fit. Scaling ``x`` by a positive factor ``s`` requires scaling ``lambda``
by ``s`` to preserve the same fit; scaling ``rho`` by ``s**2`` also preserves
the ADMM linear system. Scaling observations by ``a`` requires scaling
``lambda`` by ``abs(a)``. Center and scale very large or tightly spaced
coordinates before fitting, and choose lambda using residuals or held-out
observations rather than copying an example's value.

Inputs must be finite and real. Solver and sparse-interpolant calculations
use float64, including for integer observations. The zero-regularization
``denoise_y``/``solve`` shortcut preserves the input dtype and returns a copy.

The figure below, taken from
:ref:`sphx_glr_auto_examples_04_adaptive_regression_splines_01_adaptive_regression_splines_module.py`,
shows noisy 1D data (crosses), the TV-denoised samples, and the sparsest
piecewise-linear spline for a given value of :math:`\lambda`, together with
its knot locations.

.. image:: /auto_examples/04_adaptive_regression_splines/images/sphx_glr_01_adaptive_regression_splines_module_002.png
   :align: center
   :width: 100%

Lambda Sweep Animation
----------------------

Here a more comprehensive animation exported from the example :ref:`sphx_glr_auto_examples_04_adaptive_regression_splines_02_lambda_sweep_animation.py`, 
trying different values of :math:`\lambda`.

.. only:: html

  .. raw:: html

    <iframe
      src="../_static/animations/lambda_sweep_animation.html"
      style="width: 100%; height: 650px; border: 0;"
      loading="lazy"
      allow="fullscreen"
      allowfullscreen>
    </iframe>

Example
-------

* :ref:`sphx_glr_auto_examples_04_adaptive_regression_splines_01_adaptive_regression_splines_module.py`
* :ref:`sphx_glr_auto_examples_04_adaptive_regression_splines_02_lambda_sweep_animation.py`

References
----------

.. [1] T. Debarre, Q. Denoyelle, M. Unser, J. Fageot,
   `Sparsest Piecewise-Linear Regression of One-Dimensional Data <https://doi.org/10.1016/j.cam.2021.114044>`_, 
   Journal of Computational and Applied Mathematics, vol. 406,
   paper no. 114044, 30 p., May 1, 2022.
