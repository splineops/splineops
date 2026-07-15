.. splineops/docs/api/04_adaptive_regression_splines.rst

.. _api-adaptive_regression_splines:

Adaptive Regression Splines
===========================

Sparse piecewise-linear regression and total-variation denoising for 1-D data.
``DenoisingPlan`` reuses the factorization for fixed sample locations and can
solve an explicit lambda path with warm-start state confined to that call.
Convergence diagnostics remain opt-in, and path solving is not advertised as
an unconditional speedup.

.. automodule:: splineops.adaptive_regression_splines.denoising
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: splineops.adaptive_regression_splines.sparsification
   :members:
   :undoc-members:
   :show-inheritance:
