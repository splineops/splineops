.. splineops/docs/api/06_differentials.rst

.. _api-differentials:

Differentials
=============

The preferred public name is :class:`splineops.differentials.Differentials`;
the historical lowercase ``differentials`` name remains available.  Scalar
2-D images and 3-D volumes are supported.  The class computes differentials
using cubic B-spline interpolation, including:

- **Gradient Magnitude** - the rate of intensity change.
- **Gradient Direction** - the 2-D orientation of maximum change.
- **Laplacian** - the sum of second-order derivatives.
- **Largest Hessian Eigenvalue** - the maximal curvature.
- **Smallest Hessian Eigenvalue** - the minimal curvature.
- **Hessian Orientation** - the 2-D principal direction of curvature.

``run()`` returns a raw result and preserves the source array.  Visualization
normalization is opt-in through ``normalize=True`` and is rejected for angular
outputs.  Physical ``spacing`` contains one value per axis.
``gradient_components()`` returns derivatives in increasing axis order;
``hessian_components()`` uses packed upper-triangular order ``(00, 01, 11)``
in 2-D and ``(00, 01, 02, 11, 12, 22)`` in 3-D.

For changing arrays with fixed shape and spacing,
:class:`~splineops.differentials.differentials.DifferentialPlan` computes
gradient, Hessian, and Laplacian outputs through one per-call cached workspace.

.. automodule:: splineops.differentials.differentials
   :members:
   :exclude-members: FLT_EPSILON
   :show-inheritance:
   :special-members: __init__, run, get_cross_hessian, get_horizontal_gradient, get_horizontal_hessian, get_vertical_gradient, get_vertical_hessian, anti_symmetric_fir_mirror_on_bounds, symmetric_fir_mirror_on_bounds, get_gradient, get_hessian, get_spline_interpolation_coefficients, get_initial_causal_coefficient_mirror_on_bounds, get_initial_anti_causal_coefficient_mirror_on_bounds, gradient_magnitude, gradient_direction, laplacian, largest_hessian, smallest_hessian, hessian_orientation
   :member-order: bysource
