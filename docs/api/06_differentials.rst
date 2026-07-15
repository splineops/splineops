.. splineops/docs/api/06_differentials.rst

.. _api-differentials:

Differentials
=============

The preferred public name is :class:`splineops.differentials.Differentials`;
the historical lowercase ``differentials`` name remains available.  The class
computes image differentials using cubic B-spline interpolation, including:

- **Gradient Magnitude** - the rate of intensity change.
- **Gradient Direction** - the orientation of maximum change.
- **Laplacian** - the sum of second-order derivatives.
- **Largest Hessian Eigenvalue** - the maximal curvature.
- **Smallest Hessian Eigenvalue** - the minimal curvature.
- **Hessian Orientation** - the principal direction of curvature.

``run()`` returns a raw result and preserves the source image.  Visualization
normalization is opt-in through ``normalize=True`` and is rejected for angular
outputs.  Physical sample spacing is accepted as
``spacing=(row_spacing, column_spacing)``.  ``gradient_components()`` returns
``(vertical, horizontal)`` and ``hessian_components()`` returns
``(vertical, cross, horizontal)``.

.. automodule:: splineops.differentials.differentials
   :members:
   :exclude-members: FLT_EPSILON
   :show-inheritance:
   :special-members: __init__, run, get_cross_hessian, get_horizontal_gradient, get_horizontal_hessian, get_vertical_gradient, get_vertical_hessian, anti_symmetric_fir_mirror_on_bounds, symmetric_fir_mirror_on_bounds, get_gradient, get_hessian, get_spline_interpolation_coefficients, get_initial_causal_coefficient_mirror_on_bounds, get_initial_anti_causal_coefficient_mirror_on_bounds, gradient_magnitude, gradient_direction, laplacian, largest_hessian, smallest_hessian, hessian_orientation
   :member-order: bysource
