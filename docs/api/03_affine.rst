.. splineops/docs/api/03_affine.rst

.. _api-affine:

Affine
======

Spline-interpolated pull-back affine transforms on 2-D or 3-D spatial data.
The module provides a general matrix-and-offset function, a rotation
convenience function, and a reusable geometry plan.  Explicit spatial axes
allow remaining dimensions to act as independent batch or channel axes.
``backend="auto"`` selects compiled SciPy evaluation for degrees 1--5 with
float32/float64 and named boundaries; ``backend="numpy"`` keeps the independent
reference evaluator. SciPy plans avoid retaining geometry; NumPy plans can
cache geometry behind a caller-controlled memory cap. Both paths use the same
coefficient preparation. ``AffinePlan.prepare_coefficients``
returns an immutable, compatibility-checked ``AffineCoefficientField`` for
deliberate sharing when one sampled field is transformed through several
geometries.  The raw-array ``prefilter`` path remains available for explicit
output-buffer workflows.  ``AffineCoefficientField.save`` and
``AffinePlan.load_coefficients`` provide a JSON-plus-numeric NPZ round trip
with atomic replacement, endian-portable schema-2 values, schema-1 reading,
compatibility validation, and no NumPy object-pickle loading.  Equivalent SciPy
comparisons establish numerical parity, not a universal performance win.

.. automodule:: splineops.affine.affine
   :members:
   :undoc-members:
   :show-inheritance:

See also
--------
:class:`~splineops.spline_interpolation.tensor_spline.TensorSpline`
   The independent continuous spline model used internally for interpolation.
