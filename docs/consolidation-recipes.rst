Reusable workflow recipes
=========================

This page collects the repeated-geometry and explicit-axis workflows added by
the consolidation milestone.  The modules remain independent: the common
theme is making expensive state and spatial intent explicit, not introducing
one package-wide plan abstraction.

One coefficient field, several affine geometries
-------------------------------------------------

Changing samples require a new interpolation prefilter.  When the same sampled
frame is evaluated through several affine geometries, compute that coefficient
field once and share it deliberately:

.. code-block:: python

   from splineops.affine import AffinePlan

   first = AffinePlan(frame.shape, first_matrix, first_offset, mode="mirror")
   second = AffinePlan(frame.shape, second_matrix, second_offset, mode="mirror")

   coefficients = first.prepare_coefficients(frame)
   first_result = first.apply_coefficients(coefficients)
   second_result = second.apply_coefficients(coefficients)

The tagged field rejects a plan with a different construction shape, spline
degree, boundary mode, or precision.  ``prefilter`` remains the raw-array,
``out=``-capable alternative when the caller manages that provenance.
Use ``coefficients.save("field.npz")`` and
``compatible_plan.load_coefficients("field.npz")`` when the field crosses a
process boundary.  Saving uses atomic replacement; loading validates JSON
metadata and portable numeric values before rebuilding the tag.  Schema-1
archives remain readable.

The plans must have the same input shape, degree, mode, and dtype.  Ordinary
``plan(frame)`` remains the correct end-to-end call for one geometry.  The
coefficient API does not claim that changing frames can skip their
mathematically required prefilter.

Batch and channel axes
----------------------

Spatial axes are selected explicitly and every remaining slice is independent:

.. code-block:: python

   from splineops.smoothing_splines import SmoothingSplinePlan
   from splineops.differentials import DifferentialPlan

   # NCHW arrays: height and width are the last two axes.
   smoother = SmoothingSplinePlan(batch.shape[-2:], lamb=0.02, gamma=1.5)
   smooth = smoother(batch, axes=(-2, -1))

   derivatives = DifferentialPlan(batch.shape[-2:], spacing=(0.7, 0.7))
   maps = derivatives(batch, spatial_axes=(-2, -1))
   laplacian = derivatives(
       batch,
       gradient=False,
       hessian=False,
       laplacian=True,
       spatial_axes=(-2, -1),
   ).laplacian

``maps.gradient`` follows the order supplied in ``spatial_axes`` and packed
Hessian entries follow ``(00, 01, 11)`` in 2-D.  A Laplacian-only request skips
gradient and mixed-Hessian construction.  Smoothing executes a batched real
FFT; differentials use one batched per-call workspace.
Structured differential destinations are validated together before execution;
they may be writable strided arrays but must not overlap the source or another
component.  The complete affine-to-feature pipeline is recorded in
:doc:`stability-soak`.

Multiscale batches
------------------

Pyramids and wavelets use the same explicit convention:

.. code-block:: python

   from splineops.multiscale.pyramid import get_pyramid_filter, reduce_2d
   from splineops.multiscale.wavelets.haar import HaarWavelets

   g, _, centered = get_pyramid_filter("Spline", 3)
   coarse = reduce_2d(batch, g, centered, spatial_axes=(-2, -1))

   wavelet = HaarWavelets(scales=2)
   coefficients = wavelet.analysis(batch, spatial_axes=(-2, -1))
   reconstructed = wavelet.synthesis(coefficients, spatial_axes=(-2, -1))

Only the selected dimensions are reduced or partitioned.  Wavelet divisibility
requirements apply to those dimensions, not to batch or channel lengths.

Controlled denoising paths
--------------------------

``DenoisingPlan.solve_path`` carries ADMM state only within one explicit call:

.. code-block:: python

   from splineops.adaptive_regression_splines import DenoisingPlan

   plan = DenoisingPlan(x, rho=0.5)
   solutions, diagnostics = plan.solve_path(
       y,
       [0.01, 0.02, 0.05, 0.1],
       return_diagnostics=True,
   )

The lambda order is preserved.  Nearby values can benefit from warm starts,
but a speedup is workload-dependent.  Independent ``solve`` and ``solve_path``
calls do not share mutable iteration state.

Plan inspection
---------------

Plans expose the fixed contract and known retained storage through properties
such as ``configuration`` and ``retained_bytes``.  Denoising reports
``retained_array_bytes`` because SciPy's factorization owns additional opaque
storage.  ``TensorSplineGeometryPlan.is_compatible`` and
``incompatibility_reason`` make geometry checks available without attempting an
evaluation.

Benchmark regression checks
---------------------------

The manual development workflow stores complete JSON/CSV artifacts and applies
the broad policy in ``benchmarks/consolidation-thresholds.json``.  Its checks
use within-run ratios and numerical errors so they remain meaningful across
machines.  They are regression alarms, not release performance promises.
