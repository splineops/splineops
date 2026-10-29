2.3 numerical API review
============================

Review date: 2026-10-29. This review promotes the documented NumPy/CPU APIs in
:doc:`project-status`. It supersedes July's open graduation checklist;
:doc:`progress` retains that historical snapshot.

What changed in this review
-------------------------------

The review covered construction and query validation, coefficient filtering,
reference and accelerated dispatch, shape/dtype/boundary behavior, persistent
plans, buffers, reconstruction, convergence, and the existing release gates.

New regressions exposed and corrected these issues:

* TensorSpline's absolute grid tolerance could accept nonuniform construction
  coordinates expressed in very small physical units.
* Symmetric degree-4 O-MOMS reported the wrong polynomial degree.
* Adaptive regression's normal equations lost accuracy for large coordinate
  offsets; integer sample arithmetic could overflow before conversion.
* Sparse interpolation could wrap unsigned sample differences, and its
  evaluator could overflow integer products.
* Fractional autocorrelation did not reduce distant frequencies into one
  period before truncating its alias sum.
* Recursive smoothing could overflow or lose normalization at large finite
  regularization parameters, or lose small tails across large sample ranges.
* Affine validation treated empty batches differently across evaluators,
  could discard imaginary geometry, and did not report a plain NPY file as
  an invalid coefficient archive.
* Changing cached plan settings could leave the original factorization,
  frequency response, or interpolation geometry active.

Independent numerical references
------------------------------------

``tests/test_tensorspline_independent_reference.py`` builds basis functions
from SciPy B-splines and finite differences of lower-degree splines. It forms
dense cardinal systems with explicit boundary extension, independently of
SplineOps' support indexing and recursive/FFT/banded coefficient filters.
It covers B-spline degrees 0--9, O-MOMS degrees 0--5, symmetric discontinuity
conventions, all three boundaries, lengths 1/2/9, and mixed 3-D complex models.
The derivative construction follows Blu, Thevenaz and Unser's
`MOMS formulation <https://bigwww.epfl.ch/publications/blu0101.pdf>`_.

``tests/test_stability_numerics.py`` adds an independent box-constrained dual
optimizer for denoising, clamped cubic piecewise-polynomial derivatives for
Differentials, and a direct block-transform and energy check for Haar. These
complement the existing dense smoothing references, projection quadrature,
affine SciPy comparisons, property tests, archive fixtures, and workflow soaks.

Compatibility evidence
--------------------------

The affine persistence and differential buffer APIs have passed repeated
registration and volume-feature workflows since July, including cross-platform
standard soaks and benchmark runs. The 2.3 release preparation added minimum
dependency checks and installed-wheel numerical checks. The current workflows
rerun this evidence for the proposed release revision; a historical green run
does not replace checks on the final commit.

The review keeps existing imports and archive schemas. Plan settings now reject
reassignment: construct a new plan to change a fixed operator. The regression
objective is documented with the implemented ``0.5`` squared-error factor;
existing lambda semantics do not change. Corrected numerical edge cases are
listed in the changelog.

Follow-up repository review
---------------------------

A second pass covered the Python modules, native buffer and concurrency
contracts, fallback dispatch, package contents, examples, and release and
documentation workflows. It found additional numerical-range gaps despite
the existing suite passing. ``tests/test_audit_edge_cases.py`` records the
new regressions:

* Distant TensorSpline queries follow their named boundary mode without
  overflowing integer stencil conversion. Exact integer arithmetic checks
  queries whose physical-to-sample conversion exceeds float64's range.
  Large affine translations also agree across the SciPy and NumPy backends,
  including reuse of prepared coefficients.
* Independent decimal arithmetic checks radial smoothing responses when
  intermediate powers or products overflow or underflow.
* Upsampled time axes contain exactly one coordinate per output sample,
  including factors for which a floating-point step adds an extra endpoint.
* Image SNR remains unchanged when both signals are rescaled. MSE can still
  underflow to zero or overflow to infinity when its true value is outside
  float64's range; this no longer corrupts SNR.
* Integer configuration rejects booleans before conversion, consistently
  across supported NumPy versions.
* Numerical utility imports do not initialize the plotting stack; a fresh
  interpreter checks this separation with warnings treated as errors.

The documentation deployment now treats Sphinx warnings as failures. These
checks complement the independent algorithm references; they do not imply
that every finite floating-point input or every optional backend is supported.

Limits of the decision
--------------------------

CPU API stability is independent of experimental CuPy support. The three
smoothing formulations remain distinct, differential angular maps remain
2-D, and order-5 wavelets keep approximate source taps. Stability does not add
new boundaries, unsupported shapes, GPU guarantees, or universal speed claims.
Future extensions need their own contract and independent numerical checks.
