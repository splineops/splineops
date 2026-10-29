API stability and support
=============================

SplineOps 2.3 provides stable CPU APIs for all seven numerical modules,
including ``TensorSpline`` and its reusable geometry plans. Stability is a
compatibility commitment for the documented shapes, precisions, boundaries,
and operators below. It does not imply that different spline methods are
interchangeable or that a benchmark speedup applies to every machine.

Supported numerical APIs
----------------------------

.. list-table:: Stable CPU contracts
   :header-rows: 1
   :widths: 20 45 35

   * - Module
     - Supported contract
     - Numerical validation
   * - :doc:`Resize <user-guide/02_resize>`
     - N-D interpolation and projection antialiasing; explicit axes, output
       geometry, buffers, reusable plans, and native/Python execution.
     - Independent projection quadrature, native/reference parity, cache and
       concurrency tests, and installed-wheel checks.
   * - :doc:`TensorSpline <user-guide/01_spline_interpolation>`
     - Uniform N-D construction grids; arbitrary point and tensor-grid
       queries; all named bases and zero, mirror, and periodic modes;
       float32/float64 and complex64/complex128; reusable geometry and refitting.
     - SciPy references, independent B-spline/derivative definitions for every
       O-MOMS degree, dense cardinal solves for all boundaries and short
       signals, mixed-axis complex references, and threaded geometry reuse.
   * - :doc:`Affine <user-guide/03_affine>`
     - 2-D/3-D pull-back transforms, degrees 0--7, float32/float64, explicit
       batch/channel axes, output buffers, reusable coefficients and archives.
       SciPy and NumPy evaluators share the coefficient contract.
     - Analytical geometry, matched SciPy/reference comparisons, immutable
       archive fixtures, corruption checks, and repeated process/thread reuse.
   * - :doc:`Differentials <user-guide/06_differentials>`
     - Cubic whole-sample mirror derivatives in 2-D/3-D, physical spacing,
       selected gradient/Hessian/Laplacian families, explicit axes and exact
       structured buffers. Legacy angular maps remain 2-D.
     - Polynomial and trigonometric invariants, independent clamped cubic
       piecewise polynomials, scalar/batched parity, and repeated buffer reuse.
   * - :doc:`Smoothing <user-guide/05_smoothing_splines>`
     - Finite real inputs; periodic 1-D fractional interpolation, periodic
       N-D radial smoothing, and first-order recursive smoothing with
       steady-state ends. Each formulation keeps its own operator.
     - Dense DFT and linear-system references, analytical frequency responses,
       constant preservation, periodic autocorrelation, and bounded batch FFT
       workspace.
   * - :doc:`Adaptive regression <user-guide/04_adaptive_regression_splines>`
     - Finite real, strictly increasing 1-D sample locations; TV denoising,
       reusable factorizations, lambda paths, diagnostics, and sparse linear
       interpolation. Check convergence before using an iterative fit.
     - Independent primal and dual optimizers, known sparse hinge models,
       coordinate-translation invariance, and integer-promotion regressions.
   * - :doc:`Multiscale <user-guide/07_multiscale>`
     - Mirror pyramids with explicit odd/singleton behavior; 2-D Haar and
       spline wavelets with explicit batch axes and lengths divisible by
       ``2**scales``. Order-5 taps retain their documented approximation.
     - Independent Haar coefficients and energy, randomized reconstruction,
       legacy numerical fixtures, and scalar/batch comparisons.

Use float32 or float64 for the supported real-valued numerical paths.
``TensorSpline`` also supports the matching complex precisions; its
construction coordinates must have the corresponding real precision.
Integer promotion and result precision are specified in each module guide.
Half and extended precision are not covered by the stability promise.

Compatibility policy
------------------------

* Public imports, argument meanings, shapes, dtype rules, boundary conventions,
  and documented return types are maintained within the 2.x series.
* Bug fixes may correct numerical results or reject invalid inputs. Such
  changes are described in the changelog with regression tests.
* An intentional incompatible API change needs a deprecation path and
  migration guidance. Internal names beginning with an underscore are private.
* Cached plans describe fixed operators. Construct a new affine, smoothing,
  or denoising plan when its geometry or fixed parameters change. Reassigning
  settings cannot silently leave old caches active.
* Existing affine schema-1 and schema-2 coefficient archives remain readable.
  Archive fixtures are retained when a future format is introduced.
* ``TensorSplineQueryPlan`` remains an alias of ``TensorSplineGeometryPlan``.
  ``Differentials`` remains available alongside ``DifferentialPlan``; neither
  legacy name is deprecated in 2.3.
* Wavelet coefficient arrays preserve axis order and dtype; batched results
  may use non-contiguous views over contiguous spatial planes. C-contiguity
  is not part of the coefficient-array contract.

Scope that remains experimental
-----------------------------------

CuPy interoperability is limited to parts of ``TensorSpline`` and lacks a
dedicated GPU CI runner. It remains experimental independently of the stable
NumPy implementation. Custom basis/mode extensions are the caller's numerical
responsibility. See :doc:`backend-support` for details.

The order-5 spline wavelet is a stable API for the existing approximate tap
table, not a perfect-reconstruction guarantee. Reported reconstruction error
depends on signal magnitude, shape, and scale count; the fixture bounds in the
guide are measured cases rather than universal absolute bounds.

How stability is checked
----------------------------

The test matrix covers Python 3.11--3.13 on Linux, macOS, and Windows, plus the
declared minimum NumPy/SciPy versions. Release validation also installs the
12 platform wheels, runs reference comparisons, repeats public-API workflows,
and builds the documentation. Performance checks compare numerical outputs
before considering timings. See :doc:`graduation-audits` for the 2.3 review and
:doc:`releasing` for the exact release gates.
