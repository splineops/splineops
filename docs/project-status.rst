Project status and evidence
===========================

SplineOps publishes maturity labels so users can choose APIs with appropriate
expectations.  This page records what has been verified, what remains under
active stabilization, and how the evidence can be reproduced.

Status definitions
------------------

Stable
   The public contract is documented, edge cases and accelerated/reference
   parity are tested, and compatibility changes require migration guidance.

Stabilizing
   The central use case is sound and useful, but some combinations of shapes,
   dtypes, boundaries, backends, or memory behavior still need a completed
   contract.

Experimental
   The implementation is available for research and evaluation, but its API,
   numerical contract, provenance review, or coverage is not yet sufficient for
   a stability promise.

Current capability matrix
-------------------------

.. list-table:: Capability evidence
   :header-rows: 1
   :widths: 18 14 34 34

   * - Module
     - Status
     - Demonstrated capability
     - Remaining graduation work
   * - Resize
     - Stable
     - N-D native and Python paths, explicit axes and output geometry,
       interpolation and direct projection, reusable plans, bounded caches,
       output buffers, concurrency and fork handling.
     - Continue cross-platform performance monitoring and re-profile the v2
       direct projection pipeline before new kernel work.
   * - ``TensorSpline``
     - Stabilizing
     - Multiple bases and per-axis modes, N-D and batched evaluation, real and
       complex data, tested singleton/short-periodic behavior, bounded-memory
       queries, reusable fixed-coordinate plans, SciPy parity for B-spline
       degrees 0--5 through four dimensions, and partial CuPy interoperability.
     - Complete dedicated CuPy CI before making a stable backend-wide promise;
       continue independent references for non-B-spline bases and modes.
   * - Affine
     - Experimental
     - Analytical 2-D and 3-D rotation tests, explicit dtype/degree validation,
       bounded tiled coordinate evaluation, and equivalent SciPy parity for
       degrees 0--5; SplineOps also accepts its higher-order degrees 6 and 7.
     - Improve performance only with profile-backed changes and decide whether
       a general matrix API has a sufficiently clear contract.  SciPy is
       currently 7.46x--15.42x faster in the standard matched benchmark.
   * - Differentials
     - Experimental
     - Raw repeatable outputs, preserved source arrays, batched coefficient
       filtering, physical spacing, direct gradient/Hessian components,
       legacy-reference coverage, and polynomial and trigonometric invariants.
     - Define additional boundary/dtype contracts and add 3-D support before
       describing the module as a general N-D differential engine.
   * - Smoothing splines
     - Experimental
     - Fractional FFT and recursive examples, explicit real/finite parameter
       domains, constant preservation, periodic cosine-response checks, and an
       independent dense-system reference for the recursive formulation.
     - Add broader published numerical fixtures while preserving the clear
       distinction between exact fractional, radial approximate, and recursive
       formulations.
   * - Adaptive regression
     - Experimental
     - Deterministic denoising and piecewise-linear reconstruction tests,
       non-mutating amplitude sparsification, sorted-input validation, and
       opt-in ADMM convergence diagnostics.
     - Add larger optimization-reference comparisons and systematic penalty
       parameter guidance.
   * - Multiscale
     - Experimental
     - Explicit odd/singleton pyramid behavior and perfect reconstruction for
       supported even rectangular Haar and cubic spline-wavelet shapes, plus a
       reconstruction-error audit for spline orders 1, 3, and 5.
     - Obtain higher-precision order-5 taps or continue labeling it approximate;
       expand supported shape and scale classes only with reversible evidence.

Initial execution baseline
--------------------------

The roadmap execution began from commit ``a0bb350`` on 2026-07-15.  On the
initial Linux development environment (Python 3.12.3, NumPy 2.4.6), the full
suite completed with ``953 passed`` in 109.74 seconds.  A strict documentation
build also succeeded after the roadmap update.

The initial resize smoke comparison used:

.. code-block:: shell

   python scripts/benchmark_resize_pr.py \
     --profile smoke \
     --output-dir /tmp/splineops-analysis-smoke

On that environment, SplineOps was faster than SciPy on the three tested rows
with close numerical agreement on the interpolation rows.  OpenCV was faster
on all three generic 2-D rows, and PyTorch was faster on two.  These libraries
do not share one resize contract, so the result supports a focused claim—fast,
explicit spline semantics—not a claim of universal image-resize leadership.

Current execution result
------------------------

After the first roadmap implementation pass, the same Linux development
environment completed ``1134 passed`` in 125.37 seconds.  Black formatting,
the scoped MyPy check, a strict Sphinx build, wheel and source-distribution
builds, and a clean-environment wheel smoke test all passed.  The wheel smoke
also exercised the native resize extension and an experimental repeated-query
``TensorSpline`` plan.

The development benchmarks from this pass are recorded in
:doc:`performance`.  Their central conclusions are intentionally mixed:
``TensorSpline`` has bounded query overhead and a useful repeated-coordinate
plan, vectorized 2-D differentials substantially outperform their scalar
oracle, and affine rotation matches SciPy numerically while remaining much
slower.  These are development-machine measurements, not release promises.

Reproducing validation
----------------------

.. code-block:: shell

   python -m pytest -q
   python -m sphinx -b html docs /tmp/splineops-docs -W --keep-going \
     -D sphinx_gallery_conf.plot_gallery=0
   python -m build
   python scripts/benchmark_resize_pr.py --profile smoke \
     --output-dir /tmp/splineops-smoke

Performance reports must record the environment, threads, runtime, output
shape and dtype, numerical differences, and whether another library implements
equivalent coordinates, boundaries, interpolation, and antialiasing.

Backend support
---------------

NumPy
   Supported throughout the package on the Python versions tested in CI.

Native C++ resize
   Required and exercised in the main test matrix and published wheels.

CuPy
   Experimental and limited to portions of spline interpolation.  It is not a
   package-wide GPU backend and will not be described as supported until
   dedicated GPU CI covers the advertised combinations.

The exact graduation gates and execution order are maintained in the
:doc:`roadmap`.
