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
     - Continue cross-platform performance monitoring; validate the measured
       small-3-D scheduler policy on additional machines before generalizing
       it to other workload classes.
   * - ``TensorSpline``
     - Stabilizing
     - Multiple bases and per-axis modes, N-D and batched evaluation, real and
       complex data, tested singleton/short-periodic behavior, bounded-memory
       queries, separable grid contraction, reusable geometry plans across
       compatible sample arrays, explicit refitting/precomputed-coefficient
       paths, SciPy parity for B-spline degrees 0--5 through four dimensions,
       and partial CuPy interoperability.
     - Complete dedicated CuPy CI before making a stable backend-wide promise;
       continue independent references for non-B-spline bases and modes.
   * - Affine
     - Experimental
     - General 2-D/3-D matrix transforms, analytical rotation tests, explicit
       batch/channel axes, output buffers, bounded one-shot execution, cached
       fixed-geometry plans, vectorized batch coefficient evaluation, immutable
       compatibility-tagged coefficients reusable across affine geometries,
       safe validated coefficient persistence, concurrent field reuse, and
       equivalent SciPy parity for degrees 0--5; SplineOps also accepts its
       higher-order degrees 6 and 7.
     - Improve performance only with profile-backed changes.  SciPy remains
       8.96x--19.54x faster than the SplineOps one-shot path in the current
       standard matched benchmark, despite useful gains from cached geometry.
   * - Differentials
     - Experimental
     - Raw repeatable outputs, preserved source arrays, vectorized coefficient
       filtering, physical spacing, 2-D/3-D gradient and packed Hessian
       components, explicit batch/channel axes through one batched multi-output
       workspace, independent output-family selection, exact structured output
       buffers, legacy-reference coverage, and polynomial and trigonometric
       invariants.
     - Define additional boundary/dtype contracts before describing the module
       as a general N-D differential engine; angular maps intentionally remain
       2-D and the legacy object remains scalar.
   * - Smoothing splines
     - Experimental
     - Fractional FFT and recursive examples, reusable real-FFT half-spectrum
       plans with explicit batch/channel axes, real/finite parameter domains,
       constant preservation, periodic cosine-response checks, and an
       independent dense-system reference for the recursive formulation.
     - Add broader published numerical fixtures while preserving the clear
       distinction between exact fractional, radial approximate, and recursive
       formulations.
   * - Adaptive regression
     - Experimental
     - Deterministic denoising and piecewise-linear reconstruction tests,
       reusable fixed-sample sparse factorizations, prefix-sum spline
       evaluation, stateless warm-start lambda paths, non-mutating amplitude
       sparsification, sorted-input validation, opt-in ADMM convergence
       diagnostics, and a constrained-optimizer reference.
     - Add larger optimization-reference comparisons and systematic penalty
       and ADMM-parameter guidance.
   * - Multiscale
     - Experimental
     - Explicit odd/singleton pyramid behavior, vectorized whole-axis pyramid,
       adaptive small-plane vectorization and large-plane wavelet dispatch with
       explicit batch/channel axes, perfect
       reconstruction for supported even rectangular Haar and cubic
       spline-wavelet shapes, and a reconstruction-error audit for spline
       orders 1, 3, and 5.
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

After the second optimization pass, a recreated Python 3.12 tox environment
built the native extension from this worktree and completed ``1167 passed`` in
138.32 seconds.  The strict documentation build, Black check, scoped MyPy
check, standalone native scheduler test, benchmark smoke tests, and package
build are the accompanying branch-level gates.  This is pre-release validation,
not a claim that a new distribution has been published.

The development benchmarks from this pass are recorded in
:doc:`performance`.  Their central conclusions are intentionally mixed:
``TensorSpline`` has bounded query overhead and a useful reusable-geometry
plan, vectorized differentials substantially outperform their scalar oracle,
and affine transforms match SciPy numerically while remaining slower.  Cached
affine geometry and vectorized multiscale passes are useful workload-specific
gains.  These are development-machine measurements, not release promises.

After the consolidation pass, the recreated Python 3.12 environment completed
``1179 passed`` in 145.02 seconds.  The focused 453-test contract set, Black,
scoped MyPy, strict Sphinx, source and wheel builds, and the stored benchmark
regression policy also passed.  The standard workflow profile found a 4.01x
repeated-coordinate ``TensorSpline`` gain, a 1.18x affine coefficient-sharing
gain, and approximately neutral smoothing, denoising-path, and explicit-axis
wavelet timings.  Exact operations agreed exactly; the independently
converged denoising paths differed by at most ``5.6e-9``.  No distribution was
released from this validation pass.

After the fourth batch-execution pass, the full suite completed ``1180
passed`` in 140.13 seconds.  Black, scoped MyPy, a warning-fatal Sphinx build,
and source/wheel builds also passed.  The standard explicit-axis workflow rows
measured 1.03x affine and 1.26x differential speedups with exact output parity;
the four-plane Haar round trip measured 0.90x and is documented as a bounded
regression rather than a win.  The smoke Haar workload measured 1.73x.  This
pass also repairs the macOS FFT test's overly strict bitwise comparison and
improves CI failure annotations.  It remains intentionally unreleased.

After the fifth hardening and profiling pass, the full suite completed ``1189
passed`` in 140.51 seconds.  Black, scoped MyPy, a warning-fatal Sphinx build,
the stored smoke benchmark policy, source and native-wheel builds, and a clean
wheel import smoke test also passed.  The standard batch-memory sweep agreed
with scalar dispatch for every measured affine, Laplacian-only, and Haar row;
its worst normalized traced-memory growth was 1.08.  Direct NumPy support
contraction improved the measured affine workloads, while adaptive wavelet
dispatch restored the representative four-plane Haar round trip to
approximately neutral performance.  The subsequent `cross-platform smoke run
29424000663 <https://github.com/splineops/splineops/actions/runs/29424000663>`_
passed on Linux, macOS, and Windows and uploaded one complete benchmark artifact
per runner.  The nine-job Python/OS library matrix and quality workflow also
passed for commit ``32b9d8c``.  This remains pre-release validation and no
distribution was published.

Stability-soak and graduation review
------------------------------------

The added capabilities do not automatically promote a module.  ``TensorSpline``
remains stabilizing until dedicated CuPy CI and broader independent references
cover the advertised backend and non-B-spline surface.  Affine and
differentials are the closest experimental modules to a future stability
review, but this review deliberately does not promote them.  Their new
coefficient-persistence, output-selection, and output-buffer contracts need a
real API-soak period.  Linux/macOS/Windows smoke artifacts have now been
collected and reviewed successfully; a manually requested standard profile is
still required before making broader portable performance claims.
Smoothing, adaptive regression, and multiscale remain experimental while their
published-reference, parameter-guidance, and reconstruction gates are open.

This deliberately separates implementation quality from compatibility
promises.  Users can rely on the documented functionality today without the
project pretending that one Linux validation run establishes a stable public
contract on every platform.

Promotion from experimental requires all of the following evidence:

* successful test and development-benchmark artifacts on Linux, macOS, and
  Windows, with numerical equivalence checked before timing ratios;
* at least two representative downstream workloads exercising the new APIs
  without contract changes or unresolved correctness reports;
* reviewed batch-memory scaling and concurrency behavior at the documented
  sizes; and
* migration guidance for any naming or buffer-contract change discovered
  during the soak.

Until those gates close, changes may still refine these experimental APIs and
no release is implied by implementation completeness.

Reproducing validation
----------------------

.. code-block:: shell

   python -m pytest -q
   python -m sphinx -b html docs /tmp/splineops-docs -W --keep-going \
     -D sphinx_gallery_conf.plot_gallery=0
   python -m build
   python scripts/benchmark_tensorspline_query_plan.py \
     --output-json /tmp/splineops-bench/tensorspline-query-plan.json
   python scripts/benchmark_affine.py --profile standard \
     --output-json /tmp/splineops-bench/affine.json
   python scripts/benchmark_differentials.py --profile standard \
     --output-json /tmp/splineops-bench/differentials.json
   python scripts/benchmark_multiscale.py --profile standard \
     --output-json /tmp/splineops-bench/multiscale.json
   python scripts/benchmark_workflows.py --profile standard \
     --output-json /tmp/splineops-bench/workflows.json
   python scripts/benchmark_batch_scaling.py --profile standard \
     --output-json /tmp/splineops-bench/batch-scaling.json
   python scripts/check_benchmark_thresholds.py \
     --policy benchmarks/consolidation-thresholds.json \
     --artifacts-dir /tmp/splineops-bench
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
