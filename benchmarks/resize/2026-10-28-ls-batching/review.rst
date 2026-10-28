SplineOps competitiveness review
===============================

28 October 2026

This is the initial review and prototype snapshot. The completed LS and affine
changes, final measurements, and validation are recorded in
``benchmarks/performance/2026-10-28/report.rst``. The statements about pending
work and test counts below describe the earlier review stage.

SplineOps has worthwhile remaining potential in scientific volume resampling.
The strongest immediate finding is an execution-policy gap in cubic
least-squares projection: the existing batched native kernel is available,
but automatic 3-D dispatch only selects it for interpolation and oblique
presets. A small local patch makes the stable cubic LS path use that kernel
when enough neighboring lines are available.

Scope and reproducibility
-------------------------

The reviewed local checkout and remote HEAD were both
``a148a9ecc9daaccc74939dbb52824fbe66294f5b`` (SplineOps 2.2.0).
The candidate is an uncommitted local change on ``codex/ls-3d-batching``.
The implementation change is nine lines in ``cpp/lsresize/src/resize_nd.cpp``;
it preserves the existing fewer-than-128-lines guard and only expands automatic
routing for direct cubic equal-degree projection. No public method or numerical
formula is changed.

Fresh measurements used Windows, Intel Core Ultra 9 275HX, Python 3.12.14,
NumPy 2.5.4, and SciPy 1.18.1. The native extension was built from source.
The dedicated timing experiment uses reusable plans, three warmups, eleven
repeats, and randomized scalar/automatic call order. Plan construction and
cross-device transfers are outside its scope. The scalar control disables
batching; this uses the same per-line kernel as the parent revision's automatic
3-D cubic LS routing, although dispatch overhead can differ.

Reproduce the candidate experiment with::

    python scripts/benchmark_resize_ls_batching.py --output-json results.json

Raw timings, all 24 cases, and environment metadata are in ``windows.json``
beside this review. Tiny volumes are retained as controls and must not be
excluded when inspecting the full result.

Measured improvement
--------------------

Float32, one thread, milliseconds per warmed call:

============================= =========== =========== =======
Geometry                      Scalar      Automatic   Speedup
============================= =========== =========== =======
64x96x128 to 32x48x64          20.427      3.771       5.42x
48x256x256 to 48x32x32         29.413      5.851       5.03x
48x56x64 to 18x34x24           3.763       0.888       4.24x
24x32x40 to 36x48x60           5.056       0.986       5.13x
96x128x128 to 48x64x64         40.140      7.604       5.28x
============================= =========== =========== =======

Across the corresponding float64 cases, speedups were 4.15x to 4.92x.
With automatic threading, the three larger geometries improved by 1.58x to
1.80x across the two dtypes. The smaller anisotropic case was approximately
flat (0.98x to 1.01x); the upsampling case improved by 1.08x to 1.29x.
The tiny 8x9x10 control uses the scalar kernel in both modes and has no claimed
benefit: automatic/scalar-control speed ratios were 0.82x to 0.90x. Forcing
batching on that tiny case in the preceding unmodified-build experiment was
slower, which is why the guard remains.

All 24 timed comparisons produced exactly equal output arrays. This is a
measured property of these cases, not a bitwise compatibility guarantee for
all inputs and platforms.

Validation completed with 710 passed and four platform-specific skips across
the stable-projection, native/Python contract, plan-cache, concurrency, resize,
and native-resize tests. Four new test cases compare 3-D results with tensor
products of the existing 100-digit inner-product fixtures, covering float32,
float64, full resizing, and unchanged axes. Formatting and diff checks passed.
Linux and macOS performance still need independent validation before release.

What the least-squares method can offer
--------------------------------------

Arrate Munoz Barrutia, Thierry Blu, and Michael Unser's method remains a useful
mathematical model. Its optimum concerns continuous reconstruction error in
the chosen spline space. That does not imply an optimum for every image
appearance, spectral cutoff, segmentation endpoint, or physical measurement.

The current code already replaces the unstable high-order cumulative
integration construction with direct compact cross-Gram rows for public
zero-shift projections with analysis degree at least one. Historical June notes
about catastrophic long-line cubic LS errors therefore do not describe the
current release. Fresh ramp checks at lengths 4096, 16384, and 65536 remained
well behaved. Reimplementing that stabilization would duplicate existing work.

The repository's standard projection comparison was rerun on the unmodified
checkout, with seven timing repeats, two warmups, and one thread. Cubic oblique
was only 1.14x faster at the median across 36 cases; equal-degree cubic LS won
30 of 36 down-up reconstruction PSNR comparisons. SSIM favored oblique in
19 of 36, so the quality ordering depends on the metric. Linear LS was faster
than linear oblique in all 36 cases. These results justify evaluating a named
LS quality option; they do not justify silently replacing existing presets.

The 3-D speed gap was much larger before the patch because of dispatch, rather
than solely the mathematical cost of LS. Once batching is enabled, LS still
has wider cross-Gram rows and a higher-order output solve than cubic oblique.
It will not become an identical-cost substitute.

Competitive quality needs a stronger target
-----------------------------------------

A small independent spectral probe used nine mirror-compatible cosine mixtures
on 2049 samples reduced to 513, with an analytical target retaining only the
in-band component. Default SciPy polyphase FIR beat cubic LS in eight of nine
cases. For one representative mixture, normalized RMS error was 0.0312 for
cubic LS, 0.1446 for cubic oblique, and 0.00101 for FIR. This is exploratory,
not a dataset-wide quality claim or a timing comparison.

That finding agrees with the existing repository's controlled-field study,
where its post-hoc FIR baseline was substantially more accurate. Future
quality benchmarks should include FIR/Lanczos baselines with explicit grids,
boundary extension, and point-sample versus cell-average targets. Round-trip
PSNR alone favors the assumed reconstruction model and cannot settle the
choice for a scientific workflow.

Other modules
-------------

Affine has a clear implementation opportunity. Fresh matched cubic-transform
measurements, with whole-sample mirror boundaries, one thread, randomized
method order, and no allocation tracing, found:

* 512x384: SplineOps one-shot 144.55 ms; SciPy 11.06 ms (13.07x).
* 48x56x64: SplineOps one-shot 215.39 ms; SciPy 23.80 ms (9.05x).

Maximum absolute differences were below 7e-13. Even when both paths reused the
same SplineOps-prepared coefficients, SciPy evaluation was 2.01x and 2.87x
faster. An explicit SciPy-backed fast path for supported degrees and matching
contracts is a practical candidate; higher degrees, batching, output-buffer
behavior, and other modes must retain their appropriate implementation.

Adaptive regression does not have an equally clear easy win. A prototype
replaced generic sparse LU with banded Cholesky for its pentadiagonal SPD
system. Factor construction improved substantially, but full ADMM solves were
slightly slower at both 1000 and 10000 samples, with the same iteration counts
and agreement near 3e-15. This substitution should not be adopted from the
factorization timing alone. Whole-solver convergence and regularization paths
are more useful targets.

Smoothing, differentials, and multiscale already contain reusable plans and
vectorized/batched work. Their next competitiveness claims need an identified
application and an independent reference; this review establishes no new
speedup for those modules.

Recommended sequence
--------------------

1. Validate and review the narrow 3-D cubic LS routing patch across platforms,
   representative medical/microscopy volume layouts, and thread settings.
2. Evaluate a first-class LS quality preset using independent downstream
   metrics, and refresh historical documentation that treats LS instability
   or slowness as universal. Keep oblique available as a cheaper option.
3. Add an affine fast path using already-available compiled primitives when
   semantics match. Measure complete operations as well as coefficient reuse.
4. Investigate a device-resident, differentiable 3-D projection operator with
   an actual adjoint. Reverse resizing is generally not that adjoint; mirror
   boundaries, prefilters, and the output solve must all be transposed correctly.
   Validate dot-product identities and finite-difference gradients before any
   acceleration claim. This is proposed work, not a delivered capability.
5. Make coordinate conventions, voxel spacing/origin/direction, and
   point-sample versus voxel-average meaning explicit for imaging workflows.
   Add other conventions only with independent boundary and geometry tests.

Source references
-----------------

* `Original least-squares resizing paper
  <https://bigwww.epfl.ch/publications/munoz0101.html>`_
* `Original oblique-projection paper
  <https://bigwww.epfl.ch/publications/lee9801.pdf>`_
* `Current SplineOps resize guide
  <https://splineops.github.io/user-guide/02_resize.html>`_
* `Existing controlled-field study
  <https://splineops.github.io/wavefield3d-study.html>`_
* `SciPy polyphase resampling contract
  <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html>`_

The fresh exploratory scripts and original-checkout outputs are retained in the
parent review directory as ``review_experiments.py``, ``experiments.json``,
``projection.json``, ``projection.csv``, ``ls_batch_probe.py``, and
``ls_batch_probe.json``. They record broader assessment experiments separately
from the candidate's dedicated regression and performance evidence.
