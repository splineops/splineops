Performance evidence
====================

SplineOps treats performance as reproducible evidence, not as an adjective.
The project is especially strong when a workload needs explicit spline
semantics, projection antialiasing, N-D execution, or reusable resize plans.
It does not claim to beat specialized image libraries on every ordinary 2-D
resize.

TensorSpline baseline
---------------------

``scripts/benchmark_tensorspline.py`` separates coefficient construction from
evaluation and records both runtime and ``tracemalloc`` peak allocations.  On
2026-07-15, the standard float64 profile produced the following medians on
Linux 6.17, Python 3.12.3, and NumPy 2.4.6:

.. list-table:: Standard TensorSpline profile
   :header-rows: 1
   :widths: 28 18 18 18

   * - Case
     - Construction
     - Evaluation
     - Evaluation peak
   * - 1-D cubic grid
     - 129.2 ms
     - 36.4 ms
     - 8.50 MiB
   * - 2-D cubic grid
     - 35.9 ms
     - 14.5 ms
     - 2.54 MiB
   * - 2-D cubic, 200,000 points
     - 36.6 ms
     - 258.5 ms
     - 26.03 MiB
   * - 3-D linear grid
     - 6.0 ms
     - 37.5 ms
     - 6.01 MiB
   * - 3-D cubic grid
     - 21.3 ms
     - 10.1 ms
     - 1.73 MiB

These are a development-machine baseline, not portable promises.  The useful
result is the bounded growth of point-query temporaries and the lower working
set for separable tensor grids.  Point evaluation is tiled at 65,536 samples;
unbatched grids contract one axis at a time instead of materializing the full
coordinate-product support arrays.

An explicit point-count sweep excludes the already-allocated input coordinates
from tracing and subtracts the required output buffer.  On the same machine,
the standard cubic 2-D sweep measured:

.. list-table:: TensorSpline point-query memory scaling
   :header-rows: 1
   :widths: 25 25 25

   * - Query points
     - Traced peak
     - Temporary overhead
   * - 10,000
     - 4.20 MiB
     - 4.13 MiB
   * - 100,000
     - 27.77 MiB
     - 27.01 MiB
   * - 1,000,000
     - 34.64 MiB
     - 27.01 MiB

The flat overhead between 100,000 and 1,000,000 points is evidence that the
working set is governed by the tile size rather than the full query.  The
result array itself must still scale with the requested output.

Repeated coordinates
~~~~~~~~~~~~~~~~~~~~

``TensorSpline.query_plan`` trades capped retained memory for repeated-query
speed.  It stores coordinate geometry independently of sample values and can
therefore serve compatible splines representing changing frames.  For 200,000
random cubic 2-D points and seven changing-spline evaluations, the ordinary
path took 249.0 ms per call and the planned path 62.1 ms: a 4.01x speedup.  Plan
construction broke even after an estimated 1.15 calls and retained 24.41 MiB.
This is a strong workload-specific capability, not a reason to plan one-shot
queries.

Reproduce it with:

.. code-block:: shell

   python scripts/benchmark_tensorspline.py --profile standard \
     --output-json tensorspline.json --output-csv tensorspline.csv
   python scripts/benchmark_tensorspline_memory_scaling.py \
     --output-json tensorspline-memory.json
   python scripts/benchmark_tensorspline_query_plan.py \
     --output-json tensorspline-plan.json

Affine comparison
-----------------

``scripts/benchmark_affine.py`` compares exactly matched pull-back geometry,
whole-sample mirror boundaries, prefiltering, output shape, and spline degree
against ``scipy.ndimage.affine_transform``.  The standard 2026-07-15 run found
maximum differences near ``1e-13``, but SciPy was materially faster:

.. list-table:: Equivalent affine rotation profile
   :header-rows: 1
   :widths: 24 19 19 19 19

   * - Case
     - SplineOps one-shot
     - Cached ``AffinePlan``
     - SciPy
     - SciPy vs. one-shot
   * - 2-D linear
     - 193.6 ms
     - 51.5 ms
     - 19.0 ms
     - 10.17x
   * - 2-D cubic
     - 644.0 ms
     - 215.4 ms
     - 33.9 ms
     - 19.02x
   * - 3-D linear
     - 123.7 ms
     - 38.1 ms
     - 23.4 ms
     - 5.28x
   * - 3-D cubic
     - 538.3 ms
     - 295.1 ms
     - 53.9 ms
     - 9.99x

Affine's current achievements are exact spline semantics and bounded
coordinate memory.  A cached plan removes repeated support construction and is
materially faster than the SplineOps one-shot path, but it still does not beat
SciPy's specialized kernels on these cases.  Maximum absolute differences were
between ``7e-14`` and ``6e-13``.

Reproduce the profile with:

.. code-block:: shell

   python scripts/benchmark_affine.py --profile standard \
     --output-json affine.json --output-csv affine.csv

Differentials vectorization
---------------------------

The spline differential implementation performs coefficient conversion and
derivative stencils across complete array axes.  Against the retained scalar
row/column oracle, the standard 512x640 float64 gradient-magnitude benchmark
measured 220.9 ms cold versus 12.483 s, a 56.50x speedup.  Reusing the
``Differentials`` object's cached workspace reduced a repeated map to 5.9 ms.
A ``DifferentialPlan`` request for gradient, packed Hessian, and Laplacian
together took 358.0 ms; that row performs substantially more work and is
reported to make multi-output cost visible, not as a direct speedup ratio.
Maximum absolute difference from the scalar oracle was ``1.556e-7``.

.. code-block:: shell

   python scripts/benchmark_differentials.py --profile standard \
     --output-json differentials.json --output-csv differentials.csv

Reusable research-module plans
------------------------------

``SmoothingSplinePlan`` retains a real-FFT half-spectrum response for changing
arrays with fixed shape, regularization, and order.  ``DenoisingPlan`` retains
the sparse factorization determined by fixed sample locations and ADMM penalty
while allowing observations and regularization strength to change.

``scripts/benchmark_workflows.py`` measures complete operations, including
prefiltering and axis orchestration.  The standard three-repeat medians on the
same development machine were:

.. list-table:: Consolidated workflow profile
   :header-rows: 1
   :widths: 42 18 18 18

   * - Workflow
     - Consolidated path
     - Explicit reference
     - Speedup
   * - One coefficient field, two affine geometries
     - 104.6 ms
     - 123.3 ms
     - 1.18x
   * - Smoothing with explicit batch/channel axes
     - 22.1 ms
     - 22.8 ms
     - 1.03x
   * - Warm-start denoising lambda path
     - 1.120 s
     - 1.132 s
     - 1.01x
   * - Wavelet explicit-axis orchestration
     - 16.0 ms
     - 15.8 ms
     - 0.99x

All compared outputs agreed exactly except the independently converged ADMM
paths, whose maximum difference was ``5.6e-9``.  The mixed result is useful:
coefficient sharing earns a modest end-to-end affine gain, while denoising warm
starts and wavelet axis convenience are not presented as speed advantages on
this workload.  Nearby lambda paths can need fewer iterations, but the actual
diagnostics—not the API name—decide whether that helps.

.. code-block:: shell

   python scripts/benchmark_workflows.py --profile standard \
     --output-json workflows.json --output-csv workflows.csv
   python scripts/check_benchmark_thresholds.py \
     --policy benchmarks/consolidation-thresholds.json \
     --artifacts-dir .

Multiscale vectorization
------------------------

``scripts/benchmark_multiscale.py`` compares current whole-axis execution with
the retained row/column oracle on standard float64 images.  Pyramid reduction
measured 533.2 ms versus 3.558 s (6.67x), and Haar analysis/synthesis measured
29.1 ms versus 72.7 ms (2.49x).  These ratios measure Python dispatch and array
execution on one development machine; reconstruction tolerances and supported
shape contracts are unchanged.

.. code-block:: shell

   python scripts/benchmark_multiscale.py --profile standard \
     --output-json multiscale.json --output-csv multiscale.csv

Resize v2 scheduler profile
---------------------------

The native resize scheduler was re-profiled after the v2 direct cross-Gram
rewrite rather than carrying forward pre-v2 tuning assumptions.  A proposed
serial path for small 3-D work was rejected: one thread was typically three to
five times slower in the focused sweep.  The retained policy keeps parallel
execution but caps automatic participation at eight workers for 3-D workloads
of at most one million elements.  An explicit ``LSRESIZE_NUM_THREADS`` value
continues to override automatic selection.

In an 11-repeat focused profile, automatic/default versus explicit-eight
medians were 0.62/0.64 ms for linear downsampling, 3.51/3.12 ms for cubic
downsampling, 1.51/1.67 ms for anisotropic cubic work, 2.43/2.21 ms for linear
antialiasing, and 2.60/2.32 ms for cubic antialiasing.  The policy tracks the
measured eight-worker operating point broadly; individual rows can still move
in either direction, and it is not a claim that eight threads is ideal on every
machine.

The broader library comparison remains contextual.  SciPy, OpenCV, PyTorch,
and SplineOps differ in coordinates, boundaries, antialiasing, and output-size
rules.  SplineOps therefore reports both numerical differences and timings
rather than presenting those libraries as one universal leaderboard.  See
:doc:`project-status` for the smoke-comparison command.

Historical optimization reports
--------------------------------

The repository contains ``scripts/resize_optimization_notes.md``,
``scripts/resize_optimization_progress.md``, and
``scripts/resize_benchmark_report_20260616.md``.  They are useful engineering
history, but parts describe the finite-difference projection implementation
replaced by the 2.0 direct cross-Gram path.  They must not be quoted as current
v2 benchmark results.  New resize reports should be generated from the
versioned scripts and should state wins, losses, numerical differences,
hardware, thread settings, and semantic mismatches.
