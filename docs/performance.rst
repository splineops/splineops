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
Linux 6.17, Python 3.12.3, and NumPy 2.5.1:

.. list-table:: Standard TensorSpline profile
   :header-rows: 1
   :widths: 28 18 18 18

   * - Case
     - Construction
     - Evaluation
     - Evaluation peak
   * - 1-D cubic grid
     - 132.7 ms
     - 35.7 ms
     - 8.50 MiB
   * - 2-D cubic grid
     - 38.4 ms
     - 15.9 ms
     - 2.54 MiB
   * - 2-D cubic, 200,000 points
     - 39.6 ms
     - 276.9 ms
     - 22.03 MiB
   * - 3-D linear grid
     - 5.8 ms
     - 37.9 ms
     - 6.01 MiB
   * - 3-D cubic grid
     - 22.0 ms
     - 11.1 ms
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
     - 3.21 MiB
     - 3.13 MiB
   * - 100,000
     - 21.27 MiB
     - 20.50 MiB
   * - 1,000,000
     - 28.14 MiB
     - 20.51 MiB

The flat overhead between 100,000 and 1,000,000 points is evidence that the
working set is governed by the tile size rather than the full query.  The
result array itself must still scale with the requested output.

Repeated coordinates
~~~~~~~~~~~~~~~~~~~~

``TensorSpline.query_plan`` trades capped retained memory for repeated-query
speed.  It stores coordinate geometry independently of sample values and can
therefore serve compatible splines representing changing frames.  For 200,000
random cubic 2-D points and seven changing-spline evaluations, the ordinary
path took 187.1 ms per call and the planned path 30.0 ms: a 6.23x speedup.  Plan
construction broke even after an estimated 1.62 calls and retained 24.41 MiB.
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
     - 170.0 ms
     - 18.0 ms
     - 8.5 ms
     - 20.02x
   * - 2-D cubic
     - 569.4 ms
     - 192.4 ms
     - 21.3 ms
     - 26.69x
   * - 3-D linear
     - 123.8 ms
     - 21.1 ms
     - 12.2 ms
     - 10.17x
   * - 3-D cubic
     - 456.7 ms
     - 98.3 ms
     - 54.3 ms
     - 8.41x

Affine's current achievements are exact spline semantics and bounded
coordinate memory.  Direct NumPy support contraction materially improves the
cached plan, especially in 3-D, but it still does not beat SciPy's specialized
kernels on these cases.  Maximum absolute differences were between ``7e-14``
and ``6e-13``.  CuPy deliberately retains the previously tested contraction
until dedicated GPU evidence is available.

Reproduce the profile with:

.. code-block:: shell

   python scripts/benchmark_affine.py --profile standard \
     --output-json affine.json --output-csv affine.csv

Affine phase profile
--------------------

``scripts/profile_affine_phases.py`` instruments prefilter and coefficient
evaluation inside complete cached-plan calls and reports construction
separately.  On the current standard run, evaluation accounted for 63%--91% of
the measured in-call phase time across 2-D/3-D linear/cubic rows.  Cached-plan
construction ranged from 57 ms to 326 ms and is therefore justified only by
reuse.  The profile motivated sequential support-axis contraction for NumPy
3-D, where the gathered cubic support block is largest.  NumPy 1-D/2-D and
CuPy retain the prior direct contraction because the evidence did not justify
broadening the change.

.. code-block:: shell

   python scripts/profile_affine_phases.py --profile standard \
     --output-json affine-phases.json --output-csv affine-phases.csv

Differentials vectorization
---------------------------

The spline differential implementation performs coefficient conversion and
derivative stencils across complete array axes.  Against the retained scalar
row/column oracle, the standard 512x640 float64 gradient-magnitude benchmark
measured 210.0 ms cold versus 12.582 s, a 59.92x speedup.  Reusing the
``Differentials`` object's cached workspace reduced a repeated map to 4.0 ms.
A ``DifferentialPlan`` request for gradient, packed Hessian, and Laplacian
together took 348.1 ms with a 22.62 MiB traced peak.  Gradient-only execution
took 211.9 ms; Laplacian-only execution took 214.3 ms and used 66.8% of the
full request's traced peak because it omitted gradient and mixed-Hessian
outputs.  Maximum absolute difference from the scalar oracle was ``1.556e-7``.

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
prefiltering and axis orchestration.  After the batch-execution pass, the
standard five-repeat medians on the same development machine were:

.. list-table:: Consolidated workflow profile
   :header-rows: 1
   :widths: 42 18 18 18

   * - Workflow
     - Consolidated path
     - Explicit reference
     - Speedup
   * - One coefficient field, two affine geometries
     - 44.7 ms
     - 70.0 ms
     - 1.57x
   * - Affine with explicit batch/channel axes
     - 78.2 ms
     - 151.2 ms
     - 1.93x
   * - Smoothing with explicit batch/channel axes
     - 22.3 ms
     - 21.0 ms
     - 0.94x
   * - Multi-output differentials with explicit axes
     - 111.1 ms
     - 159.2 ms
     - 1.43x
   * - Warm-start denoising lambda path
     - 1.148 s
     - 1.155 s
     - 1.01x
   * - Haar explicit-axis analysis/synthesis
     - 30.9 ms
     - 31.0 ms
     - 1.00x

All compared outputs agreed exactly except the independently converged ADMM
paths, whose maximum difference was ``5.6e-9``.  The mixed result is useful:
coefficient sharing gains 1.57x, direct batched affine contraction gains 1.93x,
and batched multi-output differentials gain 1.43x.  Adaptive Haar execution
removes the former large-plane regression and is neutral here.  Batched
smoothing is 6% slower for these four large planes, so its axis API remains a
convenience rather than a speed claim.  Nearby denoising lambda paths can need
fewer iterations, but the actual diagnostics—not the API name—decide whether
that helps.

.. code-block:: shell

   python scripts/benchmark_workflows.py --profile standard \
     --output-json workflows.json --output-csv workflows.csv
   python scripts/check_benchmark_thresholds.py \
     --policy benchmarks/consolidation-thresholds.json \
     --artifacts-dir .

Explicit batch memory scaling
-----------------------------

``scripts/benchmark_batch_scaling.py`` sweeps batch counts 1, 2, 4, and 8 at
64x80, 128x160, and 256x320.  On the same standard profile, every batched path
agreed exactly with explicit plane dispatch.  The largest normalized traced
peak was 1.08: peak memory therefore remained linear or sublinear relative to
the one-plane baseline within measurement noise.  At 256x320 and batch eight,
traced peaks were 22.63 MiB for affine, 30.09 MiB for Laplacian-only
differentials, and 12.00 MiB for the adaptive Haar round trip.

These are Python allocation traces, not whole-process RSS measurements.  The
affine rows report cached geometry separately, and the required output buffer
is separated from temporary overhead in the JSON/CSV artifacts.

.. code-block:: shell

   python scripts/benchmark_batch_scaling.py --profile standard \
     --output-json batch-scaling.json --output-csv batch-scaling.csv

Downstream stability-soak workflows
-----------------------------------

``scripts/benchmark_downstream_workflows.py`` measures complete pipelines, not
isolated kernels.  Persisted registration includes atomic archive I/O,
compatibility validation, and three concurrent affine geometries.  Buffered
volume features apply a 3-D affine transform and write gradient and Laplacian
outputs into caller-owned arrays.  Both standard rows agreed exactly with
their explicit scalar references.  On this development machine registration
fan-out measured 288.2 ms versus 505.6 ms (1.75x), while buffered volume
features measured 156.1 ms versus 232.8 ms (1.49x).

The ratios are workload evidence, not isolated-kernel claims: registration's
timed path deliberately includes filesystem persistence and thread-pool
execution.  See :doc:`stability-soak` for the contracts being exercised.

.. code-block:: shell

   python scripts/benchmark_downstream_workflows.py --profile standard \
     --output-json downstream-workflows.json \
     --output-csv downstream-workflows.csv

Multiscale vectorization
------------------------

``scripts/benchmark_multiscale.py`` compares current whole-axis execution with
the retained row/column oracle on standard float64 images.  Pyramid reduction
measured 479.3 ms versus 3.668 s (7.65x), and one-scale Haar analysis measured
29.6 ms versus 68.8 ms (2.33x).  These ratios measure Python dispatch and array
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
