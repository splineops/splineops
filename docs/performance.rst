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
     - 136.4 ms
     - 40.5 ms
     - 8.25 MiB
   * - 2-D cubic grid
     - 42.7 ms
     - 40.4 ms
     - 16.54 MiB
   * - 2-D cubic, 200,000 points
     - 42.5 ms
     - 338.0 ms
     - 28.53 MiB
   * - 3-D linear grid
     - 16.2 ms
     - 231.0 ms
     - 22.51 MiB
   * - 3-D cubic grid
     - 26.6 ms
     - 67.9 ms
     - 28.23 MiB

These are a development-machine baseline, not portable promises.  The useful
result is the bounded growth of query temporaries: evaluation is tiled at
65,536 points while small grids retain a faster direct broadcast path.

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
speed.  For 200,000 random cubic 2-D points and seven repeated evaluations, the
ordinary path took 241.1 ms per call and the planned path 63.2 ms: a 3.81x
speedup.  Plan construction broke even after an estimated 1.28 calls and
retained 24.41 MiB.  This is a strong workload-specific capability, not a
reason to plan one-shot queries.

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
   :widths: 24 19 19 19

   * - Case
     - SplineOps
     - SciPy
     - SciPy speedup
   * - 2-D linear
     - 178.9 ms
     - 17.3 ms
     - 10.34x
   * - 2-D cubic
     - 485.4 ms
     - 31.5 ms
     - 15.42x
   * - 3-D linear
     - 190.2 ms
     - 25.5 ms
     - 7.46x
   * - 3-D cubic
     - 545.2 ms
     - 59.4 ms
     - 9.18x

Affine's current achievements are exact spline semantics and bounded
coordinate memory.  It does not beat SciPy's specialized kernels.

Reproduce the profile with:

.. code-block:: shell

   python scripts/benchmark_affine.py --profile standard \
     --output-json affine.json --output-csv affine.csv

Differentials vectorization
---------------------------

The 2-D spline differential implementation performs coefficient conversion
in batched contiguous arrays.  Against the retained scalar row/column oracle,
the standard 512x640 float64 gradient-magnitude benchmark measured 211.5 ms
versus 12.340 s, a 58.35x speedup.  Maximum absolute difference was
``1.704e-6``; the scalar oracle truncates its causal initialization at
single-precision epsilon, so it is not bit-identical to the batched helper.

.. code-block:: shell

   python scripts/benchmark_differentials.py --profile standard \
     --output-json differentials.json --output-csv differentials.csv

Resize comparisons
------------------

The initial 2026-07-15 smoke comparison found SplineOps faster than the tested
SciPy configurations on three rows with close agreement where semantics were
comparable.  OpenCV won all three generic 2-D timing rows, and PyTorch won two.
Coordinates, boundaries, antialiasing, and output sizing differ between these
libraries, so these are contextual comparisons rather than one universal
leaderboard.  See :doc:`project-status` for the exact command.

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
