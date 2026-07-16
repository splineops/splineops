Progress so far
===============

Snapshot: 2026-07-16, commit ``df94733`` on ``feature/publication``.

SplineOps completed seven repository-wide improvement passes before the 2.1
release preparation.  The work kept independent public modules separate while
strengthening numerical and resource contracts, reproducible performance
evidence, and maturity gates.  This page is the dated engineering record; the
:doc:`project-status`, :doc:`performance`, :doc:`stability-soak`, and
:doc:`roadmap` pages retain the detailed evidence.

Current outcome
---------------

SplineOps is now positioned as a precise N-D spline interpolation and
resampling library with specialized, independent tools for affine geometry,
differentials, smoothing, sparse regression, and multiscale analysis.  Its
strongest production capability remains projection-based resize.  The work so
far has made the rest of the package more coherent and dependable without
pretending that every research module is already stable or universally faster
than mature specialized libraries.

.. list-table:: Maturity at this snapshot
   :header-rows: 1
   :widths: 22 16 36 26

   * - Module
     - Status
     - What is now demonstrated
     - Main open gate
   * - Resize / ``ResizePlan``
     - Stable
     - Defined N-D sampling and projection contracts, native/Python parity,
       explicit axes, bounded plans, buffers, concurrency, and wheel coverage.
     - Continue portable performance monitoring.
   * - ``TensorSpline``
     - Stabilizing
     - Uniform construction grids, arbitrary point/grid queries, bounded
       evaluation, separable contraction, reusable changing-data geometry, and
       broad NumPy reference coverage.
     - Dedicated CuPy CI or a permanent narrowing of that experimental surface.
   * - Affine
     - Experimental
     - Planned 2-D/3-D transforms, explicit batch/channel axes, bounded cached
       geometry, coefficient sharing, immutable compatibility tags, and atomic
       schema-versioned persistence.
     - An unchanged downstream API-soak period and any resulting migration
       guidance.
   * - Differentials
     - Experimental
     - Planned 2-D/3-D gradient, Hessian, and Laplacian families with spacing,
       explicit axes, selective computation, and reusable structured buffers.
     - Compatibility history plus a deliberate legacy/angular-map migration
       decision.
   * - Smoothing / adaptive regression
     - Experimental
     - Reusable frequency or sparse-factorization plans, explicit axes,
       convergence diagnostics, lambda paths, and independent numerical
       references.
     - Broader parameter guidance and published-reference coverage.
   * - Pyramids / wavelets
     - Experimental
     - Whole-axis and explicit-axis execution, randomized reconstruction for
       reversible Haar/cubic cases, and an honest order-5 approximation label.
     - Higher-precision order-5 evidence or continued approximate status.

Seven completed passes
----------------------

1. **Trustworthy baseline.**  Reframed the package around precise spline
   semantics, documented provenance and module boundaries, established maturity
   labels, reproduced tests/builds/docs, and replaced universal speed language
   with equivalent-semantics benchmarks.
2. **Reusable numerical models.**  Added bounded ``TensorSpline`` evaluation,
   reusable query geometry, separable grid contraction, planned affine and
   differential APIs, smoothing/denoising plans, vectorized multiscale passes,
   and a measured native resize scheduling policy.
3. **Complete workflows.**  Added changing-data geometry reuse, explicit
   batch/channel axes across independent modules, controlled denoising paths,
   plan inspection, machine-readable workflow benchmarks, stored regression
   policies, and cross-platform benchmark workflows.
4. **Batched execution contracts.**  Removed avoidable per-slice affine and
   differential dispatch, introduced cache-bounded wavelet grouping, and added
   immutable affine coefficient compatibility fields.
5. **Persistence, buffers, and profile-backed optimization.**  Added selective
   differential outputs, fully checked structured buffers, portable atomic
   coefficient archives, concurrent reuse tests, memory scaling sweeps, and a
   narrower 3-D NumPy contraction justified by affine phase profiling.
6. **Realistic stability workflows.**  Added persisted registration fan-out and
   buffered 3-D feature pipelines, independent Keys/O-MOMS/fractional-smoothing
   references, cross-platform standard evidence, and an explicit approximate
   contract for inherited order-5 wavelet taps.
7. **Randomized and restart evidence.**  Added Hypothesis contracts for axes,
   dtypes, buffers, changing spline geometry, and reversible wavelets; pinned
   real schema-1/schema-2 archive bytes; expanded smoothing and sparse-hinge
   oracles; and introduced a repeated thread/process/restart/corruption soak.
   This pass intentionally changed no production numerical kernel because its
   profile did not justify one.

Evidence collected
------------------

The current local validation completed ``1222 passed`` in 146.33 seconds.
Black, scoped MyPy, warning-fatal Sphinx, source and native-wheel builds, and a
clean installed-wheel smoke test also passed.

The standard local API soak completed 12 cycles with:

* 36 atomic coefficient-archive replacements;
* 108 shared-field thread applications;
* 36 independently spawned process restorations;
* 12 deliberate corrupt-archive rejections; and
* 12 reuses of the same affine and differential destination objects.

Every numerical comparison was exact, input arrays remained unchanged, and
destination identities were preserved.  Parent profiling showed that the soak
time was dominated by deliberate process startup and teardown, so it provided
no evidence for another interpolation-kernel change.

The evidence commit passed all remote gates:

* `API stability soak #1
  <https://github.com/splineops/splineops/actions/runs/29479718318>`_ on Linux,
  macOS, and Windows;
* `Test Library #141
  <https://github.com/splineops/splineops/actions/runs/29479718365>`_ across
  Python 3.11--3.13 and all three operating systems; and
* `Quality gates #12
  <https://github.com/splineops/splineops/actions/runs/29479718142>`_.

The preceding `standard development benchmark matrix
<https://github.com/splineops/splineops/actions/runs/29429054363>`_ also passed
on all three operating systems and published complete artifacts.

Performance conclusions
-----------------------

The performance story is useful precisely because it is not simplified into
one flattering number:

* native resize is the package's strongest accelerated implementation, while
  cross-library results remain grouped by whether semantics are equivalent;
* reusable ``TensorSpline`` geometry and affine coefficient sharing eliminate
  real repeated work in matching workloads;
* vectorized differential execution substantially improves on its scalar
  oracle, but explicit-axis orchestration is not faster on every platform;
* affine cached workflows can benefit from reuse, while SciPy remains faster
  for the matched one-shot affine cases measured so far; and
* adaptive wavelet dispatch recovers useful small-plane vectorization without
  imposing large-plane transpose regressions.

Accordingly, SplineOps is proud of precise N-D contracts, projection
antialiasing, reusable geometry, bounded memory, and inspectable numerical
evidence.  It does not claim to be the fastest generic 2-D image resizer or the
best one-shot implementation of every operation.

Decisions deliberately not taken
--------------------------------

* ``TensorSpline`` remains its own continuous-model API; resize, affine,
  differentials, smoothing, regression, and multiscale tools remain separate.
* CuPy remains experimental ``TensorSpline`` interoperability.  No CuPy
  configuration is advertised as supported, and a zero-boundary coefficient
  path may transfer through a CPU solve; see :doc:`backend-support`.
* Affine and Differentials were audited but not promoted.  Their remaining gate
  is compatibility evidence over time, not another burst of implementation;
  see :doc:`graduation-audits`.
* The evidence passes did not promote Affine, Differentials, or any other
  experimental module.  The 2.1 release keeps those maturity labels.

Practical next steps
--------------------

#. Let the new cross-platform soak run on relevant changes and collect a period
   of unchanged real downstream use.
#. Record any friction found in coefficient persistence, output selection,
   buffer reuse, and explicit-axis calls before freezing those contracts.
#. Add reliable GPU CI before strengthening CuPy wording; otherwise keep the
   current narrow experimental contract.
#. Expand research references and parameter guidance where the module-specific
   graduation tables remain open, especially adaptive-regression tuning and
   the order-5 wavelet evidence.
#. Accept further performance work only when a representative profile identifies
   a numerical kernel or allocation pattern—not orchestration that the soak is
   intentionally designed to exercise.
#. Treat 2.1 as a compatibility milestone for the stable resize surface;
   defer other module promotions until their published gates close.

Reproducing the current evidence
--------------------------------

.. code-block:: shell

   python -m pytest -q
   python -m tox -e format-check,type
   python -m sphinx -b html docs /tmp/splineops-docs -W --keep-going \
     -D sphinx_gallery_conf.plot_gallery=0 \
     -D suppress_warnings=image.not_readable
   python -m build
   python scripts/run_api_stability_soak.py --profile standard \
     --output-json /tmp/splineops-api-soak.json

Longer ``extended`` soak runs are available through manual workflow dispatch;
they are not imposed on every development push.
