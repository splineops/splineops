Stability-soak workflows
========================

The stability soak exercises independent public modules in complete pipelines.
It is deliberately separate from release planning: a green soak establishes
evidence for the current experimental contracts but does not silently promote
them or require a distribution release.

Representative workloads
------------------------

Persisted registration fan-out
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A changing image frame is prefiltered once, saved as an atomic numeric-plus-JSON
coefficient archive, restored through a compatible plan, and evaluated through
three affine geometries in a thread pool.  The explicit reference performs each
complete transform independently.  This covers persistence, compatibility,
geometry reuse, concurrency, and memory-capped plans in one operation.

Buffered volume features
~~~~~~~~~~~~~~~~~~~~~~~~

A batch of 3-D volumes is transformed through one affine geometry and passed to
a differential plan requesting gradient and Laplacian outputs.  Caller-owned
structured buffers receive the features.  The explicit reference processes one
volume at a time with fresh outputs.  This covers batch axes, float32 execution,
3-D contraction, physical spacing, output selection, and buffer reuse.

The executable examples are
:ref:`sphx_glr_auto_examples_08_workflows_01_registration_fanout.py` and
:ref:`sphx_glr_auto_examples_08_workflows_02_volume_features.py`.  The
machine-readable benchmark is ``scripts/benchmark_downstream_workflows.py``.
On the 2026-07-15 Linux standard profile, both optimized pipelines agreed
exactly with their explicit references.  Registration fan-out measured 1.75x
and buffered volume features 1.49x, but these ratios remain local development
evidence rather than portable promises.

Hardened contracts
------------------

Affine coefficient archives use schema version 2.  Numeric values are stored
little-endian, metadata is scalar JSON, and loading never enables NumPy object
pickles.  Saving writes and flushes a temporary file in the destination
directory before atomic replacement.  Schema-1 archives remain readable.
Malformed, truncated, extra-member, unsafe object-array, endian-swapped, thread,
and spawned-process cases are covered by tests.

Differential output buffers are fully validated before coefficient filtering or
any destination write.  Requested arrays must have exact shape and dtype and be
writeable.  Buffers may be strided, but they may not overlap the input or each
other.  Rejecting source overlap preserves the module's non-mutating contract;
rejecting output aliasing prevents one component from corrupting another.

Independent numerical references
--------------------------------

The reference set now includes:

* Keys cubic convolution evaluated from the piecewise formula in `Keys (1981)
  <https://doi.org/10.1109/TASSP.1981.1163711>`_;
* cubic O-MOMS evaluated through the published piecewise generator and an
  independently solved dense cardinal system from `Blu, Thévenaz, and Unser
  (2001) <https://doi.org/10.1109/83.931101>`_;
* a dense discrete-Fourier fixture for the fractional smoothing response
  described by `Unser and Blu (2007)
  <https://bigwww.epfl.ch/publications/unser0701.html>`_; and
* explicit differential mirror-boundary and integer/float32 dtype contracts.

The order-5 spline-wavelet table was checked against its DeconvolutionLab2
source.  That source contains the same limited-precision, short tap table; no
authoritative higher-precision replacement was found.  SplineOps therefore
retains it reproducibly and exposes ``exact_reconstruction=False`` and the
``"bounded-approximation"`` reconstruction contract instead of inventing taps
or claiming perfect reconstruction.

Affine phase evidence
---------------------

``scripts/profile_affine_phases.py`` times plan construction and instruments
prefilter and coefficient-evaluation time inside complete plan calls.  This
avoids treating separately warmed microbenchmarks as an exact decomposition.
The current standard run found coefficient evaluation dominant in all four
rows, accounting for 63%--91% of measured in-call phase time; cached-plan
construction is reported separately and remains substantial.  The NumPy 3-D
evaluator therefore contracts one support axis at a time, reducing intermediate
work where the measured support block is largest; 1-D/2-D and CuPy retain their
prior paths.  SciPy remains faster for matched one-shot affine execution.

Evidence and graduation
-----------------------

Pushes containing ``[standard-bench]`` in the commit message deliberately run
the standard development profile on Linux, macOS, and Windows.  Other relevant
publication-branch pushes retain the smoke profile.  Both modes verify numerical
equivalence before publishing per-runner artifacts.

The first standard matrix usefully rejected two overly strong performance
floors: explicit-axis affine measured 0.70x--0.87x across the runners, and the
Windows differential row measured 1.01x.  Numerical equivalence still passed.
These APIs are consequently documented as orchestration and allocation
contracts with platform-dependent timing, not universal speedups.  The policy
retains its 0.5x complete-workflow regression floor and strict numerical gates.

This pass is eligible for a graduation review only after the standard artifacts
and the two workloads have been reviewed.  Promotion is still not automatic:
Affine and Differentials remain experimental until their soak period reveals no
contract changes, and ``TensorSpline`` remains stabilizing until dedicated CuPy
CI and broader backend evidence exist.
