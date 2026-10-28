LS batching and affine compiled evaluation
==========================================

28 October 2026. Changes against SplineOps 2.2.0, parent
``a148a9ecc9daaccc74939dbb52824fbe66294f5b``.

Implemented changes
-------------------

The native dispatcher now enables the existing batched projection kernel for
3-D direct cubic least-squares projection (degrees 3, 3, 3) when an axis has
at least 128 independent lines. This removes repeated scalar workspaces while
preserving the cross-Gram projection, boundary treatment, precision policy,
and small-workload guard. Public resize APIs are unchanged.

``AffinePlan``, ``affine_transform``, and ``rotate`` now accept
``backend="auto"``, ``"numpy"``, or ``"scipy"``. Automatic evaluation selects
SciPy for degrees 1--5, float32/float64, and named zero, mirror, or periodic
boundaries. Unsupported configurations retain the NumPy evaluator. Degree 0
retains its existing boundary-tie convention; degrees 6--7 are also unchanged.

Both evaluators use exactly the same SplineOps coefficient prefilter and
compatible coefficient fields/archives. SciPy receives ``prefilter=False``.
Coefficient extension maps zero to ``grid-constant`` with zero fill, mirror
to ``mirror``, and periodic to ``grid-wrap``. Float32 uses tiled coordinates
from the existing rounded matrix arithmetic; float64 uses the compiled affine
kernel directly. Compiled plans avoid the unused zero-valued TensorSpline and
support caches. Their retained template/geometry arrays occupy zero bytes,
versus 18.75--32.81 MiB in the standard NumPy plans; this does not count Python
objects, the small matrix/offset arrays, temporary working memory, or outputs.

Benchmark conditions
--------------------

Windows 11, Intel Core Ultra 9 275HX (24 logical processors), Python 3.12.14,
NumPy 2.5.4, SciPy 1.18.1. The native extension was built locally with MSVC in
Release mode. Runs were sequential, with no simultaneous tests or builds.
``OPENBLAS_NUM_THREADS``, ``OMP_NUM_THREADS``, and ``MKL_NUM_THREADS`` were set
to ``1`` before starting Python. LS additionally selects its own native
thread count explicitly, then tests automatic selection.

All timings below are medians. LS uses three warmups and eleven repeats with
warmed reusable plans. The scalar control sets ``LSRESIZE_BATCHED_AXIS=off``:
it uses the per-line kernel selected by the parent's automatic cubic 3-D LS
path, although dispatch overhead differs. Affine uses two warmups and nine
repeats, separating construction, one-shot transforms, repeated plan calls,
and evaluation of shared coefficients. Method order is randomized. No
allocation tracing runs during timing; the older affine comparison script
was also corrected to collect allocation peaks separately.

LS results
----------

Float32, one native thread; milliseconds per warmed call:

.. list-table:: Final LS measurement
   :header-rows: 1

   * - Input to output
     - Scalar control
     - Automatic
     - Speedup
   * - 64x96x128 to 32x48x64
     - 21.106
     - 4.465
     - 4.73x
   * - 48x256x256 to 48x32x32
     - 29.666
     - 5.981
     - 4.96x
   * - 48x56x64 to 18x34x24
     - 3.788
     - 0.929
     - 4.08x
   * - 24x32x40 to 36x48x60
     - 5.124
     - 1.014
     - 5.05x
   * - 96x128x128 to 48x64x64
     - 40.346
     - 7.415
     - 5.44x
   * - 8x9x10 to 3x4x5 (unchanged scalar control)
     - 0.073
     - 0.089
     - 0.82x

Float64 non-tiny cases improved 4.06x--4.93x. Automatic threading improved
the three larger geometries 1.58x--1.86x across both dtypes; the smaller
anisotropic volume was approximately flat (1.00x--1.01x), and upsampling
improved 1.10x--1.17x. The tiny control stays scalar in both modes and measured
0.82x--0.89x. This is a dispatch-control comparison, not evidence of a
small-volume benefit or a measured regression against the parent's automatic
path. All 24 comparisons produced identical arrays.

Raw samples: ``../../resize/2026-10-28-ls-batching/windows-final.json``.
The adjacent ``windows.json`` and ``review.rst`` preserve the earlier prototype
measurement and review rather than replacing the final results above.

Affine results
--------------

Cubic, 23-degree rotation about the array center; speedups are NumPy/SciPy:

.. list-table:: All standard affine cases
   :header-rows: 1

   * - Shape
     - Precision
     - Mode
     - One-shot
     - Reused plan
     - Shared coefficients
   * - 512x384
     - float32
     - mirror
     - 5.90x
     - 1.13x
     - 1.21x
   * - 512x384
     - float32
     - zero
     - 3.80x
     - 0.96x
     - 0.95x
   * - 512x384
     - float32
     - periodic
     - 4.34x
     - 1.16x
     - 1.22x
   * - 512x384
     - float64
     - mirror
     - 9.71x
     - 1.55x
     - 1.99x
   * - 512x384
     - float64
     - zero
     - 6.87x
     - 1.30x
     - 1.41x
   * - 512x384
     - float64
     - periodic
     - 7.46x
     - 1.59x
     - 1.96x
   * - 48x56x64
     - float32
     - mirror
     - 5.13x
     - 1.72x
     - 1.84x
   * - 48x56x64
     - float32
     - zero
     - 2.76x
     - 1.29x
     - 1.33x
   * - 48x56x64
     - float32
     - periodic
     - 3.48x
     - 1.67x
     - 1.81x
   * - 48x56x64
     - float64
     - mirror
     - 8.03x
     - 2.47x
     - 2.79x
   * - 48x56x64
     - float64
     - zero
     - 4.60x
     - 1.74x
     - 1.93x
   * - 48x56x64
     - float64
     - periodic
     - 6.05x
     - 2.39x
     - 2.77x

For a concrete float64 3-D mirror example, one-shot time fell from 206.826 to
25.757 ms; repeated plan calls from 63.053 to 25.570 ms; and coefficient
evaluation from 57.294 to 20.547 ms. The float32 2-D zero-boundary repeated
case regressed from 16.002 to 16.749 ms, and shared-coefficient evaluation
from 12.691 to 13.407 ms. Explicit ``backend="numpy"`` remains appropriate
when its cached evaluation wins for a particular workload.

Raw samples: ``../../affine/2026-10-28-backends/windows.json``.
A supplementary smoke-size sweep with degrees 1 and 5 and three channels
covers 24 further cases in ``windows-additional.json``. It measured
1.93x--3.14x one-shot, 1.02x--2.83x reused-plan, and 0.98x--3.31x
shared-coefficient speedups; these small-array results are not large-volume
performance claims.

Numerical validation
--------------------

LS timing pairs were bitwise equal in all 24 cases. Independent tests compare
3-D projections with tensor products of the existing 100-digit inner-product
fixtures and with the Python projection reference on nonseparable random
volumes, mixed resize factors, both precisions, and unchanged axes. Profile
tests verify that eligible volumes enter batching and tiny volumes do not.

The standard affine cubic sweep had maximum absolute differences below
``1.7e-6`` for float32 and ``6.8e-13`` for float64 against the independent
NumPy evaluator. Supplementary degree-5 float32 differences were below
``1.6e-5``; all supplementary float64 differences were below ``3.6e-14``.
These are measured differences on the recorded random inputs, not universal
absolute error bounds for arbitrarily scaled data. Coefficient preparation is
bitwise equal between backends; affine output equality is numerical, not
promised bitwise. Float64 coordinate arithmetic and polynomial accumulation
can round differently.

Regression tests cover degrees 1--5, both precisions, all named modes, short
and singleton axes, off-grid boundary knots and multiple extension periods,
large-offset float32 coordinates, general affine matrices, explicit spatial
and batch axes, reversed input strides, strided and aliased outputs, archive
and field reuse across backends, fallback selection, and an analytic linear
polynomial. Existing tolerances were not relaxed; tests specifically about
NumPy support caching explicitly select that backend.

The full suite passed: 1,409 passed, six skipped. Skips concern one optional
PyTorch study dependency and five Linux-only fork/RSS checks. Formatting and
static typing also passed (167 Python files and 75 source modules).
The strict Sphinx HTML build passed with gallery execution disabled and
external intersphinx inventories disabled for an offline build. Wheel and
source distribution builds passed; the source archive contains no Python
cache files. A fresh wheel-only environment passed native-extension import,
LS constant preservation, affine backend parity, packaged-resource, and
demo-command help checks.

Reproduction
------------

From a source installation with the native extension, set the three thread
environment variables listed above, then run::

    python scripts/benchmark_resize_ls_batching.py --repeats 11 --output-json ls.json
    python scripts/benchmark_affine_backends.py --profile standard --repeats 9 --output-json affine.json
    python scripts/benchmark_affine_backends.py --profile smoke --repeats 5 --degrees 1,5 --batch-size 3 --output-json affine-additional.json
    python -m pytest -q

These results establish faster execution of the existing operators on one
Windows machine. Linux/macOS timing remains unmeasured. They do not establish
better reconstruction quality than other filters or application-specific
methods. No public resize presets or mathematical accuracy targets changed.
