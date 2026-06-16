# Resize Optimization Progress Report

Date: 2026-06-15
Branch: `feature/publication`
Code checkpoint summarized: `5790dce Cache direct linear resize plans`

This report summarizes the resize optimization work completed so far. The
detailed engineering ledger remains in `scripts/resize_optimization_notes.md`;
this file is the shorter project-level summary.

## Goal

Optimize `splineops.resize` while preserving exact splineops semantics by
default. The work avoids silently switching to image-library semantics, because
OpenCV, skimage, and PyTorch differ in coordinate mapping, boundary handling,
kernel definitions, and antialiasing behavior.

## Current Status

The native `_lsresize` backend is now substantially faster on common 2-D and
3-D workloads, with stronger correctness coverage and systematic benchmark
tooling. The most important recent outcome is that exact splineops now beats
PyTorch on the close-output 3-D linear downsample benchmark on this CPU:

| Case | splineops | PyTorch | SciPy | skimage |
| --- | ---: | ---: | ---: | ---: |
| `3d_linear_down_random_f32` | `0.341 ms` | `0.575 ms` | `2.246 ms` | `2.206 ms` |

Latest full library artifact:
`/tmp/splineops_resize_libraries_full_torch_direct_plan_cache_20260615.{json,csv}`

Summary from that run:

| Backend | Comparable cases faster than splineops | Median relative-L2 vs splineops |
| --- | ---: | ---: |
| SciPy | `1/21` | `5.74e-08` |
| skimage | `0/21` | `2.39e-01` |
| PyTorch | `7/19` | `1.59e-05` |
| OpenCV | `12/18` | `2.61e-01` |

Interpreting the table:

- SciPy is the closest semantic comparison for many exact interpolation rows;
  splineops is faster on `20/21` rows in the latest full comparison.
- PyTorch is still faster on some 2-D cubic/antialiasing-style rows, but many
  of those rows have measurable semantic deltas.
- OpenCV remains very fast for 2-D image resizing, but its larger output deltas
  reflect different image-resize semantics rather than exact splineops parity.

## Major Work Completed

### Correctness and Routing

- Fixed finite-size spline IIR mirror initialization for short signals and
  constant preservation.
- Added parity regressions for native/Python behavior, batched/native behavior,
  default-auto routing, forced native routing, constants, and short axes.
- Improved axis scheduling so shrinking axes run first and pure interpolation
  identity axes are skipped.
- Made default native thread scheduling more conservative on SMT machines and
  avoided launching a worker thread for one-worker decisions.

### Benchmark and Quality Tooling

Added systematic scripts:

- `scripts/benchmark_resize_native.py`
  - native/Python/both backend timing
  - native knobs and environment metadata in JSON/CSV output
  - standard and full profiles
- `scripts/benchmark_resize_quality.py`
  - constant, ramp, impulse, checkerboard, sinusoid, and random comparisons
  - default-vs-float32 internal quality checks
- `scripts/benchmark_resize_plan.py`
  - repeated `ResizePlan` workloads
  - fresh-output and reused-output timings
  - now includes pure linear 2-D/3-D cases
- `scripts/benchmark_resize_libraries.py`
  - systematic comparisons against SciPy, skimage, OpenCV, and PyTorch
  - records timing plus max/mean/p99/relative-L2 output deltas

### Python Fallback

- Changed Python fallback defaults to `SPLINEOPS_BLOCK=256` and
  `SPLINEOPS_ACCUM=support`.
- Added Python plan-cache defaults with `SPLINEOPS_PLAN_CACHE_SIZE=32`.
- Measured `support/256` versus previous `mulsum/64`:
  - won `18/18` standard cases
  - about `1.85x` mean speedup and `1.82x` median speedup
  - artifact: `/tmp/splineops_resize_python_support_block256.{json,csv}`

### Native Batched Axis Path

- Extended batched native axis kernels from pure interpolation to projection
  and antialiasing.
- Added exact interior/boundary row mapping:
  - direct coefficient reads for interior rows
  - precomputed source/sign maps for boundary rows
  - no full extension buffer for the common batched path
- Made `LSRESIZE_BATCHED_AXIS` default to conservative `auto`.
- Cached batched row-run metadata inside immutable `Plan1D`.

Representative measurements:

- Default-auto versus explicit native off:
  - average 2-D median speedup about `2.15x`
  - weakest 2-D standard row still about `1.57x`
  - artifacts:
    `/tmp/splineops_resize_auto_default_b64_seq.{json,csv}`,
    `/tmp/splineops_resize_off_default_b64_seq.{json,csv}`
- Cached row-run/interior-boundary split:
  - 2-D mean/median speedup about `1.12x`
  - artifacts:
    `/tmp/splineops_resize_native_baseline_head_r8.{json,csv}`,
    `/tmp/splineops_resize_native_after_plan_row_runs_final_r8.{json,csv}`

### Native Preset Specialization

- Added fixed-support accumulator specializations for common presets:
  - `linear`
  - `cubic`
  - `linear-antialiasing`
  - `cubic-antialiasing`
- Added `LSRESIZE_SPECIALIZED_PRESETS=0` for A/B checks.
- Standard-profile local A/B:
  - won `30/36` medians and `33/36` best-of timings
  - about `1.15x` mean median speedup
- Full-profile local A/B:
  - won `92/130` medians and `100/130` best-of timings
  - about `1.13x` mean median speedup

### Float32 Internal Precision

- Added opt-in native float32 internals with `LSRESIZE_PRECISION=float32`.
- DC-centered projection paths before float32 recursive filters to preserve
  constant arrays better.
- Kept antialiasing/projection float32 internals opt-in because random outputs
  still show measurable drift versus conservative float64 internals.
- Auto-enabled float32 internals only for 2-D `float32` pure quadratic/cubic
  interpolation when `LSRESIZE_PRECISION` is unset.

Representative measurement:

- 2-D `float32` cubic interpolation medians improved by about `1.37x` to
  `1.48x` versus forced 64-bit internals.
- Mean median speedup about `1.42x`.
- artifacts:
  `/tmp/splineops_resize_native_auto_f32_interp_on.{json,csv}`,
  `/tmp/splineops_resize_native_auto_f32_interp_forced64.{json,csv}`

### Reusable ResizePlan

- Added public `splineops.resize.ResizePlan`.
- Native-backed plans precompute:
  - input/output shape
  - zoom factors
  - axis order
  - active axes
  - per-pass shapes and sizes
  - per-axis resize parameters
- Reused float32/float64 ping-pong intermediate buffers across repeated calls.
- Added `plan.apply(..., output=out)` direct writes into compatible
  C-contiguous output buffers.

Representative artifact:
`/tmp/splineops_resize_plan_standard.{json,csv}`

Best fit:

- fixed-geometry video/frame loops
- registration loops
- repeated batch preprocessing
- deterministic augmentation with constant shape/zoom

### Exact Linear Fast Paths

This is now the strongest exact-speed area.

Completed:

- Direct N-D pure-linear interpolation path behind default-on
  `LSRESIZE_LINEAR_INTERP`.
- Fused exact 2-D linear path behind default-on `LSRESIZE_FUSED_2D_LINEAR`.
- Runtime-dispatched AVX2/FMA kernels for selected 2-D linear rows on supported
  GNU/Clang x86 builds.
- Fused exact 3-D linear path behind default-on `LSRESIZE_FUSED_3D_LINEAR`.
- Fixed-support fused 3-D scalar kernels for common `2x2`, `1x2`, and `2x1`
  planes.
- Cached direct-linear source/weight metadata inside `Plan1D`, reused by all
  direct/fused linear kernels.

Representative measurements:

| Optimization | Representative result |
| --- | --- |
| Exact 2-D linear fast paths | mean single-thread speedup about `5.45x` vs `LSRESIZE_LINEAR_INTERP=0` |
| `2d_linear_down_1024_float32` | `4.81 ms -> 0.42 ms`, `11.44x` |
| `2d_linear_down_1024_float64` | `4.87 ms -> 0.56 ms`, `8.69x` |
| AVX2 2-D linear rows | pure 2-D linear medians won `8/12`, about `1.41x` mean speedup |
| Fused 3-D all-axis linear | `3d_linear_down_f32` `2.263 ms -> 0.452 ms`, `5.00x` |
| Fused 3-D two-axis `(0, 1)` | `1.912 ms -> 1.084 ms`, `1.76x` |
| Dedicated 3-D two-axis `(0, 2)` | `1.428 ms -> 0.423 ms`, `3.37x` |
| Dedicated 3-D two-axis `(1, 2)` | `1.073 ms -> 0.430 ms`, `2.50x` |
| Direct-plan-cache metadata | `2d_linear_down_1024_float32` `0.508 ms -> 0.427 ms`, `1.19x` |

Important dispatch decision:

- `(0, 2)` and `(1, 2)` 3-D two-axis patterns now use dedicated exact kernels
  instead of the earlier generic all-axis fused evaluator. The earlier forced
  evaluator experiment was rejected; the dedicated kernels won locally.
- A fused 3-D axis-2 interior branch-split experiment was measured and reverted
  because it regressed the important `(0, 1)` route without a durable all-axis
  win.

### Projection and Filter Cleanup

- Fused common batched `diff_sa` + `diff_as` col-major projection filters.
- Specialized half-length-2 col-major sampling FIR.
- Precomputed finite causal-initializer horizon once per pole/axis pass.
- Removed redundant integration average-buffer initialization on normal
  projection paths.
- Added direct 2-D axis-pass offset calculation in batched native kernels.
- Tightened native parallel launch policy so cheap direct-linear passes do not
  start worker threads only because they cross a line-count threshold.
- Added a default-on exact linear N-D last-axis direct path, with an escape hatch
  via `LSRESIZE_LAST_AXIS_LINEAR_DIRECT=0`, to skip generic offset unraveling
  for contiguous final-axis rows.

Representative measurement:

- Antialiasing rows won `13/16` medians with about `1.12x` mean median speedup
  after col-major filter improvements.
- Batched filter/offset cleanup won `8/8` 2-D antialiasing/projection medians,
  with about `1.05x` mean speedup.

## Current Default Knobs

| Area | Default | Override |
| --- | --- | --- |
| Native acceleration | `SPLINEOPS_ACCEL=auto` | `always`, `never` |
| Native batched axis | `auto` when unset | `LSRESIZE_BATCHED_AXIS=off/1/auto` |
| Native batch lines | adaptive by dimensionality/method | `LSRESIZE_BATCH_LINES=<n>` |
| Native row-major batched gather | enabled | `LSRESIZE_ROW_GATHER=0` |
| Native 3-D axis-1 direct scatter | enabled for large pure quadratic/cubic interpolation passes | `LSRESIZE_3D_AXIS1_DIRECT_SCATTER=0` |
| Native preset specialization | enabled | `LSRESIZE_SPECIALIZED_PRESETS=0` |
| Exact linear fast path | enabled | `LSRESIZE_LINEAR_INTERP=0` |
| Fused 2-D linear | enabled | `LSRESIZE_FUSED_2D_LINEAR=0` |
| Fused 3-D linear | enabled | `LSRESIZE_FUSED_3D_LINEAR=0` |
| Fused 3-D two-axis linear | enabled | `LSRESIZE_FUSED_3D_TWO_AXIS_LINEAR=0` |
| Fused projection average restore | explicit single-thread auto | `LSRESIZE_FUSED_PROJECTION_AVG_RESTORE=0/1/auto` |
| AVX2 linear kernels | enabled on supported x86 | `LSRESIZE_AVX2_LINEAR=0` |
| Last-axis linear direct path | enabled | `LSRESIZE_LAST_AXIS_LINEAR_DIRECT=0` |
| Native internal precision | auto f32 only for 2-D f32 pure quadratic/cubic interpolation | `LSRESIZE_PRECISION=float32` |
| Native plan cache | capacity `32` | `LSRESIZE_PLAN_CACHE_SIZE=<n>` |
| Native threads | workload-aware default | `LSRESIZE_NUM_THREADS=<n>` |
| Python block size | `256` | `SPLINEOPS_BLOCK=<n>` |
| Python accumulator | `support` | `SPLINEOPS_ACCUM=einsum/mulsum` |
| Python plan cache | capacity `32` | `SPLINEOPS_PLAN_CACHE_SIZE=<n>` |

## Validation Status

Latest local validation after the dedicated two-axis linear and projection
restore pass:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py::test_projection_avg_restore_fused_path_matches_disabled \
  tests/test_02_03_resize_cpp.py::test_batched_axis_matches_default_equal_degree_projection \
  tests/test_02_03_resize_cpp.py::test_batched_axis_matches_default_short_projection_axes \
  tests/test_02_03_resize_cpp.py::test_3d_linear_two_axis_fused_path_matches_disabled \
  tests/test_02_03_resize_cpp.py::test_3d_linear_fused_path_matches_axis_direct \
  tests/test_02_03_resize_cpp.py::test_nd_linear_interp_fast_path_matches_disabled
.venv/bin/python -m pytest -q tests/test_02_03_resize_cpp.py tests/test_02_02_resize.py
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_ab.py \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  src/splineops/utils/specs.py
git diff --check
```

Results:

| Check | Result |
| --- | ---: |
| Focused native checks | `46 passed` |
| Resize-focused suite | `253 passed` |
| Full suite | `507 passed` |
| Script py-compile | clean |
| `git diff --check` | clean |

## Recent Commit Trail

| Commit | Summary |
| --- | --- |
| `5790dce` | Cache direct linear resize plans |
| `28eb6eb` | Optimize fused 3D linear resize kernels |
| `44a65a8` | Add fused 3D linear resize path |
| `94fe06f` | Record torch resize comparison |
| `38a7152` | Record full resize library comparison |
| `3dc8900` | Optimize batched resize filter setup |
| `c6c88af` | Auto-enable float32 internals for 2D interpolation |
| `527db24` | Add AVX2 linear resize kernels |
| `f6fa714` | Generalize exact linear resize fast paths |
| `9182c73` | Specialize 2D float32 linear resize |
| `6163e24` | Add cross-library resize comparison benchmark |
| `2b011e8` | Optimize native col-major projection filters |
| `9f81b05` | Reuse native resize plan buffers |
| `59e2fb3` | Add resize plans and quality benchmarks |
| `6cda428` | Center float32 native projection residuals |
| `2145c30` | Add opt-in float32 native resize internals |
| `fa5fdac` | Specialize native resize preset accumulators |
| `b417cfe` | Document resize optimization wrap-up |
| `c91f933` | Cache native resize row-run maps |
| `c0b7140` | Optimize Python resize fallback defaults |

## Current Benchmark Artifacts

Most useful current artifacts:

- Native direct-plan-cache:
  `/tmp/splineops_resize_native_direct_plan_cache_final.{json,csv}`
- Full library comparison with PyTorch:
  `/tmp/splineops_resize_libraries_full_torch_direct_plan_cache_20260615.{json,csv}`
- Plan benchmark with added linear cases:
  `/tmp/splineops_resize_plan_direct_cache_final.{json,csv}`
- Final fused 3-D policy A/B:
  `/tmp/splineops_resize_native_fused3d_two_axis_supported_off.{json,csv}`
  and
  `/tmp/splineops_resize_native_fused3d_two_axis_final_policy_on.{json,csv}`
- Dedicated 3-D `(0, 2)` / `(1, 2)` two-axis linear A/B:
  `/tmp/splineops_ab_two_axis_fused_codex.{json,csv}`
- Projection average-restore fusion A/B:
  `/tmp/splineops_ab_projection_avg_restore_single_long_codex.{json,csv}`

## What Not To Claim

- Do not claim exact splineops cubic now matches OpenCV speed. OpenCV's cubic is
  a different fixed-kernel image operation and often has much larger output
  deltas.
- Do not claim float32 internals are universally default-safe. They are default
  only for a narrow 2-D `float32` pure interpolation scope; projection and
  antialiasing float32 internals remain opt-in.
- Do not claim every fused 3-D experiment won. The dedicated `(0, 2)` and
  `(1, 2)` two-axis kernels won locally, but batch-size and scheduler retunes
  measured during this pass were rejected.
- Do not claim projection average-restore fusion is a general threaded win. It
  is default-auto only when `LSRESIZE_NUM_THREADS=1` is explicit; `=1` forces it
  for experiments.

## Recommended Next Work

1. Specialize exact cubic interpolation more deeply, especially the 2-D pure
   interpolation case where PyTorch/OpenCV still have speed advantages but
   semantic differences.
2. Reduce temporary traffic for projection/antialiasing workloads beyond the
   single-thread average-restore fusion; threaded/default paths still need a
   different approach.
3. Expand `ResizePlan` reuse for fused/direct linear paths, possibly by caching
   output buffers or exposing a more explicit workspace API.
4. Add an explicit, opt-in image-resize semantics mode only if matching OpenCV
   style speed is a product goal; keep it separate from exact splineops
   semantics.
