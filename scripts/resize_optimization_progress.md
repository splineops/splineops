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
| Native float32 strided-offset gather | enabled for routed pure interpolation | `LSRESIZE_STRIDED_OFFSET_GATHER=0` |
| Native gather-prefilter scaling | enabled | `LSRESIZE_GATHER_PREFILTER_SCALE=0` |
| Native 2-D projection batch tuning | enabled | `LSRESIZE_2D_PROJECTION_BATCH_TUNE=0` |
| Native float32 row-wise initial causal setup | enabled | `LSRESIZE_ROWWISE_INITIAL_CAUSAL=0` |
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

- Full native/Python after fused gather-prefilter scaling:
  `/tmp/splineops_native_full_both_gather_prefilter_scale_20260616.{json,csv}`
- Full library comparison after fused gather-prefilter scaling, default
  scheduler:
  `/tmp/splineops_libraries_full_default_gather_prefilter_scale_20260616.{json,csv}`
- Full library comparison after fused gather-prefilter scaling, forced
  8 threads:
  `/tmp/splineops_libraries_full_threads8_gather_prefilter_scale_20260616.{json,csv}`
- Fused gather-prefilter scale A/B:
  `/tmp/splineops_ab_gather_prefilter_scale_final_20260616.{json,csv}`
- Full native/Python after row-wise initial-causal setup and projection batch
  tuning:
  `/tmp/splineops_native_full_both_rowwise_projection_batch_20260616.{json,csv}`
- Full library comparison after row-wise/projection-batch pass, default
  scheduler:
  `/tmp/splineops_libraries_full_default_rowwise_projection_batch_20260616.{json,csv}`
- Row-wise initial-causal A/B:
  `/tmp/splineops_ab_rowwise_initial_causal_f32_20260616.{json,csv}`
- Projection batch tuning A/B:
  `/tmp/splineops_ab_projection_batch_tune_20260616.{json,csv}`
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

## 2026-06-16 Update: Fused Gather-Prefilter Scaling

The latest optimization keeps Arrate's least-squares projection method intact
and removes one memory pass in the native batched spline prefilter:

- The interpolation-prefilter normalization factor is applied while gathering
  the axis block into coefficient storage.
- The prefilter then applies only the recursive spline poles.
- The path is default-on and A/B controlled with
  `LSRESIZE_GATHER_PREFILTER_SCALE=0`.

Focused A/B:

- Artifact: `/tmp/splineops_ab_gather_prefilter_scale_final_20260616.csv`
- Result: median `1.119x`, mean `1.155x`, 15 wins and 2 losses across 18 rows,
  no failed checks.
- Default scheduler only: median `1.124x`, 6 wins and 0 losses.

End-to-end artifacts:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_gather_prefilter_scale_20260616.csv`
  - 43 overlaps, median speedup `25.11x`, mean `29.16x`
- Library full comparison, default scheduler:
  `/tmp/splineops_libraries_full_default_gather_prefilter_scale_20260616.csv`
  - SciPy faster in `1/21`, skimage `0/21`, OpenCV `13/18`, Torch `8/19`
- Library full comparison, forced 8 threads:
  `/tmp/splineops_libraries_full_threads8_gather_prefilter_scale_20260616.csv`
  - SciPy faster in `0/21`, skimage `0/21`, OpenCV `17/18`, Torch `10/19`
- Legacy Java 2-D reference:
  `/tmp/splineops_legacy_java_full_gather_prefilter_scale_20260616.csv`
  - splineops median speedup `8.44x` over 14 overlapping 2-D cases

Validation:

- Native editable rebuild: clean.
- Focused fused-scale/row-gather/direct-scatter parity: `18 passed`.
- Full suite: `527 passed`.
- Benchmark script py-compile and `git diff --check`: clean.

## 2026-06-16 Update: Row-Wise Initial-Causal Setup

The follow-up profile after gather-prefilter scaling still showed pure cubic
time dominated by gather plus prefilter. This pass keeps the same spline
prefilter but speeds up the float32 finite-horizon initial-causal setup by
walking coefficient rows contiguously.

Accepted changes:

- `LSRESIZE_ROWWISE_INITIAL_CAUSAL` is default-on for the float32 col-major
  interpolation prefilter; double-internal paths keep the scalar initializer.
- `LSRESIZE_2D_PROJECTION_BATCH_TUNE` is default-on for targeted 2-D
  antialiasing batch-size choices.

Focused A/B:

- Row-wise initial-causal:
  `/tmp/splineops_ab_rowwise_initial_causal_f32_20260616.csv`
  - median `1.093x`, mean `1.144x`, 8 wins and 1 loss, no failed checks
  - default scheduler: median `1.177x`, 4 wins and 0 losses
- Projection batch tuning:
  `/tmp/splineops_ab_projection_batch_tune_20260616.csv`
  - median `1.033x`, mean `1.041x`, 16 wins and 7 losses, no failed checks

End-to-end artifacts:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_rowwise_projection_batch_20260616.csv`
  - 43 overlaps, median speedup `27.32x`, mean `28.39x`
- Library full comparison, default scheduler:
  `/tmp/splineops_libraries_full_default_rowwise_projection_batch_20260616.csv`
  - SciPy faster in `0/21`, skimage `0/21`, OpenCV `15/18`, Torch `7/19`
  - exact-ish SciPy and Torch rows are all slower than splineops
- Legacy Java 2-D reference:
  `/tmp/splineops_legacy_java_full_rowwise_projection_batch_20260616.csv`
  - splineops median speedup `10.15x` over 14 overlapping 2-D cases

Validation:

- Native editable rebuild: clean.
- Focused parity: `28 passed`.
- Full suite: `537 passed`.
- Benchmark script py-compile and `git diff --check`: clean.

## 2026-06-16 Update: Strided-Offset Gather And Benchmark Compare

The next profile pass showed the remaining large float32 pure-cubic time was
gather-heavy:

- `2d_cubic_down_2048_float32`, single-thread profile:
  gather `39.14%`, prefilter `27.35%`, accumulation/scatter the rest.
- `3d_cubic_down_large_f32`, single-thread profile:
  gather `49.89%`, prefilter `24.15%`, accumulate-scatter `21.51%`.
- `3d_cubic_aniso_large_f32`, single-thread profile:
  gather `37.62%`, accumulate-scatter `37.96%`, prefilter `23.69%`.

Accepted changes:

- Added `LSRESIZE_STRIDED_OFFSET_GATHER`, default-on, for batched float32 pure
  interpolation gathers when batch line offsets form an arithmetic run.
  The path avoids loading the per-line offset array inside every coefficient
  row and uses contiguous copies for unit-stride float32 runs.
- Kept projection/antialiasing and double-internal gather paths on the prior
  offset-array gather after full A/B showed projection regressions when the
  new path was applied globally.
- Added `summarize_resize_benchmarks.py compare` to compare saved CSV artifacts
  by stable row keys and report wins/losses plus largest regressions/wins.

Focused A/B:

- Strided-offset gather:
  `/tmp/splineops_ab_strided_offset_gather_routed_20260616.csv`
  - median `1.021x`, mean `1.052x`, 61 wins and 33 losses across the full
    native profile, no failed checks
  - large float32 cubic rows were the intended wins:
    `3d_cubic_down_large_f32` `1.29x/1.11x/1.14x` for
    `1/8/default` threads, `3d_cubic_aniso_large_f32`
    `1.37x/1.25x/1.30x`, and `2d_cubic_down_2048_float32`
    `1.07x/1.07x/1.19x`

End-to-end artifacts:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_strided_offset_gather_20260616.csv`
  - 43 overlaps, median native/Python speedup `23.23x`, mean `26.85x`
  - 3-D median native/Python speedup `22.89x`
  - best native thread counts: `1:12`, `8:18`, `default:13`
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_strided_offset_gather_20260616.csv`
  - SciPy faster in `1/21`, skimage `0/21`, OpenCV `14/18`, Torch `7/19`
  - exact-ish SciPy rows all slower; exact-ish Torch rows all slower
- Library full comparison, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_strided_offset_gather_20260616.csv`
  - SciPy faster in `1/21`, skimage `0/21`, OpenCV `16/18`, Torch `10/19`
  - exact-ish SciPy rows all slower; exact-ish Torch faster in `3/8`
- Legacy Java 2-D reference:
  `/tmp/splineops_legacy_java_full_strided_offset_gather_20260616.csv`
  - splineops median speedup `10.02x` over 14 overlapping 2-D cases

Validation:

- Native editable rebuild: clean.
- Focused parity:
  `34 passed` for strided-offset gather plus adjacent gather/prefilter/direct
  scatter flags.
- Full suite: `543 passed`.
- Benchmark script py-compile and `git diff --check`: clean.

## 2026-06-16 Update: Row-Wise Finite Causal Initialization

The next prefilter pass extended `LSRESIZE_ROWWISE_INITIAL_CAUSAL` beyond the
float32 truncated-horizon case. The default path now uses the same row-wise,
contiguous accumulation strategy for:

- double-internal truncated-horizon initial-causal setup
- float32 and double exact finite-length mirror initializers on short axes

This keeps Arrate's least-squares projection and spline recursion unchanged.
Only the setup of the first causal coefficient is rearranged from many strided
per-line sums into row-contiguous accumulation. The scalar implementation
remains available with `LSRESIZE_ROWWISE_INITIAL_CAUSAL=0`.

Accepted A/B:

- Artifact: `/tmp/splineops_ab_rowwise_initial_causal_finite_clean_20260616.csv`
- Full native profile, `1/8/default` threads:
  median `1.028x`, mean `1.054x`, 64 wins and 24 losses, no failed checks.
- By method: cubic median `1.054x`, cubic-antialiasing median `1.016x`,
  linear-antialiasing median `1.036x`.

Rejected experiments:

- Forcing float32 internals on projection/antialiasing was fast but not default
  safe: the selected A/B showed about `1.234x` median speedup but output drift
  up to `1.75e-3` on large cubic-antialiasing rows.
- Fusing length-2 sampling FIR with output scatter was correctness-clean but
  slower: median `0.876x`, mean `0.894x`, 1 win and 15 losses. The experiment
  was removed rather than kept as a default-off knob.

End-to-end artifacts:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_rowwise_finite_20260616.csv`
  - 43 overlaps, median native/Python speedup `24.86x`, mean `27.42x`
  - cubic median `25.18x`, antialiasing/projection median `16.22x`
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_rowwise_finite_20260616.csv`
  - SciPy faster in `1/21`, skimage `0/21`, OpenCV `14/18`, Torch `8/19`
  - exact-ish SciPy rows all slower; exact-ish Torch rows all slower
- Library full comparison, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_rowwise_finite_20260616.csv`
  - SciPy faster in `1/21`, skimage `0/21`, OpenCV `16/18`, Torch `12/19`
  - exact-ish SciPy rows all slower; exact-ish Torch faster in `5/8`
- Legacy Java 2-D reference:
  `/tmp/splineops_legacy_java_full_rowwise_finite_20260616.csv`
  - splineops median speedup `10.82x` over 14 overlapping 2-D cases

Validation:

- Native editable rebuild: clean.
- Focused parity:
  `30 passed` for row-wise initial-causal plus adjacent gather/projection flags.
- Full suite: `543 passed`.
- Benchmark script py-compile and `git diff --check`: clean.
