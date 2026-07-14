# Resize Optimization Notes

Date: 2026-06-12

This note tracks optimization ideas for `splineops.resize`, especially the native
`_lsresize` backend. The goal is to preserve the exact algorithmic behavior while
reducing runtime, memory movement, and repeated setup cost.

For a concise project-level summary, see
`scripts/resize_optimization_progress.md`. This file remains the detailed
engineering ledger with artifacts, experiments, handoffs, and roadmap notes.

## Weekend Wrap-Up: 2026-06-12

The resize optimization pass now has three solid pillars:

1. **Correctness baseline is stronger.**
   - Fixed the finite-size spline IIR mirror initializer for short signals and
     constant preservation.
   - Added regressions for constants, native/Python parity, batched/native parity,
     forced batched routing, and default-auto routing.
   - Latest full-suite validation: `507 passed`.

2. **Native default path is substantially faster for the target 2-D workloads.**
   - Axis scheduling now processes shrinking axes first and skips pure
     interpolation identity axes.
   - Default thread scheduling is more conservative on SMT machines and avoids
     launching a single worker thread for one-worker decisions.
   - The native batched axis kernel now covers pure interpolation, projection, and
     antialiasing.
   - `LSRESIZE_BATCHED_AXIS` is default-auto when unset. Use
     `LSRESIZE_BATCHED_AXIS=off` for the original line-by-line path and
     `LSRESIZE_BATCHED_AXIS=1` to force batching.
   - The native batched path skips the full extension buffer, uses direct
     coefficient reads for interior rows, and uses exact source/sign mapping for
     boundary rows.
   - Native plans are cached by default with capacity 32. Set
     `LSRESIZE_PLAN_CACHE_SIZE=0` to measure cold-plan behavior.
   - Batched row-run metadata is now precomputed inside `Plan1D`, so repeated
     same-plan calls reuse interior/boundary runs and boundary source/sign maps.

3. **Python fallback is also faster and easier to benchmark.**
   - `scripts/benchmark_resize_native.py` now supports
     `--backend {native,python,both}`.
   - Python fallback defaults are now `SPLINEOPS_BLOCK=256` and
     `SPLINEOPS_ACCUM=support`.
   - Python plan cache capacity defaults to 32 via
     `SPLINEOPS_PLAN_CACHE_SIZE`, with compatibility for
     `LSRESIZE_PLAN_CACHE_SIZE`.

Current default knobs:

| Area | Default | Override |
| --- | --- | --- |
| Native acceleration | auto via `SPLINEOPS_ACCEL=auto` | `always`, `never` |
| Native batched axis | `auto` when `LSRESIZE_BATCHED_AXIS` is unset | `off`, `1`, `auto` |
| Native batch lines | adaptive by dimensionality/method; 3-D float32 cubic-antialiasing downsample uses `256` | `LSRESIZE_BATCH_LINES=<n>` |
| Native row-major batched gather | enabled | `LSRESIZE_ROW_GATHER=0` |
| Native float32 strided-offset gather | enabled for routed pure interpolation | `LSRESIZE_STRIDED_OFFSET_GATHER=0` |
| Native gather-prefilter scaling | enabled | `LSRESIZE_GATHER_PREFILTER_SCALE=0` |
| Native projection output-prefilter scaling | auto only for explicit `LSRESIZE_NUM_THREADS=1` | `LSRESIZE_PROJECTION_OUTPUT_PREFILTER_SCALE=0/1/auto` |
| Native 2-D projection batch tuning | enabled | `LSRESIZE_2D_PROJECTION_BATCH_TUNE=0` |
| Native float32 row-wise initial causal setup | enabled | `LSRESIZE_ROWWISE_INITIAL_CAUSAL=0` |
| Native 3-D axis-1 direct scatter | enabled for large pure quadratic/cubic interpolation passes | `LSRESIZE_3D_AXIS1_DIRECT_SCATTER=0` |
| Native preset specialization | enabled | `LSRESIZE_SPECIALIZED_PRESETS=0` |
| Native exact linear interpolation fast path | enabled | `LSRESIZE_LINEAR_INTERP=0` |
| Native fused 2-D linear path | enabled | `LSRESIZE_FUSED_2D_LINEAR=0` |
| Native fused 3-D linear path | enabled | `LSRESIZE_FUSED_3D_LINEAR=0` |
| Native fused 3-D two-axis linear path | enabled | `LSRESIZE_FUSED_3D_TWO_AXIS_LINEAR=0` |
| Native fused projection average restore | auto only for explicit `LSRESIZE_NUM_THREADS=1` | `LSRESIZE_FUSED_PROJECTION_AVG_RESTORE=0/1/auto` |
| Native AVX2 2-D linear upsample path | enabled on supported x86 | `LSRESIZE_AVX2_LINEAR=0` |
| Native last-axis linear direct path | enabled | `LSRESIZE_LAST_AXIS_LINEAR_DIRECT=0` |
| Native internal precision | `float32` internals for 2-D/3-D `float32` pure quadratic/cubic interpolation and 3-D downsampling antialiasing presets; `float64` otherwise | `LSRESIZE_PRECISION=float32` to force batched float32 internals; non-empty non-f32 values keep the conservative path |
| Native plan cache | enabled, capacity `32` | `LSRESIZE_PLAN_CACHE_SIZE=<n>` |
| Native threads | workload-aware default | `LSRESIZE_NUM_THREADS=<n>` |
| Python block size | `256` | `SPLINEOPS_BLOCK=<n>` |
| Python accumulator | `support` | `SPLINEOPS_ACCUM=einsum` or `mulsum` |
| Python plan cache | enabled, capacity `32` | `SPLINEOPS_PLAN_CACHE_SIZE=<n>` |

Precision policy update, 2026-06-19:

- 3-D `float32` downsampling with the public antialiasing presets now uses the
  existing DC-centered float32 batched projection path automatically when
  `LSRESIZE_PRECISION` is unset.
- 2-D projection/antialiasing remains on the conservative float64-internal path
  by default because strict 2-D random-output parity still shows measurable
  drift under forced float32 internals.
- Explicit equal-degree least-squares projection remains conservative; the
  automatic projection float32 gate is limited to the public oblique
  antialiasing triples `(1,0,1)`, `(2,1,2)`, and `(3,1,3)`.
- `LSRESIZE_PRECISION=float64` restores the strict float64-internal path, and
  `LSRESIZE_PRECISION=float32` still forces batched float32 internals.

3-D cubic antialiasing batch retune, 2026-06-19:

- After the 3-D float32 antialiasing precision gate, the best batch size moved
  upward. The default adaptive policy now uses `256` lines for 3-D float32
  `cubic-antialiasing` downsample axes.
- Focused A/B against forced historical batch `64`
  (`--repeats 15 --warmups 4`) improved `3d_cubic_aa_down_large_f32`
  default-thread median from `7.36 ms` to `4.91 ms`; the standard
  `3d_cubic_aa_down_f32` default-thread row improved from `4.47 ms` to
  `3.92 ms`.
- A strided-offset gather experiment for the same projection rows was mixed
  and was not kept.

Measured progress so far:

- Native default-auto versus explicit native off:
  - saved sequential artifacts:
    `/tmp/splineops_resize_auto_default_b64_seq.{json,csv}` and
    `/tmp/splineops_resize_off_default_b64_seq.{json,csv}`
  - default-auto with batch size 64 was faster than explicit off on every 2-D
    standard case by median
  - 2-D median speedup averaged about `2.15x`; the weakest 2-D standard case was
    still about `1.57x`
- Native cached row-run/interior-boundary split versus the committed baseline:
  - baseline artifact:
    `/tmp/splineops_resize_native_baseline_head_r8.{json,csv}`
  - final artifact:
    `/tmp/splineops_resize_native_after_plan_row_runs_final_r8.{json,csv}`
  - default-thread medians improved in `15/18` standard cases overall and
    `14/16` 2-D cases
  - 2-D median speedup was about `1.12x` mean / `1.12x` median
  - best-of timings improved in `15/18` cases with about `1.16x` mean /
    `1.12x` median speedup
- Native plan cache:
  - mostly helps small/repeated workloads
  - observed wins include about `1.17x` for 1-D length 64 cubic, `1.08x` for
    16x16 cubic, `1.06x` for 32x32 cubic, and `1.02x` for 64x64
    cubic-antialiasing on the local CPU
- Python fallback:
  - `support`/256 versus previous `mulsum`/64 won `18/18` standard cases
  - mean speedup about `1.85x`, median speedup about `1.82x`
  - saved artifact:
    `/tmp/splineops_resize_python_support_block256.{json,csv}`
- Native fixed-support preset specialization:
  - implemented for the batched native path for `linear`, `cubic`,
    `linear-antialiasing`, and `cubic-antialiasing`
  - `LSRESIZE_SPECIALIZED_PRESETS=0` disables it for A/B checks
  - local standard-profile A/B artifacts:
    `/tmp/splineops_resize_specialized_on.{json,csv}` and
    `/tmp/splineops_resize_specialized_off.{json,csv}`
  - with `--threads 1,default`, `--repeats 5`, `--warmups 2`,
    specialization won `30/36` medians and `33/36` best-of timings, with about
    `1.15x` mean median speedup across the standard profile
  - full-profile A/B artifacts:
    `/tmp/splineops_resize_full_specialized_on.{json,csv}` and
    `/tmp/splineops_resize_full_specialized_off.{json,csv}`
  - with `--threads 1,2,4,8,default`, `--repeats 3`, `--warmups 1`,
    specialization won `92/130` medians and `100/130` best-of timings, with
    about `1.13x` mean median speedup across the full profile
- Opt-in native float32 internals:
  - enabled with `LSRESIZE_PRECISION=float32` (also accepts `single` and `f32`)
  - scope is intentionally limited to native batched passes over `float32`
    arrays; `float64` workloads still use the existing 64-bit internal path
  - original local standard-profile artifacts:
    `/tmp/splineops_resize_float32_internal_default.{json,csv}` and
    `/tmp/splineops_resize_float32_internal_f32.{json,csv}`
  - on selected 2-D `float32` cases (`--threads 1,default`, `--repeats 5`,
    `--warmups 2`), the opt-in path won `15/16` medians and `16/16` best-of
    timings, with about `1.52x` mean median speedup
  - the projection path is now DC-centered before the float32 recursive filters
    and adds the DC component back on output; this preserves constant arrays in
    projection/antialiasing spot checks while keeping the computation in
    `float32` scratch space
  - centered sequential standard-profile artifacts:
    `/tmp/splineops_resize_float32_centered_seq_default.{json,csv}` and
    `/tmp/splineops_resize_float32_centered_seq_f32.{json,csv}`
  - on selected 2-D `float32` cases, the centered opt-in path won `16/16`
    medians and `16/16` best-of timings versus the default 64-bit internal
    path, with about `1.43x` mean median speedup
  - random antialiasing/projection outputs still differ from the default 64-bit
    internal path, so the mode remains opt-in
- Automatic native float32 internals for 2-D pure interpolation:
  - default precision policy now uses the existing float32-internal batched
    path for 2-D `float32` pure quadratic/cubic interpolation when
    `LSRESIZE_PRECISION` is unset
  - linear interpolation keeps using the exact linear fast paths by default;
    antialiasing/projection stays on the conservative 64-bit internal path
    unless `LSRESIZE_PRECISION=float32` is explicitly requested
  - local standard-profile A/B artifacts:
    `/tmp/splineops_resize_native_auto_f32_interp_on.{json,csv}` and
    `/tmp/splineops_resize_native_auto_f32_interp_forced64.{json,csv}`
  - with `--threads 1`, `--repeats 9`, `--warmups 3`, the 2-D `float32`
    cubic interpolation medians improved by about `1.37x` to `1.48x`
    versus forced 64-bit internals, with about `1.42x` mean median speedup
  - local quality artifact:
    `/tmp/splineops_resize_quality_auto_f32_interp_standard.{json,csv}`;
    explicit float32 still shows measurable antialiasing/projection drift, so
    those modes remain opt-in
  - local library comparison artifact:
    `/tmp/splineops_resize_libraries_auto_f32_interp_standard.{json,csv}`;
    splineops was faster than SciPy on `14/15` comparable standard-profile
    cases on this CPU, while OpenCV was faster on `11/13` 2-D cases but uses
    different image-resize semantics in this comparison
  - latest full-profile local library comparison artifact:
    `/tmp/splineops_resize_libraries_full_20260615.{json,csv}`;
    splineops was faster than SciPy on `20/21` comparable rows and faster than
    skimage on `21/21`; OpenCV was faster on `16/18` 2-D rows but had median
    relative-L2 delta about `2.61e-01` versus splineops, and PyTorch was not
    installed in this virtual environment
  - latest full-profile local library comparison with PyTorch installed:
    `/tmp/splineops_resize_libraries_full_torch_20260615.{json,csv}`;
    PyTorch `2.12.0+cpu` was faster on `8/19` supported rows, with median
    speed `0.79x` versus splineops and median relative-L2 delta `1.59e-05`;
    the most actionable close-output PyTorch win was
    `3d_linear_down_random_f32` (`2.22x`, `rel_l2=2.59e-06`)
- Native exact linear interpolation fast paths:
  - implemented behind default-on `LSRESIZE_LINEAR_INTERP`; older
    `LSRESIZE_2D_LINEAR_INTERP` and `LSRESIZE_2D_FLOAT_INTERP` remain accepted
    as compatibility aliases when the new knob is unset
  - scope is pure `method="linear"`; `LSRESIZE_PRECISION=float32` keeps using
    the existing opt-in float32-internal batched path
  - the N-D direct path precomputes exact source/weight entries from `Plan1D`
    once per axis pass and avoids generic line buffers, spline prefilter calls,
    and sparse row-plan traversal at every sample
  - the direct linear path now uses storage-matched accumulator weights:
    `float32` arrays use float weights/accumulation in the exact direct linear
    kernel, while `float64` arrays keep double weights/accumulation
  - the fused 2-D path is default-on whenever both 2-D axes are active for pure
    linear interpolation; it evaluates the exact separable linear plan directly
    into the final output and avoids the intermediate ping-pong array
  - the fused 2-D inner loop hoists the output-row support branch out of the
    column loop, which materially improves the common 2-tap row case
  - the exact 2-D linear paths now have runtime-dispatched AVX2/FMA row kernels
    for `float32` and `float64` on GNU/Clang x86 builds; they remain scalar on
    unsupported CPUs and can be disabled with `LSRESIZE_AVX2_LINEAR=0`
  - the fused all-axis upsample path uses an AVX2 row kernel for the common
    2-row/2-column support region; the single-axis 2-D path uses a smaller
    horizontal AVX2 kernel for the common 2-column support region
  - the fused AVX2 path is intentionally gated to all-axis growth because local
    A/B runs showed four-gather fused rows winning on upsample rows but not on
    downsample/aniso rows on the i7-7820X test machine
  - latest AVX2 native A/B artifacts:
    `/tmp/splineops_resize_native_avx2_axis1_on.{json,csv}` and
    `/tmp/splineops_resize_native_avx2_axis1_off.{json,csv}`
  - in that A/B, pure 2-D linear medians won `8/12` rows with about `1.41x`
    mean speedup; 2-D linear upsample rows won `4/4` with about `2.00x` mean
    speedup, and 2-D linear anisotropic rows won `4/4` with about `1.28x` mean
    speedup
  - final native A/B artifacts:
    `/tmp/splineops_resize_native_linear_final_current.{json,csv}` and
    `/tmp/splineops_resize_native_linear_final_current_disabled.{json,csv}`
  - on the expanded native standard profile, the fast paths won all pure
    linear medians except one noisy default-thread 2-D anisotropic row:
    2-D rows won `12/12` single-thread and `11/12` default-thread; 3-D rows
    won `2/2` single-thread and `2/2` default-thread
  - 2-D pure linear median speedup versus `LSRESIZE_LINEAR_INTERP=0`: about
    `5.45x` mean / `3.93x` median single-thread, and `3.16x` mean / `3.25x`
    median default-thread
  - 3-D pure linear median speedup versus `LSRESIZE_LINEAR_INTERP=0`: about
    `1.87x` mean / median single-thread, and `1.63x` mean / median
    default-thread
  - selected fused 2-D downsample rows improved strongly, for example
    `2d_linear_down_1024_float32` single-thread `4.81 ms -> 0.42 ms`
    (`11.44x`) and `2d_linear_down_1024_float64` single-thread
    `4.87 ms -> 0.56 ms` (`8.69x`)
  - exact fused 3-D linear interpolation is implemented behind default-on
    `LSRESIZE_FUSED_3D_LINEAR`; it evaluates direct linear plans into the final
    output for all-axes-active 3-D pure linear workloads and the measured-winning
    two-axis `(0, 1)` pattern, avoiding intermediate arrays and extra
    full-volume passes
  - 3-D linear fused A/B artifacts:
    `/tmp/splineops_resize_native_fused3d_linear_on.{json,csv}` and
    `/tmp/splineops_resize_native_fused3d_linear_off.{json,csv}`
  - on the native standard profile (`--threads 1`, `--repeats 9`,
    `--warmups 3`), `3d_linear_down_f32` improved from `2.287 ms` to
    `1.101 ms` median (`2.08x`), while the single-axis 3-D linear aniso row
    stayed on the existing direct axis path
  - the fused 3-D scalar inner loop now has fixed-support `2x2`, `1x2`, and
    `2x1` specializations to remove the generic support loops from common
    linear rows
  - final two-axis policy artifacts:
    `/tmp/splineops_resize_native_fused3d_two_axis_supported_off.{json,csv}`
    and
    `/tmp/splineops_resize_native_fused3d_two_axis_final_policy_on.{json,csv}`
  - on the final policy A/B (`--threads 1`, `--repeats 9`, `--warmups 3`),
    `3d_linear_down_f32` improved from `2.263 ms` to `0.452 ms` median
    (`5.00x`), and `3d_linear_two_axis01_f32` improved from `1.912 ms` to
    `1.084 ms` median (`1.76x`)
  - dedicated exact `(0, 2)` and `(1, 2)` two-axis kernels now replace the
    earlier forced-through-all-axis experiment; they keep Arrate's direct
    linear plan weights/boundary handling and avoid the poor memory pattern
    that made the generic fused evaluator lose
  - dedicated two-axis A/B artifact:
    `/tmp/splineops_ab_two_axis_fused_codex.{json,csv}`; on local standard
    cases, `(0, 2)` improved `1.428 ms -> 0.423 ms` (`3.37x`) with
    `LSRESIZE_NUM_THREADS=1` and `1.466 ms -> 0.474 ms` (`3.10x`) with
    default scheduling, while `(1, 2)` improved `1.073 ms -> 0.430 ms`
    (`2.50x`) single-thread and `1.051 ms -> 0.706 ms` (`1.49x`) default
  - direct linear source/weight metadata is now precomputed inside `Plan1D`
    for pure linear interpolation and reused by all direct/fused linear
    kernels; this removes the per-call compact-plan rebuild from one-shot
    cached plans and `ResizePlan.apply`
  - direct-plan-cache native artifact:
    `/tmp/splineops_resize_native_direct_plan_cache_final.{json,csv}`;
    versus the previous final-policy artifact, selected setup-sensitive rows
    improved by median: `2d_linear_down_1024_float32` `0.508 ms -> 0.427 ms`
    (`1.19x`), `2d_linear_aniso_1024_float32` `0.761 ms -> 0.692 ms`
    (`1.10x`), and `3d_linear_two_axis01_f32` `1.084 ms -> 1.064 ms`
    (`1.02x`)
  - a branch-split experiment for the fused 3-D axis-2 2-tap interior was
    measured and reverted because it regressed the important `(0, 1)` two-axis
    route (`3d_linear_two_axis01_f32`) without a durable all-axis win
  - post-direct-plan-cache full PyTorch comparison artifact:
    `/tmp/splineops_resize_libraries_full_torch_direct_plan_cache_20260615.{json,csv}`;
    splineops was faster than SciPy on `20/21` supported full-profile rows,
    faster than skimage on `21/21`, faster than PyTorch on `12/19`, and faster
    than OpenCV on `6/18` comparable 2-D rows; `3d_linear_down_random_f32`
    stayed in splineops' favor at `0.341 ms` versus PyTorch `0.575 ms`
  - post-fused-3-D full PyTorch comparison artifact:
    `/tmp/splineops_resize_libraries_full_torch_fused3d_20260615.{json,csv}`;
    the close-output `3d_linear_down_random_f32` row improved from `1.310 ms`
    to `0.747 ms` for splineops in the full library profile, reducing
    PyTorch's lead on that row from `2.22x` to `1.26x`
  - final full PyTorch comparison artifact:
    `/tmp/splineops_resize_libraries_full_torch_fused3d_final_policy_20260615.{json,csv}`;
    `3d_linear_down_random_f32` improved further to `0.347 ms` for splineops
    versus `0.579 ms` for PyTorch, flipping that close-output row in favor of
    splineops; overall PyTorch was faster on `7/19` supported full-profile rows
    with median speed `0.70x` versus splineops
  - latest cross-library standard artifact:
    `/tmp/splineops_resize_libraries_avx2_axis1_standard.{json,csv}`
  - in that run, SciPy was faster on only `1/15` standard rows and skimage on
    `0/15`; splineops beat OpenCV on both standard 2-D linear anisotropic rows
    (`0.191 ms` vs `0.227 ms` for `float32`, `0.300 ms` vs `0.329 ms` for
    `float64`) and was effectively tied on 2-D random linear downsample
  - OpenCV remained faster on most 2-D rows overall, but its median relative-L2
    delta versus splineops was about `2.22e-01`, reflecting different
    coordinate, boundary, kernel, and antialiasing semantics
  - exact cubic was not fused because it requires the B-spline IIR prefilter as
    part of the exact splineops semantics; OpenCV's cubic path is a fixed-kernel
    interpolation with much larger output deltas
  - fixed-support preset accumulation now also covers the opt-in float32
    internal batched path; sequential A/B artifacts:
    `/tmp/splineops_resize_f32_preset_seq_on.{json,csv}` and
    `/tmp/splineops_resize_f32_preset_seq_off.{json,csv}`
  - on selected 2-D `float32` cases, the float32 preset dispatch won `12/16`
    medians and `15/16` best-of timings versus the generic float32 accumulator,
    with about `1.05x` median speedup
- Native col-major projection filters:
  - fused the common batched `diff_sa` + `diff_as` pair into one col-major pass
    for both default double-internal and opt-in float32-internal paths
  - specialized half-length-2 col-major sampling FIR, avoiding per-row mirror
    modulo work and work-buffer zero-fill for the supported public synthesis
    degrees
  - standard native benchmark artifacts:
    `/tmp/splineops_resize_sampling_fir_baseline.{json,csv}` and
    `/tmp/splineops_resize_colmajor_filters_fast.{json,csv}`
  - on local standard native timings (`--threads 1,default`, `--repeats 5`,
    `--warmups 2`, `--batched-axis auto`), the combined filter pass won `24/36`
    medians and `23/36` best-of timings overall
  - antialiasing rows, where the fused projection diff applies most directly,
    won `13/16` medians and `12/16` best-of timings, with about `1.12x` mean
    median speedup
  - fused projection average restore into the final diff pass, but only
    default-auto when `LSRESIZE_NUM_THREADS=1` is explicit; forced default
    threaded runs were too noisy/regressive to keep enabled generally
  - single-thread projection restore artifact:
    `/tmp/splineops_ab_projection_avg_restore_single_long_codex.{json,csv}`;
    on selected 2-D antialiasing rows it won `3/8` medians, lost `0/8`, with
    about `1.018x` median and `1.030x` mean speedup
- Native batched filter and offset cleanup:
  - precomputes the finite causal-initializer horizon once per pole/axis pass
    instead of recalculating the same logarithms for every line in a batch
  - avoids redundant integration average-buffer initialization on normal
    projection paths
  - uses a direct 2-D axis-pass offset calculation in the batched native
    kernels instead of generic N-D index unraveling
  - local standard-profile A/B artifacts:
    `/tmp/splineops_resize_projection_next_baseline.{json,csv}` and
    `/tmp/splineops_resize_projection_next_fast_r2.{json,csv}`
  - with `--threads 1`, `--repeats 9`, `--warmups 3`, 2-D
    antialiasing/projection rows won `8/8` medians, with about `1.05x` mean
    and `1.03x` median speedup
  - the same pass improved all 2-D pure cubic standard rows (`8/8` medians),
    with about `1.15x` mean and median speedup, because cubic interpolation
    also uses the batched prefilter and 2-D offset path
  - an earlier 2026-06-16 retune kept `LSRESIZE_BATCH_LINES=64` before the
    row-major gather pass; after row-major gather, the unset default became
    narrowly adaptive for 2-D pure quadratic/cubic downsampling
  - in that earlier run, `LSRESIZE_BATCH_LINES=64`
    beat tested `32` and `16` alternatives on the focused float32 rows, and
    `LSRESIZE_PARALLEL_THRESHOLD=1e6` beat `750000` on the focused scheduler
    sweep
  - rejected retune artifacts:
    `/tmp/splineops_ab_batch_64_32_f32_codex.{json,csv}`,
    `/tmp/splineops_ab_batch_64_16_f32_codex.{json,csv}`, and
    `/tmp/splineops_ab_parallel_threshold_1m_750k_codex.{json,csv}`
  - local quality artifact:
    `/tmp/splineops_resize_quality_projection_next_fast_standard.{json,csv}`;
    default-vs-explicit-float32 quality deltas were unchanged from the prior
    precision-policy run
  - local plan-reuse and library artifacts:
    `/tmp/splineops_resize_plan_projection_next_fast_standard.{json,csv}` and
    `/tmp/splineops_resize_libraries_projection_next_fast_standard.{json,csv}`
- Public reusable `ResizePlan`:
  - added as `splineops.resize.ResizePlan`
  - native-backed plans precompute input/output shape, zoom factors, axis order,
    active axes, per-pass shapes, per-pass output sizes, and per-axis resize
    parameters once; native per-axis `Plan1D` metadata is still reused by the
    existing process-local cache
  - repeated native plan calls now reuse float32/float64 ping-pong intermediate
    buffers instead of allocating them on every call
  - `ResizePlan.apply(..., output=out)` writes directly into a matching
    C-contiguous native output buffer when it is safe to do so; mismatched
    dtypes, non-contiguous outputs, and input/output aliases keep the previous
    temporary-and-copy semantics
  - standard plan benchmark artifacts:
    `/tmp/splineops_resize_plan_standard.{json,csv}`
  - on the local standard profile, plan reuse preserved exact output parity;
    median speedups versus one-shot `resize` ranged from about `1.00x` to
    `1.08x` for fresh output and from about `0.98x` to `1.15x` with a reused
    output buffer
  - pure-Python fallback remains available when `SPLINEOPS_ACCEL=never` or the
    native extension is unavailable
- Quality and benchmark automation:
  - added `scripts/benchmark_resize_quality.py` for constant, ramp, impulse,
    checkerboard, sinusoid, and random comparisons between default internals
    and `LSRESIZE_PRECISION=float32`
  - added `scripts/benchmark_resize_plan.py` for repeated same-shape
    `ResizePlan` workloads with fresh-output and reused-output timings
  - added `scripts/benchmark_resize_libraries.py` for systematic timing and
    quality comparisons against optional external libraries such as SciPy,
    scikit-image, OpenCV, and PyTorch
  - library comparison artifacts:
    `/tmp/splineops_resize_libraries_smoke.{json,csv}` and
    `/tmp/splineops_resize_libraries_standard.{json,csv}`
  - local standard library comparison, single-threaded where controllable:
    SciPy matched splineops closely on non-antialiasing linear/cubic cases but
    was slower on most standard rows; OpenCV was faster on 2-D rows but uses
    different interpolation/antialiasing semantics and showed much larger
    deltas versus splineops; PyTorch was not installed in the local venv and
    was reported as skipped
  - added a manual GitHub Actions workflow,
    `.github/workflows/resize-benchmark.yml`, to collect native benchmark,
    quality, and plan-reuse artifacts on CI hardware without gating PRs on
    noisy timings

Validation status:

- Native editable rebuild after general exact linear/fused 2-D fast paths:
  clean.
- Direct batched-axis parity tests: `68 passed`.
- Opt-in float32 internal focused tests: `11 passed`.
- Native resize module: `130 passed`.
- Native resize module with `LSRESIZE_SPECIALIZED_PRESETS=0`: `130 passed`.
- Native resize module with `LSRESIZE_LINEAR_INTERP=0`: `130 passed`.
- Native resize module with `LSRESIZE_FUSED_2D_LINEAR=0`: `130 passed`.
- Resize API suite: `95 passed`.
- Resize API suite with `SPLINEOPS_ACCEL=never`: `95 passed`.
- Focused resize suite: `225 passed`.
- Full suite: `479 passed`.
- Native editable rebuild after AVX2 linear kernels: clean.
- Focused resize suite after AVX2 linear kernels: `229 passed`.
- Native resize module with `LSRESIZE_AVX2_LINEAR=0`: `134 passed`.
- Full suite after AVX2 linear kernels: `483 passed`.
- Fresh external virtualenv editable rebuild after general exact linear/fused
  2-D fast paths: clean.
- Fresh external virtualenv focused resize suite: `225 passed`.
- Fresh external virtualenv native resize module with
  `LSRESIZE_SPECIALIZED_PRESETS=0`: `130 passed`.
- Fresh external virtualenv full suite: `479 passed`.
- Fresh external virtualenv quality smoke:
  `/tmp/splineops_resize_quality_fresh_quick.{json,csv}`.
- Python compile checks for updated scripts/specs: clean.
- Native linear/fused A/B:
  `/tmp/splineops_resize_native_linear_final_current.{json,csv}` and
  `/tmp/splineops_resize_native_linear_final_current_disabled.{json,csv}`.
- Cross-library standard comparison after general linear/fused path:
  `/tmp/splineops_resize_libraries_linear_final_current.{json,csv}`.
- Cross-library standard comparison after AVX2 linear kernels:
  `/tmp/splineops_resize_libraries_avx2_axis1_standard.{json,csv}`.
- Standard quality sweep:
  `/tmp/splineops_resize_quality_standard.{json,csv}`.
- Col-major filter quality smoke:
  `/tmp/splineops_resize_quality_colmajor_filters_quick.{json,csv}`.
- Col-major filter native benchmark:
  `/tmp/splineops_resize_colmajor_filters_fast.{json,csv}`.
- ResizePlan reuse smoke:
  `/tmp/splineops_resize_plan_smoke.{json,csv}`.
- ResizePlan reuse standard:
  `/tmp/splineops_resize_plan_standard.{json,csv}`.
- Cross-library resize smoke:
  `/tmp/splineops_resize_libraries_smoke.{json,csv}`.
- Cross-library resize standard:
  `/tmp/splineops_resize_libraries_standard.{json,csv}`.
- Native editable rebuild after batched filter/offset cleanup: clean.
- Native resize module after batched filter/offset cleanup: `138 passed`.
- Focused ResizePlan parity after batched filter/offset cleanup: `10 passed`.
- Full suite after batched filter/offset cleanup: `487 passed`.
- Native editable rebuild after fused 3-D linear path: clean.
- Focused resize suite after fused 3-D linear path: `237 passed`.
- Full suite after fused 3-D linear path: `491 passed`.
- Native editable rebuild after fused 3-D linear fixed-support policy: clean.
- Focused fused 3-D checks after final policy: `12 passed`.
- Focused resize suite after final fused 3-D policy: `243 passed`.
- Full suite after final fused 3-D policy: `497 passed`.
- `scripts/benchmark_resize_native.py` py-compile after final policy: clean.
- Native editable rebuild after direct linear plan-cache metadata: clean.
- Focused direct/fused linear checks after direct plan cache: `30 passed`.
- Focused resize suite after direct plan cache: `243 passed`.
- Full suite after direct plan cache: `497 passed`.
- `scripts/benchmark_resize_plan.py` and `scripts/benchmark_resize_native.py`
  py-compile after direct plan cache: clean.
- `git diff --check`: clean on the latest implementation pass.

### Full Optimization Roadmap

The algorithm is now in a good exact, default-on state. The remaining high-upside
work is mostly specialization, precision policy, and repeated-workload API
design.

1. **Specialized native preset kernels.**
   - Status: fixed-support specialization is implemented in the batched native
     accumulator for the common preset triples, and the common col-major
     projection filter sequence now has a fused pass.
   - Target methods: `linear`, `cubic`, `linear-antialiasing`,
     `cubic-antialiasing`.
   - Remaining deeper stage: specialize more of the full pipeline, including
     degree-specific IIR code and broader projection sequences, not just the
     accumulation and first diff/filter pieces.
   - Keep the current generic path as the fallback for uncommon degree triples.
   - Continue to keep the generic path available through
     `LSRESIZE_SPECIALIZED_PRESETS=0` for A/B checks.

2. **Opt-in float32 internal mode.**
   - Status: implemented for native batched `float32` axis passes behind
     `LSRESIZE_PRECISION=float32`.
   - The projection path is DC-centered to remove the observed constant-array
     boundary drift from the pure float32 recursive filters.
   - The default path still computes with 64-bit scratch/accumulation.
   - Validation covers default-precision parity, opt-in agreement thresholds,
     and constant preservation for projection presets.
   - Remaining deeper stage: test high-frequency image workloads and decide
     whether a public API precision option is warranted beyond the environment
     flag.
   - Keep the mode opt-in unless a future accuracy study supports changing the
     default precision policy.

3. **Public reusable `ResizePlan`.**
   - Status: implemented as `splineops.resize.ResizePlan`.
   - The plan object makes repeated same-shape workloads explicit:
     `plan = ResizePlan(input_shape, zoom, method); out = plan.apply(x)`.
   - Native-backed plans precompute axis order, active axes, output shapes,
     per-pass geometry, and per-axis parameters, while per-axis `Plan1D`
     metadata remains reused by the native process-local cache.
   - Native-backed plans reuse intermediate ping-pong scratch buffers across
     calls and can write directly into a compatible user-provided output array.
   - Best fit: video frames, registration loops, batch processing, and repeated
     augmentation with fixed geometry.

4. **Memory and temporary-buffer strategy.**
   - Status: first stage implemented for reusable native plans.
   - Remaining deeper stage: reduce temporary traffic inside the axis kernels,
     where antialiasing/projection workloads still spend most of their time.
   - Explore fused final-axis writes or permutation-aware scheduling for
     workloads where strided passes dominate.
   - Consider tiling for large volumes where memory traffic, not arithmetic,
     dominates.

5. **Routing and scheduler calibration on more hardware.**
   - Re-run standard/full artifacts on machines with different core counts,
     cache sizes, and SMT behavior.
  - Revisit the adaptive `LSRESIZE_BATCH_LINES` heuristic, default-auto
    thresholds, direct last-axis routing, and default thread caps only after cross-machine
    artifacts show a consistent better choice.

6. **Benchmark hygiene before every default change.**
   - Always save before/after JSON+CSV artifacts.
   - Compare medians and best-of timings; short sub-3 ms cases can be noisy.
   - Keep parity checks enabled for smoke/small profiles, and use
     `--skip-checks` only for large timing sweeps.
   - For cross-library comparisons, report both timing and quality deltas
     because coordinate, boundary, and antialiasing semantics differ between
     libraries.

## Handoff: 2026-06-12

Work completed today:

- Extended the native batched axis kernel behind `LSRESIZE_BATCHED_AXIS=1` from
  pure interpolation to projection and antialiasing.
- Promoted conservative automatic routing to the default when
  `LSRESIZE_BATCHED_AXIS` is unset. Use `LSRESIZE_BATCHED_AXIS=off` to
  force the original line-by-line native path, and `LSRESIZE_BATCHED_AXIS=1`
  to force the batched path everywhere.
- Reworked the batched axis accumulator to skip materializing the full
  `[left pad | line | right pad]` extension buffer. Interior rows now read
  coefficient columns directly, and boundary/tail rows use a precomputed
  per-weight source/sign table.
- Split pure interpolation into its own batched native kernel. It skips
  projection scratch, correction filtering, and projection-only branches.
- Added a pure-interpolation direct-write path for non-contiguous output axes,
  avoiding the intermediate `y` buffer plus strided scatter on those passes.
- Moved the col-major batched IIR prefilter into `cpp/lsresize/src/filters.*`
  so the finite-size mirror initializer is shared instead of duplicated in
  `resize_nd.cpp`.
- Added batched native helpers for:
  - `do_integ`
  - `do_diff`
  - output interpolation prefilter
  - output sampling FIR
- Added native parity regressions comparing the batched path against the
  default native path for:
  - pure `linear`, `quadratic`, and `cubic`
  - `linear-antialiasing`, `quadratic-antialiasing`, and `cubic-antialiasing`
  - equal-degree projection (`interp = analy = synthe`) for degrees 1, 2, 3
- Added an `auto`-mode parity regression covering large 2-D cases and a 3-D
  fallback case.
- Extended `scripts/benchmark_resize_native.py` with:
  - `--batched-axis {env,off,on,auto}`
  - `--batch-lines-sweep`
  - per-result batched mode and batch-size metadata
  - per-case batch-size sweep summaries
- Added workload-aware default scheduling in `parallel_utils.h`:
  - explicit `LSRESIZE_NUM_THREADS` remains a hard override
  - unset/default scheduling caps larger pools near a physical-core estimate
    instead of always using every logical CPU
  - one-worker decisions now run directly in the current thread instead of
    launching a single `std::thread`
- Added a bounded process-local native `Plan1D` cache:
  - keyed by line length, realized output length, degrees, and shift bits
  - plans are immutable/read-only after construction and shared by axis workers
  - `LSRESIZE_PLAN_CACHE_SIZE=0` disables the cache for measurements/debugging
- Moved the native batched row map into cached `Plan1D` metadata:
  - contiguous interior/boundary row runs are precomputed once per plan
  - boundary rows use cached per-weight coefficient source/sign mapping
  - batched ND kernels now execute branch-free direct interior runs and fall
    back to the exact mapped boundary runs
  - this removes per-call row-map allocation/setup from repeated same-plan
    workloads
- Promoted the batched-axis default block size from 32 to 64 after a saved
  standard `--batch-lines-sweep` on the local target CPU.
- Extended `scripts/benchmark_resize_native.py` beyond native-only measurements:
  - `--backend {native,python,both}` selects the implementation under test
  - Python rows report `<n/a>` for native thread/batch-line controls
  - smoke parity can now compare native against Python and Python against native
  - JSON metadata records the active Python fallback knobs
- Tuned the pure-Python fallback defaults:
  - `SPLINEOPS_BLOCK` default is now 256 instead of 64
  - `SPLINEOPS_ACCUM` default is now `support`
  - `SPLINEOPS_ACCUM=einsum` and `SPLINEOPS_ACCUM=mulsum` remain available for
    comparison/debugging
  - Python plan cache capacity now defaults to 32 via
    `SPLINEOPS_PLAN_CACHE_SIZE`, with fallback compatibility for
    `LSRESIZE_PLAN_CACHE_SIZE`
  - `SPLINEOPS_PLAN_CACHE=0` or `SPLINEOPS_PLAN_CACHE_SIZE=0` disables the
    Python plan cache

Current changed files to expect in the working tree:

- `cpp/lsresize/src/resize_1d.cpp`
- `cpp/lsresize/src/resize_1d.h`
- `cpp/lsresize/src/resize_nd.cpp`
- `scripts/resize_optimization_notes.md`

Validation run:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py::test_batched_axis_matches_default_pure_interpolation \
  tests/test_02_03_resize_cpp.py::test_batched_axis_matches_default_antialiasing \
  tests/test_02_03_resize_cpp.py::test_batched_axis_matches_default_equal_degree_projection \
  tests/test_02_03_resize_cpp.py::test_batched_axis_auto_matches_default
.venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
LSRESIZE_BATCHED_AXIS=off .venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
LSRESIZE_BATCHED_AXIS=1 .venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
LSRESIZE_PLAN_CACHE_SIZE=0 .venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
SPLINEOPS_ACCEL=never .venv/bin/python -m pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  src/splineops/resize/_pycore/resize_nd.py \
  src/splineops/utils/specs.py
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile smoke \
  --backend both \
  --threads default \
  --repeats 2 \
  --warmups 1
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile smoke \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis env \
  --skip-checks
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile smoke \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis off \
  --skip-checks
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,2,4,8,16,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis env \
  --skip-checks \
  --output-json /tmp/splineops_resize_scheduler_before_auto.json \
  --output-csv /tmp/splineops_resize_scheduler_before_auto.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 8,16,default \
  --repeats 5 \
  --warmups 2 \
  --batched-axis env \
  --skip-checks \
  --output-json /tmp/splineops_resize_scheduler_after_physicalcap.json \
  --output-csv /tmp/splineops_resize_scheduler_after_physicalcap.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --backend python \
  --threads default \
  --repeats 3 \
  --warmups 1 \
  --skip-checks \
  --output-json /tmp/splineops_resize_python_support_block256.json \
  --output-csv /tmp/splineops_resize_python_support_block256.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --backend native \
  --threads default \
  --repeats 8 \
  --warmups 2 \
  --batched-axis auto \
  --skip-checks \
  --output-json /tmp/splineops_resize_native_after_plan_row_runs_final_r8.json \
  --output-csv /tmp/splineops_resize_native_after_plan_row_runs_final_r8.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads default \
  --repeats 5 \
  --warmups 2 \
  --batched-axis auto \
  --batch-lines-sweep 8,16,32,64,128 \
  --skip-checks \
  --output-json /tmp/splineops_resize_batch_sweep_plan_cache.json \
  --output-csv /tmp/splineops_resize_batch_sweep_plan_cache.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads default \
  --repeats 5 \
  --warmups 2 \
  --batched-axis env \
  --skip-checks \
  --output-json /tmp/splineops_resize_auto_default_b64_seq.json \
  --output-csv /tmp/splineops_resize_auto_default_b64_seq.csv
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads default \
  --repeats 5 \
  --warmups 2 \
  --batched-axis off \
  --skip-checks \
  --output-json /tmp/splineops_resize_off_default_b64_seq.json \
  --output-csv /tmp/splineops_resize_off_default_b64_seq.csv
git diff --check
```

Observed results:

- After promoting auto routing to the unset default:
  - `py_compile`: clean
  - new batched/default-auto/off parity tests: `60 passed`
  - focused resize suite default/unset: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=off`: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=1`: `175 passed`
  - full suite default/unset: `429 passed`
  - explicit `auto` is covered by `test_batched_axis_auto_matches_default`
  - smoke benchmark reports unset/env mode as `batched_axis=<default:auto>`
- After the scheduler default-thread update:
  - focused resize suite default/unset: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=off`: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=1`: `175 passed`
  - full suite default/unset: `429 passed`
  - final standard thread sweep saved to
    `/tmp/splineops_resize_scheduler_after_physicalcap.{json,csv}`
  - on the local 16-logical/8-core CPU, default scheduling now tracks the
    8-thread/physical-core-style setting instead of the previous all-logical
    default, while explicit `LSRESIZE_NUM_THREADS=16` remains available
- After native plan caching and batch-size tuning:
  - focused resize suite default/unset: `175 passed`
  - focused resize suite with `LSRESIZE_PLAN_CACHE_SIZE=0`: `175 passed`
  - focused resize suite with `LSRESIZE_BATCHED_AXIS=1`: `175 passed`
  - full suite default/unset: `429 passed`
  - repeated-call microbenchmarks show cache wins mostly on small workloads:
    about `1.17x` for 1-D length 64 cubic, `1.08x` for 16x16 cubic,
    `1.06x` for 32x32 cubic, and `1.02x` for 64x64 cubic-antialiasing
    on this CPU; 64-256 squared cases were closer to `1.01x-1.04x`
  - standard default-thread batch sweep saved to
    `/tmp/splineops_resize_batch_sweep_plan_cache.{json,csv}`
  - batch size 64 won the most median-time cases in the standard sweep and
    gave about `1.07x` median / `1.10x` mean speedup versus batch size 32
    across standard cases
  - sequential standard auto-vs-off artifacts saved to
    `/tmp/splineops_resize_auto_default_b64_seq.{json,csv}` and
    `/tmp/splineops_resize_off_default_b64_seq.{json,csv}`
  - default-auto with batch size 64 was faster than explicit off on every 2-D
    standard case by median; 2-D median speedup averaged about `2.15x`, with
    the weakest 2-D case still about `1.57x`
- After Python fallback tuning:
  - `py_compile`: clean for the benchmark script, Python core, and runtime specs
  - smoke `--backend both`: native/Python parity passed for the smoke profile;
    max absolute differences stayed at `1.19e-7` for float32 cubic and
    `5.55e-16` for float64 cubic
  - focused resize suite default/unset: `175 passed`
  - focused resize suite with `SPLINEOPS_ACCEL=never`: `175 passed`
  - full suite default/unset: `429 passed`
  - standard Python benchmark saved to
    `/tmp/splineops_resize_python_support_block256.{json,csv}`
- After cached native row-run/interior-boundary split:
  - editable native rebuild: clean
  - direct batched-axis parity tests: `60 passed`
  - focused resize suite default/unset: `175 passed`
  - fair baseline from committed `HEAD` saved to
    `/tmp/splineops_resize_native_baseline_head_r8.{json,csv}`
  - final default-thread standard artifact saved to
    `/tmp/splineops_resize_native_after_plan_row_runs_final_r8.{json,csv}`
  - versus the committed baseline, standard default-thread medians improved in
    `15/18` cases overall and `14/16` 2-D cases
  - 2-D median speedup: `1.12x` mean / `1.12x` median, with noise/regression
    on small anisotropic float32/large anisotropic float64 medians but best-of
    timings still mostly improved
  - best-of timings improved in `15/18` cases overall, with `1.16x` mean /
    `1.12x` median speedup
- `git diff --check`: clean

Fresh local benchmark command:

```bash
.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis off \
  --skip-checks

.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis on \
  --batch-lines 64 \
  --skip-checks

.venv/bin/python scripts/benchmark_resize_native.py \
  --profile smoke \
  --threads 1,default \
  --repeats 2 \
  --warmups 1 \
  --batched-axis on \
  --batch-lines-sweep 8,16,32,64,128 \
  --skip-checks

.venv/bin/python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --batched-axis auto \
  --skip-checks
```

Selected best-of timings from the forced batched run:

| Case | Default path | Batched path | Speedup |
| --- | ---: | ---: | ---: |
| float32, 512x512, cubic-antialiasing, 1 thread | 31.42 ms | 13.22 ms | 2.38x |
| float32, 1024x1024, cubic-antialiasing, 1 thread | 108.73 ms | 52.82 ms | 2.06x |
| float64, 1024x1024, cubic-antialiasing, 1 thread | 110.56 ms | 49.70 ms | 2.22x |
| float32, 1024x1024, cubic-antialiasing, default threads | 14.67 ms | 8.22 ms | 1.78x |
| float64, 1024x1024, cubic-antialiasing, default threads | 11.29 ms | 9.17 ms | 1.23x |
| float32, 3-D cubic down, default threads | 6.15 ms | 7.21 ms | 0.85x |

Selected best-of timings from the auto-routing run:

| Case | Default path | Auto path | Speedup |
| --- | ---: | ---: | ---: |
| float32, 512x512, cubic down, 1 thread | 12.38 ms | 4.91 ms | 2.52x |
| float32, 512x512, cubic-antialiasing, 1 thread | 26.63 ms | 14.70 ms | 1.81x |
| float32, 1024x1024, cubic-antialiasing, 1 thread | 119.30 ms | 41.60 ms | 2.87x |
| float64, 1024x1024, cubic-antialiasing, 1 thread | 105.72 ms | 42.56 ms | 2.48x |
| float32, 1024x1024, cubic-antialiasing, default threads | 10.76 ms | 7.83 ms | 1.37x |
| float32, 3-D cubic down, default threads | 6.19 ms | 5.82 ms | skipped by auto, timing noise |

Selected best-of timings after the interior/boundary split and precomputed
source/sign table:

| Case | Default path | Auto path | Speedup |
| --- | ---: | ---: | ---: |
| float32, 512x512, cubic down, 1 thread | 12.03 ms | 7.06 ms | 1.70x |
| float32, 1024x1024, cubic down, 1 thread | 25.33 ms | 21.74 ms | 1.17x |
| float32, 1024x1024, cubic down, default threads | 6.98 ms | 4.63 ms | 1.51x |
| float32, 1024x1024, cubic-antialiasing, 1 thread | 80.82 ms | 42.71 ms | 1.89x |
| float32, 1024x1024, cubic-antialiasing, default threads | 11.08 ms | 6.78 ms | 1.63x |
| float64, 1024x1024, cubic-antialiasing, 1 thread | 110.87 ms | 36.20 ms | 3.06x |

The focused repeat benchmark showed high best-time noise for the default path,
so median times should be checked before overfitting the auto heuristic.

Selected best-of timings after the pure-interpolation specialization and
non-contiguous direct-write path:

| Case | Default path | Auto path | Speedup |
| --- | ---: | ---: | ---: |
| float32, 512x512, cubic down, 1 thread | 14.00 ms | 6.49 ms | 2.16x |
| float32, 1024x1024, cubic down, 1 thread | 54.71 ms | 24.28 ms | 2.25x |
| float32, 1024x1024, cubic down, default threads | 7.12 ms | 5.57 ms | 1.28x |
| float64, 1024x1024, cubic down, 1 thread | 51.40 ms | 23.27 ms | 2.21x |
| float32, 1024x1024, cubic-antialiasing, 1 thread | 119.39 ms | 41.86 ms | 2.85x |
| float32, 1024x1024, cubic-antialiasing, default threads | 12.41 ms | 7.44 ms | 1.67x |
| float64, 1024x1024, cubic-antialiasing, default threads | 14.48 ms | 7.50 ms | 1.93x |

Median-focused repeat timings from the same code path showed:

- float32 1024x1024 cubic down, 1 thread: `54.39 ms -> 23.77 ms`
  (`2.31x` median speedup).
- float64 1024x1024 cubic down, 1 thread: `50.92 ms -> 22.27 ms`
  (`2.29x` median speedup).
- float32 1024x1024 cubic-antialiasing, default threads:
  `13.96 ms -> 8.26 ms` (`1.69x` median speedup).

Smoke/standard `--batch-lines-sweep` observations:

- Best batch size is workload/thread dependent.
- After the no-extension-buffer split, the latest smoke sweep picked different
  winners across cases: 8 for small single-thread pure cubic, 16 for some
  default-thread pure/anisotropic cases, 32 for single-thread
  cubic-antialiasing/anisotropic cases, and 64 for default-thread
  cubic-antialiasing.
- After plan caching and the default-thread scheduler update, a saved standard
  sweep over 8, 16, 32, 64, and 128 picked 64 most often by median time.
  The native default is now 64. Re-run a saved target-CPU sweep before changing
  it again.

Python fallback observations:

- `SPLINEOPS_ACCUM=einsum` at the old block size 64 beat the previous
  `mulsum`/64 default in 17/18 standard-profile cases, with `1.06x` mean
  median speedup.
- Raising the Python block size to 256 was the larger win. `einsum`/256 beat
  `mulsum`/64 in 18/18 cases, with `1.67x` mean and `1.55x` median speedup.
- Block size 512 was only marginally faster on average than 256 and regressed
  several 2-D float32/linear cases, so 256 is the safer default.
- The support-wise streaming accumulator beat `einsum`/256 in 18/18 cases,
  with `1.12x` mean, `1.07x` median, `1.04x` minimum, and `1.32x` maximum
  speedup.
- Against the previous `mulsum`/64 default, `support`/256 beat 18/18 cases,
  with `1.85x` mean and `1.82x` median speedup.
- Saved Python artifacts:
  - `/tmp/splineops_resize_python_mulsum_block64.{json,csv}`
  - `/tmp/splineops_resize_python_einsum_block256.{json,csv}`
  - `/tmp/splineops_resize_python_support_block256.{json,csv}`

Important caveat:

- `LSRESIZE_BATCHED_AXIS` is now default-auto. Unset and `auto` both route only
  sufficiently large 2-D axis passes through batching and leave 3-D on the
  original path because 3-D forced-batched timings remain mixed.
- Use `LSRESIZE_BATCHED_AXIS=off` to force the original line-by-line path for
  debugging, comparisons, and conservative deployments.
- `LSRESIZE_BATCHED_AXIS=1` remains useful for stress/parity testing the
  batched implementation directly.
- Default thread scheduling is now conservative on SMT-style machines. Use
  `LSRESIZE_NUM_THREADS=<n>` to force a particular thread count for benchmark
  sweeps or deployments that benefit from all logical CPUs.
- Native plan caching is on by default with capacity 32. Use
  `LSRESIZE_PLAN_CACHE_SIZE=0` to disable it when measuring cold-plan behavior.

Suggested next steps:

1. Run a saved standard/full `--batch-lines-sweep` artifact on additional target
   CPUs before changing the default batch size from 64.
2. Run default-auto versus `LSRESIZE_BATCHED_AXIS=off` saved artifacts on
   additional target CPUs before tuning the auto heuristic further.
3. Run a standard/full saved thread sweep on target hardware before changing
   the default scheduler constants.
4. Add template-specialized native preset kernels for the common methods
   (`linear`, `cubic`, `linear-antialiasing`, `cubic-antialiasing`).
5. Consider exposing an explicit reusable native `ResizePlan` object if repeated
   same-shape workloads remain important; the private cache is the low-risk
   first step.
6. Consider fused resize/permutation writes for N-D cases where strided passes
   dominate.

## Handoff: 2026-06-11

Work completed today:

- Fixed the finite-size constant-preservation bug in the spline IIR prefilter.
  The fix uses the exact mirror-boundary finite-length causal initializer with
  the correct reflected term `z^(2N-2-n)`.
- Added regressions for short constant signals and end-to-end constant resizing.
- Added a native scheduling optimization:
  - process shrinking axes before non-shrinking axes
  - process stronger shrink factors earlier
  - skip pure-interpolation identity axes at the binding layer
- Added a reproducible benchmark harness:
  `scripts/benchmark_resize_native.py`.
- Added the first batched native axis-kernel prototype behind a feature flag:
  `LSRESIZE_BATCHED_AXIS=1`.

Current changed files to expect in the working tree:

- `cpp/lsresize/src/resize_nd.cpp`
  - contains the feature-flagged batched pure-interpolation axis path
  - note: this 2026-06-11 scope was superseded on 2026-06-12 with projection
    and antialiasing support
- `tests/test_02_03_resize_cpp.py`
  - includes parity tests comparing default native output against
    `LSRESIZE_BATCHED_AXIS=1`
- `scripts/benchmark_resize_native.py`
  - new benchmark script
- `scripts/resize_optimization_notes.md`
  - this handoff and roadmap

Validation already run:

```bash
python -m py_compile scripts/benchmark_resize_native.py
git diff --check
pytest -q
LSRESIZE_BATCHED_AXIS=1 pytest -q
LSRESIZE_BATCHED_AXIS=1 pytest -q tests/test_02_02_resize.py tests/test_02_03_resize_cpp.py
```

Observed results:

- Full suite default: `387 passed`
- Full suite with `LSRESIZE_BATCHED_AXIS=1`: `387 passed`
- Focused suite with `LSRESIZE_BATCHED_AXIS=1`: `133 passed`
- `git diff --check`: clean
- `py_compile`: clean

Benchmark artifacts written under `/tmp`:

- `/tmp/splineops_resize_baseline_standard.json`
- `/tmp/splineops_resize_baseline_standard.csv`
- `/tmp/splineops_resize_batched_standard.json`
- `/tmp/splineops_resize_batched_standard.csv`

Important caveat as of 2026-06-11:

- The batched path is promising for pure cubic interpolation, especially larger
  2-D cases, but it is disabled by default. Do not make it default yet.
- This caveat was superseded on 2026-06-12: antialiasing/projection is now
  covered by the feature-flagged batched path, but the path remains disabled by
  default pending routing/tuning.

Suggested next steps tomorrow:

1. Re-run the standard benchmark once to re-establish local timing stability:

   ```bash
   python scripts/benchmark_resize_native.py \
     --profile standard \
     --threads 1,2,4,8,16,default \
     --repeats 5 \
     --warmups 2 \
     --batched-axis off

   python scripts/benchmark_resize_native.py \
     --profile standard \
     --threads 1,2,4,8,16,default \
     --repeats 5 \
     --warmups 2 \
     --batched-axis on \
     --batch-lines 32
   ```

2. Tune `LSRESIZE_BATCH_LINES`; early runs suggested 32 is good, but 8/16/64
   should be compared systematically.
3. If batched interpolation and antialiasing remain consistently faster in a
   subset of cases, add an automatic routing condition for those cases.

## Current Baseline

The current native implementation is a separable N-D resize:

1. Build a 1-D plan for one axis.
2. Process each independent 1-D line.
3. Ping-pong intermediate arrays between axis passes.
4. Use double precision internally, including for float32 storage.

Useful existing strengths:

- Per-axis weights are precomputed once per axis pass.
- Work buffers are reused per worker.
- Small spline dot products are already unrolled.
- C++ and Python fallback paths agree closely.
- Constant-preservation regressions now cover short finite signals.

Recent safe native scheduling change:

- Shrinking axes are processed before non-shrinking axes.
- Stronger shrink factors are processed earlier.
- Pure interpolation identity axes are skipped at the binding layer.

Local rough timing deltas, default threading:

| Case | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| float32, 1024x1024, cubic, zoom 0.37 | 9.76 ms | 7.32 ms | 1.33x |
| float64, 1024x1024, cubic, zoom 0.37 | 9.69 ms | 6.11 ms | 1.59x |
| float64, 1024x1024, cubic-antialiasing, zoom 0.37 | 13.70 ms | 10.32 ms | 1.33x |
| float32, 512x512, cubic-antialiasing, zoom 0.37 | 4.32 ms | 3.91 ms | 1.10x |

These numbers are local best-of timings, not a formal benchmark suite.

## Rejected Experiment

A 2-D transpose fast path was tested:

1. Resize rows contiguously.
2. Transpose the intermediate.
3. Resize original columns as contiguous rows.
4. Transpose back.

This was not kept. It helped some single-thread antialiasing cases but was not
consistently better under default threading. The extra memory movement became a
bottleneck.

## Suggested Next Steps

### 1. Formal Benchmark Harness

Create a small reproducible benchmark script before deeper optimization.

Initial script: `scripts/benchmark_resize_native.py`.

Useful commands:

```bash
python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,2,4,8,16,default \
  --repeats 5 \
  --warmups 2 \
  --output-json /tmp/splineops_resize_baseline_standard.json \
  --output-csv /tmp/splineops_resize_baseline_standard.csv

python scripts/benchmark_resize_native.py \
  --profile standard \
  --threads 1,default \
  --repeats 3 \
  --warmups 1 \
  --skip-checks \
  --batched-axis on \
  --batch-lines 32 \
  --output-json /tmp/splineops_resize_batched_standard.json \
  --output-csv /tmp/splineops_resize_batched_standard.csv
```

Requirements:

- Fixed inputs and random seeds.
- Shapes: 512x512, 1024x1024, 2048x2048, and representative 3-D volumes.
- Methods: cubic, cubic-antialiasing, linear-antialiasing.
- Dtypes: float32 and float64.
- Zooms: uniform downsampling, anisotropic downsampling, upsampling, identity on
  one axis.
- Thread counts: 1, 2, 4, 8, 16, default.
- Correctness checks against the Python fallback for selected cases.

This avoids optimizing against noisy ad hoc timings.

### 2. Native Plan Cache

The C++ binding currently rebuilds per-axis plans for each call. Repeated
same-shape/same-zoom workloads, such as video frames or registration loops, can
reuse the same plans.

Options:

- Add a small process-local LRU cache keyed by realized grid, degrees, and
  shift.
- Or expose a Python-facing `ResizePlan` object:
  `plan = ResizePlan(input_shape, output_shape, method); plan.apply(frame)`.

The explicit plan object is better long-term. A private LRU cache is easier to
add first.

### 3. Threading Heuristic

Local measurements showed oversubscription:

- 1024x1024 cubic-antialiasing, float32:
  - 1 thread: about 119 ms
  - 8 threads: about 17 ms
  - 16 threads: about 11 ms
  - 32 requested threads: slower on the test machine

Possible improvements:

- Cap default worker count more conservatively.
- Use workload size to choose fewer threads for small images.
- Add a small environment-controlled autotune mode.
- Eventually replace per-axis thread creation with a persistent thread pool.

The environment override `LSRESIZE_NUM_THREADS` should remain available.

### 4. Batched Native Axis Kernel

Status: implemented for pure interpolation, projection, and antialiasing.

The original native code processed one 1-D line at a time. But all lines for an
axis share the same plan and weights. The batched C++ kernel now processes a
block of lines together:

```text
for output index l:
    y[:, l] = w0 * x[:, k0] + w1 * x[:, k1] + ...
```

Benefits:

- Reuses weights across many lines.
- Vectorizes across lines instead of only across the tiny spline support.
- Reduces per-line overhead.
- More closely matches the vectorized strategy already used in the Python
  fallback, but without NumPy gather overhead.

This should be exact, not an approximation.

Implementation status:

- Feature flag: `LSRESIZE_BATCHED_AXIS=1`.
- Batch size override: `LSRESIZE_BATCH_LINES=<positive integer>`.
- Default routing: `LSRESIZE_BATCHED_AXIS` unset means conservative `auto`.
- Current scope: pure interpolation plus projection/antialiasing.
- Tests compare flagged native output against default native output for
  interpolation, antialiasing presets, and equal-degree projection on 2-D and
  3-D cases.

Early local timings with `LSRESIZE_BATCHED_AXIS=1` and batch size 32:

| Case | Default path | Batched path | Notes |
| --- | ---: | ---: | --- |
| float32, 1024x1024, cubic, 1 thread | 51.1 ms | 30.4 ms | side-by-side best-of |
| float32, 1024x1024, cubic, default threads | 7.1 ms | 5.3 ms | side-by-side best-of |
| float64, 512x512, cubic anisotropic, 1 thread | 6.6 ms | 3.3 ms | side-by-side best-of |
| float64, 512x512, cubic anisotropic, default threads | 1.5 ms | 1.2 ms | side-by-side best-of |

Later local timings after projection support was added are recorded in the
2026-06-12 handoff above.

### 5. Interior/Boundary Split

Status: implemented inside the batched native path.

Most output samples are far from mirrored boundaries. The hot loop should not
pay boundary-extension costs for those samples.

Implemented structure:

- Fast interior kernel:
  - no mirror handling
  - no extension buffer
  - direct weighted reads
  - contiguous row runs precomputed in `Plan1D`
- Slow boundary kernel:
  - exact mirror or antisymmetric boundary behavior
  - cached per-weight source/sign mapping in `Plan1D`

For large images, most rows hit the fast interior path. The current
implementation keeps the row-run metadata in the private plan cache; an explicit
public `ResizePlan` could reuse the same metadata across user-managed workloads.

### 6. Template-Specialized Presets

The generic degree triple is useful, but common presets can be specialized:

- `linear`
- `cubic`
- `linear-antialiasing`
- `cubic-antialiasing`

Compile-time specialization can remove degree branches, fix support sizes, and
make the compiler more aggressive in unrolling and inlining.

This should come after the benchmark harness, since specialized kernels are easy
to make faster in one case and slower in another.

### 7. Optional Float32 Internal Mode

The current native path uses double internally even for float32 storage. That is
robust, but image workloads may not need double precision.

Possible API:

```python
resize(x, zoom_factors=z, method="cubic-antialiasing", precision="float32")
resize(x, zoom_factors=z, method="cubic-antialiasing", precision="float64")
```

Risks:

- Different numerical results.
- Potentially worse constant preservation if not implemented carefully.
- More testing required for small images and high-frequency inputs.

This has high performance upside but should be opt-in at first.

### 8. Output and Temporary Memory Strategy

Current ping-pong intermediates are simple and safe. For large N-D data, memory
traffic may dominate.

Ideas:

- Process smaller tiles for large arrays.
- Fuse adjacent axis passes for small output tiles where practical.
- Reuse native temporary buffers through a plan object.
- Avoid reallocating ping-pong buffers in repeated calls.

This is probably more valuable for video, volumes, and batch processing than for
single small 2-D images.

## Recommended Priority

1. Add specialized native preset kernels.
2. Add opt-in float32 internal mode.
3. Expose an explicit reusable native `ResizePlan` if repeated same-shape
   workloads are a priority.
4. Explore tiled/fused memory strategies for large N-D workloads.
5. Revisit the auto-routing/thread heuristics with artifacts from additional
   target CPUs.

The batched native axis kernel and interior/boundary split are now implemented.
The next highest-upside exact redesign is method-specialized native kernels,
followed by an opt-in float32 internal mode for image-oriented workloads.

## Handoff: 2026-06-16 Follow-Up

Accepted changes after the latest committed baseline:

- Refactored the col-major recursive interpolation prefilter so the double and
  float32 paths share one pole-application helper instead of duplicating the
  causal/anti-causal loops.
- Tightened the initial-causal scalar helper to walk each line with pointer
  increments instead of recomputing `n * B + b` indexing in the inner loop.
- Added `--cases` to `scripts/benchmark_resize_native.py` so targeted native
  sweeps can use the same case selection style as `benchmark_resize_ab.py`.
- Added `scripts/summarize_resize_benchmarks.py` for raw CSV artifacts:
  `native`, `libraries`, `legacy`, and `ab` summaries.
- Added a guarded direct-scatter path for large 3-D axis-1 pure
  quadratic/cubic interpolation passes. It evaluates the same spline row plan
  directly into the strided destination when the pass has at least about
  `1M` axis outputs and the selected worker pool is at most 8 threads. It
  remains A/B controllable with `LSRESIZE_3D_AXIS1_DIRECT_SCATTER=0`.

Measured but rejected on this machine:

- Batched initial-causal setup:
  `/tmp/splineops_ab_batched_initial_causal_20260616.csv`
  - median `0.987x`, mean `0.979x`, 10 wins and 20 losses across 48 rows
  - especially weak at 8 threads: median `0.909x`
- Degree-specific one-pole prefilter setup:
  `/tmp/splineops_ab_specialized_prefilters_20260616.csv`
  - median `0.981x`, mean `0.977x`, 11 wins and 21 losses across 48 rows
- Batch-line retunes:
  - `64 -> 32`: median `1.024x`, but 13 losses and threaded/default regressions
  - `64 -> 128`: median `0.946x`, 20 losses

New A/B artifact:

- Thresholded 3-D axis-1 direct scatter:
  `/tmp/splineops_ab_direct_scatter_threadguard_20260616.csv`
  - median `1.079x`, mean `1.078x`, 9 wins and 0 losses across the focused
    3-D cubic sweep
  - first unthresholded attempt was rejected as a blanket policy because the
    small anisotropic case regressed under explicit thread counts; a later
    threshold-only attempt exposed thread-pool sensitivity on the active large
    anisotropic case, so the final router also checks selected worker count

Fresh final artifacts after the direct-scatter pass:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_direct_scatter_20260616.csv`
  - 43 overlaps, median native/Python speedup `24.62x`, mean `27.83x`
  - best native thread counts: `1:8`, `8:17`, `default:18`
  - 3-D median native/Python speedup `20.67x`
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_direct_scatter_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows are all slower
  - skimage faster in `0/21`
  - OpenCV faster in `14/18`, but with different coordinate/AA semantics
  - Torch faster in `8/19`; exact-ish Torch rows are all slower
- Library full comparison, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_direct_scatter_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows are all slower
  - skimage faster in `0/21`
  - OpenCV faster in `17/18`, with different coordinate/AA semantics
  - Torch faster in `10/19`; exact-ish Torch rows faster in `3/8`
- Legacy Java 2-D reference rerun:
  `/tmp/splineops_legacy_java_full_direct_scatter_20260616.csv`
  - 14 overlaps against current splineops default library artifact, median
    speedup `9.18x`, mean `12.93x`

Validation:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  scripts/benchmark_resize_ab.py \
  scripts/summarize_resize_benchmarks.py \
  src/splineops/utils/specs.py
```

Latest validation result: full suite `519 passed`; script py-compile and
`git diff --check` clean.

## Handoff: 2026-06-16 Fused Gather-Prefilter Scale

Accepted change after the direct-scatter pass:

- Fused the interpolation-prefilter normalization factor into the batched gather
  for default double-internal interpolation/projection and default float32 pure
  interpolation. The recursive pole pass is still identical to the previous
  path; the optimization removes the separate full coefficient-block scaling
  pass. It is controllable with `LSRESIZE_GATHER_PREFILTER_SCALE=0`.
- Added standalone `apply_interpolation_poles_colmajor*` helpers so the fused
  path can apply only the recursive spline poles after gathering pre-scaled
  coefficients.
- Kept float32 projection on the existing path because its DC centering and
  restore logic needs separate accounting.
- Guarded the fused path with `N > 1` to preserve the previous short-line
  behavior exactly.

Measured but rejected on this machine:

- Unit-stride gather specialization:
  `/tmp/splineops_ab_unit_stride_gather_20260616.csv`
  - median `0.985x`, mean `0.971x`, 3 wins and 8 losses
  - removed from the implementation rather than keeping a default-off dead knob
- 3-D batch-line retune:
  `/tmp/splineops_batch_lines_3d_next_20260616.csv`
  - no single tested larger batch size was safer than the current default `64`
    across the sampled 3-D cubic rows

New A/B artifact:

- Gather-prefilter scale fusion:
  `/tmp/splineops_ab_gather_prefilter_scale_final_20260616.csv`
  - median `1.119x`, mean `1.155x`, 15 wins and 2 losses across 18 focused
    cubic/projection rows, no failed checks
  - default scheduler: median `1.124x`, mean `1.217x`, 6 wins and 0 losses
  - single-thread: median `1.160x`, mean `1.141x`, 5 wins and 0 losses
  - forced 8 threads: median `1.069x`, mean `1.107x`, 4 wins and 2 losses

Fresh final artifacts after the fused gather-prefilter scale pass:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_gather_prefilter_scale_20260616.csv`
  - 43 overlaps, median native/Python speedup `25.11x`, mean `29.16x`
  - range `7.67x` to `90.82x`
  - 3-D median native/Python speedup `25.10x`
  - best native thread counts: `1:8`, `8:23`, `default:12`
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_gather_prefilter_scale_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows are all slower
  - skimage faster in `0/21`
  - OpenCV faster in `13/18`, but with different coordinate/AA semantics
  - Torch faster in `8/19`; exact-ish Torch rows are all slower
- Library full comparison, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_gather_prefilter_scale_20260616.csv`
  - SciPy faster in `0/21`; exact-ish SciPy rows are all slower
  - skimage faster in `0/21`
  - OpenCV faster in `17/18`, with different coordinate/AA semantics
  - Torch faster in `10/19`; exact-ish Torch rows faster in `3/8`
- Legacy Java 2-D reference rerun:
  `/tmp/splineops_legacy_java_full_gather_prefilter_scale_20260616.csv`
  - 14 overlaps against current splineops default library artifact, median
    speedup `8.44x`, mean `11.04x`

Validation:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py -k 'gather_prefilter_scale or row_gather or direct_scatter'
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  scripts/benchmark_resize_ab.py \
  scripts/summarize_resize_benchmarks.py
git diff --check
```

Latest validation result: focused parity `18 passed`, full suite
`527 passed`; script py-compile and `git diff --check` clean.

## Handoff: 2026-06-16 Row-Wise Prefilter Setup And Projection Batches

Accepted changes after the fused gather-prefilter scale pass:

- Added a default-on row-wise finite-horizon initial-causal setup for the
  float32 col-major interpolation prefilter. This keeps the same recursive
  spline pole math but initializes the first coefficient row by walking
  coefficient rows contiguously instead of striding one line at a time. It is
  controllable with `LSRESIZE_ROWWISE_INITIAL_CAUSAL=0`.
- Kept the double-internal prefilter on the scalar initializer after A/B showed
  the row-wise version was useful mainly for float32 pure interpolation and
  noisy for double-internal projection paths.
- Added a guarded 2-D projection batch-size heuristic behind
  `LSRESIZE_2D_PROJECTION_BATCH_TUNE=0`: 2-D linear antialiasing uses batch
  `32`, and large 2-D cubic antialiasing uses batch `96`; pure interpolation
  and 3-D cubic behavior are unchanged.

Fresh profiles after the previous commit:

- `/tmp/splineops_profile_post_fused_scale_2d_cubic_down_2048_f32_thr1_20260616.log`
- `/tmp/splineops_profile_post_fused_scale_2d_cubic_aa_down_2048_f32_thr1_20260616.log`
- `/tmp/splineops_profile_post_fused_scale_3d_cubic_down_large_f32_thr1_20260616.log`
- `/tmp/splineops_profile_post_fused_scale_3d_cubic_aniso_large_f32_thr1_20260616.log`

New A/B artifacts:

- Projection batch tuning:
  `/tmp/splineops_ab_projection_batch_tune_20260616.csv`
  - median `1.033x`, mean `1.041x`, 16 wins and 7 losses across 30 2-D
    projection rows, no failed checks
  - default scheduler: median `1.053x`, mean `1.047x`
  - linear-antialiasing rows were stronger than cubic-antialiasing rows:
    median `1.093x` versus `1.005x`
- Float32 row-wise initial-causal setup:
  `/tmp/splineops_ab_rowwise_initial_causal_f32_20260616.csv`
  - median `1.093x`, mean `1.144x`, 8 wins and 1 loss across 12 focused
    float32 cubic rows, no failed checks
  - default scheduler: median `1.177x`, mean `1.203x`, 4 wins and 0 losses

Fresh final artifacts after this pass:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_rowwise_projection_batch_20260616.csv`
  - 43 overlaps, median native/Python speedup `27.32x`, mean `28.39x`
  - antialiasing median native/Python speedup `20.07x`
  - best native thread counts: `1:8`, `8:22`, `default:13`
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_rowwise_projection_batch_20260616.csv`
  - SciPy faster in `0/21`; exact-ish SciPy rows are all slower
  - skimage faster in `0/21`
  - OpenCV faster in `15/18`, with different coordinate/AA semantics
  - Torch faster in `7/19`; exact-ish Torch rows are all slower
- Library full comparison, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_rowwise_projection_batch_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows are all slower
  - skimage faster in `0/21`
  - OpenCV faster in `16/18`, with different coordinate/AA semantics
  - Torch faster in `11/19`; exact-ish Torch rows faster in `4/8`
- Legacy Java 2-D reference rerun:
  `/tmp/splineops_legacy_java_full_rowwise_projection_batch_20260616.csv`
  - 14 overlaps against current splineops default library artifact, median
    speedup `10.15x`, mean `12.44x`

Validation:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py -k 'rowwise_initial_causal or projection_batch_tune or gather_prefilter_scale or row_gather or direct_scatter'
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  scripts/benchmark_resize_ab.py \
  scripts/summarize_resize_benchmarks.py
git diff --check
```

Latest validation result: focused parity `28 passed`, full suite
`537 passed`; script py-compile and `git diff --check` clean.

## Handoff: 2026-06-16 Strided-Offset Gather

Accepted changes after the row-wise initial-causal/projection-batch pass:

- Added `LSRESIZE_STRIDED_OFFSET_GATHER`, default-on, scoped to batched
  float32 pure interpolation gathers where batch line offsets form a constant
  stride. It keeps Arrate's least-squares/spline math unchanged; only the input
  coefficient block gather is specialized.
- The first global version was A/B tested and rejected for projection/double
  paths because antialiasing rows regressed. The committed route therefore
  leaves projection/antialiasing and double-internal gather on the previous
  offset-array code path.
- Added `scripts/summarize_resize_benchmarks.py compare` for artifact-to-
  artifact CSV comparisons. It compares rows by inferred or explicit keys,
  reports aggregate baseline/current speedups, and lists largest wins/losses.

Fresh profiles before this pass:

- `/tmp/splineops_profile_2d_cubic_down_2048_f32_thr1_20260616.csv`
  - profile summary: gather `39.14%`, prefilter `27.35%`
- `/tmp/splineops_profile_2d_cubic_aa_down_2048_f32_thr1_20260616.csv`
  - profile summary: gather `28.90%`, integrate `17.64%`, prefilter `12.69%`
- `/tmp/splineops_profile_3d_cubic_down_large_f32_thr1_20260616.csv`
  - profile summary: gather `49.89%`, prefilter `24.15%`
- `/tmp/splineops_profile_3d_cubic_aniso_large_f32_thr1_20260616.csv`
  - profile summary: gather `37.62%`, accumulate-scatter `37.96%`

New A/B artifact:

- Strided-offset gather:
  `/tmp/splineops_ab_strided_offset_gather_routed_20260616.csv`
  - median `1.021x`, mean `1.052x`, 61 wins and 33 losses across the full
    native profile, no failed checks
  - key intended wins:
    `3d_cubic_down_large_f32` up to `1.29x`,
    `3d_cubic_aniso_large_f32` up to `1.37x`, and
    `2d_cubic_down_2048_float32` up to `1.19x`

Fresh final artifacts after this pass:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_strided_offset_gather_20260616.csv`
  - 43 overlaps, median native/Python speedup `23.23x`, mean `26.85x`
  - 3-D median native/Python speedup `22.89x`
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_strided_offset_gather_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `14/18`, with different coordinate/AA semantics
  - Torch faster in `7/19`; exact-ish Torch rows all slower
- Library full comparison, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_strided_offset_gather_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `16/18`, with different coordinate/AA semantics
  - Torch faster in `10/19`; exact-ish Torch rows faster in `3/8`
- Legacy Java 2-D reference:
  `/tmp/splineops_legacy_java_full_strided_offset_gather_20260616.csv`
  - 14 overlaps against current splineops default library artifact, median
    speedup `10.02x`, mean `12.08x`

Validation:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py -k 'strided_offset_gather or row_gather or gather_prefilter_scale or rowwise_initial_causal or projection_batch_tune or direct_scatter'
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  scripts/benchmark_resize_ab.py \
  scripts/summarize_resize_benchmarks.py
git diff --check
```

Latest validation result: focused parity `34 passed`, full suite
`543 passed`; script py-compile and `git diff --check` clean.

## Handoff: 2026-06-16 Row-Wise Finite Causal Initializer

Accepted changes after the strided-offset gather pass:

- Extended `LSRESIZE_ROWWISE_INITIAL_CAUSAL` to double-internal col-major
  interpolation prefilters.
- Extended the same row-wise setup to the exact finite-length mirror
  initializer used when the causal horizon reaches the whole axis. This covers
  short axes for both float32 and double internal paths.
- Preserved the scalar reference path with `LSRESIZE_ROWWISE_INITIAL_CAUSAL=0`.
  The optimized path changes memory traversal, not the least-squares projection
  math or spline pole recursion.

Rejected experiments in this pass:

- `LSRESIZE_PRECISION=float32`-style projection internals as an automatic
  default. The selected projection-heavy A/B was fast, with about `1.234x`
  median speedup, but failed output checks on 9 rows and reached max drift
  around `1.75e-3` on large cubic-antialiasing output.
- Fusing the length-2 sampling FIR directly into output scatter. The temporary
  implementation matched the separate sampling path but was slower:
  median `0.876x`, mean `0.894x`, with 1 win and 15 losses. It was removed
  before finalizing this patch.

New A/B artifact:

- Row-wise truncated plus finite causal setup:
  `/tmp/splineops_ab_rowwise_initial_causal_finite_clean_20260616.csv`
  - median `1.028x`, mean `1.054x`, 64 wins and 24 losses across the full
    native profile, no failed checks
  - thread breakdown: `1` median `1.020x`, `8` median `1.050x`, default
    median `1.025x`
  - method breakdown: cubic median `1.054x`, cubic-antialiasing median
    `1.016x`, linear-antialiasing median `1.036x`

Fresh final artifacts after this pass:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_rowwise_finite_20260616.csv`
  - 43 overlaps, median native/Python speedup `24.86x`, mean `27.42x`
  - best native thread counts: `1:12`, `8:18`, `default:13`
  - cubic median speedup `25.18x`, antialiasing median `16.22x`, 3-D median
    `24.86x`
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_rowwise_finite_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `14/18`, with different coordinate/AA semantics
  - Torch faster in `8/19`; exact-ish Torch rows all slower
- Library full comparison, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_rowwise_finite_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `16/18`, with different coordinate/AA semantics
  - Torch faster in `12/19`; exact-ish Torch faster in `5/8`
- Legacy Java 2-D reference:
  `/tmp/splineops_legacy_java_full_rowwise_finite_20260616.csv`
  - 14 overlaps against current splineops default library artifact, median
    speedup `10.82x`, mean `12.60x`

Validation so far:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py -k 'rowwise_initial_causal or projection_batch_tune or gather_prefilter_scale or row_gather or strided_offset_gather'
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  scripts/benchmark_resize_ab.py \
  scripts/summarize_resize_benchmarks.py
git diff --check
```

Latest validation result: focused parity `30 passed`, full suite `543 passed`;
editable rebuild, script py-compile, and `git diff --check` clean.

## Handoff: 2026-06-16 Double-Internal Strided Gather

Accepted change after the row-wise finite-causal pass:

- Extended `LSRESIZE_STRIDED_OFFSET_GATHER` to double-internal pure
  interpolation gathers for large non-upsampling axes. The route is deliberately
  narrow: 2-D double paths require `max(input_shape) >= 1024`, and non-2-D
  double paths require a large axis. This keeps small images and upsampling on
  the previous offset-array gather.
- The feature changes address generation during input gather only. It does not
  alter the least-squares projection, spline poles, boundary handling,
  correction filters, or accumulation weights.

Rejected experiment in this pass:

- Compact fixed-support row maps were tried in both vector-of-fields and
  interleaved row layouts. They matched outputs, but A/B was flat to negative:
  first layout median `0.998x`, interleaved median `1.000x`, and interleaved
  explicit 8-thread median `0.943x`. The code was removed to avoid extra plan
  memory and branch surface in the hot accumulators.

New A/B artifacts:

- Active large double 2-D gather:
  `/tmp/splineops_ab_strided_offset_gather_double_active_large2d_20260616.csv`
  - median `1.066x`, mean `1.061x`, 6 wins and 2 losses, no failed checks
  - default scheduler median `1.088x`, 3 wins and 0 losses
- 2048 double repeat:
  `/tmp/splineops_ab_strided_offset_gather_double_2048_repeat_20260616.csv`
  - median `1.083x`, mean `1.088x`, 3 wins and 0 losses

Fresh final artifacts after this pass:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_double_strided_gather_20260616.csv`
  - 43 overlaps, median native/Python speedup `22.25x`, mean `27.34x`
  - best native thread counts: `1:10`, `8:16`, `default:17`
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_double_strided_gather_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `14/18`, with different coordinate/AA semantics
  - Torch faster in `8/19`; exact-ish Torch rows all slower
- Library full comparison, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_double_strided_gather_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `17/18`, with different coordinate/AA semantics
  - Torch faster in `12/19`; exact-ish Torch faster in `4/8`
- Legacy Java 2-D reference:
  `/tmp/splineops_legacy_java_full_double_strided_gather_20260616.csv`
  - 14 overlaps against current splineops default library artifact, median
    speedup `10.66x`, mean `13.73x`

Validation so far:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py -k 'strided_offset_gather or row_gather or gather_prefilter_scale or rowwise_initial_causal or projection_batch_tune or direct_scatter'
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  scripts/benchmark_resize_ab.py \
  scripts/summarize_resize_benchmarks.py
git diff --check
```

Latest validation result: focused parity `34 passed`, full suite `543 passed`;
script py-compile and `git diff --check` clean.

## Handoff: 2026-06-16 Batch Tune V2 and 2-D Axis-0 Direct Scatter

Accepted changes after the double-internal strided gather pass:

- Added `LSRESIZE_2D_AXIS0_DIRECT_SCATTER`, default-on only for large 2-D pure
  quadratic/cubic upsampling on axis 0. The first broader gate covering
  downsample/aniso rows was noisy and produced losses, so the committed gate
  requires `p.zoom > 1.0 + 1e-12` and at least `250000` axis-output samples.
- Added `LSRESIZE_BATCH_TUNE_V2`, default-on for a very narrow 2-D pure
  quadratic/cubic downsampling batch adjustment. Threaded/default paths keep
  the previous 16-line batch. Explicit single-thread f32 rows and large
  double rows use 24 lines.
- Both changes preserve Arrate's least-squares projection method. They only
  alter batching and whether a non-contiguous output axis writes through a
  temporary accumulator row.

Rejected experiment in this pass:

- Cached f32 plan weights were tested behind `LSRESIZE_F32_PLAN_WEIGHTS`.
  Output parity was exact, but performance was not stable: the focused outlier
  repeat reported median `0.980x`, mean `0.978x`, with 5 losses and 1 win. The
  experiment was removed rather than left as dead plan memory and dispatch
  surface.

New A/B artifacts:

- 2-D axis-0 direct scatter, affected cubic upsample rows:
  `/tmp/splineops_ab_2d_axis0_direct_scatter_up_final_20260616.csv`
  - median `1.078x`, mean `1.076x`, 6 wins and 0 losses, no failed checks
  - thread breakdown: `1` median `1.094x`, `8` median `1.073x`, default
    median `1.062x`
- Batch tune v2, affected single-thread cubic downsample rows:
  `/tmp/splineops_ab_batch_tune_v2_single_thread_final_20260616.csv`
  - median `1.036x`, mean `1.030x`, 2 wins and 0 losses, no failed checks
  - f32 downsample rows were the useful wins: `1.062x` to `1.068x`

Fresh final artifacts after this pass:

- Native full sweep, current defaults:
  `/tmp/splineops_native_full_current_batch_axis0_20260616.csv`
- Native full sweep with this pass disabled:
  `/tmp/splineops_native_full_without_batch_axis0_20260616.csv`
  - direct same-build A/B remains the better source for claims; the
    two-separate-run aggregate CSV comparison is visibly noisy on unaffected
    linear and 3-D rows
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_batch_axis0_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `12/18`, with different coordinate/AA semantics
  - Torch faster in `8/19`; exact-ish Torch rows all slower
- Library full comparison, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_batch_axis0_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `17/18`, with different coordinate/AA semantics
  - Torch faster in `11/19`; exact-ish Torch faster in `4/8`
- Legacy Java 2-D reference:
  `/tmp/splineops_legacy_java_full_batch_axis0_20260616.csv`
  - 14 overlaps against current default-scheduler splineops artifact, median
    speedup `7.94x`, mean `11.25x`

Validation so far:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py -k 'batch_tune or direct_scatter or strided_offset_gather or row_gather or gather_prefilter_scale or rowwise_initial_causal or projection_batch_tune'
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  scripts/benchmark_resize_ab.py \
  scripts/summarize_resize_benchmarks.py
git diff --check
```

Latest validation result: focused parity `40 passed`, full suite
`549 passed`; script py-compile and `git diff --check` clean.

## Handoff: 2026-06-16 Projection Batch Single-Thread Tuning

Accepted change after the batch/direct-scatter pass:

- Large 2-D cubic-antialiasing projection passes now use 32-line batches when
  `LSRESIZE_NUM_THREADS=1` is set explicitly. The previous 96-line batch
  remains the policy for default and other threaded scheduling. This changes
  only workspace batch size; Arrate's projection, spline recursion, correction
  filters, and weights are unchanged.

Focused accepted A/B:

- `/tmp/splineops_ab_cubic_aa_batch96_vs32_single_20260616.csv`
  - old forced 96-line batch versus 32-line batch, explicit single-thread
  - median `1.072x`, mean `1.096x`, 3 wins and 0 losses, no failed checks
  - largest row: `2d_cubic_aa_down_2048_float64` improved `1.24x`

Rejected experiments in this pass:

- 2-D axis-1 direct scatter for pure upsampling:
  `/tmp/splineops_ab_2d_axis1_direct_scatter_up_20260616.csv`, median
  `0.873x`, mean `0.844x`, 0 wins and 6 losses.
- Blocked axis-contiguous scatter:
  `/tmp/splineops_ab_blocked_axis_contig_scatter_cubic_20260616.csv` had
  median `1.020x`, but 7 losses; tile-16 and transpose variants were also
  mixed or negative.
- Tiled last-axis gather:
  `/tmp/splineops_ab_tiled_axis_contig_gather_cubic_20260616.csv`, median
  `0.971x`, mean `0.945x`, 15 losses.
- Run-length fixed-window accumulation:
  `/tmp/splineops_ab_run_length_accumulate_cubic_aa_20260616.csv`, median
  `1.008x`, mean `1.009x`, 8 losses. A narrower cubic-only gate still had
  unstable losses, so it was removed.
- Projection constant-stride gather:
  `/tmp/splineops_ab_projection_strided_gather_aa_20260616.csv`, median
  `0.999x`, mean `1.005x`, 8 losses.
- Cubic-antialiasing 128-line batches for explicit/threaded routes were
  inconsistent: forced-8 rows looked good in one run, but the direct policy
  A/B did not reproduce cleanly. The default/threaded policy remains 96 lines.

Fresh final artifacts after this pass:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_projection_batch_single_20260616.csv`
  - 43 overlaps, median native/Python speedup `24.85x`, mean `28.45x`
  - best native thread counts: `1:6`, `8:12`, `default:25`
  - antialiasing median speedup `20.07x`
- Library full comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_projection_batch_single_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `13/18`, with different coordinate/AA semantics
  - Torch faster in `8/19`; exact-ish Torch rows all slower

Validation for this handoff:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py -k 'projection_batch_tune or rowwise_initial_causal or gather_prefilter_scale'
.venv/bin/python -m pytest -q
.venv/bin/python -m py_compile \
  scripts/benchmark_resize_native.py \
  scripts/benchmark_resize_libraries.py \
  scripts/benchmark_resize_ab.py \
  scripts/summarize_resize_benchmarks.py
git diff --check
```

Latest validation result: focused parity `18 passed`, full suite
`549 passed`; script py-compile and `git diff --check` clean.

## Session Wrap-Up: Current State and Next Steps

Current state:

- The implementation is still Arrate's least-squares projection method. The
  accepted changes rearrange execution only: batching, gather/scatter memory
  traversal, prefilter setup, fused scale factors, specialized dispatch, and
  workspace sizing.
- Latest native/Python full artifact:
  `/tmp/splineops_native_full_both_projection_batch_single_20260616.csv`
  - median speedup `24.85x`, mean speedup `28.45x` across 43 overlaps
  - cubic median speedup `25.79x`
  - antialiasing/projection median speedup `20.07x`
  - 3-D median speedup `23.87x`
- Latest default-scheduler library artifact:
  `/tmp/splineops_libraries_full_default_projection_batch_single_20260616.csv`
  - exact-ish SciPy rows: SciPy faster in `0/16`
  - exact-ish Torch rows: Torch faster in `0/8`
  - OpenCV remains faster on many 2-D image-resize rows, but those rows use
    different coordinate and antialiasing semantics.

Accepted changes accumulated so far:

- Batched ND axis execution for interpolation and projection.
- Fused 2-D/3-D linear kernels and linear direct paths.
- Row-major gather and strided-offset gather for large pure interpolation.
- Gather-prefilter scale fusion.
- Row-wise initial-causal setup for spline recursion, including finite mirror
  initializers.
- Projection batch tuning and fused single-thread projection average restore.
- Specialized fixed-support accumulation dispatch.
- 3-D axis-1 and 2-D axis-0 direct scatter in narrow pure interpolation cases.
- 2-D cubic/quadratic downsampling batch tune v2.
- Explicit-single-thread large cubic-antialiasing batch reduction to 32 lines.

Rejected paths to avoid retesting without new evidence:

- f32 projection internals as default: fast but not output-stable enough.
- f32 cached plan weights: slower on focused outliers.
- Compact fixed-support row maps: flat to negative.
- Axis-1 direct scatter and blocked/transpose axis-contiguous scatter:
  correctness-clean but mixed or negative.
- Tiled last-axis gather: negative on threaded/default rows.
- Run-length fixed-window accumulation: too mixed after narrowing.
- Projection constant-stride gather: flat to negative.
- Larger threaded cubic-antialiasing batches: promising in one sweep but not
  reproducible enough for default routing.

Recommended next work:

1. Create a benchmark-report generator that consumes the native, A/B, library,
   compare, and legacy CSV artifacts and writes one reviewable Markdown report.
2. Define a PR-readiness profile with exact-ish comparisons separated from
   image-resize comparisons. For major libraries, the exact-spline/SciPy-like
   rows should carry the argument.
3. Run hardware-counter profiling on current defaults. Prior timing profiles
   show gather, prefilter, projection integration, and projection temp-buffer
   traffic as remaining costs, but the next decisions need cache/branch data.
4. Consider a deeper pure-cubic 2-D specialization only after profiling proves
   where memory traffic can be reduced. Previous scatter/gather reorderings did
   not generalize.
5. Add CI-style benchmark smoke tests that compare a small set of A/B knobs
   against stored baselines with loose thresholds. This would catch obvious
   regressions without requiring full benchmark runs.
6. If preparing a library PR, freeze a clean benchmark corpus, publish semantic
   notes, and include a minimal implementation story: exact boundary handling,
   spline poles, LS projection filters, and why the optimized memory traversal
   does not change the mathematical method.

## Handoff: 2026-06-17 Projection Output-Prefilter Scale Fusion

Accepted change:

- Projection paths now optionally fold the output prefilter normalization factor
  into the existing finite-difference average-restore step, then run the
  pole-only output prefilter. This removes one full normalization pass over the
  projection output buffer without changing Arrate's finite-difference
  projection, mirror boundaries, spline weights, or recursive poles.
- The default policy is conservative: `auto` enables the fused scale only when
  `LSRESIZE_NUM_THREADS=1` is set explicitly. It can be forced with
  `LSRESIZE_PROJECTION_OUTPUT_PREFILTER_SCALE=1` or disabled with
  `LSRESIZE_PROJECTION_OUTPUT_PREFILTER_SCALE=0`.

Focused profiling:

- `/tmp/splineops_profile_projection_scale_auto_1024_f64_thr1.err`
- `/tmp/splineops_profile_projection_scale_disabled_1024_f64_thr1.err`
- In the 1024x1024 float64 cubic-antialiasing single-thread profile, output
  prefilter time dropped from about `5.57 ms` to `5.03 ms`; diff/average time
  dropped from about `2.72 ms` to `2.21 ms`. Total profile time moved from
  about `51.32 ms` to `50.62 ms` across the profiled benchmark process.

Accepted A/B:

- `/tmp/splineops_ab_projection_output_scale_auto_thr1_r9.csv`
  - disabled versus default-auto, explicit `LSRESIZE_NUM_THREADS=1`
  - 8 projection rows, `9` repeats, `3` warmups
  - median speedup `1.019x`, mean speedup `1.039x`
  - 2 wins, 0 losses, all checks passed with `max_abs_diff=0`
- `/tmp/splineops_ab_projection_output_scale_full_large.csv`
  - force-on large 2048x2048 cubic-antialiasing rows showed good
    single-thread results but a default-thread float64 regression
  - this is why the default is explicit-single-thread auto, not unconditional

Validation:

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m pytest -q \
  tests/test_02_03_resize_cpp.py::test_projection_output_prefilter_scaled_path_matches_disabled \
  tests/test_02_03_resize_cpp.py::test_projection_avg_restore_fused_path_matches_disabled \
  tests/test_02_03_resize_cpp.py::test_batched_axis_matches_default_antialiasing \
  tests/test_02_03_resize_cpp.py::test_float32_internal_preserves_constant_arrays
```

Latest focused validation result: `28 passed`.

## Rejected: 2026-06-17 Fused Projection Integration

Experiment:

- Combined the `nb == 2` projection integration path by computing the weighted
  average first, then fusing the symmetric and anti-symmetric integration
  recurrences into one pass. This targets the common quadratic/cubic
  antialiasing path while preserving the same Arrate finite-difference
  recurrence and output values.

Result:

- Correctness was clean in focused tests and same-build A/B checks
  (`max_abs_diff=0` in the measured rows).
- End-to-end performance was mixed and not stable enough to keep:
  - `/tmp/splineops_ab_fused_projection_integrate_standard.csv`
  - `/tmp/splineops_ab_fused_projection_integrate_standard_rerun.csv`
  - `/tmp/splineops_ab_fused_projection_integrate_large.csv`
- The large-row run had median `1.014x` overall, but default-thread rows had
  median `0.954x` and included regressions. Single-thread rows were noisy:
  some large rows won, while others regressed.
- Phase profiles showed the integration phase itself can drop, e.g.
  `/tmp/splineops_profile_fused_integrate_on_1024_f64_thr1.err` versus
  `/tmp/splineops_profile_fused_integrate_off_1024_f64_thr1.err`, but total
  runtime remained effectively flat because adjacent gather/accumulate/scatter
  phases moved within noise.

Decision:

- Removed the experimental native code path and did not add a public knob.
- Do not retry this as a default optimization without lower-level evidence
  such as hardware-counter data showing a cache or bandwidth win.

## Pause/Audit: 2026-06-22 Provenance and Remaining Speed Headroom

Status:

- Upstream-facing work is paused. The SciPy draft PR was closed after the
  missing AI/LLM disclosure issue was identified.
- The local `splineops` resize implementation and the SciPy prototype were both
  developed with extensive LLM assistance. Future upstream use needs explicit
  disclosure and a human audit that can explain and maintain the code.
- No code changes were made in this pause audit. The purpose was to preserve
  the technical state and identify whether there is obvious remaining speed
  headroom.

Reference-method interpretation:

- The implementation follows the Muñoz Barrutia/Blu/Unser finite-difference
  spline resize method and the Lee/Eden/Unser oblique-projection production
  variant.
- Legacy `Resize.java` and current C++ agree on the core cost model:
  projection performs `analy_degree + 1` integrations before geometric
  resampling, then the matching differences, correction prefilter, and synthesis
  sampling after resampling.
- Equal-degree least-squares remains the orthogonal/reference path, but the
  public `*-antialiasing` presets are better production defaults because their
  oblique triples use lower integration/correction order.

Fresh artifacts:

```text
/tmp/splineops_resize_evidence_check_20260622.md
/tmp/splineops_resize_thread_sweep_20260622.csv
/tmp/splineops_resize_thread_sweep_20260622.json
```

Hardware-counter profiling:

- `perf` is installed, but hardware counters are blocked locally by
  `perf_event_paranoid=4`.
- The audit therefore used `LSRESIZE_PROFILE=1` phase profiles plus controlled
  same-build A/B timing.

Phase-profile summary, `LSRESIZE_NUM_THREADS=1`:

| Case | Main phases |
| --- | --- |
| `2d_cubic_down_1024_f32` | gather `31.9%`, prefilter `29.3%`, accumulate `17.9%`, accumulate/scatter plus scatter about `19.8%` |
| `2d_cubic_aa_down_1024_f32` | gather `21.6%`, integrate `21.1%`, accumulate `16.8%`, input prefilter `12.7%`, output prefilter `10.4%` |
| `2d_linear_aa_down_1024_f32` | gather `34.1%`, integrate `23.7%`, accumulate `13.5%`, scatter `12.5%` |
| `3d_cubic_down_f32` | gather `42.8%`, prefilter `20.2%`, accumulate `17.4%`, accumulate/scatter `11.9%` |
| `3d_cubic_aa_down_f32` | spread across gather `17.6%`, accumulate `16.0%`, output prefilter `14.2%`, integrate/input prefilter about `22.0%` |

Interpretation:

- Pure linear paths are already heavily optimized by direct/fused kernels.
- Pure cubic interpolation still has some headroom, mostly in gather and
  intermediate memory movement, especially for 3-D.
- Projection antialiasing does not have one dominant hot loop. The time is
  distributed over the required projection passes, so large gains likely need a
  layout or algorithmic change rather than another small loop fusion.

Same-build A/B observations:

- Fused 2-D linear remains a large default-on win: about `2.1x` to `2.7x` on
  sampled 2-D linear rows.
- Fused 3-D linear remains a large default-on win: about `2.4x` to `3.3x` on
  common down/two-axis rows.
- Row-major gather remains useful and default-on: `1.16x` to `1.49x` on sampled
  2-D cubic/projection rows.
- Strided-offset gather is useful in the sampled 2-D cubic row and mostly
  neutral elsewhere.
- Gather-prefilter scale fusion remains useful for pure cubic interpolation
  (`1.18x` to `1.25x` in sampled rows) and mostly neutral for projection.
- Projection output-prefilter scale fusion is a small neutral-to-slight win in
  the sampled rows, not a broad new source of speed.
- Fixed `LSRESIZE_BATCH_LINES=64` was worse than the adaptive default in the
  sampled cubic/projection cases.

Thread-policy finding:

- Explicit `LSRESIZE_NUM_THREADS=2` was consistently bad in the focused sweep.
- Small sampled 3-D cubic/projection cases preferred serial execution:
  - `3d_cubic_down_f32`: best median at `1` thread; default was about `1.09x`
    slower.
  - `3d_cubic_aa_down_f32`: best median at `1` thread; default was about
    `1.48x` slower.
- Forcing serial globally with a very high `LSRESIZE_PARALLEL_THRESHOLD`
  improved the small 3-D cases but hurt 1024x1024 2-D projection and larger 3-D
  rows badly.
- Conclusion: do not globally raise the parallel threshold. If work resumes,
  prototype a narrow small-volume 3-D serial gate and validate against large
  2-D and large 3-D benchmark rows.

Current technical conclusion:

- `splineops.resize` appears to be a real improvement for exact-ish N-D spline
  resize semantics, especially compared with SciPy-like rows and 3-D oblique
  antialiasing.
- It should not be positioned as the fastest generic 2-D image resize; OpenCV
  and Torch can be faster in non-equivalent image-resize semantics.
- The current native backend is past the easy optimization stage. Further CPU
  gains are likely incremental and workload-specific unless the project changes
  layout strategy, algorithmic semantics, or backend target.

Recommended if work resumes:

1. Keep provenance and AI/LLM disclosure explicit in all upstream work.
2. Rebuild a narrow human-owned branch from the documented method and tests.
3. Start with thread-policy validation for small 3-D cubic/projection cases.
4. Only consider a deeper 3-D gather/layout rewrite after cache-counter
   profiling is available.
5. Keep projection antialiasing claims focused on exact N-D spline semantics,
   lower-order oblique projection, and measured 3-D behavior.
