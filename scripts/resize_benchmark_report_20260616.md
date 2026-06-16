# Resize Benchmark Report - 2026-06-16

Commit measured: `4aa3846 Implement optimization pass`

Host:

- CPU: Intel Core i7-7820X, 8 physical cores / 16 logical CPUs
- Python: 3.12.3
- NumPy: 2.4.6
- SciPy: 1.17.1
- scikit-image: 0.26.0
- OpenCV: 4.13.0
- PyTorch: 2.12.0+cpu

## Artifacts

- Native full, same-algorithm Python fallback:
  `/tmp/splineops_native_full_both_postcommit_20260616.{json,csv}`
- Native full after direct-scatter pass:
  `/tmp/splineops_native_full_both_direct_scatter_20260616.{json,csv}`
- Native full after fused gather-prefilter scale pass:
  `/tmp/splineops_native_full_both_gather_prefilter_scale_20260616.{json,csv}`
- Native full after row-wise initial-causal/projection-batch pass:
  `/tmp/splineops_native_full_both_rowwise_projection_batch_20260616.{json,csv}`
- Native full after row-wise finite-causal initializer pass:
  `/tmp/splineops_native_full_both_rowwise_finite_20260616.{json,csv}`
- Cross-library full, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_postcommit_20260616.{json,csv}`
- Cross-library full after direct-scatter pass, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_direct_scatter_20260616.{json,csv}`
- Cross-library full after fused gather-prefilter scale pass, splineops default
  scheduler:
  `/tmp/splineops_libraries_full_default_gather_prefilter_scale_20260616.{json,csv}`
- Cross-library full after row-wise initial-causal/projection-batch pass,
  splineops default scheduler:
  `/tmp/splineops_libraries_full_default_rowwise_projection_batch_20260616.{json,csv}`
- Cross-library full after row-wise finite-causal initializer pass, splineops
  default scheduler:
  `/tmp/splineops_libraries_full_default_rowwise_finite_20260616.{json,csv}`
- Cross-library full, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_postcommit_20260616.{json,csv}`
- Cross-library full after direct-scatter pass, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_direct_scatter_20260616.{json,csv}`
- Cross-library full after fused gather-prefilter scale pass, splineops forced
  to 8 threads:
  `/tmp/splineops_libraries_full_threads8_gather_prefilter_scale_20260616.{json,csv}`
- Cross-library full after row-wise initial-causal/projection-batch pass,
  splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_rowwise_projection_batch_20260616.{json,csv}`
- Cross-library full after row-wise finite-causal initializer pass, splineops
  forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_rowwise_finite_20260616.{json,csv}`
- Legacy Java 2-D Arrate implementation harness:
  `/tmp/splineops_legacy_java_full_20260616.csv`
- Legacy Java 2-D Arrate harness after direct-scatter pass:
  `/tmp/splineops_legacy_java_full_direct_scatter_20260616.csv`
- Legacy Java 2-D Arrate harness after fused gather-prefilter scale pass:
  `/tmp/splineops_legacy_java_full_gather_prefilter_scale_20260616.csv`
- Legacy Java 2-D Arrate harness after row-wise initial-causal/projection-batch
  pass:
  `/tmp/splineops_legacy_java_full_rowwise_projection_batch_20260616.csv`
- Legacy Java 2-D Arrate harness after row-wise finite-causal initializer pass:
  `/tmp/splineops_legacy_java_full_rowwise_finite_20260616.csv`
- Temporary legacy harness sources:
  `/tmp/legacy_resize_bench/ImageAccess.java`
  and `/tmp/legacy_resize_bench/LegacyResizeBench.java`

## Same-Algorithm Results

Native C++ versus the current Python fallback on the full native profile:

| Scope | Cases | Median speedup | Mean speedup |
| --- | ---: | ---: | ---: |
| All native/Python overlaps | 43 | `20.25x` | `23.39x` |
| Linear | 17 | `25.49x` | `31.08x` |
| Cubic | 16 | `18.53x` | `19.43x` |
| Antialiasing/projection | 10 | `14.31x` | `16.65x` |
| 3-D | 9 | `11.76x` | `14.74x` |

Best native thread setting by median over the full profile:

| Thread setting | Winning cases |
| --- | ---: |
| `1` | 5 |
| `8` | 17 |
| `16` | 8 |
| `default` | 13 |

The workload-aware default is still the best general default. Forced `8` helps
many heavy cubic/projection rows on this 8-core CPU, while forced `16` is often
worse on medium rows.

After the 3-D axis-1 direct-scatter pass, the full native/Python artifact
reported 43 overlaps with median speedup `24.62x` and mean speedup `27.83x`.
Best native thread counts were `1:8`, `8:17`, and `default:18`; the 3-D median
speedup was `20.67x`.

After the fused gather-prefilter scale pass, the full native/Python artifact
reported 43 overlaps with median speedup `25.11x` and mean speedup `29.16x`.
Best native thread counts were `1:8`, `8:23`, and `default:12`; the 3-D median
speedup was `25.10x`.

After the row-wise initial-causal/projection-batch pass, the full native/Python
artifact reported 43 overlaps with median speedup `27.32x` and mean speedup
`28.39x`. Best native thread counts were `1:8`, `8:22`, and `default:13`; the
antialiasing median speedup was `20.07x`.

After the strided-offset gather pass, the full native/Python artifact reported
43 overlaps with median speedup `23.23x` and mean speedup `26.85x`. Best native
thread counts were `1:12`, `8:18`, and `default:13`; the 3-D median speedup
was `22.89x`.

After the row-wise finite-causal initializer pass, the full native/Python
artifact reported 43 overlaps with median speedup `24.86x` and mean speedup
`27.42x`. Best native thread counts were `1:12`, `8:18`, and `default:13`;
cubic median speedup was `25.18x` and antialiasing/projection median speedup
was `16.22x`.

## Legacy Java Baseline

A temporary standalone shim was used around `legacy_code/resize/Resize.java`.
This is useful as a historical Arrate-method timing reference, but it is not a
drop-in library benchmark: it is Java, 2-D only, and uses a minimal local
`ImageAccess` implementation.

Over the 14 overlapping 2-D cases, current splineops default scheduling was:

| Statistic | Speedup versus legacy Java |
| --- | ---: |
| Median | `9.39x` |
| Mean | `11.42x` |
| Minimum | `2.71x` |
| Maximum | `35.07x` |

The weakest same-method legacy gap was `2d_linear_aa_down_random_f32`
(`4.557 ms -> 1.679 ms`, `2.71x`). The strongest was
`2d_linear_up_sinusoid_f32_large` (`46.708 ms -> 1.332 ms`, `35.07x`).

The legacy harness was rerun after the 3-D direct-scatter pass for a fresh
reference artifact. Against the current splineops default library artifact, the
14 overlapping 2-D cases showed median speedup `9.18x`, mean `12.93x`, minimum
`4.36x`, and maximum `50.44x`.

After the fused gather-prefilter scale pass, the same legacy harness showed
median speedup `8.44x`, mean `11.04x`, minimum `3.53x`, and maximum `34.33x`
over 14 overlapping 2-D cases.

After the row-wise initial-causal/projection-batch pass, the legacy harness
showed median speedup `10.15x`, mean `12.44x`, minimum `3.74x`, and maximum
`35.44x` over 14 overlapping 2-D cases.

After the strided-offset gather pass, the legacy harness showed median speedup
`10.02x`, mean `12.08x`, minimum `3.86x`, and maximum `31.00x` over 14
overlapping 2-D cases.

After the row-wise finite-causal initializer pass, the legacy harness showed
median speedup `10.82x`, mean `12.60x`, minimum `5.57x`, and maximum `32.74x`
over 14 overlapping 2-D cases.

## Cross-Library Results

Full profile, splineops default scheduler:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.15x` | `5.74e-08` |
| scikit-image | 21 | 0 | `0.14x` | `2.39e-01` |
| OpenCV | 18 | 15 | `1.49x` | `2.61e-01` |
| PyTorch | 19 | 8 | `0.75x` | `1.59e-05` |

Full profile, splineops forced to `LSRESIZE_NUM_THREADS=8`:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.17x` | `5.74e-08` |
| scikit-image | 21 | 0 | `0.17x` | `2.39e-01` |
| OpenCV | 18 | 16 | `1.93x` | `2.61e-01` |
| PyTorch | 19 | 12 | `1.26x` | `1.59e-05` |

After the direct-scatter pass, splineops default scheduler:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.12x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.11x` | `2.39e-01` |
| OpenCV | 18 | 14 | `1.41x` | `2.61e-01` |
| PyTorch | 19 | 8 | `0.75x` | `1.59e-05` |

After the direct-scatter pass, splineops forced to `LSRESIZE_NUM_THREADS=8`:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.16x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.16x` | `2.39e-01` |
| OpenCV | 18 | 17 | `1.91x` | `2.61e-01` |
| PyTorch | 19 | 10 | `1.10x` | `1.59e-05` |

After the fused gather-prefilter scale pass, splineops default scheduler:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.14x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.11x` | `2.39e-01` |
| OpenCV | 18 | 13 | `1.41x` | `2.61e-01` |
| PyTorch | 19 | 8 | `0.71x` | `1.59e-05` |

After the fused gather-prefilter scale pass, splineops forced to
`LSRESIZE_NUM_THREADS=8`:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 0 | `0.17x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.15x` | `2.39e-01` |
| OpenCV | 18 | 17 | `1.90x` | `2.61e-01` |
| PyTorch | 19 | 10 | `1.14x` | `1.59e-05` |

After the row-wise initial-causal/projection-batch pass, splineops default
scheduler:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 0 | `0.14x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.10x` | `2.39e-01` |
| OpenCV | 18 | 15 | `1.41x` | `2.61e-01` |
| PyTorch | 19 | 7 | `0.75x` | `1.59e-05` |

After the row-wise initial-causal/projection-batch pass, splineops forced to
`LSRESIZE_NUM_THREADS=8`:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.17x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.15x` | `2.39e-01` |
| OpenCV | 18 | 16 | `1.96x` | `2.61e-01` |
| PyTorch | 19 | 11 | `1.29x` | `1.59e-05` |

After the strided-offset gather pass, splineops default scheduler:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.15x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.11x` | `2.39e-01` |
| OpenCV | 18 | 14 | `1.33x` | `2.61e-01` |
| PyTorch | 19 | 7 | `0.68x` | `1.59e-05` |

Exact-ish rows under the default scheduler still favor splineops: SciPy faster
in `0/16`, Torch faster in `0/8`.

After the strided-offset gather pass, splineops forced to
`LSRESIZE_NUM_THREADS=8`:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.17x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.15x` | `2.39e-01` |
| OpenCV | 18 | 16 | `1.85x` | `2.61e-01` |
| PyTorch | 19 | 10 | `1.17x` | `1.59e-05` |

Exact-ish rows with forced 8 threads: SciPy faster in `0/16`, Torch faster in
`3/8`.

After the row-wise finite-causal initializer pass, splineops default scheduler:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.15x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.11x` | `2.39e-01` |
| OpenCV | 18 | 14 | `1.39x` | `2.61e-01` |
| PyTorch | 19 | 8 | `0.65x` | `1.59e-05` |

Exact-ish rows under the default scheduler still favor splineops: SciPy faster
in `0/16`, Torch faster in `0/8`.

After the row-wise finite-causal initializer pass, splineops forced to
`LSRESIZE_NUM_THREADS=8`:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.17x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.15x` | `2.39e-01` |
| OpenCV | 18 | 16 | `2.04x` | `2.61e-01` |
| PyTorch | 19 | 12 | `1.19x` | `1.59e-05` |

Exact-ish rows with forced 8 threads: SciPy faster in `0/16`, Torch faster in
`5/8`.

Important interpretation:

- SciPy is the closest semantic comparison for spline interpolation. In the
  latest full-profile default run, splineops is faster on `20/21` SciPy rows
  and on all exact-ish SciPy rows.
- OpenCV is faster on most 2-D rows, but its median rel-L2 delta is large
  because it implements different coordinate, boundary, cubic, and antialiasing
  semantics.
- PyTorch is close on some linear rows and faster on some image-style 2-D rows,
  but exact-ish rows with rel-L2 below `1e-5` still favor splineops under the
  default scheduler.

## Profiling Notes

Linux `perf` could not be used in this session because
`/proc/sys/kernel/perf_event_paranoid` is `4`. Benchmark slicing therefore
served as the profiling proxy.

Observed remaining hotspots by timing behavior:

- exact cubic interpolation remains the largest exact-semantics target
- projection/antialiasing rows still have substantial memory traffic
- default scheduling is generally good, but heavy cubic/projection workloads can
  benefit from an explicit physical-core thread count on this host

Controlled A/B for the row-wise finite-causal initializer:

- Artifact:
  `/tmp/splineops_ab_rowwise_initial_causal_finite_clean_20260616.{json,csv}`
- Full native profile, `1/8/default` threads: median `1.028x`, mean `1.054x`,
  64 wins and 24 losses, no failed checks.
- Cubic rows had median `1.054x`; cubic-antialiasing rows had median `1.016x`.

Rejected experiments in the same pass:

- Automatic float32 projection internals were fast but not default-safe:
  selected A/B median speedup about `1.234x`, with output drift up to
  `1.75e-3`.
- Fusing length-2 sampling FIR into output scatter matched outputs but was
  slower: median `0.876x`, mean `0.894x`, 1 win and 15 losses.

## Double-Internal Strided Gather

The latest pass extends the existing strided-offset gather to large
double-internal pure interpolation axes. It is gated to non-upsampling axes and
large inputs so small-image and projection/antialiasing rows stay on the prior
offset-array gather.

Focused A/B:

- Active large float64 2-D cubic rows:
  `/tmp/splineops_ab_strided_offset_gather_double_active_large2d_20260616.csv`
  - median `1.066x`, mean `1.061x`, 6 wins and 2 losses, no failed checks
  - default scheduler median `1.088x`, 3 wins and 0 losses
- Isolated 2048 float64 cubic repeat:
  `/tmp/splineops_ab_strided_offset_gather_double_2048_repeat_20260616.csv`
  - median `1.083x`, mean `1.088x`, 3 wins and 0 losses

Current full native/Python sweep:

- Artifact:
  `/tmp/splineops_native_full_both_double_strided_gather_20260616.csv`
- 43 overlaps, median native/Python speedup `22.25x`, mean `27.34x`
- By method: linear `24.67x`, cubic `23.88x`, antialiasing `16.20x`
- 3-D median native/Python speedup: `21.30x`

Current library comparison, splineops default scheduler:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.15x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.11x` | `2.39e-01` |
| OpenCV | 18 | 14 | `1.61x` | `2.61e-01` |
| PyTorch | 19 | 8 | `0.70x` | `1.59e-05` |

Exact-ish rows under the default scheduler: SciPy faster in `0/16`, Torch
faster in `0/8`.

Current library comparison, splineops forced to `LSRESIZE_NUM_THREADS=8`:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.18x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.17x` | `2.39e-01` |
| OpenCV | 18 | 17 | `2.08x` | `2.61e-01` |
| PyTorch | 19 | 12 | `1.29x` | `1.59e-05` |

Exact-ish rows with forced 8 threads: SciPy faster in `0/16`, Torch faster in
`4/8`.

Legacy Java 2-D reference:

- Artifact:
  `/tmp/splineops_legacy_java_full_double_strided_gather_20260616.csv`
- 14 overlaps against the current default-scheduler splineops artifact, median
  speedup `10.66x`, mean `13.73x`.

Rejected in this pass:

- Compact fixed-support row maps for support <= 7 were correctness-clean but
  flat to negative: first layout median `0.998x`, interleaved layout median
  `1.000x`, and interleaved explicit 8-thread median `0.943x`. The experiment
  was removed.

## Next Engineering Targets

1. Add deeper exact cubic specialization for 2-D pure interpolation and common
   contiguous-axis passes.
2. Investigate projection/antialiasing temp-buffer traffic with a strategy that
   helps threaded/default paths, not only explicit single-thread mode.
3. Extend the CSV artifact comparison tooling into a first-class benchmark
   report generator.
4. If a library PR becomes the target, package SciPy-like exact-semantics
   comparisons separately from OpenCV/PyTorch image-resize comparisons.

## Addendum: Batch Tune V2 and 2-D Axis-0 Direct Scatter

This addendum covers the next pass after the double-internal strided-gather
run. Two narrow changes were accepted:

- 2-D pure quadratic/cubic upsampling on axis 0 can use direct
  accumulate-scatter when the output axis is non-contiguous and large enough.
- 2-D pure quadratic/cubic downsampling gets a single-thread batch adjustment
  from 16 to 24 lines for f32 and large double rows.

Focused same-build A/B:

| Flag | Artifact | Median | Mean | Wins | Losses |
| --- | --- | ---: | ---: | ---: | ---: |
| `LSRESIZE_2D_AXIS0_DIRECT_SCATTER` | `/tmp/splineops_ab_2d_axis0_direct_scatter_up_final_20260616.csv` | `1.078x` | `1.076x` | 6 | 0 |
| `LSRESIZE_BATCH_TUNE_V2` | `/tmp/splineops_ab_batch_tune_v2_single_thread_final_20260616.csv` | `1.036x` | `1.030x` | 2 | 0 |

Rejected in this pass:

- Cached f32 plan weights. The focused repeat was slower overall
  (`0.980x` median, `0.978x` mean), so the experiment was removed.

Current library comparison, splineops default scheduler:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.15x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.11x` | `2.39e-01` |
| OpenCV | 18 | 12 | `1.43x` | `2.61e-01` |
| PyTorch | 19 | 8 | `0.68x` | `1.59e-05` |

Exact-ish rows under the default scheduler: SciPy faster in `0/16`, Torch
faster in `0/8`.

Current library comparison, splineops forced to `LSRESIZE_NUM_THREADS=8`:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Median rel-L2 delta |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.18x` | `5.79e-08` |
| scikit-image | 21 | 0 | `0.16x` | `2.39e-01` |
| OpenCV | 18 | 17 | `2.08x` | `2.61e-01` |
| PyTorch | 19 | 11 | `1.30x` | `1.59e-05` |

Exact-ish rows with forced 8 threads: SciPy faster in `0/16`, Torch faster in
`4/8`.

Fresh artifacts:

- Native current defaults:
  `/tmp/splineops_native_full_current_batch_axis0_20260616.csv`
- Native with the new knobs disabled:
  `/tmp/splineops_native_full_without_batch_axis0_20260616.csv`
- Libraries, default scheduler:
  `/tmp/splineops_libraries_full_default_batch_axis0_20260616.csv`
- Libraries, forced 8 threads:
  `/tmp/splineops_libraries_full_threads8_batch_axis0_20260616.csv`
- Legacy Java:
  `/tmp/splineops_legacy_java_full_batch_axis0_20260616.csv`
  - current splineops default median speedup `7.94x` over 14 overlapping 2-D
    cases

## Addendum: Projection Batch Single-Thread Tuning

The next optimization pass accepted one narrow batch-policy change:

- Large 2-D cubic-antialiasing projection uses 32-line batches for explicit
  `LSRESIZE_NUM_THREADS=1`; default/threaded scheduling keeps the existing
  96-line policy.

Focused same-build A/B:

| Change | Artifact | Median | Mean | Wins | Losses |
| --- | --- | ---: | ---: | ---: | ---: |
| cubic-AA `96 -> 32` batch, explicit single-thread | `/tmp/splineops_ab_cubic_aa_batch96_vs32_single_20260616.csv` | `1.072x` | `1.096x` | 3 | 0 |

Rejected in this pass:

- 2-D axis-1 direct scatter, blocked/transpose axis-contiguous scatter, tiled
  last-axis gather, run-length fixed-window accumulation, and projection
  constant-stride gather. All were correctness-clean where applicable, but
  mixed or negative on the focused A/B matrices.

Fresh artifacts after this pass:

- Native/Python full sweep:
  `/tmp/splineops_native_full_both_projection_batch_single_20260616.csv`
  - 43 overlaps, median native/Python speedup `24.85x`, mean `28.45x`
  - antialiasing median speedup `20.07x`
- Library comparison, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_projection_batch_single_20260616.csv`
  - SciPy faster in `1/21`; exact-ish SciPy rows all slower
  - skimage faster in `0/21`
  - OpenCV faster in `13/18`, with different semantics
  - Torch faster in `8/19`; exact-ish Torch rows all slower

## Current Position and Next Steps

Current native position:

| Scope | Cases | Median speedup vs Python fallback | Mean speedup |
| --- | ---: | ---: | ---: |
| All overlaps | 43 | `24.85x` | `28.45x` |
| Cubic | 16 | `25.79x` | `29.17x` |
| Antialiasing/projection | 10 | `20.07x` | `21.60x` |
| 3-D | 9 | `23.87x` | `25.11x` |

Current default-scheduler library position:

| Backend | Comparable cases | Faster than splineops | Median speed vs splineops | Exact-ish faster |
| --- | ---: | ---: | ---: | ---: |
| SciPy | 21 | 1 | `0.14x` | `0/16` |
| scikit-image | 21 | 0 | `0.11x` | n/a |
| OpenCV | 18 | 13 | `1.37x` | n/a |
| PyTorch | 19 | 8 | `0.67x` | `0/8` |

Interpretation:

- Against exact-ish SciPy/Torch rows, splineops is now consistently faster on
  this benchmark set.
- OpenCV remains faster on many image-resize-style 2-D rows, but those rows are
  not exact semantic matches for Arrate's spline projection method.
- The remaining performance work should be guided by profiling, not by adding
  speculative scatter/gather kernels. Several such attempts were measured and
  rejected.

Recommended next steps:

1. Turn `scripts/summarize_resize_benchmarks.py` into a full report generator
   that can produce this document automatically from CSV artifacts.
2. Build a PR-focused benchmark pack that isolates exact-semantics comparisons
   from contextual image-resize comparisons.
3. Add hardware-counter profiling for the remaining hotspots: cubic gather,
   prefilter recursion, projection integration/diff/output prefilter, and
   projection scratch-buffer traffic.
4. Revisit pure-cubic specialization only with counter evidence showing a
   specific memory-traffic reduction. Prior tiled gather/scatter and fixed-run
   accumulator experiments were too mixed.
5. For a future library PR, document the exact numerical method first, then
   present performance as an implementation of the same spline/LS projection
   semantics rather than as a generic image-resize replacement.
