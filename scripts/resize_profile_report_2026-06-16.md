# Resize Profiling Report: 2026-06-16

This pass profiled the native `_lsresize` backend after the previous resize
optimization work. The goal was to identify where time is still spent, validate
3-D routing decisions, and separate real optimization wins from noisy knobs.

## Instrumentation

Linux `perf` was not usable on this machine because
`/proc/sys/kernel/perf_event_paranoid` is `4`; `perf stat` exits before
collecting counters. To keep profiling reproducible without elevated
permissions, the native backend now has an opt-in phase profiler:

```bash
LSRESIZE_PROFILE=1 LSRESIZE_PROFILE_LABEL=<label> python ...
```

The profiler prints per-phase call counts, total time, average time, and percent
of `nd.axis.total` at process exit. In threaded runs, phase totals are summed
across worker threads, so use them as CPU-time distribution rather than wall
percentages.

## Changes From The Profiles

- `LSRESIZE_BATCHED_AXIS=auto` now routes pure 3-D quadratic/cubic
  interpolation through the batched axis kernel, including short axes.
- Default precision now uses float32 internals for 2-D and 3-D float32 pure
  quadratic/cubic interpolation. Projection and antialiasing remain on the
  conservative default unless `LSRESIZE_PRECISION=float32` is explicitly set.
- A follow-up dataflow pass changed batched gather from line-wise strided
  coefficient stores to coefficient-row-major stores. `LSRESIZE_ROW_GATHER=0`
  forces the old gather order for A/B checks.
- `LSRESIZE_BATCH_LINES` is now adaptive only for 2-D pure quadratic/cubic
  downsampling; all other unset cases keep the historical batch size.
- Large 3-D axis-1 pure quadratic/cubic interpolation passes now skip the
  temporary accumulation row and scatter directly into the destination.
  `LSRESIZE_3D_AXIS1_DIRECT_SCATTER=0` restores the buffered scatter path for
  A/B checks. The router keeps the path off for smaller passes where direct
  destination traffic was not a stable win, and for larger worker pools where
  the extra destination traffic can dominate.
- The latest dataflow pass fuses the interpolation-prefilter normalization
  factor into batched gather for double-internal interpolation/projection and
  float32 pure interpolation. `LSRESIZE_GATHER_PREFILTER_SCALE=0` restores the
  prior gather-then-scale path for A/B checks. The recursive pole application
  remains the same Arrate-method spline prefilter.

These keep Arrate's least-squares projection method intact: the changes are
routing, storage precision for pure interpolation, and batched execution of the
same separable spline operations.

## Final Single-Thread Phase Profiles

Artifacts:

- `/tmp/splineops_profile_final_2d_cubic_down_2048_float32_thr1_20260616.log`
- `/tmp/splineops_profile_final_2d_cubic_aa_down_2048_float32_thr1_20260616.log`
- `/tmp/splineops_profile_final_3d_cubic_down_f32_thr1_20260616.log`
- `/tmp/splineops_profile_final_3d_cubic_down_large_f32_thr1_20260616.log`

| Case | Native median | Main remaining costs |
| --- | ---: | --- |
| `2d_cubic_down_2048_float32` | `18.71 ms` | gather/transpose `65.64%`, prefilter `15.06%`, accumulate `6.58%`, scatter `6.07%` |
| `2d_cubic_aa_down_2048_float32` | `58.12 ms` | gather/transpose `52.48%`, integrate `11.37%`, input prefilter `8.92%`, accumulate `7.56%`, output prefilter `5.97%` |
| `3d_cubic_down_f32` | `3.23 ms` | gather/transpose `48.80%`, prefilter `23.56%`, accumulate `12.04%`, accumulate+scatter `9.74%` |
| `3d_cubic_down_large_f32` | `19.40 ms` | gather/transpose `63.43%`, prefilter `18.47%`, accumulate+scatter `15.14%` |

Interpretation: after removing the old 3-D line-by-line fallback bottleneck,
the dominant cost is memory movement into the batched coefficient layout. The
spline math is no longer the main cost for pure interpolation.

## A/B Results

Artifacts:

- `/tmp/splineops_ab_3d_off_auto_after_precision3d_20260616.csv`
- `/tmp/splineops_ab_precision_f64_default_3d_after_auto3d_20260616.csv`
- `/tmp/splineops_batch_lines_final_profile_20260616.csv`
- `/tmp/splineops_ab_row_gather_20260616.csv`
- `/tmp/splineops_ab_adaptive_batch_lines_narrow_20260616.csv`

Key results:

- `LSRESIZE_BATCHED_AXIS=off -> <unset>` on 3-D cubic cases:
  median `2.562x`, mean `2.509x`, `12/12` wins, no failed checks.
- `LSRESIZE_PRECISION=float64 -> <unset>` on 3-D cubic float32 cases:
  median `1.443x`, mean `1.458x`, `12/12` wins, no failed checks.
  Maximum observed absolute difference was `7.75e-7`.
- `LSRESIZE_ROW_GATHER=0 -> <unset>` on hotspot cubic/projection cases:
  median `1.237x`, mean `1.267x`, `26/27` wins, no failed checks.
- Fixed `LSRESIZE_BATCH_LINES=64 -> <unset>` after narrowing the adaptive
  heuristic:
  median `1.047x`, mean `1.095x`, `16/27` wins, `2/27` losses, no failed
  checks.
- `LSRESIZE_3D_AXIS1_DIRECT_SCATTER=0 -> <unset>` after adding the
  thresholded/thread-guarded router:
  median `1.079x`, mean `1.078x`, `9/12` wins, no losses, no failed checks.
  The unthresholded attempt was rejected as a blanket default because small
  anisotropic 3-D cases regressed under explicit thread counts; a threshold-only
  attempt exposed thread-pool sensitivity, so the final router also checks
  selected worker count.
- `LSRESIZE_BATCH_LINES` remains cache-sensitive but not solved by a single
  obvious default. In the sampled cases, pure 2-D cubic liked `16`, 2-D cubic
  antialiasing liked `16`-`96`, and large 3-D cubic liked `96`-`192`.

## End-To-End Benchmarks

Native versus Python fallback artifact:
`/tmp/splineops_native_full_both_profiled_routing_precision_20260616.csv`

- Native/Python overlaps: `43`
- Median speedup: `21.44x`
- Mean speedup: `24.64x`
- Range: `6.97x` to `72.23x`
- 3-D median speedup: `21.08x`

After the row-major gather/adaptive-batch pass:
`/tmp/splineops_native_full_after_row_gather_adaptive_20260616.csv`

- Native/Python overlaps: `43`
- Median speedup: `25.43x`
- Mean speedup: `28.25x`
- Range: `7.87x` to `79.32x`
- 3-D median speedup: `23.79x`
- Versus the prior profiled native artifact, overlapping native rows improved
  by median `1.130x`; 2-D cubic rows improved by median `1.449x`, and 3-D
  cubic rows by median `1.400x`.

After the direct-scatter pass:
`/tmp/splineops_native_full_both_direct_scatter_20260616.csv`

- Native/Python overlaps: `43`
- Median speedup: `24.62x`
- Mean speedup: `27.83x`
- Range: `7.78x` to `85.68x`
- 3-D median speedup: `20.67x`
- Best native thread counts: `1:8`, `8:17`, `default:18`

After the fused gather-prefilter scale pass:
`/tmp/splineops_native_full_both_gather_prefilter_scale_20260616.csv`

- Native/Python overlaps: `43`
- Median speedup: `25.11x`
- Mean speedup: `29.16x`
- Range: `7.67x` to `90.82x`
- 3-D median speedup: `25.10x`
- Best native thread counts: `1:8`, `8:23`, `default:12`

Cross-library artifact:
`/tmp/splineops_libraries_full_after_profile_20260616.csv`

- SciPy: median speed `0.14x` versus splineops, exact-ish cases `0.09x`.
- skimage: median speed `0.12x`, with different resize semantics in most rows.
- OpenCV: median speed `1.47x`, but median relative L2 difference was
  `2.61e-1`; this is generally not the same operation.
- Torch: median speed `0.75x`; exact-ish cases `0.57x`.

After the row-major gather/adaptive-batch pass:
`/tmp/splineops_libraries_full_after_row_gather_adaptive_20260616.csv`

- SciPy: median speed `0.15x` versus splineops, exact-ish cases `0.07x`.
- skimage: median speed `0.11x`, with different resize semantics in most rows.
- OpenCV: median speed `1.44x`, with median relative L2 difference `2.61e-1`.
- Torch: median speed `0.73x`; exact-ish cases `0.59x`.

After the direct-scatter pass, splineops default scheduler:
`/tmp/splineops_libraries_full_default_direct_scatter_20260616.csv`

- SciPy: median speed `0.12x` versus splineops, exact-ish cases `0.10x`.
- skimage: median speed `0.11x`, with different resize semantics in most rows.
- OpenCV: median speed `1.41x`, with median relative L2 difference `2.61e-1`.
- Torch: median speed `0.75x`; exact-ish cases `0.59x`.

After the direct-scatter pass, splineops forced to 8 threads:
`/tmp/splineops_libraries_full_threads8_direct_scatter_20260616.csv`

- SciPy: median speed `0.16x` versus splineops, exact-ish cases `0.12x`.
- skimage: median speed `0.16x`, with different resize semantics in most rows.
- OpenCV: median speed `1.91x`, with median relative L2 difference `2.61e-1`.
- Torch: median speed `1.10x`; exact-ish cases `0.93x`.

After the fused gather-prefilter scale pass, splineops default scheduler:
`/tmp/splineops_libraries_full_default_gather_prefilter_scale_20260616.csv`

- SciPy: median speed `0.14x` versus splineops, exact-ish cases `0.08x`.
- skimage: median speed `0.11x`, with different resize semantics in most rows.
- OpenCV: median speed `1.41x`, with median relative L2 difference `2.61e-1`.
- Torch: median speed `0.71x`; exact-ish cases `0.59x`.

After the fused gather-prefilter scale pass, splineops forced to 8 threads:
`/tmp/splineops_libraries_full_threads8_gather_prefilter_scale_20260616.csv`

- SciPy: median speed `0.17x` versus splineops, exact-ish cases `0.10x`.
- skimage: median speed `0.15x`, with different resize semantics in most rows.
- OpenCV: median speed `1.90x`, with median relative L2 difference `2.61e-1`.
- Torch: median speed `1.14x`; exact-ish cases `0.98x`.

## Remaining Optimization Targets

1. Continue reducing gather/transpose cost in the batched coefficient layout.
   Row-major gather cut it substantially, but it remains material in large pure
   interpolation cases.
2. Improve non-contiguous output accumulate+scatter for anisotropic 3-D. After
   row-major gather, this can be the largest phase for single-axis 3-D cubic.
3. Consider direct contiguous-axis kernels for pure cubic/quadratic that avoid
   some temporary layout conversion while preserving the recursive spline
   prefiltering and the existing boundary model.
4. For antialiasing/projection, focus on integration/prefilter/accumulate phases;
   float32 projection remains opt-in because random projection outputs differ
   more than pure interpolation.
