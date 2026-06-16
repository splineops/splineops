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
- Cross-library full, splineops default scheduler:
  `/tmp/splineops_libraries_full_default_postcommit_20260616.{json,csv}`
- Cross-library full, splineops forced to 8 threads:
  `/tmp/splineops_libraries_full_threads8_postcommit_20260616.{json,csv}`
- Legacy Java 2-D Arrate implementation harness:
  `/tmp/splineops_legacy_java_full_20260616.csv`
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

Important interpretation:

- SciPy is the closest semantic comparison for spline interpolation. Splineops
  is faster on `20/21` full-profile SciPy rows with default scheduling.
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

## Next Engineering Targets

1. Add deeper exact cubic specialization for 2-D pure interpolation and common
   contiguous-axis passes.
2. Investigate projection/antialiasing temp-buffer traffic with a strategy that
   helps threaded/default paths, not only explicit single-thread mode.
3. Add a first-class benchmark report generator so these summaries can be
   reproduced from CSV artifacts without ad hoc parsing.
4. If a library PR becomes the target, package SciPy-like exact-semantics
   comparisons separately from OpenCV/PyTorch image-resize comparisons.
