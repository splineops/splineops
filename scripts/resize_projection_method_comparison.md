# Resize Projection Method Comparison

Date: 2026-06-17

This note records the focused comparison between equal-degree least-squares
projection and the oblique antialiasing presets.

## Method Mapping

In `splineops.resize`, the public presets encode spline degree triples:

| Method | Interp | Analysis | Synthesis |
| --- | ---: | ---: | ---: |
| `linear` | 1 | -1 | 1 |
| `cubic` | 3 | -1 | 3 |
| `linear-antialiasing` | 1 | 0 | 1 |
| `quadratic-antialiasing` | 2 | 1 | 2 |
| `cubic-antialiasing` | 3 | 1 | 3 |

Equal-degree least-squares is still available through `resize_degrees`, for
example `(interp=3, analy=3, synthe=3)`, but it is not a public preset.

The legacy Java implementation confirms the core cost model: projection calls
`doInteg(inputVector, analyDegree + 1)` before resampling and
`doDiff(addOutputVector, analyDegree + 1)` after resampling. Cubic
least-squares therefore uses four integrations/differences per projected axis;
cubic oblique uses two.

## Literature And Local Docs

The local resize guide makes the same distinction:

- least-squares is the orthogonal projection onto the output spline space;
- oblique projection uses a lower-degree analysis space while keeping the same
  synthesis space;
- oblique keeps the same approximation order and similar practical quality
  with much lower cost;
- exact equal-degree cubic least-squares is not recommended as a routine preset
  because the required high-order integration can amplify roundoff on long
  lines.

This matches the EPFL oblique-projection paper summary: the oblique method
achieves essentially the same performance as the least-squares solution with
considerably fewer computations.

## Local Sweep

Artifacts:

- `/tmp/splineops_projection_methods_legacy_relevant_20260617.csv`
- `/tmp/splineops_ls_vs_oblique_20260617.csv`

The focused sweep used:

- patterns: `camera`, `coins`, `low_sine`, `near_nyquist`, `checker`,
  `random`
- zooms: `(0.5, 0.5)`, `(0.37, 0.61)`, `(0.25, 0.25)`
- dtypes: `float32`, `float64`
- degrees: 1 and 3
- metrics: one-way timing, down-up round-trip PSNR, round-trip SSIM
- environment: native backend, explicit single thread

Summary:

| Degree | Dtype | Oblique faster | Median oblique speedup vs LS | PSNR wins LS/Oblique/Interp | SSIM wins LS/Oblique/Interp |
| ---: | --- | ---: | ---: | --- | --- |
| 1 | `float32` | 15/18 | `1.22x` | 12 / 6 / 0 | 4 / 10 / 4 |
| 1 | `float64` | 16/18 | `1.22x` | 12 / 6 / 0 | 4 / 10 / 4 |
| 3 | `float32` | 18/18 | `2.17x` | 15 / 2 / 1 | 4 / 10 / 4 |
| 3 | `float64` | 18/18 | `2.16x` | 15 / 2 / 1 | 4 / 10 / 4 |

The small-image round-trip PSNR metric gives least-squares a slight edge in
many rows, especially cubic. The median PSNR difference is small:

| Degree | Dtype | Median PSNR LS | Median PSNR Oblique |
| ---: | --- | ---: | ---: |
| 1 | `float32` | `18.14 dB` | `18.10 dB` |
| 1 | `float64` | `18.14 dB` | `18.10 dB` |
| 3 | `float32` | `18.43 dB` | `18.40 dB` |
| 3 | `float64` | `18.43 dB` | `18.40 dB` |

SSIM is mixed: oblique wins more individual cases, while cubic least-squares
has a slightly higher median SSIM in this small sweep.

## Long-Line Stability

The decisive difference is robustness on long lines. With a 1-D ramp, cubic
least-squares begins to amplify numerical error as line length grows, while
cubic oblique stays stable.

One-way downsample by `0.37`, `float64`, cubic:

| Input length | Method | Output min | Output max | Output mean |
| ---: | --- | ---: | ---: | ---: |
| 4096 | LS `(3,3,3)` | `-1.11e-04` | `1.0046` | `0.5000` |
| 4096 | Oblique `(3,1,3)` | `-1.96e-05` | `1.0000` | `0.5000` |
| 16384 | LS `(3,3,3)` | `-4.2869` | `5.8711` | `0.5001` |
| 16384 | Oblique `(3,1,3)` | `-9.73e-05` | `1.0001` | `0.5000` |
| 65536 | LS `(3,3,3)` | `-1563.49` | `1422.42` | `0.4940` |
| 65536 | Oblique `(3,1,3)` | `-3.42e-05` | `1.0000` | `0.5000` |

This is exactly the failure mode expected from high-order finite-difference
integration in floating point arithmetic.

## Conclusion

For a production default and for an upstream PR, the better method is oblique
projection antialiasing, not equal-degree least-squares projection.

Least-squares is still important:

- it is the orthogonal-projection reference;
- it can have a small quality edge on controlled, small round-trip benchmarks;
- it should remain available through `resize_degrees`.

But oblique is the better default:

- much faster, especially cubic;
- lower integration order;
- stable on long lines where cubic equal-degree least-squares can fail badly;
- similar practical quality and often better SSIM;
- aligned with the public `*-antialiasing` presets.

Optimization work should therefore continue to target the oblique projection
path first. Equal-degree least-squares should be treated as an advanced
reference path and tested for parity, but not used as the main PR narrative.
