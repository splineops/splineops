# Oblique Antialiasing Resize PR Brief

> [!WARNING]
> **Historical and superseded (2026-06-17).** This brief preserves the PR
> framing and benchmark evidence available at that checkpoint; its numerical
> rationale is not the current resize contract. In particular, public
> zero-shift projections with analysis degree one or greater now use stable
> direct compact cross-Gram rows. See the current
> [resize API guide](../docs/api/02_resize.rst) and
> [resize user guide](../docs/user-guide/02_resize.rst).

Date: 2026-06-17

## Headline

Fast N-D spline resize with oblique-projection antialiasing, especially for
3-D/volumetric downsampling.

## Scope

Promote one public method family:

- `linear-antialiasing`
- `quadratic-antialiasing`
- `cubic-antialiasing`

These presets are oblique projection methods. Equal-degree least-squares
remains an advanced/reference configuration through `resize_degrees`, but it
is not part of the PR-facing public story.

## Core Claim

The oblique antialiasing presets provide a faster and more robust production
downsampling method than equal-degree least-squares, while retaining spline
semantics and strong N-D behavior.

The clearest differentiator is 3-D cubic antialiasing:

| Case | splineops | SciPy | skimage | OpenCV | Torch |
| --- | ---: | ---: | ---: | --- | --- |
| `3d_cubic_aa_down_random_f32` | `2.454 ms` | `8.070 ms` | `10.186 ms` | skipped, 2-D only | skipped, no 3-D AA |
| `3d_cubic_aa_down_random_f32_large` | `4.179 ms` | `23.610 ms` | `28.675 ms` | skipped, 2-D only | skipped, no 3-D AA |

## Evidence Bundle

Canonical artifact:

- `/tmp/splineops_resize_pr_oblique_pr_6411817_20260617/resize_pr_report_oblique_pr_6411817_20260617.md`

Reproduce with:

```shell
python scripts/benchmark_resize_pr.py \
  --profile oblique-pr \
  --output-dir /tmp/splineops_resize_pr_oblique_pr
```

Important rows from the current bundle:

| Evidence | Result |
| --- | --- |
| Native/Python full report | 46 overlaps, median `22.95x` speedup |
| Oblique antialiasing native/Python bucket | 13 cases, median `14.07x` |
| 3-D oblique antialiasing bucket | 3 cases, median `13.01x` |
| Oblique vs equal-degree LS, degree 1 | faster in `36/36`, median `1.25x` |
| Oblique vs equal-degree LS, degree 3 | faster in `36/36`, median `1.41x` |
| Exact-ish SciPy rows | SciPy faster in `0/16`, median speed `0.08x` |

## Non-Claims

Do not claim that splineops is the fastest generic 2-D image resize library.
OpenCV and Torch can be faster on some 2-D image-resize rows, but those rows
use different coordinate rules, boundary conditions, antialiasing filters, or
dtype behavior.

Do not claim oblique has higher PSNR than equal-degree least-squares in every
round-trip benchmark. Equal-degree least-squares is still the orthogonal
projection reference and can win small controlled PSNR rows. The production
argument is speed, lower integration order, long-line robustness, and N-D
coverage.

## Suggested PR Framing

1. Introduce the public method as oblique-projection antialiasing for spline
   resize.
2. Lead with correctness: native/Python parity and bounded long-ramp behavior.
3. Lead performance with 3-D cubic antialiasing and exact-ish SciPy-compatible
   rows.
4. Keep OpenCV, skimage, and Torch as contextual baselines, not exact
   competitors.
5. Keep equal-degree least-squares in the background as reference machinery,
   not as a promoted user-facing method.
