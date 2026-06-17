# Resize Optimization Session Handoff

Date: 2026-06-17
Branch: `feature/publication`
Base head before this handoff: `bf34f95 Finalize oblique resize PR package`

This is the close-out handoff for the resize optimization work done across
`splineops` and the local legacy/reference material. It summarizes what was
implemented, what was measured, what was rejected, and what should happen next.

## Executive State

The branch is ready for submission-mode review. The strongest public story is:

> Fast N-D spline resize with oblique-projection antialiasing, especially for
> 3-D/volumetric downsampling.

The public method family to promote is:

- `linear-antialiasing`
- `quadratic-antialiasing`
- `cubic-antialiasing`

Equal-degree least-squares projection remains available through
`resize_degrees` for advanced/reference use, but it is not the recommended
routine downsampling preset.

## Final Commits

Most recent commits on `feature/publication`:

| Commit | Purpose |
| --- | --- |
| `bf34f95` | Final PR package, fresh benchmark brief, PR draft, strict docs warning fix |
| `6411817` | PR wrapper profiles, oblique PR brief, public docs/docstrings |
| `8129791` | 3-D oblique resize batching and quadratic antialiasing specialization |
| `7fafed3` | Benchmark/story refocus around oblique projection |
| `9a0da90` | Projection method tradeoff documentation |
| `b8d3d49` | PR benchmark wrapper |
| `cde4671` | Rejected projection integration fusion documented |
| `1761de4` | Projection output prefilter scaling fusion |
| `a9a97a8` | Composite operator experiment |
| `7eb3158` | Single-thread cubic-antialiasing batch optimization |

## Files To Open First

- `scripts/resize_upstream_pr_description.md`
  Ready-to-paste PR text.
- `scripts/resize_oblique_pr_brief.md`
  Short PR-facing evidence and non-claims.
- `scripts/resize_pr_readiness.md`
  Checklist and interpretation guide.
- `scripts/resize_projection_method_comparison.md`
  Focused least-squares versus oblique comparison.
- `scripts/resize_optimization_notes.md`
  Detailed engineering ledger.
- `scripts/resize_optimization_progress.md`
  Earlier project-level summary; this handoff supersedes it for final state.

## Fresh Evidence Bundle

Canonical fresh bundle from commit `6411817`:

```text
/tmp/splineops_resize_pr_oblique_pr_6411817_20260617/
```

Main report:

```text
/tmp/splineops_resize_pr_oblique_pr_6411817_20260617/resize_pr_report_oblique_pr_6411817_20260617.md
```

Reproduce:

```shell
python scripts/benchmark_resize_pr.py \
  --profile oblique-pr \
  --output-dir /tmp/splineops_resize_pr_oblique_pr_6411817_20260617 \
  --tag oblique_pr_6411817_20260617
```

Key metrics:

| Evidence | Result |
| --- | --- |
| Native/Python full report | 46 overlaps, median `22.95x` speedup |
| Oblique antialiasing native/Python bucket | 13 cases, median `14.07x` |
| 3-D oblique antialiasing bucket | 3 cases, median `13.01x` |
| Oblique vs equal-degree LS, degree 1 | faster in `36/36`, median `1.25x` |
| Oblique vs equal-degree LS, degree 3 | faster in `36/36`, median `1.41x` |
| Exact-ish SciPy rows | SciPy faster in `0/16`, median speed `0.08x` |

3-D cubic antialiasing comparison:

| Case | splineops | SciPy | skimage | OpenCV | Torch |
| --- | ---: | ---: | ---: | --- | --- |
| `3d_cubic_aa_down_random_f32` | `2.454 ms` | `8.070 ms` | `10.186 ms` | skipped, 2-D only | skipped, no 3-D AA |
| `3d_cubic_aa_down_random_f32_large` | `4.179 ms` | `23.610 ms` | `28.675 ms` | skipped, 2-D only | skipped, no 3-D AA |

## Method Decision

The key method decision is to promote oblique projection, not equal-degree
least-squares, as the production downsampling path.

Reasoning:

- Legacy/reference code confirms the cost model: projection performs
  `analy_degree + 1` integrations before resampling and the matching
  differences after resampling.
- Cubic equal-degree least-squares therefore needs fourth-order integration per
  projected axis; cubic oblique needs second-order integration.
- Equal-degree least-squares remains the orthogonal projection reference and
  wins some small PSNR rows.
- Oblique is faster in every focused timing row measured for degree 1 and
  degree 3, and is much more stable on long lines.
- Long-ramp stability checks show cubic equal-degree least-squares can amplify
  floating-point error badly at large lengths, while cubic oblique remains
  bounded.

The PR should not claim oblique has higher PSNR everywhere. The production
argument is speed, lower integration order, long-line robustness, and N-D
coverage.

## Implemented Optimization Areas

Native backend:

- Batched N-D axis pipeline with automatic routing.
- Automatic batched projection routing for 3-D oblique antialiasing presets.
- Fixed-support accumulator specializations for common presets, including
  quadratic antialiasing.
- Row-major batched gather with exact boundary source/sign mapping.
- Strided-offset gather for routed pure interpolation cases.
- Gather-prefilter normalization fusion for supported exact paths.
- Projection output-prefilter scale fusion under conservative dispatch.
- Workload-aware threading and batch policies.

Exact linear fast paths:

- Direct N-D pure-linear interpolation path.
- Fused exact 2-D linear path.
- Fused exact 3-D linear path.
- Dedicated 3-D two-axis linear kernels.
- AVX2/FMA dispatch for selected 2-D linear rows where supported.
- Last-axis linear direct path to avoid generic offset unraveling.

Reusable plans:

- Public `ResizePlan`.
- Cached immutable 1-D/axis plans.
- Reused intermediate buffers for repeated fixed-geometry workloads.
- Caller-owned output support.
- Plan benchmark and report integration.

Precision policy:

- Conservative default precision preserved for projection/antialiasing.
- Native float32 internals enabled only for safe pure `float32`
  interpolation rows.
- Projection/antialiasing float32 internals kept opt-in due to measurable
  random-output drift.

Python fallback:

- Faster default block/support accumulation settings.
- Python plan cache defaults.
- Fallback remains the exact reference for native parity checks.

Benchmarking and reporting:

- `benchmark_resize_native.py`
- `benchmark_resize_libraries.py`
- `benchmark_resize_plan.py`
- `benchmark_resize_projection_methods.py`
- `benchmark_resize_pr.py`
- `summarize_resize_benchmarks.py`

The PR wrapper now has:

- `--profile smoke`
- `--profile oblique-pr`

Docs and public API framing:

- User guide recommends `*-antialiasing` presets for production downsampling.
- API docs and `resize()` docstrings describe those presets as oblique
  projection.
- `resize_degrees()` is framed as the advanced/reference degrees API.

## Rejected Or Deferred Ideas

Rejected for the current PR:

- Composite projection operator reformulation as a replacement for the current
  separable projection path. It was interesting but too risky and not a clear
  implementation win.
- Fused `nb == 2` projection input integration. It was exact in focused checks
  but not a stable timing win.
- Broad projection strided-gather routing. Prior runs did not show enough
  durable end-to-end gain.
- Default float32 internals for projection/antialiasing. Quality drift remains
  measurable.
- Promoting equal-degree least-squares as the public default. It is the
  reference path, not the best production path.

Deferred follow-up work:

- Improve `ResizePlan` reuse for `linear-antialiasing`, which still has a weak
  row in the standard plan benchmark.
- Further 3-D oblique batching/autotuning.
- Reduce docs benchmark runtime; full gallery build spends most time in
  resize benchmarking examples.
- Consider additional targeted SIMD only after the upstream API/story is
  accepted.

## Cross-Library Positioning

SciPy:

- Best close-semantics baseline for many spline interpolation rows.
- In the fresh exact-ish rows, SciPy is faster in `0/16`.

OpenCV and Torch:

- Can be faster in some generic 2-D image-resize rows.
- Not exact semantic equivalents because coordinate rules, antialiasing
  filters, boundaries, kernels, and dtype handling differ.
- Torch has no comparable 3-D cubic antialiasing row in the benchmark.

skimage:

- Useful contextual baseline.
- Slower than splineops in the fresh library benchmark rows, with larger
  semantic deltas.

Do not lead with "fastest generic 2-D resize." Lead with exact spline
semantics, N-D coverage, and 3-D oblique antialiasing.

## Validation Completed

Latest validation from the final state:

- `python -m py_compile` on benchmark/report scripts.
- Fresh full `benchmark_resize_pr.py --profile oblique-pr`.
- `pytest -q`: `558 passed`.
- `python -m build --outdir /tmp/splineops_dist_check_6411817`.
- Strict Sphinx build with `-W --keep-going`.
- Clean venv install from built wheel.
- Clean-install PR smoke benchmark from the wheel.
- `git diff --check`.

Generated artifacts:

```text
/tmp/splineops_dist_check_6411817/
/tmp/splineops_docs_build_6411817/
/tmp/splineops_clean_pr_smoke_6411817/
```

Clean-install package check confirmed:

- package imported from `/tmp/splineops_clean_install_6411817/.../site-packages`
- native extension `splineops._lsresize` present
- `resize(..., method="cubic-antialiasing")` works from the installed wheel

## Current Repo State At Handoff

At the time this document was written:

- branch: `feature/publication`
- head before this handoff commit: `bf34f95`
- previous final package commit: `bf34f95 Finalize oblique resize PR package`
- expected next commit: this documentation handoff

The repository should remain clean after committing this file and its links.

## Recommended Next Steps

1. Push `feature/publication`.
2. Open the upstream PR using `scripts/resize_upstream_pr_description.md`.
3. Attach or summarize the fresh benchmark report from
   `/tmp/splineops_resize_pr_oblique_pr_6411817_20260617/`.
4. Keep the PR story focused on oblique antialiasing and exact N-D spline
   semantics.
5. Do not add more algorithmic work to this PR before review.
6. Create a separate follow-up branch for additional optimization experiments.

## Reviewer Response Prep

Likely questions and answers:

- Why oblique instead of least-squares?
  Oblique is faster, lower-order, and long-line stable; least-squares remains
  available as the orthogonal reference through `resize_degrees`.
- Why not claim fastest 2-D resize?
  OpenCV/Torch can win non-equivalent 2-D rows; this PR is about exact spline
  semantics, N-D behavior, and oblique antialiasing.
- Why include benchmark scripts?
  They make the claims reproducible and separate exact comparisons from
  contextual image-library baselines.
- Is the public API more complex now?
  No. The public recommendation is one method family: `*-antialiasing`.
  Advanced degree triples stay behind `resize_degrees`.
