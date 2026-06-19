#!/usr/bin/env python3
"""Audit splineops resize against scipy.ndimage.zoom.

This script is intentionally narrower than ``benchmark_resize_libraries.py``.
It varies SciPy's public ``zoom`` semantics knobs and separates rows that are
credible same-semantics optimization targets from rows that are only useful as
context.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import warnings
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable


os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCIPY_ZOOM_DOC = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html"
SCIPY_BENCHMARK_DOC = "https://docs.scipy.org/doc/scipy/dev/contributor/benchmarking.html"
SCIPY_INTERPOLATION_SRC = "https://github.com/scipy/scipy/blob/main/scipy/ndimage/src/ni_interpolation.c"
SCIPY_SPLINES_SRC = "https://github.com/scipy/scipy/blob/main/scipy/ndimage/src/ni_splines.c"


@dataclass(frozen=True)
class ZoomCase:
    name: str
    shape: tuple[int, ...]
    zoom: tuple[float, ...]
    order: int
    dtype: str
    pattern: str


@dataclass(frozen=True)
class ScipyVariant:
    name: str
    mode: str
    grid_mode: bool
    prefilter_policy: str


@dataclass
class ZoomAuditResult:
    case: str
    status: str
    reason: str | None
    shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None
    scipy_output_shape: tuple[int, ...] | None
    zoom: tuple[float, ...]
    order: int
    splineops_method: str
    dtype: str
    pattern: str
    scipy_mode: str
    scipy_grid_mode: bool
    scipy_prefilter: bool
    scipy_variant: str
    repeats: int
    warmups: int
    splineops_best_ms: float | None
    splineops_median_ms: float | None
    splineops_mean_ms: float | None
    scipy_best_ms: float | None
    scipy_median_ms: float | None
    scipy_mean_ms: float | None
    potential_scipy_speedup: float | None
    max_abs_diff: float | None
    mean_abs_diff: float | None
    p99_abs_diff: float | None
    rel_l2_diff: float | None
    exactish: bool
    same_semantics_candidate: bool
    interpretation: str


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def method_for_order(order: int) -> str:
    mapping = {
        0: "fast",
        1: "linear",
        2: "quadratic",
        3: "cubic",
    }
    try:
        return mapping[int(order)]
    except KeyError as exc:
        raise ValueError(f"splineops does not expose order {order}") from exc


def module_version(name: str) -> str | None:
    try:
        if name == "splineops":
            return importlib.metadata.version(name)
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "<unknown>"))


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "<unknown>"


def dtype_from_name(name: str) -> np.dtype:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported dtype {name!r}")


def output_shape_for(case: ZoomCase) -> tuple[int, ...]:
    return tuple(max(1, int(round(n * z))) for n, z in zip(case.shape, case.zoom))


def stable_seed(case: ZoomCase) -> int:
    payload = (
        f"{case.name}|{case.shape}|{case.zoom}|{case.order}|"
        f"{case.dtype}|{case.pattern}"
    )
    return zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF


def make_input(case: ZoomCase) -> np.ndarray:
    dtype = dtype_from_name(case.dtype)
    rng = np.random.default_rng(stable_seed(case))
    shape = case.shape

    if case.pattern == "random":
        return rng.random(shape, dtype=dtype)
    if case.pattern == "constant":
        return np.full(shape, 3.25, dtype=dtype)
    if case.pattern == "impulse":
        out = np.zeros(shape, dtype=dtype)
        out[tuple(n // 2 for n in shape)] = 1.0
        return out
    if case.pattern == "ramp":
        grids = np.meshgrid(
            *[np.linspace(0.0, 1.0, n, dtype=np.float64) for n in shape],
            indexing="ij",
        )
        return (sum(grids) / float(len(grids))).astype(dtype, copy=False)
    if case.pattern == "sinusoid":
        grids = np.meshgrid(
            *[np.arange(n, dtype=np.float64) for n in shape],
            indexing="ij",
        )
        out = np.zeros(shape, dtype=np.float64)
        for axis, grid in enumerate(grids):
            out += np.sin(2.0 * np.pi * (0.07 + 0.05 * axis) * grid)
        return (out / float(len(grids))).astype(dtype, copy=False)
    if case.pattern == "checkerboard":
        grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        board = sum((grid // 4) for grid in grids) % 2
        return board.astype(dtype, copy=False)

    raise ValueError(f"unknown pattern {case.pattern!r}")


def _case(
    name: str,
    shape: tuple[int, ...],
    zoom: tuple[float, ...],
    order: int,
    dtype: str,
    pattern: str,
) -> ZoomCase:
    if len(shape) != len(zoom):
        raise ValueError(f"{name}: shape and zoom rank differ")
    return ZoomCase(name, shape, zoom, order, dtype, pattern)


def smoke_cases() -> list[ZoomCase]:
    return [
        _case("1d_linear_down_ramp_f64", (129,), (0.5,), 1, "float64", "ramp"),
        _case("2d_linear_down_random_f32", (192, 160), (0.5, 0.5), 1, "float32", "random"),
        _case("2d_cubic_down_random_f32", (160, 192), (0.5, 0.5), 3, "float32", "random"),
    ]


def standard_cases() -> list[ZoomCase]:
    return smoke_cases() + [
        _case("2d_linear_aniso_random_f32", (384, 320), (1.0, 0.37), 1, "float32", "random"),
        _case("2d_linear_up_sinusoid_f64", (192, 224), (1.25, 1.25), 1, "float64", "sinusoid"),
        _case("2d_cubic_down_random_f64", (256, 256), (0.37, 0.37), 3, "float64", "random"),
        _case("2d_cubic_up_sinusoid_f32", (160, 192), (1.7, 1.25), 3, "float32", "sinusoid"),
        _case("2d_cubic_short_axis_f64", (7, 96), (1.0, 0.5), 3, "float64", "random"),
        _case("3d_linear_down_random_f32", (80, 72, 28), (0.5, 0.5, 0.5), 1, "float32", "random"),
        _case("3d_cubic_down_random_f32", (64, 56, 24), (0.5, 0.5, 0.5), 3, "float32", "random"),
    ]


def full_cases() -> list[ZoomCase]:
    return standard_cases() + [
        _case("1d_nearest_up_impulse_f64", (257,), (1.75,), 0, "float64", "impulse"),
        _case("1d_quadratic_down_ramp_f64", (257,), (0.37,), 2, "float64", "ramp"),
        _case("2d_quadratic_down_random_f32", (320, 320), (0.5, 0.5), 2, "float32", "random"),
        _case("2d_linear_down_random_f32_large", (768, 768), (0.37, 0.37), 1, "float32", "random"),
        _case("2d_cubic_down_random_f32_large", (768, 768), (0.37, 0.37), 3, "float32", "random"),
        _case("3d_linear_aniso_random_f32", (96, 96, 32), (1.0, 0.5, 1.0), 1, "float32", "random"),
        _case("3d_cubic_aniso_random_f32", (80, 80, 28), (1.0, 0.5, 1.0), 3, "float32", "random"),
    ]


def cases_for_profile(profile: str) -> list[ZoomCase]:
    if profile == "smoke":
        return smoke_cases()
    if profile == "standard":
        return standard_cases()
    if profile == "full":
        return full_cases()
    raise ValueError(f"unknown profile {profile!r}")


def variants_for_profile(profile: str) -> list[ScipyVariant]:
    if profile == "focused":
        return [
            ScipyVariant("mirror_grid_false_auto", "mirror", False, "auto"),
            ScipyVariant("reflect_grid_false_auto", "reflect", False, "auto"),
            ScipyVariant("mirror_grid_true_auto", "mirror", True, "auto"),
            ScipyVariant("mirror_grid_false_off", "mirror", False, "off"),
        ]
    if profile == "broad":
        variants: list[ScipyVariant] = []
        for mode in ("mirror", "reflect", "nearest", "constant", "wrap"):
            for grid_mode in (False, True):
                variants.append(
                    ScipyVariant(
                        f"{mode}_grid_{str(grid_mode).lower()}_auto",
                        mode,
                        grid_mode,
                        "auto",
                    )
                )
                variants.append(
                    ScipyVariant(
                        f"{mode}_grid_{str(grid_mode).lower()}_off",
                        mode,
                        grid_mode,
                        "off",
                    )
                )
        return variants
    raise ValueError(f"unknown variant profile {profile!r}")


def prefilter_for(policy: str, order: int) -> bool:
    if policy == "auto":
        return order > 1
    if policy == "off":
        return False
    if policy == "on":
        return True
    raise ValueError(f"unknown prefilter policy {policy!r}")


def resolved_variants(profile: str, order: int) -> list[tuple[ScipyVariant, bool]]:
    out: list[tuple[ScipyVariant, bool]] = []
    seen: set[tuple[str, bool, bool]] = set()
    for variant in variants_for_profile(profile):
        prefilter = prefilter_for(variant.prefilter_policy, order)
        key = (variant.mode, variant.grid_mode, prefilter)
        if key in seen:
            continue
        seen.add(key)
        out.append((variant, prefilter))
    return out


def time_call(
    func: Callable[[], np.ndarray],
    repeats: int,
    warmups: int,
) -> tuple[np.ndarray, float, float, float]:
    out: np.ndarray | None = None
    for _ in range(warmups):
        out = np.asarray(func())

    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        out = np.asarray(func())
        samples.append((time.perf_counter() - t0) * 1000.0)

    if out is None:
        out = np.asarray(func())
    return out, min(samples), float(statistics.median(samples)), float(statistics.fmean(samples))


def run_splineops(
    arr: np.ndarray,
    case: ZoomCase,
    out_shape: tuple[int, ...],
) -> np.ndarray:
    from splineops.resize import resize

    method = method_for_order(case.order)
    return np.asarray(resize(arr, output_size=out_shape, method=method))


def run_scipy(
    arr: np.ndarray,
    case: ZoomCase,
    out_shape: tuple[int, ...],
    variant: ScipyVariant,
    prefilter: bool,
) -> np.ndarray:
    from scipy import ndimage

    # Passing zoom as output/input gives SciPy the same requested output shape
    # policy as splineops for this audit. The result shape is still checked.
    zoom = tuple(float(new) / float(old) for new, old in zip(out_shape, arr.shape))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(
            ndimage.zoom(
                arr,
                zoom,
                order=case.order,
                mode=variant.mode,
                prefilter=prefilter,
                grid_mode=variant.grid_mode,
            )
        )


def quality_metrics(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float, float, float]:
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    p99_abs = float(np.quantile(diff, 0.99)) if diff.size else 0.0
    denom = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    rel_l2 = float(np.linalg.norm(diff.ravel()) / denom) if denom > 0.0 else 0.0
    return max_abs, mean_abs, p99_abs, rel_l2


def interpretation_for(
    *,
    exactish: bool,
    same_semantics_candidate: bool,
    variant: ScipyVariant,
    prefilter: bool,
    order: int,
) -> str:
    if same_semantics_candidate:
        return "same-semantics candidate"
    if not exactish:
        return "not close enough for upstream correctness evidence"
    if variant.mode != "mirror":
        return "close output but different boundary mode"
    if variant.grid_mode:
        return "close output but different grid_mode"
    if prefilter != (order > 1):
        return "close output but different prefilter policy"
    return "close output but not a first-pass target"


def audit_case(
    case: ZoomCase,
    *,
    variant_profile: str,
    repeats: int,
    warmups: int,
    exact_rel_l2: float,
) -> list[ZoomAuditResult]:
    arr = make_input(case)
    out_shape = output_shape_for(case)
    method = method_for_order(case.order)

    reference, ref_best, ref_median, ref_mean = time_call(
        lambda: run_splineops(arr, case, out_shape),
        repeats,
        warmups,
    )

    rows: list[ZoomAuditResult] = []
    for variant, prefilter in resolved_variants(variant_profile, case.order):
        variant_name = (
            f"{variant.mode}|grid_mode={variant.grid_mode}|prefilter={prefilter}"
        )
        try:
            scipy_out, scipy_best, scipy_median, scipy_mean = time_call(
                lambda v=variant, p=prefilter: run_scipy(arr, case, out_shape, v, p),
                repeats,
                warmups,
            )
            scipy_shape = tuple(int(n) for n in scipy_out.shape)
            if scipy_shape != out_shape:
                rows.append(
                    ZoomAuditResult(
                        case=case.name,
                        status="shape-mismatch",
                        reason=f"SciPy returned {scipy_shape}, expected {out_shape}",
                        shape=case.shape,
                        output_shape=out_shape,
                        scipy_output_shape=scipy_shape,
                        zoom=case.zoom,
                        order=case.order,
                        splineops_method=method,
                        dtype=case.dtype,
                        pattern=case.pattern,
                        scipy_mode=variant.mode,
                        scipy_grid_mode=variant.grid_mode,
                        scipy_prefilter=prefilter,
                        scipy_variant=variant_name,
                        repeats=repeats,
                        warmups=warmups,
                        splineops_best_ms=ref_best,
                        splineops_median_ms=ref_median,
                        splineops_mean_ms=ref_mean,
                        scipy_best_ms=scipy_best,
                        scipy_median_ms=scipy_median,
                        scipy_mean_ms=scipy_mean,
                        potential_scipy_speedup=None,
                        max_abs_diff=None,
                        mean_abs_diff=None,
                        p99_abs_diff=None,
                        rel_l2_diff=None,
                        exactish=False,
                        same_semantics_candidate=False,
                        interpretation="SciPy produced a different shape",
                    )
                )
                continue
            max_abs, mean_abs, p99_abs, rel_l2 = quality_metrics(scipy_out, reference)
            exactish = rel_l2 < exact_rel_l2
            same_semantics_candidate = (
                exactish
                and variant.mode == "mirror"
                and not variant.grid_mode
                and prefilter == (case.order > 1)
                and case.order in {1, 3}
            )
            rows.append(
                ZoomAuditResult(
                    case=case.name,
                    status="ok",
                    reason=None,
                    shape=case.shape,
                    output_shape=out_shape,
                    scipy_output_shape=scipy_shape,
                    zoom=case.zoom,
                    order=case.order,
                    splineops_method=method,
                    dtype=case.dtype,
                    pattern=case.pattern,
                    scipy_mode=variant.mode,
                    scipy_grid_mode=variant.grid_mode,
                    scipy_prefilter=prefilter,
                    scipy_variant=variant_name,
                    repeats=repeats,
                    warmups=warmups,
                    splineops_best_ms=ref_best,
                    splineops_median_ms=ref_median,
                    splineops_mean_ms=ref_mean,
                    scipy_best_ms=scipy_best,
                    scipy_median_ms=scipy_median,
                    scipy_mean_ms=scipy_mean,
                    potential_scipy_speedup=(
                        scipy_median / ref_median if ref_median > 0.0 else float("inf")
                    ),
                    max_abs_diff=max_abs,
                    mean_abs_diff=mean_abs,
                    p99_abs_diff=p99_abs,
                    rel_l2_diff=rel_l2,
                    exactish=exactish,
                    same_semantics_candidate=same_semantics_candidate,
                    interpretation=interpretation_for(
                        exactish=exactish,
                        same_semantics_candidate=same_semantics_candidate,
                        variant=variant,
                        prefilter=prefilter,
                        order=case.order,
                    ),
                )
            )
        except Exception as exc:
            rows.append(
                ZoomAuditResult(
                    case=case.name,
                    status="error",
                    reason=f"{type(exc).__name__}: {exc}",
                    shape=case.shape,
                    output_shape=out_shape,
                    scipy_output_shape=None,
                    zoom=case.zoom,
                    order=case.order,
                    splineops_method=method,
                    dtype=case.dtype,
                    pattern=case.pattern,
                    scipy_mode=variant.mode,
                    scipy_grid_mode=variant.grid_mode,
                    scipy_prefilter=prefilter,
                    scipy_variant=variant_name,
                    repeats=repeats,
                    warmups=warmups,
                    splineops_best_ms=ref_best,
                    splineops_median_ms=ref_median,
                    splineops_mean_ms=ref_mean,
                    scipy_best_ms=None,
                    scipy_median_ms=None,
                    scipy_mean_ms=None,
                    potential_scipy_speedup=None,
                    max_abs_diff=None,
                    mean_abs_diff=None,
                    p99_abs_diff=None,
                    rel_l2_diff=None,
                    exactish=False,
                    same_semantics_candidate=False,
                    interpretation="SciPy run failed",
                )
            )
    return rows


def _float_values(rows: Iterable[ZoomAuditResult], attr: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = getattr(row, attr)
        if value is not None and not math.isnan(float(value)):
            out.append(float(value))
    return out


def _fmt(value: float | None, *, digits: int = 2, suffix: str = "") -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"`{value:.{digits}f}{suffix}`"


def _fmt_sci(value: float | None, *, digits: int = 2) -> str:
    if value is None or math.isnan(value):
        return "n/a"
    return f"`{value:.{digits}e}`"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["No usable rows."]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def summarize_by_variant(rows: list[ZoomAuditResult]) -> list[list[str]]:
    out: list[list[str]] = []
    keys = sorted(
        {
            (row.scipy_mode, row.scipy_grid_mode, row.scipy_prefilter)
            for row in rows
            if row.status == "ok"
        }
    )
    for mode, grid_mode, prefilter in keys:
        group = [
            row
            for row in rows
            if row.status == "ok"
            and row.scipy_mode == mode
            and row.scipy_grid_mode == grid_mode
            and row.scipy_prefilter == prefilter
        ]
        speedups = _float_values(group, "potential_scipy_speedup")
        rel_l2 = _float_values(group, "rel_l2_diff")
        if not speedups:
            continue
        out.append(
            [
                f"`{mode}`",
                f"`{grid_mode}`",
                f"`{prefilter}`",
                str(len(group)),
                f"{sum(row.exactish for row in group)}/{len(group)}",
                f"{sum(row.same_semantics_candidate for row in group)}/{len(group)}",
                _fmt(statistics.median(speedups), suffix="x"),
                _fmt(statistics.fmean(speedups), suffix="x"),
                _fmt_sci(statistics.median(rel_l2) if rel_l2 else None),
            ]
        )
    return out


def render_source_audit(*, tag: str) -> str:
    return f"""# SciPy `ndimage.zoom` Source Audit

Tag: `{tag}`

This note identifies the likely implementation points for a same-semantics
`ndimage.zoom` acceleration prototype. It is not a patch.

## Current Public Surface

The first target should preserve the documented `scipy.ndimage.zoom` behavior:
order, mode, cval, prefilter and grid_mode must remain unchanged.

Primary API reference: {SCIPY_ZOOM_DOC}

## Likely Source Touch Points

| Area | SciPy file | Why it matters |
| --- | --- | --- |
| Python argument handling | `scipy/ndimage/_interpolation.py` | Owns `zoom` validation, output shape and public defaults. |
| Geometric interpolation loop | `scipy/ndimage/src/ni_interpolation.c` | Owns coordinate mapping, boundary handling, per-point spline weights and accumulation. |
| Spline prefilter recursion | `scipy/ndimage/src/ni_splines.c` | Owns spline poles and recursive prefilter setup for order > 1. |
| ndimage tests | `scipy/ndimage/tests/` | Needs parity tests before optimization. |
| ASV benchmarks | `benchmarks/benchmarks/` | Needs `ndimage.zoom` cases before implementation changes. |

## Optimization Hypotheses To Test

1. Precompute per-output source indices and weights per axis for fixed
   `shape`, `zoom`, `order`, `mode`, `grid_mode` and `prefilter`.
2. Add specialized order-1 and order-3 paths before considering broader order
   coverage.
3. Batch contiguous or near-contiguous lines to reduce iterator overhead and
   improve cache behavior.
4. Keep the existing spline prefilter as the oracle first; optimize the gather
   and accumulation path before changing recursion code.
5. Gate any fast path behind exact parity tests and fall back to the existing
   generic path for unsupported modes or dtypes.

## First Patch Boundary

The first SciPy PR should be boring on purpose:

- no public API change,
- no new antialiasing option,
- no change to output shape, coordinate mapping or boundary behavior,
- benchmarks first, implementation second.

Source links:

- {SCIPY_INTERPOLATION_SRC}
- {SCIPY_SPLINES_SRC}
"""


def asv_case_specs(rows: list[ZoomAuditResult]) -> list[ZoomAuditResult]:
    candidates = [row for row in rows if row.same_semantics_candidate]
    selected: list[ZoomAuditResult] = []
    seen: set[tuple[str, int, str, tuple[int, ...], tuple[float, ...]]] = set()
    for row in sorted(
        candidates,
        key=lambda r: (
            -float(r.potential_scipy_speedup or 0.0),
            r.case,
        ),
    ):
        key = (row.dtype, row.order, row.pattern, row.shape, row.zoom)
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= 8:
            break
    return selected


def render_asv_benchmark(rows: list[ZoomAuditResult], *, tag: str) -> str:
    specs = asv_case_specs(rows)
    if not specs:
        specs = [
            ZoomAuditResult(
                case="2d_linear_down_float32",
                status="ok",
                reason=None,
                shape=(512, 512),
                output_shape=(256, 256),
                scipy_output_shape=(256, 256),
                zoom=(0.5, 0.5),
                order=1,
                splineops_method="linear",
                dtype="float32",
                pattern="random",
                scipy_mode="mirror",
                scipy_grid_mode=False,
                scipy_prefilter=False,
                scipy_variant="mirror|grid_mode=False|prefilter=False",
                repeats=0,
                warmups=0,
                splineops_best_ms=None,
                splineops_median_ms=None,
                splineops_mean_ms=None,
                scipy_best_ms=None,
                scipy_median_ms=None,
                scipy_mean_ms=None,
                potential_scipy_speedup=None,
                max_abs_diff=None,
                mean_abs_diff=None,
                p99_abs_diff=None,
                rel_l2_diff=None,
                exactish=True,
                same_semantics_candidate=True,
                interpretation="fallback ASV case",
            )
        ]

    case_lines = []
    for row in specs:
        shape = tuple(int(n) for n in row.shape)
        zoom = tuple(float(z) for z in row.zoom)
        case_lines.append(
            f"    ({row.case!r}, {shape!r}, {zoom!r}, {row.order}, {row.dtype!r}, {row.pattern!r}),"
        )
    cases = "\n".join(case_lines)
    return f'''"""ASV benchmarks for scipy.ndimage.zoom same-semantics candidates.

Generated by splineops/scripts/audit_scipy_zoom.py with tag {tag}.
Copy this file into a SciPy checkout under benchmarks/benchmarks/ and run:

    spin bench -t ndimage_zoom_splineops.NdimageZoomSplineCandidates --compare main

or run the same benchmark from scipy/benchmarks with direct asv commands.
"""

from __future__ import annotations

import zlib

import numpy as np
from scipy import ndimage


CASES = [
{cases}
]


def _seed(name, shape, zoom, order, dtype, pattern):
    payload = f"{{name}}|{{shape}}|{{zoom}}|{{order}}|{{dtype}}|{{pattern}}"
    return zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF


def _make_input(name, shape, zoom, order, dtype_name, pattern):
    dtype = np.dtype(dtype_name)
    rng = np.random.default_rng(_seed(name, shape, zoom, order, dtype_name, pattern))
    if pattern == "random":
        return rng.random(shape, dtype=dtype)
    if pattern == "ramp":
        grids = np.meshgrid(
            *[np.linspace(0.0, 1.0, n, dtype=np.float64) for n in shape],
            indexing="ij",
        )
        return (sum(grids) / float(len(grids))).astype(dtype, copy=False)
    if pattern == "sinusoid":
        grids = np.meshgrid(
            *[np.arange(n, dtype=np.float64) for n in shape],
            indexing="ij",
        )
        out = np.zeros(shape, dtype=np.float64)
        for axis, grid in enumerate(grids):
            out += np.sin(2.0 * np.pi * (0.07 + 0.05 * axis) * grid)
        return (out / float(len(grids))).astype(dtype, copy=False)
    if pattern == "impulse":
        out = np.zeros(shape, dtype=dtype)
        out[tuple(n // 2 for n in shape)] = 1.0
        return out
    raise NotImplementedError(pattern)


class NdimageZoomSplineCandidates:
    params = [CASES]
    param_names = ["case"]

    def setup(self, case):
        name, shape, zoom, order, dtype_name, pattern = case
        self.x = _make_input(name, shape, zoom, order, dtype_name, pattern)
        self.zoom = zoom
        self.order = int(order)
        self.prefilter = self.order > 1

    def time_zoom_mirror_grid_false(self, case):
        ndimage.zoom(
            self.x,
            self.zoom,
            order=self.order,
            mode="mirror",
            prefilter=self.prefilter,
            grid_mode=False,
        )
'''


def render_report(
    *,
    tag: str,
    rows: list[ZoomAuditResult],
    metadata: dict[str, object],
    csv_path: Path,
    json_path: Path,
    asv_path: Path,
    source_audit_path: Path,
    exact_rel_l2: float,
) -> str:
    ok_rows = [row for row in rows if row.status == "ok"]
    exact_rows = [row for row in ok_rows if row.exactish]
    candidates = [row for row in ok_rows if row.same_semantics_candidate]
    candidate_speedups = _float_values(candidates, "potential_scipy_speedup")

    lines = [
        "# SciPy `ndimage.zoom` Audit",
        "",
        f"Tag: `{tag}`",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(
        _markdown_table(
            ["Kind", "Path"],
            [
                ["CSV", f"`{csv_path}`"],
                ["JSON", f"`{json_path}`"],
                ["SciPy ASV benchmark stub", f"`{asv_path}`"],
                ["Source audit", f"`{source_audit_path}`"],
            ],
        )
    )
    lines.extend(
        [
            "",
            "## Environment",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Field", "Value"],
            [[str(key), f"`{value}`"] for key, value in metadata.items()],
        )
    )
    lines.extend(
        [
            "",
            "## Stage-Gate Summary",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ["Scope", "Rows", "Median Potential SciPy Speedup", "Mean Potential SciPy Speedup"],
            [
                [
                    "All successful rows",
                    str(len(ok_rows)),
                    _fmt(
                        statistics.median(_float_values(ok_rows, "potential_scipy_speedup"))
                        if ok_rows
                        else None,
                        suffix="x",
                    ),
                    _fmt(
                        statistics.fmean(_float_values(ok_rows, "potential_scipy_speedup"))
                        if ok_rows
                        else None,
                        suffix="x",
                    ),
                ],
                [
                    f"Exact-ish rows (`rel_l2 < {exact_rel_l2:g}`)",
                    str(len(exact_rows)),
                    _fmt(
                        statistics.median(_float_values(exact_rows, "potential_scipy_speedup"))
                        if exact_rows
                        else None,
                        suffix="x",
                    ),
                    _fmt(
                        statistics.fmean(_float_values(exact_rows, "potential_scipy_speedup"))
                        if exact_rows
                        else None,
                        suffix="x",
                    ),
                ],
                [
                    "Same-semantics first-pass candidates",
                    str(len(candidates)),
                    _fmt(
                        statistics.median(candidate_speedups)
                        if candidate_speedups
                        else None,
                        suffix="x",
                    ),
                    _fmt(
                        statistics.fmean(candidate_speedups)
                        if candidate_speedups
                        else None,
                        suffix="x",
                    ),
                ],
            ],
        )
    )
    lines.extend(
        [
            "",
            "A potential SciPy speedup above `1.0x` means the installed SciPy row",
            "was slower than splineops for the same audited input and SciPy public",
            "semantics.",
            "",
            "## By SciPy Semantic Variant",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            [
                "mode",
                "grid_mode",
                "prefilter",
                "Rows",
                "Exact-ish",
                "First-pass candidates",
                "Median speedup",
                "Mean speedup",
                "Median rel-L2",
            ],
            summarize_by_variant(rows),
        )
    )
    lines.extend(
        [
            "",
            "## First-Pass Candidate Rows",
            "",
        ]
    )
    candidate_rows: list[list[str]] = []
    for row in sorted(
        candidates,
        key=lambda r: (
            -float(r.potential_scipy_speedup or 0.0),
            r.case,
        ),
    ):
        candidate_rows.append(
            [
                row.case,
                f"`{row.shape}`",
                f"`{row.zoom}`",
                f"`{row.order}`",
                f"`{row.dtype}`",
                _fmt(row.splineops_median_ms, suffix=" ms"),
                _fmt(row.scipy_median_ms, suffix=" ms"),
                _fmt(row.potential_scipy_speedup, suffix="x"),
                _fmt_sci(row.rel_l2_diff),
            ]
        )
    lines.extend(
        _markdown_table(
            [
                "Case",
                "Shape",
                "Zoom",
                "Order",
                "Dtype",
                "splineops median",
                "SciPy median",
                "Potential speedup",
                "rel-L2",
            ],
            candidate_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Treat only first-pass candidate rows as evidence for optimizing",
            "  existing `scipy.ndimage.zoom` semantics.",
            "- Rows that are exact-ish but use a different `mode`, `grid_mode` or",
            "  `prefilter` setting are useful for source study, but should not be",
            "  presented as first PR targets.",
            "- Rows that are not exact-ish are evidence that the libraries are doing",
            "  different operations, not evidence that either implementation is wrong.",
            "- The ASV benchmark stub is deliberately SciPy-only. It does not import",
            "  splineops and is suitable as a starting point for a SciPy PR branch.",
            "",
            "## Sources",
            "",
            f"- SciPy `ndimage.zoom`: {SCIPY_ZOOM_DOC}",
            f"- SciPy benchmarking guide: {SCIPY_BENCHMARK_DOC}",
            f"- SciPy interpolation source entry point: {SCIPY_INTERPOLATION_SRC}",
        ]
    )
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[ZoomAuditResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dict_rows = [asdict(row) for row in rows]
    fieldnames = list(dict_rows[0]) if dict_rows else list(ZoomAuditResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dict_rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def print_result(row: ZoomAuditResult) -> None:
    if row.status != "ok":
        print(f"{row.case:34s} {row.scipy_variant:42s} {row.status:14s} {row.reason}")
        return
    assert row.scipy_median_ms is not None
    assert row.splineops_median_ms is not None
    assert row.potential_scipy_speedup is not None
    assert row.rel_l2_diff is not None
    marker = "candidate" if row.same_semantics_candidate else ("exactish" if row.exactish else "context")
    print(
        f"{row.case:34s} "
        f"{row.scipy_variant:42s} "
        f"sp={row.splineops_median_ms:8.3f} ms "
        f"scipy={row.scipy_median_ms:8.3f} ms "
        f"potential={row.potential_scipy_speedup:6.2f}x "
        f"rel_l2={row.rel_l2_diff:9.2e} "
        f"{marker}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard", "full"), default="standard")
    parser.add_argument(
        "--variant-profile",
        choices=("focused", "broad"),
        default="focused",
        help="SciPy semantic knobs to audit.",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--exact-rel-l2", type=float, default=1e-5)
    parser.add_argument("--tag", default=timestamp_tag())
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory. Defaults to /tmp/splineops_scipy_zoom_audit_<tag>.",
    )
    parser.add_argument(
        "--splineops-threads",
        default="1",
        help="Value for LSRESIZE_NUM_THREADS; use 'default' to unset.",
    )
    parser.add_argument(
        "--splineops-accel",
        choices=("always", "auto", "never"),
        default="always",
        help="Value for SPLINEOPS_ACCEL.",
    )
    return parser.parse_args()


def configure_runtime(args: argparse.Namespace) -> None:
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")
    if args.exact_rel_l2 <= 0.0:
        raise SystemExit("--exact-rel-l2 must be positive")

    os.environ["SPLINEOPS_ACCEL"] = args.splineops_accel
    if args.splineops_threads == "default":
        os.environ.pop("LSRESIZE_NUM_THREADS", None)
    else:
        threads = int(args.splineops_threads)
        if threads <= 0:
            raise SystemExit("--splineops-threads must be positive or 'default'")
        os.environ["LSRESIZE_NUM_THREADS"] = str(threads)


def main() -> int:
    args = parse_args()
    configure_runtime(args)

    output_dir = args.output_dir or Path(f"/tmp/splineops_scipy_zoom_audit_{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"scipy_zoom_audit_{args.tag}.csv"
    json_path = output_dir / f"scipy_zoom_audit_{args.tag}.json"
    report_path = output_dir / f"scipy_zoom_audit_report_{args.tag}.md"
    asv_path = output_dir / f"scipy_zoom_asv_benchmark_{args.tag}.py"
    source_audit_path = output_dir / f"scipy_zoom_source_audit_{args.tag}.md"

    metadata: dict[str, object] = {
        "tag": args.tag,
        "profile": args.profile,
        "variant_profile": args.variant_profile,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "exact_rel_l2": args.exact_rel_l2,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": module_version("scipy"),
        "splineops": module_version("splineops"),
        "branch": git_output(["branch", "--show-current"]),
        "commit": git_output(["rev-parse", "--short", "HEAD"]),
        "SPLINEOPS_ACCEL": os.environ.get("SPLINEOPS_ACCEL", "<unset>"),
        "LSRESIZE_NUM_THREADS": os.environ.get("LSRESIZE_NUM_THREADS", "<default>"),
    }

    print("SciPy ndimage.zoom audit")
    print(
        f"profile={args.profile} variant_profile={args.variant_profile} "
        f"repeats={args.repeats} warmups={args.warmups}"
    )
    print(
        "versions="
        f"numpy={metadata['numpy']} scipy={metadata['scipy']} "
        f"splineops={metadata['splineops']}"
    )
    print(
        "runtime="
        f"SPLINEOPS_ACCEL={metadata['SPLINEOPS_ACCEL']} "
        f"LSRESIZE_NUM_THREADS={metadata['LSRESIZE_NUM_THREADS']}"
    )
    print()

    # Keep import/setup overhead out of the first timed row even when a caller
    # asks for ``--warmups 0`` in a fast smoke test.
    importlib.import_module("scipy.ndimage")
    importlib.import_module("splineops.resize")

    rows: list[ZoomAuditResult] = []
    for case in cases_for_profile(args.profile):
        case_rows = audit_case(
            case,
            variant_profile=args.variant_profile,
            repeats=args.repeats,
            warmups=args.warmups,
            exact_rel_l2=args.exact_rel_l2,
        )
        rows.extend(case_rows)
        for row in case_rows:
            print_result(row)
        print()

    source_audit_path.write_text(render_source_audit(tag=args.tag), encoding="utf-8")
    asv_path.write_text(render_asv_benchmark(rows, tag=args.tag), encoding="utf-8")
    write_csv(csv_path, rows)
    write_json(
        json_path,
        {
            "metadata": metadata,
            "rows": [asdict(row) for row in rows],
            "sources": {
                "scipy_zoom_doc": SCIPY_ZOOM_DOC,
                "scipy_benchmark_doc": SCIPY_BENCHMARK_DOC,
                "scipy_interpolation_source": SCIPY_INTERPOLATION_SRC,
                "scipy_splines_source": SCIPY_SPLINES_SRC,
            },
        },
    )
    report_path.write_text(
        render_report(
            tag=args.tag,
            rows=rows,
            metadata=metadata,
            csv_path=csv_path,
            json_path=json_path,
            asv_path=asv_path,
            source_audit_path=source_audit_path,
            exact_rel_l2=args.exact_rel_l2,
        ),
        encoding="utf-8",
    )

    candidates = [row for row in rows if row.same_semantics_candidate]
    speedups = _float_values(candidates, "potential_scipy_speedup")
    print("summary")
    print(f"rows={len(rows)} candidates={len(candidates)}")
    if speedups:
        print(
            "candidate potential speedup "
            f"median={statistics.median(speedups):.2f}x "
            f"mean={statistics.fmean(speedups):.2f}x"
        )
    print(f"report: {report_path}")
    print(f"asv: {asv_path}")
    print(f"source audit: {source_audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
