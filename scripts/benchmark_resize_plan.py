#!/usr/bin/env python3
"""Benchmark reusable ResizePlan against one-shot resize calls."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class PlanBenchCase:
    name: str
    shape: tuple[int, ...]
    zoom: tuple[float, ...]
    method: str
    dtype: str


@dataclass
class PlanBenchResult:
    case: str
    shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    zoom: tuple[float, ...]
    method: str
    dtype: str
    frames: int
    repeats: int
    oneshot_best_ms_per_frame: float
    oneshot_median_ms_per_frame: float
    plan_best_ms_per_frame: float
    plan_median_ms_per_frame: float
    median_speedup: float
    plan_into_best_ms_per_frame: float
    plan_into_median_ms_per_frame: float
    median_into_speedup: float
    max_abs_diff: float


def bench_cases(profile: str) -> list[PlanBenchCase]:
    smoke = [
        PlanBenchCase(
            "2d_cubic_down_f32", (256, 256), (0.37, 0.37), "cubic", "float32"
        ),
        PlanBenchCase(
            "2d_cubic_aa_down_f32",
            (256, 256),
            (0.37, 0.37),
            "cubic-antialiasing",
            "float32",
        ),
    ]
    if profile == "smoke":
        return smoke
    if profile != "standard":
        raise ValueError(f"unknown profile {profile!r}")
    return smoke + [
        PlanBenchCase(
            "2d_linear_down_f32", (512, 512), (0.37, 0.37), "linear", "float32"
        ),
        PlanBenchCase(
            "2d_linear_aniso_f32", (512, 512), (1.0, 0.37), "linear", "float32"
        ),
        PlanBenchCase(
            "2d_cubic_aniso_f32", (512, 512), (1.0, 0.37), "cubic", "float32"
        ),
        PlanBenchCase(
            "2d_linear_aa_down_f32",
            (512, 512),
            (0.37, 0.37),
            "linear-antialiasing",
            "float32",
        ),
        PlanBenchCase(
            "2d_cubic_down_f64", (512, 512), (0.37, 0.37), "cubic", "float64"
        ),
        PlanBenchCase(
            "3d_cubic_aniso_f32",
            (96, 96, 24),
            (1.0, 0.5, 1.0),
            "cubic",
            "float32",
        ),
        PlanBenchCase(
            "3d_linear_down_f32",
            (128, 128, 32),
            (0.5, 0.5, 0.5),
            "linear",
            "float32",
        ),
        PlanBenchCase(
            "3d_linear_two_axis01_f32",
            (128, 128, 32),
            (0.5, 0.5, 1.0),
            "linear",
            "float32",
        ),
        PlanBenchCase(
            "3d_cubic_aa_down_f32",
            (64, 64, 24),
            (0.5, 0.5, 0.5),
            "cubic-antialiasing",
            "float32",
        ),
    ]


def run_timed(
    fn, frames: list[np.ndarray], repeats: int, warmups: int
) -> tuple[float, float]:
    for _ in range(warmups):
        for frame in frames:
            fn(frame)

    times: list[float] = []
    for _ in range(repeats):
        t0 = perf_counter()
        for frame in frames:
            fn(frame)
        times.append((perf_counter() - t0) * 1000.0 / len(frames))
    return min(times), float(median(times))


def run_timed_into(
    fn,
    frames: list[np.ndarray],
    output: np.ndarray,
    repeats: int,
    warmups: int,
) -> tuple[float, float]:
    for _ in range(warmups):
        for frame in frames:
            fn(frame, output=output)

    times: list[float] = []
    for _ in range(repeats):
        t0 = perf_counter()
        for frame in frames:
            fn(frame, output=output)
        times.append((perf_counter() - t0) * 1000.0 / len(frames))
    return min(times), float(median(times))


def run_case(
    case: PlanBenchCase,
    *,
    frames: int,
    repeats: int,
    warmups: int,
    rng: np.random.Generator,
) -> PlanBenchResult:
    from splineops.resize import ResizePlan, resize

    dtype = np.dtype(case.dtype)
    data = [rng.random(case.shape, dtype=dtype) for _ in range(frames)]
    plan = ResizePlan(case.shape, zoom_factors=case.zoom, method=case.method)

    y_oneshot = resize(data[0], zoom_factors=case.zoom, method=case.method)
    y_plan = plan(data[0])
    output = np.empty_like(y_plan)
    y_into = plan(data[0], output=output)
    diff = np.abs(y_oneshot.astype(np.float64) - y_plan.astype(np.float64))
    diff_into = np.abs(y_oneshot.astype(np.float64) - y_into.astype(np.float64))
    max_abs = max(
        float(np.max(diff)) if diff.size else 0.0,
        float(np.max(diff_into)) if diff_into.size else 0.0,
    )

    oneshot_best, oneshot_median = run_timed(
        lambda x: resize(x, zoom_factors=case.zoom, method=case.method),
        data,
        repeats,
        warmups,
    )
    plan_best, plan_median = run_timed(plan, data, repeats, warmups)
    plan_into_best, plan_into_median = run_timed_into(
        plan.apply,
        data,
        output,
        repeats,
        warmups,
    )
    speedup = oneshot_median / plan_median if plan_median > 0.0 else float("inf")
    into_speedup = (
        oneshot_median / plan_into_median if plan_into_median > 0.0 else float("inf")
    )

    return PlanBenchResult(
        case=case.name,
        shape=case.shape,
        output_shape=tuple(int(n) for n in y_plan.shape),
        zoom=case.zoom,
        method=case.method,
        dtype=case.dtype,
        frames=frames,
        repeats=repeats,
        oneshot_best_ms_per_frame=oneshot_best,
        oneshot_median_ms_per_frame=oneshot_median,
        plan_best_ms_per_frame=plan_best,
        plan_median_ms_per_frame=plan_median,
        median_speedup=speedup,
        plan_into_best_ms_per_frame=plan_into_best,
        plan_into_median_ms_per_frame=plan_into_median,
        median_into_speedup=into_speedup,
        max_abs_diff=max_abs,
    )


def write_json(path: Path, results: list[PlanBenchResult]) -> None:
    payload = {
        "metadata": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "LSRESIZE_PRECISION": os.environ.get(
                "LSRESIZE_PRECISION",
                "<default:auto-f32-2d/3d-interp+3d-down-proj-else-float64>",
            ),
            "LSRESIZE_BATCHED_AXIS": os.environ.get("LSRESIZE_BATCHED_AXIS", "<unset>"),
        },
        "results": [asdict(r) for r in results],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, results: list[PlanBenchResult]) -> None:
    rows = [asdict(r) for r in results]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["SPLINEOPS_ACCEL"] = "always"
    rng = np.random.default_rng(args.seed)
    results = [
        run_case(
            case,
            frames=max(1, args.frames),
            repeats=max(1, args.repeats),
            warmups=max(0, args.warmups),
            rng=rng,
        )
        for case in bench_cases(args.profile)
    ]

    for r in results:
        print(
            f"{r.case:24s} "
            f"resize={r.oneshot_median_ms_per_frame:7.3f} ms/frame "
            f"plan={r.plan_median_ms_per_frame:7.3f} ms/frame "
            f"plan_out={r.plan_into_median_ms_per_frame:7.3f} ms/frame "
            f"speedup={r.median_speedup:5.2f}x "
            f"out_speedup={r.median_into_speedup:5.2f}x "
            f"diff={r.max_abs_diff:.3e}"
        )

    if args.output_json is not None:
        write_json(args.output_json, results)
        print(f"wrote JSON: {args.output_json}")
    if args.output_csv is not None:
        write_csv(args.output_csv, results)
        print(f"wrote CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
