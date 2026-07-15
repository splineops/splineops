"""Benchmark TensorSpline construction and evaluation runtime and memory.

The benchmark separates coefficient construction from evaluation, reuses one
constructed spline for repeated calls, and records Python/NumPy traced peak
allocations. It is intentionally independent of resize: TensorSpline models
arbitrary continuous coordinates and has a different workload contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from splineops import TensorSpline, __version__


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    query_shape: tuple[int, ...]
    basis: str
    mode: str = "mirror"
    query_kind: str = "grid"


SMOKE_CASES = (
    Case("1d_cubic_grid", (1024,), (4096,), "bspline3"),
    Case("2d_cubic_grid", (64, 64), (96, 96), "bspline3"),
    Case("2d_cubic_points", (64, 64), (20_000,), "bspline3", query_kind="points"),
    Case("3d_linear_grid", (16, 24, 24), (24, 32, 32), "linear"),
)

STANDARD_CASES = (
    Case("1d_cubic_grid", (4096,), (32768,), "bspline3"),
    Case("2d_cubic_grid", (128, 128), (256, 256), "bspline3"),
    Case("2d_cubic_points", (128, 128), (200_000,), "bspline3", query_kind="points"),
    Case("3d_linear_grid", (32, 48, 48), (48, 64, 64), "linear"),
    Case("3d_cubic_grid", (20, 24, 24), (28, 32, 32), "bspline3"),
)


def _measure(
    call: Callable[[], object], repeats: int, warmups: int
) -> tuple[float, float, int]:
    for _ in range(warmups):
        call()

    timings = []
    peak = 0
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        call()
        timings.append(time.perf_counter() - started)
        _, call_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, call_peak)
    return min(timings), float(np.median(timings)), peak


def _make_case(case: Case, dtype: np.dtype, seed: int):
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(case.shape).astype(dtype)
    construction = tuple(np.arange(length, dtype=dtype) for length in case.shape)
    if case.query_kind == "grid":
        query = tuple(
            np.linspace(0, source - 1, target, dtype=dtype)
            for source, target in zip(case.shape, case.query_shape)
        )
        grid = True
    else:
        count = case.query_shape[0]
        query = tuple(
            rng.uniform(0, length - 1, size=count).astype(dtype)
            for length in case.shape
        )
        grid = False
    return data, construction, query, grid


def run_case(
    case: Case, dtype: np.dtype, repeats: int, warmups: int, seed: int
) -> dict:
    data, construction, query, grid = _make_case(case, dtype, seed)

    def construct():
        return TensorSpline(data, construction, bases=case.basis, modes=case.mode)

    construction_best, construction_median, construction_peak = _measure(
        construct, repeats, warmups
    )
    spline = construct()

    def evaluate():
        return spline(query, grid=grid)

    evaluation_best, evaluation_median, evaluation_peak = _measure(
        evaluate, repeats, warmups
    )
    output = evaluate()
    return {
        "case": case.name,
        "shape": list(case.shape),
        "query_shape": list(case.query_shape),
        "query_kind": case.query_kind,
        "basis": case.basis,
        "mode": case.mode,
        "dtype": dtype.name,
        "output_shape": list(output.shape),
        "construction_best_seconds": construction_best,
        "construction_median_seconds": construction_median,
        "construction_peak_bytes": construction_peak,
        "evaluation_best_seconds": evaluation_best,
        "evaluation_median_seconds": evaluation_median,
        "evaluation_peak_bytes": evaluation_peak,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            serialized = {
                key: (
                    "x".join(str(value) for value in item)
                    if isinstance(item, list)
                    else item
                )
                for key, item in row.items()
            }
            writer.writerow(serialized)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("--repeats must be positive and --warmups non-negative")

    cases = SMOKE_CASES if args.profile == "smoke" else STANDARD_CASES
    dtype = np.dtype(args.dtype)
    rows = []
    for index, case in enumerate(cases):
        row = run_case(case, dtype, args.repeats, args.warmups, args.seed + index)
        rows.append(row)
        print(
            f"{case.name:22s} construct={row['construction_median_seconds'] * 1e3:9.3f} ms "
            f"eval={row['evaluation_median_seconds'] * 1e3:9.3f} ms "
            f"peak={row['evaluation_peak_bytes'] / (1024 ** 2):8.2f} MiB"
        )

    metadata = {
        "schema_version": 1,
        "profile": args.profile,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "seed": args.seed,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "splineops": __version__,
        },
        "memory_metric": "tracemalloc peak bytes during one call",
        "results": rows,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    if args.output_csv:
        _write_csv(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
