"""Benchmark repeated TensorSpline coordinates with and without a query plan."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from splineops import TensorSpline, __version__


def _measure(call, repeats: int) -> tuple[float, int]:
    timings = []
    tracemalloc.start()
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        timings.append(time.perf_counter() - started)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return float(np.median(timings)), peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=200_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--shape", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.points <= 0 or args.repeats <= 0 or args.shape < 2:
        parser.error("points/repeats must be positive and shape at least two")

    rng = np.random.default_rng(args.seed)
    data = rng.standard_normal((args.shape, args.shape))
    construction = (np.arange(args.shape, dtype=float),) * 2
    query = tuple(rng.uniform(0.0, args.shape - 1.0, args.points) for _ in range(2))
    spline = TensorSpline(data, construction, bases="bspline3", modes="mirror")

    started = time.perf_counter()
    plan = spline.query_plan(query, grid=False)
    construction_seconds = time.perf_counter() - started
    ordinary_seconds, ordinary_peak = _measure(
        lambda: spline(query, grid=False), args.repeats
    )
    planned_seconds, planned_peak = _measure(plan.apply, args.repeats)
    np.testing.assert_equal(plan(), spline(query, grid=False))

    rows = [
        {
            "path": "ordinary",
            "median_seconds": ordinary_seconds,
            "peak_bytes": ordinary_peak,
            "plan_construction_seconds": 0.0,
            "retained_bytes": 0,
        },
        {
            "path": "query_plan",
            "median_seconds": planned_seconds,
            "peak_bytes": planned_peak,
            "plan_construction_seconds": construction_seconds,
            "retained_bytes": plan.retained_bytes,
        },
    ]
    payload = {
        "schema_version": 1,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "splineops": __version__,
        },
        "configuration": {
            "points": args.points,
            "repeats": args.repeats,
            "shape": [args.shape, args.shape],
            "basis": "bspline3",
            "mode": "mirror",
        },
        "speedup": ordinary_seconds / planned_seconds,
        "break_even_repetitions": construction_seconds
        / max(ordinary_seconds - planned_seconds, np.finfo(float).eps),
        "results": rows,
    }
    print(
        f"ordinary={ordinary_seconds:.6f}s planned={planned_seconds:.6f}s "
        f"speedup={payload['speedup']:.2f}x "
        f"break-even={payload['break_even_repetitions']:.2f} calls "
        f"retained={plan.retained_bytes / 2**20:.2f} MiB"
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
