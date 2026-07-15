"""Measure TensorSpline point-query memory growth across output sizes.

Coordinate arrays are allocated before tracing starts.  The reported peak
therefore contains the returned output and evaluation temporaries, but not the
benchmark's input coordinates.  A bounded tiled implementation should show an
approximately constant ``temporary_overhead_bytes`` once one tile is filled.
"""

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


def _parse_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "counts must be comma-separated integers"
        ) from exc
    if not counts or any(count <= 0 for count in counts):
        raise argparse.ArgumentTypeError("counts must be positive")
    return counts


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts", type=_parse_counts, default=(10_000, 100_000, 1_000_000)
    )
    parser.add_argument("--shape", type=int, default=128)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.shape < 2:
        parser.error("--shape must be at least two")

    dtype = np.dtype(args.dtype)
    rng = np.random.default_rng(args.seed)
    data = rng.standard_normal((args.shape, args.shape)).astype(dtype)
    construction = (
        np.arange(args.shape, dtype=dtype),
        np.arange(args.shape, dtype=dtype),
    )
    spline = TensorSpline(data, construction, bases="bspline3", modes="mirror")
    rows = []
    for count in args.counts:
        query = (
            rng.uniform(0, args.shape - 1, size=count).astype(dtype),
            rng.uniform(0, args.shape - 1, size=count).astype(dtype),
        )
        tracemalloc.start()
        started = time.perf_counter()
        output = spline(query, grid=False)
        seconds = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        output_bytes = output.nbytes
        row = {
            "query_count": count,
            "seconds": seconds,
            "peak_bytes": peak_bytes,
            "output_bytes": output_bytes,
            "temporary_overhead_bytes": max(0, peak_bytes - output_bytes),
            "bytes_per_output": peak_bytes / count,
        }
        rows.append(row)
        print(
            f"{count:>10,d} points  {seconds:8.3f} s  "
            f"peak={peak_bytes / 2**20:8.2f} MiB  "
            f"temporary={row['temporary_overhead_bytes'] / 2**20:8.2f} MiB"
        )

    payload = {
        "schema_version": 1,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "splineops": __version__,
        },
        "configuration": {
            "shape": [args.shape, args.shape],
            "basis": "bspline3",
            "mode": "mirror",
            "dtype": dtype.name,
            "tile_size": spline._EVALUATION_TILE_SIZE,
            "coordinate_memory_traced": False,
        },
        "results": rows,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    if args.output_csv:
        _write_csv(args.output_csv, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
