"""Benchmark vectorized multiscale axis passes against row/column dispatch."""

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

from splineops import __version__
from splineops.multiscale.pyramid import (
    get_pyramid_filter,
    reduce_1d,
    reduce_2d,
)
from splineops.multiscale.wavelets.haar import HaarWavelets


def _measure(call, repeats, warmups):
    for _ in range(warmups):
        call()
    samples = []
    peak = 0
    result = None
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
        _, current_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, current_peak)
    return result, float(np.median(samples)), peak


def _row_column_reduce(image, filter_, centered):
    rows = np.vstack([reduce_1d(row, filter_, centered) for row in image])
    return np.column_stack(
        [
            reduce_1d(rows[:, column], filter_, centered)
            for column in range(rows.shape[1])
        ]
    )


def _row_column_haar(image, wavelet):
    result = image.copy()
    for row in range(result.shape[0]):
        result[row] = wavelet._split(result[row])
    for column in range(result.shape[1]):
        result[:, column] = wavelet._split(result[:, column])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("repeats must be positive and warmups non-negative")

    shape = (128, 160) if args.profile == "smoke" else (1024, 1280)
    image = np.random.default_rng(args.seed).standard_normal(shape).astype(np.float64)
    reduce_filter, _, centered = get_pyramid_filter("Spline", 3)
    wavelet = HaarWavelets(scales=1)

    pyramid, pyramid_seconds, pyramid_peak = _measure(
        lambda: reduce_2d(image, reduce_filter, centered), args.repeats, args.warmups
    )
    pyramid_reference, pyramid_reference_seconds, pyramid_reference_peak = _measure(
        lambda: _row_column_reduce(image, reduce_filter, centered),
        args.repeats,
        args.warmups,
    )
    np.testing.assert_allclose(pyramid, pyramid_reference, rtol=2e-13, atol=1e-14)

    haar, haar_seconds, haar_peak = _measure(
        lambda: wavelet.analysis1(image), args.repeats, args.warmups
    )
    haar_reference, haar_reference_seconds, haar_reference_peak = _measure(
        lambda: _row_column_haar(image, wavelet), args.repeats, args.warmups
    )
    np.testing.assert_allclose(haar, haar_reference, rtol=0.0, atol=2e-15)

    rows = [
        {
            "operation": "pyramid_reduce_2d",
            "vectorized_median_seconds": pyramid_seconds,
            "row_column_median_seconds": pyramid_reference_seconds,
            "speedup": pyramid_reference_seconds / pyramid_seconds,
            "vectorized_peak_bytes": pyramid_peak,
            "row_column_peak_bytes": pyramid_reference_peak,
        },
        {
            "operation": "haar_analysis_2d",
            "vectorized_median_seconds": haar_seconds,
            "row_column_median_seconds": haar_reference_seconds,
            "speedup": haar_reference_seconds / haar_seconds,
            "vectorized_peak_bytes": haar_peak,
            "row_column_peak_bytes": haar_reference_peak,
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
        "configuration": {"profile": args.profile, "shape": list(shape)},
        "results": rows,
    }
    for row in rows:
        print(
            f"{row['operation']:22s} vectorized={row['vectorized_median_seconds']:.6f}s "
            f"row-column={row['row_column_median_seconds']:.6f}s "
            f"speedup={row['speedup']:.2f}x"
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
