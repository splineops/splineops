"""Measure runtime and traced-memory scaling across explicit batch sizes."""

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
from splineops.affine import AffinePlan
from splineops.differentials import DifferentialPlan
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


def _rotation(shape):
    radians = np.radians(-9.0)
    matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    center = (np.asarray(shape) - 1.0) / 2.0
    return matrix, center - matrix @ center


def _loop_affine(plan, data):
    return np.stack([plan(plane) for plane in data])


def _loop_laplacian(plan, data):
    return np.stack(
        [
            plan(plane, gradient=False, hessian=False, laplacian=True).laplacian
            for plane in data
        ]
    )


def _loop_wavelet(wavelet, data):
    return np.stack([wavelet.synthesis(wavelet.analysis(plane)) for plane in data])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("repeats must be positive and warmups non-negative")

    if args.profile == "smoke":
        shapes = ((32, 40), (64, 80))
        batch_counts = (1, 2, 4)
    else:
        shapes = ((64, 80), (128, 160), (256, 320))
        batch_counts = (1, 2, 4, 8)
    rng = np.random.default_rng(args.seed)
    rows = []
    baseline_peaks = {}

    for shape in shapes:
        matrix, offset = _rotation(shape)
        affine = AffinePlan(shape, matrix, offset, degree=3, mode="mirror")
        differentials = DifferentialPlan(shape, spacing=(0.7, 1.3))
        wavelet = HaarWavelets(scales=2)
        for batch_count in batch_counts:
            data = rng.standard_normal((batch_count,) + shape)
            cases = (
                (
                    "affine",
                    lambda: affine(data, spatial_axes=(1, 2)),
                    lambda: _loop_affine(affine, data),
                    affine.retained_bytes,
                ),
                (
                    "laplacian_only",
                    lambda: differentials(
                        data,
                        gradient=False,
                        hessian=False,
                        laplacian=True,
                        spatial_axes=(1, 2),
                    ).laplacian,
                    lambda: _loop_laplacian(differentials, data),
                    differentials.retained_bytes,
                ),
                (
                    "haar_roundtrip",
                    lambda: wavelet.synthesis(
                        wavelet.analysis(data, spatial_axes=(1, 2)),
                        spatial_axes=(1, 2),
                    ),
                    lambda: _loop_wavelet(wavelet, data),
                    0,
                ),
            )
            for operation, batched_call, looped_call, retained_bytes in cases:
                batched, seconds, peak = _measure(
                    batched_call, args.repeats, args.warmups
                )
                reference = looped_call()
                max_difference = float(np.max(np.abs(batched - reference)))
                output_bytes = batched.nbytes
                key = (operation, shape)
                if batch_count == 1:
                    baseline_peaks[key] = peak
                normalized_growth = peak / baseline_peaks[key] / batch_count
                row = {
                    "operation": operation,
                    "shape": "x".join(str(value) for value in shape),
                    "batch_count": batch_count,
                    "median_seconds": seconds,
                    "tracemalloc_peak_bytes": peak,
                    "required_output_bytes": output_bytes,
                    "temporary_overhead_bytes": max(0, peak - output_bytes),
                    "normalized_peak_growth": normalized_growth,
                    "retained_bytes": retained_bytes,
                    "max_abs_difference": max_difference,
                }
                rows.append(row)
                print(
                    f"{operation:16s} shape={row['shape']:7s} "
                    f"batch={batch_count:2d} time={seconds:.6f}s "
                    f"peak={peak / 2**20:.2f} MiB "
                    f"normalized-growth={normalized_growth:.2f} "
                    f"max-error={max_difference:.3e}"
                )

    payload = {
        "schema_version": 1,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "splineops": __version__,
        },
        "profile": args.profile,
        "summary": {
            "max_normalized_peak_growth": max(
                row["normalized_peak_growth"] for row in rows
            ),
            "max_abs_difference": max(row["max_abs_difference"] for row in rows),
        },
        "results": rows,
    }
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
