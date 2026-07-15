"""Benchmark complete repeated-geometry and explicit-axis workflows."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from splineops import __version__
from splineops.adaptive_regression_splines import DenoisingPlan
from splineops.affine import AffinePlan
from splineops.multiscale.wavelets.haar import HaarWavelets
from splineops.smoothing_splines import SmoothingSplinePlan


def _measure(call, repeats, warmups):
    for _ in range(warmups):
        call()
    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        timings.append(time.perf_counter() - started)
    return result, float(np.median(timings))


def _rotation(shape, degrees):
    radians = np.radians(-degrees)
    matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    center = (np.asarray(shape) - 1.0) / 2.0
    return matrix, center - matrix @ center


def _max_difference(first, second):
    if isinstance(first, tuple):
        return max(_max_difference(a, b) for a, b in zip(first, second))
    return float(np.max(np.abs(first - second)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("repeats must be positive and warmups non-negative")

    spatial_shape = (64, 80) if args.profile == "smoke" else (256, 320)
    batch_shape = (2, spatial_shape[0], spatial_shape[1], 2)
    rng = np.random.default_rng(args.seed)
    rows = []

    image = rng.standard_normal(spatial_shape)
    first_matrix, first_offset = _rotation(spatial_shape, 9.0)
    second_matrix, second_offset = _rotation(spatial_shape, -13.0)
    first_plan = AffinePlan(
        spatial_shape, first_matrix, first_offset, degree=3, mode="mirror"
    )
    second_plan = AffinePlan(
        spatial_shape, second_matrix, second_offset, degree=3, mode="mirror"
    )

    def ordinary_affine():
        return first_plan(image), second_plan(image)

    def prepared_affine():
        coefficients = first_plan.prefilter(image)
        return (
            first_plan.apply_coefficients(coefficients),
            second_plan.apply_coefficients(coefficients),
        )

    reference, reference_seconds = _measure(ordinary_affine, args.repeats, args.warmups)
    optimized, optimized_seconds = _measure(prepared_affine, args.repeats, args.warmups)
    rows.append(
        {
            "workflow": "affine_two_geometries_one_prefilter",
            "optimized_seconds": optimized_seconds,
            "reference_seconds": reference_seconds,
            "speedup": reference_seconds / optimized_seconds,
            "max_abs_difference": _max_difference(optimized, reference),
            "retained_bytes": first_plan.retained_bytes + second_plan.retained_bytes,
        }
    )

    batch = rng.standard_normal(batch_shape)
    smoothing_plan = SmoothingSplinePlan(spatial_shape, lamb=0.2, gamma=1.3)

    def batched_smoothing():
        return smoothing_plan(batch, axes=(1, 2))

    def looped_smoothing():
        result = np.empty(batch_shape)
        for batch_index in range(batch_shape[0]):
            for channel in range(batch_shape[-1]):
                result[batch_index, :, :, channel] = smoothing_plan(
                    batch[batch_index, :, :, channel]
                )
        return result

    reference, reference_seconds = _measure(
        looped_smoothing, args.repeats, args.warmups
    )
    optimized, optimized_seconds = _measure(
        batched_smoothing, args.repeats, args.warmups
    )
    rows.append(
        {
            "workflow": "smoothing_explicit_axes",
            "optimized_seconds": optimized_seconds,
            "reference_seconds": reference_seconds,
            "speedup": reference_seconds / optimized_seconds,
            "max_abs_difference": _max_difference(optimized, reference),
            "retained_bytes": smoothing_plan.retained_bytes,
        }
    )

    sample_count = 96 if args.profile == "smoke" else 384
    x = np.arange(sample_count, dtype=np.float64)
    signal = np.sin(0.08 * x) + 0.05 * np.cos(0.73 * x)
    lambdas = (0.01, 0.02, 0.05, 0.1)
    denoising_plan = DenoisingPlan(x, rho=0.5)
    reference, reference_seconds = _measure(
        lambda: np.stack([denoising_plan.solve(signal, value) for value in lambdas]),
        args.repeats,
        args.warmups,
    )
    optimized, optimized_seconds = _measure(
        lambda: denoising_plan.solve_path(signal, lambdas),
        args.repeats,
        args.warmups,
    )
    rows.append(
        {
            "workflow": "denoising_lambda_path",
            "optimized_seconds": optimized_seconds,
            "reference_seconds": reference_seconds,
            "speedup": reference_seconds / optimized_seconds,
            "max_abs_difference": _max_difference(optimized, reference),
            "retained_bytes": denoising_plan.retained_array_bytes,
        }
    )

    wavelet = HaarWavelets(scales=2)

    def batched_wavelet():
        return wavelet.analysis(batch, spatial_axes=(1, 2))

    def looped_wavelet():
        result = np.empty(batch_shape)
        for batch_index in range(batch_shape[0]):
            for channel in range(batch_shape[-1]):
                result[batch_index, :, :, channel] = wavelet.analysis(
                    batch[batch_index, :, :, channel]
                )
        return result

    reference, reference_seconds = _measure(looped_wavelet, args.repeats, args.warmups)
    optimized, optimized_seconds = _measure(batched_wavelet, args.repeats, args.warmups)
    rows.append(
        {
            "workflow": "wavelet_explicit_axes",
            "optimized_seconds": optimized_seconds,
            "reference_seconds": reference_seconds,
            "speedup": reference_seconds / optimized_seconds,
            "max_abs_difference": _max_difference(optimized, reference),
            "retained_bytes": 0,
        }
    )

    for row in rows:
        print(
            f"{row['workflow']:38s} optimized={row['optimized_seconds']:.6f}s "
            f"reference={row['reference_seconds']:.6f}s "
            f"speedup={row['speedup']:.2f}x "
            f"max-error={row['max_abs_difference']:.3e}"
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
