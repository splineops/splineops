"""Benchmark two realistic multi-module stability-soak workflows."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import platform
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from splineops import __version__
from splineops.affine import AffinePlan
from splineops.differentials import DifferentialPlan, DifferentialResult


def _measure(call, repeats, warmups):
    for _ in range(warmups):
        call()
    timings = []
    peak = 0
    result = None
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        result = call()
        timings.append(time.perf_counter() - started)
        _, current_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, current_peak)
    return result, float(np.median(timings)), peak


def _rotation_2d(shape, angle):
    radians = np.radians(-angle)
    matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    center = (np.asarray(shape) - 1.0) / 2.0
    return matrix, center - matrix @ center


def _rotation_3d(shape, angle):
    radians = np.radians(-angle)
    matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians), 0.0],
            [np.sin(radians), np.cos(radians), 0.0],
            [0.0, 0.0, 1.0],
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
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("repeats must be positive and warmups non-negative")

    image_shape = (48, 64) if args.profile == "smoke" else (192, 256)
    volume_shape = (10, 12, 14) if args.profile == "smoke" else (32, 40, 48)
    rng = np.random.default_rng(args.seed)
    rows = []

    frames = rng.standard_normal((2,) + image_shape)
    affine_plans = []
    for angle in (7.0, -11.0, 18.0):
        matrix, offset = _rotation_2d(image_shape, angle)
        affine_plans.append(
            AffinePlan(image_shape, matrix, offset, degree=3, mode="mirror")
        )

    def reference_registration():
        return tuple(tuple(plan(frame) for plan in affine_plans) for frame in frames)

    with tempfile.TemporaryDirectory() as directory:
        archive_directory = Path(directory)

        def persisted_registration():
            output = []
            with ThreadPoolExecutor(max_workers=len(affine_plans)) as executor:
                for index, frame in enumerate(frames):
                    field = affine_plans[0].prepare_coefficients(frame)
                    archive = archive_directory / f"frame-{index}.npz"
                    field.save(archive)
                    restored = affine_plans[-1].load_coefficients(archive)
                    output.append(
                        tuple(
                            executor.map(
                                lambda plan: plan.apply_coefficients(restored),
                                affine_plans,
                            )
                        )
                    )
            return tuple(output)

        reference, reference_seconds, reference_peak = _measure(
            reference_registration, args.repeats, args.warmups
        )
        optimized, optimized_seconds, optimized_peak = _measure(
            persisted_registration, args.repeats, args.warmups
        )
    rows.append(
        {
            "workflow": "persisted_registration_fanout",
            "optimized_seconds": optimized_seconds,
            "reference_seconds": reference_seconds,
            "speedup": reference_seconds / optimized_seconds,
            "optimized_tracemalloc_peak_bytes": optimized_peak,
            "reference_tracemalloc_peak_bytes": reference_peak,
            "retained_bytes": sum(plan.retained_bytes for plan in affine_plans),
            "max_abs_difference": _max_difference(optimized, reference),
        }
    )

    volumes = rng.standard_normal((2,) + volume_shape)
    matrix, offset = _rotation_3d(volume_shape, 6.0)
    affine = AffinePlan(
        volume_shape,
        matrix,
        offset,
        degree=3,
        mode="mirror",
    )
    differentials = DifferentialPlan(volume_shape, spacing=(0.8, 0.8, 1.5))
    warped_buffer = np.empty_like(volumes)
    feature_buffer = DifferentialResult(
        tuple(np.empty_like(volumes) for _ in range(3)),
        None,
        np.empty_like(volumes),
    )

    def buffered_volume_features():
        affine(volumes, spatial_axes=(1, 2, 3), out=warped_buffer)
        result = differentials(
            warped_buffer,
            gradient=True,
            hessian=False,
            laplacian=True,
            spatial_axes=(1, 2, 3),
            out=feature_buffer,
        )
        return result.gradient + (result.laplacian,)

    def scalar_volume_features():
        components = [np.empty_like(volumes) for _ in range(4)]
        for batch in range(volumes.shape[0]):
            warped = affine(volumes[batch])
            result = differentials(warped, gradient=True, hessian=False, laplacian=True)
            for index, component in enumerate(result.gradient + (result.laplacian,)):
                components[index][batch] = component
        return tuple(components)

    reference, reference_seconds, reference_peak = _measure(
        scalar_volume_features, args.repeats, args.warmups
    )
    optimized, optimized_seconds, optimized_peak = _measure(
        buffered_volume_features, args.repeats, args.warmups
    )
    rows.append(
        {
            "workflow": "buffered_volume_features",
            "optimized_seconds": optimized_seconds,
            "reference_seconds": reference_seconds,
            "speedup": reference_seconds / optimized_seconds,
            "optimized_tracemalloc_peak_bytes": optimized_peak,
            "reference_tracemalloc_peak_bytes": reference_peak,
            "retained_bytes": affine.retained_bytes + differentials.retained_bytes,
            "max_abs_difference": _max_difference(optimized, reference),
        }
    )

    for row in rows:
        print(
            f"{row['workflow']:34s} optimized={row['optimized_seconds']:.6f}s "
            f"reference={row['reference_seconds']:.6f}s "
            f"ratio={row['speedup']:.2f}x "
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
        "semantics": {
            "registration": "persisted coefficient fan-out across compatible geometries",
            "features": "3-D affine then gradient and Laplacian into caller buffers",
            "timing": "complete workflow; registration includes archive I/O and threads",
        },
        "results": rows,
        "summary": {
            "max_abs_difference": max(row["max_abs_difference"] for row in rows),
        },
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
