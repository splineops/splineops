"""Benchmark SplineOps rotation against an equivalent SciPy affine transform."""

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

import numpy as np
import scipy
from scipy.ndimage import affine_transform

from splineops import __version__
from splineops.affine import rotate


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    degree: int
    axis: tuple[float, float, float] | None = None


SMOKE_CASES = (
    Case("2d_linear", (96, 80), 1),
    Case("2d_cubic", (96, 80), 3),
    Case("3d_linear", (16, 20, 24), 1, (1.0, 2.0, 3.0)),
    Case("3d_cubic", (16, 20, 24), 3, (1.0, 2.0, 3.0)),
)

STANDARD_CASES = (
    Case("2d_linear", (512, 384), 1),
    Case("2d_cubic", (512, 384), 3),
    Case("3d_linear", (48, 56, 64), 1, (1.0, 2.0, 3.0)),
    Case("3d_cubic", (48, 56, 64), 3, (1.0, 2.0, 3.0)),
)


def _rotation_matrix(ndim: int, angle: float, axis):
    radians = np.radians(-angle)
    cosine = np.cos(radians)
    sine = np.sin(radians)
    if ndim == 2:
        return np.array([[cosine, -sine], [sine, cosine]])
    vector = np.asarray(axis, dtype=float)
    vector /= np.linalg.norm(vector)
    ux, uy, uz = vector
    factor = 1.0 - cosine
    return np.array(
        [
            [
                cosine + ux**2 * factor,
                ux * uy * factor - uz * sine,
                ux * uz * factor + uy * sine,
            ],
            [
                uy * ux * factor + uz * sine,
                cosine + uy**2 * factor,
                uy * uz * factor - ux * sine,
            ],
            [
                uz * ux * factor - uy * sine,
                uz * uy * factor + ux * sine,
                cosine + uz**2 * factor,
            ],
        ]
    )


def _measure(call, repeats: int, warmups: int):
    for _ in range(warmups):
        call()
    timings = []
    peak = 0
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        output = call()
        timings.append(time.perf_counter() - started)
        _, current_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, current_peak)
    return output, float(np.median(timings)), peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--angle", type=float, default=23.0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("repeats must be positive and warmups non-negative")

    cases = SMOKE_CASES if args.profile == "smoke" else STANDARD_CASES
    rng = np.random.default_rng(args.seed)
    rows = []
    for case in cases:
        data = rng.standard_normal(case.shape)
        center = np.array([(length - 1) / 2 for length in case.shape])
        matrix = _rotation_matrix(data.ndim, args.angle, case.axis)
        offset = center - matrix @ center
        spline_output, spline_seconds, spline_peak = _measure(
            lambda: rotate(
                data,
                args.angle,
                axis=case.axis,
                degree=case.degree,
                mode="mirror",
            ),
            args.repeats,
            args.warmups,
        )
        scipy_output, scipy_seconds, scipy_peak = _measure(
            lambda: affine_transform(
                data,
                matrix,
                offset=offset,
                output_shape=case.shape,
                order=case.degree,
                mode="mirror",
                prefilter=True,
            ),
            args.repeats,
            args.warmups,
        )
        row = {
            "case": case.name,
            "shape": "x".join(str(value) for value in case.shape),
            "degree": case.degree,
            "splineops_median_seconds": spline_seconds,
            "scipy_median_seconds": scipy_seconds,
            "scipy_speedup_over_splineops": spline_seconds / scipy_seconds,
            "splineops_tracemalloc_peak_bytes": spline_peak,
            "scipy_tracemalloc_peak_bytes": scipy_peak,
            "max_abs_difference": float(np.max(np.abs(spline_output - scipy_output))),
        }
        rows.append(row)
        print(
            f"{case.name:12s} splineops={spline_seconds:8.4f}s "
            f"scipy={scipy_seconds:8.4f}s "
            f"scipy-speedup={row['scipy_speedup_over_splineops']:6.2f}x "
            f"max-error={row['max_abs_difference']:.3e}"
        )

    payload = {
        "schema_version": 1,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "splineops": __version__,
        },
        "semantics": {
            "coordinates": "pull-back rotation about array center",
            "boundary": "whole-sample mirror",
            "prefilter": True,
            "memory_metric": "tracemalloc peak; native allocations may be omitted",
        },
        "profile": args.profile,
        "angle_degrees": args.angle,
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
