"""Benchmark vectorized spline differentials against the scalar reference path."""

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
from splineops.differentials import DifferentialPlan, Differentials


def _measure(call, repeats: int, warmups: int):
    for _ in range(warmups):
        call()
    timings = []
    peak = 0
    output = None
    for _ in range(repeats):
        tracemalloc.start()
        started = time.perf_counter()
        output = call()
        timings.append(time.perf_counter() - started)
        _, current_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak = max(peak, current_peak)
    return output, float(np.median(timings)), peak


def _scalar_gradient_magnitude(operator: Differentials) -> np.ndarray:
    """Historical row/column loop retained as an independent benchmark oracle."""
    image = operator.image
    vertical = np.zeros_like(image)
    horizontal = np.zeros_like(image)
    for row in range(operator.height):
        coefficients = image[row, :].copy()
        operator.get_spline_interpolation_coefficients(
            coefficients, operator.FLT_EPSILON
        )
        horizontal[row, :] = operator.get_gradient(coefficients) / operator.spacing[1]
    for column in range(operator.width):
        coefficients = image[:, column].copy()
        operator.get_spline_interpolation_coefficients(
            coefficients, operator.FLT_EPSILON
        )
        vertical[:, column] = operator.get_gradient(coefficients) / operator.spacing[0]
    return np.sqrt(horizontal**2 + vertical**2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("repeats must be positive and warmups non-negative")

    shape = (64, 80) if args.profile == "smoke" else (512, 640)
    spacing = (0.7, 1.3)
    image = np.random.default_rng(args.seed).standard_normal(shape)
    vectorized, vectorized_seconds, vectorized_peak = _measure(
        lambda: Differentials(image, spacing=spacing).gradient_magnitude(),
        args.repeats,
        args.warmups,
    )
    operator = Differentials(image, spacing=spacing)
    cached, cached_seconds, cached_peak = _measure(
        operator.gradient_magnitude, args.repeats, args.warmups
    )
    plan = DifferentialPlan(shape, spacing=spacing)
    multi_output, multi_output_seconds, multi_output_peak = _measure(
        lambda: plan(image), args.repeats, args.warmups
    )
    gradient_only, gradient_only_seconds, gradient_only_peak = _measure(
        lambda: plan(image, hessian=False, laplacian=False),
        args.repeats,
        args.warmups,
    )
    laplacian_only, laplacian_only_seconds, laplacian_only_peak = _measure(
        lambda: plan(image, gradient=False, hessian=False, laplacian=True),
        args.repeats,
        args.warmups,
    )
    scalar, scalar_seconds, scalar_peak = _measure(
        lambda: _scalar_gradient_magnitude(operator), args.repeats, args.warmups
    )
    # The historical scalar recursion truncates its causal initialization at
    # ``FLT_EPSILON``.  It is therefore a close numerical oracle, not a
    # bit-identical implementation of the batched coefficient helper.
    np.testing.assert_allclose(vectorized, scalar, rtol=1e-5, atol=2e-6)
    np.testing.assert_equal(cached, vectorized)
    assert multi_output.gradient is not None
    assert multi_output.hessian is not None
    assert gradient_only.gradient is not None
    assert gradient_only.hessian is None
    assert gradient_only.laplacian is None
    assert laplacian_only.gradient is None
    assert laplacian_only.hessian is None
    np.testing.assert_equal(laplacian_only.laplacian, multi_output.laplacian)

    rows = [
        {
            "path": "vectorized_cold",
            "median_seconds": vectorized_seconds,
            "tracemalloc_peak_bytes": vectorized_peak,
        },
        {
            "path": "cached_instance",
            "median_seconds": cached_seconds,
            "tracemalloc_peak_bytes": cached_peak,
        },
        {
            "path": "multi_output_plan",
            "median_seconds": multi_output_seconds,
            "tracemalloc_peak_bytes": multi_output_peak,
        },
        {
            "path": "gradient_only_plan",
            "median_seconds": gradient_only_seconds,
            "tracemalloc_peak_bytes": gradient_only_peak,
        },
        {
            "path": "laplacian_only_plan",
            "median_seconds": laplacian_only_seconds,
            "tracemalloc_peak_bytes": laplacian_only_peak,
        },
        {
            "path": "scalar_reference",
            "median_seconds": scalar_seconds,
            "tracemalloc_peak_bytes": scalar_peak,
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
            "profile": args.profile,
            "shape": list(shape),
            "dtype": str(image.dtype),
            "spacing": list(spacing),
            "operation": "gradient_magnitude",
            "boundary": "whole-sample mirror",
        },
        "vectorized_speedup": scalar_seconds / vectorized_seconds,
        "cached_speedup": vectorized_seconds / cached_seconds,
        "multi_output_seconds": multi_output_seconds,
        "gradient_only_seconds": gradient_only_seconds,
        "laplacian_only_seconds": laplacian_only_seconds,
        "laplacian_peak_fraction_of_full": (laplacian_only_peak / multi_output_peak),
        "max_abs_difference": float(np.max(np.abs(vectorized - scalar))),
        "results": rows,
    }
    print(
        f"shape={shape} vectorized={vectorized_seconds:.6f}s "
        f"scalar={scalar_seconds:.6f}s "
        f"speedup={payload['vectorized_speedup']:.2f}x "
        f"cached={cached_seconds:.6f}s "
        f"laplacian-only={laplacian_only_seconds:.6f}s "
        f"max-error={payload['max_abs_difference']:.3e}"
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
