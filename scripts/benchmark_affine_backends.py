"""Benchmark affine backends with matched coefficients and geometry.

Separates plan construction, one-shot calls, repeated calls, and precomputed
coefficient evaluation. Timings are uninstrumented and method order is
randomized. Differences use the independent NumPy spline evaluator as the
reference; they measure numerical parity, not application quality.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import scipy

from splineops import __version__
from splineops.affine import AffinePlan, affine_transform


def measure(calls, repeats, rng):
    outputs = {}
    samples = {name: [] for name in calls}
    for name, call in calls.items():
        for _ in range(2):
            outputs[name] = call()
    for _ in range(repeats):
        for name in rng.permutation(list(calls)):
            started = time.perf_counter()
            value = calls[name]()
            samples[name].append((time.perf_counter() - started) * 1000)
            # Destruction of a previous plan is outside construction timing.
            outputs[name] = value
    return (
        outputs,
        {name: float(np.median(values)) for name, values in samples.items()},
        samples,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard"], default="standard")
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--degrees", default="3", help="Comma-separated degrees 1--5.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats <= 0 or args.batch_size <= 0:
        parser.error("--repeats and --batch-size must be positive")
    try:
        degrees = [int(value) for value in args.degrees.split(",")]
    except ValueError:
        parser.error("--degrees must contain comma-separated integers")
    if not degrees or any(degree < 1 or degree > 5 for degree in degrees):
        parser.error("--degrees must be in 1--5")
    rng = np.random.default_rng(20261028)
    shapes = (
        [(32, 40), (8, 10, 12)]
        if args.profile == "smoke"
        else [(512, 384), (48, 56, 64)]
    )
    rows = []
    for shape in shapes:
        ndim = len(shape)
        angle = np.deg2rad(23)
        matrix = np.eye(ndim)
        matrix[:2, :2] = [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
        center = (np.asarray(shape) - 1) / 2
        offset = center - matrix @ center
        for dtype in (np.float32, np.float64):
            sample_shape = shape if args.batch_size == 1 else shape + (args.batch_size,)
            data = rng.standard_normal(sample_shape).astype(dtype)
            spatial_kw = (
                {} if args.batch_size == 1 else {"spatial_axes": tuple(range(ndim))}
            )
            for mode, degree in itertools.product(
                ("mirror", "zero", "periodic"), degrees
            ):
                params = dict(degree=degree, mode=mode)
                factories = {
                    backend: lambda backend=backend: AffinePlan(
                        shape, matrix, offset, dtype=dtype, backend=backend, **params
                    )
                    for backend in ("numpy", "scipy")
                }
                plans, build_ms, build_samples = measure(factories, args.repeats, rng)
                # Reuse exactly one coefficient field for both evaluators.
                coefficients = plans["numpy"].prepare_coefficients(data, **spatial_kw)
                calls = {}
                for backend, plan in plans.items():
                    calls[f"{backend}_one_shot"] = (
                        lambda backend=backend: affine_transform(
                            data,
                            matrix,
                            offset,
                            backend=backend,
                            **spatial_kw,
                            **params,
                        )
                    )
                    calls[f"{backend}_planned"] = lambda plan=plan: plan(
                        data, **spatial_kw
                    )
                    calls[f"{backend}_coefficients"] = (
                        lambda plan=plan: plan.apply_coefficients(coefficients)
                    )
                outputs, medians, samples = measure(calls, args.repeats, rng)
                reference = outputs["numpy_one_shot"]
                errors = {}
                for name, values in outputs.items():
                    tolerance = (
                        (3e-5 if degree >= 4 else 5e-6)
                        if dtype == np.float32
                        else 3e-11
                    )
                    np.testing.assert_allclose(
                        values, reference, rtol=tolerance, atol=tolerance
                    )
                    errors[name] = {
                        "max_abs": float(np.max(np.abs(values - reference))),
                        "relative_l2": float(
                            np.linalg.norm(
                                (values - reference).astype(np.float64).ravel()
                            )
                            / np.linalg.norm(reference.astype(np.float64).ravel())
                        ),
                    }
                speedups = {
                    stage: medians[f"numpy_{stage}"] / medians[f"scipy_{stage}"]
                    for stage in ("one_shot", "planned", "coefficients")
                }
                row = dict(
                    shape=shape,
                    dtype=np.dtype(dtype).name,
                    degree=degree,
                    mode=mode,
                    batch_size=args.batch_size,
                    construction_ms=build_ms,
                    construction_samples_ms=build_samples,
                    median_ms=medians,
                    samples_ms=samples,
                    speedups=speedups,
                    error_against_numpy=errors,
                    retained_bytes={
                        name: plan.retained_bytes for name, plan in plans.items()
                    },
                )
                rows.append(row)
                print(
                    f"{shape} {np.dtype(dtype).name} {mode} degree={degree} batch={args.batch_size}: one-shot {speedups['one_shot']:.2f}x, "
                    f"planned {speedups['planned']:.2f}x, coefficients {speedups['coefficients']:.2f}x; "
                    f"max difference {errors['scipy_one_shot']['max_abs']:.3g}",
                    flush=True,
                )
    payload = {
        "schema_version": 1,
        "profile": args.profile,
        "repeats": args.repeats,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "splineops": __version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "semantics": {
            "reference": "NumPy TensorSpline evaluator",
            "coefficients": "identical SplineOps prefilter for both backends",
            "geometry": "23-degree pull-back rotation about array center",
            "float32": "same rounded coordinates; double accumulation in SciPy",
            "timing": "two warmups; randomized method order; no tracemalloc",
        },
        "results": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
