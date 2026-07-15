"""Profile complete affine execution as construction, prefilter, and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from splineops import __version__
from splineops.affine import AffinePlan, affine_transform


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    degree: int


SMOKE_CASES = (Case("2d_cubic", (64, 80), 3), Case("3d_cubic", (12, 14, 16), 3))
STANDARD_CASES = (
    Case("2d_linear", (384, 512), 1),
    Case("2d_cubic", (384, 512), 3),
    Case("3d_linear", (40, 48, 56), 1),
    Case("3d_cubic", (40, 48, 56), 3),
)


def _measure(call, repeats, warmups):
    for _ in range(warmups):
        call()
    samples = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = call()
        samples.append(time.perf_counter() - started)
    return result, float(np.median(samples))


def _measure_plan_phases(plan, data, repeats, warmups):
    for _ in range(warmups):
        plan(data)
    original_prefilter = plan._prefilter_canonical
    original_evaluation = plan._apply_canonical_coefficients
    prefilter_samples = []
    evaluation_samples = []
    total_samples = []
    result = None
    try:
        for _ in range(repeats):
            phase_times = {}

            def timed_prefilter(values):
                started = time.perf_counter()
                output = original_prefilter(values)
                phase_times["prefilter"] = time.perf_counter() - started
                return output

            def timed_evaluation(values):
                started = time.perf_counter()
                output = original_evaluation(values)
                phase_times["evaluation"] = time.perf_counter() - started
                return output

            plan._prefilter_canonical = timed_prefilter
            plan._apply_canonical_coefficients = timed_evaluation
            started = time.perf_counter()
            result = plan(data)
            total_samples.append(time.perf_counter() - started)
            prefilter_samples.append(phase_times["prefilter"])
            evaluation_samples.append(phase_times["evaluation"])
    finally:
        plan.__dict__.pop("_prefilter_canonical", None)
        plan.__dict__.pop("_apply_canonical_coefficients", None)
    return (
        result,
        float(np.median(prefilter_samples)),
        float(np.median(evaluation_samples)),
        float(np.median(total_samples)),
    )


def _geometry(shape):
    radians = np.radians(-9.0)
    if len(shape) == 2:
        matrix = np.array(
            [
                [np.cos(radians), -np.sin(radians)],
                [np.sin(radians), np.cos(radians)],
            ]
        )
    else:
        matrix = np.array(
            [
                [np.cos(radians), -np.sin(radians), 0.0],
                [np.sin(radians), np.cos(radians), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
    center = (np.asarray(shape) - 1.0) / 2.0
    return matrix, center - matrix @ center


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="smoke")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        parser.error("repeats must be positive and warmups non-negative")

    rng = np.random.default_rng(args.seed)
    cases = SMOKE_CASES if args.profile == "smoke" else STANDARD_CASES
    rows = []
    for case in cases:
        data = rng.standard_normal(case.shape)
        matrix, offset = _geometry(case.shape)

        def construct():
            return AffinePlan(
                case.shape,
                matrix,
                offset,
                degree=case.degree,
                mode="mirror",
            )

        plan, construction_seconds = _measure(construct, args.repeats, args.warmups)
        (
            end_to_end,
            prefilter_seconds,
            evaluation_seconds,
            end_to_end_seconds,
        ) = _measure_plan_phases(
            plan,
            data,
            args.repeats,
            args.warmups,
        )
        streaming, streaming_seconds = _measure(
            lambda: affine_transform(
                data,
                matrix,
                offset,
                degree=case.degree,
                mode="mirror",
            ),
            args.repeats,
            args.warmups,
        )
        np.testing.assert_equal(streaming, end_to_end)
        measured_phases = prefilter_seconds + evaluation_seconds
        phase_values = {
            "prefilter": prefilter_seconds,
            "evaluation": evaluation_seconds,
        }
        dominant_phase = max(phase_values, key=phase_values.get)
        rows.append(
            {
                "case": case.name,
                "shape": "x".join(str(value) for value in case.shape),
                "degree": case.degree,
                "construction_seconds": construction_seconds,
                "prefilter_seconds": prefilter_seconds,
                "evaluation_seconds": evaluation_seconds,
                "end_to_end_seconds": end_to_end_seconds,
                "streaming_one_shot_seconds": streaming_seconds,
                "prefilter_fraction": prefilter_seconds / measured_phases,
                "evaluation_fraction": evaluation_seconds / measured_phases,
                "dominant_phase": dominant_phase,
                "phase_sum_over_end_to_end": measured_phases / end_to_end_seconds,
                "retained_bytes": plan.retained_bytes,
                "max_abs_difference": float(np.max(np.abs(end_to_end - streaming))),
            }
        )

    for row in rows:
        print(
            f"{row['case']:12s} construct={row['construction_seconds']:.6f}s "
            f"prefilter={row['prefilter_seconds']:.6f}s "
            f"evaluate={row['evaluation_seconds']:.6f}s "
            f"end-to-end={row['end_to_end_seconds']:.6f}s "
            f"dominant={row['dominant_phase']}"
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
