#!/usr/bin/env python3
"""Quality sweep for native resize precision modes.

This script compares the default 64-bit internal native path against the
opt-in float32 internal path on synthetic signals that are useful for spotting
precision and high-frequency artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class QualityCase:
    name: str
    shape: tuple[int, ...]
    zoom: tuple[float, ...]
    method: str
    pattern: str


@dataclass
class QualityResult:
    case: str
    pattern: str
    shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    zoom: tuple[float, ...]
    method: str
    dtype: str
    max_abs_diff: float
    mean_abs_diff: float
    p99_abs_diff: float
    rel_l2_diff: float
    default_ms: float
    float32_ms: float


def quality_cases(profile: str) -> list[QualityCase]:
    base = [
        QualityCase(
            "constant_cubic_aa_down",
            (257, 251),
            (0.37, 0.53),
            "cubic-antialiasing",
            "constant",
        ),
        QualityCase("ramp_cubic_down", (256, 256), (0.37, 0.37), "cubic", "ramp"),
        QualityCase(
            "ramp_cubic_aa_aniso",
            (256, 256),
            (0.37, 0.91),
            "cubic-antialiasing",
            "ramp",
        ),
        QualityCase(
            "impulse_cubic_aa_down",
            (256, 256),
            (0.37, 0.37),
            "cubic-antialiasing",
            "impulse",
        ),
        QualityCase(
            "checker_linear_aa_down",
            (256, 256),
            (0.37, 0.37),
            "linear-antialiasing",
            "checkerboard",
        ),
        QualityCase(
            "checker_cubic_aa_down",
            (256, 256),
            (0.37, 0.37),
            "cubic-antialiasing",
            "checkerboard",
        ),
        QualityCase(
            "sinusoid_cubic_aa_down",
            (256, 256),
            (0.37, 0.53),
            "cubic-antialiasing",
            "sinusoid",
        ),
        QualityCase(
            "random_cubic_aa_down",
            (256, 256),
            (0.37, 0.37),
            "cubic-antialiasing",
            "random",
        ),
    ]
    if profile == "quick":
        return base[:4]
    if profile != "standard":
        raise ValueError(f"unknown profile {profile!r}")
    return base


def make_pattern(case: QualityCase, rng: np.random.Generator) -> np.ndarray:
    shape = case.shape
    if case.pattern == "constant":
        return np.full(shape, 3.25, dtype=np.float32)
    if case.pattern == "random":
        return rng.random(shape, dtype=np.float32)
    if case.pattern == "impulse":
        x = np.zeros(shape, dtype=np.float32)
        center = tuple(n // 2 for n in shape)
        x[center] = 1.0
        return x
    if case.pattern == "ramp":
        grids = np.meshgrid(
            *[np.linspace(0.0, 1.0, n, dtype=np.float32) for n in shape],
            indexing="ij",
        )
        return (sum(grids) / float(len(grids))).astype(np.float32, copy=False)
    if case.pattern == "checkerboard":
        grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        block = 4
        board = sum((g // block) for g in grids) % 2
        return board.astype(np.float32)
    if case.pattern == "sinusoid":
        grids = np.meshgrid(
            *[np.arange(n, dtype=np.float32) for n in shape],
            indexing="ij",
        )
        out = np.zeros(shape, dtype=np.float32)
        for ax, g in enumerate(grids):
            freq = 0.31 + 0.13 * ax
            out += np.sin(2.0 * np.pi * freq * g).astype(np.float32)
        out /= float(len(grids))
        return out
    raise ValueError(f"unknown pattern {case.pattern!r}")


def timed_resize(arr: np.ndarray, case: QualityCase, precision: str | None) -> tuple[np.ndarray, float]:
    if precision is None:
        os.environ.pop("LSRESIZE_PRECISION", None)
    else:
        os.environ["LSRESIZE_PRECISION"] = precision

    from splineops.resize import resize

    t0 = perf_counter()
    out = resize(arr, zoom_factors=case.zoom, method=case.method)
    dt_ms = (perf_counter() - t0) * 1000.0
    return out, dt_ms


def run_case(case: QualityCase, rng: np.random.Generator) -> QualityResult:
    arr = make_pattern(case, rng)
    y_default, default_ms = timed_resize(arr, case, None)
    y_float32, float32_ms = timed_resize(arr, case, "float32")

    diff = np.abs(y_default.astype(np.float64) - y_float32.astype(np.float64))
    denom = float(np.linalg.norm(y_default.astype(np.float64).ravel()))
    rel_l2 = float(np.linalg.norm(diff.ravel()) / denom) if denom > 0.0 else 0.0
    return QualityResult(
        case=case.name,
        pattern=case.pattern,
        shape=case.shape,
        output_shape=tuple(int(n) for n in y_default.shape),
        zoom=case.zoom,
        method=case.method,
        dtype=str(arr.dtype),
        max_abs_diff=float(np.max(diff)) if diff.size else 0.0,
        mean_abs_diff=float(np.mean(diff)) if diff.size else 0.0,
        p99_abs_diff=float(np.quantile(diff, 0.99)) if diff.size else 0.0,
        rel_l2_diff=rel_l2,
        default_ms=default_ms,
        float32_ms=float32_ms,
    )


def write_json(path: Path, results: Iterable[QualityResult]) -> None:
    rows = [asdict(r) for r in results]
    payload = {
        "metadata": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "batched_axis_env": os.environ.get("LSRESIZE_BATCHED_AXIS"),
        },
        "results": rows,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, results: Iterable[QualityResult]) -> None:
    rows = [asdict(r) for r in results]
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["quick", "standard"], default="standard")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--batched-axis",
        choices=["unset", "auto", "1", "off"],
        default="auto",
        help="Set LSRESIZE_BATCHED_AXIS before running the sweep.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["SPLINEOPS_ACCEL"] = "always"
    if args.batched_axis == "unset":
        os.environ.pop("LSRESIZE_BATCHED_AXIS", None)
    else:
        os.environ["LSRESIZE_BATCHED_AXIS"] = args.batched_axis

    rng = np.random.default_rng(args.seed)
    results = [run_case(case, rng) for case in quality_cases(args.profile)]

    for result in results:
        speedup = result.default_ms / result.float32_ms if result.float32_ms > 0.0 else float("inf")
        print(
            f"{result.case:28s} "
            f"max={result.max_abs_diff:.3e} "
            f"p99={result.p99_abs_diff:.3e} "
            f"rel_l2={result.rel_l2_diff:.3e} "
            f"speedup={speedup:.2f}x"
        )

    if args.output_json is not None:
        write_json(args.output_json, results)
        print(f"wrote JSON: {args.output_json}")
    if args.output_csv is not None:
        write_csv(args.output_csv, results)
        print(f"wrote CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
