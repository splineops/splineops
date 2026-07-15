#!/usr/bin/env python3
"""
Same-build A/B benchmark for native resize feature flags.

This script toggles one environment variable between two values while reusing
the fixed cases from benchmark_resize_native.py. It is intended for native
kernel/routing decisions where rebuild-to-rebuild timing noise would hide the
real effect.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_resize_native as native  # noqa: E402


@dataclass
class ABResult:
    case: str
    shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    zoom: tuple[float, ...]
    method: str
    dtype: str
    threads: str
    flag: str
    a_value: str
    b_value: str
    a_best_ms: float
    b_best_ms: float
    a_median_ms: float
    b_median_ms: float
    a_mean_ms: float
    b_mean_ms: float
    best_speedup: float
    median_speedup: float
    mean_speedup: float
    repeats: int
    warmups: int
    max_abs_diff: float
    mean_abs_diff: float
    passed_check: bool


def set_flag(name: str, value: str) -> None:
    if value == "<unset>":
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


def parse_case_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


def run_variant(
    case: native.BenchCase,
    *,
    flag: str,
    value: str,
    threads: str,
    repeats: int,
    warmups: int,
) -> tuple[list[float], np.ndarray]:
    set_flag(flag, value)
    return native.time_case(
        case,
        backend="native",
        threads=threads,
        repeats=repeats,
        warmups=warmups,
    )


def compare_outputs(
    a: np.ndarray, b: np.ndarray, atol: float, rtol: float
) -> tuple[bool, float, float]:
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    return bool(np.allclose(a, b, atol=atol, rtol=rtol)), max_abs, mean_abs


def print_result(result: ABResult) -> None:
    status = "ok" if result.passed_check else "FAIL"
    print(
        f"{result.case:30s} "
        f"{result.dtype:7s} "
        f"thr={result.threads:>7s} "
        f"{result.flag}={result.a_value}->{result.b_value} "
        f"median={result.a_median_ms:8.3f}->{result.b_median_ms:8.3f} ms "
        f"speedup={result.median_speedup:5.2f}x "
        f"{status} max={result.max_abs_diff:.2e}"
    )


def write_csv(path: Path, results: list[ABResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    fieldnames = list(rows[0].keys()) if rows else list(ABResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def summarize(results: list[ABResult]) -> None:
    if not results:
        return
    speedups = [r.median_speedup for r in results]
    print()
    print(
        "summary "
        f"cases={len(results)} "
        f"median_speedup={statistics.median(speedups):.3f}x "
        f"mean_speedup={statistics.fmean(speedups):.3f}x "
        f"wins={sum(v > 1.03 for v in speedups)} "
        f"losses={sum(v < 0.97 for v in speedups)}"
    )
    for threads in sorted({r.threads for r in results}):
        group = [r.median_speedup for r in results if r.threads == threads]
        print(
            f"threads={threads:>7s} "
            f"median_speedup={statistics.median(group):.3f}x "
            f"mean_speedup={statistics.fmean(group):.3f}x "
            f"wins={sum(v > 1.03 for v in group)} "
            f"losses={sum(v < 0.97 for v in group)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["smoke", "standard", "full"], default="standard"
    )
    parser.add_argument("--flag", required=True, help="Environment variable to toggle.")
    parser.add_argument(
        "--a", default="0", help="Baseline value. Use <unset> to unset the variable."
    )
    parser.add_argument(
        "--b", default="1", help="Candidate value. Use <unset> to unset the variable."
    )
    parser.add_argument(
        "--threads", type=native.parse_threads, default=native.parse_threads("default")
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--cases", help="Comma-separated case names to include.")
    parser.add_argument("--atol", type=float, default=5e-6)
    parser.add_argument("--rtol", type=float, default=5e-6)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    if args.repeats <= 0:
        print("--repeats must be positive", file=sys.stderr)
        return 2
    if args.warmups < 0:
        print("--warmups must be non-negative", file=sys.stderr)
        return 2

    case_filter = parse_case_filter(args.cases)
    cases = native.cases_for_profile(args.profile)
    if case_filter is not None:
        cases = [case for case in cases if case.name in case_filter]
        missing = sorted(case_filter - {case.name for case in cases})
        if missing:
            print(f"unknown case(s): {', '.join(missing)}", file=sys.stderr)
            return 2

    original_flag = os.environ.get(args.flag)
    results: list[ABResult] = []

    print("splineops resize native A/B benchmark")
    print(
        f"profile={args.profile} flag={args.flag} "
        f"a={args.a} b={args.b} repeats={args.repeats} warmups={args.warmups}"
    )
    print(f"threads={','.join(args.threads)}")
    print(f"python={platform.python_version()} numpy={np.__version__}")
    print(f"platform={platform.platform()}")
    print()

    try:
        for case in cases:
            for threads in args.threads:
                a_samples, a_out = run_variant(
                    case,
                    flag=args.flag,
                    value=args.a,
                    threads=threads,
                    repeats=args.repeats,
                    warmups=args.warmups,
                )
                b_samples, b_out = run_variant(
                    case,
                    flag=args.flag,
                    value=args.b,
                    threads=threads,
                    repeats=args.repeats,
                    warmups=args.warmups,
                )

                passed, max_abs, mean_abs = compare_outputs(
                    a_out, b_out, args.atol, args.rtol
                )
                result = ABResult(
                    case=case.name,
                    shape=case.shape,
                    output_shape=tuple(int(v) for v in b_out.shape),
                    zoom=case.zoom,
                    method=case.method,
                    dtype=case.dtype,
                    threads=threads,
                    flag=args.flag,
                    a_value=args.a,
                    b_value=args.b,
                    a_best_ms=min(a_samples) * 1000.0,
                    b_best_ms=min(b_samples) * 1000.0,
                    a_median_ms=statistics.median(a_samples) * 1000.0,
                    b_median_ms=statistics.median(b_samples) * 1000.0,
                    a_mean_ms=statistics.fmean(a_samples) * 1000.0,
                    b_mean_ms=statistics.fmean(b_samples) * 1000.0,
                    best_speedup=min(a_samples) / min(b_samples),
                    median_speedup=statistics.median(a_samples)
                    / statistics.median(b_samples),
                    mean_speedup=statistics.fmean(a_samples)
                    / statistics.fmean(b_samples),
                    repeats=args.repeats,
                    warmups=args.warmups,
                    max_abs_diff=max_abs,
                    mean_abs_diff=mean_abs,
                    passed_check=passed,
                )
                results.append(result)
                print_result(result)
    finally:
        if original_flag is None:
            os.environ.pop(args.flag, None)
        else:
            os.environ[args.flag] = original_flag

    summarize(results)

    payload = {
        "metadata": {
            "profile": args.profile,
            "flag": args.flag,
            "a": args.a,
            "b": args.b,
            "threads": args.threads,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "atol": args.atol,
            "rtol": args.rtol,
            "native_knobs": native.native_knobs(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "results": [asdict(r) for r in results],
    }
    if args.output_json is not None:
        write_json(args.output_json, payload)
        print(f"\nwrote JSON: {args.output_json}")
    if args.output_csv is not None:
        write_csv(args.output_csv, results)
        print(f"wrote CSV: {args.output_csv}")

    failed = [r for r in results if not r.passed_check]
    if failed:
        print(f"\n{len(failed)} A/B output check(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
