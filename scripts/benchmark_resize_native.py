# splineops/scripts/benchmark_resize_native.py
"""
Benchmark the native spline resize backend.

This script is intended as the scoreboard before deeper native-kernel rewrites.
It times the C++ backend across fixed resize cases and, for selected workloads,
checks native output against the pure-Python fallback.

Examples
--------
Quick smoke run:

    python scripts/benchmark_resize_native.py --profile smoke

Standard run with saved artifacts:

    python scripts/benchmark_resize_native.py \\
        --profile standard \\
        --output-json /tmp/splineops_resize_bench.json \\
        --output-csv /tmp/splineops_resize_bench.csv

Full thread sweep:

    python scripts/benchmark_resize_native.py \\
        --profile full \\
        --threads 1,2,4,8,16,default
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Keep BLAS stacks from interfering with the native resize thread sweep.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


@dataclass(frozen=True)
class BenchCase:
    name: str
    shape: tuple[int, ...]
    zoom: tuple[float, ...]
    method: str
    dtype: str


@dataclass
class BenchResult:
    case: str
    shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    zoom: tuple[float, ...]
    method: str
    dtype: str
    threads: str
    best_ms: float
    median_ms: float
    mean_ms: float
    repeats: int
    warmups: int
    checked: bool
    max_abs_diff: float | None
    mean_abs_diff: float | None
    passed_check: bool | None


def has_cpp() -> bool:
    return importlib.util.find_spec("splineops._lsresize") is not None


def load_resize_module(mode: str):
    os.environ["SPLINEOPS_ACCEL"] = mode
    name = "splineops.resize.resize"
    if name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def parse_threads(value: str) -> list[str]:
    threads: list[str] = []
    for raw in value.split(","):
        item = raw.strip().lower()
        if not item:
            continue
        if item == "default":
            threads.append(item)
            continue
        parsed = int(item)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("thread counts must be positive")
        threads.append(str(parsed))
    if not threads:
        raise argparse.ArgumentTypeError("at least one thread setting is required")
    return threads


def set_threads(value: str) -> None:
    if value == "default":
        os.environ.pop("LSRESIZE_NUM_THREADS", None)
    else:
        os.environ["LSRESIZE_NUM_THREADS"] = value


def dtype_from_name(name: str) -> np.dtype:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"Unsupported dtype: {name}")


def stable_seed(case: BenchCase) -> int:
    payload = f"{case.name}|{case.shape}|{case.zoom}|{case.method}|{case.dtype}"
    return zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF


def make_input(case: BenchCase) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(case))
    return rng.random(case.shape, dtype=dtype_from_name(case.dtype))


def run_once(module: Any, x: np.ndarray, case: BenchCase) -> np.ndarray:
    return module.resize(x, zoom_factors=case.zoom, method=case.method)


def time_native_case(
    case: BenchCase,
    *,
    threads: str,
    warmups: int,
    repeats: int,
) -> tuple[list[float], np.ndarray]:
    set_threads(threads)
    module = load_resize_module("always")
    x = make_input(case)

    out = run_once(module, x, case)
    for _ in range(max(0, warmups - 1)):
        out = run_once(module, x, case)

    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        out = run_once(module, x, case)
        samples.append(time.perf_counter() - t0)

    return samples, out


def check_against_python(
    case: BenchCase,
    native_out: np.ndarray,
    *,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float]:
    x = make_input(case)
    module = load_resize_module("never")
    py_out = run_once(module, x, case)

    diff = np.abs(native_out.astype(np.float64) - py_out.astype(np.float64))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    passed = bool(np.allclose(native_out, py_out, atol=atol, rtol=rtol))
    return passed, max_abs, mean_abs


def smoke_cases() -> list[BenchCase]:
    return [
        BenchCase("2d_cubic_down_f32", (256, 256), (0.37, 0.37), "cubic", "float32"),
        BenchCase(
            "2d_cubic_aa_down_f32",
            (256, 256),
            (0.37, 0.37),
            "cubic-antialiasing",
            "float32",
        ),
        BenchCase(
            "2d_cubic_aniso_f64",
            (256, 384),
            (1.0, 0.5),
            "cubic",
            "float64",
        ),
    ]


def standard_cases() -> list[BenchCase]:
    cases: list[BenchCase] = []
    for dtype in ("float32", "float64"):
        for shape in ((512, 512), (1024, 1024)):
            cases.extend(
                [
                    BenchCase(
                        f"2d_cubic_down_{shape[0]}_{dtype}",
                        shape,
                        (0.37, 0.37),
                        "cubic",
                        dtype,
                    ),
                    BenchCase(
                        f"2d_cubic_aa_down_{shape[0]}_{dtype}",
                        shape,
                        (0.37, 0.37),
                        "cubic-antialiasing",
                        dtype,
                    ),
                    BenchCase(
                        f"2d_linear_aa_down_{shape[0]}_{dtype}",
                        shape,
                        (0.37, 0.37),
                        "linear-antialiasing",
                        dtype,
                    ),
                    BenchCase(
                        f"2d_cubic_aniso_{shape[0]}_{dtype}",
                        shape,
                        (1.0, 0.37),
                        "cubic",
                        dtype,
                    ),
                ]
            )
    cases.extend(
        [
            BenchCase("3d_cubic_down_f32", (128, 128, 32), (0.5, 0.5, 0.5), "cubic", "float32"),
            BenchCase("3d_cubic_aniso_f32", (128, 128, 32), (1.0, 0.5, 1.0), "cubic", "float32"),
        ]
    )
    return cases


def full_cases() -> list[BenchCase]:
    cases = standard_cases()
    for dtype in ("float32", "float64"):
        cases.extend(
            [
                BenchCase(
                    f"2d_cubic_down_2048_{dtype}",
                    (2048, 2048),
                    (0.37, 0.37),
                    "cubic",
                    dtype,
                ),
                BenchCase(
                    f"2d_cubic_aa_down_2048_{dtype}",
                    (2048, 2048),
                    (0.37, 0.37),
                    "cubic-antialiasing",
                    dtype,
                ),
                BenchCase(
                    f"2d_cubic_up_512_{dtype}",
                    (512, 512),
                    (1.7, 1.7),
                    "cubic",
                    dtype,
                ),
            ]
        )
    cases.extend(
        [
            BenchCase("3d_cubic_down_large_f32", (256, 256, 64), (0.37, 0.37, 0.37), "cubic", "float32"),
            BenchCase("3d_cubic_aniso_large_f32", (256, 256, 64), (1.0, 0.5, 1.0), "cubic", "float32"),
        ]
    )
    return cases


def cases_for_profile(profile: str) -> list[BenchCase]:
    if profile == "smoke":
        return smoke_cases()
    if profile == "standard":
        return standard_cases()
    if profile == "full":
        return full_cases()
    raise ValueError(f"Unknown profile: {profile}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_csv(path: Path, results: list[BenchResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    fieldnames = list(rows[0].keys()) if rows else list(BenchResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_result(result: BenchResult) -> None:
    check = "skipped"
    if result.passed_check is True:
        check = f"ok max={result.max_abs_diff:.2e}"
    elif result.passed_check is False:
        check = f"FAIL max={result.max_abs_diff:.2e}"

    print(
        f"{result.case:30s} "
        f"{result.dtype:7s} "
        f"thr={result.threads:>7s} "
        f"best={result.best_ms:8.2f} ms "
        f"median={result.median_ms:8.2f} ms "
        f"out={str(result.output_shape):16s} "
        f"check={check}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("smoke", "standard", "full"),
        default="standard",
        help="Benchmark case profile.",
    )
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats per case.")
    parser.add_argument("--warmups", type=int, default=1, help="Warmup calls before timing.")
    parser.add_argument(
        "--threads",
        type=parse_threads,
        default=parse_threads("1,4,8,default"),
        help="Comma-separated LSRESIZE_NUM_THREADS values; use 'default' to unset.",
    )
    parser.add_argument(
        "--check-max-elements",
        type=int,
        default=300_000,
        help="Run Python fallback parity checks only when input size is at most this value.",
    )
    parser.add_argument("--skip-checks", action="store_true", help="Disable Python fallback checks.")
    parser.add_argument("--atol", type=float, default=5e-5, help="Absolute tolerance for parity checks.")
    parser.add_argument("--rtol", type=float, default=5e-5, help="Relative tolerance for parity checks.")
    parser.add_argument(
        "--batched-axis",
        choices=("env", "off", "on"),
        default="env",
        help="Control LSRESIZE_BATCHED_AXIS for this run.",
    )
    parser.add_argument(
        "--batch-lines",
        type=int,
        default=None,
        help="Set LSRESIZE_BATCH_LINES for batched-axis runs.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not has_cpp():
        print("Native extension splineops._lsresize is not available.", file=sys.stderr)
        return 2

    if args.repeats <= 0:
        print("--repeats must be positive", file=sys.stderr)
        return 2
    if args.warmups < 0:
        print("--warmups must be non-negative", file=sys.stderr)
        return 2
    if args.batch_lines is not None and args.batch_lines <= 0:
        print("--batch-lines must be positive", file=sys.stderr)
        return 2

    if args.batched_axis == "on":
        os.environ["LSRESIZE_BATCHED_AXIS"] = "1"
    elif args.batched_axis == "off":
        os.environ.pop("LSRESIZE_BATCHED_AXIS", None)
    if args.batch_lines is not None:
        os.environ["LSRESIZE_BATCH_LINES"] = str(args.batch_lines)

    cases = cases_for_profile(args.profile)
    results: list[BenchResult] = []

    print("splineops native resize benchmark")
    print(f"profile={args.profile} repeats={args.repeats} warmups={args.warmups}")
    print(f"threads={','.join(args.threads)}")
    print(
        "batched_axis="
        f"{os.environ.get('LSRESIZE_BATCHED_AXIS', '<unset>')} "
        f"batch_lines={os.environ.get('LSRESIZE_BATCH_LINES', '<unset>')}"
    )
    print(f"python={platform.python_version()} numpy={np.__version__}")
    print(f"platform={platform.platform()}")
    print()

    for case in cases:
        input_size = int(np.prod(case.shape))
        should_check = (not args.skip_checks) and (input_size <= args.check_max_elements)

        for thread_value in args.threads:
            samples, native_out = time_native_case(
                case,
                threads=thread_value,
                warmups=args.warmups,
                repeats=args.repeats,
            )

            passed_check: bool | None = None
            max_abs_diff: float | None = None
            mean_abs_diff: float | None = None
            if should_check:
                passed_check, max_abs_diff, mean_abs_diff = check_against_python(
                    case,
                    native_out,
                    atol=args.atol,
                    rtol=args.rtol,
                )

            result = BenchResult(
                case=case.name,
                shape=case.shape,
                output_shape=tuple(int(v) for v in native_out.shape),
                zoom=case.zoom,
                method=case.method,
                dtype=case.dtype,
                threads=thread_value,
                best_ms=min(samples) * 1000.0,
                median_ms=statistics.median(samples) * 1000.0,
                mean_ms=statistics.fmean(samples) * 1000.0,
                repeats=args.repeats,
                warmups=args.warmups,
                checked=should_check,
                max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff,
                passed_check=passed_check,
            )
            results.append(result)
            print_result(result)

    failed = [r for r in results if r.passed_check is False]

    payload = {
        "metadata": {
            "profile": args.profile,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "threads": args.threads,
            "batched_axis": os.environ.get("LSRESIZE_BATCHED_AXIS"),
            "batch_lines": os.environ.get("LSRESIZE_BATCH_LINES"),
            "check_max_elements": args.check_max_elements,
            "atol": args.atol,
            "rtol": args.rtol,
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

    if failed:
        print(f"\n{len(failed)} parity check(s) failed.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
