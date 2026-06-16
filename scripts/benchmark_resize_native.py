# splineops/scripts/benchmark_resize_native.py
"""
Benchmark spline resize backends.

This script is intended as the scoreboard before deeper resize-kernel rewrites.
It times the C++ backend and/or pure-Python fallback across fixed resize cases,
and can check one backend against the other for selected workloads.

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

Compare native and Python fallback:

    python scripts/benchmark_resize_native.py \\
        --profile smoke \\
        --backend both \\
        --threads default
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
    backend: str
    case: str
    shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    zoom: tuple[float, ...]
    method: str
    dtype: str
    threads: str
    batched_axis: str
    batch_lines: str
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


def parse_positive_int_list(value: str) -> list[int]:
    items: list[int] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        items.append(parsed)
    if not items:
        raise argparse.ArgumentTypeError("at least one value is required")
    return items


def parse_case_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


def set_threads(value: str) -> None:
    if value == "default":
        os.environ.pop("LSRESIZE_NUM_THREADS", None)
    else:
        os.environ["LSRESIZE_NUM_THREADS"] = value


def set_batched_axis(value: str) -> None:
    if value == "env":
        return
    if value == "off":
        os.environ["LSRESIZE_BATCHED_AXIS"] = "off"
    elif value == "on":
        os.environ["LSRESIZE_BATCHED_AXIS"] = "1"
    elif value == "auto":
        os.environ["LSRESIZE_BATCHED_AXIS"] = "auto"
    else:
        raise ValueError(f"Unsupported batched-axis setting: {value}")


def batched_axis_label() -> str:
    return os.environ.get("LSRESIZE_BATCHED_AXIS", "<default:auto>")


def native_knobs() -> dict[str, str]:
    return {
        "LSRESIZE_BATCHED_AXIS": os.environ.get("LSRESIZE_BATCHED_AXIS", "<default:auto>"),
        "LSRESIZE_BATCH_LINES": os.environ.get("LSRESIZE_BATCH_LINES", "<default:adaptive>"),
        "LSRESIZE_ROW_GATHER": os.environ.get("LSRESIZE_ROW_GATHER", "<default:on>"),
        "LSRESIZE_GATHER_PREFILTER_SCALE": os.environ.get(
            "LSRESIZE_GATHER_PREFILTER_SCALE",
            "<default:on>",
        ),
        "LSRESIZE_3D_AXIS1_DIRECT_SCATTER": os.environ.get(
            "LSRESIZE_3D_AXIS1_DIRECT_SCATTER",
            "<default:on>",
        ),
        "LSRESIZE_SPECIALIZED_PRESETS": os.environ.get(
            "LSRESIZE_SPECIALIZED_PRESETS",
            "<default:on>",
        ),
        "LSRESIZE_LINEAR_INTERP": os.environ.get(
            "LSRESIZE_LINEAR_INTERP",
            "<default:on>",
        ),
        "LSRESIZE_FUSED_2D_LINEAR": os.environ.get(
            "LSRESIZE_FUSED_2D_LINEAR",
            "<default:on>",
        ),
        "LSRESIZE_FUSED_3D_LINEAR": os.environ.get(
            "LSRESIZE_FUSED_3D_LINEAR",
            "<default:on>",
        ),
        "LSRESIZE_FUSED_3D_TWO_AXIS_LINEAR": os.environ.get(
            "LSRESIZE_FUSED_3D_TWO_AXIS_LINEAR",
            "<default:on>",
        ),
        "LSRESIZE_FUSED_PROJECTION_AVG_RESTORE": os.environ.get(
            "LSRESIZE_FUSED_PROJECTION_AVG_RESTORE",
            "<default:single-thread-auto>",
        ),
        "LSRESIZE_AVX2_LINEAR": os.environ.get(
            "LSRESIZE_AVX2_LINEAR",
            "<default:on-if-supported>",
        ),
        "LSRESIZE_LAST_AXIS_LINEAR_DIRECT": os.environ.get(
            "LSRESIZE_LAST_AXIS_LINEAR_DIRECT",
            "<default:on>",
        ),
        "LSRESIZE_2D_LINEAR_INTERP": os.environ.get(
            "LSRESIZE_2D_LINEAR_INTERP",
            "<default:on>",
        ),
        "LSRESIZE_2D_FLOAT_INTERP": os.environ.get(
            "LSRESIZE_2D_FLOAT_INTERP",
            "<default:on>",
        ),
        "LSRESIZE_PRECISION": os.environ.get(
            "LSRESIZE_PRECISION",
            "<default:auto-f32-2d/3d-interp-else-float64>",
        ),
        "LSRESIZE_NUM_THREADS": os.environ.get("LSRESIZE_NUM_THREADS", "<default:auto>"),
        "LSRESIZE_PLAN_CACHE_SIZE": os.environ.get("LSRESIZE_PLAN_CACHE_SIZE", "<default:32>"),
    }


def python_knobs() -> dict[str, str]:
    return {
        "SPLINEOPS_BLOCK": os.environ.get("SPLINEOPS_BLOCK", "<default:256>"),
        "SPLINEOPS_ACCUM": os.environ.get("SPLINEOPS_ACCUM", "<default:support>"),
        "SPLINEOPS_TILE_W": os.environ.get("SPLINEOPS_TILE_W", "<default:0>"),
        "SPLINEOPS_PLAN_CACHE": os.environ.get("SPLINEOPS_PLAN_CACHE", "<default:on>"),
        "SPLINEOPS_PLAN_CACHE_SIZE": os.environ.get("SPLINEOPS_PLAN_CACHE_SIZE", "<default:32>"),
        "LSRESIZE_PLAN_CACHE_SIZE": os.environ.get("LSRESIZE_PLAN_CACHE_SIZE", "<unset>"),
        "SPLINEOPS_AUTOTUNE": os.environ.get("SPLINEOPS_AUTOTUNE", "<default:off>"),
    }


def set_batch_lines(value: int | None) -> None:
    if value is None:
        os.environ.pop("LSRESIZE_BATCH_LINES", None)
    else:
        os.environ["LSRESIZE_BATCH_LINES"] = str(value)


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


def time_case(
    case: BenchCase,
    *,
    backend: str,
    threads: str,
    warmups: int,
    repeats: int,
) -> tuple[list[float], np.ndarray]:
    if backend == "native":
        set_threads(threads)
        module = load_resize_module("always")
    elif backend == "python":
        module = load_resize_module("never")
    else:
        raise ValueError(f"Unsupported backend: {backend}")

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


def check_against_reference(
    case: BenchCase,
    candidate_out: np.ndarray,
    *,
    backend: str,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float]:
    if backend == "native":
        mode = "never"
    elif backend == "python":
        if not has_cpp():
            raise RuntimeError("Native extension is not available for Python backend parity check")
        mode = "always"
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    x = make_input(case)
    module = load_resize_module(mode)
    ref_out = run_once(module, x, case)

    diff = np.abs(candidate_out.astype(np.float64) - ref_out.astype(np.float64))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    passed = bool(np.allclose(candidate_out, ref_out, atol=atol, rtol=rtol))
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
                        f"2d_linear_down_{shape[0]}_{dtype}",
                        shape,
                        (0.37, 0.37),
                        "linear",
                        dtype,
                    ),
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
                        f"2d_linear_aniso_{shape[0]}_{dtype}",
                        shape,
                        (1.0, 0.37),
                        "linear",
                        dtype,
                    ),
                    BenchCase(
                        f"2d_cubic_aniso_{shape[0]}_{dtype}",
                        shape,
                        (1.0, 0.37),
                        "cubic",
                        dtype,
                    ),
                    BenchCase(
                        f"2d_linear_up_{shape[0]}_{dtype}",
                        shape,
                        (1.25, 1.25),
                        "linear",
                        dtype,
                    ),
                ]
            )
    cases.extend(
        [
            BenchCase("3d_linear_down_f32", (128, 128, 32), (0.5, 0.5, 0.5), "linear", "float32"),
            BenchCase("3d_linear_two_axis01_f32", (128, 128, 32), (0.5, 0.5, 1.0), "linear", "float32"),
            BenchCase("3d_linear_two_axis02_f32", (128, 128, 32), (0.5, 1.0, 0.5), "linear", "float32"),
            BenchCase("3d_linear_two_axis12_f32", (128, 128, 32), (1.0, 0.5, 0.5), "linear", "float32"),
            BenchCase("3d_linear_aniso_f32", (128, 128, 32), (1.0, 0.5, 1.0), "linear", "float32"),
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
        f"{result.backend:6s} "
        f"{result.case:30s} "
        f"{result.dtype:7s} "
        f"thr={result.threads:>7s} "
        f"batch={result.batch_lines:>7s} "
        f"mode={result.batched_axis:>5s} "
        f"best={result.best_ms:8.2f} ms "
        f"median={result.median_ms:8.2f} ms "
        f"out={str(result.output_shape):16s} "
        f"check={check}"
    )


def print_batch_sweep_summary(results: list[BenchResult], batch_values: list[int | None]) -> None:
    if len(batch_values) <= 1:
        return

    print("\nbatch-lines sweep summary")
    groups: dict[tuple[str, str, str, str], list[BenchResult]] = {}
    for result in results:
        key = (result.backend, result.case, result.dtype, result.threads)
        groups.setdefault(key, []).append(result)

    first_label = "<unset>" if batch_values[0] is None else str(batch_values[0])
    for key in sorted(groups):
        group = groups[key]
        if len({r.batch_lines for r in group}) <= 1:
            continue
        baseline = next((r for r in group if r.batch_lines == first_label), None)
        if baseline is None:
            baseline = group[0]
        best = min(group, key=lambda r: r.best_ms)
        speedup = baseline.best_ms / best.best_ms if best.best_ms > 0.0 else float("inf")
        backend, case, dtype, threads = key
        print(
            f"{backend:6s} {case:30s} {dtype:7s} thr={threads:>7s} "
            f"best_batch={best.batch_lines:>7s} "
            f"best={best.best_ms:8.2f} ms "
            f"speedup_vs_{baseline.batch_lines}={speedup:5.2f}x"
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
        "--backend",
        choices=("native", "python", "both"),
        default="native",
        help="Backend to time. 'native' preserves the historical behavior.",
    )
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
        choices=("env", "off", "on", "auto"),
        default="env",
        help="Control LSRESIZE_BATCHED_AXIS for this run; unset/env uses the native default auto router.",
    )
    parser.add_argument(
        "--batch-lines",
        type=int,
        default=None,
        help="Set LSRESIZE_BATCH_LINES for batched-axis runs.",
    )
    parser.add_argument(
        "--batch-lines-sweep",
        type=parse_positive_int_list,
        default=None,
        help=(
            "Comma-separated LSRESIZE_BATCH_LINES values to sweep. "
            "When set, this overrides --batch-lines and runs each case for each value."
        ),
    )
    parser.add_argument("--cases", help="Comma-separated benchmark case names to include.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    backends = ("native", "python") if args.backend == "both" else (args.backend,)
    needs_cpp = "native" in backends
    cpp_available = has_cpp()

    if needs_cpp and not cpp_available:
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
    if args.batch_lines_sweep is not None and args.batched_axis == "off":
        print("--batch-lines-sweep has no effect with --batched-axis off", file=sys.stderr)
        return 2

    set_batched_axis(args.batched_axis)
    native_batch_line_values = (
        [int(v) for v in args.batch_lines_sweep]
        if args.batch_lines_sweep is not None
        else [args.batch_lines]
    )
    python_batch_line_values: list[int | None] = [None]

    cases = cases_for_profile(args.profile)
    case_filter = parse_case_filter(args.cases)
    if case_filter is not None:
        cases = [case for case in cases if case.name in case_filter]
        missing = sorted(case_filter - {case.name for case in cases})
        if missing:
            print(f"unknown case(s): {', '.join(missing)}", file=sys.stderr)
            return 2

    results: list[BenchResult] = []

    print("splineops resize benchmark")
    print(
        f"profile={args.profile} backend={args.backend} "
        f"repeats={args.repeats} warmups={args.warmups}"
    )
    print(f"threads={','.join(args.threads)}")
    print(
        "batched_axis="
        f"{batched_axis_label()} "
        f"batch_lines={','.join('<unset>' if v is None else str(v) for v in native_batch_line_values)}"
    )
    print("native_knobs=" + " ".join(f"{k}={v}" for k, v in native_knobs().items()))
    print("python_knobs=" + " ".join(f"{k}={v}" for k, v in python_knobs().items()))
    print(f"python={platform.python_version()} numpy={np.__version__}")
    print(f"platform={platform.platform()}")
    print()

    for case in cases:
        input_size = int(np.prod(case.shape))
        should_check = (not args.skip_checks) and (input_size <= args.check_max_elements)

        for backend in backends:
            thread_values = args.threads if backend == "native" else ["<n/a>"]
            batch_line_values = (
                native_batch_line_values
                if backend == "native"
                else python_batch_line_values
            )
            for thread_value in thread_values:
                for batch_lines_value in batch_line_values:
                    if backend == "native":
                        set_batch_lines(batch_lines_value)
                    else:
                        os.environ.pop("LSRESIZE_BATCH_LINES", None)

                    samples, out = time_case(
                        case,
                        backend=backend,
                        threads=thread_value,
                        warmups=args.warmups,
                        repeats=args.repeats,
                    )

                    passed_check: bool | None = None
                    max_abs_diff: float | None = None
                    mean_abs_diff: float | None = None
                    if should_check and (backend == "native" or cpp_available):
                        passed_check, max_abs_diff, mean_abs_diff = check_against_reference(
                            case,
                            out,
                            backend=backend,
                            atol=args.atol,
                            rtol=args.rtol,
                        )

                    result = BenchResult(
                        backend=backend,
                        case=case.name,
                        shape=case.shape,
                        output_shape=tuple(int(v) for v in out.shape),
                        zoom=case.zoom,
                        method=case.method,
                        dtype=case.dtype,
                        threads=thread_value,
                        batched_axis=batched_axis_label() if backend == "native" else "<n/a>",
                        batch_lines=(
                            os.environ.get("LSRESIZE_BATCH_LINES", "<unset>")
                            if backend == "native"
                            else "<n/a>"
                        ),
                        best_ms=min(samples) * 1000.0,
                        median_ms=statistics.median(samples) * 1000.0,
                        mean_ms=statistics.fmean(samples) * 1000.0,
                        repeats=args.repeats,
                        warmups=args.warmups,
                        checked=should_check and (backend == "native" or cpp_available),
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
            "backend": args.backend,
            "backends": list(backends),
            "repeats": args.repeats,
            "warmups": args.warmups,
            "threads": args.threads,
            "batched_axis": batched_axis_label(),
            "batched_axis_env": os.environ.get("LSRESIZE_BATCHED_AXIS"),
            "batch_lines": [None if v is None else int(v) for v in native_batch_line_values],
            "native_knobs": native_knobs(),
            "python_knobs": python_knobs(),
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

    print_batch_sweep_summary(results, native_batch_line_values)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
