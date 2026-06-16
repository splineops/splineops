#!/usr/bin/env python3
"""Compare splineops resize against common Python imaging/numerics libraries.

The libraries do not all implement the same boundary conditions, coordinate
mapping, or antialiasing model. Treat quality deltas as semantic comparisons
against splineops, not as strict parity failures.
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
from typing import Callable

# Keep default runs reasonably comparable and reproducible. Users can override
# these before launching the script.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np


@dataclass(frozen=True)
class LibraryCase:
    name: str
    shape: tuple[int, ...]
    zoom: tuple[float, ...]
    method: str
    dtype: str
    pattern: str


@dataclass
class LibraryResult:
    backend: str
    implementation: str
    case: str
    status: str
    reason: str | None
    shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None
    zoom: tuple[float, ...]
    method: str
    dtype: str
    pattern: str
    repeats: int
    warmups: int
    best_ms: float | None
    median_ms: float | None
    mean_ms: float | None
    speedup_vs_splineops: float | None
    max_abs_diff: float | None
    mean_abs_diff: float | None
    p99_abs_diff: float | None
    rel_l2_diff: float | None


Runner = Callable[[np.ndarray, LibraryCase, tuple[int, ...]], tuple[np.ndarray, str]]


def module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except Exception:
        return None
    return str(getattr(module, "__version__", "<unknown>"))


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dtype_from_name(name: str) -> np.dtype:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported dtype {name!r}")


def output_shape_for(case: LibraryCase) -> tuple[int, ...]:
    return tuple(max(1, int(round(n * z))) for n, z in zip(case.shape, case.zoom))


def stable_seed(case: LibraryCase) -> int:
    payload = f"{case.name}|{case.shape}|{case.zoom}|{case.method}|{case.dtype}|{case.pattern}"
    return zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF


def make_input(case: LibraryCase) -> np.ndarray:
    dtype = dtype_from_name(case.dtype)
    rng = np.random.default_rng(stable_seed(case))
    shape = case.shape

    if case.pattern == "random":
        return rng.random(shape, dtype=dtype)
    if case.pattern == "constant":
        return np.full(shape, 3.25, dtype=dtype)
    if case.pattern == "impulse":
        x = np.zeros(shape, dtype=dtype)
        x[tuple(n // 2 for n in shape)] = 1.0
        return x
    if case.pattern == "ramp":
        grids = np.meshgrid(
            *[np.linspace(0.0, 1.0, n, dtype=np.float64) for n in shape],
            indexing="ij",
        )
        return (sum(grids) / float(len(grids))).astype(dtype, copy=False)
    if case.pattern == "checkerboard":
        grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        block = 4
        board = sum((g // block) for g in grids) % 2
        return board.astype(dtype, copy=False)
    if case.pattern == "sinusoid":
        grids = np.meshgrid(
            *[np.arange(n, dtype=np.float64) for n in shape],
            indexing="ij",
        )
        out = np.zeros(shape, dtype=np.float64)
        for ax, grid in enumerate(grids):
            freq = 0.09 + 0.07 * ax
            out += np.sin(2.0 * np.pi * freq * grid)
        return (out / float(len(grids))).astype(dtype, copy=False)

    raise ValueError(f"unknown pattern {case.pattern!r}")


def method_base(method: str) -> str:
    return method.removesuffix("-antialiasing")


def method_order(method: str) -> int:
    base = method_base(method)
    if base == "fast":
        return 0
    if base == "linear":
        return 1
    if base == "quadratic":
        return 2
    if base == "cubic":
        return 3
    raise ValueError(f"unsupported method {method!r}")


def run_splineops(arr: np.ndarray, case: LibraryCase, out_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    os.environ["SPLINEOPS_ACCEL"] = "always"
    from splineops.resize import resize

    out = resize(arr, output_size=out_shape, method=case.method)
    return np.asarray(out), "splineops.resize"


def run_scipy_ndimage(arr: np.ndarray, case: LibraryCase, out_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    from scipy import ndimage

    order = method_order(case.method)
    zoom = tuple(float(o) / float(i) for o, i in zip(out_shape, arr.shape))
    out = ndimage.zoom(
        arr,
        zoom,
        order=order,
        mode="mirror",
        prefilter=order > 1,
        grid_mode=False,
    )
    return np.asarray(out), f"scipy.ndimage.zoom(order={order}, mode=mirror, no-aa)"


def run_skimage_resize(arr: np.ndarray, case: LibraryCase, out_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    from skimage.transform import resize

    order = method_order(case.method)
    anti_aliasing = case.method.endswith("-antialiasing")
    out = resize(
        arr,
        out_shape,
        order=order,
        mode="reflect",
        anti_aliasing=anti_aliasing,
        preserve_range=True,
        clip=False,
    )
    return np.asarray(out, dtype=arr.dtype), (
        f"skimage.transform.resize(order={order}, aa={anti_aliasing})"
    )


def run_opencv(arr: np.ndarray, case: LibraryCase, out_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    if arr.ndim != 2:
        raise NotImplementedError("OpenCV backend is limited to 2-D scalar arrays")

    import cv2

    base = method_base(case.method)
    is_down = all(o <= i for o, i in zip(out_shape, arr.shape))
    if case.method.endswith("-antialiasing") and is_down:
        interpolation = cv2.INTER_AREA
        label = "INTER_AREA"
    elif base == "fast":
        interpolation = cv2.INTER_NEAREST
        label = "INTER_NEAREST"
    elif base == "linear":
        interpolation = cv2.INTER_LINEAR
        label = "INTER_LINEAR"
    elif base in {"quadratic", "cubic"}:
        interpolation = cv2.INTER_CUBIC
        label = "INTER_CUBIC"
    else:
        raise NotImplementedError(f"OpenCV unsupported method {case.method!r}")

    # OpenCV size is (width, height).
    out = cv2.resize(arr, (out_shape[1], out_shape[0]), interpolation=interpolation)
    return np.asarray(out), f"cv2.resize({label})"


def run_torch_interpolate(arr: np.ndarray, case: LibraryCase, out_shape: tuple[int, ...]) -> tuple[np.ndarray, str]:
    import torch
    import torch.nn.functional as F

    base = method_base(case.method)
    antialias = case.method.endswith("-antialiasing")
    if arr.ndim == 2:
        if base == "fast":
            mode = "nearest"
        elif base == "linear":
            mode = "bilinear"
        elif base == "cubic":
            mode = "bicubic"
        else:
            raise NotImplementedError(f"PyTorch unsupported 2-D method {case.method!r}")
        x = torch.from_numpy(np.ascontiguousarray(arr))[None, None]
    elif arr.ndim == 3:
        if antialias:
            raise NotImplementedError("PyTorch antialias is not available for 3-D interpolate")
        if base == "fast":
            mode = "nearest"
        elif base == "linear":
            mode = "trilinear"
        else:
            raise NotImplementedError(f"PyTorch unsupported 3-D method {case.method!r}")
        x = torch.from_numpy(np.ascontiguousarray(arr))[None, None]
    else:
        raise NotImplementedError("PyTorch backend supports only 2-D and 3-D scalar arrays")

    kwargs: dict[str, object] = {"size": out_shape, "mode": mode}
    if mode != "nearest":
        kwargs["align_corners"] = True
    if antialias and mode in {"bilinear", "bicubic"}:
        kwargs["antialias"] = True

    y = F.interpolate(x, **kwargs)
    return y[0, 0].detach().cpu().numpy(), (
        f"torch.nn.functional.interpolate(mode={mode}, "
        f"align_corners={kwargs.get('align_corners')}, aa={kwargs.get('antialias', False)})"
    )


BACKENDS: dict[str, tuple[str, Runner]] = {
    "splineops": ("splineops", run_splineops),
    "scipy": ("scipy", run_scipy_ndimage),
    "skimage": ("skimage", run_skimage_resize),
    "opencv": ("cv2", run_opencv),
    "torch": ("torch", run_torch_interpolate),
}


def smoke_cases() -> list[LibraryCase]:
    return [
        LibraryCase("2d_linear_down_ramp_f32", (256, 256), (0.5, 0.5), "linear", "float32", "ramp"),
        LibraryCase("2d_cubic_down_random_f32", (256, 256), (0.37, 0.37), "cubic", "float32", "random"),
        LibraryCase(
            "2d_cubic_aa_down_checker_f32",
            (256, 256),
            (0.37, 0.37),
            "cubic-antialiasing",
            "float32",
            "checkerboard",
        ),
    ]


def standard_cases() -> list[LibraryCase]:
    return smoke_cases() + [
        LibraryCase("2d_linear_down_random_f32", (512, 512), (0.37, 0.37), "linear", "float32", "random"),
        LibraryCase("2d_linear_aniso_random_f32", (512, 512), (1.0, 0.37), "linear", "float32", "random"),
        LibraryCase("2d_linear_up_sinusoid_f32", (512, 512), (1.25, 1.25), "linear", "float32", "sinusoid"),
        LibraryCase("2d_linear_down_random_f64", (512, 512), (0.37, 0.37), "linear", "float64", "random"),
        LibraryCase("2d_linear_aniso_random_f64", (512, 512), (1.0, 0.37), "linear", "float64", "random"),
        LibraryCase("2d_linear_up_sinusoid_f64", (512, 512), (1.25, 1.25), "linear", "float64", "sinusoid"),
        LibraryCase("2d_cubic_up_sinusoid_f32", (256, 256), (1.7, 1.7), "cubic", "float32", "sinusoid"),
        LibraryCase(
            "2d_linear_aa_down_random_f32",
            (512, 512),
            (0.37, 0.37),
            "linear-antialiasing",
            "float32",
            "random",
        ),
        LibraryCase(
            "2d_cubic_aa_down_random_f32",
            (512, 512),
            (0.37, 0.37),
            "cubic-antialiasing",
            "float32",
            "random",
        ),
        LibraryCase(
            "2d_cubic_aa_down_random_f64",
            (512, 512),
            (0.37, 0.37),
            "cubic-antialiasing",
            "float64",
            "random",
        ),
        LibraryCase("3d_linear_down_random_f32", (96, 96, 32), (0.5, 0.5, 0.5), "linear", "float32", "random"),
        LibraryCase("3d_cubic_down_random_f32", (96, 96, 32), (0.5, 0.5, 0.5), "cubic", "float32", "random"),
    ]


def full_cases() -> list[LibraryCase]:
    return standard_cases() + [
        LibraryCase("2d_linear_down_random_f32_large", (1024, 1024), (0.37, 0.37), "linear", "float32", "random"),
        LibraryCase("2d_linear_aniso_random_f32_large", (1024, 1024), (1.0, 0.37), "linear", "float32", "random"),
        LibraryCase("2d_linear_up_sinusoid_f32_large", (1024, 1024), (1.25, 1.25), "linear", "float32", "sinusoid"),
        LibraryCase("2d_cubic_down_random_f64", (1024, 1024), (0.37, 0.37), "cubic", "float64", "random"),
        LibraryCase(
            "2d_cubic_aa_down_random_f64_large",
            (1024, 1024),
            (0.37, 0.37),
            "cubic-antialiasing",
            "float64",
            "random",
        ),
        LibraryCase("3d_cubic_aniso_random_f32", (128, 128, 32), (1.0, 0.5, 1.0), "cubic", "float32", "random"),
    ]


def cases_for_profile(profile: str) -> list[LibraryCase]:
    if profile == "smoke":
        return smoke_cases()
    if profile == "standard":
        return standard_cases()
    if profile == "full":
        return full_cases()
    raise ValueError(f"unknown profile {profile!r}")


def parse_backends(value: str) -> list[str]:
    if value == "all":
        return list(BACKENDS)
    out = []
    for raw in value.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        if name not in BACKENDS:
            raise argparse.ArgumentTypeError(
                f"unknown backend {name!r}; choose from {', '.join(BACKENDS)} or all"
            )
        out.append(name)
    if not out:
        raise argparse.ArgumentTypeError("at least one backend is required")
    if "splineops" not in out:
        out.insert(0, "splineops")
    return out


def time_runner(
    runner: Runner,
    arr: np.ndarray,
    case: LibraryCase,
    out_shape: tuple[int, ...],
    repeats: int,
    warmups: int,
) -> tuple[np.ndarray, str, list[float]]:
    out: np.ndarray | None = None
    implementation = ""
    for _ in range(warmups):
        out, implementation = runner(arr, case, out_shape)

    samples: list[float] = []
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        out, implementation = runner(arr, case, out_shape)
        samples.append((time.perf_counter() - t0) * 1000.0)

    if out is None:
        out, implementation = runner(arr, case, out_shape)
    return np.asarray(out), implementation, samples


def quality_metrics(candidate: np.ndarray, reference: np.ndarray) -> tuple[float, float, float, float]:
    diff = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    max_abs = float(np.max(diff)) if diff.size else 0.0
    mean_abs = float(np.mean(diff)) if diff.size else 0.0
    p99 = float(np.quantile(diff, 0.99)) if diff.size else 0.0
    denom = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    rel_l2 = float(np.linalg.norm(diff.ravel()) / denom) if denom > 0.0 else 0.0
    return max_abs, mean_abs, p99, rel_l2


def skipped_result(
    backend: str,
    case: LibraryCase,
    reason: str,
) -> LibraryResult:
    return LibraryResult(
        backend=backend,
        implementation="",
        case=case.name,
        status="skipped",
        reason=reason,
        shape=case.shape,
        output_shape=None,
        zoom=case.zoom,
        method=case.method,
        dtype=case.dtype,
        pattern=case.pattern,
        repeats=0,
        warmups=0,
        best_ms=None,
        median_ms=None,
        mean_ms=None,
        speedup_vs_splineops=None,
        max_abs_diff=None,
        mean_abs_diff=None,
        p99_abs_diff=None,
        rel_l2_diff=None,
    )


def run_case(
    case: LibraryCase,
    backends: list[str],
    repeats: int,
    warmups: int,
) -> list[LibraryResult]:
    arr = make_input(case)
    out_shape = output_shape_for(case)

    reference, ref_impl, ref_samples = time_runner(
        run_splineops,
        arr,
        case,
        out_shape,
        repeats,
        warmups,
    )
    ref_best = min(ref_samples)
    ref_median = float(statistics.median(ref_samples))
    ref_mean = float(statistics.mean(ref_samples))

    results = [
        LibraryResult(
            backend="splineops",
            implementation=ref_impl,
            case=case.name,
            status="ok",
            reason=None,
            shape=case.shape,
            output_shape=tuple(int(n) for n in reference.shape),
            zoom=case.zoom,
            method=case.method,
            dtype=case.dtype,
            pattern=case.pattern,
            repeats=repeats,
            warmups=warmups,
            best_ms=ref_best,
            median_ms=ref_median,
            mean_ms=ref_mean,
            speedup_vs_splineops=1.0,
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
            p99_abs_diff=0.0,
            rel_l2_diff=0.0,
        )
    ]

    for backend in backends:
        if backend == "splineops":
            continue
        module_name, runner = BACKENDS[backend]
        if not has_module(module_name):
            results.append(skipped_result(backend, case, f"missing optional dependency {module_name!r}"))
            continue

        try:
            out, implementation, samples = time_runner(
                runner,
                arr,
                case,
                out_shape,
                repeats,
                warmups,
            )
            if tuple(out.shape) != out_shape:
                raise RuntimeError(f"returned shape {tuple(out.shape)}, expected {out_shape}")
            max_abs, mean_abs, p99, rel_l2 = quality_metrics(out, reference)
            median_ms = float(statistics.median(samples))
            results.append(
                LibraryResult(
                    backend=backend,
                    implementation=implementation,
                    case=case.name,
                    status="ok",
                    reason=None,
                    shape=case.shape,
                    output_shape=tuple(int(n) for n in out.shape),
                    zoom=case.zoom,
                    method=case.method,
                    dtype=case.dtype,
                    pattern=case.pattern,
                    repeats=repeats,
                    warmups=warmups,
                    best_ms=min(samples),
                    median_ms=median_ms,
                    mean_ms=float(statistics.mean(samples)),
                    speedup_vs_splineops=ref_median / median_ms if median_ms > 0.0 else float("inf"),
                    max_abs_diff=max_abs,
                    mean_abs_diff=mean_abs,
                    p99_abs_diff=p99,
                    rel_l2_diff=rel_l2,
                )
            )
        except NotImplementedError as exc:
            results.append(skipped_result(backend, case, str(exc)))
        except Exception as exc:
            failed = skipped_result(backend, case, f"{type(exc).__name__}: {exc}")
            failed.status = "error"
            results.append(failed)

    return results


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, results: list[LibraryResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    fieldnames = list(rows[0].keys()) if rows else list(LibraryResult.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_result(result: LibraryResult) -> None:
    if result.status != "ok":
        print(f"{result.case:34s} {result.backend:9s} {result.status:7s} {result.reason}")
        return
    assert result.median_ms is not None
    assert result.speedup_vs_splineops is not None
    assert result.max_abs_diff is not None
    assert result.rel_l2_diff is not None
    print(
        f"{result.case:34s} "
        f"{result.backend:9s} "
        f"median={result.median_ms:8.3f} ms "
        f"speed={result.speedup_vs_splineops:6.2f}x "
        f"max={result.max_abs_diff:9.2e} "
        f"rel_l2={result.rel_l2_diff:9.2e}"
    )


def print_summary(results: list[LibraryResult]) -> None:
    print("\nsummary versus splineops median time")
    by_backend: dict[str, list[LibraryResult]] = {}
    for result in results:
        if result.backend == "splineops" or result.status != "ok":
            continue
        by_backend.setdefault(result.backend, []).append(result)

    for backend in sorted(by_backend):
        rows = by_backend[backend]
        speedups = [r.speedup_vs_splineops for r in rows if r.speedup_vs_splineops is not None]
        rel = [r.rel_l2_diff for r in rows if r.rel_l2_diff is not None]
        if not speedups:
            continue
        print(
            f"{backend:9s} "
            f"cases={len(rows):2d} "
            f"faster={sum(s > 1.0 for s in speedups):2d}/{len(speedups):2d} "
            f"mean_speed={statistics.mean(speedups):6.2f}x "
            f"median_speed={statistics.median(speedups):6.2f}x "
            f"median_rel_l2={statistics.median(rel):9.2e}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard", "full"], default="standard")
    parser.add_argument("--backends", type=parse_backends, default=parse_backends("all"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument(
        "--splineops-threads",
        default="1",
        help="Value for LSRESIZE_NUM_THREADS; use 'default' to unset.",
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="Thread count passed to cv2.setNumThreads when OpenCV is available.",
    )
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Thread count passed to torch.set_num_threads when PyTorch is available.",
    )
    return parser.parse_args()


def configure_threads(args: argparse.Namespace) -> None:
    if args.splineops_threads == "default":
        os.environ.pop("LSRESIZE_NUM_THREADS", None)
    else:
        parsed = int(args.splineops_threads)
        if parsed <= 0:
            raise ValueError("--splineops-threads must be positive or 'default'")
        os.environ["LSRESIZE_NUM_THREADS"] = str(parsed)

    if has_module("cv2"):
        try:
            import cv2

            cv2.setNumThreads(max(1, int(args.opencv_threads)))
        except Exception:
            pass

    if has_module("torch"):
        try:
            import torch

            torch.set_num_threads(max(1, int(args.torch_threads)))
        except Exception:
            pass


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        print("--repeats must be positive", file=sys.stderr)
        return 2
    if args.warmups < 0:
        print("--warmups must be non-negative", file=sys.stderr)
        return 2

    configure_threads(args)
    os.environ["SPLINEOPS_ACCEL"] = "always"
    os.environ.setdefault("LSRESIZE_BATCHED_AXIS", "auto")

    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "splineops": module_version("splineops"),
        "scipy": module_version("scipy"),
        "skimage": module_version("skimage"),
        "opencv": module_version("cv2"),
        "torch": module_version("torch"),
    }

    print("resize library comparison")
    print(f"profile={args.profile} repeats={args.repeats} warmups={args.warmups}")
    print(f"backends={','.join(args.backends)}")
    print(
        "threads="
        f"splineops:{os.environ.get('LSRESIZE_NUM_THREADS', '<default>')} "
        f"opencv:{args.opencv_threads} torch:{args.torch_threads}"
    )
    print("versions=" + " ".join(f"{k}={v}" for k, v in versions.items()))
    print()

    results: list[LibraryResult] = []
    for case in cases_for_profile(args.profile):
        case_results = run_case(case, args.backends, args.repeats, args.warmups)
        results.extend(case_results)
        for result in case_results:
            print_result(result)
        print()

    print_summary(results)

    payload = {
        "metadata": {
            **versions,
            "profile": args.profile,
            "repeats": args.repeats,
            "warmups": args.warmups,
            "backends": args.backends,
            "LSRESIZE_NUM_THREADS": os.environ.get("LSRESIZE_NUM_THREADS", "<default>"),
            "LSRESIZE_BATCHED_AXIS": os.environ.get("LSRESIZE_BATCHED_AXIS", "<default:auto>"),
            "LSRESIZE_AVX2_LINEAR": os.environ.get(
                "LSRESIZE_AVX2_LINEAR",
                "<default:on-if-supported>",
            ),
            "LSRESIZE_FUSED_3D_TWO_AXIS_LINEAR": os.environ.get(
                "LSRESIZE_FUSED_3D_TWO_AXIS_LINEAR",
                "<default:on>",
            ),
            "LSRESIZE_FUSED_PROJECTION_AVG_RESTORE": os.environ.get(
                "LSRESIZE_FUSED_PROJECTION_AVG_RESTORE",
                "<default:single-thread-auto>",
            ),
            "LSRESIZE_LAST_AXIS_LINEAR_DIRECT": os.environ.get(
                "LSRESIZE_LAST_AXIS_LINEAR_DIRECT",
                "<default:on>",
            ),
            "LSRESIZE_GATHER_PREFILTER_SCALE": os.environ.get(
                "LSRESIZE_GATHER_PREFILTER_SCALE",
                "<default:on>",
            ),
            "LSRESIZE_3D_AXIS1_DIRECT_SCATTER": os.environ.get(
                "LSRESIZE_3D_AXIS1_DIRECT_SCATTER",
                "<default:on>",
            ),
            "coordinate_note": (
                "Backends use different coordinate, boundary, and antialiasing semantics; "
                "quality metrics are deltas against splineops, not parity checks."
            ),
        },
        "results": [asdict(r) for r in results],
    }

    if args.output_json is not None:
        write_json(args.output_json, payload)
        print(f"wrote JSON: {args.output_json}")
    if args.output_csv is not None:
        write_csv(args.output_csv, results)
        print(f"wrote CSV: {args.output_csv}")

    errored = [r for r in results if r.status == "error"]
    return 1 if errored else 0


if __name__ == "__main__":
    raise SystemExit(main())
