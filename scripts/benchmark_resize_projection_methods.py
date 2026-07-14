#!/usr/bin/env python3
"""Compare equal-degree least-squares and oblique resize projection methods.

This benchmark is intentionally separate from the library comparison suite. It
compares splineops implementations of three related method families:

* interpolation: no projection, ``analy_degree=-1``
* oblique: lower-degree analysis with the same synthesis degree
* least-squares: equal interpolation/analysis/synthesis degrees

The goal is to keep the method-positioning evidence reproducible: oblique is
the production antialiasing default, while equal-degree least-squares remains a
reference/advanced configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


@dataclass(frozen=True)
class ProjectionCase:
    name: str
    shape: tuple[int, ...]
    zoom: tuple[float, ...]
    pattern: str


@dataclass
class ProjectionResult:
    case: str
    pattern: str
    shape: tuple[int, ...]
    zoom: tuple[float, ...]
    output_shape: tuple[int, ...]
    degree: int
    dtype: str
    repeats: int
    warmups: int
    least_squares_ms: float
    oblique_ms: float
    interpolation_ms: float
    oblique_speedup_vs_least_squares: float
    interpolation_speedup_vs_least_squares: float
    down_rel_l2_oblique_vs_least_squares: float
    down_max_abs_oblique_vs_least_squares: float
    roundtrip_psnr_least_squares: float
    roundtrip_psnr_oblique: float
    roundtrip_psnr_interpolation: float
    roundtrip_rel_l2_least_squares: float
    roundtrip_rel_l2_oblique: float
    roundtrip_rel_l2_interpolation: float
    roundtrip_ssim_least_squares: float | None
    roundtrip_ssim_oblique: float | None
    roundtrip_ssim_interpolation: float | None
    down_min_least_squares: float
    down_max_least_squares: float
    down_mean_least_squares: float
    down_min_oblique: float
    down_max_oblique: float
    down_mean_oblique: float


def parse_csv_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected at least one value")
    return items


def parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for item in parse_csv_list(value):
        parsed = int(item)
        if parsed <= 0:
            raise argparse.ArgumentTypeError("values must be positive integers")
        out.append(parsed)
    return out


def dtype_from_name(name: str) -> np.dtype:
    if name == "float32":
        return np.dtype(np.float32)
    if name == "float64":
        return np.dtype(np.float64)
    raise ValueError(f"unsupported dtype {name!r}")


def stable_seed(case: ProjectionCase) -> int:
    payload = f"{case.name}|{case.shape}|{case.zoom}|{case.pattern}"
    return zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF


def cases_for_profile(profile: str) -> list[ProjectionCase]:
    if profile == "smoke":
        return [
            ProjectionCase("camera_half", (384, 384), (0.5, 0.5), "camera"),
            ProjectionCase("random_half", (384, 384), (0.5, 0.5), "random"),
        ]
    if profile == "standard":
        patterns = [
            "camera",
            "coins",
            "low_sine",
            "near_nyquist",
            "checkerboard",
            "random",
        ]
        zooms = [(0.5, 0.5), (0.37, 0.61), (0.25, 0.25)]
        return [
            ProjectionCase(
                f"{pattern}_{int(100 * zoom[0])}_{int(100 * zoom[1])}",
                (384, 384),
                zoom,
                pattern,
            )
            for pattern in patterns
            for zoom in zooms
        ]
    if profile == "stability":
        return [
            ProjectionCase(f"ramp_1d_{length}", (length,), (0.37,), "ramp")
            for length in (4096, 16384, 65536)
        ]
    raise ValueError(f"unknown profile {profile!r}")


def resize_image_fixture(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if arr.shape == shape:
        return arr.astype(np.float64, copy=False)
    try:
        from skimage.transform import resize
    except Exception:
        slices = tuple(slice(0, min(n, s)) for n, s in zip(arr.shape, shape))
        out = np.zeros(shape, dtype=np.float64)
        target = tuple(slice(0, s.stop) for s in slices)
        out[target] = arr[slices]
        return out
    return resize(
        arr,
        shape,
        order=3,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
        clip=False,
    ).astype(np.float64, copy=False)


def generated_image_mix(shape: tuple[int, ...]) -> np.ndarray:
    grids = np.meshgrid(
        *[np.linspace(0.0, 1.0, n, dtype=np.float64) for n in shape],
        indexing="ij",
    )
    out = sum(grids) / float(len(grids))
    for axis, grid in enumerate(grids):
        out += 0.08 * np.sin(2.0 * np.pi * (4 + 2 * axis) * grid)
    if len(shape) >= 2:
        out += 0.25 * ((grids[0] > 0.35) & (grids[1] > 0.55))
    out -= float(np.min(out))
    peak = float(np.max(out))
    return out / peak if peak > 0.0 else out


def make_input(case: ProjectionCase, dtype: np.dtype) -> np.ndarray:
    shape = case.shape
    rng = np.random.default_rng(stable_seed(case))

    if case.pattern in {"camera", "coins"} and len(shape) == 2:
        try:
            from skimage import data as skdata

            if case.pattern == "camera":
                arr = skdata.camera().astype(np.float64) / 255.0
            else:
                arr = skdata.coins().astype(np.float64) / 255.0
            return resize_image_fixture(arr, shape).astype(dtype, copy=False)
        except Exception:
            return generated_image_mix(shape).astype(dtype, copy=False)

    if case.pattern == "random":
        return rng.random(shape, dtype=dtype)
    if case.pattern == "constant":
        return np.full(shape, 0.75, dtype=dtype)
    if case.pattern == "ramp":
        grids = np.meshgrid(
            *[np.linspace(0.0, 1.0, n, dtype=np.float64) for n in shape],
            indexing="ij",
        )
        return (sum(grids) / float(len(grids))).astype(dtype, copy=False)
    if case.pattern == "low_sine":
        grids = np.meshgrid(
            *[np.arange(n, dtype=np.float64) for n in shape],
            indexing="ij",
        )
        out = np.zeros(shape, dtype=np.float64)
        for axis, grid in enumerate(grids):
            out += np.sin(2.0 * np.pi * (5 + 2 * axis) * grid / shape[axis])
        return (0.5 + 0.2 * out / float(len(shape))).astype(dtype, copy=False)
    if case.pattern == "near_nyquist":
        grids = np.meshgrid(
            *[np.arange(n, dtype=np.float64) for n in shape],
            indexing="ij",
        )
        out = np.zeros(shape, dtype=np.float64)
        for axis, grid in enumerate(grids):
            out += np.sin(2.0 * np.pi * (0.31 + 0.08 * axis) * grid)
        return (0.5 + 0.25 * out / float(len(shape))).astype(dtype, copy=False)
    if case.pattern == "checkerboard":
        grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
        board = sum((grid // 3) for grid in grids) % 2
        return board.astype(dtype, copy=False)

    raise ValueError(f"unknown pattern {case.pattern!r}")


def crop_center(arr: np.ndarray) -> np.ndarray:
    slices = []
    for n in arr.shape:
        if n < 32:
            pad = max(1, n // 20)
        else:
            pad = min(n // 4, max(8, int(0.08 * n)))
        slices.append(slice(pad, n - pad if pad else n))
    return arr[tuple(slices)]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((aa - bb) ** 2))
    if mse == 0.0:
        return float("inf")
    data_range = float(np.max(bb) - np.min(bb))
    if data_range <= 0.0:
        data_range = 1.0
    return 20.0 * math.log10(data_range / math.sqrt(mse))


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(bb.ravel()))
    return float(np.linalg.norm((aa - bb).ravel()) / denom) if denom > 0.0 else 0.0


def maybe_ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.ndim != 2:
        return None
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return None
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    data_range = float(np.max(bb) - np.min(bb))
    if data_range <= 0.0:
        data_range = 1.0
    return float(structural_similarity(bb, aa, data_range=data_range))


def degrees_for(kind: str, degree: int) -> tuple[int, int, int]:
    if kind == "least_squares":
        return degree, degree, degree
    if kind == "oblique":
        analy_degree = 0 if degree == 1 else 1
        return degree, analy_degree, degree
    if kind == "interpolation":
        return degree, -1, degree
    raise ValueError(f"unknown method family {kind!r}")


def resize_kind(
    arr: np.ndarray,
    *,
    degree: int,
    kind: str,
    zoom: tuple[float, ...] | None = None,
    output_size: tuple[int, ...] | None = None,
) -> np.ndarray:
    from splineops.resize import resize_degrees

    interp_degree, analy_degree, synthe_degree = degrees_for(kind, degree)
    return resize_degrees(
        arr,
        zoom_factors=zoom,
        output_size=output_size,
        interp_degree=interp_degree,
        analy_degree=analy_degree,
        synthe_degree=synthe_degree,
    )


def timed(fn: Callable[[], np.ndarray], repeats: int, warmups: int) -> tuple[float, np.ndarray]:
    for _ in range(warmups):
        fn()
    values: list[float] = []
    out: np.ndarray | None = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        values.append((time.perf_counter() - t0) * 1000.0)
    assert out is not None
    return float(statistics.median(values)), out


def run_case(
    case: ProjectionCase,
    *,
    degree: int,
    dtype_name: str,
    repeats: int,
    warmups: int,
) -> ProjectionResult:
    arr = make_input(case, dtype_from_name(dtype_name))
    down: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}
    for kind in ("least_squares", "oblique", "interpolation"):
        timings[kind], down[kind] = timed(
            lambda kind=kind: resize_kind(arr, degree=degree, kind=kind, zoom=case.zoom),
            repeats,
            warmups,
        )

    roundtrip: dict[str, np.ndarray] = {}
    for kind, y in down.items():
        roundtrip[kind] = resize_kind(
            y,
            degree=degree,
            kind=kind,
            output_size=tuple(arr.shape),
        )

    arr_crop = crop_center(arr.astype(np.float64, copy=False))
    rt_crop = {
        kind: crop_center(value.astype(np.float64, copy=False))
        for kind, value in roundtrip.items()
    }
    ls_down = down["least_squares"].astype(np.float64, copy=False)
    oblique_down = down["oblique"].astype(np.float64, copy=False)

    return ProjectionResult(
        case=case.name,
        pattern=case.pattern,
        shape=case.shape,
        zoom=case.zoom,
        output_shape=tuple(int(n) for n in ls_down.shape),
        degree=degree,
        dtype=dtype_name,
        repeats=repeats,
        warmups=warmups,
        least_squares_ms=timings["least_squares"],
        oblique_ms=timings["oblique"],
        interpolation_ms=timings["interpolation"],
        oblique_speedup_vs_least_squares=timings["least_squares"] / timings["oblique"],
        interpolation_speedup_vs_least_squares=timings["least_squares"] / timings["interpolation"],
        down_rel_l2_oblique_vs_least_squares=rel_l2(oblique_down, ls_down),
        down_max_abs_oblique_vs_least_squares=float(np.max(np.abs(oblique_down - ls_down))),
        roundtrip_psnr_least_squares=psnr(rt_crop["least_squares"], arr_crop),
        roundtrip_psnr_oblique=psnr(rt_crop["oblique"], arr_crop),
        roundtrip_psnr_interpolation=psnr(rt_crop["interpolation"], arr_crop),
        roundtrip_rel_l2_least_squares=rel_l2(rt_crop["least_squares"], arr_crop),
        roundtrip_rel_l2_oblique=rel_l2(rt_crop["oblique"], arr_crop),
        roundtrip_rel_l2_interpolation=rel_l2(rt_crop["interpolation"], arr_crop),
        roundtrip_ssim_least_squares=maybe_ssim(rt_crop["least_squares"], arr_crop),
        roundtrip_ssim_oblique=maybe_ssim(rt_crop["oblique"], arr_crop),
        roundtrip_ssim_interpolation=maybe_ssim(rt_crop["interpolation"], arr_crop),
        down_min_least_squares=float(np.min(ls_down)),
        down_max_least_squares=float(np.max(ls_down)),
        down_mean_least_squares=float(np.mean(ls_down)),
        down_min_oblique=float(np.min(oblique_down)),
        down_max_oblique=float(np.max(oblique_down)),
        down_mean_oblique=float(np.mean(oblique_down)),
    )


def write_csv(path: Path, rows: list[ProjectionResult]) -> None:
    data = [asdict(row) for row in rows]
    if not data:
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        writer.writeheader()
        writer.writerows(data)


def write_json(path: Path, rows: list[ProjectionResult], args: argparse.Namespace) -> None:
    payload: dict[str, Any] = {
        "metadata": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "splineops_accel": os.environ.get("SPLINEOPS_ACCEL", "<unset>"),
            "threads": os.environ.get("LSRESIZE_NUM_THREADS", "<default>"),
            "persistent_threads": os.environ.get(
                "LSRESIZE_PERSISTENT_THREADS", "<default:on>"
            ),
            "plan_cache_size": os.environ.get(
                "LSRESIZE_PLAN_CACHE_SIZE", "<default:32>"
            ),
            "plan_cache_bytes": os.environ.get(
                "LSRESIZE_PLAN_CACHE_BYTES", "<default:134217728>"
            ),
            "profile": args.profile,
            "degrees": args.degrees,
            "dtypes": args.dtypes,
            "repeats": args.repeats,
            "warmups": args.warmups,
        },
        "results": [asdict(row) for row in rows],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def summarize(rows: list[ProjectionResult]) -> None:
    if not rows:
        print("no rows")
        return
    for degree in sorted({row.degree for row in rows}):
        group = [row for row in rows if row.degree == degree]
        speedups = [row.oblique_speedup_vs_least_squares for row in group]
        psnr_ls_wins = sum(
            row.roundtrip_psnr_least_squares
            >= max(row.roundtrip_psnr_oblique, row.roundtrip_psnr_interpolation)
            for row in group
        )
        psnr_oblique_wins = sum(
            row.roundtrip_psnr_oblique
            >= max(row.roundtrip_psnr_least_squares, row.roundtrip_psnr_interpolation)
            for row in group
        )
        psnr_interp_wins = sum(
            row.roundtrip_psnr_interpolation
            >= max(row.roundtrip_psnr_least_squares, row.roundtrip_psnr_oblique)
            for row in group
        )
        ssim_rows = [
            row for row in group
            if row.roundtrip_ssim_least_squares is not None
            and row.roundtrip_ssim_oblique is not None
            and row.roundtrip_ssim_interpolation is not None
        ]
        ssim_oblique_wins = sum(
            (row.roundtrip_ssim_oblique or -1.0)
            >= max(row.roundtrip_ssim_least_squares or -1.0, row.roundtrip_ssim_interpolation or -1.0)
            for row in ssim_rows
        )
        print(
            f"degree={degree} cases={len(group)} "
            f"oblique_faster={sum(v > 1.0 for v in speedups)}/{len(speedups)} "
            f"median_oblique_speedup={statistics.median(speedups):.2f}x "
            f"psnr_wins_ls/oblique/interp={psnr_ls_wins}/{psnr_oblique_wins}/{psnr_interp_wins} "
            f"ssim_oblique_wins={ssim_oblique_wins}/{len(ssim_rows)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "standard", "stability"], default="standard")
    parser.add_argument("--degrees", default="1,3", help="Comma-separated spline degrees.")
    parser.add_argument("--dtypes", default="float32,float64", help="Comma-separated dtypes.")
    parser.add_argument("--threads", default="1", help="LSRESIZE_NUM_THREADS value, or 'default'.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups must be non-negative")

    os.environ["SPLINEOPS_ACCEL"] = "always"
    if args.threads == "default":
        os.environ.pop("LSRESIZE_NUM_THREADS", None)
    else:
        int(args.threads)
        os.environ["LSRESIZE_NUM_THREADS"] = args.threads

    degrees = parse_int_csv(args.degrees)
    dtypes = parse_csv_list(args.dtypes)

    rows: list[ProjectionResult] = []
    for case in cases_for_profile(args.profile):
        for dtype_name in dtypes:
            dtype_from_name(dtype_name)
            for degree in degrees:
                if degree < 1 or degree > 3:
                    raise SystemExit("projection method comparison supports degrees 1..3")
                result = run_case(
                    case,
                    degree=degree,
                    dtype_name=dtype_name,
                    repeats=args.repeats,
                    warmups=args.warmups,
                )
                rows.append(result)
                print(
                    f"{result.case:24s} dtype={dtype_name:7s} degree={degree} "
                    f"oblique/ls={result.oblique_speedup_vs_least_squares:.2f}x "
                    f"psnr_ls/oblique/interp="
                    f"{result.roundtrip_psnr_least_squares:.2f}/"
                    f"{result.roundtrip_psnr_oblique:.2f}/"
                    f"{result.roundtrip_psnr_interpolation:.2f} "
                    f"down_range_ls=[{result.down_min_least_squares:.3g},"
                    f"{result.down_max_least_squares:.3g}] "
                    f"down_range_oblique=[{result.down_min_oblique:.3g},"
                    f"{result.down_max_oblique:.3g}]"
                )

    summarize(rows)
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
        print(f"wrote CSV: {args.output_csv}")
    if args.output_json is not None:
        write_json(args.output_json, rows, args)
        print(f"wrote JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
