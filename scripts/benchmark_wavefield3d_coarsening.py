#!/usr/bin/env python3
"""Run the frozen controlled 3-D spectral-coarsening validation.

The protocol is recorded in ``benchmarks/wavefield3d/PROTOCOL.md``.  It uses
manufactured mirror-compatible cosine fields so the resolvable coarse-grid
target is known analytically rather than supplied by one of the libraries.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence, cast

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("SPLINEOPS_ACCEL", "always")

import numpy as np
from scipy import ndimage

LOW_MODE_COUNT = 12
HIGH_MODE_COUNT = 12
LOW_NYQUIST_FRACTION = 0.55
HIGH_NYQUIST_FRACTION = 1.15
SOURCE_NYQUIST_FRACTION = 0.95
NUISANCE_RATIOS = (0.25, 0.65, 1.00)
TEST_SEEDS_PER_GEOMETRY = 8
BOOTSTRAP_SEED = 20260716
BOOTSTRAP_RESAMPLES = 20_000
MEAN_REDUCTION_MARGIN = 0.20
CI_REDUCTION_MARGIN = 0.10
WIN_FRACTION_MARGIN = 0.90
TIMING_WARMUPS = 2
TIMING_REPEATS = 7
TIMING_FRAMES = 8

ArrayCall = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Geometry:
    key: str
    source_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    seed_start: int


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    semantics: str
    target_grid: str
    predeclared: bool
    prepare: Callable[[Geometry], ArrayCall]


GEOMETRIES = (
    Geometry("isotropic_2x", (65, 65, 65), (33, 33, 33), 2000),
    Geometry("anisotropic_noninteger", (73, 81, 65), (29, 41, 33), 2100),
    Geometry("mixed_reduction", (97, 81, 65), (25, 41, 49), 2200),
)


def protocol_path() -> Path:
    return (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "wavefield3d"
        / "PROTOCOL.md"
    )


def sha256_text(path: Path) -> str:
    """Hash UTF-8 text after universal-newline normalization."""

    normalized = path.read_text(encoding="utf-8").encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip() or None
    return platform.processor() or None


def normalized_coordinates(
    source_shape: Sequence[int], output_shape: Sequence[int], *, grid: str
) -> tuple[np.ndarray, ...]:
    coordinates: list[np.ndarray] = []
    for source, output in zip(source_shape, output_shape):
        if grid == "endpoint":
            source_indices = np.linspace(0.0, float(source - 1), output)
        elif grid == "half_pixel":
            source_indices = (np.arange(output, dtype=np.float64) + 0.5) * (
                float(source) / float(output)
            ) - 0.5
        else:
            raise ValueError(f"unknown grid {grid!r}")
        coordinates.append(source_indices / float(source - 1))
    return tuple(coordinates)


def source_coordinates(shape: Sequence[int]) -> tuple[np.ndarray, ...]:
    return tuple(np.linspace(0.0, 1.0, size) for size in shape)


def evaluate_modes(
    coordinates: Sequence[np.ndarray],
    modes: Sequence[tuple[int, int, int]],
    coefficients: Sequence[float] | np.ndarray,
    *,
    dtype: np.dtype | type = np.float64,
) -> np.ndarray:
    if len(coordinates) != 3:
        raise ValueError("the frozen study requires three-dimensional coordinates")
    if len(modes) != len(coefficients):
        raise ValueError("mode and coefficient counts differ")
    shape = tuple(len(axis) for axis in coordinates)
    output = np.zeros(shape, dtype=np.float64)
    for mode, coefficient in zip(modes, coefficients):
        term: np.ndarray | float = float(coefficient)
        for axis, (values, frequency) in enumerate(zip(coordinates, mode)):
            view_shape = [1, 1, 1]
            view_shape[axis] = len(values)
            term = term * np.cos(np.pi * frequency * values).reshape(view_shape)
        output += term
    return output.astype(dtype, copy=False)


def generate_mode_groups(geometry: Geometry, seed: int) -> tuple[
    list[tuple[int, int, int]],
    list[tuple[int, int, int]],
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(seed)

    low_maxima = [
        max(1, int(np.floor(LOW_NYQUIST_FRACTION * (output - 1))))
        for output in geometry.output_shape
    ]
    low_modes = [
        cast(
            tuple[int, int, int],
            tuple(int(rng.integers(1, maximum + 1)) for maximum in low_maxima),
        )
        for _ in range(LOW_MODE_COUNT)
    ]

    high_modes: list[tuple[int, int, int]] = []
    for _ in range(HIGH_MODE_COUNT):
        high_axis = int(rng.integers(0, 3))
        mode: list[int] = []
        for axis, (source, output, low_maximum) in enumerate(
            zip(geometry.source_shape, geometry.output_shape, low_maxima)
        ):
            if axis == high_axis:
                lower = int(np.ceil(HIGH_NYQUIST_FRACTION * (output - 1)))
                upper = int(np.floor(SOURCE_NYQUIST_FRACTION * (source - 1)))
                if upper < lower:
                    raise ValueError(f"invalid high-frequency range for {geometry.key}")
                mode.append(int(rng.integers(lower, upper + 1)))
            else:
                mode.append(int(rng.integers(1, low_maximum + 1)))
        high_modes.append(cast(tuple[int, int, int], tuple(mode)))

    low_coefficients = rng.normal(size=LOW_MODE_COUNT)
    high_coefficients = rng.normal(size=HIGH_MODE_COUNT)
    low_coefficients /= np.linalg.norm(low_coefficients)
    high_coefficients /= np.linalg.norm(high_coefficients)
    return low_modes, high_modes, low_coefficients, high_coefficients


def relative_l2(actual: np.ndarray, target: np.ndarray) -> float:
    actual64 = np.asarray(actual, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    denominator = float(np.linalg.norm(target64.ravel()))
    if denominator == 0.0:
        raise ValueError("target norm is zero")
    return float(np.linalg.norm((actual64 - target64).ravel()) / denominator)


def endpoint_index_grid(
    source_shape: Sequence[int], output_shape: Sequence[int]
) -> tuple[np.ndarray, ...]:
    vectors = [
        np.linspace(0.0, float(source - 1), output, dtype=np.float64)
        for source, output in zip(source_shape, output_shape)
    ]
    return tuple(np.meshgrid(*vectors, indexing="ij", sparse=False))


def prepare_splineops(geometry: Geometry, *, antialias: bool) -> ArrayCall:
    from splineops import ResizePlan

    method = "cubic-antialiasing" if antialias else "cubic"
    plan = ResizePlan(
        geometry.source_shape,
        output_size=geometry.output_shape,
        method=method,
    )

    def apply(values: np.ndarray) -> np.ndarray:
        return np.asarray(plan(values), dtype=np.float32)

    return apply


def prepare_scipy(geometry: Geometry, *, antialias: bool) -> ArrayCall:
    grid = endpoint_index_grid(geometry.source_shape, geometry.output_shape)
    factors = tuple(
        (source - 1) / (output - 1)
        for source, output in zip(geometry.source_shape, geometry.output_shape)
    )
    sigma = tuple(max(0.0, (factor - 1.0) / 2.0) for factor in factors)

    def apply(values: np.ndarray) -> np.ndarray:
        source = np.asarray(values, dtype=np.float32)
        if antialias:
            source = ndimage.gaussian_filter(source, sigma=sigma, mode="reflect")
        return np.asarray(
            ndimage.map_coordinates(
                source,
                grid,
                order=3,
                mode="reflect",
                prefilter=True,
            ),
            dtype=np.float32,
        )

    return apply


def prepare_skimage(geometry: Geometry) -> ArrayCall:
    try:
        from skimage.transform import resize
    except ImportError as exc:  # pragma: no cover - optional dependency guidance
        raise RuntimeError("Install the study dependencies with `.[study]`.") from exc

    def apply(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            resize(
                values,
                geometry.output_shape,
                order=3,
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
                clip=False,
            ),
            dtype=np.float32,
        )

    return apply


def prepare_scipy_polyphase(geometry: Geometry) -> ArrayCall:
    """Prepare a separable FIR resampler matching the endpoint interval ratio."""

    from scipy.signal import resample_poly

    ratios: list[tuple[int, int]] = []
    for source, output in zip(geometry.source_shape, geometry.output_shape):
        divisor = math.gcd(source - 1, output - 1)
        ratios.append(((output - 1) // divisor, (source - 1) // divisor))

    def apply(values: np.ndarray) -> np.ndarray:
        output = np.asarray(values, dtype=np.float32)
        for axis, (up, down) in enumerate(ratios):
            output = resample_poly(
                output,
                up,
                down,
                axis=axis,
                padtype="reflect",
            )
        if output.shape != geometry.output_shape:
            raise RuntimeError(
                f"polyphase output has shape {output.shape}, "
                f"expected {geometry.output_shape}"
            )
        return np.asarray(output, dtype=np.float32)

    return apply


def prepare_torch(geometry: Geometry, *, mode: str) -> ArrayCall:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - optional dependency guidance
        raise RuntimeError("Install the study dependencies with `.[study]`.") from exc

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    def apply(values: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32))[
            None, None
        ]
        if mode == "trilinear":
            output = functional.interpolate(
                tensor,
                size=geometry.output_shape,
                mode="trilinear",
                align_corners=True,
            )
        elif mode == "area":
            output = functional.interpolate(
                tensor,
                size=geometry.output_shape,
                mode="area",
            )
        else:  # pragma: no cover - internal invariant
            raise ValueError(f"unsupported torch mode {mode!r}")
        return output[0, 0].detach().cpu().numpy()

    return apply


def study_methods() -> list[Method]:
    return [
        Method(
            "splineops_projection",
            "SplineOps projection AA",
            "endpoint-aligned cubic oblique projection antialiasing",
            "endpoint",
            True,
            lambda geometry: prepare_splineops(geometry, antialias=True),
        ),
        Method(
            "splineops_interpolation",
            "SplineOps cubic, no AA",
            "endpoint-aligned cubic interpolation without antialiasing",
            "endpoint",
            True,
            lambda geometry: prepare_splineops(geometry, antialias=False),
        ),
        Method(
            "scipy_gaussian",
            "SciPy Gaussian + cubic",
            "Gaussian prefilter followed by endpoint-aligned cubic sampling",
            "endpoint",
            True,
            lambda geometry: prepare_scipy(geometry, antialias=True),
        ),
        Method(
            "scipy_cubic",
            "SciPy cubic, no AA",
            "endpoint-aligned cubic sampling without antialiasing",
            "endpoint",
            True,
            lambda geometry: prepare_scipy(geometry, antialias=False),
        ),
        Method(
            "skimage_resize",
            "scikit-image cubic AA",
            "half-pixel cubic resize with Gaussian antialiasing",
            "half_pixel",
            True,
            prepare_skimage,
        ),
        Method(
            "torch_trilinear",
            "PyTorch trilinear, no AA",
            "endpoint-aligned trilinear interpolation without antialiasing",
            "endpoint",
            True,
            lambda geometry: prepare_torch(geometry, mode="trilinear"),
        ),
        Method(
            "torch_area",
            "PyTorch area",
            "regional area resize evaluated against the half-pixel point-grid target",
            "half_pixel",
            True,
            lambda geometry: prepare_torch(geometry, mode="area"),
        ),
        Method(
            "scipy_polyphase",
            "SciPy polyphase FIR",
            "separable endpoint-ratio polyphase FIR with reflect padding",
            "endpoint",
            False,
            prepare_scipy_polyphase,
        ),
    ]


def evaluate_accuracy(
    methods: list[Method], *, seeds_per_geometry: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    case_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    total = len(GEOMETRIES) * seeds_per_geometry
    completed = 0

    for geometry in GEOMETRIES:
        prepared = {method.key: method.prepare(geometry) for method in methods}
        coordinates_source = source_coordinates(geometry.source_shape)
        target_coordinates = {
            grid: normalized_coordinates(
                geometry.source_shape,
                geometry.output_shape,
                grid=grid,
            )
            for grid in {method.target_grid for method in methods}
        }
        for offset in range(seeds_per_geometry):
            seed = geometry.seed_start + offset
            low_modes, high_modes, low_coefficients, high_coefficients = (
                generate_mode_groups(geometry, seed)
            )
            source_low = evaluate_modes(
                coordinates_source,
                low_modes,
                low_coefficients,
                dtype=np.float32,
            )
            source_high = evaluate_modes(
                coordinates_source,
                high_modes,
                high_coefficients,
                dtype=np.float32,
            )
            targets = {
                grid: evaluate_modes(
                    coordinates,
                    low_modes,
                    low_coefficients,
                    dtype=np.float64,
                )
                for grid, coordinates in target_coordinates.items()
            }
            base_id = f"{geometry.key}:seed{seed}"
            completed += 1
            print(f"[{completed:2d}/{total}] {base_id}", flush=True)

            for method in methods:
                output_low = prepared[method.key](source_low)
                output_high = prepared[method.key](source_high)
                if output_low.shape != geometry.output_shape:
                    raise RuntimeError(
                        f"{method.label} returned {output_low.shape}, "
                        f"expected {geometry.output_shape}"
                    )
                target = targets[method.target_grid]
                target_norm = float(np.linalg.norm(target.ravel()))
                passband_nrmse = relative_l2(output_low, target)
                stopband_leakage = float(
                    np.linalg.norm(np.asarray(output_high, np.float64).ravel())
                    / target_norm
                )
                component_rows.append(
                    {
                        "base_id": base_id,
                        "geometry": geometry.key,
                        "seed": seed,
                        "method": method.key,
                        "passband_nrmse": passband_nrmse,
                        "stopband_leakage": stopband_leakage,
                    }
                )
                for ratio in NUISANCE_RATIOS:
                    mixed_output = output_low + np.float32(ratio) * output_high
                    case_rows.append(
                        {
                            "case_id": f"{base_id}:ratio{ratio:.2f}",
                            "base_id": base_id,
                            "geometry": geometry.key,
                            "source_shape": "x".join(map(str, geometry.source_shape)),
                            "output_shape": "x".join(map(str, geometry.output_shape)),
                            "seed": seed,
                            "nuisance_ratio": ratio,
                            "method": method.key,
                            "target_grid": method.target_grid,
                            "nrmse": relative_l2(mixed_output, target),
                        }
                    )
    return case_rows, component_rows


def comparison_seed(method: str) -> int:
    return BOOTSTRAP_SEED + (zlib.crc32(method.encode("utf-8")) & 0xFFFF)


def bootstrap_relative_reduction_ci(
    projection: np.ndarray,
    baseline: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    projection = np.asarray(projection, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    if projection.shape != baseline.shape or projection.ndim != 1:
        raise ValueError("bootstrap arrays must be paired one-dimensional arrays")
    if projection.size < 2 or np.any(baseline <= 0.0):
        raise ValueError("bootstrap needs at least two positive baseline blocks")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        projection.size,
        size=(resamples, projection.size),
    )
    projection_means = np.mean(projection[indices], axis=1)
    baseline_means = np.mean(baseline[indices], axis=1)
    reductions = 1.0 - projection_means / baseline_means
    low, high = np.percentile(reductions, (2.5, 97.5))
    return float(low), float(high)


def summarize_accuracy(
    case_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
    methods: list[Method],
    *,
    bootstrap_resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, bool]:
    summaries: list[dict[str, Any]] = []
    for method in methods:
        errors = np.asarray(
            [float(row["nrmse"]) for row in case_rows if row["method"] == method.key]
        )
        components = [row for row in component_rows if row["method"] == method.key]
        summaries.append(
            {
                "method": method.key,
                "label": method.label,
                "semantics": method.semantics,
                "target_grid": method.target_grid,
                "predeclared": method.predeclared,
                "mean_nrmse": float(np.mean(errors)),
                "median_nrmse": float(np.median(errors)),
                "max_nrmse": float(np.max(errors)),
                "mean_passband_nrmse": statistics.fmean(
                    float(row["passband_nrmse"]) for row in components
                ),
                "mean_stopband_leakage": statistics.fmean(
                    float(row["stopband_leakage"]) for row in components
                ),
            }
        )

    block_ids = sorted({str(row["base_id"]) for row in case_rows})
    projection_rows = {
        (str(row["base_id"]), float(row["nuisance_ratio"])): float(row["nrmse"])
        for row in case_rows
        if row["method"] == "splineops_projection"
    }
    projection_blocks = np.asarray(
        [
            statistics.fmean(
                projection_rows[(base_id, ratio)] for ratio in NUISANCE_RATIOS
            )
            for base_id in block_ids
        ]
    )

    comparisons: list[dict[str, Any]] = []
    for method in methods:
        if method.key == "splineops_projection":
            continue
        baseline_rows = {
            (str(row["base_id"]), float(row["nuisance_ratio"])): float(row["nrmse"])
            for row in case_rows
            if row["method"] == method.key
        }
        baseline_blocks = np.asarray(
            [
                statistics.fmean(
                    baseline_rows[(base_id, ratio)] for ratio in NUISANCE_RATIOS
                )
                for base_id in block_ids
            ]
        )
        all_projection = np.asarray(
            [
                float(row["nrmse"])
                for row in case_rows
                if row["method"] == "splineops_projection"
            ]
        )
        all_baseline = np.asarray(
            [float(row["nrmse"]) for row in case_rows if row["method"] == method.key]
        )
        mean_reduction = float(1.0 - np.mean(all_projection) / np.mean(all_baseline))
        ci_low, ci_high = bootstrap_relative_reduction_ci(
            projection_blocks,
            baseline_blocks,
            resamples=bootstrap_resamples,
            seed=comparison_seed(method.key),
        )
        wins = [
            projection_rows[(str(row["base_id"]), float(row["nuisance_ratio"]))]
            < float(row["nrmse"])
            for row in case_rows
            if row["method"] == method.key
        ]
        stratum_reductions: dict[str, float] = {}
        for geometry in GEOMETRIES:
            for ratio in NUISANCE_RATIOS:
                projection_values = [
                    float(row["nrmse"])
                    for row in case_rows
                    if row["method"] == "splineops_projection"
                    and row["geometry"] == geometry.key
                    and float(row["nuisance_ratio"]) == ratio
                ]
                baseline_values = [
                    float(row["nrmse"])
                    for row in case_rows
                    if row["method"] == method.key
                    and row["geometry"] == geometry.key
                    and float(row["nuisance_ratio"]) == ratio
                ]
                key = f"{geometry.key}:ratio{ratio:.2f}"
                stratum_reductions[key] = float(
                    1.0
                    - statistics.fmean(projection_values)
                    / statistics.fmean(baseline_values)
                )
        win_fraction = float(sum(wins) / len(wins))
        no_worse_stratum = bool(min(stratum_reductions.values()) >= 0.0)
        demonstrated = bool(
            mean_reduction >= MEAN_REDUCTION_MARGIN
            and ci_low > CI_REDUCTION_MARGIN
            and win_fraction >= WIN_FRACTION_MARGIN
            and no_worse_stratum
        )
        comparisons.append(
            {
                "baseline": method.key,
                "baseline_label": method.label,
                "predeclared": method.predeclared,
                "mean_relative_nrmse_reduction": mean_reduction,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "individual_win_fraction": win_fraction,
                "minimum_stratum_relative_reduction": min(stratum_reductions.values()),
                "stratum_relative_reductions": stratum_reductions,
                "mean_reduction_margin": MEAN_REDUCTION_MARGIN,
                "ci_reduction_margin": CI_REDUCTION_MARGIN,
                "win_fraction_margin": WIN_FRACTION_MARGIN,
                "demonstrated": demonstrated,
            }
        )
    frozen_passed = all(
        bool(row["demonstrated"]) for row in comparisons if bool(row["predeclared"])
    )
    all_passed = all(bool(row["demonstrated"]) for row in comparisons)
    return summaries, comparisons, frozen_passed, all_passed


def run_timings(
    methods: list[Method],
    *,
    frames: int,
    warmups: int,
    repeats: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for geometry_index, geometry in enumerate(GEOMETRIES):
        rng = np.random.default_rng(9000 + geometry_index)
        inputs = [
            rng.standard_normal(geometry.source_shape, dtype=np.float32)
            for _ in range(frames)
        ]
        for method in methods:
            setup_values: list[float] = []
            prepared: ArrayCall | None = None
            for _ in range(5):
                gc.collect()
                started = time.perf_counter()
                prepared = method.prepare(geometry)
                setup_values.append(time.perf_counter() - started)
            assert prepared is not None
            for _ in range(warmups):
                for values in inputs:
                    prepared(values)
            elapsed: list[float] = []
            for _ in range(repeats):
                gc.collect()
                started = time.perf_counter()
                for values in inputs:
                    prepared(values)
                elapsed.append((time.perf_counter() - started) / len(inputs))
            rows.append(
                {
                    "geometry": geometry.key,
                    "source_shape": "x".join(map(str, geometry.source_shape)),
                    "output_shape": "x".join(map(str, geometry.output_shape)),
                    "method": method.key,
                    "setup_median_ms": 1000.0 * statistics.median(setup_values),
                    "apply_median_ms_per_volume": 1000.0 * statistics.median(elapsed),
                    "apply_min_ms_per_volume": 1000.0 * min(elapsed),
                    "frames": frames,
                    "warmups": warmups,
                    "repeats": repeats,
                }
            )
            print(
                f"timing {geometry.key:24s} {method.label:28s} "
                f"{rows[-1]['apply_median_ms_per_volume']:8.3f} ms",
                flush=True,
            )

    projection = {
        str(row["geometry"]): float(row["apply_median_ms_per_volume"])
        for row in rows
        if row["method"] == "splineops_projection"
    }
    for row in rows:
        row["speedup_vs_splineops_projection"] = float(
            float(row["apply_median_ms_per_volume"]) / projection[str(row["geometry"])]
        )
    return rows


def runtime_comparisons(
    timing_rows: list[dict[str, Any]], methods: list[Method]
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for method in methods:
        if method.key == "splineops_projection":
            continue
        rows = [row for row in timing_rows if row["method"] == method.key]
        speedups = [float(row["speedup_vs_splineops_projection"]) for row in rows]
        comparisons.append(
            {
                "baseline": method.key,
                "baseline_label": method.label,
                "geometric_mean_speedup": float(
                    np.exp(np.mean(np.log(np.asarray(speedups))))
                ),
                "minimum_geometry_speedup": min(speedups),
                "faster_in_all_geometries": bool(min(speedups) > 1.0),
            }
        )
    return comparisons


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_accuracy_plot(
    case_rows: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    methods: list[Method],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = (
        "#2463a5",
        "#7d8597",
        "#6a994e",
        "#9c6644",
        "#bc6c25",
        "#7b2cbf",
        "#168aad",
        "#d62828",
    )
    figure, axes = plt.subplots(1, 2, figsize=(14.2, 5.2))
    x = np.arange(len(NUISANCE_RATIOS))
    for method, color in zip(methods, colors):
        values = [
            statistics.fmean(
                float(row["nrmse"])
                for row in case_rows
                if row["method"] == method.key and float(row["nuisance_ratio"]) == ratio
            )
            for ratio in NUISANCE_RATIOS
        ]
        axes[0].plot(x, values, marker="o", label=method.label, color=color)
    axes[0].set_xticks(x, [f"{ratio:.2f}" for ratio in NUISANCE_RATIOS])
    axes[0].set_xlabel("Above-Nyquist nuisance coefficient ratio")
    axes[0].set_ylabel("Exact-target NRMSE (lower is better)")
    axes[0].set_title("72 controlled 3-D coarsening cases")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    frozen_comparisons = [row for row in comparisons if bool(row["predeclared"])]
    y = np.arange(len(frozen_comparisons))
    means = np.asarray(
        [float(row["mean_relative_nrmse_reduction"]) for row in frozen_comparisons]
    )
    low = np.asarray([float(row["bootstrap_ci_low"]) for row in frozen_comparisons])
    high = np.asarray([float(row["bootstrap_ci_high"]) for row in frozen_comparisons])
    axes[1].errorbar(
        100.0 * means,
        y,
        xerr=100.0 * np.vstack((means - low, high - means)),
        fmt="o",
        color="#2463a5",
        capsize=5,
    )
    axes[1].axvline(
        100.0 * CI_REDUCTION_MARGIN, color="#bc6c25", linestyle=":", label="CI margin"
    )
    axes[1].axvline(
        100.0 * MEAN_REDUCTION_MARGIN,
        color="#6a994e",
        linestyle="--",
        label="mean margin",
    )
    axes[1].set_yticks(y, [str(row["baseline_label"]) for row in frozen_comparisons])
    axes[1].set_xlabel("SplineOps relative NRMSE reduction (%)")
    axes[1].set_title("Blocked 95% bootstrap intervals")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "benchmarks" / "wavefield3d",
    )
    parser.add_argument("--docs-static-dir", type=Path)
    parser.add_argument(
        "--seeds-per-geometry", type=int, default=TEST_SEEDS_PER_GEOMETRY
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--timing-frames", type=int, default=TIMING_FRAMES)
    parser.add_argument("--timing-warmups", type=int, default=TIMING_WARMUPS)
    parser.add_argument("--timing-repeats", type=int, default=TIMING_REPEATS)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds_per_geometry < 1:
        parser.error("--seeds-per-geometry must be positive")
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    if args.timing_frames < 1 or args.timing_warmups < 0 or args.timing_repeats < 1:
        parser.error("timing counts must be positive, except warmups may be zero")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = study_methods()
    case_rows, component_rows = evaluate_accuracy(
        methods,
        seeds_per_geometry=args.seeds_per_geometry,
    )
    summaries, comparisons, frozen_superiority, all_method_superiority = (
        summarize_accuracy(
            case_rows,
            component_rows,
            methods,
            bootstrap_resamples=args.bootstrap_resamples,
        )
    )
    timing_rows = (
        []
        if args.skip_timing
        else run_timings(
            methods,
            frames=args.timing_frames,
            warmups=args.timing_warmups,
            repeats=args.timing_repeats,
        )
    )
    timing_comparisons = (
        runtime_comparisons(timing_rows, methods) if timing_rows else []
    )
    protocol_conforming = bool(
        args.seeds_per_geometry == TEST_SEEDS_PER_GEOMETRY
        and args.bootstrap_resamples == BOOTSTRAP_RESAMPLES
        and (
            args.skip_timing
            or (
                args.timing_frames == TIMING_FRAMES
                and args.timing_warmups == TIMING_WARMUPS
                and args.timing_repeats == TIMING_REPEATS
            )
        )
    )
    frozen_demonstrated = bool(protocol_conforming and frozen_superiority)
    field_wide_demonstrated = bool(protocol_conforming and all_method_superiority)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "path": "benchmarks/wavefield3d/PROTOCOL.md",
            "sha256": sha256_text(protocol_path()),
            "frozen_before_confirmation_run": True,
            "independently_preregistered": False,
            "excluded_before_confirmation": {
                "feasibility": [0, 1, 2, 3],
                "implementation_validation": {
                    "isotropic_2x": [1000, 1007],
                    "anisotropic_noninteger": [1100, 1107],
                    "mixed_reduction": [1200, 1207],
                },
            },
            "protocol_conforming_run": protocol_conforming,
            "posthoc_scientific_resampler_audit": True,
        },
        "configuration": {
            "geometries": [asdict(geometry) for geometry in GEOMETRIES],
            "seeds_per_geometry": args.seeds_per_geometry,
            "nuisance_ratios": NUISANCE_RATIOS,
            "low_mode_count": LOW_MODE_COUNT,
            "high_mode_count": HIGH_MODE_COUNT,
            "low_nyquist_fraction": LOW_NYQUIST_FRACTION,
            "high_nyquist_fraction": HIGH_NYQUIST_FRACTION,
            "source_nyquist_fraction": SOURCE_NYQUIST_FRACTION,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": args.bootstrap_resamples,
            "timing_frames": args.timing_frames,
            "timing_warmups": args.timing_warmups,
            "timing_repeats": args.timing_repeats,
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": cpu_model(),
            "packages": {
                name: package_version(name)
                for name in (
                    "splineops",
                    "numpy",
                    "scipy",
                    "scikit-image",
                    "torch",
                )
            },
        },
        "metric_contract": {
            "primary": "NRMSE against the analytically evaluated low-frequency component",
            "bootstrap_unit": "base field, with three nuisance-ratio variants kept together",
            "target_grids": {
                "endpoint": "physical points including 0 and 1",
                "half_pixel": "pipeline-specific interior half-pixel point locations",
            },
            "runtime": "local single-thread repeated-application median; not portable",
        },
        "accuracy_summary": summaries,
        "comparisons": comparisons,
        "runtime_comparisons": timing_comparisons,
        "case_scores": case_rows,
        "component_scores": component_rows,
        "timings": timing_rows,
        "conclusion": {
            "generic_resize_superiority_demonstrated": frozen_demonstrated,
            "scientific_resampling_superiority_demonstrated": field_wide_demonstrated,
            "claim": (
                "SplineOps cubic projection has lower exact-target NRMSE than every "
                "predeclared generic N-D resize alternative for the frozen controlled "
                "field class. A post-hoc SciPy polyphase FIR audit is more accurate, so "
                "field-wide scientific-resampling superiority is not demonstrated."
                if frozen_demonstrated and not field_wide_demonstrated
                else "No numerical-superiority claim survived the recorded comparisons."
            ),
            "scope_limit": (
                "Controlled continuous mirror-compatible 3-D cosine fields on the three "
                "tested endpoint/half-pixel geometries; not generic image or application "
                "superiority."
            ),
        },
    }

    results_path = args.output_dir / "results.json"
    summary_path = args.output_dir / "summary.csv"
    comparisons_path = args.output_dir / "comparisons.csv"
    cases_path = args.output_dir / "case_scores.csv"
    components_path = args.output_dir / "component_scores.csv"
    results_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    save_csv(summary_path, summaries)
    save_csv(comparisons_path, comparisons)
    save_csv(cases_path, case_rows)
    save_csv(components_path, component_rows)
    if timing_rows:
        save_csv(args.output_dir / "timings.csv", timing_rows)

    generated_plot: Path | None = None
    if not args.no_plots:
        generated_plot = args.output_dir / "wavefield3d_accuracy.png"
        make_accuracy_plot(case_rows, comparisons, methods, generated_plot)
        if args.docs_static_dir is not None:
            args.docs_static_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                generated_plot,
                args.docs_static_dir / "wavefield3d-study-accuracy.png",
            )

    print(f"Wrote {results_path}")
    print(f"Generic-resize superiority demonstrated: {frozen_demonstrated}")
    print(f"Scientific-resampling superiority demonstrated: {field_wide_demonstrated}")
    if generated_plot is not None:
        print(f"Wrote {generated_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
