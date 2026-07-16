#!/usr/bin/env python3
"""Reproduce the SplineOps 3-D microscopy downsampling case study.

The real-data metrics in this script are diagnostics, not biological ground
truth.  A separate mirror-compatible cosine field supplies a known numerical
target for a small calibration check.

The public ``cells3d.tif`` input is pinned by URL and SHA-256.  It is the CC0
sample distributed by scikit-image and attributed there to the Allen Institute
for Cell Science.
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
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy import ndimage

DATASET_URL = (
    "https://gitlab.com/scikit-image/data/-/raw/"
    "2cdc5ce89b334d28f06a58c9f0ca21aa6992a5ba/cells3d.tif"
)
DATASET_SHA256 = "afc7c7d80d38bfde09788b4064ac1e64ec14e88454ab785ebdc8dbba5ca3b222"
DATASET_SHAPE = (60, 2, 256, 256)
INPUT_SPACING_UM = (0.29, 0.26, 0.26)
OUTPUT_SPATIAL_SHAPE = (30, 128, 128)
SPATIAL_AXES = (0, 2, 3)
CHANNEL_NAMES = ("membrane", "nuclei")

VolumeMethod = Callable[[np.ndarray, tuple[int, int, int]], np.ndarray]


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    semantics: str
    call: VolumeMethod


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_dataset(cache_dir: Path) -> Path:
    """Return the verified public TIFF, downloading it atomically if needed."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / "cells3d.tif"
    if destination.exists():
        actual = sha256_file(destination)
        if actual == DATASET_SHA256:
            return destination
        raise RuntimeError(
            f"cached dataset checksum mismatch: {destination} has SHA-256 {actual}"
        )

    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "SplineOps cells3d reproducibility study"},
    )
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=cache_dir, prefix="cells3d-", suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                shutil.copyfileobj(response, handle)
        actual = sha256_file(temporary)
        if actual != DATASET_SHA256:
            raise RuntimeError(
                f"downloaded dataset checksum mismatch: expected {DATASET_SHA256}, "
                f"got {actual}"
            )
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_dataset(path: Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError(
            "This study needs its optional dependencies. Install them with "
            "`python -m pip install -e '.[study]'`."
        ) from exc

    volume = np.asarray(tifffile.imread(path))
    if volume.shape != DATASET_SHAPE:
        raise ValueError(
            f"expected the pinned dataset shape {DATASET_SHAPE}, got {volume.shape}"
        )
    return volume


def normalize_channels(volume: np.ndarray) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Percentile-normalize each channel for comparable diagnostics and plots."""

    normalized = np.empty(volume.shape, dtype=np.float32)
    parameters: list[dict[str, float]] = []
    for channel, name in enumerate(CHANNEL_NAMES):
        source = np.asarray(volume[:, channel], dtype=np.float64)
        low, high = np.percentile(source, (0.5, 99.5))
        if high <= low:
            raise ValueError(f"channel {name!r} has no usable intensity range")
        normalized[:, channel] = np.clip((source - low) / (high - low), 0.0, 1.0)
        parameters.append(
            {
                "channel": name,
                "percentile_low": float(low),
                "percentile_high": float(high),
            }
        )
    return normalized, parameters


def endpoint_coordinates(
    input_shape: Sequence[int], output_shape: Sequence[int]
) -> tuple[np.ndarray, ...]:
    if len(input_shape) != len(output_shape):
        raise ValueError("input and output dimensionality must match")
    return tuple(
        (
            np.linspace(0.0, float(source - 1), target, dtype=np.float64)
            if target > 1
            else np.zeros(1, dtype=np.float64)
        )
        for source, target in zip(input_shape, output_shape)
    )


def scipy_endpoint_resize(
    channel: np.ndarray, output_shape: tuple[int, ...]
) -> np.ndarray:
    coordinates = endpoint_coordinates(channel.shape, output_shape)
    grid = np.meshgrid(*coordinates, indexing="ij", sparse=False)
    return np.asarray(
        ndimage.map_coordinates(
            channel,
            grid,
            order=3,
            mode="reflect",
            prefilter=True,
        ),
        dtype=np.float32,
    )


def resize_splineops(
    volume: np.ndarray, output_shape: tuple[int, int, int], *, antialias: bool
) -> np.ndarray:
    from splineops import resize

    method = "cubic-antialiasing" if antialias else "cubic"
    return np.asarray(
        resize(volume, output_size=output_shape, axes=SPATIAL_AXES, method=method),
        dtype=np.float32,
    )


def resize_scipy_gaussian(
    volume: np.ndarray, output_shape: tuple[int, int, int]
) -> np.ndarray:
    """Gaussian prefilter followed by endpoint-aligned cubic sampling."""

    input_shape = tuple(volume.shape[axis] for axis in SPATIAL_AXES)
    factors = tuple(
        (source - 1) / (target - 1) if target > 1 else float(source)
        for source, target in zip(input_shape, output_shape)
    )
    sigma = tuple(max(0.0, (factor - 1.0) / 2.0) for factor in factors)
    output = np.empty((output_shape[0], volume.shape[1], *output_shape[1:]), np.float32)
    for channel in range(volume.shape[1]):
        filtered = ndimage.gaussian_filter(
            volume[:, channel], sigma=sigma, mode="reflect"
        )
        output[:, channel] = scipy_endpoint_resize(filtered, output_shape)
    return output


def resize_skimage(
    volume: np.ndarray, output_shape: tuple[int, int, int]
) -> np.ndarray:
    try:
        from skimage.transform import resize
    except ImportError as exc:
        raise RuntimeError(
            "This study needs its optional dependencies. Install them with "
            "`python -m pip install -e '.[study]'`."
        ) from exc

    output = np.empty((output_shape[0], volume.shape[1], *output_shape[1:]), np.float32)
    for channel in range(volume.shape[1]):
        output[:, channel] = resize(
            volume[:, channel],
            output_shape,
            order=3,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
            clip=False,
        )
    return output


def study_methods() -> list[Method]:
    return [
        Method(
            "splineops_projection",
            "SplineOps projection AA",
            "endpoint-aligned cubic spline projection antialiasing",
            lambda volume, shape: resize_splineops(volume, shape, antialias=True),
        ),
        Method(
            "splineops_interpolation",
            "SplineOps cubic, no AA",
            "endpoint-aligned cubic interpolation without antialiasing",
            lambda volume, shape: resize_splineops(volume, shape, antialias=False),
        ),
        Method(
            "scipy_gaussian",
            "SciPy Gaussian + cubic",
            "Gaussian prefilter and endpoint-aligned cubic sampling",
            resize_scipy_gaussian,
        ),
        Method(
            "skimage_resize",
            "scikit-image cubic AA",
            "scikit-image resize grid, reflect boundary, cubic antialiasing",
            resize_skimage,
        ),
    ]


def timed_call(
    call: Callable[[], np.ndarray], *, warmups: int, repeats: int
) -> tuple[np.ndarray, list[float]]:
    for _ in range(warmups):
        call()
    times: list[float] = []
    output: np.ndarray | None = None
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        output = call()
        times.append(time.perf_counter() - start)
    if output is None:
        raise ValueError("repeats must be at least one")
    return output, times


def nrmse(actual: np.ndarray, target: np.ndarray) -> float:
    error = np.asarray(actual, np.float64) - np.asarray(target, np.float64)
    scale = float(np.sqrt(np.mean(np.asarray(target, np.float64) ** 2)))
    if scale == 0.0:
        raise ValueError("target RMS is zero")
    return float(np.sqrt(np.mean(error**2)) / scale)


def psnr(
    actual: np.ndarray, target: np.ndarray, *, data_range: float = 1.0
) -> float | None:
    mse = float(
        np.mean((np.asarray(actual, np.float64) - np.asarray(target, np.float64)) ** 2)
    )
    if mse == 0.0:
        return None
    return float(20.0 * math.log10(data_range / math.sqrt(mse)))


def mean_ssim(actual: np.ndarray, target: np.ndarray) -> float:
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        raise RuntimeError(
            "This study needs its optional dependencies. Install them with "
            "`python -m pip install -e '.[study]'`."
        ) from exc

    values = [
        structural_similarity(target[:, channel], actual[:, channel], data_range=1.0)
        for channel in range(target.shape[1])
    ]
    return float(statistics.fmean(values))


def high_band_fraction(volume: np.ndarray, *, threshold: float = 0.70) -> float:
    """Return average energy fraction above a normalized radial frequency."""

    fractions: list[float] = []
    spatial_shape = (volume.shape[0], volume.shape[2], volume.shape[3])
    frequencies = [np.fft.fftfreq(size) / 0.5 for size in spatial_shape]
    radius_grid = np.meshgrid(*frequencies, indexing="ij", sparse=True)
    radius_squared = sum(component**2 for component in radius_grid)
    mask = radius_squared >= threshold**2
    for channel in range(volume.shape[1]):
        centered = np.asarray(volume[:, channel], np.float64)
        centered = centered - float(np.mean(centered))
        power = np.abs(np.fft.fftn(centered)) ** 2
        total = float(np.sum(power))
        fractions.append(float(np.sum(power[mask]) / total) if total > 0.0 else 0.0)
    return float(statistics.fmean(fractions))


def roundtrip_psnr(coarse: np.ndarray, original: np.ndarray) -> float | None:
    reconstructed = np.empty_like(original, dtype=np.float32)
    output_shape = (original.shape[0], original.shape[2], original.shape[3])
    for channel in range(original.shape[1]):
        reconstructed[:, channel] = scipy_endpoint_resize(
            coarse[:, channel], output_shape
        )
    return psnr(reconstructed, original)


def analytic_calibration(
    input_shape: tuple[int, int, int] = (48, 64, 64),
    output_shape: tuple[int, int, int] = (24, 32, 32),
) -> tuple[np.ndarray, np.ndarray]:
    """Create a sampled cosine field and its exact low-band coarse target."""

    def coordinates(shape: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
        vectors = [np.linspace(0.0, 1.0, size) for size in shape]
        return tuple(np.meshgrid(*vectors, indexing="ij", sparse=True))

    def low_field(shape: tuple[int, int, int]) -> np.ndarray:
        z, y, x = coordinates(shape)
        return (
            0.50
            + 0.13 * np.cos(4 * np.pi * z) * np.cos(5 * np.pi * y)
            + 0.11 * np.cos(7 * np.pi * x)
            + 0.07 * np.cos(3 * np.pi * z) * np.cos(6 * np.pi * x)
        )

    z, y, x = coordinates(input_shape)
    # These DCT-like modes are above the corresponding output-grid Nyquist
    # while remaining compatible with a mirrored boundary extension.
    high = (
        0.09 * np.cos(27 * np.pi * z) * np.cos(2 * np.pi * y)
        + 0.08 * np.cos(36 * np.pi * y) * np.cos(3 * np.pi * x)
        + 0.07 * np.cos(35 * np.pi * x) * np.cos(2 * np.pi * z)
    )
    sampled = (low_field(input_shape) + high).astype(np.float32)
    target = low_field(output_shape).astype(np.float32)
    return sampled[:, np.newaxis, :, :], target[:, np.newaxis, :, :]


def output_spacing_um(
    input_shape: tuple[int, int, int], output_shape: tuple[int, int, int]
) -> tuple[float, float, float]:
    return tuple(
        spacing * (source - 1) / (target - 1)
        for spacing, source, target in zip(INPUT_SPACING_UM, input_shape, output_shape)
    )


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


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "method",
        "label",
        "semantics",
        "runtime_median_s",
        "runtime_min_s",
        "gaussian_reference_nrmse",
        "gaussian_reference_ssim",
        "roundtrip_psnr_db",
        "high_band_energy_fraction",
        "synthetic_nrmse",
        "synthetic_psnr_db",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_montage(
    original: np.ndarray,
    outputs: dict[str, np.ndarray],
    methods: list[Method],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    columns = 1 + len(methods)
    figure, axes = plt.subplots(2, columns, figsize=(3.05 * columns, 5.8))
    for channel, channel_name in enumerate(CHANNEL_NAMES):
        images = [original[:, channel], *[outputs[m.key][:, channel] for m in methods]]
        labels = ["Original", *[method.label for method in methods]]
        for column, (image, label) in enumerate(zip(images, labels)):
            axis = axes[channel, column]
            axis.imshow(image[image.shape[0] // 2], cmap="gray", vmin=0.0, vmax=1.0)
            axis.set_title(label, fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(channel_name.capitalize())
    figure.suptitle("cells3d: central slices after 2x spatial downsampling")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def make_metrics_plot(rows: list[dict[str, object]], destination: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["label"]) for row in rows]
    runtime = [float(row["runtime_median_s"]) for row in rows]
    roundtrip = [float(row["roundtrip_psnr_db"]) for row in rows]
    calibration = [float(row["synthetic_nrmse"]) for row in rows]
    x = np.arange(len(rows))

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    panels = (
        (runtime, "Median runtime (s)", False),
        (roundtrip, "Round-trip PSNR (dB)\nproxy", False),
        (calibration, "Known-target NRMSE\nlower is better", True),
    )
    for axis, (values, title, log_scale) in zip(axes, panels):
        axis.bar(x, values, color=("#2a6fbb", "#7d8597", "#6a994e", "#bc6c25"))
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
        if log_scale:
            axis.set_yscale("log")
    figure.suptitle("cells3d study: speed and quality answer different questions")
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="local cells3d.tif override")
    parser.add_argument(
        "--cache-dir", type=Path, default=default_cache / "splineops" / "cells3d"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "benchmarks" / "cells3d"
    )
    parser.add_argument(
        "--docs-static-dir",
        type=Path,
        help="also copy generated plots here for the documentation",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="skip download and real-data metrics (useful for local checks)",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repeats < 1:
        parser.error("--warmups must be non-negative and --repeats must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = study_methods()

    synthetic_input, synthetic_target = analytic_calibration()
    synthetic_shape = (
        synthetic_target.shape[0],
        synthetic_target.shape[2],
        synthetic_target.shape[3],
    )
    synthetic_results: dict[str, dict[str, float | None]] = {}
    for method in methods:
        output = method.call(synthetic_input, synthetic_shape)
        synthetic_results[method.key] = {
            "nrmse": nrmse(output, synthetic_target),
            "psnr_db": psnr(output, synthetic_target),
        }

    source_path: Path | None = None
    normalization: list[dict[str, float]] = []
    outputs: dict[str, np.ndarray] = {}
    timings: dict[str, list[float]] = {}
    rows: list[dict[str, object]] = []
    original: np.ndarray | None = None

    if not args.synthetic_only:
        source_path = (
            args.input if args.input is not None else fetch_dataset(args.cache_dir)
        )
        if sha256_file(source_path) != DATASET_SHA256:
            raise RuntimeError(
                f"input does not match the pinned SHA-256: {source_path}"
            )
        raw = load_dataset(source_path)
        original, normalization = normalize_channels(raw)

        for method in methods:
            print(f"Running {method.label} ...", flush=True)
            output, elapsed = timed_call(
                lambda method=method: method.call(original, OUTPUT_SPATIAL_SHAPE),
                warmups=args.warmups,
                repeats=args.repeats,
            )
            expected_shape = (
                OUTPUT_SPATIAL_SHAPE[0],
                original.shape[1],
                *OUTPUT_SPATIAL_SHAPE[1:],
            )
            if output.shape != expected_shape:
                raise RuntimeError(
                    f"{method.label} returned {output.shape}, expected {expected_shape}"
                )
            outputs[method.key] = output
            timings[method.key] = elapsed

        gaussian_reference = outputs["scipy_gaussian"]
        for method in methods:
            output = outputs[method.key]
            rows.append(
                {
                    "method": method.key,
                    "label": method.label,
                    "semantics": method.semantics,
                    "runtime_median_s": statistics.median(timings[method.key]),
                    "runtime_min_s": min(timings[method.key]),
                    "gaussian_reference_nrmse": nrmse(output, gaussian_reference),
                    "gaussian_reference_ssim": mean_ssim(output, gaussian_reference),
                    "roundtrip_psnr_db": roundtrip_psnr(output, original),
                    "high_band_energy_fraction": high_band_fraction(output),
                    "synthetic_nrmse": synthetic_results[method.key]["nrmse"],
                    "synthetic_psnr_db": synthetic_results[method.key]["psnr_db"],
                }
            )
    else:
        for method in methods:
            rows.append(
                {
                    "method": method.key,
                    "label": method.label,
                    "semantics": method.semantics,
                    "runtime_median_s": None,
                    "runtime_min_s": None,
                    "gaussian_reference_nrmse": None,
                    "gaussian_reference_ssim": None,
                    "roundtrip_psnr_db": None,
                    "high_band_energy_fraction": None,
                    "synthetic_nrmse": synthetic_results[method.key]["nrmse"],
                    "synthetic_psnr_db": synthetic_results[method.key]["psnr_db"],
                }
            )

    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "name": "cells3d.tif",
            "source": "scikit-image data repository",
            "origin": "Allen Institute for Cell Science",
            "license": "CC0-1.0",
            "url": DATASET_URL,
            "sha256": DATASET_SHA256,
            "source_shape_zcyx": DATASET_SHAPE,
            "channel_names": CHANNEL_NAMES,
            "input_spacing_um_zyx": INPUT_SPACING_UM,
            "output_spacing_um_zyx": output_spacing_um(
                (DATASET_SHAPE[0], DATASET_SHAPE[2], DATASET_SHAPE[3]),
                OUTPUT_SPATIAL_SHAPE,
            ),
            "normalization_for_measurement": normalization,
        },
        "configuration": {
            "spatial_axes": SPATIAL_AXES,
            "output_spatial_shape": OUTPUT_SPATIAL_SHAPE,
            "warmups": args.warmups,
            "repeats": args.repeats,
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
                for name in ("splineops", "numpy", "scipy", "scikit-image", "tifffile")
            },
        },
        "metric_limits": {
            "gaussian_reference": (
                "agreement with one chosen practical reference; not ground truth"
            ),
            "roundtrip_psnr": "information-retention proxy; can reward aliased detail",
            "high_band_energy_fraction": (
                "spectral diagnostic; lower can mean less aliasing or more blur"
            ),
            "synthetic_metrics": (
                "known target on a constructed mirror-compatible cosine field"
            ),
            "runtime": "local end-to-end median; not a universal speed ranking",
        },
        "results": rows,
    }

    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    save_csv(csv_path, rows)

    generated_plots: list[Path] = []
    if not args.no_plots and original is not None:
        montage = args.output_dir / "cells3d_montage.png"
        metrics = args.output_dir / "cells3d_metrics.png"
        make_montage(original, outputs, methods, montage)
        make_metrics_plot(rows, metrics)
        generated_plots.extend((montage, metrics))
        if args.docs_static_dir is not None:
            args.docs_static_dir.mkdir(parents=True, exist_ok=True)
            for source in generated_plots:
                docs_name = f"cells3d-study-{source.name.removeprefix('cells3d_')}"
                shutil.copy2(
                    source,
                    args.docs_static_dir / docs_name,
                )

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    for path in generated_plots:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
