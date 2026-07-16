#!/usr/bin/env python3
"""Run the frozen BBBC050 downsampling-to-segmentation study.

The protocol is recorded in ``benchmarks/bbbc050/PROTOCOL.md``.  This script
downloads two checksum-pinned CC BY 3.0 archives, keeps embryo identities
separate during threshold fitting, and publishes a negative result as readily
as a positive one.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import os
import platform
import re
import shutil
import statistics
import sys
import tempfile
import time
import urllib.request
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence
from zipfile import ZipFile

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy import ndimage

DATASET_PAGE = "https://bbbc.broadinstitute.org/BBBC050"
IMAGES_URL = "https://data.broadinstitute.org/bbbc/BBBC050/Images.zip"
GROUND_TRUTH_URL = "https://data.broadinstitute.org/bbbc/BBBC050/GroundTruth.zip"
IMAGES_SHA256 = "29f100abbfebfb1986b8e87eac091e86d8ec27cd8194f9a1c02c805e76b6dcd8"
GROUND_TRUTH_SHA256 = "1f19b308730dccf217c4d4dcf5745ad0fcde4eeb9f9c9306b2c8abd1fe73e5d1"
EXPECTED_IMAGE_COUNTS = {"train": 121, "test": 44}
EXPECTED_EMBRYO_COUNTS = {"train": 11, "test": 4}
PILOT_TIMEPOINT = "251"
THRESHOLDS = np.round(np.arange(0.040, 0.300 + 0.0001, 0.005), 3)
BOOTSTRAP_SEED = 20260716
BOOTSTRAP_RESAMPLES = 20_000
PRACTICAL_MARGIN = 0.005
SPATIAL_AXES = (1, 2)

VolumeMethod = Callable[[np.ndarray, tuple[int, int, int]], np.ndarray]


@dataclass(frozen=True)
class Archive:
    filename: str
    url: str
    sha256: str


@dataclass(frozen=True)
class Sample:
    split: str
    embryo: str
    timepoint: str
    image_member: str
    ground_truth_member: str

    @property
    def key(self) -> str:
        return f"{self.split}:{self.embryo}:t{self.timepoint}"


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    semantics: str
    ground_truth_grid: str
    call: VolumeMethod


@dataclass
class SampleCurve:
    sample: Sample
    method: str
    source_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    runtime_s: float
    dice: np.ndarray


ARCHIVES = (
    Archive("Images.zip", IMAGES_URL, IMAGES_SHA256),
    Archive("GroundTruth.zip", GROUND_TRUTH_URL, GROUND_TRUTH_SHA256),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_archive(archive: Archive, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / archive.filename
    if destination.exists():
        actual = sha256_file(destination)
        if actual == archive.sha256:
            return destination
        raise RuntimeError(
            f"cached archive checksum mismatch: {destination} has {actual}"
        )

    request = urllib.request.Request(
        archive.url,
        headers={"User-Agent": "SplineOps BBBC050 reproducibility study"},
    )
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(
                dir=cache_dir,
                prefix=f"{archive.filename}-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                shutil.copyfileobj(response, handle)
        actual = sha256_file(temporary)
        if actual != archive.sha256:
            raise RuntimeError(
                f"downloaded {archive.filename} checksum mismatch: "
                f"expected {archive.sha256}, got {actual}"
            )
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def verified_override(path: Path, archive: Archive) -> Path:
    actual = sha256_file(path)
    if actual != archive.sha256:
        raise RuntimeError(
            f"{path} does not match the pinned {archive.filename} SHA-256: {actual}"
        )
    return path


def discover_samples(images: ZipFile, ground_truth: ZipFile) -> list[Sample]:
    pattern = re.compile(
        r"^Images/(?P<split>train|test)/Images/"
        r"(?P<embryo>Emb\d+)_t(?P<timepoint>\d+)\.tif$"
    )
    ground_truth_names = set(ground_truth.namelist())
    samples: list[Sample] = []
    for member in images.namelist():
        match = pattern.match(member)
        if match is None:
            continue
        split = match.group("split")
        filename = PurePosixPath(member).name
        target = f"GroundTruth/{split}/GroundTruth_QCANet/{filename}"
        if target not in ground_truth_names:
            raise RuntimeError(f"missing QCANet ground truth for {member}")
        samples.append(
            Sample(
                split=split,
                embryo=match.group("embryo"),
                timepoint=match.group("timepoint"),
                image_member=member,
                ground_truth_member=target,
            )
        )

    samples.sort(key=lambda item: (item.split, item.embryo, item.timepoint))
    counts = {
        split: sum(sample.split == split for sample in samples)
        for split in EXPECTED_IMAGE_COUNTS
    }
    embryos = {
        split: len({sample.embryo for sample in samples if sample.split == split})
        for split in EXPECTED_EMBRYO_COUNTS
    }
    if counts != EXPECTED_IMAGE_COUNTS:
        raise RuntimeError(f"unexpected image counts: {counts}")
    if embryos != EXPECTED_EMBRYO_COUNTS:
        raise RuntimeError(f"unexpected embryo counts: {embryos}")
    return samples


def read_tiff(archive: ZipFile, member: str) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError(
            "Install the study dependencies with "
            "`python -m pip install -e '.[study]'`."
        ) from exc
    return np.asarray(tifffile.imread(BytesIO(archive.read(member))))


def normalize_image(image: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    source = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(source, (1.0, 99.9))
    if high <= low:
        raise ValueError("image has no usable percentile intensity range")
    normalized = np.clip((source - low) / (high - low), 0.0, 1.0)
    return normalized.astype(np.float32), (float(low), float(high))


def half_xy_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    if len(shape) != 3:
        raise ValueError("BBBC050 images must be three-dimensional")
    return int(shape[0]), (int(shape[1]) + 1) // 2, (int(shape[2]) + 1) // 2


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
    image: np.ndarray, output_shape: tuple[int, int, int]
) -> np.ndarray:
    coordinates = endpoint_coordinates(image.shape, output_shape)
    grid = np.meshgrid(*coordinates, indexing="ij", sparse=False)
    return np.asarray(
        ndimage.map_coordinates(
            image,
            grid,
            order=3,
            mode="reflect",
            prefilter=True,
        ),
        dtype=np.float32,
    )


def resize_splineops(
    image: np.ndarray, output_shape: tuple[int, int, int], *, antialias: bool
) -> np.ndarray:
    from splineops import resize

    method = "cubic-antialiasing" if antialias else "cubic"
    return np.asarray(
        resize(
            image,
            output_size=output_shape[1:],
            axes=SPATIAL_AXES,
            method=method,
        ),
        dtype=np.float32,
    )


def resize_scipy_gaussian(
    image: np.ndarray, output_shape: tuple[int, int, int]
) -> np.ndarray:
    factors = tuple(
        (source - 1) / (target - 1) if target > 1 else float(source)
        for source, target in zip(image.shape, output_shape)
    )
    sigma = (0.0,) + tuple(max(0.0, (factor - 1.0) / 2.0) for factor in factors[1:])
    filtered = ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")
    return scipy_endpoint_resize(filtered, output_shape)


def resize_skimage(image: np.ndarray, output_shape: tuple[int, int, int]) -> np.ndarray:
    try:
        from skimage.transform import resize
    except ImportError as exc:
        raise RuntimeError(
            "Install the study dependencies with "
            "`python -m pip install -e '.[study]'`."
        ) from exc
    return np.asarray(
        resize(
            image,
            output_shape,
            order=3,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
            clip=False,
        ),
        dtype=np.float32,
    )


def study_methods() -> list[Method]:
    return [
        Method(
            "splineops_projection",
            "SplineOps projection AA",
            "endpoint-aligned cubic spline projection antialiasing",
            "endpoint",
            lambda image, shape: resize_splineops(image, shape, antialias=True),
        ),
        Method(
            "splineops_interpolation",
            "SplineOps cubic, no AA",
            "endpoint-aligned cubic interpolation without antialiasing",
            "endpoint",
            lambda image, shape: resize_splineops(image, shape, antialias=False),
        ),
        Method(
            "scipy_gaussian",
            "SciPy Gaussian + cubic",
            "Gaussian prefilter and endpoint-aligned cubic sampling",
            "endpoint",
            resize_scipy_gaussian,
        ),
        Method(
            "skimage_resize",
            "scikit-image cubic AA",
            "scikit-image resize grid, reflect boundary, cubic antialiasing",
            "skimage",
            resize_skimage,
        ),
    ]


def resize_ground_truth(
    labels: np.ndarray,
    output_shape: tuple[int, int, int],
    *,
    grid: str,
) -> np.ndarray:
    foreground = np.asarray(labels) > 0
    if grid == "endpoint":
        indices = tuple(
            np.rint(axis).astype(np.intp)
            for axis in endpoint_coordinates(foreground.shape, output_shape)
        )
        return foreground[np.ix_(*indices)]
    if grid == "skimage":
        try:
            from skimage.transform import resize
        except ImportError as exc:
            raise RuntimeError(
                "Install the study dependencies with "
                "`python -m pip install -e '.[study]'`."
            ) from exc
        return (
            resize(
                foreground.astype(np.uint8),
                output_shape,
                order=0,
                mode="reflect",
                anti_aliasing=False,
                preserve_range=True,
            )
            > 0
        )
    raise ValueError(f"unknown ground-truth grid {grid!r}")


def dice_curve(
    values: np.ndarray, target: np.ndarray, thresholds: np.ndarray = THRESHOLDS
) -> np.ndarray:
    """Compute every threshold Dice score with one histogram pass."""

    scores = np.asarray(values, dtype=np.float64).ravel()
    truth = np.asarray(target, dtype=bool).ravel()
    if scores.shape != truth.shape:
        raise ValueError("values and target must have the same shape")
    bins = np.searchsorted(thresholds, scores, side="right")
    count_all = np.bincount(bins, minlength=len(thresholds) + 1)
    count_true = np.bincount(
        bins, weights=truth.astype(np.int64), minlength=len(thresholds) + 1
    )
    predicted = np.cumsum(count_all[::-1])[::-1][1:]
    true_positive = np.cumsum(count_true[::-1])[::-1][1:]
    denominator = predicted + int(np.count_nonzero(truth))
    return np.divide(
        2.0 * true_positive,
        denominator,
        out=np.ones_like(true_positive, dtype=np.float64),
        where=denominator > 0,
    )


def process_samples(
    samples: list[Sample],
    images: ZipFile,
    ground_truth: ZipFile,
    methods: list[Method],
) -> list[SampleCurve]:
    records: list[SampleCurve] = []
    for index, sample in enumerate(samples, start=1):
        image_raw = read_tiff(images, sample.image_member)
        labels = read_tiff(ground_truth, sample.ground_truth_member)
        if image_raw.shape != labels.shape:
            raise RuntimeError(f"shape mismatch for {sample.key}")
        image, _ = normalize_image(image_raw)
        output_shape = half_xy_shape(image.shape)
        targets = {
            grid: resize_ground_truth(labels, output_shape, grid=grid)
            for grid in {method.ground_truth_grid for method in methods}
        }
        print(f"[{index:3d}/{len(samples)}] {sample.key}", flush=True)
        for method in methods:
            start = time.perf_counter()
            output = method.call(image, output_shape)
            elapsed = time.perf_counter() - start
            if output.shape != output_shape:
                raise RuntimeError(
                    f"{method.label} returned {output.shape}, expected {output_shape}"
                )
            records.append(
                SampleCurve(
                    sample=sample,
                    method=method.key,
                    source_shape=tuple(int(value) for value in image.shape),
                    output_shape=output_shape,
                    runtime_s=elapsed,
                    dice=dice_curve(output, targets[method.ground_truth_grid]),
                )
            )
    return records


def embryo_curves(
    records: list[SampleCurve],
    method: str,
    split: str,
    *,
    exclude_timepoint: str | None,
) -> dict[str, np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = {}
    for record in records:
        if record.method != method or record.sample.split != split:
            continue
        if (
            exclude_timepoint is not None
            and record.sample.timepoint == exclude_timepoint
        ):
            continue
        grouped.setdefault(record.sample.embryo, []).append(record.dice)
    return {
        embryo: np.mean(np.stack(curves), axis=0)
        for embryo, curves in sorted(grouped.items())
    }


def best_threshold_index(curves: Sequence[np.ndarray]) -> int:
    if not curves:
        raise ValueError("at least one threshold curve is required")
    mean_curve = np.mean(np.stack(curves), axis=0)
    return int(np.argmax(mean_curve))


def validate_method(
    records: list[SampleCurve], method: Method
) -> tuple[dict[str, float], dict[str, float], dict[str, float], float]:
    train = embryo_curves(
        records,
        method.key,
        "train",
        exclude_timepoint=PILOT_TIMEPOINT,
    )
    test = embryo_curves(
        records,
        method.key,
        "test",
        exclude_timepoint=None,
    )
    if len(train) != EXPECTED_EMBRYO_COUNTS["train"]:
        raise RuntimeError(f"incomplete training embryos for {method.key}")
    if len(test) != EXPECTED_EMBRYO_COUNTS["test"]:
        raise RuntimeError(f"incomplete test embryos for {method.key}")

    cross_validation: dict[str, float] = {}
    fold_thresholds: dict[str, float] = {}
    for held_out in train:
        threshold_index = best_threshold_index(
            [curve for embryo, curve in train.items() if embryo != held_out]
        )
        cross_validation[held_out] = float(train[held_out][threshold_index])
        fold_thresholds[held_out] = float(THRESHOLDS[threshold_index])

    global_index = best_threshold_index(list(train.values()))
    global_threshold = float(THRESHOLDS[global_index])
    external = {embryo: float(curve[global_index]) for embryo, curve in test.items()}
    return cross_validation, fold_thresholds, external, global_threshold


def bootstrap_mean_ci(
    differences: np.ndarray,
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap input must contain at least two paired values")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    means = np.mean(values[indices], axis=1)
    low, high = np.percentile(means, (2.5, 97.5))
    return float(low), float(high)


def sign_flip_pvalue(differences: np.ndarray) -> float:
    """Exact one-sided paired sign-flip test for a positive mean difference."""

    values = np.asarray(differences, dtype=np.float64)
    observed = float(np.mean(values))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=values.size)))
    null_means = np.mean(signs * values[np.newaxis, :], axis=1)
    return float(np.mean(null_means >= observed - 1e-15))


def comparison_seed(baseline: str) -> int:
    return BOOTSTRAP_SEED + (zlib.crc32(baseline.encode("utf-8")) & 0xFFFF)


def build_validation_results(
    records: list[SampleCurve],
    methods: list[Method],
    *,
    bootstrap_resamples: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, dict[str, object]],
]:
    validation: dict[str, dict[str, object]] = {}
    summaries: list[dict[str, object]] = []
    embryo_rows: list[dict[str, object]] = []
    for method in methods:
        cv, folds, external, global_threshold = validate_method(records, method)
        runtimes = [
            record.runtime_s for record in records if record.method == method.key
        ]
        validation[method.key] = {
            "cv": cv,
            "fold_thresholds": folds,
            "external": external,
            "global_threshold": global_threshold,
        }
        summaries.append(
            {
                "method": method.key,
                "label": method.label,
                "cv_mean_dice": statistics.fmean(cv.values()),
                "cv_std_dice": statistics.stdev(cv.values()),
                "external_mean_dice": statistics.fmean(external.values()),
                "external_std_dice": statistics.stdev(external.values()),
                "global_threshold": global_threshold,
                "runtime_median_ms": 1000.0 * statistics.median(runtimes),
            }
        )
        for embryo, score in cv.items():
            threshold = folds[embryo]
            embryo_rows.append(
                {
                    "split": "cross_validation",
                    "embryo": embryo,
                    "method": method.key,
                    "dice": score,
                    "threshold": threshold,
                }
            )
        for embryo, score in external.items():
            embryo_rows.append(
                {
                    "split": "external_test",
                    "embryo": embryo,
                    "method": method.key,
                    "dice": score,
                    "threshold": global_threshold,
                }
            )

    projection = validation["splineops_projection"]
    comparisons: list[dict[str, object]] = []
    for method in methods:
        if method.key == "splineops_projection":
            continue
        baseline = validation[method.key]
        cv_embryos = sorted(projection["cv"])
        cv_delta = np.asarray(
            [projection["cv"][embryo] - baseline["cv"][embryo] for embryo in cv_embryos]
        )
        test_embryos = sorted(projection["external"])
        test_delta = np.asarray(
            [
                projection["external"][embryo] - baseline["external"][embryo]
                for embryo in test_embryos
            ]
        )
        ci_low, ci_high = bootstrap_mean_ci(
            cv_delta,
            resamples=bootstrap_resamples,
            seed=comparison_seed(method.key),
        )
        mean_delta = float(np.mean(cv_delta))
        external_mean_delta = float(np.mean(test_delta))
        demonstrated = bool(
            mean_delta >= PRACTICAL_MARGIN
            and ci_low > 0.0
            and external_mean_delta >= 0.0
        )
        comparisons.append(
            {
                "baseline": method.key,
                "baseline_label": method.label,
                "cv_mean_dice_delta": mean_delta,
                "cv_bootstrap_ci_low": ci_low,
                "cv_bootstrap_ci_high": ci_high,
                "cv_sign_flip_pvalue_one_sided": sign_flip_pvalue(cv_delta),
                "external_mean_dice_delta": external_mean_delta,
                "practical_margin": PRACTICAL_MARGIN,
                "demonstrated": demonstrated,
            }
        )
    return summaries, comparisons, embryo_rows, validation


def volume_score_rows(
    records: list[SampleCurve], validation: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        if record.sample.split == "train":
            if record.sample.timepoint == PILOT_TIMEPOINT:
                continue
            threshold = validation[record.method]["fold_thresholds"][
                record.sample.embryo
            ]
            analysis_split = "cross_validation"
        else:
            threshold = validation[record.method]["global_threshold"]
            analysis_split = "external_test"
        threshold_index = int(np.flatnonzero(THRESHOLDS == threshold)[0])
        rows.append(
            {
                "split": analysis_split,
                "embryo": record.sample.embryo,
                "timepoint": record.sample.timepoint,
                "method": record.method,
                "threshold": threshold,
                "dice": float(record.dice[threshold_index]),
                "runtime_ms": 1000.0 * record.runtime_s,
                "source_shape": "x".join(map(str, record.source_shape)),
                "output_shape": "x".join(map(str, record.output_shape)),
            }
        )
    return rows


def save_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_scores_plot(
    embryo_rows: list[dict[str, object]],
    methods: list[Method],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(10.5, 5.2))
    rng = np.random.default_rng(20260716)
    for index, method in enumerate(methods):
        for split, color, marker, offset in (
            ("cross_validation", "#2a6fbb", "o", -0.10),
            ("external_test", "#bc6c25", "D", 0.10),
        ):
            values = [
                float(row["dice"])
                for row in embryo_rows
                if row["method"] == method.key and row["split"] == split
            ]
            jitter = rng.normal(0.0, 0.018, size=len(values))
            axis.scatter(
                np.full(len(values), index + offset) + jitter,
                values,
                color=color,
                marker=marker,
                alpha=0.82,
                label=(
                    "11-embryo cross-validation"
                    if index == 0 and split == "cross_validation"
                    else "4-embryo external test" if index == 0 else None
                ),
            )
            axis.hlines(
                statistics.fmean(values),
                index + offset - 0.13,
                index + offset + 0.13,
                color=color,
                linewidth=3,
            )
    axis.set_xticks(range(len(methods)), [method.label for method in methods])
    axis.tick_params(axis="x", rotation=18)
    axis.set_ylabel("Embryo-level semantic Dice")
    axis.set_title("BBBC050: each point is one embryo")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def make_comparison_plot(
    comparisons: list[dict[str, object]], destination: Path
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8.5, 4.2))
    y = np.arange(len(comparisons))
    means = np.asarray([row["cv_mean_dice_delta"] for row in comparisons], float)
    low = np.asarray([row["cv_bootstrap_ci_low"] for row in comparisons], float)
    high = np.asarray([row["cv_bootstrap_ci_high"] for row in comparisons], float)
    axis.errorbar(
        means,
        y,
        xerr=np.vstack((means - low, high - means)),
        fmt="o",
        color="#2a6fbb",
        capsize=5,
    )
    axis.axvline(0.0, color="black", linewidth=1)
    axis.axvline(
        PRACTICAL_MARGIN,
        color="#6a994e",
        linestyle="--",
        label=f"predeclared margin ({PRACTICAL_MARGIN:.3f})",
    )
    axis.set_yticks(y, [str(row["baseline_label"]) for row in comparisons])
    axis.set_xlabel("SplineOps projection minus baseline Dice")
    axis.set_title("Primary paired embryo comparison (95% bootstrap CI)")
    axis.grid(axis="x", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def make_example_plot(
    sample: Sample,
    images: ZipFile,
    ground_truth: ZipFile,
    methods: list[Method],
    validation: dict[str, dict[str, object]],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    image_raw = read_tiff(images, sample.image_member)
    labels = read_tiff(ground_truth, sample.ground_truth_member)
    image, _ = normalize_image(image_raw)
    output_shape = half_xy_shape(image.shape)
    figure, axes = plt.subplots(1, len(methods) + 1, figsize=(15.5, 3.4))
    original_plane = image[image.shape[0] // 2]
    axes[0].imshow(original_plane, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].contour(
        (labels[labels.shape[0] // 2] > 0).astype(float),
        levels=(0.5,),
        colors=("#55d66b",),
        linewidths=0.7,
    )
    axes[0].set_title("Original + truth")
    for axis, method in zip(axes[1:], methods):
        output = method.call(image, output_shape)
        target = resize_ground_truth(
            labels, output_shape, grid=method.ground_truth_grid
        )
        threshold = float(validation[method.key]["global_threshold"])
        plane = output[output.shape[0] // 2]
        prediction = output >= threshold
        axis.imshow(plane, cmap="gray", vmin=0.0, vmax=1.0)
        axis.contour(
            target[target.shape[0] // 2].astype(float),
            levels=(0.5,),
            colors=("#55d66b",),
            linewidths=0.7,
        )
        axis.contour(
            prediction[prediction.shape[0] // 2].astype(float),
            levels=(0.5,),
            colors=("#e6508f",),
            linewidths=0.6,
        )
        axis.set_title(f"{method.label}\nt={threshold:.3f}", fontsize=9)
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        f"Preselected external example {sample.embryo}_t{sample.timepoint}: "
        "truth green, prediction magenta"
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-zip", type=Path)
    parser.add_argument("--ground-truth-zip", type=Path)
    parser.add_argument(
        "--cache-dir", type=Path, default=default_cache / "splineops" / "bbbc050"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repo_root / "benchmarks" / "bbbc050"
    )
    parser.add_argument("--docs-static-dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if (args.images_zip is None) != (args.ground_truth_zip is None):
        parser.error("provide both archive overrides or neither")
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.images_zip is None:
        image_path = fetch_archive(ARCHIVES[0], args.cache_dir)
        truth_path = fetch_archive(ARCHIVES[1], args.cache_dir)
    else:
        image_path = verified_override(args.images_zip, ARCHIVES[0])
        truth_path = verified_override(args.ground_truth_zip, ARCHIVES[1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = study_methods()
    with ZipFile(image_path) as images, ZipFile(truth_path) as ground_truth:
        samples = discover_samples(images, ground_truth)
        records = process_samples(samples, images, ground_truth, methods)
        summaries, comparisons, embryo_rows, validation = build_validation_results(
            records,
            methods,
            bootstrap_resamples=args.bootstrap_resamples,
        )
        volume_rows = volume_score_rows(records, validation)
        demonstrated = all(bool(row["demonstrated"]) for row in comparisons)
        conclusion = {
            "superiority_demonstrated": demonstrated,
            "claim": (
                "SplineOps projection is superior for the frozen BBBC050 "
                "threshold-segmentation protocol"
            ),
            "statement": (
                "The predeclared superiority criterion was met against every "
                "baseline."
                if demonstrated
                else "The predeclared superiority criterion was not met against "
                "every baseline."
            ),
        }
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": {
                "name": "BBBC050 version 2",
                "page": DATASET_PAGE,
                "license": "CC BY 3.0",
                "images": asdict(ARCHIVES[0]),
                "ground_truth": asdict(ARCHIVES[1]),
                "counts": EXPECTED_IMAGE_COUNTS,
                "embryos": EXPECTED_EMBRYO_COUNTS,
                "ground_truth_kind": "GroundTruth_QCANet > 0",
            },
            "protocol": {
                "primary_metric": "mean semantic Dice per held-out embryo",
                "validation": "leave-one-training-embryo-out",
                "external_validation": "four acquisition-shifted test embryos",
                "pilot_timepoint_excluded": PILOT_TIMEPOINT,
                "normalization_percentiles": (1.0, 99.9),
                "resize_axes": SPATIAL_AXES,
                "resize_rule": "retain Z; Y and X become ceil(n / 2)",
                "thresholds": THRESHOLDS.tolist(),
                "practical_margin": PRACTICAL_MARGIN,
                "bootstrap_resamples": args.bootstrap_resamples,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "success_rule": (
                    "against every baseline: CV mean delta >= 0.005, paired "
                    "95% bootstrap lower bound > 0, external mean delta >= 0"
                ),
            },
            "methods": [
                {
                    "key": method.key,
                    "label": method.label,
                    "semantics": method.semantics,
                    "ground_truth_grid": method.ground_truth_grid,
                }
                for method in methods
            ],
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "processor": cpu_model(),
                "thread_environment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                    )
                },
                "packages": {
                    name: package_version(name)
                    for name in (
                        "splineops",
                        "numpy",
                        "scipy",
                        "scikit-image",
                        "tifffile",
                    )
                },
            },
            "summary": summaries,
            "comparisons": comparisons,
            "embryo_scores": embryo_rows,
            "fold_thresholds": [
                {
                    "method": method,
                    "held_out_embryo": embryo,
                    "threshold": threshold,
                }
                for method, values in validation.items()
                for embryo, threshold in values["fold_thresholds"].items()
            ],
            "conclusion": conclusion,
            "limits": [
                "The downstream segmenter is a fitted scalar threshold, not a neural network.",
                "The primary analysis has 11 independent embryos; frames are not treated as independent.",
                "The external test has four embryos and is directional confirmation only.",
                "Runtime is machine-specific and the scikit-image grid is not endpoint-identical.",
            ],
        }

        result_json = args.output_dir / "results.json"
        result_json.write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        save_csv(args.output_dir / "summary.csv", summaries)
        save_csv(args.output_dir / "comparisons.csv", comparisons)
        save_csv(args.output_dir / "embryo_scores.csv", embryo_rows)
        save_csv(args.output_dir / "volume_scores.csv", volume_rows)

        plots: list[Path] = []
        if not args.no_plots:
            scores_plot = args.output_dir / "bbbc050_scores.png"
            comparison_plot = args.output_dir / "bbbc050_comparisons.png"
            example_plot = args.output_dir / "bbbc050_example.png"
            make_scores_plot(embryo_rows, methods, scores_plot)
            make_comparison_plot(comparisons, comparison_plot)
            example = next(
                sample
                for sample in samples
                if sample.split == "test"
                and sample.embryo == "Emb1"
                and sample.timepoint == "001"
            )
            make_example_plot(
                example,
                images,
                ground_truth,
                methods,
                validation,
                example_plot,
            )
            plots.extend((scores_plot, comparison_plot, example_plot))
            if args.docs_static_dir is not None:
                args.docs_static_dir.mkdir(parents=True, exist_ok=True)
                for source in plots:
                    docs_name = f"bbbc050-study-{source.name.removeprefix('bbbc050_')}"
                    shutil.copy2(source, args.docs_static_dir / docs_name)

    print(conclusion["statement"])
    print(f"Wrote {result_json}")
    for plot in plots:
        print(f"Wrote {plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
