#!/usr/bin/env python3
"""Run the frozen SELMA3D nuclei quality-at-speed confirmation study.

The protocol is recorded in ``benchmarks/selma3d/PROTOCOL.md``. The script
downloads the official files, verifies every SHA-256 digest, and publishes
negative results as readily as positive ones.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy import ndimage, stats

ACCESSION = "S-BIAD1196"
DATASET_PAGE = (
    "https://www.ebi.ac.uk/bioimage-archive/galleries/ai/analysed-dataset/"
    "S-BIAD1196/"
)
CHALLENGE_PAGE = "https://selma3d.grand-challenge.org/data/"
BASE_URL = (
    "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/196/S-BIAD1196/Files/"
    "SELMA3D_training_annotated/shannel_cells"
)
EXPECTED_SHAPE = (200, 200, 200)
OUTPUT_SHAPE = (100, 100, 100)
SAMPLE_IDS = tuple(range(12))
BOOTSTRAP_SEED = 20260716
BOOTSTRAP_RESAMPLES = 20_000
NONINFERIORITY_MARGIN = 0.01
MINIMUM_SPEEDUP = 10.0
PILOT_AUC = {
    "splineops_projection": 0.96144689,
    "scipy_gaussian": 0.95704538,
    "skimage_resize": 0.94753048,
    "torch_area": 0.94708409,
}

CHECKSUMS = {
    "gt/patchvolume_000.nii.gz": "ad59e596aba5bb3562360fa824f9f23474facf2d80242c79edcc5cd8b71f5491",
    "gt/patchvolume_001.nii.gz": "9669072ce2a4cad1a96b5393afc0f10c6f146f3452ebfe6c6e7adbe941ca4eaa",
    "gt/patchvolume_002.nii.gz": "3cebf4f81169b3349971b06a673d2d948bc3dfc929379c9d6664cb35553d8a98",
    "gt/patchvolume_003.nii.gz": "648b71876b83c2d65c8cc5f6ecbaf2176e39b400483ebe233d7d17c52d95fa53",
    "gt/patchvolume_004.nii.gz": "3ad54ed31e99513916f1bdf931b8694e32b524c70791d1dfba8031bc279a1ffb",
    "gt/patchvolume_005.nii.gz": "09b860c25291b1a99f82d790cfa2b6296f50056e863ce8004e68d5c113dab666",
    "gt/patchvolume_006.nii.gz": "839cd4b5a8ca6a54ba8c805fcfbaa03b3fa2e570d4136e497aa611275980d7e4",
    "gt/patchvolume_007.nii.gz": "a067cc4587290745eb8964faadb8037e8a3b48ee0f471efc08ab11fbf9205ec8",
    "gt/patchvolume_008.nii.gz": "908bf1bce5ea16da4eb8275dd39cfbec03bbb330f1e0e3fd59fdb4fbdd16ebaf",
    "gt/patchvolume_009.nii.gz": "2bb38e298870fd783b986e3bb953a648f07dbe50b55e242b2eb2e740984d66ca",
    "gt/patchvolume_010.nii.gz": "41f240844354753118398633278733d14ddbc04f74121f4cfe8f524761e7616f",
    "gt/patchvolume_011.nii.gz": "3568b78e697997995123d270f3ae217659811949d3c79f9298acb2ac93c34522",
    "raw/patchvolume_000_0000.nii.gz": "25e1e351872db53a22f2ff9196892e4cefcc44c3d0be98186c13d3c77f8e0397",
    "raw/patchvolume_001_0000.nii.gz": "d12e43353982148b6726110ddef6331c0f057b964924accbd492c03ae2af09b7",
    "raw/patchvolume_002_0000.nii.gz": "74915c737bb470052808e3cdbeb6303fb4c64e81f0a0b0e375057116a9b10590",
    "raw/patchvolume_003_0000.nii.gz": "631a43a1f511f17ae2a2c92f2fb7841fd6f7eab683fcaf0ddaf530b7eb9a9501",
    "raw/patchvolume_004_0000.nii.gz": "f2043a0db9247d1a24349c499712223fc04050c54bbc2d09e4b854905faa5510",
    "raw/patchvolume_005_0000.nii.gz": "297e6e93418497a7f6af4814f08cd03741a21723a0e06ed7003db60ca59d8b9f",
    "raw/patchvolume_006_0000.nii.gz": "b2ec6c7d6d3a6c4d6fc99237a04c2ceb392aaed7505e33387bb10f11f1dc031a",
    "raw/patchvolume_007_0000.nii.gz": "f5e6219cc27a07772d696eb8f2c235b8f5aea96a1e6d1874a1e1254b38c90017",
    "raw/patchvolume_008_0000.nii.gz": "09bf8dac83abfeb575f76feaf4fb252258e78d7631b4c9f51a55460cfdaf9529",
    "raw/patchvolume_009_0000.nii.gz": "7f28921b85cb7a69c3937c3acfe1a4c1fa4091bd8d1d80a0c223e0f1ab82240c",
    "raw/patchvolume_010_0000.nii.gz": "acad1cdb850705d8263482c1a0c196f4ae801ddc6ef1a64e8f1b2132a93da486",
    "raw/patchvolume_011_0000.nii.gz": "acbb1c96da13958c892c9351aa0520988263640e3d247e961d93256bc2970f82",
}

VolumeMethod = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    semantics: str
    grid: str
    antialiased_comparison: bool
    call: VolumeMethod


@dataclass(frozen=True)
class Scores:
    roc_auc: float
    average_precision: float
    top_prevalence_dice: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_file(path: Path, expected: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"checksum mismatch for {path}: {actual}")
    return path


def fetch_file(relative: str, cache_dir: Path) -> Path:
    expected = CHECKSUMS[relative]
    destination = cache_dir / "shannel_cells" / relative
    if destination.exists():
        return verified_file(destination, expected)

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_URL}/{relative}"
    request = urllib.request.Request(
        url, headers={"User-Agent": "SplineOps SELMA3D reproducibility study"}
    )
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f"{destination.name}-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                shutil.copyfileobj(response, handle)
        verified_file(temporary, expected)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def data_root(data_dir: Path | None, cache_dir: Path) -> Path:
    if data_dir is None:
        for relative in CHECKSUMS:
            fetch_file(relative, cache_dir)
        return cache_dir / "shannel_cells"

    candidate = data_dir / "shannel_cells"
    root = candidate if candidate.is_dir() else data_dir
    for relative, expected in CHECKSUMS.items():
        verified_file(root / relative, expected)
    return root


def require_nibabel() -> Any:
    try:
        import nibabel
    except ImportError as exc:
        raise RuntimeError(
            "Install the study dependencies with "
            "`python -m pip install -e '.[selma3d-study]'`."
        ) from exc
    return nibabel


def read_sample(root: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    nibabel = require_nibabel()
    image_path = root / "raw" / f"patchvolume_{sample_id:03d}_0000.nii.gz"
    mask_path = root / "gt" / f"patchvolume_{sample_id:03d}.nii.gz"
    image = np.asarray(nibabel.load(image_path).dataobj, dtype=np.float32)
    labels = np.asarray(nibabel.load(mask_path).dataobj) > 0
    if image.shape != EXPECTED_SHAPE or labels.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"sample {sample_id:03d} has shapes {image.shape} and {labels.shape}"
        )
    if not np.any(labels) or np.all(labels):
        raise RuntimeError(f"sample {sample_id:03d} has a degenerate mask")
    return image, labels


def normalize_image(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    low, high = np.percentile(image, (1.0, 99.9))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("image has no usable percentile intensity range")
    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.asarray(normalized, dtype=np.float32), float(low), float(high)


def endpoint_coordinates(
    input_shape: Sequence[int], output_shape: Sequence[int]
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.linspace(0.0, float(source - 1), target, dtype=np.float64)
        for source, target in zip(input_shape, output_shape)
    )


def nearest_indices(source: int, target: int, *, grid: str) -> np.ndarray:
    if grid == "endpoint":
        coordinates = np.linspace(0.0, float(source - 1), target)
    elif grid == "half_pixel":
        coordinates = (np.arange(target, dtype=np.float64) + 0.5) * (
            source / target
        ) - 0.5
    else:
        raise ValueError(f"unknown grid {grid!r}")
    # floor(x + 0.5) is explicit round-half-up for half-grid ties.
    return np.clip(np.floor(coordinates + 0.5), 0, source - 1).astype(np.intp)


def sample_labels(
    labels: np.ndarray, output_shape: Sequence[int], *, grid: str
) -> np.ndarray:
    indices = tuple(
        nearest_indices(source, target, grid=grid)
        for source, target in zip(labels.shape, output_shape)
    )
    return np.asarray(labels[np.ix_(*indices)], dtype=bool)


def resize_splineops(image: np.ndarray, *, antialias: bool) -> np.ndarray:
    from splineops import resize

    method = "cubic-antialiasing" if antialias else "cubic"
    return np.asarray(
        resize(image, output_size=OUTPUT_SHAPE, axes=(0, 1, 2), method=method),
        dtype=np.float32,
    )


def resize_scipy_gaussian(image: np.ndarray) -> np.ndarray:
    factors = tuple(
        (source - 1) / (target - 1) for source, target in zip(image.shape, OUTPUT_SHAPE)
    )
    filtered = ndimage.gaussian_filter(
        image,
        sigma=tuple(max(0.0, (factor - 1.0) / 2.0) for factor in factors),
        mode="reflect",
    )
    coordinates = endpoint_coordinates(image.shape, OUTPUT_SHAPE)
    grid = np.meshgrid(*coordinates, indexing="ij", sparse=False)
    return np.asarray(
        ndimage.map_coordinates(
            filtered, grid, order=3, mode="reflect", prefilter=True
        ),
        dtype=np.float32,
    )


def resize_skimage(image: np.ndarray) -> np.ndarray:
    try:
        from skimage.transform import resize
    except ImportError as exc:
        raise RuntimeError(
            "Install the study dependencies with "
            "`python -m pip install -e '.[selma3d-study]'`."
        ) from exc
    return np.asarray(
        resize(
            image,
            OUTPUT_SHAPE,
            order=3,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
            clip=False,
        ),
        dtype=np.float32,
    )


def torch_area_method() -> VolumeMethod:
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError(
            "Install the study dependencies with "
            "`python -m pip install -e '.[selma3d-study]'`."
        ) from exc

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    def apply(image: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None]
        output = functional.interpolate(tensor, size=OUTPUT_SHAPE, mode="area")
        return output[0, 0].detach().cpu().numpy()

    return apply


def study_methods() -> list[Method]:
    return [
        Method(
            "splineops_projection",
            "SplineOps projection AA",
            "endpoint-aligned cubic projection antialiasing",
            "endpoint",
            False,
            lambda image: resize_splineops(image, antialias=True),
        ),
        Method(
            "splineops_interpolation",
            "SplineOps cubic, no AA",
            "endpoint-aligned cubic interpolation without antialiasing",
            "endpoint",
            False,
            lambda image: resize_splineops(image, antialias=False),
        ),
        Method(
            "scipy_gaussian",
            "SciPy Gaussian + cubic",
            "Gaussian prefilter and endpoint-aligned cubic sampling",
            "endpoint",
            True,
            resize_scipy_gaussian,
        ),
        Method(
            "skimage_resize",
            "scikit-image cubic AA",
            "native half-pixel cubic resize with Gaussian antialiasing",
            "half_pixel",
            True,
            resize_skimage,
        ),
        Method(
            "torch_area",
            "PyTorch area",
            "native regional area resize on the half-pixel geometry",
            "half_pixel",
            True,
            torch_area_method(),
        ),
    ]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=bool).ravel()
    positives = int(np.count_nonzero(truth))
    negatives = truth.size - positives
    if positives == 0 or negatives == 0:
        raise ValueError("ROC AUC requires both classes")
    ranks = stats.rankdata(values, method="average")
    rank_sum = float(np.sum(ranks[truth]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=bool).ravel()
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        raise ValueError("average precision requires positives")
    order = np.argsort(-values, kind="stable")
    ordered_values = values[order]
    ordered_truth = truth[order]
    cumulative = np.cumsum(ordered_truth, dtype=np.int64)
    group_ends = np.r_[
        np.flatnonzero(ordered_values[1:] != ordered_values[:-1]), truth.size - 1
    ]
    group_positive = np.diff(np.r_[0, cumulative[group_ends]])
    precision = cumulative[group_ends] / (group_ends + 1)
    return float(np.sum(group_positive * precision) / positives)


def top_prevalence_dice(scores: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(scores, dtype=np.float64).ravel()
    truth = np.asarray(labels, dtype=bool).ravel()
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        raise ValueError("top-prevalence Dice requires positives")
    boundary = np.partition(values, values.size - positives)[values.size - positives]
    above = values > boundary
    equal = values == boundary
    slots = positives - int(np.count_nonzero(above))
    true_positive = float(np.count_nonzero(truth & above))
    equal_count = int(np.count_nonzero(equal))
    if slots > 0 and equal_count > 0:
        true_positive += slots * np.count_nonzero(truth & equal) / equal_count
    # Predicted and true foreground counts both equal positives.
    return true_positive / positives


def compute_scores(values: np.ndarray, labels: np.ndarray) -> Scores:
    if values.shape != labels.shape:
        raise ValueError(f"score/label shape mismatch: {values.shape}, {labels.shape}")
    return Scores(
        roc_auc=roc_auc(values, labels),
        average_precision=average_precision(values, labels),
        top_prevalence_dice=top_prevalence_dice(values, labels),
    )


def timed_calls(
    method: Method,
    image: np.ndarray,
    *,
    warmups: int,
    repetitions: int,
) -> list[float]:
    for _ in range(warmups):
        output = method.call(image)
        if output.shape != OUTPUT_SHAPE:
            raise RuntimeError(f"{method.key} returned {output.shape}")
    durations: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = method.call(image)
        durations.append(time.perf_counter() - start)
        if output.shape != OUTPUT_SHAPE or not np.all(np.isfinite(output)):
            raise RuntimeError(f"invalid output from {method.key}")
    return durations


def paired_interval(
    differences: np.ndarray,
    *,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    estimates = differences[indices].mean(axis=1)
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    score_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    methods: list[Method],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = {method.key: method.label for method in methods}
    colors = {
        "splineops_projection": "#1877b8",
        "splineops_interpolation": "#83b8d9",
        "scipy_gaussian": "#e58b2b",
        "skimage_resize": "#4a9d63",
        "torch_area": "#9067b0",
    }
    keys = [method.key for method in methods]
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))

    for position, key in enumerate(keys):
        values = [row["roc_auc"] for row in score_rows if row["method"] == key]
        x = np.full(len(values), position, dtype=float)
        axes[0].scatter(
            x,
            values,
            s=24,
            alpha=0.65,
            color=colors[key],
            edgecolor="none",
        )
        axes[0].plot(
            [position - 0.22, position + 0.22],
            [np.mean(values), np.mean(values)],
            color="black",
            linewidth=2,
        )
    axes[0].set_ylabel("Voxel-ranking ROC AUC")
    axes[0].set_ylim(0.5, 1.005)
    axes[0].set_xticks(
        range(len(keys)), [labels[key] for key in keys], rotation=25, ha="right"
    )
    axes[0].set_title("12 held-out nuclei patches")
    axes[0].grid(axis="y", alpha=0.25)

    medians = [
        next(row["median_runtime_s"] for row in summary_rows if row["method"] == key)
        for key in keys
    ]
    axes[1].bar(
        range(len(keys)),
        np.asarray(medians) * 1000.0,
        color=[colors[key] for key in keys],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median one-volume runtime (ms, log scale)")
    axes[1].set_xticks(
        range(len(keys)), [labels[key] for key in keys], rotation=25, ha="right"
    )
    axes[1].set_title("200³ → 100³, one CPU thread")
    axes[1].grid(axis="y", alpha=0.25)

    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_study(
    root: Path,
    output_dir: Path,
    *,
    warmups: int,
    repetitions: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    methods = study_methods()
    score_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for sample_index, sample_id in enumerate(SAMPLE_IDS, start=1):
        print(
            f"[{sample_index:2d}/{len(SAMPLE_IDS)}] patch {sample_id:03d}", flush=True
        )
        source, labels = read_sample(root, sample_id)
        image, low, high = normalize_image(source)
        targets = {
            grid: sample_labels(labels, OUTPUT_SHAPE, grid=grid)
            for grid in {method.grid for method in methods}
        }
        for method in methods:
            output = method.call(image)
            if output.shape != OUTPUT_SHAPE or not np.all(np.isfinite(output)):
                raise RuntimeError(f"invalid output from {method.key}")
            scores = compute_scores(output, targets[method.grid])
            score_rows.append(
                {
                    "sample": f"patchvolume_{sample_id:03d}",
                    "method": method.key,
                    **asdict(scores),
                    "positive_voxels": int(np.count_nonzero(targets[method.grid])),
                    "total_voxels": int(targets[method.grid].size),
                    "normalization_low": low,
                    "normalization_high": high,
                }
            )
            durations = timed_calls(
                method,
                image,
                warmups=warmups,
                repetitions=repetitions,
            )
            for repetition, duration in enumerate(durations):
                timing_rows.append(
                    {
                        "sample": f"patchvolume_{sample_id:03d}",
                        "method": method.key,
                        "repetition": repetition,
                        "runtime_s": duration,
                    }
                )
            print(
                f"  {method.key:24s} AUC={scores.roc_auc:.6f} "
                f"median={statistics.median(durations) * 1000:.2f} ms",
                flush=True,
            )
        del source, labels, image, targets

    summary_rows: list[dict[str, Any]] = []
    per_patch_times: dict[str, dict[str, float]] = {}
    for method in methods:
        selected = [row for row in score_rows if row["method"] == method.key]
        patch_times: dict[str, float] = {}
        for sample_id in SAMPLE_IDS:
            sample = f"patchvolume_{sample_id:03d}"
            values = [
                row["runtime_s"]
                for row in timing_rows
                if row["method"] == method.key and row["sample"] == sample
            ]
            patch_times[sample] = statistics.median(values)
        per_patch_times[method.key] = patch_times
        summary_rows.append(
            {
                "method": method.key,
                "label": method.label,
                "mean_roc_auc": statistics.fmean(row["roc_auc"] for row in selected),
                "mean_average_precision": statistics.fmean(
                    row["average_precision"] for row in selected
                ),
                "mean_top_prevalence_dice": statistics.fmean(
                    row["top_prevalence_dice"] for row in selected
                ),
                "median_runtime_s": statistics.median(patch_times.values()),
            }
        )

    score_by_method = {
        method.key: {
            row["sample"]: float(row["roc_auc"])
            for row in score_rows
            if row["method"] == method.key
        }
        for method in methods
    }
    comparison_rows: list[dict[str, Any]] = []
    for comparison_index, method in enumerate(
        [method for method in methods if method.antialiased_comparison]
    ):
        differences = np.asarray(
            [
                score_by_method["splineops_projection"][sample]
                - score_by_method[method.key][sample]
                for sample in sorted(score_by_method[method.key])
            ],
            dtype=np.float64,
        )
        low, high = paired_interval(
            differences,
            resamples=bootstrap_resamples,
            seed=BOOTSTRAP_SEED + comparison_index,
        )
        speedup = statistics.median(
            per_patch_times[method.key].values()
        ) / statistics.median(per_patch_times["splineops_projection"].values())
        comparison_rows.append(
            {
                "comparison": method.key,
                "mean_auc_difference": float(np.mean(differences)),
                "ci95_low": low,
                "ci95_high": high,
                "noninferior_margin": NONINFERIORITY_MARGIN,
                "noninferior": bool(low > -NONINFERIORITY_MARGIN),
                "median_runtime_ratio": speedup,
                "speed_target_applies": method.key
                in {"scipy_gaussian", "skimage_resize"},
                "speed_target_pass": (
                    bool(speedup >= MINIMUM_SPEEDUP)
                    if method.key in {"scipy_gaussian", "skimage_resize"}
                    else None
                ),
            }
        )

    quality_pass = all(row["noninferior"] for row in comparison_rows)
    speed_pass = all(
        row["speed_target_pass"]
        for row in comparison_rows
        if row["speed_target_applies"]
    )
    overall_pass = quality_pass and speed_pass

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        output_dir / "scores.csv",
        score_rows,
        (
            "sample",
            "method",
            "roc_auc",
            "average_precision",
            "top_prevalence_dice",
            "positive_voxels",
            "total_voxels",
            "normalization_low",
            "normalization_high",
        ),
    )
    write_csv(
        output_dir / "timings.csv",
        timing_rows,
        ("sample", "method", "repetition", "runtime_s"),
    )
    write_csv(
        output_dir / "summary.csv",
        summary_rows,
        (
            "method",
            "label",
            "mean_roc_auc",
            "mean_average_precision",
            "mean_top_prevalence_dice",
            "median_runtime_s",
        ),
    )
    write_csv(
        output_dir / "comparisons.csv",
        comparison_rows,
        (
            "comparison",
            "mean_auc_difference",
            "ci95_low",
            "ci95_high",
            "noninferior_margin",
            "noninferior",
            "median_runtime_ratio",
            "speed_target_applies",
            "speed_target_pass",
        ),
    )
    make_figure(
        score_rows,
        summary_rows,
        methods,
        output_dir / "selma3d_nuclei.png",
    )

    result = {
        "study": "SELMA3D nuclei quality-at-speed confirmation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "accession": ACCESSION,
            "dataset_page": DATASET_PAGE,
            "challenge_page": CHALLENGE_PAGE,
            "subset": "shannel_cells",
            "samples": len(SAMPLE_IDS),
            "source_shape": list(EXPECTED_SHAPE),
            "output_shape": list(OUTPUT_SHAPE),
            "license_interpretation": "CC BY-NC (stricter of conflicting official metadata)",
            "source_files_redistributed": False,
            "specimen_grouping_available": False,
        },
        "protocol": {
            "pilot_subset": "cFos-Active_Neurons",
            "pilot_mean_auc": PILOT_AUC,
            "otsu_pilot_mean_dice": 0.03263731515806877,
            "confirmation_subset": "shannel_cells",
            "primary_metric": "voxel-ranking ROC AUC",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": bootstrap_resamples,
            "minimum_speedup": MINIMUM_SPEEDUP,
            "timing_warmups": warmups,
            "timing_repetitions": repetitions,
            "threads": 1,
        },
        "methods": [
            {
                "key": method.key,
                "label": method.label,
                "semantics": method.semantics,
                "grid": method.grid,
            }
            for method in methods
        ],
        "summary": summary_rows,
        "comparisons": comparison_rows,
        "decision": {
            "quality_noninferiority_pass": quality_pass,
            "speed_pass": speed_pass,
            "narrow_quality_at_speed_claim_pass": overall_pass,
            "segmentation_superiority_demonstrated": False,
            "broad_resampling_superiority_demonstrated": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": {
                name: package_version(name)
                for name in (
                    "splineops",
                    "numpy",
                    "scipy",
                    "scikit-image",
                    "torch",
                    "nibabel",
                    "matplotlib",
                )
            },
        },
    }
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="verified shannel_cells directory (or its parent); otherwise download",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "splineops" / "selma3d",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/selma3d"),
    )
    parser.add_argument("--timing-warmups", type=int, default=2)
    parser.add_argument("--timing-repetitions", type=int, default=7)
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timing_warmups < 0 or args.timing_repetitions < 1:
        raise ValueError(
            "timing counts must be non-negative warmups and positive repetitions"
        )
    if args.bootstrap_resamples < 1:
        raise ValueError("bootstrap resamples must be positive")
    root = data_root(args.data_dir, args.cache_dir)
    result = run_study(
        root,
        args.output_dir,
        warmups=args.timing_warmups,
        repetitions=args.timing_repetitions,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    decision = result["decision"]
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
