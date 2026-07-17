#!/usr/bin/env python3
"""Run the frozen MiniVess 8x vessel-overview confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

THREAD_ENVIRONMENT = (
    "LSRESIZE_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
for variable in THREAD_ENVIRONMENT:
    os.environ[variable] = "1"

import numpy as np
from scipy import ndimage, signal

try:
    from scripts import benchmark_selma3d_nuclei as common
except ModuleNotFoundError:  # Executed directly from the scripts directory.
    import benchmark_selma3d_nuclei as common  # type: ignore[no-redef]

DATASET_ID = "bf268b89-1420-476b-b428-b85a913eb523"
DATASET_DOI = "10.25493/HPBE-YHK"
DATASET_PAGE = (
    "https://search.kg.ebrains.eu/instances/" "bf268b89-1420-476b-b428-b85a913eb523"
)
PAPER_PAGE = "https://doi.org/10.1038/s41597-023-02048-8"
CODE_PAGE = "https://github.com/ctpn/minivess"
API_BASE = f"https://data-proxy.ebrains.eu/api/v1/datasets/{DATASET_ID}"

EXPECTED_XY = (512, 512)
OUTPUT_XY = (64, 64)
PILOT_IDS = (6, 28, 35, 38, 44, 48, 58, 68)
CONFIRMATION_IDS = tuple(index for index in range(1, 71) if index not in PILOT_IDS)
BOOTSTRAP_SEED = 20260716
BOOTSTRAP_RESAMPLES = 20_000
QUALITY_MARGIN = 0.005
BONFERRONI_LOWER_QUANTILE = 0.05 / 5.0
MINIMUM_SPEEDUPS: dict[str, float] = {
    "scipy_gaussian": 3.0,
    "skimage_resize": 3.0,
    "torch_area": 1.0,
    "scipy_polyphase": 3.0,
}
STRICT_SPEEDUPS = frozenset({"torch_area"})

VolumeMethod = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    semantics: str
    grid: str
    comparison: bool
    call: VolumeMethod


def _manifest_url(prefix: str) -> str:
    query = urllib.parse.urlencode({"prefix": f"{prefix}/", "limit": 100})
    return f"{API_BASE}?{query}"


def read_manifest(cache_dir: Path, prefix: str) -> dict[str, dict[str, Any]]:
    destination = cache_dir / "manifests" / f"{prefix}.json"
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(
            _manifest_url(prefix),
            headers={"User-Agent": "SplineOps MiniVess reproducibility study"},
        )
        temporary: Path | None = None
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f"{destination.name}-",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                    shutil.copyfileobj(response, handle)
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    with destination.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    objects = payload.get("objects", [])
    selected = {
        str(item["name"]): item
        for item in objects
        if str(item.get("name", "")).startswith(f"{prefix}/")
    }
    if len(selected) != 70:
        raise RuntimeError(
            f"expected 70 {prefix} manifest objects, got {len(selected)}"
        )
    return selected


def _md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - archive identity, not authentication.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_file(path: Path, metadata: dict[str, Any]) -> Path:
    expected_size = int(metadata["bytes"])
    expected_hash = str(metadata["hash"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != expected_size or _md5(path) != expected_hash:
        raise RuntimeError(f"archive identity mismatch for {path}")
    return path


def fetch_file(relative: str, cache_dir: Path, metadata: dict[str, Any]) -> Path:
    destination = cache_dir / relative
    if destination.exists():
        return verified_file(destination, metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"{API_BASE}/{relative}",
        headers={"User-Agent": "SplineOps MiniVess reproducibility study"},
    )
    temporary: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                prefix=f"{destination.name}-",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                shutil.copyfileobj(response, handle)
        verified_file(temporary, metadata)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def prepare_files(
    data_dir: Path | None, cache_dir: Path, sample_ids: Sequence[int]
) -> tuple[Path, dict[str, dict[str, Any]]]:
    manifest = {
        **read_manifest(cache_dir, "raw"),
        **read_manifest(cache_dir, "seg"),
    }
    root = cache_dir if data_dir is None else data_dir
    for sample_id in sample_ids:
        relatives = (
            f"raw/mv{sample_id:02d}.nii.gz",
            f"seg/mv{sample_id:02d}_y.nii.gz",
        )
        for relative in relatives:
            if data_dir is None:
                fetch_file(relative, cache_dir, manifest[relative])
            else:
                verified_file(root / relative, manifest[relative])
    return root, manifest


def read_sample(root: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    nibabel = common.require_nibabel()
    image = np.asarray(
        nibabel.load(root / "raw" / f"mv{sample_id:02d}.nii.gz").dataobj,
        dtype=np.float32,
    )
    labels = (
        np.asarray(nibabel.load(root / "seg" / f"mv{sample_id:02d}_y.nii.gz").dataobj)
        > 0
    )
    if image.shape != labels.shape or image.shape[:2] != EXPECTED_XY:
        raise RuntimeError(
            f"mv{sample_id:02d} has unexpected shapes {image.shape}, {labels.shape}"
        )
    if image.ndim != 3 or not np.any(labels) or np.all(labels):
        raise RuntimeError(f"mv{sample_id:02d} has unusable image or labels")
    return image, labels


def output_shape(image: np.ndarray) -> tuple[int, int, int]:
    return (*OUTPUT_XY, int(image.shape[2]))


def resize_splineops(image: np.ndarray, *, antialias: bool) -> np.ndarray:
    from splineops import resize

    method = "cubic-antialiasing" if antialias else "cubic"
    return np.asarray(
        resize(image, output_size=OUTPUT_XY, axes=(0, 1), method=method),
        dtype=np.float32,
    )


def resize_scipy_gaussian(image: np.ndarray) -> np.ndarray:
    factor = (EXPECTED_XY[0] - 1) / (OUTPUT_XY[0] - 1)
    filtered = ndimage.gaussian_filter(
        image,
        sigma=((factor - 1.0) / 2.0, (factor - 1.0) / 2.0, 0.0),
        mode="reflect",
    )
    return np.asarray(
        ndimage.zoom(
            filtered,
            zoom=(OUTPUT_XY[0] / EXPECTED_XY[0], OUTPUT_XY[1] / EXPECTED_XY[1], 1),
            order=3,
            mode="reflect",
            prefilter=True,
            grid_mode=False,
        ),
        dtype=np.float32,
    )


def resize_skimage(image: np.ndarray) -> np.ndarray:
    try:
        from skimage.transform import resize
    except ImportError as exc:
        raise RuntimeError("Install `splineops[minivess-study]`.") from exc
    return np.asarray(
        resize(
            image,
            output_shape(image),
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
        raise RuntimeError("Install `splineops[minivess-study]`.") from exc
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    def apply(image: np.ndarray) -> np.ndarray:
        tensor = torch.from_numpy(np.ascontiguousarray(image))[None, None]
        result = functional.interpolate(tensor, size=output_shape(image), mode="area")
        return result[0, 0].detach().cpu().numpy()

    return apply


def opencv_area_method() -> VolumeMethod:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Install `splineops[minivess-study]`.") from exc
    cv2.setNumThreads(1)

    def apply(image: np.ndarray) -> np.ndarray:
        return np.asarray(
            cv2.resize(image, OUTPUT_XY, interpolation=cv2.INTER_AREA),
            dtype=np.float32,
        )

    return apply


def resize_scipy_polyphase(image: np.ndarray) -> np.ndarray:
    reduced = signal.resample_poly(image, 1, 8, axis=0, padtype="line")
    return np.asarray(
        signal.resample_poly(reduced, 1, 8, axis=1, padtype="line"),
        dtype=np.float32,
    )


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
            "Gaussian-prefiltered endpoint-aligned cubic sampling",
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
        Method(
            "opencv_area",
            "OpenCV area",
            "native INTER_AREA resize on the half-pixel geometry",
            "half_pixel",
            True,
            opencv_area_method(),
        ),
        Method(
            "scipy_polyphase",
            "SciPy polyphase FIR",
            "separable default polyphase FIR with linear boundary continuation",
            "phase_zero",
            True,
            resize_scipy_polyphase,
        ),
    ]


def sample_labels(labels: np.ndarray, *, grid: str) -> np.ndarray:
    shape = (*OUTPUT_XY, labels.shape[2])
    if grid in {"endpoint", "half_pixel"}:
        return common.sample_labels(labels, shape, grid=grid)
    if grid == "phase_zero":
        lateral = np.arange(OUTPUT_XY[0], dtype=np.intp) * 8
        depth = np.arange(labels.shape[2], dtype=np.intp)
        return np.asarray(labels[np.ix_(lateral, lateral, depth)], dtype=bool)
    raise ValueError(f"unknown grid {grid!r}")


def timed_calls(
    method: Method, image: np.ndarray, *, warmups: int, repetitions: int
) -> tuple[np.ndarray, list[float]]:
    expected = output_shape(image)
    for _ in range(warmups):
        output = method.call(image)
        if output.shape != expected or not np.all(np.isfinite(output)):
            raise RuntimeError(f"invalid output from {method.key}")
    durations: list[float] = []
    output = np.empty(expected, dtype=np.float32)
    for _ in range(repetitions):
        start = time.perf_counter()
        output = method.call(image)
        durations.append(time.perf_counter() - start)
        if output.shape != expected or not np.all(np.isfinite(output)):
            raise RuntimeError(f"invalid output from {method.key}")
    return np.asarray(output, dtype=np.float32), durations


def bootstrap_indices(sample_count: int, *, resamples: int, seed: int) -> np.ndarray:
    if sample_count < 1 or resamples < 1:
        raise ValueError("bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    return rng.integers(0, sample_count, size=(resamples, sample_count))


def bootstrap_distribution(differences: np.ndarray, indices: np.ndarray) -> np.ndarray:
    values = np.asarray(differences, dtype=np.float64)
    selected = np.asarray(indices, dtype=np.intp)
    if values.ndim != 1 or selected.ndim != 2 or selected.shape[1] != values.size:
        raise ValueError("bootstrap indices do not match paired differences")
    if np.any(selected < 0) or np.any(selected >= values.size):
        raise ValueError("bootstrap index is out of range")
    return values[selected].mean(axis=1)


def speed_condition_pass(method_key: str, runtime_ratio: float) -> bool | None:
    required = MINIMUM_SPEEDUPS.get(method_key)
    if required is None:
        return None
    if method_key in STRICT_SPEEDUPS:
        return bool(runtime_ratio > required)
    return bool(runtime_ratio >= required)


def make_figure(
    score_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    methods: list[Method],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    keys = [method.key for method in methods]
    labels = {method.key: method.label for method in methods}
    colors = (
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    )
    figure, axes = plt.subplots(1, 2, figsize=(12.2, 4.5))
    for position, key in enumerate(keys):
        values = [
            row["top_prevalence_dice"] for row in score_rows if row["method"] == key
        ]
        axes[0].scatter(
            np.full(len(values), position),
            values,
            s=16,
            alpha=0.45,
            color=colors[position],
            edgecolor="none",
        )
        axes[0].plot(
            [position - 0.22, position + 0.22],
            [np.mean(values), np.mean(values)],
            color="black",
            linewidth=2,
        )
    axes[0].set_ylabel("Top-prevalence vessel Dice")
    axes[0].set_ylim(0.0, 1.005)
    axes[0].set_xticks(
        range(len(keys)), [labels[key] for key in keys], rotation=28, ha="right"
    )
    axes[0].set_title("62 held-out MiniVess volumes")
    axes[0].grid(axis="y", alpha=0.25)

    medians = [
        next(row["median_runtime_s"] for row in summary_rows if row["method"] == key)
        for key in keys
    ]
    axes[1].bar(
        range(len(keys)),
        np.asarray(medians) * 1000.0,
        color=[colors[position] for position in range(len(keys))],
    )
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Median one-volume runtime (ms, log scale)")
    axes[1].set_xticks(
        range(len(keys)), [labels[key] for key in keys], rotation=28, ha="right"
    )
    axes[1].set_title("512×512×Z → 64×64×Z, one CPU thread")
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_study(
    root: Path,
    manifest: dict[str, dict[str, Any]],
    output_dir: Path,
    *,
    sample_ids: Sequence[int],
    sample_set: str,
    warmups: int,
    repetitions: int,
    bootstrap_resamples: int,
) -> dict[str, Any]:
    methods = study_methods()
    score_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    for sample_index, sample_id in enumerate(sample_ids):
        print(
            f"[{sample_index + 1:2d}/{len(sample_ids)}] mv{sample_id:02d}", flush=True
        )
        source, labels = read_sample(root, sample_id)
        image, low, high = common.normalize_image(source)
        targets = {
            grid: sample_labels(labels, grid=grid)
            for grid in {method.grid for method in methods}
        }
        ordered = (
            methods[sample_index % len(methods) :]
            + methods[: sample_index % len(methods)]
        )
        for method in ordered:
            output, durations = timed_calls(
                method, image, warmups=warmups, repetitions=repetitions
            )
            scores = common.compute_scores(output, targets[method.grid])
            score_rows.append(
                {
                    "sample": f"mv{sample_id:02d}",
                    "method": method.key,
                    **asdict(scores),
                    "positive_voxels": int(np.count_nonzero(targets[method.grid])),
                    "total_voxels": int(targets[method.grid].size),
                    "source_depth": int(image.shape[2]),
                    "normalization_low": low,
                    "normalization_high": high,
                }
            )
            for repetition, duration in enumerate(durations):
                timing_rows.append(
                    {
                        "sample": f"mv{sample_id:02d}",
                        "method": method.key,
                        "repetition": repetition,
                        "runtime_s": duration,
                    }
                )
            print(
                f"  {method.key:24s} Dice={scores.top_prevalence_dice:.6f} "
                f"median={statistics.median(durations) * 1000:.2f} ms",
                flush=True,
            )
        del source, labels, image, targets

    per_volume_times: dict[str, dict[str, float]] = {}
    summary_rows: list[dict[str, Any]] = []
    for method in methods:
        selected = [row for row in score_rows if row["method"] == method.key]
        volume_times = {
            row["sample"]: statistics.median(
                timing["runtime_s"]
                for timing in timing_rows
                if timing["method"] == method.key and timing["sample"] == row["sample"]
            )
            for row in selected
        }
        per_volume_times[method.key] = volume_times
        summary_rows.append(
            {
                "method": method.key,
                "label": method.label,
                "mean_top_prevalence_dice": statistics.fmean(
                    row["top_prevalence_dice"] for row in selected
                ),
                "mean_average_precision": statistics.fmean(
                    row["average_precision"] for row in selected
                ),
                "mean_roc_auc": statistics.fmean(row["roc_auc"] for row in selected),
                "median_runtime_s": statistics.median(volume_times.values()),
            }
        )

    dice_by_method = {
        method.key: {
            row["sample"]: float(row["top_prevalence_dice"])
            for row in score_rows
            if row["method"] == method.key
        }
        for method in methods
    }
    spline_time = statistics.median(per_volume_times["splineops_projection"].values())
    comparison_rows: list[dict[str, Any]] = []
    comparison_methods = [method for method in methods if method.comparison]
    shared_bootstrap_indices = bootstrap_indices(
        len(sample_ids), resamples=bootstrap_resamples, seed=BOOTSTRAP_SEED
    )
    for method in comparison_methods:
        differences = np.asarray(
            [
                dice_by_method["splineops_projection"][sample]
                - dice_by_method[method.key][sample]
                for sample in sorted(dice_by_method[method.key])
            ]
        )
        distribution = bootstrap_distribution(
            differences,
            shared_bootstrap_indices,
        )
        ci_low, ci_high = np.quantile(distribution, (0.025, 0.975))
        familywise_low = np.quantile(distribution, BONFERRONI_LOWER_QUANTILE)
        runtime_ratio = (
            statistics.median(per_volume_times[method.key].values()) / spline_time
        )
        required_speedup = MINIMUM_SPEEDUPS.get(method.key)
        comparison_rows.append(
            {
                "comparison": method.key,
                "mean_dice_difference": float(np.mean(differences)),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "bonferroni_one_sided_low": float(familywise_low),
                "required_quality_margin": QUALITY_MARGIN,
                "quality_margin_pass": bool(familywise_low > QUALITY_MARGIN),
                "median_runtime_ratio": runtime_ratio,
                "required_runtime_ratio": required_speedup,
                "runtime_ratio_strict": method.key in STRICT_SPEEDUPS,
                "speed_pass": speed_condition_pass(method.key, runtime_ratio),
            }
        )

    confirmation_eligible = tuple(sample_ids) == CONFIRMATION_IDS
    quality_pass = all(row["quality_margin_pass"] for row in comparison_rows)
    speed_pass = all(
        row["speed_pass"]
        for row in comparison_rows
        if row["required_runtime_ratio"] is not None
    )
    strict_pass = confirmation_eligible and quality_pass and speed_pass

    output_dir.mkdir(parents=True, exist_ok=True)
    common.write_csv(
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
            "source_depth",
            "normalization_low",
            "normalization_high",
        ),
    )
    common.write_csv(
        output_dir / "timings.csv",
        timing_rows,
        ("sample", "method", "repetition", "runtime_s"),
    )
    common.write_csv(
        output_dir / "summary.csv",
        summary_rows,
        (
            "method",
            "label",
            "mean_top_prevalence_dice",
            "mean_average_precision",
            "mean_roc_auc",
            "median_runtime_s",
        ),
    )
    common.write_csv(
        output_dir / "comparisons.csv",
        comparison_rows,
        (
            "comparison",
            "mean_dice_difference",
            "ci95_low",
            "ci95_high",
            "bonferroni_one_sided_low",
            "required_quality_margin",
            "quality_margin_pass",
            "median_runtime_ratio",
            "required_runtime_ratio",
            "runtime_ratio_strict",
            "speed_pass",
        ),
    )
    make_figure(score_rows, summary_rows, methods, output_dir / "minivess_overview.png")

    used_relatives = [
        relative
        for sample_id in sample_ids
        for relative in (
            f"raw/mv{sample_id:02d}.nii.gz",
            f"seg/mv{sample_id:02d}_y.nii.gz",
        )
    ]
    result = {
        "study": "MiniVess 8x vessel-overview confirmation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "id": DATASET_ID,
            "doi": DATASET_DOI,
            "dataset_page": DATASET_PAGE,
            "paper_page": PAPER_PAGE,
            "code_page": CODE_PAGE,
            "sample_set": sample_set,
            "pilot_sample_ids": list(PILOT_IDS),
            "confirmation_sample_ids": list(CONFIRMATION_IDS),
            "evaluated_sample_ids": list(sample_ids),
            "source_xy": list(EXPECTED_XY),
            "output_xy": list(OUTPUT_XY),
            "license": "CC BY-NC-SA 4.0",
            "source_files_redistributed": False,
            "specimen_grouping_available": False,
            "archive_objects": {
                relative: {
                    "bytes": int(manifest[relative]["bytes"]),
                    "md5": str(manifest[relative]["hash"]),
                }
                for relative in used_relatives
            },
        },
        "protocol": {
            "primary_metric": "top-prevalence vessel Dice",
            "required_quality_margin": QUALITY_MARGIN,
            "superiority_one_sided_quantile": BONFERRONI_LOWER_QUANTILE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": bootstrap_resamples,
            "bootstrap_resamples_shared_across_comparisons": True,
            "minimum_speedups": MINIMUM_SPEEDUPS,
            "strict_speedups": sorted(STRICT_SPEEDUPS),
            "timing_warmups": warmups,
            "timing_repetitions": repetitions,
            "threads": 1,
            "timing_cache_state": "warm process-local public-call caches enabled",
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
            "confirmation_eligible": confirmation_eligible,
            "quality_margin_pass": quality_pass,
            "speed_pass": speed_pass,
            "strict_vessel_overview_claim_pass": strict_pass,
            "segmentation_superiority_demonstrated": False,
            "broad_resampling_superiority_demonstrated": False,
            "fastest_method_claimed": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "thread_controls": {
                **{name: os.environ[name] for name in THREAD_ENVIRONMENT},
                "opencv_api_threads": 1,
                "torch_api_threads": 1,
            },
            "packages": {
                name: common.package_version(name)
                for name in (
                    "splineops",
                    "numpy",
                    "scipy",
                    "scikit-image",
                    "torch",
                    "opencv-python-headless",
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
        help="verified directory containing raw/ and seg/; otherwise download",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "splineops" / "minivess",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/minivess"))
    parser.add_argument(
        "--sample-set", choices=("pilot", "confirmation"), default="confirmation"
    )
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repetitions", type=int, default=3)
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
    sample_ids = PILOT_IDS if args.sample_set == "pilot" else CONFIRMATION_IDS
    root, manifest = prepare_files(args.data_dir, args.cache_dir, sample_ids)
    result = run_study(
        root,
        manifest,
        args.output_dir,
        sample_ids=sample_ids,
        sample_set=args.sample_set,
        warmups=args.timing_warmups,
        repetitions=args.timing_repetitions,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
