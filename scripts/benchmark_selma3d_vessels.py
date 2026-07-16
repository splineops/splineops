#!/usr/bin/env python3
"""Run the frozen SELMA3D microvessel quality-at-speed confirmation."""

from __future__ import annotations

import argparse
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
from scipy import ndimage

try:
    from scripts import benchmark_selma3d_nuclei as common
except ModuleNotFoundError:  # Executed directly from the scripts directory.
    import benchmark_selma3d_nuclei as common  # type: ignore[no-redef]

ACCESSION = "S-BIAD1196"
DATASET_PAGE = common.DATASET_PAGE
CHALLENGE_PAGE = common.CHALLENGE_PAGE
BASE_URL = (
    "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/196/S-BIAD1196/Files/"
    "SELMA3D_training_annotated/VessAP_vessel"
)
EXPECTED_SHAPE = (500, 500, 50)
OUTPUT_SHAPE = (250, 250, 50)
SAMPLE_IDS = tuple(range(6, 24))
BOOTSTRAP_SEED = 20260716
BOOTSTRAP_RESAMPLES = 20_000
NONINFERIORITY_MARGIN = 0.01
MINIMUM_CPU_SPEEDUP = 10.0
MINIMUM_TORCH_SPEEDUP = 1.0
BONFERRONI_LOWER_QUANTILE = 0.05 / 3.0
PILOT_AUC = {
    "splineops_projection": 0.9216086422123885,
    "scipy_gaussian": 0.9204433549034956,
    "skimage_resize": 0.9195062241983747,
    "torch_area": 0.9193569129087099,
}

CHECKSUMS = {
    "gt/patchvolume_006.nii.gz": "fe7f87450aad5eb1354541c443f1bf66ff2a6d44dcef13a28f86f348b489a270",
    "gt/patchvolume_007.nii.gz": "3bd73d006aaac5c24997513902feeccb98380fc9518fe6c4ea103fe48c23c5e9",
    "gt/patchvolume_008.nii.gz": "47bdc8a93d3e3a6b9966c9e9889367a61be44393fdc854aacaf8f11e637d3b87",
    "gt/patchvolume_009.nii.gz": "3aed4ad9b1b265287a4a0e2c6ed6f50da751ee879221cadf3b338eb2f88d22e6",
    "gt/patchvolume_010.nii.gz": "c005b0af10f047a33d56ee5e78974650b248346105e107d856c702fc2bc0d205",
    "gt/patchvolume_011.nii.gz": "ab5f702d549bf1aecb9305417fc4525d05a6b052b7965f55ffa4fcb5ea350d04",
    "gt/patchvolume_012.nii.gz": "6bd5eed2d2cb06ebe73f248eb8c21764b4f1e7906790d97610af58918ed968ef",
    "gt/patchvolume_013.nii.gz": "019069f8d37c721eca8aeab93641ae2057d1dce79c0792f769d2b3358813e4ed",
    "gt/patchvolume_014.nii.gz": "a3eaa498398d5d2f29e45f3820581e35c170ec5efe2b13dfc8fe31adcfcb08eb",
    "gt/patchvolume_015.nii.gz": "1bf501dba8ff5b5fde802983533e61392273c9dd296684a164afba909b3f67e8",
    "gt/patchvolume_016.nii.gz": "12695b31af9a17d1647437df933f82819ed8d16c30fbacad771397df63a6a76c",
    "gt/patchvolume_017.nii.gz": "5763227a220cbb45465dc90b9c534094b38b0f149d78c59bed4be7714efc44cf",
    "gt/patchvolume_018.nii.gz": "1a23b4393965fb6d15230465aa04616c7787133802ef4f7067d4e9a4f86608e6",
    "gt/patchvolume_019.nii.gz": "7036a3594ed8c539482fb17110788ddf3b75383125ea2f3ee693909761a0d611",
    "gt/patchvolume_020.nii.gz": "df6b8fe68d92982c0f88f6be1b5d715bc2c0e96cd715e67a2368f836a62d5301",
    "gt/patchvolume_021.nii.gz": "49460f4e4103ff03f904befaa5e517eee185755f5116c638917125c44add5cd4",
    "gt/patchvolume_022.nii.gz": "1fc3f5c071f6d723b66d46812254735d87a3b2a2ca0600a6e38656041f85d172",
    "gt/patchvolume_023.nii.gz": "4da2c1e827a1e156aa5ca6b22cd2f1d45ad2e54c288bea7649013d30b8138692",
    "raw/patchvolume_006_0000.nii.gz": "b163136d154d3355f56fd96ef30505de48b70a446c3ce6ccb7207ea85989d78b",
    "raw/patchvolume_007_0000.nii.gz": "e7f559ab3cb9881cae2d3027307a7a854669dd7525b43bc1aa00c5133a25b660",
    "raw/patchvolume_008_0000.nii.gz": "35e75de1734aa79288e26cee92b9730f65da948e8002dc2253c37dbee0d12c1b",
    "raw/patchvolume_009_0000.nii.gz": "baae9bb83d10cf5245bf4897f683ad319e621c75591db24d737f9666cdb37959",
    "raw/patchvolume_010_0000.nii.gz": "c0ced366d5e2131fdbe35a7933a5904765b79bde1945ce45a876fea3d9fe8bf8",
    "raw/patchvolume_011_0000.nii.gz": "e4a814cb073eb509c7d40ebefdd94a59b33cafcf178b0eca8df8e9e60c2da2f2",
    "raw/patchvolume_012_0000.nii.gz": "505680523ee5ae51d726c2bc200efdf465642a063ef969991320fe1a139eb42a",
    "raw/patchvolume_013_0000.nii.gz": "f0c53ba36f2bfadff0458d2a93ae1a23545a1b48dd8c10d026a83c48e5e74843",
    "raw/patchvolume_014_0000.nii.gz": "f16682d3fe62b43684716a7fb4544fd59155ea058aede6484a1cd1fbdf60759b",
    "raw/patchvolume_015_0000.nii.gz": "ee054dca06539d89b1b5715df69186fec9da7d2fd781b509db5ad5604717047f",
    "raw/patchvolume_016_0000.nii.gz": "6dc5bacb5e72382f9bf7d2409c1c75632dd1bc1a99a194b5f023c89feb4610e8",
    "raw/patchvolume_017_0000.nii.gz": "2153a6709d2fcd458c08e7db6694808d3237e9ba653782e6ba1075183f29321c",
    "raw/patchvolume_018_0000.nii.gz": "d12c92b441af26f6a2f89e784520fbc2ca07036dbf0b2a2d93ae60c6db66b443",
    "raw/patchvolume_019_0000.nii.gz": "ee4bbee8939913ac3e8329e1969a019befa089b34ff8dc1f33c06b5acd9b622d",
    "raw/patchvolume_020_0000.nii.gz": "3fa3afb069715406ff18c57256290357b601eb649325357b1021fbb753583fb7",
    "raw/patchvolume_021_0000.nii.gz": "0b6b67926698e11700a1ce91307c8a0fe1ea437a55c10a4b1d276522858783ff",
    "raw/patchvolume_022_0000.nii.gz": "103fc18bbbbb4097230bb93e66b1df548c38229a0d57cd406ecd799ded0d0857",
    "raw/patchvolume_023_0000.nii.gz": "ac2bfa0b47f65db4ef8f859efcad1cceb04da64c2a6b3c3edb0492b898dd6da6",
}

VolumeMethod = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    semantics: str
    grid: str
    comparison: bool
    call: VolumeMethod


def fetch_file(relative: str, cache_dir: Path) -> Path:
    expected = CHECKSUMS[relative]
    destination = cache_dir / "VessAP_vessel" / relative
    if destination.exists():
        return common.verified_file(destination, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"{BASE_URL}/{relative}",
        headers={"User-Agent": "SplineOps SELMA3D reproducibility study"},
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
        common.verified_file(temporary, expected)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def data_root(data_dir: Path | None, cache_dir: Path) -> Path:
    if data_dir is None:
        for relative in CHECKSUMS:
            fetch_file(relative, cache_dir)
        return cache_dir / "VessAP_vessel"
    candidate = data_dir / "VessAP_vessel"
    root = candidate if candidate.is_dir() else data_dir
    for relative, expected in CHECKSUMS.items():
        common.verified_file(root / relative, expected)
    return root


def read_sample(root: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    nibabel = common.require_nibabel()
    image_path = root / "raw" / f"patchvolume_{sample_id:03d}_0000.nii.gz"
    mask_path = root / "gt" / f"patchvolume_{sample_id:03d}.nii.gz"
    image = np.squeeze(np.asarray(nibabel.load(image_path).dataobj, dtype=np.float32))
    labels = np.squeeze(np.asarray(nibabel.load(mask_path).dataobj)) > 0
    if image.shape != EXPECTED_SHAPE or labels.shape != EXPECTED_SHAPE:
        raise RuntimeError(
            f"sample {sample_id:03d} has shapes {image.shape} and {labels.shape}"
        )
    if not np.any(labels) or np.all(labels):
        raise RuntimeError(f"sample {sample_id:03d} has a degenerate mask")
    return image, labels


def resize_splineops(image: np.ndarray, *, antialias: bool) -> np.ndarray:
    from splineops import resize

    method = "cubic-antialiasing" if antialias else "cubic"
    return np.asarray(
        resize(image, output_size=(250, 250), axes=(0, 1), method=method),
        dtype=np.float32,
    )


def resize_scipy_gaussian(image: np.ndarray) -> np.ndarray:
    factors = ((500 - 1) / (250 - 1), (500 - 1) / (250 - 1), 1.0)
    filtered = ndimage.gaussian_filter(
        image,
        sigma=tuple(max(0.0, (factor - 1.0) / 2.0) for factor in factors),
        mode="reflect",
    )
    coordinates = (
        np.linspace(0.0, 499.0, 250),
        np.linspace(0.0, 499.0, 250),
        np.arange(50, dtype=np.float64),
    )
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
        raise RuntimeError("Install `splineops[selma3d-study]`.") from exc
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
        raise RuntimeError("Install `splineops[selma3d-study]`.") from exc
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
    ]


def timed_calls(
    method: Method, image: np.ndarray, *, warmups: int, repetitions: int
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


def bootstrap_distribution(
    differences: np.ndarray, *, resamples: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    return differences[indices].mean(axis=1)


def make_figure(
    score_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    methods: list[Method],
    destination: Path,
) -> None:
    import matplotlib.pyplot as plt

    keys = [method.key for method in methods]
    labels = {method.key: method.label for method in methods}
    colors = {
        "splineops_projection": "#1877b8",
        "splineops_interpolation": "#83b8d9",
        "scipy_gaussian": "#e58b2b",
        "skimage_resize": "#4a9d63",
        "torch_area": "#9067b0",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    for position, key in enumerate(keys):
        values = [row["roc_auc"] for row in score_rows if row["method"] == key]
        axes[0].scatter(
            np.full(len(values), position),
            values,
            s=23,
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
    axes[0].set_ylabel("Microvessel voxel-ranking ROC AUC")
    axes[0].set_ylim(0.5, 1.005)
    axes[0].set_xticks(
        range(len(keys)), [labels[key] for key in keys], rotation=25, ha="right"
    )
    axes[0].set_title("18 held-out vessel patches")
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
    axes[1].set_title("500×500×50 → 250×250×50, one CPU thread")
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
        image, low, high = common.normalize_image(source)
        targets = {
            grid: common.sample_labels(labels, OUTPUT_SHAPE, grid=grid)
            for grid in {method.grid for method in methods}
        }
        for method in methods:
            output = method.call(image)
            scores = common.compute_scores(output, targets[method.grid])
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
                method, image, warmups=warmups, repetitions=repetitions
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

    per_patch_times: dict[str, dict[str, float]] = {}
    summary_rows: list[dict[str, Any]] = []
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
    spline_time = statistics.median(per_patch_times["splineops_projection"].values())
    comparison_rows: list[dict[str, Any]] = []
    for comparison_index, method in enumerate(
        [method for method in methods if method.comparison]
    ):
        differences = np.asarray(
            [
                score_by_method["splineops_projection"][sample]
                - score_by_method[method.key][sample]
                for sample in sorted(score_by_method[method.key])
            ]
        )
        distribution = bootstrap_distribution(
            differences,
            resamples=bootstrap_resamples,
            seed=BOOTSTRAP_SEED + comparison_index,
        )
        low, high = np.quantile(distribution, (0.025, 0.975))
        superiority_low = np.quantile(distribution, BONFERRONI_LOWER_QUANTILE)
        speedup = statistics.median(per_patch_times[method.key].values()) / spline_time
        required_speedup = (
            MINIMUM_TORCH_SPEEDUP if method.key == "torch_area" else MINIMUM_CPU_SPEEDUP
        )
        comparison_rows.append(
            {
                "comparison": method.key,
                "mean_auc_difference": float(np.mean(differences)),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bonferroni_one_sided_low": float(superiority_low),
                "noninferior_margin": NONINFERIORITY_MARGIN,
                "noninferior": bool(low > -NONINFERIORITY_MARGIN),
                "quality_superior": bool(superiority_low > 0.0),
                "median_runtime_ratio": speedup,
                "required_runtime_ratio": required_speedup,
                "speed_pass": bool(speedup > required_speedup),
            }
        )

    quality_noninferiority_pass = all(row["noninferior"] for row in comparison_rows)
    speed_pass = all(row["speed_pass"] for row in comparison_rows)
    strict_superiority_pass = speed_pass and all(
        row["quality_superior"] for row in comparison_rows
    )
    narrow_quality_at_speed_pass = quality_noninferiority_pass and speed_pass

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
            "mean_roc_auc",
            "mean_average_precision",
            "mean_top_prevalence_dice",
            "median_runtime_s",
        ),
    )
    common.write_csv(
        output_dir / "comparisons.csv",
        comparison_rows,
        (
            "comparison",
            "mean_auc_difference",
            "ci95_low",
            "ci95_high",
            "bonferroni_one_sided_low",
            "noninferior_margin",
            "noninferior",
            "quality_superior",
            "median_runtime_ratio",
            "required_runtime_ratio",
            "speed_pass",
        ),
    )
    make_figure(
        score_rows,
        summary_rows,
        methods,
        output_dir / "selma3d_vessels.png",
    )

    result = {
        "study": "SELMA3D microvessel quality-at-speed confirmation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "accession": ACCESSION,
            "dataset_page": DATASET_PAGE,
            "challenge_page": CHALLENGE_PAGE,
            "subset": "VessAP_vessel",
            "channel": "_0000 WGA microvessels",
            "pilot_sample_ids": list(range(6)),
            "confirmation_sample_ids": list(SAMPLE_IDS),
            "source_shape": list(EXPECTED_SHAPE),
            "output_shape": list(OUTPUT_SHAPE),
            "license_interpretation": "CC BY-NC (stricter of conflicting official metadata)",
            "source_files_redistributed": False,
            "specimen_grouping_available": False,
        },
        "protocol": {
            "pilot_mean_auc": PILOT_AUC,
            "primary_metric": "microvessel voxel-ranking ROC AUC",
            "noninferiority_margin": NONINFERIORITY_MARGIN,
            "superiority_one_sided_quantile": BONFERRONI_LOWER_QUANTILE,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_resamples": bootstrap_resamples,
            "minimum_cpu_speedup": MINIMUM_CPU_SPEEDUP,
            "minimum_torch_speedup": MINIMUM_TORCH_SPEEDUP,
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
            "quality_noninferiority_pass": quality_noninferiority_pass,
            "speed_pass": speed_pass,
            "narrow_quality_at_speed_claim_pass": narrow_quality_at_speed_pass,
            "strict_named_method_superiority_pass": strict_superiority_pass,
            "segmentation_superiority_demonstrated": False,
            "broad_resampling_superiority_demonstrated": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": {
                name: common.package_version(name)
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
        help="verified VessAP_vessel directory (or its parent); otherwise download",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "splineops" / "selma3d",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/selma3d-vessels"),
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
    print(json.dumps(result["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
