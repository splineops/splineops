#!/usr/bin/env python3
"""Interactively compare SELMA3D vessel downsampling methods in napari.

The demo downloads one checksum-pinned held-out patch by default. It reproduces
the preprocessing, methods, grids, and ranking metric from the frozen SELMA3D
microvessel confirmation and carries only its non-image aggregate summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import statistics
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Callable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from scipy import ndimage, stats

ACCESSION = "S-BIAD1196"
BASE_URL = (
    "https://ftp.ebi.ac.uk/biostudies/fire/S-BIAD/196/S-BIAD1196/Files/"
    "SELMA3D_training_annotated/VessAP_vessel"
)
EXPECTED_SHAPE = (500, 500, 50)
OUTPUT_SHAPE = (250, 250, 50)
SAMPLE_IDS = tuple(range(6, 24))
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "splineops" / "selma3d"
DEFAULT_SAMPLE_ID = 13
ANCHOR_KEY = "splineops_projection"
DEFAULT_COMPARISONS = (
    "skimage_resize",
    "scipy_gaussian",
    "splineops_interpolation",
)
OPTIONAL_COMPARISONS = ("torch_area",)
COMPARISON_KEYS = (*DEFAULT_COMPARISONS, *OPTIONAL_COMPARISONS)
METHOD_KEYS = (ANCHOR_KEY, *COMPARISON_KEYS)
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


@dataclass(frozen=True)
class Scores:
    roc_auc: float
    average_precision: float
    top_prevalence_dice: float


@dataclass(frozen=True)
class MethodResult:
    """One method's output and locally measured evidence for a single patch."""

    key: str
    label: str
    semantics: str
    grid: str
    output: np.ndarray
    target: np.ndarray
    roc_auc: float
    average_precision: float
    top_prevalence_dice: float
    runtime_s: float


@dataclass(frozen=True)
class PublishedResult:
    """Frozen aggregate evidence loaded from the repository artifact."""

    mean_roc_auc: float
    median_runtime_s: float


@dataclass(frozen=True)
class DemoRun:
    """All inputs and evidence required by the console and napari views."""

    sample_id: int
    normalization_low: float
    normalization_high: float
    warmups: int
    repetitions: int
    methods: tuple[MethodResult, ...]
    published: dict[str, PublishedResult]
    published_speedups: dict[str, float]

    def result(self, key: str) -> MethodResult:
        return next(result for result in self.methods if result.key == key)


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
    destination = cache_dir / "VessAP_vessel" / relative
    if destination.exists():
        return verified_file(destination, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"{BASE_URL}/{relative}",
        headers={"User-Agent": "SplineOps SELMA3D napari demo"},
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
        verified_file(temporary, expected)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def require_nibabel() -> Any:
    try:
        import nibabel
    except ImportError as exc:
        raise RuntimeError(
            "Install the demo with `python -m pip install 'splineops[selma3d-demo]'`."
        ) from exc
    return nibabel


def read_sample(root: Path, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    nibabel = require_nibabel()
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


def normalize_image(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    low, high = np.percentile(image, (1.0, 99.9))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("image has no usable percentile intensity range")
    normalized = np.clip((image - low) / (high - low), 0.0, 1.0)
    return np.asarray(normalized, dtype=np.float32), float(low), float(high)


def nearest_indices(source: int, target: int, *, grid: str) -> np.ndarray:
    if grid == "endpoint":
        coordinates = np.linspace(0.0, float(source - 1), target)
    elif grid == "half_pixel":
        coordinates = (np.arange(target, dtype=np.float64) + 0.5) * (
            source / target
        ) - 0.5
    else:
        raise ValueError(f"unknown grid {grid!r}")
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
        raise RuntimeError(
            "Install the demo with `python -m pip install 'splineops[selma3d-demo]'`."
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
            "Install the PyTorch comparison with "
            "`python -m pip install 'splineops[selma3d-demo-all]'`."
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
    return true_positive / positives


def compute_scores(values: np.ndarray, labels: np.ndarray) -> Scores:
    if values.shape != labels.shape:
        raise ValueError(f"score/label shape mismatch: {values.shape}, {labels.shape}")
    return Scores(
        roc_auc=roc_auc(values, labels),
        average_precision=average_precision(values, labels),
        top_prevalence_dice=top_prevalence_dice(values, labels),
    )


def _relative_paths(sample_id: int) -> tuple[str, str]:
    return (
        f"raw/patchvolume_{sample_id:03d}_0000.nii.gz",
        f"gt/patchvolume_{sample_id:03d}.nii.gz",
    )


def validate_sample_id(sample_id: int) -> None:
    """Reject samples outside the held-out, checksum-pinned confirmation set."""

    if sample_id not in SAMPLE_IDS:
        allowed = f"{min(SAMPLE_IDS)} through {max(SAMPLE_IDS)}"
        raise ValueError(f"sample must be a held-out patch ID from {allowed}")


def resolve_sample_root(
    sample_id: int, *, data_dir: Path | None, cache_dir: Path
) -> Path:
    """Fetch or verify only the two files needed by the selected sample."""

    validate_sample_id(sample_id)
    relatives = _relative_paths(sample_id)
    if data_dir is None:
        for relative in relatives:
            fetch_file(relative, cache_dir)
        return cache_dir / "VessAP_vessel"

    candidate = data_dir / "VessAP_vessel"
    root = candidate if candidate.is_dir() else data_dir
    for relative in relatives:
        verified_file(root / relative, CHECKSUMS[relative])
    return root


def build_methods(keys: Sequence[str]) -> list[Method]:
    """Construct only requested methods so optional imports stay genuinely optional."""

    unknown = set(keys).difference(METHOD_KEYS)
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(sorted(unknown))}")

    methods: list[Method] = []
    for key in keys:
        if key == "splineops_projection":
            call = lambda image: resize_splineops(image, antialias=True)
            label = "SplineOps projection AA"
            semantics = "endpoint-aligned cubic projection antialiasing"
            grid = "endpoint"
        elif key == "splineops_interpolation":
            call = lambda image: resize_splineops(image, antialias=False)
            label = "SplineOps cubic, no AA"
            semantics = "endpoint-aligned cubic interpolation without antialiasing"
            grid = "endpoint"
        elif key == "scipy_gaussian":
            call = resize_scipy_gaussian
            label = "SciPy Gaussian + cubic"
            semantics = "Gaussian-prefiltered endpoint-aligned cubic sampling"
            grid = "endpoint"
        elif key == "skimage_resize":
            call = resize_skimage
            label = "scikit-image cubic AA"
            semantics = "native half-pixel cubic resize with Gaussian antialiasing"
            grid = "half_pixel"
        else:
            call = torch_area_method()
            label = "PyTorch area"
            semantics = "native regional area resize on half-pixel geometry"
            grid = "half_pixel"
        methods.append(Method(key, label, semantics, grid, key != ANCHOR_KEY, call))
    return methods


def timed_output(
    method: Method,
    image: np.ndarray,
    *,
    warmups: int,
    repetitions: int,
) -> tuple[np.ndarray, float]:
    """Return the final output and median wall time, including each method call."""

    output: np.ndarray | None = None
    for _ in range(warmups):
        output = np.asarray(method.call(image), dtype=np.float32)

    durations: list[float] = []
    for _ in range(repetitions):
        start = time.perf_counter()
        output = np.asarray(method.call(image), dtype=np.float32)
        durations.append(time.perf_counter() - start)

    if output is None or output.shape != OUTPUT_SHAPE:
        shape = None if output is None else output.shape
        raise RuntimeError(f"{method.key} returned unexpected shape {shape}")
    if not np.all(np.isfinite(output)):
        raise RuntimeError(f"{method.key} returned non-finite values")
    return output, statistics.median(durations)


def load_published_results(
    path: Path | None = None,
) -> tuple[dict[str, PublishedResult], dict[str, float]]:
    """Load full-precision frozen aggregate metrics used in the viewer panel."""

    if path is None:
        resource = files("splineops.demos.data").joinpath(
            "selma3d_vessels_summary.json"
        )
        with resource.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
    else:
        with path.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
    published = {
        row["method"]: PublishedResult(
            mean_roc_auc=float(row["mean_roc_auc"]),
            median_runtime_s=float(row["median_runtime_s"]),
        )
        for row in artifact["summary"]
    }
    speedups = {
        row["comparison"]: float(row["median_runtime_ratio"])
        for row in artifact["comparisons"]
    }
    return published, speedups


def prepare_demo(
    sample_id: int,
    *,
    comparisons: Sequence[str] = DEFAULT_COMPARISONS,
    data_dir: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    warmups: int = 1,
    repetitions: int = 3,
    results_path: Path | None = None,
) -> DemoRun:
    """Load one patch, execute selected methods, and compute transparent evidence."""

    if warmups < 0 or repetitions < 1:
        raise ValueError(
            "timings require non-negative warmups and positive repetitions"
        )
    if ANCHOR_KEY in comparisons:
        raise ValueError("comparisons must not repeat the SplineOps anchor method")
    if len(set(comparisons)) != len(comparisons):
        raise ValueError("comparisons must not contain duplicates")

    root = resolve_sample_root(sample_id, data_dir=data_dir, cache_dir=cache_dir)
    source, labels = read_sample(root, sample_id)
    image, low, high = normalize_image(source)
    requested = (ANCHOR_KEY, *comparisons)

    results: list[MethodResult] = []
    for method in build_methods(requested):
        print(f"Running {method.label}...", flush=True)
        output, runtime_s = timed_output(
            method, image, warmups=warmups, repetitions=repetitions
        )
        target = sample_labels(labels, OUTPUT_SHAPE, grid=method.grid)
        scores = compute_scores(output, target)
        results.append(
            MethodResult(
                key=method.key,
                label=method.label,
                semantics=method.semantics,
                grid=method.grid,
                output=output,
                target=target,
                roc_auc=scores.roc_auc,
                average_precision=scores.average_precision,
                top_prevalence_dice=scores.top_prevalence_dice,
                runtime_s=runtime_s,
            )
        )
        print(
            f"  ROC AUC {scores.roc_auc:.6f}; local median "
            f"{runtime_s * 1000.0:.1f} ms",
            flush=True,
        )

    published, speedups = load_published_results(results_path)
    return DemoRun(
        sample_id=sample_id,
        normalization_low=low,
        normalization_high=high,
        warmups=warmups,
        repetitions=repetitions,
        methods=tuple(results),
        published=published,
        published_speedups=speedups,
    )


def napari_order(array: np.ndarray) -> np.ndarray:
    """Convert SELMA3D's (x, y, z) storage to napari's (z, y, x) display order."""

    if array.ndim != 3:
        raise ValueError(f"expected a 3-D array, received shape {array.shape}")
    return np.transpose(array, (2, 1, 0))


def curtain(
    anchor: np.ndarray, comparison: np.ndarray, fraction: float
) -> tuple[np.ndarray, int]:
    """Reveal ``anchor`` from the left and keep ``comparison`` on the right."""

    if anchor.shape != comparison.shape:
        raise ValueError("curtain inputs must have identical shapes")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("curtain fraction must lie between zero and one")
    split = int(round(anchor.shape[-1] * fraction))
    output = np.array(comparison, copy=True)
    output[..., :split] = anchor[..., :split]
    return output, split


def console_summary(run: DemoRun) -> str:
    """Format local single-patch measurements without implying study-level proof."""

    anchor = run.result(ANCHOR_KEY)
    rows = [
        f"SELMA3D patch {run.sample_id:03d} — local measurements",
        f"{'method':29s} {'ROC AUC':>10s} {'median ms':>11s} {'vs SplineOps':>13s}",
    ]
    for result in run.methods:
        ratio = result.runtime_s / anchor.runtime_s
        rows.append(
            f"{result.label:29.29s} {result.roc_auc:10.6f} "
            f"{result.runtime_s * 1000.0:11.1f} {ratio:12.2f}x"
        )
    rows.extend(
        (
            "",
            f"Timing: {run.warmups} warm-up(s), {run.repetitions} measured call(s), "
            "one CPU thread.",
            "Metric: expert vessel/background voxel-ranking ROC AUC; not segmentation.",
        )
    )
    return "\n".join(rows)


def metrics_html(run: DemoRun, comparison_key: str) -> str:
    """Build the dock panel text for the currently selected comparison."""

    anchor = run.result(ANCHOR_KEY)
    comparison = run.result(comparison_key)
    local_delta = anchor.roc_auc - comparison.roc_auc
    local_speedup = comparison.runtime_s / anchor.runtime_s
    anchor_published = run.published[ANCHOR_KEY]
    comparison_published = run.published[comparison_key]
    published_delta = anchor_published.mean_roc_auc - comparison_published.mean_roc_auc
    published_speedup = (
        run.published_speedups.get(comparison_key)
        or comparison_published.median_runtime_s / anchor_published.median_runtime_s
    )
    grid_note = (
        "Both sides use endpoint coordinates."
        if comparison.grid == anchor.grid
        else "Left uses endpoint coordinates; right uses the comparison's half-pixel grid."
    )
    return f"""
    <h3>{comparison.label}</h3>
    <p>{comparison.semantics}<br><i>{grid_note}</i></p>
    <h4>This patch: {run.sample_id:03d}</h4>
    <table cellspacing="4">
      <tr><th></th><th>ROC AUC</th><th>Local median</th></tr>
      <tr><td>SplineOps</td><td>{anchor.roc_auc:.6f}</td><td>{anchor.runtime_s * 1000:.1f} ms</td></tr>
      <tr><td>Comparison</td><td>{comparison.roc_auc:.6f}</td><td>{comparison.runtime_s * 1000:.1f} ms</td></tr>
    </table>
    <p><b>Difference:</b> {local_delta:+.6f} AUC; {local_speedup:.2f}× runtime ratio.<br>
    <small>{run.warmups} warm-up(s), median of {run.repetitions} call(s), one CPU thread.</small></p>
    <h4>Frozen 18-patch result</h4>
    <table cellspacing="4">
      <tr><th></th><th>Mean AUC</th><th>Recorded median</th></tr>
      <tr><td>SplineOps</td><td>{anchor_published.mean_roc_auc:.6f}</td><td>{anchor_published.median_runtime_s * 1000:.0f} ms</td></tr>
      <tr><td>Comparison</td><td>{comparison_published.mean_roc_auc:.6f}</td><td>{comparison_published.median_runtime_s * 1000:.0f} ms</td></tr>
    </table>
    <p><b>Difference:</b> {published_delta:+.6f} mean AUC; {published_speedup:.2f}× recorded runtime ratio.</p>
    <p><small>Positive AUC differences favour SplineOps. ROC AUC measures vessel/background ranking,
    not segmentation. Claims are limited to the named methods, data, geometry, metric, and recorded machine.</small></p>
    """


def create_comparison_widget(
    run: DemoRun,
    image_layer: Any,
    labels_layer: Any,
    divider_layer: Any,
) -> Any:
    """Create the Qt comparison control without importing Qt in headless runs."""

    try:
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QLabel,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:  # pragma: no cover - requires GUI dependency
        raise RuntimeError(
            "Install the demo dependencies with "
            "`python -m pip install 'splineops[selma3d-demo]'`."
        ) from exc

    anchor = run.result(ANCHOR_KEY)
    comparisons = [result for result in run.methods if result.key != ANCHOR_KEY]
    anchor_image = napari_order(anchor.output)
    anchor_target = napari_order(anchor.target.astype(np.uint8))

    widget = QWidget()
    layout = QVBoxLayout(widget)

    title = QLabel("<h2>SplineOps vessel-preservation demo</h2>")
    title.setWordWrap(True)
    layout.addWidget(title)

    method_label = QLabel("Comparison on the right")
    method_box = QComboBox()
    for result in comparisons:
        method_box.addItem(result.label, result.key)
    layout.addWidget(method_label)
    layout.addWidget(method_box)

    reveal_label = QLabel("Reveal SplineOps from the left: 50%")
    reveal_slider = QSlider(Qt.Orientation.Horizontal)
    reveal_slider.setRange(0, 100)
    reveal_slider.setValue(50)
    reveal_slider.setTickInterval(10)
    reveal_slider.setToolTip("0% = comparison only; 100% = SplineOps only")
    layout.addWidget(reveal_label)
    layout.addWidget(reveal_slider)

    mask_box = QCheckBox("Show expert vessel mask")
    mask_box.setChecked(True)
    layout.addWidget(mask_box)

    metrics = QLabel()
    metrics.setWordWrap(True)
    metrics.setTextFormat(Qt.TextFormat.RichText)
    metrics.setMinimumWidth(390)
    layout.addWidget(metrics)
    layout.addStretch(1)

    def update_curtain() -> None:
        comparison_key = str(method_box.currentData())
        comparison = run.result(comparison_key)
        fraction = reveal_slider.value() / 100.0
        comparison_image = napari_order(comparison.output)
        comparison_target = napari_order(comparison.target.astype(np.uint8))
        image_layer.data, split = curtain(anchor_image, comparison_image, fraction)
        labels_layer.data, _ = curtain(anchor_target, comparison_target, fraction)
        divider = np.zeros_like(anchor_image, dtype=np.float32)
        if 0 < split < divider.shape[-1]:
            divider[..., max(0, split - 1) : min(split + 1, divider.shape[-1])] = 1.0
        divider_layer.data = divider
        reveal_label.setText(
            f"Reveal SplineOps from the left: {reveal_slider.value()}%"
        )
        image_layer.name = f"SplineOps (left) | {comparison.label} (right)"
        metrics.setText(metrics_html(run, comparison_key))

    method_box.currentIndexChanged.connect(update_curtain)
    reveal_slider.valueChanged.connect(update_curtain)
    mask_box.toggled.connect(lambda checked: setattr(labels_layer, "visible", checked))
    update_curtain()
    return widget


def launch_napari(run: DemoRun) -> None:
    """Open the comparison layers and evidence panel in a napari event loop."""

    try:
        import napari
    except ImportError as exc:  # pragma: no cover - requires GUI dependency
        raise RuntimeError(
            "Install the demo dependencies with "
            "`python -m pip install 'splineops[selma3d-demo]'`."
        ) from exc

    anchor = run.result(ANCHOR_KEY)
    comparison = next(result for result in run.methods if result.key != ANCHOR_KEY)
    image, split = curtain(
        napari_order(anchor.output), napari_order(comparison.output), 0.5
    )
    labels, _ = curtain(
        napari_order(anchor.target.astype(np.uint8)),
        napari_order(comparison.target.astype(np.uint8)),
        0.5,
    )
    divider = np.zeros_like(image, dtype=np.float32)
    divider[..., split - 1 : split + 1] = 1.0

    viewer = napari.Viewer(title=f"SplineOps — SELMA3D patch {run.sample_id:03d}")
    image_layer = viewer.add_image(
        image,
        name=f"SplineOps (left) | {comparison.label} (right)",
        colormap="gray",
        contrast_limits=(0.0, 1.0),
    )
    labels_layer = viewer.add_labels(
        labels,
        name="Expert vessel mask (grid-aware)",
        opacity=0.32,
    )
    divider_layer = viewer.add_image(
        divider,
        name="Curtain boundary",
        colormap="red",
        contrast_limits=(0.0, 1.0),
        opacity=0.9,
        blending="additive",
    )
    widget = create_comparison_widget(run, image_layer, labels_layer, divider_layer)
    viewer.window.add_dock_widget(widget, name="SplineOps comparison", area="right")

    densest_plane = int(np.argmax(labels.sum(axis=(1, 2))))
    viewer.dims.set_current_step(0, densest_plane)
    viewer.layers.selection.active = image_layer
    napari.run()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE_ID,
        choices=SAMPLE_IDS,
        help="held-out SELMA3D patch ID (default: %(default)s)",
    )
    parser.add_argument(
        "--comparisons",
        nargs="+",
        choices=COMPARISON_KEYS,
        default=list(DEFAULT_COMPARISONS),
        help="methods available in the viewer (default: installed core set)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="verified VessAP_vessel directory (or its parent); otherwise download",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="download cache (default: %(default)s)",
    )
    parser.add_argument("--timing-warmups", type=int, default=1)
    parser.add_argument("--timing-repetitions", type=int, default=3)
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="run downloads, methods, metrics, and timings without opening napari",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        "SELMA3D source: BioImage Archive S-BIAD1196 (VessAP_vessel).\n"
        "License note: official metadata conflict; this demo applies CC BY-NC.\n"
        "Source images are downloaded on demand and are not redistributed.",
        flush=True,
    )
    run = prepare_demo(
        args.sample,
        comparisons=args.comparisons,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        warmups=args.timing_warmups,
        repetitions=args.timing_repetitions,
    )
    print(console_summary(run), flush=True)
    if not args.no_gui:
        launch_napari(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
