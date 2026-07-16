from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_selma3d_nuclei as nuclei_script
from scripts import benchmark_selma3d_vessels as vessel_script

ROOT = Path(__file__).resolve().parents[1]
VESSEL_DIR = ROOT / "benchmarks" / "selma3d-vessels"
NUCLEI_DIR = ROOT / "benchmarks" / "selma3d"


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_vessel_confirmation_passes_only_its_narrow_claim() -> None:
    result = read_json(VESSEL_DIR / "results.json")
    decision = result["decision"]
    assert decision == {
        "broad_resampling_superiority_demonstrated": False,
        "narrow_quality_at_speed_claim_pass": True,
        "quality_noninferiority_pass": True,
        "segmentation_superiority_demonstrated": False,
        "speed_pass": True,
        "strict_named_method_superiority_pass": True,
    }
    dataset = result["dataset"]
    assert dataset["pilot_sample_ids"] == list(range(6))
    assert dataset["confirmation_sample_ids"] == list(range(6, 24))
    assert dataset["specimen_grouping_available"] is False
    assert dataset["source_files_redistributed"] is False


def test_vessel_scores_and_comparisons_are_internally_consistent() -> None:
    scores = read_csv(VESSEL_DIR / "scores.csv")
    comparisons = read_csv(VESSEL_DIR / "comparisons.csv")
    assert len(scores) == 18 * 5
    assert {row["comparison"] for row in comparisons} == {
        "scipy_gaussian",
        "skimage_resize",
        "torch_area",
    }

    by_method = {
        method: {
            row["sample"]: float(row["roc_auc"])
            for row in scores
            if row["method"] == method
        }
        for method in {row["method"] for row in scores}
    }
    for row in comparisons:
        comparison = row["comparison"]
        differences = [
            by_method["splineops_projection"][sample] - by_method[comparison][sample]
            for sample in sorted(by_method[comparison])
        ]
        assert float(row["mean_auc_difference"]) == pytest.approx(
            statistics.fmean(differences), abs=1e-15
        )
        assert float(row["bonferroni_one_sided_low"]) > 0.0
        assert row["quality_superior"] == "True"
        assert row["speed_pass"] == "True"


def test_vessel_runtime_ratios_recompute_from_patch_medians() -> None:
    timings = read_csv(VESSEL_DIR / "timings.csv")
    comparisons = read_csv(VESSEL_DIR / "comparisons.csv")

    def study_median(method: str) -> float:
        patches = sorted({row["sample"] for row in timings if row["method"] == method})
        patch_medians = [
            statistics.median(
                float(row["runtime_s"])
                for row in timings
                if row["method"] == method and row["sample"] == patch
            )
            for patch in patches
        ]
        return statistics.median(patch_medians)

    spline_time = study_median("splineops_projection")
    for row in comparisons:
        ratio = study_median(row["comparison"]) / spline_time
        assert float(row["median_runtime_ratio"]) == pytest.approx(ratio)


def test_vessel_protocol_runner_and_artifacts_preserve_provenance() -> None:
    protocol = (VESSEL_DIR / "PROTOCOL.md").read_text(encoding="utf-8")
    readme = (VESSEL_DIR / "README.md").read_text(encoding="utf-8")
    docs = (ROOT / "docs" / "selma3d-vessels-study.rst").read_text(encoding="utf-8")
    assert "before running any resizing method" in protocol
    assert "Bonferroni" in protocol
    assert "CC BY-NC" in protocol
    assert "specimen" in protocol
    assert "not segmentation superiority" in readme
    assert "Do not shorten this" in docs
    assert "One individual patch favoured SciPy" in docs
    assert len(vessel_script.CHECKSUMS) == 36
    assert vessel_script.SAMPLE_IDS == tuple(range(6, 24))
    assert (VESSEL_DIR / "selma3d_vessels.png").stat().st_size > 50_000
    assert not list(VESSEL_DIR.glob("*.nii*"))
    assert not list(VESSEL_DIR.glob("*.gz"))


def test_nuclei_confirmation_retains_failed_speed_decision() -> None:
    result = read_json(NUCLEI_DIR / "results.json")
    decision = result["decision"]
    assert decision["quality_noninferiority_pass"] is True
    assert decision["speed_pass"] is False
    assert decision["narrow_quality_at_speed_claim_pass"] is False
    assert decision["segmentation_superiority_demonstrated"] is False
    assert len(nuclei_script.CHECKSUMS) == 24
    assert (NUCLEI_DIR / "selma3d_nuclei.png").stat().st_size > 50_000


def test_ranking_metrics_handle_perfect_reverse_and_ties() -> None:
    labels = np.asarray([False, False, True, True])
    perfect = nuclei_script.compute_scores(np.asarray([0.0, 1.0, 2.0, 3.0]), labels)
    reverse = nuclei_script.compute_scores(np.asarray([3.0, 2.0, 1.0, 0.0]), labels)
    tied = nuclei_script.compute_scores(np.ones(4), labels)
    assert perfect.roc_auc == 1.0
    assert perfect.average_precision == 1.0
    assert perfect.top_prevalence_dice == 1.0
    assert reverse.roc_auc == 0.0
    assert tied.roc_auc == 0.5
    assert tied.average_precision == 0.5
    assert tied.top_prevalence_dice == 0.5
