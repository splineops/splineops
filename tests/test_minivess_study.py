from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_minivess_overview as study

ROOT = Path(__file__).resolve().parents[1]


def test_pilot_and_confirmation_are_disjoint_and_complete() -> None:
    assert len(study.PILOT_IDS) == 8
    assert len(study.CONFIRMATION_IDS) == 62
    assert set(study.PILOT_IDS).isdisjoint(study.CONFIRMATION_IDS)
    assert set(study.PILOT_IDS) | set(study.CONFIRMATION_IDS) == set(range(1, 71))
    assert study.BONFERRONI_LOWER_QUANTILE == pytest.approx(0.01)
    assert study.QUALITY_MARGIN == pytest.approx(0.005)


def test_native_grid_label_sampling_has_expected_shape() -> None:
    labels = np.zeros((512, 512, 5), dtype=bool)
    labels[::8, ::8] = True
    for grid in ("endpoint", "half_pixel", "phase_zero"):
        sampled = study.sample_labels(labels, grid=grid)
        assert sampled.shape == (64, 64, 5)
        assert sampled.dtype == np.bool_
    assert np.all(study.sample_labels(labels, grid="phase_zero"))


def test_bootstrap_is_deterministic_and_uses_paired_means() -> None:
    differences = np.asarray([0.01, 0.02, 0.03, 0.04])
    first_indices = study.bootstrap_indices(4, resamples=250, seed=19)
    second_indices = study.bootstrap_indices(4, resamples=250, seed=19)
    first = study.bootstrap_distribution(differences, first_indices)
    second = study.bootstrap_distribution(differences, second_indices)
    assert np.array_equal(first_indices, second_indices)
    assert np.array_equal(first, second)
    assert np.all((first >= differences.min()) & (first <= differences.max()))


def test_bootstrap_indices_can_be_shared_across_comparisons() -> None:
    indices = study.bootstrap_indices(3, resamples=50, seed=study.BOOTSTRAP_SEED)
    first = study.bootstrap_distribution(np.asarray([0.1, 0.2, 0.3]), indices)
    second = study.bootstrap_distribution(np.asarray([1.1, 1.2, 1.3]), indices)
    np.testing.assert_allclose(second - first, 1.0)


def test_frozen_runner_forces_all_thread_controls_to_one() -> None:
    assert study.THREAD_ENVIRONMENT == (
        "LSRESIZE_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    assert {name: os.environ[name] for name in study.THREAD_ENVIRONMENT} == {
        name: "1" for name in study.THREAD_ENVIRONMENT
    }


def test_speed_conditions_match_frozen_inclusive_and_strict_rules() -> None:
    for method in ("scipy_gaussian", "skimage_resize", "scipy_polyphase"):
        assert study.speed_condition_pass(method, 3.0) is True
        assert study.speed_condition_pass(method, np.nextafter(3.0, 0.0)) is False
    assert study.speed_condition_pass("torch_area", 1.0) is False
    assert study.speed_condition_pass("torch_area", np.nextafter(1.0, 2.0)) is True
    assert study.speed_condition_pass("opencv_area", 100.0) is None


def test_protocol_freezes_narrow_claim_and_limitations() -> None:
    protocol = (ROOT / "benchmarks" / "minivess" / "PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    normalized = " ".join(protocol.split())
    assert "before running any resizing method" in normalized
    assert "warm fixed-grid" in normalized
    assert "same seeded resample indices" in normalized
    assert "top-prevalence Dice" in normalized
    assert "+0.005 Dice" in normalized
    assert "OpenCV" in normalized
    assert "not demonstrate segmentation superiority" in normalized
    assert "specimen/animal identifier" in normalized


def test_method_registry_includes_strong_and_fast_counterchecks() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    methods = study.study_methods()
    assert [method.key for method in methods] == [
        "splineops_projection",
        "splineops_interpolation",
        "scipy_gaussian",
        "skimage_resize",
        "torch_area",
        "opencv_area",
        "scipy_polyphase",
    ]
    comparisons = {method.key for method in methods if method.comparison}
    assert comparisons == {
        "scipy_gaussian",
        "skimage_resize",
        "torch_area",
        "opencv_area",
        "scipy_polyphase",
    }
    assert "opencv_area" not in study.MINIMUM_SPEEDUPS
