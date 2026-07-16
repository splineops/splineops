import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def load_study_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_bbbc050_segmentation.py"
    )
    spec = importlib.util.spec_from_file_location("bbbc050_study", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dice_curve_matches_direct_thresholding():
    study = load_study_module()
    values = np.asarray([0.1, 0.2, 0.49, 0.5, 0.9]).reshape(1, 1, 5)
    target = np.asarray([False, True, True, False, True]).reshape(1, 1, 5)
    thresholds = np.asarray([0.2, 0.5, 0.8])

    actual = study.dice_curve(values, target, thresholds)
    expected = []
    for threshold in thresholds:
        prediction = values >= threshold
        expected.append(
            2.0
            * np.count_nonzero(prediction & target)
            / (np.count_nonzero(prediction) + np.count_nonzero(target))
        )

    assert actual == pytest.approx(expected)


def test_geometry_and_ground_truth_are_endpoint_aligned():
    study = load_study_module()
    labels = np.arange(2 * 5 * 7).reshape(2, 5, 7)

    output_shape = study.half_xy_shape(labels.shape)
    resized = study.resize_ground_truth(labels, output_shape, grid="endpoint")

    assert output_shape == (2, 3, 4)
    assert resized.shape == output_shape
    assert resized.dtype == np.bool_
    assert not resized[0, 0, 0]
    assert resized[-1, -1, -1]


def test_bootstrap_and_success_constants_are_deterministic():
    study = load_study_module()
    differences = np.asarray([0.01, 0.02, 0.03, 0.04])

    first = study.bootstrap_mean_ci(differences, resamples=500, seed=19)
    second = study.bootstrap_mean_ci(differences, resamples=500, seed=19)

    assert first == second
    assert first[0] > 0.0
    assert study.sign_flip_pvalue(differences) == pytest.approx(1.0 / 16.0)
    assert study.PILOT_TIMEPOINT == "251"
    assert study.PRACTICAL_MARGIN == 0.005
    assert len(study.IMAGES_SHA256) == 64


def test_recorded_result_does_not_overclaim_superiority():
    result_path = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "bbbc050" / "results.json"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    comparisons = {row["baseline"]: row for row in payload["comparisons"]}

    assert payload["schema_version"] == 1
    assert payload["dataset"]["counts"] == {"train": 121, "test": 44}
    assert payload["conclusion"]["superiority_demonstrated"] is False
    assert comparisons["splineops_interpolation"]["demonstrated"] is True
    assert comparisons["scipy_gaussian"]["demonstrated"] is False
    assert comparisons["skimage_resize"]["external_mean_dice_delta"] < 0.0
