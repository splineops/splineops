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
        / "benchmark_cells3d_downsampling.py"
    )
    spec = importlib.util.spec_from_file_location("cells3d_study", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_known_target_favors_projection_over_no_antialiasing():
    study = load_study_module()
    source, target = study.analytic_calibration()
    output_shape = (target.shape[0], target.shape[2], target.shape[3])

    projection = study.resize_splineops(source, output_shape, antialias=True)
    interpolation = study.resize_splineops(source, output_shape, antialias=False)

    assert source.shape == (48, 1, 64, 64)
    assert target.shape == (24, 1, 32, 32)
    assert study.nrmse(projection, target) < study.nrmse(interpolation, target)


def test_endpoint_spacing_and_dataset_pin_are_explicit():
    study = load_study_module()

    spacing = study.output_spacing_um((60, 256, 256), (30, 128, 128))

    assert spacing == pytest.approx((0.59, 0.5220472441, 0.5220472441))
    assert len(study.DATASET_SHA256) == 64
    assert np.dtype(np.uint16).itemsize == 2
    assert study.DATASET_SHAPE == (60, 2, 256, 256)


def test_recorded_artifact_records_metric_limits():
    result_path = (
        Path(__file__).resolve().parents[1] / "benchmarks" / "cells3d" / "results.json"
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    results = {row["method"]: row for row in payload["results"]}

    assert payload["schema_version"] == 1
    assert payload["dataset"]["license"] == "CC0-1.0"
    assert set(results) == {
        "splineops_projection",
        "splineops_interpolation",
        "scipy_gaussian",
        "skimage_resize",
    }
    assert (
        results["splineops_projection"]["synthetic_nrmse"]
        < results["splineops_interpolation"]["synthetic_nrmse"]
    )
    assert "not ground truth" in payload["metric_limits"]["gaussian_reference"]
