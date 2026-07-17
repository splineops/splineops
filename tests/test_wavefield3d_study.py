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
        / "benchmark_wavefield3d_coarsening.py"
    )
    spec = importlib.util.spec_from_file_location("wavefield3d_study", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_modes_respect_the_frequency_bands():
    study = load_study_module()
    geometry = study.GEOMETRIES[2]

    low, high, low_coefficients, high_coefficients = study.generate_mode_groups(
        geometry, geometry.seed_start
    )

    assert len(low) == study.LOW_MODE_COUNT
    assert len(high) == study.HIGH_MODE_COUNT
    assert np.linalg.norm(low_coefficients) == pytest.approx(1.0)
    assert np.linalg.norm(high_coefficients) == pytest.approx(1.0)
    for mode in low:
        assert all(
            frequency <= np.floor(study.LOW_NYQUIST_FRACTION * (output - 1))
            for frequency, output in zip(mode, geometry.output_shape)
        )
    for mode in high:
        assert any(
            frequency >= np.ceil(study.HIGH_NYQUIST_FRACTION * (output - 1))
            for frequency, output in zip(mode, geometry.output_shape)
        )
        assert all(
            frequency <= np.floor(study.SOURCE_NYQUIST_FRACTION * (source - 1))
            for frequency, source in zip(mode, geometry.source_shape)
        )


def test_target_grids_and_bootstrap_are_deterministic():
    study = load_study_module()
    endpoint = study.normalized_coordinates((9, 11, 13), (5, 6, 7), grid="endpoint")
    half_pixel = study.normalized_coordinates((9, 11, 13), (5, 6, 7), grid="half_pixel")

    assert endpoint[0][[0, -1]] == pytest.approx((0.0, 1.0))
    assert half_pixel[0][0] > 0.0
    assert half_pixel[0][-1] < 1.0

    projection = np.asarray([0.1, 0.2, 0.15, 0.18])
    baseline = np.asarray([0.3, 0.4, 0.35, 0.38])
    first = study.bootstrap_relative_reduction_ci(
        projection, baseline, resamples=500, seed=19
    )
    second = study.bootstrap_relative_reduction_ci(
        projection, baseline, resamples=500, seed=19
    )
    assert first == second
    assert first[0] > study.CI_REDUCTION_MARGIN


def test_protocol_digest_is_portable_across_line_endings(tmp_path):
    study = load_study_module()
    unix = tmp_path / "unix.md"
    windows = tmp_path / "windows.md"
    unix.write_bytes(b"frozen\nprotocol\n")
    windows.write_bytes(b"frozen\r\nprotocol\r\n")
    assert study.sha256_text(unix) == study.sha256_text(windows)


def test_recorded_superiority_claim_is_narrow_and_complete():
    study = load_study_module()
    root = Path(__file__).resolve().parents[1]
    result_path = root / "benchmarks" / "wavefield3d" / "results.json"
    protocol_path = root / "benchmarks" / "wavefield3d" / "PROTOCOL.md"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    summaries = {row["method"]: row for row in payload["accuracy_summary"]}
    comparisons = {row["baseline"]: row for row in payload["comparisons"]}
    runtimes = {row["baseline"]: row for row in payload["runtime_comparisons"]}

    assert payload["schema_version"] == 1
    assert payload["protocol"]["protocol_conforming_run"] is True
    assert payload["protocol"]["frozen_before_confirmation_run"] is True
    assert payload["protocol"]["sha256"] == study.sha256_text(protocol_path)
    assert payload["conclusion"]["generic_resize_superiority_demonstrated"] is True
    assert (
        payload["conclusion"]["scientific_resampling_superiority_demonstrated"] is False
    )
    assert len(payload["case_scores"]) == 72 * 8
    assert all(
        row["demonstrated"] for row in comparisons.values() if row["predeclared"]
    )
    assert comparisons["scipy_polyphase"]["predeclared"] is False
    assert comparisons["scipy_polyphase"]["demonstrated"] is False
    assert comparisons["torch_area"]["individual_win_fraction"] < 1.0
    assert comparisons["torch_area"]["individual_win_fraction"] >= 0.90
    assert {row["seed"] for row in payload["case_scores"]} == set(
        range(2000, 2008)
    ) | set(range(2100, 2108)) | set(range(2200, 2208))

    # The artifact must retain the two important limits rather than only the win.
    assert (
        summaries["splineops_interpolation"]["mean_passband_nrmse"]
        < summaries["splineops_projection"]["mean_passband_nrmse"]
    )
    assert (
        summaries["scipy_polyphase"]["mean_nrmse"]
        < summaries["splineops_projection"]["mean_nrmse"]
    )
    assert runtimes["torch_trilinear"]["faster_in_all_geometries"] is False
