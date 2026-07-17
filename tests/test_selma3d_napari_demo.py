from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_selma3d_vessels as study
from splineops.demos import selma3d as demo

ROOT = Path(__file__).resolve().parents[1]


def test_curtain_reveals_anchor_from_left() -> None:
    anchor = np.ones((2, 3, 10), dtype=np.float32)
    comparison = np.zeros_like(anchor)

    all_comparison, split_zero = demo.curtain(anchor, comparison, 0.0)
    half, split_half = demo.curtain(anchor, comparison, 0.5)
    all_anchor, split_full = demo.curtain(anchor, comparison, 1.0)

    assert split_zero == 0
    assert split_half == 5
    assert split_full == 10
    assert np.array_equal(all_comparison, comparison)
    assert np.all(half[..., :5] == 1.0)
    assert np.all(half[..., 5:] == 0.0)
    assert np.array_equal(all_anchor, anchor)


def test_curtain_validates_shape_and_fraction() -> None:
    data = np.zeros((2, 3, 4))
    with pytest.raises(ValueError, match="identical shapes"):
        demo.curtain(data, np.zeros((2, 3, 5)), 0.5)
    with pytest.raises(ValueError, match="between zero and one"):
        demo.curtain(data, data, 1.1)


def test_napari_order_makes_plane_axis_first() -> None:
    array = np.arange(4 * 5 * 2).reshape(4, 5, 2)
    displayed = demo.napari_order(array)
    assert displayed.shape == (2, 5, 4)
    assert displayed[1, 3, 2] == array[2, 3, 1]


def test_build_methods_preserves_frozen_semantics_without_napari() -> None:
    methods = demo.build_methods(
        ("splineops_projection", "scipy_gaussian", "skimage_resize")
    )
    assert [method.key for method in methods] == [
        "splineops_projection",
        "scipy_gaussian",
        "skimage_resize",
    ]
    assert [method.grid for method in methods] == [
        "endpoint",
        "endpoint",
        "half_pixel",
    ]
    assert demo.CHECKSUMS == study.CHECKSUMS


def test_default_demo_avoids_the_optional_torch_comparison() -> None:
    assert "torch_area" not in demo.DEFAULT_COMPARISONS
    assert "torch_area" in demo.COMPARISON_KEYS
    assert demo.parse_args([]).comparisons == list(demo.DEFAULT_COMPARISONS)
    assert demo.parse_args(["--comparisons", "torch_area"]).comparisons == [
        "torch_area"
    ]


def test_published_panel_data_matches_frozen_artifact() -> None:
    published, speedups = demo.load_published_results()
    with (ROOT / "benchmarks/selma3d-vessels/results.json").open(
        encoding="utf-8"
    ) as handle:
        artifact = json.load(handle)
    summary = {row["method"]: row for row in artifact["summary"]}

    assert published["splineops_projection"].mean_roc_auc == pytest.approx(
        summary["splineops_projection"]["mean_roc_auc"]
    )
    assert speedups["scipy_gaussian"] == pytest.approx(12.494583619323219)

    repository, repository_speedups = demo.load_published_results(
        ROOT / "benchmarks/selma3d-vessels/results.json"
    )
    assert published == repository
    assert speedups == repository_speedups


def test_demo_is_documented_as_narrow_ranking_evidence() -> None:
    readme = (ROOT / "demos/README.md").read_text(encoding="utf-8")
    assert "not a segmentation metric" in readme
    assert "half-pixel" in readme
    assert "CC BY-NC" in readme
    assert "only patch 013" in readme
    assert "post-study presentation choice" in readme


def test_demo_script_can_show_help_from_repository_root() -> None:
    completed = subprocess.run(
        [sys.executable, "demos/selma3d_napari.py", "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--no-gui" in completed.stdout
