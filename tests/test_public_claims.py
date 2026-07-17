from __future__ import annotations

import json
import tomllib
from importlib.resources import files
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "selma3d-vessels" / "results.json"


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def test_canonical_claim_matches_frozen_decision_and_scope() -> None:
    result = load_json(RESULTS)
    decision = result["decision"]
    assert decision["strict_named_method_superiority_pass"] is True
    assert decision["narrow_quality_at_speed_claim_pass"] is True
    assert decision["segmentation_superiority_demonstrated"] is False
    assert decision["broad_resampling_superiority_demonstrated"] is False

    claims = (ROOT / "docs/claims.rst").read_text(encoding="utf-8")
    required_scope = (
        "18 held-out SELMA3D WGA",
        "vessel-ranking ROC AUC",
        "predeclared family-wise rule",
        "recorded one-thread CPU run",
        "12.49 times",
        "15.17 times",
        "1.36 times",
        "not segmentation",
        "One patch favoured SciPy",
    )
    for phrase in required_scope:
        assert phrase in claims


def test_packaged_demo_summary_is_a_narrow_copy_of_frozen_evidence() -> None:
    result = load_json(RESULTS)
    resource = files("splineops.demos.data").joinpath("selma3d_vessels_summary.json")
    with resource.open(encoding="utf-8") as handle:
        packaged = json.load(handle)

    for packaged_row, source_row in zip(
        packaged["summary"], result["summary"], strict=True
    ):
        for key in ("method", "label", "mean_roc_auc", "median_runtime_s"):
            assert packaged_row[key] == source_row[key]
    for packaged_row, source_row in zip(
        packaged["comparisons"], result["comparisons"], strict=True
    ):
        for key in (
            "comparison",
            "mean_auc_difference",
            "median_runtime_ratio",
            "quality_superior",
            "speed_pass",
        ):
            assert packaged_row[key] == source_row[key]
    assert packaged["decision"] == result["decision"]


def test_distribution_metadata_exposes_the_installed_demo() -> None:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    project = pyproject["project"]
    assert project["version"] == "2.2.0"
    assert project["scripts"]["splineops-selma3d-demo"] == (
        "splineops.demos.selma3d:main"
    )
    assert project["urls"]["Evidence"].endswith("/claims.html")
    assert {"resampling", "antialiasing", "microscopy"}.issubset(project["keywords"])
    extras = project["optional-dependencies"]
    assert "splineops[study]" in extras["selma3d-demo"]
    assert not any("torch" in requirement for requirement in extras["selma3d-demo"])
    assert "splineops[selma3d-demo]" in extras["selma3d-demo-all"]
    assert any("torch" in requirement for requirement in extras["selma3d-demo-all"])


def test_primary_navigation_keeps_modules_and_examples_visible() -> None:
    index = (ROOT / "docs/index.rst").read_text(encoding="utf-8")
    for destination in (
        "getting-started",
        "modules",
        "examples",
        "evidence",
        "project",
        "api/index",
    ):
        assert f"   {destination}\n" in index
    conf = (ROOT / "docs/conf.py").read_text(encoding="utf-8")
    assert '"header_links_before_dropdown": 8' in conf


def test_public_surfaces_point_to_the_claim_registry() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    docs_home = (ROOT / "docs/index.rst").read_text(encoding="utf-8")
    demo = (ROOT / "docs/selma3d-demo.rst").read_text(encoding="utf-8")
    assert "https://splineops.github.io/claims.html" in readme
    assert ":link: evidence" in docs_home
    assert ":doc:`claims`" in demo
    assert "post-study presentation choice" in demo
