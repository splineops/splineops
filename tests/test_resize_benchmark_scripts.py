import csv
import json
import subprocess
import sys
from pathlib import Path


def test_projection_methods_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_csv = tmp_path / "projection_methods.csv"
    output_json = tmp_path / "projection_methods.json"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_resize_projection_methods.py"),
            "--profile",
            "smoke",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--degrees",
            "1",
            "--dtypes",
            "float64",
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    with output_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert output_json.exists()
    assert {row["case"] for row in rows} == {"camera_half", "random_half"}
    assert all(float(row["oblique_speedup_vs_least_squares"]) > 0.0 for row in rows)


def test_resize_pr_oblique_profile_dry_run(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "oblique_pr"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_resize_pr.py"),
            "--profile",
            "oblique-pr",
            "--output-dir",
            str(output_dir),
            "--tag",
            "test_oblique_pr",
            "--dry-run",
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads(
        (output_dir / "resize_pr_manifest_test_oblique_pr.json").read_text(
            encoding="utf-8"
        )
    )
    options = manifest["options"]

    assert options["profile"] == "oblique-pr"
    assert options["native_profile"] == "full"
    assert options["library_profile"] == "full"
    assert options["plan_profile"] == "standard"
    assert options["projection_methods_profile"] == "standard"
    assert options["title"] == "Oblique Antialiasing Resize PR Benchmark Report"
    assert "--projection-methods" in (
        output_dir / "resize_pr_commands_test_oblique_pr.txt"
    ).read_text(encoding="utf-8")


def test_resize_upstream_package_docs_only(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "upstream"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "prepare_resize_upstream_package.py"),
            "--output-dir",
            str(output_dir),
            "--tag",
            "test_upstream",
            "--skip-benchmarks",
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    manifest = json.loads(
        (output_dir / "resize_upstream_manifest_test_upstream.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["options"]["skip_benchmarks"] is True
    assert (output_dir / "resize_scipy_rfc_test_upstream.md").exists()
    assert (output_dir / "resize_pytorch_bridge_test_upstream.md").exists()
    semantics = (output_dir / "resize_upstream_semantics_test_upstream.md").read_text(
        encoding="utf-8"
    )
    assert "scipy.ndimage.zoom" in semantics
    assert "torch.nn.functional.interpolate" in semantics


def test_scipy_zoom_audit_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "scipy_audit"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "audit_scipy_zoom.py"),
            "--profile",
            "smoke",
            "--variant-profile",
            "focused",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--tag",
            "test_scipy_audit",
            "--output-dir",
            str(output_dir),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(
        (output_dir / "scipy_zoom_audit_test_scipy_audit.json").read_text(
            encoding="utf-8"
        )
    )
    rows = payload["rows"]
    assert rows
    assert any(row["same_semantics_candidate"] for row in rows)
    assert all("scipy_variant" in row for row in rows)
    assert (
        output_dir / "scipy_zoom_audit_report_test_scipy_audit.md"
    ).exists()
    asv = (output_dir / "scipy_zoom_asv_benchmark_test_scipy_audit.py").read_text(
        encoding="utf-8"
    )
    assert "NdimageZoomSplineCandidates" in asv
    source_audit = (
        output_dir / "scipy_zoom_source_audit_test_scipy_audit.md"
    ).read_text(encoding="utf-8")
    assert "ni_interpolation.c" in source_audit
