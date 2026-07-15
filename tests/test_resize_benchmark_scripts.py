import csv
import json
import os
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
    assert (output_dir / "scipy_zoom_audit_report_test_scipy_audit.md").exists()
    asv = (output_dir / "scipy_zoom_asv_benchmark_test_scipy_audit.py").read_text(
        encoding="utf-8"
    )
    assert "NdimageZoomSplineCandidates" in asv
    source_audit = (
        output_dir / "scipy_zoom_source_audit_test_scipy_audit.md"
    ).read_text(encoding="utf-8")
    assert "ni_interpolation.c" in source_audit


def test_tensorspline_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "tensorspline.json"
    output_csv = tmp_path / "tensorspline.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_tensorspline.py"),
            "--profile",
            "smoke",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["memory_metric"] == "tracemalloc peak bytes during one call"
    assert {row["query_kind"] for row in payload["results"]} == {"grid", "points"}
    assert all(row["evaluation_peak_bytes"] > 0 for row in payload["results"])
    with output_csv.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == len(payload["results"])


def test_tensorspline_memory_scaling_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "tensorspline-memory.json"
    output_csv = tmp_path / "tensorspline-memory.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_tensorspline_memory_scaling.py"),
            "--counts",
            "100,1000",
            "--shape",
            "16",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["configuration"]["coordinate_memory_traced"] is False
    assert [row["query_count"] for row in payload["results"]] == [100, 1000]
    assert all(row["temporary_overhead_bytes"] > 0 for row in payload["results"])


def test_tensorspline_query_plan_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "tensorspline-plan.json"
    output_csv = tmp_path / "tensorspline-plan.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_tensorspline_query_plan.py"),
            "--points",
            "1000",
            "--repeats",
            "2",
            "--shape",
            "16",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["speedup"] > 0
    assert {row["path"] for row in payload["results"]} == {
        "ordinary",
        "query_plan",
    }
    assert payload["results"][1]["retained_bytes"] > 0
    assert payload["configuration"]["data_workload"].startswith("changing")
    assert output_csv.exists()


def test_affine_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "affine.json"
    output_csv = tmp_path / "affine.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_affine.py"),
            "--profile",
            "smoke",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["semantics"]["boundary"] == "whole-sample mirror"
    assert {row["case"] for row in payload["results"]} == {
        "2d_linear",
        "2d_cubic",
        "3d_linear",
        "3d_cubic",
    }
    assert all(row["max_abs_difference"] < 1e-10 for row in payload["results"])
    assert all(row["plan_retained_bytes"] > 0 for row in payload["results"])
    assert output_csv.exists()


def test_differentials_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "differentials.json"
    output_csv = tmp_path / "differentials.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_differentials.py"),
            "--profile",
            "smoke",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["configuration"]["operation"] == "gradient_magnitude"
    assert payload["vectorized_speedup"] > 0
    assert payload["max_abs_difference"] < 2e-6
    assert {row["path"] for row in payload["results"]} == {
        "vectorized_cold",
        "cached_instance",
        "multi_output_plan",
        "gradient_only_plan",
        "laplacian_only_plan",
        "scalar_reference",
    }
    assert payload["laplacian_peak_fraction_of_full"] < 1.0
    assert output_csv.exists()


def test_multiscale_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "multiscale.json"
    output_csv = tmp_path / "multiscale.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_multiscale.py"),
            "--profile",
            "smoke",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert {row["operation"] for row in payload["results"]} == {
        "pyramid_reduce_2d",
        "haar_analysis_2d",
    }
    assert all(row["speedup"] > 0 for row in payload["results"])
    assert output_csv.exists()


def test_consolidated_workflow_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "workflows.json"
    output_csv = tmp_path / "workflows.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_workflows.py"),
            "--profile",
            "smoke",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert {row["workflow"] for row in payload["results"]} == {
        "affine_two_geometries_one_prefilter",
        "affine_explicit_axes",
        "smoothing_explicit_axes",
        "differentials_explicit_axes",
        "denoising_lambda_path",
        "wavelet_roundtrip_explicit_axes",
    }
    assert set(payload["speedups_by_workflow"]) == {
        row["workflow"] for row in payload["results"]
    }
    assert all(row["speedup"] > 0 for row in payload["results"])
    assert all(row["max_abs_difference"] < 2e-4 for row in payload["results"])
    assert output_csv.exists()


def test_batch_scaling_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "batch-scaling.json"
    output_csv = tmp_path / "batch-scaling.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_batch_scaling.py"),
            "--profile",
            "smoke",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert {row["operation"] for row in payload["results"]} == {
        "affine",
        "laplacian_only",
        "haar_roundtrip",
    }
    assert {row["batch_count"] for row in payload["results"]} == {1, 2, 4}
    assert payload["summary"]["max_normalized_peak_growth"] < 1.5
    assert payload["summary"]["max_abs_difference"] < 1e-12
    assert output_csv.exists()


def test_downstream_workflow_benchmark_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "downstream-workflows.json"
    output_csv = tmp_path / "downstream-workflows.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "benchmark_downstream_workflows.py"),
            "--profile",
            "smoke",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert {row["workflow"] for row in payload["results"]} == {
        "persisted_registration_fanout",
        "buffered_volume_features",
    }
    assert payload["summary"]["max_abs_difference"] < 1e-12
    assert output_csv.exists()


def test_affine_phase_profile_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_json = tmp_path / "affine-phases.json"
    output_csv = tmp_path / "affine-phases.csv"

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "profile_affine_phases.py"),
            "--profile",
            "smoke",
            "--repeats",
            "1",
            "--warmups",
            "0",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert {row["case"] for row in payload["results"]} == {
        "2d_cubic",
        "3d_cubic",
    }
    assert all(
        row["dominant_phase"] in {"prefilter", "evaluation"}
        for row in payload["results"]
    )
    assert all(row["max_abs_difference"] == 0.0 for row in payload["results"])
    assert output_csv.exists()


def test_benchmark_threshold_failure_emits_github_annotation(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    policy = tmp_path / "policy.json"
    artifact = tmp_path / "result.json"
    policy.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checks": [
                    {
                        "name": "deliberate floor",
                        "artifact": artifact.name,
                        "json_path": "value",
                        "minimum": 2.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact.write_text(json.dumps({"value": 1.0}), encoding="utf-8")
    environment = os.environ.copy()
    environment["GITHUB_ACTIONS"] = "true"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "check_benchmark_thresholds.py"),
            "--policy",
            str(policy),
            "--artifacts-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        check=False,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 1
    assert "FAIL: deliberate floor[0]: 1 is below 2" in completed.stdout
    assert "::error title=Benchmark threshold::deliberate floor" in completed.stdout


def test_benchmark_threshold_checker(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    artifact = {
        "results": [
            {"speedup": 1.2, "max_abs_difference": 1e-12},
            {"speedup": 0.9, "max_abs_difference": 2e-12},
        ]
    }
    policy = {
        "schema_version": 1,
        "checks": [
            {
                "name": "relative runtime",
                "artifact": "artifact.json",
                "json_path": "results.*.speedup",
                "minimum": 0.5,
            },
            {
                "name": "equivalence",
                "artifact": "artifact.json",
                "json_path": "results.*.max_abs_difference",
                "maximum": 1e-10,
            },
        ],
    }
    (tmp_path / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "check_benchmark_thresholds.py"),
            "--policy",
            str(policy_path),
            "--artifacts-dir",
            str(tmp_path),
        ],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert result.returncode == 0
    assert "passed" in result.stdout
