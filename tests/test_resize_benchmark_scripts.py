import csv
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
