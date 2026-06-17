#!/usr/bin/env python3
"""Run the resize benchmark artifact set used for PR preparation.

This is an orchestration wrapper around the existing benchmark scripts. It does
not implement benchmark logic itself; it just creates timestamped artifact
paths, runs the native/library/ResizePlan suites, and builds one Markdown
report from the resulting CSVs.
"""

from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class ArtifactSet:
    output_dir: str
    tag: str
    native_csv: str | None
    native_json: str | None
    libraries_csv: str | None
    libraries_json: str | None
    plan_csv: str | None
    plan_json: str | None
    projection_methods_csv: str | None
    projection_methods_json: str | None
    report_md: str
    manifest_json: str
    commands_txt: str


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print(flush=True)
    print("$ " + shell_join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated artifacts. Defaults to /tmp/splineops_resize_pr_<tag>.",
    )
    parser.add_argument("--tag", default=timestamp_tag())
    parser.add_argument(
        "--native-profile",
        choices=["smoke", "standard", "full"],
        default="full",
    )
    parser.add_argument(
        "--library-profile",
        choices=["smoke", "standard", "full"],
        default="full",
    )
    parser.add_argument(
        "--plan-profile",
        choices=["smoke", "standard"],
        default="standard",
    )
    parser.add_argument(
        "--projection-methods-profile",
        choices=["smoke", "standard", "stability"],
        default="standard",
    )
    parser.add_argument("--threads", default="1,8,default")
    parser.add_argument("--native-repeats", type=int, default=3)
    parser.add_argument("--native-warmups", type=int, default=1)
    parser.add_argument("--library-repeats", type=int, default=3)
    parser.add_argument("--library-warmups", type=int, default=1)
    parser.add_argument(
        "--library-splineops-threads",
        default="default",
        help="Value passed to benchmark_resize_libraries.py --splineops-threads.",
    )
    parser.add_argument("--plan-repeats", type=int, default=5)
    parser.add_argument("--plan-warmups", type=int, default=1)
    parser.add_argument("--plan-frames", type=int, default=8)
    parser.add_argument("--projection-methods-repeats", type=int, default=3)
    parser.add_argument("--projection-methods-warmups", type=int, default=1)
    parser.add_argument("--projection-methods-degrees", default="1,3")
    parser.add_argument("--projection-methods-dtypes", default="float32,float64")
    parser.add_argument("--exact-rel-l2", type=float, default=1e-5)
    parser.add_argument("--title", default="Resize PR Benchmark Report")
    parser.add_argument("--skip-native", action="store_true")
    parser.add_argument("--skip-libraries", action="store_true")
    parser.add_argument("--skip-plan", action="store_true")
    parser.add_argument("--skip-projection-methods", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def ensure_positive(name: str, value: int) -> None:
    if value <= 0:
        raise SystemExit(f"{name} must be positive")


def json_ready_options(args: argparse.Namespace) -> dict[str, object]:
    options: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            options[key] = str(value)
        else:
            options[key] = value
    return options


def main() -> int:
    args = parse_args()
    ensure_positive("--native-repeats", args.native_repeats)
    ensure_positive("--library-repeats", args.library_repeats)
    ensure_positive("--plan-repeats", args.plan_repeats)
    ensure_positive("--projection-methods-repeats", args.projection_methods_repeats)
    ensure_positive("--plan-frames", args.plan_frames)
    if (
        args.native_warmups < 0
        or args.library_warmups < 0
        or args.plan_warmups < 0
        or args.projection_methods_warmups < 0
    ):
        raise SystemExit("warmup counts must be non-negative")
    if (
        args.skip_native
        and args.skip_libraries
        and args.skip_plan
        and args.skip_projection_methods
    ):
        raise SystemExit("at least one benchmark suite must be enabled")

    output_dir = args.output_dir or Path(f"/tmp/splineops_resize_pr_{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    native_csv = output_dir / f"resize_native_{args.native_profile}_{args.tag}.csv"
    native_json = output_dir / f"resize_native_{args.native_profile}_{args.tag}.json"
    libraries_csv = output_dir / f"resize_libraries_{args.library_profile}_{args.tag}.csv"
    libraries_json = output_dir / f"resize_libraries_{args.library_profile}_{args.tag}.json"
    plan_csv = output_dir / f"resize_plan_{args.plan_profile}_{args.tag}.csv"
    plan_json = output_dir / f"resize_plan_{args.plan_profile}_{args.tag}.json"
    projection_methods_csv = output_dir / (
        f"resize_projection_methods_{args.projection_methods_profile}_{args.tag}.csv"
    )
    projection_methods_json = output_dir / (
        f"resize_projection_methods_{args.projection_methods_profile}_{args.tag}.json"
    )
    report_md = output_dir / f"resize_pr_report_{args.tag}.md"
    manifest_json = output_dir / f"resize_pr_manifest_{args.tag}.json"
    commands_txt = output_dir / f"resize_pr_commands_{args.tag}.txt"

    commands: list[list[str]] = []
    py = sys.executable

    if not args.skip_native:
        commands.append([
            py,
            str(SCRIPT_DIR / "benchmark_resize_native.py"),
            "--profile",
            args.native_profile,
            "--backend",
            "both",
            "--threads",
            args.threads,
            "--repeats",
            str(args.native_repeats),
            "--warmups",
            str(args.native_warmups),
            "--output-csv",
            str(native_csv),
            "--output-json",
            str(native_json),
        ])

    if not args.skip_libraries:
        commands.append([
            py,
            str(SCRIPT_DIR / "benchmark_resize_libraries.py"),
            "--profile",
            args.library_profile,
            "--backends",
            "all",
            "--repeats",
            str(args.library_repeats),
            "--warmups",
            str(args.library_warmups),
            "--splineops-threads",
            args.library_splineops_threads,
            "--output-csv",
            str(libraries_csv),
            "--output-json",
            str(libraries_json),
        ])

    if not args.skip_plan:
        commands.append([
            py,
            str(SCRIPT_DIR / "benchmark_resize_plan.py"),
            "--profile",
            args.plan_profile,
            "--frames",
            str(args.plan_frames),
            "--repeats",
            str(args.plan_repeats),
            "--warmups",
            str(args.plan_warmups),
            "--output-csv",
            str(plan_csv),
            "--output-json",
            str(plan_json),
        ])

    if not args.skip_projection_methods:
        commands.append([
            py,
            str(SCRIPT_DIR / "benchmark_resize_projection_methods.py"),
            "--profile",
            args.projection_methods_profile,
            "--repeats",
            str(args.projection_methods_repeats),
            "--warmups",
            str(args.projection_methods_warmups),
            "--degrees",
            args.projection_methods_degrees,
            "--dtypes",
            args.projection_methods_dtypes,
            "--output-csv",
            str(projection_methods_csv),
            "--output-json",
            str(projection_methods_json),
        ])

    report_cmd = [
        py,
        str(SCRIPT_DIR / "summarize_resize_benchmarks.py"),
        "report",
        "--title",
        args.title,
        "--exact-rel-l2",
        str(args.exact_rel_l2),
        "--output",
        str(report_md),
    ]
    if not args.skip_native:
        report_cmd.extend(["--native", str(native_csv)])
    if not args.skip_libraries:
        report_cmd.extend(["--libraries", str(libraries_csv)])
    if not args.skip_plan:
        report_cmd.extend(["--plan", str(plan_csv)])
    if not args.skip_projection_methods:
        report_cmd.extend(["--projection-methods", str(projection_methods_csv)])
    commands.append(report_cmd)

    artifact_set = ArtifactSet(
        output_dir=str(output_dir),
        tag=args.tag,
        native_csv=None if args.skip_native else str(native_csv),
        native_json=None if args.skip_native else str(native_json),
        libraries_csv=None if args.skip_libraries else str(libraries_csv),
        libraries_json=None if args.skip_libraries else str(libraries_json),
        plan_csv=None if args.skip_plan else str(plan_csv),
        plan_json=None if args.skip_plan else str(plan_json),
        projection_methods_csv=(
            None if args.skip_projection_methods else str(projection_methods_csv)
        ),
        projection_methods_json=(
            None if args.skip_projection_methods else str(projection_methods_json)
        ),
        report_md=str(report_md),
        manifest_json=str(manifest_json),
        commands_txt=str(commands_txt),
    )

    commands_text = "\n".join(shell_join(cmd) for cmd in commands) + "\n"
    commands_txt.write_text(commands_text, encoding="utf-8")
    manifest = {
        "artifact_set": asdict(artifact_set),
        "options": json_ready_options(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "commands": commands,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"artifact directory: {output_dir}", flush=True)
    print(f"manifest: {manifest_json}", flush=True)
    print(f"commands: {commands_txt}", flush=True)
    for cmd in commands:
        run_command(cmd, dry_run=args.dry_run)

    print(flush=True)
    print(f"report: {report_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
