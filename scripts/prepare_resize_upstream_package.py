#!/usr/bin/env python3
"""Prepare an upstream-facing resize evidence package.

The generated package is meant for early SciPy/PyTorch discussions. It wraps
the existing benchmark bundle and adds stable Markdown notes:

* a semantics matrix,
* a correctness/tolerance policy,
* a SciPy RFC draft,
* a SciPy prototype patch plan,
* a PyTorch custom-operator bridge note,
* a README and manifest.

Use ``--benchmark-profile smoke`` for a quick local check and
``--benchmark-profile oblique-pr`` for a formal evidence bundle.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shlex
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent


SCIPY_ZOOM_DOC = "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html"
SCIPY_CONTRIBUTOR_DOC = "https://docs.scipy.org/doc/scipy/dev/contributor/contributor_toc.html"
SCIPY_INTERPOLATION_SRC = "https://github.com/scipy/scipy/blob/main/scipy/ndimage/src/ni_interpolation.c"
PYTORCH_INTERPOLATE_DOC = "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html"
PYTORCH_CUSTOM_OP_DOC = "https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html"
PYTORCH_ATEN_NATIVE_DOC = "https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md"


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_command(cmd: list[str], *, dry_run: bool) -> None:
    print("$ " + shell_join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def read_project_version() -> str:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def installed_package_version() -> str | None:
    try:
        return importlib.metadata.version("splineops")
    except importlib.metadata.PackageNotFoundError:
        return None


def git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except subprocess.CalledProcessError:
        return "<unknown>"


def repo_state() -> dict[str, str]:
    installed = installed_package_version()
    return {
        "branch": git_output(["branch", "--show-current"]),
        "commit": git_output(["rev-parse", "HEAD"]),
        "short_commit": git_output(["rev-parse", "--short", "HEAD"]),
        "status": git_output(["status", "--short"]),
        "version": read_project_version(),
        "installed_version": installed or "<not installed>",
    }


def markdown_path(path: Path | None) -> str:
    return "`not generated`" if path is None else f"`{path}`"


def render_semantics_matrix(*, tag: str, exact_rel_l2: float) -> str:
    return f"""# Resize Upstream Semantics Matrix

Tag: `{tag}`

This table separates same-semantics evidence from contextual image-resize
comparisons. It is the first thing to keep clear in any upstream discussion.

| Operation family | splineops API | Closest SciPy/PyTorch surface | Semantic relationship | Upstream role |
| --- | --- | --- | --- | --- |
| Pure spline interpolation | `resize(..., method="linear"|"quadratic"|"cubic")` | `scipy.ndimage.zoom(order=1..3, mode="mirror", prefilter=order>1)` | Closest like-for-like family when output shape, boundary mode and coordinate mapping are aligned. | Best first SciPy optimization target. |
| Oblique antialiasing projection | `resize(..., method="*-antialiasing")` | No direct SciPy or PyTorch equivalent. | Same spline framework with projection before sampling. On the public zero-shift grid, analysis degree one uses direct compact cross-Gram rows; analysis degree zero retains the finite-difference form. | RFC/API discussion after same-semantics acceleration evidence. |
| Equal-degree least-squares projection | `resize_degrees(..., interp_degree=d, analy_degree=d, synthe_degree=d)` | No direct SciPy/PyTorch public equivalent. | Orthogonal projection control. At analysis degree one or greater it uses stable direct compact cross-Gram rows, with support and plan cost depending on the degree. | Keep as advanced degree-control evidence, not a separate preset. |
| PyTorch image interpolation | `torch.nn.functional.interpolate` | PyTorch tensor resize modes. | Different tensor layout, coordinate conventions, antialias support, device dispatch and autograd expectations. | Prototype as custom op before upstreaming. |
| OpenCV/Pillow/skimage image resize | External benchmark rows | Contextual image-processing baselines. | Often different coordinate rules, filters, boundaries and dtype behavior. | Do not use as correctness evidence. |

## Recommended Public Method Boundary

The production downsampling story remains the oblique antialiasing preset
family:

| Preset | Degrees | Intended use |
| --- | --- | --- |
| `linear-antialiasing` | `(interp=1, analy=0, synthe=1)` | Fast low-order antialiasing. |
| `quadratic-antialiasing` | `(interp=2, analy=1, synthe=2)` | Middle ground when quadratic synthesis is desired. |
| `cubic-antialiasing` | `(interp=3, analy=1, synthe=3)` | Recommended high-quality production downsampling preset. |

## Exact-ish External Rows

When using external library benchmark rows, treat a row as exact-ish only when
the relative L2 output difference is below `{exact_rel_l2:g}` and the method
metadata explains why the coordinate, boundary and interpolation model match
closely enough. Anything outside that threshold is still useful context, but
not a proof that another library implements the same operation faster.

Primary sources for upstream semantics:

- SciPy `ndimage.zoom`: {SCIPY_ZOOM_DOC}
- SciPy interpolation source entry point: {SCIPY_INTERPOLATION_SRC}
- PyTorch `interpolate`: {PYTORCH_INTERPOLATE_DOC}
"""


def render_correctness_policy(*, tag: str, exact_rel_l2: float) -> str:
    return f"""# Resize Upstream Correctness Policy

Tag: `{tag}`

The upstream package should lead with correctness before speed. The native
backend is an implementation of the splineops Python fallback, so same-package
parity is the highest-value oracle.

## Correctness Oracles

| Scope | Oracle | Default tolerance policy |
| --- | --- | --- |
| Pure interpolation, `float64` | Native path vs Python fallback | `atol=rtol=5e-11` in focused tests. |
| Pure interpolation, `float32` | Native path vs Python fallback | `atol=rtol=5e-6` in focused tests. |
| Projection/antialiasing, `float64` | Native path vs Python fallback | `atol=rtol=2e-9` in focused tests. |
| Projection/antialiasing, `float32` | Native path vs Python fallback | `atol=rtol=3e-5` in focused tests. |
| External library rows | Relative L2 vs splineops output | Exact-ish threshold `{exact_rel_l2:g}` for PR interpretation. |
| `ResizePlan` reuse | Plan output vs one-shot resize output | Max-abs diff reported per row; zero or roundoff-level drift expected. |
| Constant projection rows | Constant input preservation | Dedicated float32 internal checks use strict constant-preservation tolerances. |

## What Must Not Be Hidden

- Coordinate mapping differences can dominate apparent output quality.
- Boundary modes are not interchangeable; SciPy `mirror` is the closest first
  target for many splineops rows, while image libraries often use other rules.
- OpenCV/Pillow/skimage/PyTorch rows are useful context, but not correctness
  evidence unless the benchmark row is marked exact-ish.
- Float32 internal projection is intentionally conservative by default: 3-D
  public downsampling antialiasing presets are enabled automatically, while 2-D
  projection/antialiasing remains double-internal unless explicitly requested.

## Required Validation Before A Formal Upstream PR

```shell
python -m py_compile \\
  scripts/benchmark_resize_pr.py \\
  scripts/benchmark_resize_native.py \\
  scripts/benchmark_resize_libraries.py \\
  scripts/benchmark_resize_plan.py \\
  scripts/benchmark_resize_projection_methods.py \\
  scripts/audit_scipy_zoom.py \\
  scripts/summarize_resize_benchmarks.py \\
  scripts/prepare_resize_upstream_package.py
python -m pytest -q
python scripts/audit_scipy_zoom.py \\
  --profile standard \\
  --variant-profile focused \\
  --output-dir /tmp/splineops_scipy_zoom_audit_standard
python scripts/prepare_resize_upstream_package.py \\
  --benchmark-profile oblique-pr \\
  --output-dir /tmp/splineops_resize_upstream_oblique_pr
git diff --check
```
"""


def render_scipy_rfc(
    *,
    tag: str,
    report_md: Path | None,
    semantics_md: Path,
    correctness_md: Path,
    prototype_md: Path,
) -> str:
    return f"""# Draft SciPy RFC: Accelerated Spline Zoom And Antialiasing Evidence

Tag: `{tag}`

## Proposed Title

RFC: benchmark and prototype accelerated spline resize paths for `scipy.ndimage`

## Short Summary

SplineOps implements N-D spline resize using the Munoz/Blu/Unser projection
framework and now has a native CPU backend with strong speedups over its Python
fallback. The first SciPy-facing proposal should be conservative: benchmark and
prototype implementation-level acceleration for `scipy.ndimage.zoom`-like
semantics before discussing any new antialiasing/projection API.

## Why This Might Belong In SciPy

- SciPy already exposes N-D spline zoom through `scipy.ndimage.zoom`.
- The closest initial target is same-semantics acceleration for order-1/order-3
  interpolation rows with aligned output shape, coordinate mapping and boundary
  mode.
- Oblique antialiasing projection can be discussed separately as an API/RFC
  after maintainers agree the method is in scope.

## Evidence Package

- Benchmark report: {markdown_path(report_md)}
- Semantics matrix: `{semantics_md}`
- Correctness policy: `{correctness_md}`
- Prototype patch plan: `{prototype_md}`

## Proposed Upstream Phases

1. Open an issue/RFC with the evidence package and ask whether maintainers are
   interested in a backend optimization, a new antialiasing mode, or neither.
2. Add SciPy ASV-style benchmarks for representative `ndimage.zoom` rows before
   changing implementation code.
3. Prototype a same-semantics CPU acceleration path behind private internals:
   cached per-axis plan construction, batched line execution and specialized
   fixed-support accumulation for order 1 and 3.
4. Only after same-semantics behavior is stable, discuss whether projection
   antialiasing belongs as a new public option or a separate function.

## Non-Claims

- This is not a request to replace all SciPy image interpolation.
- This does not claim OpenCV/PyTorch/Pillow rows are equivalent operations.
- This does not claim oblique projection is the orthogonal least-squares
  optimum. Equal-degree least-squares remains an advanced explicit control;
  oblique is the public quality/cost preset family. Relative speed and quality
  claims must come from the current benchmark bundle.

## Links

- SciPy contributor guide: {SCIPY_CONTRIBUTOR_DOC}
- SciPy `ndimage.zoom` documentation: {SCIPY_ZOOM_DOC}
- SciPy interpolation C source entry point: {SCIPY_INTERPOLATION_SRC}
"""


def render_scipy_prototype_plan(*, tag: str) -> str:
    return f"""# SciPy Prototype Patch Plan

Tag: `{tag}`

This is a prototype plan, not a ready patch. The goal is to reduce risk by
starting with SciPy-compatible semantics and adding benchmarks before changing
core behavior.

## Likely Source Touch Points

Verify file names against the current SciPy checkout before editing:

| Area | Likely files | Prototype purpose |
| --- | --- | --- |
| Python wrapper/API validation | `scipy/ndimage/_interpolation.py` | Keep public `zoom` signature stable; route only private experiments if maintainers agree. |
| C interpolation loop | `scipy/ndimage/src/ni_interpolation.c` | Study current coordinate mapping, boundary handling and spline coefficient accumulation. |
| Spline filter/poles | `scipy/ndimage/src/ni_splines.c`, `ni_splines.h` | Compare pole recursion and prefilter setup against splineops filters. |
| Tests | `scipy/ndimage/tests/test_ndimage.py` or nearby ndimage interpolation tests | Add parity tests before optimization. |
| Benchmarks | SciPy ASV benchmark suite | Add `zoom` benchmarks for 2-D/3-D order-1/order-3 cases. |

## Minimum Prototype

1. Run the splineops SciPy-only audit to identify exact-ish first-pass rows:

   ```shell
   python scripts/audit_scipy_zoom.py \\
     --profile standard \\
     --variant-profile focused \\
     --output-dir /tmp/splineops_scipy_zoom_audit_standard
   ```

2. Add ASV benchmarks for `ndimage.zoom`:
   - 2-D `float32` and `float64`, order 1 and 3,
   - 3-D volume order 1 and 3,
   - `mode="mirror"`, `prefilter=True` for order > 1,
   - downsample and mixed-axis zoom rows.
3. Add a private C helper that precomputes per-output bases/weights for one
   axis and reuses it across lines.
4. Add a batched line loop for contiguous or near-contiguous lines while
   preserving SciPy's current coordinate and boundary behavior.
5. Prove bitwise or tolerance-level parity with the existing SciPy path.
6. Benchmark before proposing any public API changes.

## What To Avoid In The First Patch

- Do not introduce oblique antialiasing as a public SciPy option in the first
  patch.
- Do not change `zoom` output sizes, `grid_mode`, boundary behavior or default
  dtype behavior.
- Do not port splineops environment-variable tuning knobs directly into SciPy.
- Do not mix performance refactors with semantics changes.

## Follow-On RFC If Phase 1 Works

If SciPy maintainers are interested after the same-semantics prototype, propose
a separate RFC for projection antialiasing. That RFC should include method
names, mathematical references, quality metrics, and migration guidance for
users currently approximating antialiasing with external filters.
"""


def render_pytorch_bridge(*, tag: str, report_md: Path | None) -> str:
    return f"""# PyTorch Bridge Plan: Custom Operator First

Tag: `{tag}`

PyTorch is a harder upstream target than SciPy because `interpolate` is part of
the ATen/native operator ecosystem and carries tensor layout, device dispatch,
autograd and export expectations. The right first artifact is a standalone
custom operator, not a direct upstream PR.

## Phase 0: External Custom Op

Build a small package exposing:

```python
spline_resize(input, size=None, scale_factor=None, method="cubic-antialiasing")
```

Initial constraints:

- CPU only.
- Contiguous `float32`/`float64`.
- 2-D and 3-D spatial tensors, with explicit batch/channel handling.
- Inference/preprocessing use first; autograd can be a follow-up via the
  adjoint linear operator if the use case demands it.

## Phase 1: Evidence

- Compare against `torch.nn.functional.interpolate` only where semantics are
  clearly explained.
- Lead with quality/antialiasing and N-D CPU behavior, not a blanket speed
  claim.
- Use the same benchmark evidence report when useful:
  {markdown_path(report_md)}

## Phase 2: Upstream Discussion

Only after the custom op is stable:

1. Ask whether PyTorch maintainers prefer a new mode, an extension package, or
   no upstream surface.
2. If upstream is viable, map it to ATen/native conventions and dispatch keys.
3. Add CPU kernels first; CUDA is a separate project.
4. Define backward/autograd semantics explicitly.

## Links

- PyTorch `interpolate` documentation: {PYTORCH_INTERPOLATE_DOC}
- PyTorch custom C++ operator guide: {PYTORCH_CUSTOM_OP_DOC}
- ATen native functions overview: {PYTORCH_ATEN_NATIVE_DOC}
"""


def render_readme(
    *,
    tag: str,
    state: dict[str, str],
    benchmark_report: Path | None,
    semantics: Path,
    correctness: Path,
    scipy_rfc: Path,
    scipy_prototype: Path,
    pytorch_bridge: Path,
    benchmark_manifest: Path | None,
    commands_txt: Path,
) -> str:
    status = (state["status"] or "clean").replace("\n", "<br>")
    return f"""# Resize Upstream Evidence Package

Tag: `{tag}`

Repository state:

| Field | Value |
| --- | --- |
| Branch | `{state["branch"]}` |
| Commit | `{state["short_commit"]}` |
| Version | `{state["version"]}` |
| Installed version | `{state["installed_version"]}` |
| Worktree status at generation | `{status}` |
| Python | `{platform.python_version()}` |
| Platform | `{platform.platform()}` |

## Files

| Purpose | Path |
| --- | --- |
| Benchmark report | {markdown_path(benchmark_report)} |
| Benchmark manifest | {markdown_path(benchmark_manifest)} |
| Semantics matrix | `{semantics}` |
| Correctness policy | `{correctness}` |
| SciPy RFC draft | `{scipy_rfc}` |
| SciPy prototype patch plan | `{scipy_prototype}` |
| PyTorch custom-op bridge plan | `{pytorch_bridge}` |
| Command log | `{commands_txt}` |

## Recommended Use

1. Use the semantics matrix and correctness policy to keep claims precise.
2. Use the benchmark report as quantitative evidence.
3. Run `scripts/audit_scipy_zoom.py` before opening a SciPy issue/RFC; use only
   exact-ish first-pass candidate rows as evidence for existing
   `scipy.ndimage.zoom` semantics.
4. Open a SciPy issue/RFC before writing a large patch.
5. Keep the PyTorch path as a custom operator until there is separate evidence
   for tensor/autograd/device integration.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for generated artifacts. Defaults to /tmp/splineops_resize_upstream_<tag>.",
    )
    parser.add_argument("--tag", default=timestamp_tag())
    parser.add_argument(
        "--benchmark-profile",
        choices=("smoke", "default", "oblique-pr"),
        default="smoke",
        help="Profile passed to benchmark_resize_pr.py.",
    )
    parser.add_argument("--exact-rel-l2", type=float, default=1e-5)
    parser.add_argument(
        "--skip-benchmarks",
        action="store_true",
        help="Only generate Markdown package files; do not run benchmark_resize_pr.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print benchmark command without executing it; Markdown files are still written.",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Allow benchmark execution when installed splineops metadata differs from pyproject.toml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or Path(f"/tmp/splineops_resize_upstream_{args.tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    commands: list[list[str]] = []
    benchmark_report: Path | None = None
    benchmark_manifest: Path | None = None
    project_version = read_project_version()
    installed_version = installed_package_version()

    if (
        not args.skip_benchmarks
        and not args.dry_run
        and not args.allow_version_mismatch
        and installed_version != project_version
    ):
        raise SystemExit(
            "Installed splineops metadata does not match pyproject.toml: "
            f"installed={installed_version!r}, pyproject={project_version!r}. "
            "Run `python -m pip install -e .` in the benchmark environment, "
            "or pass --allow-version-mismatch if this is intentional."
        )

    if not args.skip_benchmarks:
        title = (
            "Resize Upstream Smoke Evidence Report"
            if args.benchmark_profile == "smoke"
            else "Resize Upstream Evidence Report"
        )
        benchmark_cmd = [
            sys.executable,
            str(SCRIPT_DIR / "benchmark_resize_pr.py"),
            "--profile",
            args.benchmark_profile,
            "--output-dir",
            str(output_dir),
            "--tag",
            args.tag,
            "--title",
            title,
            "--exact-rel-l2",
            str(args.exact_rel_l2),
        ]
        if args.dry_run:
            benchmark_cmd.append("--dry-run")
        commands.append(benchmark_cmd)
        run_command(benchmark_cmd, dry_run=args.dry_run)
        if not args.dry_run:
            benchmark_report = output_dir / f"resize_pr_report_{args.tag}.md"
        benchmark_manifest = output_dir / f"resize_pr_manifest_{args.tag}.json"

    semantics = output_dir / f"resize_upstream_semantics_{args.tag}.md"
    correctness = output_dir / f"resize_upstream_correctness_{args.tag}.md"
    scipy_prototype = output_dir / f"resize_scipy_prototype_plan_{args.tag}.md"
    scipy_rfc = output_dir / f"resize_scipy_rfc_{args.tag}.md"
    pytorch_bridge = output_dir / f"resize_pytorch_bridge_{args.tag}.md"
    readme = output_dir / f"README_resize_upstream_{args.tag}.md"
    manifest = output_dir / f"resize_upstream_manifest_{args.tag}.json"
    commands_txt = output_dir / f"resize_upstream_commands_{args.tag}.txt"

    semantics.write_text(
        render_semantics_matrix(tag=args.tag, exact_rel_l2=args.exact_rel_l2),
        encoding="utf-8",
    )
    correctness.write_text(
        render_correctness_policy(tag=args.tag, exact_rel_l2=args.exact_rel_l2),
        encoding="utf-8",
    )
    scipy_prototype.write_text(
        render_scipy_prototype_plan(tag=args.tag),
        encoding="utf-8",
    )
    scipy_rfc.write_text(
        render_scipy_rfc(
            tag=args.tag,
            report_md=benchmark_report,
            semantics_md=semantics,
            correctness_md=correctness,
            prototype_md=scipy_prototype,
        ),
        encoding="utf-8",
    )
    pytorch_bridge.write_text(
        render_pytorch_bridge(tag=args.tag, report_md=benchmark_report),
        encoding="utf-8",
    )

    state = repo_state()
    readme.write_text(
        render_readme(
            tag=args.tag,
            state=state,
            benchmark_report=benchmark_report,
            semantics=semantics,
            correctness=correctness,
            scipy_rfc=scipy_rfc,
            scipy_prototype=scipy_prototype,
            pytorch_bridge=pytorch_bridge,
            benchmark_manifest=benchmark_manifest,
            commands_txt=commands_txt,
        ),
        encoding="utf-8",
    )

    commands_txt.write_text(
        "\n".join(shell_join(cmd) for cmd in commands) + ("\n" if commands else ""),
        encoding="utf-8",
    )
    generated = {
        "readme": str(readme),
        "semantics": str(semantics),
        "correctness": str(correctness),
        "scipy_rfc": str(scipy_rfc),
        "scipy_prototype_plan": str(scipy_prototype),
        "pytorch_bridge": str(pytorch_bridge),
        "commands": str(commands_txt),
        "benchmark_report": None if benchmark_report is None else str(benchmark_report),
        "benchmark_manifest": None if benchmark_manifest is None else str(benchmark_manifest),
    }
    manifest.write_text(
        json.dumps(
            {
                "tag": args.tag,
                "repo": state,
                "options": {
                    "benchmark_profile": args.benchmark_profile,
                    "exact_rel_l2": args.exact_rel_l2,
                    "skip_benchmarks": args.skip_benchmarks,
                    "dry_run": args.dry_run,
                    "allow_version_mismatch": args.allow_version_mismatch,
                },
                "generated": generated,
                "commands": commands,
                "sources": {
                    "scipy_zoom_doc": SCIPY_ZOOM_DOC,
                    "scipy_contributor_doc": SCIPY_CONTRIBUTOR_DOC,
                    "scipy_interpolation_source": SCIPY_INTERPOLATION_SRC,
                    "pytorch_interpolate_doc": PYTORCH_INTERPOLATE_DOC,
                    "pytorch_custom_op_doc": PYTORCH_CUSTOM_OP_DOC,
                    "pytorch_aten_native_doc": PYTORCH_ATEN_NATIVE_DOC,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"upstream package: {output_dir}")
    print(f"readme: {readme}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
