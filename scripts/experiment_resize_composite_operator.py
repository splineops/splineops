#!/usr/bin/env python3
"""
Probe whether resize's 1-D spline operator can be collapsed into a compact
precomputed matrix.

The Arrate/Unser projection framework is separable, so any fundamental
replacement for the per-axis operator should first show useful locality in 1-D.
This script builds dense 1-D response matrices for small signals and summarizes
threshold sparsity and energy outside local neighborhoods. It follows the
current plan policy: public zero-shift projections with analysis degree one or
greater use direct compact cross-Gram rows, while analysis degree zero retains
the finite-difference pipeline.

The historical ``coeff_after_diff`` stage label is retained for CSV
compatibility. It means the projected coefficient response before the output
solve; a direct plan performs no differentiation at that stage.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from splineops.resize._pycore.diff_integ import do_diff, do_integ  # noqa: E402
from splineops.resize._pycore.filters import (  # noqa: E402
    get_interpolation_coefficients,
    get_samples,
)
from splineops.resize._pycore.params import LSParams, Work1D  # noqa: E402
from splineops.resize._pycore.plan_1d import make_plan_1d  # noqa: E402
from splineops.resize._pycore.resize_1d import (  # noqa: E402
    _build_extension_inplace,
    _ensure_ws,
    resize_1d_ws,
)

METHOD_PARAMS: dict[str, tuple[int, int, int]] = {
    "linear": (1, -1, 1),
    "cubic": (3, -1, 3),
    "linear-antialiasing": (1, 0, 1),
    "cubic-antialiasing": (3, 1, 3),
}

STAGES_ALL = (
    "coeff_after_diff",
    "coeff_after_output_tail",
    "sample_to_sample",
)


@dataclass
class SummaryRow:
    method: str
    n: int
    out_n: int
    zoom: float
    stage: str
    threshold: float
    radius: int
    max_abs: float
    median_count: float
    max_count: int
    median_span: float
    max_span: int
    median_energy_outside: float
    max_energy_outside: float
    trunc_rel_l2_median: float
    trunc_rel_l2_max: float
    trunc_max_abs_median: float
    trunc_max_abs_max: float


def parse_csv_floats(value: str) -> list[float]:
    out: list[float] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(float(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one float")
    return out


def parse_csv_ints(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(v < 0 for v in out):
        raise argparse.ArgumentTypeError("radii must be non-negative")
    return out


def params_for_method(method: str, zoom: float, shift: float) -> LSParams:
    interp, analy, synthe = METHOD_PARAMS[method]
    return LSParams(
        interp_degree=interp,
        analy_degree=analy,
        synthe_degree=synthe,
        zoom=float(zoom),
        shift=float(shift),
    )


def visible_centers(n: int, out_n: int, p: LSParams) -> np.ndarray:
    if n > 1 and out_n > 1:
        effective_zoom = (out_n - 1) / float(n - 1)
        step = 1.0 / effective_zoom
    else:
        effective_zoom = 1.0
        step = 0.0

    shift = float(p.shift)
    if p.analy_degree >= 0:
        t = (p.analy_degree + 1.0) / 2.0
        shift += (t - np.floor(t)) * (1.0 / effective_zoom - 1.0)
    return step * np.arange(out_n, dtype=np.float64) + shift


def projection_from_coeff(coeff: np.ndarray, p: LSParams, tail: bool) -> np.ndarray:
    plan = make_plan_1d(coeff.size, p)
    ws = Work1D()
    _ensure_ws(ws, plan, coeff.size)

    ws.coeff[...] = coeff
    average = 0.0
    if p.analy_degree >= 0 and not plan.direct_projection:
        average = do_integ(ws.coeff, p.analy_degree + 1)

    if plan.direct_projection:
        source = ws.coeff
    else:
        _build_extension_inplace(ws.coeff, plan, ws)
        source = ws.ext_full
    if plan.win_len_max > 0 and plan.out_total > 0:
        np.take(source, plan.idx2d, out=ws.gather2d)
        np.multiply(plan.weights2d, ws.gather2d, out=ws.gather2d)
        np.sum(ws.gather2d, axis=1, out=ws.y)
    else:
        ws.y[:] = 0.0

    if p.analy_degree >= 0 and not plan.direct_projection:
        do_diff(ws.y, p.analy_degree + 1)
        ws.y += average

    if tail and p.analy_degree >= 0:
        corr_degree = p.analy_degree + p.synthe_degree + 1
        get_interpolation_coefficients(ws.y, corr_degree)
        get_samples(ws.y, p.synthe_degree)

    return ws.y[: plan.outN].copy()


def sample_to_sample(x: np.ndarray, p: LSParams) -> np.ndarray:
    plan = make_plan_1d(x.size, p)
    ws = Work1D()
    return resize_1d_ws(x, p, plan, ws)


def build_matrix(n: int, p: LSParams, stage: str) -> np.ndarray:
    plan = make_plan_1d(n, p)
    mat = np.empty((plan.outN, n), dtype=np.float64)
    basis = np.zeros(n, dtype=np.float64)
    for col in range(n):
        basis[col] = 1.0
        if stage == "coeff_after_diff":
            # Historical artifact label: for direct plans this is the compact
            # cross-Gram response before the output solve, with no difference.
            mat[:, col] = projection_from_coeff(basis, p, tail=False)
        elif stage == "coeff_after_output_tail":
            mat[:, col] = projection_from_coeff(basis, p, tail=True)
        elif stage == "sample_to_sample":
            mat[:, col] = sample_to_sample(basis, p)
        else:
            raise ValueError(f"unknown stage: {stage}")
        basis[col] = 0.0
    return mat


def threshold_stats(
    mat: np.ndarray, threshold: float, abs_floor: float
) -> tuple[float, int, float, int]:
    counts: list[int] = []
    spans: list[int] = []
    for row in mat:
        limit = max(abs_floor, threshold * float(np.max(np.abs(row))))
        nz = np.flatnonzero(np.abs(row) > limit)
        counts.append(int(nz.size))
        spans.append(int(nz[-1] - nz[0] + 1) if nz.size else 0)
    return (
        float(statistics.median(counts)),
        int(max(counts, default=0)),
        float(statistics.median(spans)),
        int(max(spans, default=0)),
    )


def energy_outside_stats(
    mat: np.ndarray, centers: np.ndarray, radius: int
) -> tuple[float, float]:
    outside: list[float] = []
    cols = np.arange(mat.shape[1], dtype=np.float64)
    for row, center in zip(mat, centers):
        total = float(np.sum(row * row))
        if total == 0.0:
            outside.append(0.0)
            continue
        mask = np.abs(cols - center) > radius
        outside.append(float(np.sum(row[mask] * row[mask]) / total))
    return float(statistics.median(outside)), float(max(outside, default=0.0))


def truncation_stats(
    mat: np.ndarray,
    centers: np.ndarray,
    radius: int,
    samples: np.ndarray | None,
) -> tuple[float, float, float, float]:
    if samples is None:
        nan = float("nan")
        return nan, nan, nan, nan

    cols = np.arange(mat.shape[1], dtype=np.float64)
    truncated = mat.copy()
    for row_index, center in enumerate(centers):
        truncated[row_index, np.abs(cols - center) > radius] = 0.0

    reference = samples @ mat.T
    candidate = samples @ truncated.T
    diff = candidate - reference
    denom = np.maximum(np.linalg.norm(reference, axis=1), 1e-300)
    rel_l2 = np.linalg.norm(diff, axis=1) / denom
    max_abs = np.max(np.abs(diff), axis=1) if diff.size else np.array([0.0])

    return (
        float(np.median(rel_l2)),
        float(np.max(rel_l2)),
        float(np.median(max_abs)),
        float(np.max(max_abs)),
    )


def summarize_matrix(
    method: str,
    n: int,
    zoom: float,
    p: LSParams,
    stage: str,
    mat: np.ndarray,
    thresholds: Iterable[float],
    radii: Iterable[int],
    abs_floor: float,
    truncation_trials: int,
    seed: int,
) -> list[SummaryRow]:
    centers = visible_centers(n, mat.shape[0], p)
    rows: list[SummaryRow] = []
    max_abs = float(np.max(np.abs(mat))) if mat.size else 0.0
    samples = None
    if truncation_trials > 0:
        rng = np.random.default_rng(seed)
        samples = rng.standard_normal((truncation_trials, mat.shape[1]))
    trunc_by_radius = {
        radius: truncation_stats(mat, centers, radius, samples) for radius in radii
    }
    for threshold in thresholds:
        median_count, max_count, median_span, max_span = threshold_stats(
            mat, threshold, abs_floor
        )
        for radius in radii:
            median_energy, max_energy = energy_outside_stats(mat, centers, radius)
            trunc_rel_median, trunc_rel_max, trunc_abs_median, trunc_abs_max = (
                trunc_by_radius[radius]
            )
            rows.append(
                SummaryRow(
                    method=method,
                    n=n,
                    out_n=mat.shape[0],
                    zoom=zoom,
                    stage=stage,
                    threshold=threshold,
                    radius=radius,
                    max_abs=max_abs,
                    median_count=median_count,
                    max_count=max_count,
                    median_span=median_span,
                    max_span=max_span,
                    median_energy_outside=median_energy,
                    max_energy_outside=max_energy,
                    trunc_rel_l2_median=trunc_rel_median,
                    trunc_rel_l2_max=trunc_rel_max,
                    trunc_max_abs_median=trunc_abs_median,
                    trunc_max_abs_max=trunc_abs_max,
                )
            )
    return rows


def print_summary(rows: list[SummaryRow]) -> None:
    current: tuple[str, str] | None = None
    for row in rows:
        key = (row.method, row.stage)
        if key != current:
            current = key
            print()
            print(
                f"{row.method} {row.stage} "
                f"n={row.n} out={row.out_n} zoom={row.zoom:g} max_abs={row.max_abs:.6g}"
            )
        print(
            f"  thr={row.threshold:.0e} rad={row.radius:2d} "
            f"count med/max={row.median_count:5.1f}/{row.max_count:<4d} "
            f"span med/max={row.median_span:5.1f}/{row.max_span:<4d} "
            f"energy_out med/max={row.median_energy_outside:.3e}/{row.max_energy_outside:.3e}"
        )
        if not np.isnan(row.trunc_rel_l2_median):
            print(
                f"       trunc rel_l2 med/max={row.trunc_rel_l2_median:.3e}/"
                f"{row.trunc_rel_l2_max:.3e} "
                f"max_abs med/max={row.trunc_max_abs_median:.3e}/"
                f"{row.trunc_max_abs_max:.3e}"
            )


def write_csv(path: Path, rows: list[SummaryRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(SummaryRow.__dataclass_fields__)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=64, help="1-D input length.")
    parser.add_argument("--zoom", type=float, default=0.37, help="Resize zoom factor.")
    parser.add_argument("--shift", type=float, default=0.0, help="Input shift.")
    parser.add_argument(
        "--method",
        choices=["all", *METHOD_PARAMS],
        default="all",
        help="Method to probe.",
    )
    parser.add_argument(
        "--thresholds",
        type=parse_csv_floats,
        default=parse_csv_floats("1e-6,1e-8,1e-10"),
        help="Relative thresholds for sparsity summaries.",
    )
    parser.add_argument(
        "--radii",
        type=parse_csv_ints,
        default=parse_csv_ints("4,8,16,32"),
        help="Input-space radii for outside-energy summaries.",
    )
    parser.add_argument(
        "--abs-floor",
        type=float,
        default=1e-14,
        help="Absolute floor used with relative thresholds.",
    )
    parser.add_argument(
        "--truncation-trials",
        type=int,
        default=0,
        help="Random inputs for reporting radius-truncated operator error.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output-csv", type=Path)
    args = parser.parse_args()

    if args.n <= 0:
        print("--n must be positive", file=sys.stderr)
        return 2
    if args.zoom <= 0:
        print("--zoom must be positive", file=sys.stderr)
        return 2
    if args.truncation_trials < 0:
        print("--truncation-trials must be non-negative", file=sys.stderr)
        return 2

    methods = list(METHOD_PARAMS) if args.method == "all" else [args.method]
    rows: list[SummaryRow] = []

    for method in methods:
        p = params_for_method(method, args.zoom, args.shift)
        stages = list(STAGES_ALL)
        if p.analy_degree < 0:
            stages = ["sample_to_sample"]
        for stage in stages:
            mat = build_matrix(args.n, p, stage)
            rows.extend(
                summarize_matrix(
                    method=method,
                    n=args.n,
                    zoom=args.zoom,
                    p=p,
                    stage=stage,
                    mat=mat,
                    thresholds=args.thresholds,
                    radii=args.radii,
                    abs_floor=args.abs_floor,
                    truncation_trials=args.truncation_trials,
                    seed=args.seed,
                )
            )

    print_summary(rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
        print()
        print(f"wrote {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
