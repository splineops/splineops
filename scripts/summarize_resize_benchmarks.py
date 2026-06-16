#!/usr/bin/env python3
"""Summarize resize benchmark CSV artifacts.

The benchmark scripts intentionally write raw per-case CSVs. This helper turns
those artifacts into stable, reviewable summaries without relying on ad hoc
notebook snippets.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Iterable


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out):
        return None
    return out


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _median(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.median(data) if data else float("nan")


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return statistics.fmean(data) if data else float("nan")


def summarize_native(path: Path) -> None:
    rows = _rows(path)
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)

    ratios: list[float] = []
    thread_winners: dict[str, int] = {}
    buckets: dict[str, list[float]] = {
        "linear": [],
        "cubic": [],
        "antialiasing": [],
        "3d": [],
    }

    for case, group in sorted(by_case.items()):
        py_rows = [r for r in group if r.get("backend") == "python"]
        native_rows = [
            r for r in group
            if r.get("backend") == "native" and _float(r, "median_ms") is not None
        ]
        if not py_rows or not native_rows:
            continue
        py_ms = _float(py_rows[0], "median_ms")
        if py_ms is None:
            continue
        best_native = min(native_rows, key=lambda r: _float(r, "median_ms") or float("inf"))
        native_ms = _float(best_native, "median_ms")
        if native_ms is None or native_ms <= 0.0:
            continue

        speedup = py_ms / native_ms
        ratios.append(speedup)
        thread = best_native.get("threads", "<unknown>")
        thread_winners[thread] = thread_winners.get(thread, 0) + 1
        if case.startswith("3d_"):
            buckets["3d"].append(speedup)
        if "_aa_" in case:
            buckets["antialiasing"].append(speedup)
        elif "cubic" in case:
            buckets["cubic"].append(speedup)
        elif "linear" in case:
            buckets["linear"].append(speedup)

    print(f"native artifact: {path}")
    print(f"native/python overlaps: {len(ratios)}")
    if ratios:
        print(
            "native vs python fallback: "
            f"median={_median(ratios):.2f}x "
            f"mean={_mean(ratios):.2f}x "
            f"min={min(ratios):.2f}x "
            f"max={max(ratios):.2f}x"
        )
    if thread_winners:
        winners = " ".join(
            f"{thread}:{count}"
            for thread, count in sorted(thread_winners.items())
        )
        print(f"best native thread counts: {winners}")
    for name, values in buckets.items():
        if values:
            print(
                f"{name:12s} cases={len(values):2d} "
                f"median={_median(values):.2f}x "
                f"mean={_mean(values):.2f}x"
            )


def summarize_libraries(path: Path, *, exact_rel_l2: float) -> None:
    rows = _rows(path)
    print(f"library artifact: {path}")
    for backend in ("scipy", "skimage", "opencv", "torch"):
        br = [
            r for r in rows
            if r.get("backend") == backend and r.get("status") == "ok"
        ]
        speeds = [
            v for r in br
            if (v := _float(r, "speedup_vs_splineops")) is not None
        ]
        rel = [
            v for r in br
            if (v := _float(r, "rel_l2_diff")) is not None
        ]
        if not speeds:
            continue
        print(
            f"{backend:9s} cases={len(speeds):2d} "
            f"faster={sum(v > 1.0 for v in speeds):2d}/{len(speeds):2d} "
            f"median_speed={_median(speeds):.2f}x "
            f"mean_speed={_mean(speeds):.2f}x "
            f"median_rel_l2={_median(rel):.2e}"
        )

    for backend in ("scipy", "torch"):
        br = [
            r for r in rows
            if r.get("backend") == backend
            and r.get("status") == "ok"
            and (_float(r, "rel_l2_diff") or float("inf")) < exact_rel_l2
        ]
        speeds = [
            v for r in br
            if (v := _float(r, "speedup_vs_splineops")) is not None
        ]
        if speeds:
            print(
                f"exact-ish {backend:5s} cases={len(speeds):2d} "
                f"faster={sum(v > 1.0 for v in speeds):2d}/{len(speeds):2d} "
                f"median_speed={_median(speeds):.2f}x "
                f"mean_speed={_mean(speeds):.2f}x"
            )


def summarize_legacy(path: Path, reference: Path) -> None:
    legacy_rows = _rows(path)
    ref_rows = _rows(reference)
    splineops = {
        r["case"]: _float(r, "median_ms")
        for r in ref_rows
        if r.get("backend") == "splineops" and r.get("status") == "ok"
    }

    ratios: list[float] = []
    print(f"legacy artifact: {path}")
    print(f"reference artifact: {reference}")
    for row in legacy_rows:
        case = row["case"]
        legacy_ms = _float(row, "median_ms")
        ref_ms = splineops.get(case)
        if legacy_ms is None or ref_ms is None or ref_ms <= 0.0:
            continue
        ratios.append(legacy_ms / ref_ms)

    if ratios:
        print(
            "legacy vs splineops: "
            f"cases={len(ratios)} "
            f"median={_median(ratios):.2f}x "
            f"mean={_mean(ratios):.2f}x "
            f"min={min(ratios):.2f}x "
            f"max={max(ratios):.2f}x"
        )


def summarize_ab(path: Path) -> None:
    rows = _rows(path)
    usable = [
        r for r in rows
        if (v := _float(r, "median_speedup")) is not None and v > 0.0
    ]

    print(f"A/B artifact: {path}")
    if not usable:
        print("no usable A/B rows")
        return

    flag = usable[0].get("flag", "<unknown>")
    a_value = usable[0].get("a_value", "<a>")
    b_value = usable[0].get("b_value", "<b>")
    speedups = [_float(r, "median_speedup") or 0.0 for r in usable]
    failed = [r for r in usable if r.get("passed_check") not in ("True", "true", "1")]
    print(
        f"{flag}: {a_value} -> {b_value} "
        f"rows={len(usable)} "
        f"median={_median(speedups):.3f}x "
        f"mean={_mean(speedups):.3f}x "
        f"wins={sum(v > 1.03 for v in speedups)} "
        f"losses={sum(v < 0.97 for v in speedups)} "
        f"failed_checks={len(failed)}"
    )

    for threads in sorted({r.get("threads", "<unknown>") for r in usable}):
        group = [
            _float(r, "median_speedup") or 0.0
            for r in usable
            if r.get("threads", "<unknown>") == threads
        ]
        print(
            f"threads={threads:>7s} "
            f"rows={len(group):2d} "
            f"median={_median(group):.3f}x "
            f"mean={_mean(group):.3f}x "
            f"wins={sum(v > 1.03 for v in group)} "
            f"losses={sum(v < 0.97 for v in group)}"
        )

    for method in sorted({r.get("method", "<unknown>") for r in usable}):
        group = [
            _float(r, "median_speedup") or 0.0
            for r in usable
            if r.get("method", "<unknown>") == method
        ]
        print(
            f"method={method:20s} "
            f"rows={len(group):2d} "
            f"median={_median(group):.3f}x "
            f"mean={_mean(group):.3f}x"
        )

    ranked = sorted(
        usable,
        key=lambda r: _float(r, "median_speedup") or 0.0,
    )
    print("largest losses:")
    for row in ranked[:5]:
        print(
            f"  {row.get('case', '<case>'):30s} "
            f"thr={row.get('threads', '<threads>'):>7s} "
            f"speedup={(_float(row, 'median_speedup') or 0.0):.3f}x"
        )
    print("largest wins:")
    for row in reversed(ranked[-5:]):
        print(
            f"  {row.get('case', '<case>'):30s} "
            f"thr={row.get('threads', '<threads>'):>7s} "
            f"speedup={(_float(row, 'median_speedup') or 0.0):.3f}x"
        )


def _parse_key_fields(value: str | None, rows: list[dict[str, str]]) -> list[str]:
    if value:
        return [part.strip() for part in value.split(",") if part.strip()]

    fieldnames = set(rows[0].keys()) if rows else set()
    if {"backend", "case", "dtype", "threads"}.issubset(fieldnames):
        return ["backend", "case", "dtype", "threads"]
    if {"backend", "case"}.issubset(fieldnames):
        return ["backend", "case"]
    if "case" in fieldnames:
        return ["case"]
    raise ValueError("could not infer comparison key; pass --key")


def _filtered_rows(
    rows: list[dict[str, str]],
    *,
    metric: str,
    backends: set[str],
    include_failed: bool,
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        if backends and row.get("backend") not in backends:
            continue
        if not include_failed and row.get("status", "ok") != "ok":
            continue
        value = _float(row, metric)
        if value is None or value <= 0.0:
            continue
        out.append(row)
    return out


def _row_key(row: dict[str, str], key_fields: list[str]) -> tuple[str, ...]:
    return tuple(row.get(field, "") for field in key_fields)


def _best_rows_by_key(
    rows: list[dict[str, str]],
    *,
    key_fields: list[str],
    metric: str,
) -> tuple[dict[tuple[str, ...], dict[str, str]], int]:
    best: dict[tuple[str, ...], dict[str, str]] = {}
    duplicates = 0
    for row in rows:
        key = _row_key(row, key_fields)
        if key in best:
            duplicates += 1
            current = _float(row, metric) or float("inf")
            previous = _float(best[key], metric) or float("inf")
            if current >= previous:
                continue
        best[key] = row
    return best, duplicates


def _format_key(key_fields: list[str], key: tuple[str, ...]) -> str:
    return " ".join(
        f"{field}={value}" for field, value in zip(key_fields, key)
    )


def compare_artifacts(
    baseline: Path,
    current: Path,
    *,
    metric: str,
    key: str | None,
    backend: list[str],
    include_failed: bool,
    win_threshold: float,
    loss_threshold: float,
    top: int,
) -> None:
    base_rows_raw = _rows(baseline)
    cur_rows_raw = _rows(current)
    key_fields = _parse_key_fields(key, base_rows_raw or cur_rows_raw)
    backends = set(backend)

    base_rows = _filtered_rows(
        base_rows_raw,
        metric=metric,
        backends=backends,
        include_failed=include_failed,
    )
    cur_rows = _filtered_rows(
        cur_rows_raw,
        metric=metric,
        backends=backends,
        include_failed=include_failed,
    )
    base_by_key, base_duplicates = _best_rows_by_key(
        base_rows,
        key_fields=key_fields,
        metric=metric,
    )
    cur_by_key, cur_duplicates = _best_rows_by_key(
        cur_rows,
        key_fields=key_fields,
        metric=metric,
    )

    entries: list[tuple[float, tuple[str, ...], float, float]] = []
    for key_tuple, cur_row in cur_by_key.items():
        base_row = base_by_key.get(key_tuple)
        if base_row is None:
            continue
        base_value = _float(base_row, metric)
        cur_value = _float(cur_row, metric)
        if base_value is None or cur_value is None or cur_value <= 0.0:
            continue
        entries.append((base_value / cur_value, key_tuple, base_value, cur_value))

    missing_current = len(set(base_by_key) - set(cur_by_key))
    missing_baseline = len(set(cur_by_key) - set(base_by_key))
    speedups = [entry[0] for entry in entries]

    print(f"baseline artifact: {baseline}")
    print(f"current artifact:  {current}")
    print(f"metric={metric} key={','.join(key_fields)}")
    if backends:
        print(f"backend filter: {','.join(sorted(backends))}")
    print(
        f"overlaps={len(entries)} "
        f"baseline_only={missing_current} "
        f"current_only={missing_baseline} "
        f"duplicate_rows={base_duplicates + cur_duplicates}"
    )
    if not entries:
        return

    print(
        "baseline/current speedup: "
        f"median={_median(speedups):.3f}x "
        f"mean={_mean(speedups):.3f}x "
        f"min={min(speedups):.3f}x "
        f"max={max(speedups):.3f}x "
        f"wins={sum(v > win_threshold for v in speedups)} "
        f"losses={sum(v < loss_threshold for v in speedups)}"
    )

    if "backend" in key_fields:
        backend_index = key_fields.index("backend")
        for backend_name in sorted({key_tuple[backend_index] for _, key_tuple, _, _ in entries}):
            group = [
                speedup for speedup, key_tuple, _, _ in entries
                if key_tuple[backend_index] == backend_name
            ]
            print(
                f"backend={backend_name:9s} "
                f"rows={len(group):2d} "
                f"median={_median(group):.3f}x "
                f"mean={_mean(group):.3f}x "
                f"wins={sum(v > win_threshold for v in group)} "
                f"losses={sum(v < loss_threshold for v in group)}"
            )

    ranked = sorted(entries, key=lambda entry: entry[0])
    print("largest regressions:")
    for speedup, key_tuple, base_value, cur_value in ranked[:top]:
        print(
            f"  {_format_key(key_fields, key_tuple)} "
            f"{base_value:.3f}->{cur_value:.3f} ms "
            f"speedup={speedup:.3f}x"
        )
    print("largest wins:")
    for speedup, key_tuple, base_value, cur_value in reversed(ranked[-top:]):
        print(
            f"  {_format_key(key_fields, key_tuple)} "
            f"{base_value:.3f}->{cur_value:.3f} ms "
            f"speedup={speedup:.3f}x"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)

    native = sub.add_parser("native", help="Summarize benchmark_resize_native.py CSV output.")
    native.add_argument("csv", type=Path)

    libraries = sub.add_parser(
        "libraries",
        help="Summarize benchmark_resize_libraries.py CSV output.",
    )
    libraries.add_argument("csv", type=Path)
    libraries.add_argument("--exact-rel-l2", type=float, default=1e-5)

    legacy = sub.add_parser("legacy", help="Summarize legacy Java CSV against a library CSV.")
    legacy.add_argument("csv", type=Path)
    legacy.add_argument("--reference", type=Path, required=True)

    ab = sub.add_parser("ab", help="Summarize benchmark_resize_ab.py CSV output.")
    ab.add_argument("csv", type=Path)

    compare = sub.add_parser("compare", help="Compare two benchmark CSV artifacts.")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("current", type=Path)
    compare.add_argument("--metric", default="median_ms")
    compare.add_argument(
        "--key",
        help="Comma-separated key fields. Defaults to backend,case,dtype,threads; backend,case; or case.",
    )
    compare.add_argument(
        "--backend",
        action="append",
        default=[],
        help="Restrict comparison to a backend. May be passed more than once.",
    )
    compare.add_argument(
        "--include-failed",
        action="store_true",
        help="Include rows whose status column is not ok.",
    )
    compare.add_argument("--win-threshold", type=float, default=1.03)
    compare.add_argument("--loss-threshold", type=float, default=0.97)
    compare.add_argument("--top", type=int, default=5)

    args = parser.parse_args()
    if args.kind == "native":
        summarize_native(args.csv)
    elif args.kind == "libraries":
        summarize_libraries(args.csv, exact_rel_l2=args.exact_rel_l2)
    elif args.kind == "legacy":
        summarize_legacy(args.csv, args.reference)
    elif args.kind == "ab":
        summarize_ab(args.csv)
    elif args.kind == "compare":
        compare_artifacts(
            args.baseline,
            args.current,
            metric=args.metric,
            key=args.key,
            backend=args.backend,
            include_failed=args.include_failed,
            win_threshold=args.win_threshold,
            loss_threshold=args.loss_threshold,
            top=args.top,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
