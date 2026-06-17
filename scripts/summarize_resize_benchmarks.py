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


def _float_or(row: dict[str, str], key: str, default: float) -> float:
    value = _float(row, key)
    return default if value is None else value


def _fmt(value: float, *, digits: int = 2, suffix: str = "") -> str:
    if math.isnan(value):
        return "n/a"
    return f"`{value:.{digits}f}{suffix}`"


def _fmt_sci(value: float, *, digits: int = 2) -> str:
    if math.isnan(value):
        return "n/a"
    return f"`{value:.{digits}e}`"


def _is_oblique_antialiasing(row: dict[str, str]) -> bool:
    return row.get("method", "").endswith("-antialiasing")


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["No usable rows."]
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def _native_speedup_stats(
    path: Path,
) -> tuple[list[float], dict[str, list[float]], dict[str, int]]:
    rows = _rows(path)
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)

    ratios: list[float] = []
    thread_winners: dict[str, int] = {}
    buckets: dict[str, list[float]] = {
        "Linear interpolation": [],
        "Cubic interpolation": [],
        "Oblique antialiasing presets": [],
        "3-D": [],
        "3-D oblique antialiasing": [],
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
        method = best_native.get("method", "")
        is_oblique = method.endswith("-antialiasing")
        if case.startswith("3d_"):
            buckets["3-D"].append(speedup)
            if is_oblique:
                buckets["3-D oblique antialiasing"].append(speedup)
        if is_oblique:
            buckets["Oblique antialiasing presets"].append(speedup)
        elif method == "cubic":
            buckets["Cubic interpolation"].append(speedup)
        elif method == "linear":
            buckets["Linear interpolation"].append(speedup)

    return ratios, buckets, thread_winners


def _native_report_lines(path: Path) -> list[str]:
    ratios, buckets, thread_winners = _native_speedup_stats(path)
    lines = [
        "## Native vs Python Fallback",
        "",
        f"Artifact: `{path}`",
        "",
        "Speedup is computed as Python fallback median divided by the best native median for each case.",
        "",
    ]
    rows: list[list[str]] = []
    for label, values in [
        ("All native/Python overlaps", ratios),
        *buckets.items(),
    ]:
        if not values:
            continue
        rows.append([
            label,
            str(len(values)),
            _fmt(_median(values), suffix="x"),
            _fmt(_mean(values), suffix="x"),
            _fmt(min(values), suffix="x"),
            _fmt(max(values), suffix="x"),
        ])
    lines.extend(_markdown_table(
        ["Scope", "Cases", "Median", "Mean", "Min", "Max"],
        rows,
    ))
    if thread_winners:
        lines.extend([
            "",
            "Best native thread setting by case:",
            "",
        ])
        lines.extend(_markdown_table(
            ["Thread setting", "Winning cases"],
            [[f"`{thread}`", str(count)] for thread, count in sorted(thread_winners.items())],
        ))
    lines.append("")
    return lines


def _libraries_report_lines(path: Path, *, exact_rel_l2: float) -> list[str]:
    rows = _rows(path)
    lines = [
        "## Cross-Library Comparison",
        "",
        f"Artifact: `{path}`",
        "",
        "Here, speed values greater than `1.0x` mean the other backend was faster than splineops.",
        "",
    ]

    backend_rows: list[list[str]] = []
    exact_rows: list[list[str]] = []
    oblique_rows: list[list[str]] = []
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
        if speeds:
            backend_rows.append([
                backend,
                str(len(speeds)),
                f"{sum(v > 1.0 for v in speeds)}/{len(speeds)}",
                _fmt(_median(speeds), suffix="x"),
                _fmt(_mean(speeds), suffix="x"),
                _fmt_sci(_median(rel)),
            ])

        exact = [
            r for r in br
            if (_float(r, "rel_l2_diff") or float("inf")) < exact_rel_l2
        ]
        exact_speeds = [
            v for r in exact
            if (v := _float(r, "speedup_vs_splineops")) is not None
        ]
        if exact_speeds:
            exact_rows.append([
                backend,
                str(len(exact_speeds)),
                f"{sum(v > 1.0 for v in exact_speeds)}/{len(exact_speeds)}",
                _fmt(_median(exact_speeds), suffix="x"),
                _fmt(_mean(exact_speeds), suffix="x"),
            ])

        oblique = [r for r in br if _is_oblique_antialiasing(r)]
        oblique_speeds = [
            v for r in oblique
            if (v := _float(r, "speedup_vs_splineops")) is not None
        ]
        oblique_rel = [
            v for r in oblique
            if (v := _float(r, "rel_l2_diff")) is not None
        ]
        if oblique_speeds:
            oblique_rows.append([
                backend,
                str(len(oblique_speeds)),
                f"{sum(v > 1.0 for v in oblique_speeds)}/{len(oblique_speeds)}",
                _fmt(_median(oblique_speeds), suffix="x"),
                _fmt(_mean(oblique_speeds), suffix="x"),
                _fmt_sci(_median(oblique_rel)),
            ])

    lines.extend(_markdown_table(
        [
            "Backend",
            "Comparable cases",
            "Faster than splineops",
            "Median speed",
            "Mean speed",
            "Median rel-L2",
        ],
        backend_rows,
    ))
    lines.extend([
        "",
        f"Exact-ish rows use `rel_l2_diff < {exact_rel_l2:g}`.",
        "",
    ])
    lines.extend(_markdown_table(
        ["Backend", "Exact-ish cases", "Faster than splineops", "Median speed", "Mean speed"],
        exact_rows,
    ))
    if oblique_rows:
        lines.extend([
            "",
            "Oblique antialiasing rows use splineops `*-antialiasing` presets. Other libraries are contextual baselines here, not exact oblique projection implementations.",
            "",
        ])
        lines.extend(_markdown_table(
            [
                "Backend",
                "Oblique cases",
                "Faster than splineops",
                "Median speed",
                "Mean speed",
                "Median rel-L2",
            ],
            oblique_rows,
        ))
    lines.append("")
    return lines


def _legacy_report_lines(path: Path, reference: Path) -> list[str]:
    legacy_rows = _rows(path)
    ref_rows = _rows(reference)
    splineops = {
        r["case"]: _float(r, "median_ms")
        for r in ref_rows
        if r.get("backend") == "splineops" and r.get("status") == "ok"
    }

    ratios: list[float] = []
    for row in legacy_rows:
        case = row["case"]
        legacy_ms = _float(row, "median_ms")
        ref_ms = splineops.get(case)
        if legacy_ms is None or ref_ms is None or ref_ms <= 0.0:
            continue
        ratios.append(legacy_ms / ref_ms)

    lines = [
        "## Legacy Java Baseline",
        "",
        f"Legacy artifact: `{path}`",
        f"Reference artifact: `{reference}`",
        "",
        "Speedup is computed as legacy Java median divided by splineops median on overlapping 2-D cases.",
        "",
    ]
    rows: list[list[str]] = []
    if ratios:
        rows.append([
            str(len(ratios)),
            _fmt(_median(ratios), suffix="x"),
            _fmt(_mean(ratios), suffix="x"),
            _fmt(min(ratios), suffix="x"),
            _fmt(max(ratios), suffix="x"),
        ])
    lines.extend(_markdown_table(
        ["Cases", "Median", "Mean", "Min", "Max"],
        rows,
    ))
    lines.append("")
    return lines


def _ab_report_lines(path: Path) -> list[str]:
    rows = _rows(path)
    usable = [
        r for r in rows
        if (v := _float(r, "median_speedup")) is not None and v > 0.0
    ]
    lines = [
        f"## A/B Sweep: {path.name}",
        "",
        f"Artifact: `{path}`",
        "",
    ]
    if not usable:
        lines.extend(["No usable A/B rows.", ""])
        return lines

    flag = usable[0].get("flag", "<unknown>")
    a_value = usable[0].get("a_value", "<a>")
    b_value = usable[0].get("b_value", "<b>")
    speedups = [_float(r, "median_speedup") or 0.0 for r in usable]
    failed = [r for r in usable if r.get("passed_check") not in ("True", "true", "1")]
    lines.extend([
        f"`{flag}`: `{a_value}` to `{b_value}`. Speedup values greater than `1.0x` favor `{b_value}`.",
        "",
    ])
    lines.extend(_markdown_table(
        ["Rows", "Median", "Mean", "Wins >1.03x", "Losses <0.97x", "Failed checks"],
        [[
            str(len(usable)),
            _fmt(_median(speedups), digits=3, suffix="x"),
            _fmt(_mean(speedups), digits=3, suffix="x"),
            str(sum(v > 1.03 for v in speedups)),
            str(sum(v < 0.97 for v in speedups)),
            str(len(failed)),
        ]],
    ))

    thread_rows: list[list[str]] = []
    for threads in sorted({r.get("threads", "<unknown>") for r in usable}):
        group = [
            _float(r, "median_speedup") or 0.0
            for r in usable
            if r.get("threads", "<unknown>") == threads
        ]
        thread_rows.append([
            f"`{threads}`",
            str(len(group)),
            _fmt(_median(group), digits=3, suffix="x"),
            _fmt(_mean(group), digits=3, suffix="x"),
            str(sum(v > 1.03 for v in group)),
            str(sum(v < 0.97 for v in group)),
        ])
    if thread_rows:
        lines.extend(["", "By thread setting:", ""])
        lines.extend(_markdown_table(
            ["Threads", "Rows", "Median", "Mean", "Wins", "Losses"],
            thread_rows,
        ))

    method_rows: list[list[str]] = []
    for method in sorted({r.get("method", "<unknown>") for r in usable}):
        group = [
            _float(r, "median_speedup") or 0.0
            for r in usable
            if r.get("method", "<unknown>") == method
        ]
        method_rows.append([
            f"`{method}`",
            str(len(group)),
            _fmt(_median(group), digits=3, suffix="x"),
            _fmt(_mean(group), digits=3, suffix="x"),
        ])
    if method_rows:
        lines.extend(["", "By method:", ""])
        lines.extend(_markdown_table(
            ["Method", "Rows", "Median", "Mean"],
            method_rows,
        ))

    ranked = sorted(
        usable,
        key=lambda r: _float(r, "median_speedup") or 0.0,
    )
    lowest = ranked[:3]
    highest = [row for row in reversed(ranked[-3:]) if row not in lowest]
    extrema_rows = [
        [
            "Lowest",
            row.get("case", "<case>"),
            f"`{row.get('threads', '<threads>')}`",
            _fmt(_float(row, "median_speedup") or float("nan"), digits=3, suffix="x"),
        ]
        for row in lowest
    ]
    extrema_rows.extend(
        [
            [
                "Highest",
                row.get("case", "<case>"),
                f"`{row.get('threads', '<threads>')}`",
                _fmt(_float(row, "median_speedup") or float("nan"), digits=3, suffix="x"),
            ]
            for row in highest
        ]
    )
    lines.extend(["", "Lowest and highest movements:", ""])
    lines.extend(_markdown_table(
        ["Type", "Case", "Threads", "Median speedup"],
        extrema_rows,
    ))
    lines.append("")
    return lines


def _plan_report_lines(path: Path) -> list[str]:
    rows = _rows(path)
    usable = [
        r for r in rows
        if (_float(r, "median_speedup") is not None or
            _float(r, "median_into_speedup") is not None)
    ]
    lines = [
        "## ResizePlan Reuse",
        "",
        f"Artifact: `{path}`",
        "",
    ]
    if not usable:
        lines.extend(["No usable ResizePlan rows.", ""])
        return lines

    speedups = [
        v for r in usable
        if (v := _float(r, "median_speedup")) is not None
    ]
    into_speedups = [
        v for r in usable
        if (v := _float(r, "median_into_speedup")) is not None
    ]
    lines.extend(_markdown_table(
        ["Mode", "Cases", "Median speedup", "Mean speedup"],
        [
            [
                "Plan",
                str(len(speedups)),
                _fmt(_median(speedups), digits=3, suffix="x"),
                _fmt(_mean(speedups), digits=3, suffix="x"),
            ],
            [
                "Plan with output",
                str(len(into_speedups)),
                _fmt(_median(into_speedups), digits=3, suffix="x"),
                _fmt(_mean(into_speedups), digits=3, suffix="x"),
            ],
        ],
    ))
    case_rows: list[list[str]] = []
    for row in usable:
        max_abs_diff = _float(row, "max_abs_diff")
        case_rows.append([
            row.get("case", "<case>"),
            f"`{row.get('method', '<method>')}`",
            f"`{row.get('dtype', '<dtype>')}`",
            _fmt(_float(row, "median_speedup") or float("nan"), digits=3, suffix="x"),
            _fmt(_float(row, "median_into_speedup") or float("nan"), digits=3, suffix="x"),
            _fmt_sci(max_abs_diff if max_abs_diff is not None else float("nan")),
        ])
    lines.extend(["", "Per case:", ""])
    lines.extend(_markdown_table(
        ["Case", "Method", "Dtype", "Plan speedup", "Plan output speedup", "Max abs diff"],
        case_rows,
    ))
    lines.append("")
    return lines


def _projection_methods_report_lines(path: Path) -> list[str]:
    rows = _rows(path)
    usable = [
        r for r in rows
        if _float(r, "oblique_speedup_vs_least_squares") is not None
    ]
    lines = [
        "## Projection Method Comparison",
        "",
        f"Artifact: `{path}`",
        "",
        "This compares splineops equal-degree least-squares projection against the oblique `*-antialiasing` method family.",
        "",
    ]
    if not usable:
        lines.extend(["No usable projection-method rows.", ""])
        return lines

    table_rows: list[list[str]] = []
    for degree in sorted({r.get("degree", "<degree>") for r in usable}):
        group = [r for r in usable if r.get("degree", "<degree>") == degree]
        speedups = [
            _float(r, "oblique_speedup_vs_least_squares") or 0.0
            for r in group
        ]
        psnr_ls = psnr_oblique = psnr_interp = 0
        ssim_rows = 0
        ssim_oblique = 0
        for row in group:
            psnr_values = {
                "LS": _float_or(row, "roundtrip_psnr_least_squares", -float("inf")),
                "Oblique": _float_or(row, "roundtrip_psnr_oblique", -float("inf")),
                "Interp": _float_or(row, "roundtrip_psnr_interpolation", -float("inf")),
            }
            best_psnr = max(psnr_values, key=psnr_values.get)
            if best_psnr == "LS":
                psnr_ls += 1
            elif best_psnr == "Oblique":
                psnr_oblique += 1
            else:
                psnr_interp += 1

            ssim_values = {
                "LS": _float(row, "roundtrip_ssim_least_squares"),
                "Oblique": _float(row, "roundtrip_ssim_oblique"),
                "Interp": _float(row, "roundtrip_ssim_interpolation"),
            }
            if all(value is not None for value in ssim_values.values()):
                ssim_rows += 1
                if ssim_values["Oblique"] >= max(ssim_values["LS"], ssim_values["Interp"]):  # type: ignore[arg-type]
                    ssim_oblique += 1

        table_rows.append([
            f"`{degree}`",
            str(len(group)),
            f"{sum(v > 1.0 for v in speedups)}/{len(speedups)}",
            _fmt(_median(speedups), suffix="x"),
            f"{psnr_ls}/{psnr_oblique}/{psnr_interp}",
            f"{ssim_oblique}/{ssim_rows}" if ssim_rows else "n/a",
        ])

    lines.extend(_markdown_table(
        [
            "Degree",
            "Cases",
            "Oblique faster",
            "Median oblique speedup",
            "PSNR wins LS/Oblique/Interp",
            "SSIM oblique wins",
        ],
        table_rows,
    ))
    lines.extend([
        "",
        "Interpretation: least-squares can retain a small PSNR edge on controlled small round trips, but oblique is the production default because it is faster, lower-order, and more robust on long lines.",
        "",
    ])
    return lines


def build_markdown_report(
    *,
    title: str,
    native: Path | None,
    libraries: Path | None,
    legacy: Path | None,
    legacy_reference: Path | None,
    ab: list[Path],
    plan: Path | None,
    projection_methods: Path | None,
    exact_rel_l2: float,
) -> str:
    artifacts: list[list[str]] = []
    if native is not None:
        artifacts.append(["Native/Python", f"`{native}`"])
    if libraries is not None:
        artifacts.append(["Libraries", f"`{libraries}`"])
    if legacy is not None:
        artifacts.append(["Legacy Java", f"`{legacy}`"])
    if legacy_reference is not None:
        artifacts.append(["Legacy reference", f"`{legacy_reference}`"])
    for path in ab:
        artifacts.append(["A/B", f"`{path}`"])
    if plan is not None:
        artifacts.append(["ResizePlan", f"`{plan}`"])
    if projection_methods is not None:
        artifacts.append(["Projection methods", f"`{projection_methods}`"])

    lines = [
        f"# {title}",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(_markdown_table(["Kind", "Path"], artifacts))
    lines.append("")

    if native is not None:
        lines.extend(_native_report_lines(native))
    if libraries is not None:
        lines.extend(_libraries_report_lines(libraries, exact_rel_l2=exact_rel_l2))
    if legacy is not None and legacy_reference is not None:
        lines.extend(_legacy_report_lines(legacy, legacy_reference))
    for path in ab:
        lines.extend(_ab_report_lines(path))
    if plan is not None:
        lines.extend(_plan_report_lines(plan))
    if projection_methods is not None:
        lines.extend(_projection_methods_report_lines(projection_methods))

    lines.extend([
        "## PR Interpretation",
        "",
        "- Use the native/Python section to justify same-algorithm acceleration.",
        "- Treat splineops `*-antialiasing` rows as oblique projection presets; do not describe them as equal-degree least-squares defaults.",
        "- Use exact-ish SciPy or PyTorch rows from the library comparison when arguing about like-for-like semantics.",
        "- Treat OpenCV, scikit-image and non-exact PyTorch rows as contextual image-resize baselines, because they can use different coordinate, boundary, antialiasing and dtype behavior.",
        "- Use A/B sections to defend individual default-on knobs and to identify rows that need another pass before an upstream PR.",
        "",
    ])
    return "\n".join(lines).rstrip() + "\n"


def summarize_native(path: Path) -> None:
    rows = _rows(path)
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)

    ratios: list[float] = []
    thread_winners: dict[str, int] = {}
    buckets: dict[str, list[float]] = {
        "linear interpolation": [],
        "cubic interpolation": [],
        "oblique antialiasing": [],
        "3d": [],
        "3d oblique antialiasing": [],
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
        method = best_native.get("method", "")
        is_oblique = method.endswith("-antialiasing")
        if case.startswith("3d_"):
            buckets["3d"].append(speedup)
            if is_oblique:
                buckets["3d oblique antialiasing"].append(speedup)
        if is_oblique:
            buckets["oblique antialiasing"].append(speedup)
        elif method == "cubic":
            buckets["cubic interpolation"].append(speedup)
        elif method == "linear":
            buckets["linear interpolation"].append(speedup)

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

    report = sub.add_parser("report", help="Build a Markdown resize benchmark report.")
    report.add_argument("--title", default="Resize Benchmark Report")
    report.add_argument("--native", type=Path, help="benchmark_resize_native.py CSV output.")
    report.add_argument("--libraries", type=Path, help="benchmark_resize_libraries.py CSV output.")
    report.add_argument("--legacy", type=Path, help="Legacy Java CSV output.")
    report.add_argument(
        "--legacy-reference",
        type=Path,
        help="Library CSV with splineops rows used as the legacy comparison reference.",
    )
    report.add_argument(
        "--ab",
        action="append",
        default=[],
        type=Path,
        help="benchmark_resize_ab.py CSV output. May be passed more than once.",
    )
    report.add_argument("--plan", type=Path, help="benchmark_resize_plan.py CSV output.")
    report.add_argument(
        "--projection-methods",
        type=Path,
        help="benchmark_resize_projection_methods.py CSV output.",
    )
    report.add_argument("--exact-rel-l2", type=float, default=1e-5)
    report.add_argument("--output", type=Path, help="Write Markdown to this path.")

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
    elif args.kind == "report":
        if not any([
            args.native,
            args.libraries,
            args.legacy,
            args.ab,
            args.plan,
            args.projection_methods,
        ]):
            parser.error("report requires at least one artifact argument")
        if args.legacy is not None and args.legacy_reference is None:
            parser.error("report --legacy requires --legacy-reference")
        report_text = build_markdown_report(
            title=args.title,
            native=args.native,
            libraries=args.libraries,
            legacy=args.legacy,
            legacy_reference=args.legacy_reference,
            ab=args.ab,
            plan=args.plan,
            projection_methods=args.projection_methods,
            exact_rel_l2=args.exact_rel_l2,
        )
        if args.output is not None:
            args.output.write_text(report_text, encoding="utf-8")
            print(f"wrote report: {args.output}")
        else:
            print(report_text, end="")
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
