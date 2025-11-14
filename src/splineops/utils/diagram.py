# splineops/src/splineops/utils/diagram.py

"""
splineops.utils.diagram
=======================
Lightweight diagram helpers for example gallery/tutorials (Matplotlib-only).

Design goals
------------
- No external deps (only Matplotlib).
- Coordinates are in *data units* so you can port TikZ coordinates verbatim.
- Helpers are stateless; pass an Axes or create one via `figure_for_extents`.
- Functions return created artists where it’s useful.

Typical usage
-------------
>>> from splineops.utils.diagram import draw_standard_vs_scipy_pipeline
>>> draw_standard_vs_scipy_pipeline()  # captured by Sphinx-Gallery

"""

from __future__ import annotations
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Circle as _Circle

__all__ = [
    # primitives
    "figure_for_extents",
    "box",
    "circle",
    "capsule",
    "dot",
    "seg",
    "arrow",
    "label",
    # ready-made diagrams
    "draw_standard_vs_scipy_pipeline",
    "draw_standard_vs_leastsq_pipeline",
    "draw_leastsq_vs_oblique_pipeline",
    "draw_two_method_comparisons",
]

# -----------------------------------------------------------------------------#
# Canvas / primitives
# -----------------------------------------------------------------------------#

def figure_for_extents(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    *,
    width: float = 12.0,
    hide_axes: bool = True,
    equal_aspect: bool = True,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Create (or configure) an Axes covering given data extents.

    Height is computed to preserve the data aspect so shapes remain undistorted.
    If *ax* is provided, limits and aspect are applied to it and its figure is
    returned unchanged.
    """
    ratio = (xmax - xmin) / (ymax - ymin)
    height = width / ratio
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if hide_axes:
        ax.axis("off")
    return fig, ax


def box(
    ax: Axes,
    x1: float, y1: float, x2: float, y2: float,
    label_text: Optional[str] = None,
    *,
    fontsize: int = 12,
    linewidth: float = 1.6,
    facecolor: str = "white",
    edgecolor: str = "black",
    rounding: float = 0.18,
):
    """Rounded rectangle given opposite corners (TikZ-style)."""
    x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
    y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
    w, h = x_hi - x_lo, y_hi - y_lo
    r = FancyBboxPatch(
        (x_lo, y_lo), w, h,
        boxstyle=f"round,pad={rounding},rounding_size={rounding}",
        linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(r)
    if label_text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, label_text,
                ha="center", va="center", fontsize=fontsize)
    return r

def capsule(
    ax: Axes,
    x1: float, y1: float, x2: float, y2: float,
    label_text: Optional[str] = None,
    *,
    fontsize: int = 12,
    linewidth: float = 1.6,
    facecolor: str = "white",
    edgecolor: str = "black",
    padding: float = 0.12,
):
    x_lo, x_hi = (x1, x2) if x1 <= x2 else (x2, x1)
    y_lo, y_hi = (y1, y2) if y1 <= y2 else (y2, y1)
    w, h = x_hi - x_lo, y_hi - y_lo
    r = min(w, h) * 0.5  # semicircular ends

    patch = FancyBboxPatch(
        (x_lo, y_lo), w, h,
        boxstyle=f"round,pad={padding},rounding_size={r}",
        linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor,
    )
    ax.add_patch(patch)
    if label_text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, label_text,
                ha="center", va="center", fontsize=fontsize)
    return patch

def circle(
    ax: Axes,
    cx: float, cy: float, radius: float,
    label_text: Optional[str] = None,
    *,
    fontsize: int = 16,
    linewidth: float = 1.6,
    edgecolor: str = "black",
    facecolor: Optional[str] = None,
):
    c = _Circle((cx, cy), radius, fill=facecolor is not None,
                linewidth=linewidth, edgecolor=edgecolor,
                facecolor=facecolor or "none")
    ax.add_patch(c)
    if label_text is not None:
        ax.text(cx, cy, label_text, ha="center", va="center", fontsize=fontsize)
    return c


def dot(ax: Axes, x: float, y: float, *, size: float = 4.5, color: str = "black"):
    return ax.plot([x], [y], marker="o", markersize=size, color=color)[0]


def seg(
    ax: Axes,
    x1: float, y1: float, x2: float, y2: float,
    *,
    style: str = "solid",
    linewidth: float = 1.6,
    color: str = "black",
    zorder: int = 2,
):
    return ax.plot([x1, x2], [y1, y2], linestyle=style,
                   linewidth=linewidth, color=color, zorder=zorder)[0]


def arrow(
    ax: Axes,
    x1: float, y1: float, x2: float, y2: float,
    *,
    linewidth: float = 1.6,
    color: str = "black",
):
    """Data→data arrow via annotate (gives nicer heads than quiver)."""
    return ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=linewidth,
                        shrinkA=0, shrinkB=0, color=color),
    )


def label(
    ax: Axes,
    x: float, y: float, text: str,
    *,
    fontsize: int = 12,
    ha: str = "center",
    va: str = "center",
):
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize)

def draw_standard_vs_scipy_pipeline(
    *,
    show_separator: bool = True,
    show_plus: bool = False,
    include_upsample_labels: bool = True,
    width: float = 12.0,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Standard/SciPy pipeline with TensorSpline branch, equal rail spacing,
    independent outputs, and collector arrows that stop before the box."""
    # canvas
    xmin, xmax = -2.5, (46.5 if show_plus else 34.8)
    ymin, ymax = -3.0, 16.0
    fig, ax = figure_for_extents(xmin, xmax, ymin, ymax, width=width, ax=ax)

    # layout constants
    y_std, y_scipy, y_ts = 13.0, 8.25, 3.5
    box_left = 14.5
    box_right = 21.75
    box_gap = 0.25  # inbound arrows stop before method box edges

    # Left: Original
    capsule(ax, -2, 14.25, 4.25, 12.5, "Original Image", fontsize=12)
    arrow(ax, 4.25, 13.25, 8, 13.25)

    # Downsample circle ↓4
    circle(ax, 9, 13.25, 1.0, r"$\downarrow 4$", fontsize=18)
    # into Standard box (stop before the edge)
    arrow(ax, 10, 13.25, box_left - box_gap, 13.25)
    # label the outgoing (resized) signal from the ↓4 stage
    label(
        ax,
        (10 + (box_left - box_gap)) / 2.0,  # midpoint of the segment
        13.25 + 0.6,                        # a bit above the line
        "Resized",
        fontsize=12,
    )

    # Junctions and split to lower rails
    dot(ax, 12, 13.25)
    seg(ax, 12, 13.25, 12, y_scipy)
    # dot at the beginning of the arrow into SciPy
    dot(ax, 12, y_scipy)
    # into SciPy box (stop before the edge)
    arrow(ax, 12, y_scipy, box_left - box_gap, y_scipy)

    # Method boxes (wider right edge = 21.75)
    ups = "\n$\\uparrow 4$" if include_upsample_labels else ""
    box(ax, box_left, 14, box_right, 12.25, f"Standard Interpolation{ups}", fontsize=12)
    box(ax, box_left, 9,  box_right, 7.25,  f"SciPy Interpolation{ups}",    fontsize=12)

    # TensorSpline branch (rail spacing matched to Standard↔SciPy)
    seg(ax, 12, y_scipy, 12, y_ts)
    # dot at the beginning of the arrow into TensorSpline
    dot(ax, 12, y_ts)
    # into TensorSpline box (stop before the edge)
    arrow(ax, 12, y_ts, box_left - box_gap, y_ts)
    box(ax, box_left, y_ts + 0.875, box_right, y_ts - 0.875,
        f"TensorSpline Interpolation{ups}", fontsize=12)

    # Rails to the right of the three boxes
    seg(ax, box_right, y_std,   32.25, y_std)   # Standard rail
    seg(ax, box_right, y_scipy, 32.25, y_scipy) # SciPy rail
    seg(ax, box_right, y_ts,    32.25, y_ts)    # TensorSpline rail
    # Labels over the outgoing rails from each interpolation ("Recovered")
    rail_label_dx, rail_label_dy = 1.75, 0.6
    label(ax, box_right + rail_label_dx, y_std   + rail_label_dy, "Recovered", fontsize=12)
    label(ax, box_right + rail_label_dx, y_scipy + rail_label_dy, "Recovered", fontsize=12)
    label(ax, box_right + rail_label_dx, y_ts    + rail_label_dy, "Recovered", fontsize=12)

    # Taps / junction dots
    dot(ax, 27.25, y_std)
    dot(ax, 30.00, y_std)
    dot(ax, 32.00, y_std)   # tap for Standard↔TensorSpline sum
    dot(ax, 30.00, y_scipy) # SciPy→sum junction
    dot(ax, 32.00, y_ts)    # TensorSpline→sum junction

    # Standard + SciPy sum
    mid_cx, mid_cy, mid_r = 30.0, 10.75, 1.0
    circle(ax, mid_cx, mid_cy, mid_r, r"$\sum$", fontsize=18)
    arrow(ax, 30.0, y_std,   mid_cx, mid_cy + mid_r)  # from Standard ↓
    arrow(ax, 30.0, y_scipy, mid_cx, mid_cy - mid_r)  # from SciPy ↑
    seg(ax, mid_cx + mid_r, mid_cy, 33.5, mid_cy)
    dot(ax, 33.5, mid_cy)
    label(ax, mid_cx - 0.7, mid_cy + mid_r + 0.6, r"$+$", fontsize=18)
    label(ax, mid_cx - 0.7, mid_cy - mid_r - 0.6, r"$-$", fontsize=18)

    # Standard ± TensorSpline sum (centered between y_scipy & y_ts)
    st_ts_cx, st_ts_cy, st_ts_r = 32.0, (y_scipy + y_ts) / 2.0, 1.0  # 5.875
    circle(ax, st_ts_cx, st_ts_cy, st_ts_r, r"$\sum$", fontsize=18)
    arrow(ax, 32.0, y_std, st_ts_cx, st_ts_cy + st_ts_r)    # from Standard ↓
    arrow(ax, st_ts_cx, y_ts, st_ts_cx, st_ts_cy - st_ts_r) # from TensorSpline ↑
    label(ax, st_ts_cx - 0.7, st_ts_cy + st_ts_r + 0.2, r"$+$", fontsize=18)
    label(ax, st_ts_cx - 0.7, st_ts_cy - st_ts_r - 0.2, r"$-$", fontsize=18)
    exit_x = 34.0
    seg(ax, st_ts_cx + st_ts_r, st_ts_cy, exit_x, st_ts_cy)
    dot(ax, exit_x, st_ts_cy)

    # Bottom sum (Standard vs Original) — lower for more space
    sum_cx, sum_cy, sum_r = 27.25, 1.25, 1.0
    circle(ax, sum_cx, sum_cy, sum_r, r"$\sum$", fontsize=18)
    label(ax, sum_cx - sum_r - 0.7, sum_cy + 0.6, r"$+$", fontsize=18)
    label(ax, sum_cx - 0.7,         sum_cy + sum_r + 0.6, r"$-$", fontsize=18)
    # Original lowest rail aligned to bottom sum y
    seg(ax, 5.5, 13.25, 5.5, sum_cy)
    dot(ax, 5.5, 13.25)
    dot(ax, 5.5, sum_cy)  
    arrow(ax, 5.5, sum_cy, 26.25, sum_cy)
    # tap from Standard to bottom sum
    arrow(ax, 27.25, y_std, sum_cx, sum_cy + sum_r)

    # Collector box & down arrows that STOP BEFORE the box
    collector_left, collector_right = 23.75, 34.25
    collector_top, collector_bottom = -1.0, -2.6
    collector_gap = 0.25  # how far above the box the arrows should end

    # arrows landing just above the collector top (no overlap)
    arrow(ax, 33.5,  mid_cy,        33.5,  collector_top + collector_gap)   # from Std↔SciPy sum
    arrow(ax, exit_x, st_ts_cy,     exit_x, collector_top + collector_gap)  # from Std↔TS sum
    arrow(ax, sum_cx, sum_cy - sum_r, sum_cx, collector_top + collector_gap) # from bottom sum

    # draw the collector box last
    capsule(ax, collector_left, collector_top, collector_right, collector_bottom,
        "Difference Images", fontsize=12)

    if show_separator:
        seg(ax, 10.75, 15.5, 10.75, 0.25, style="dashed", linewidth=1.2, zorder=1)

    if show_plus:
        label(ax, 45.5, 6.25, "+", fontsize=28)

    fig.tight_layout(pad=0.4)
    plt.show()
    return fig, ax

def draw_standard_vs_leastsq_pipeline(
    *, 
    include_upsample_labels: bool = True, 
    width: float = 12.0, 
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    return draw_two_method_comparisons(
        "Standard Interpolation",
        "Least-Squares Projection",
        include_downsample_labels=True,
        include_upsample_labels=include_upsample_labels,
        scale_factor=4,
        width=width,
        ax=ax,
    )

def draw_leastsq_vs_oblique_pipeline(
    *, 
    include_upsample_labels: bool = True, 
    width: float = 12.0, 
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    return draw_two_method_comparisons(
        "Least-Squares Projection",
        "Oblique Projection",
        include_downsample_labels=True,
        include_upsample_labels=include_upsample_labels,
        scale_factor=4,
        width=width,
        ax=ax,
    )

def draw_two_method_comparisons(
    top_method: str,
    bottom_method: str,
    *,
    include_downsample_labels: bool = True,
    include_upsample_labels: bool = True,
    scale_factor: int = 4,
    width: float = 12.0,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Draw a two-rail resampling pipeline diagram with identical layout to
    `draw_standard_vs_leastsq_pipeline`, but with configurable method names.

    Parameters
    ----------
    top_method : str
        Label for the top rail (e.g., "Standard Interpolation").
    bottom_method : str
        Label for the bottom rail (e.g., "Least-Squares Projection").
    include_downsample_labels : bool
        If True, shows "$\\downarrow K$" inside the left boxes.
    include_upsample_labels : bool
        If True, shows "$\\uparrow K$" inside the right boxes.
    scale_factor : int
        The K in up/down arrows (default 4).
    width : float
        Figure width in inches.
    ax : matplotlib.axes.Axes or None
        Optionally draw into an existing Axes.

    Returns
    -------
    (fig, ax)
    """
    fig, ax = figure_for_extents(-2.5, 35.0, -1.5, 16.0, width=width, ax=ax)

    rail_y = 13.25
    y_top  = rail_y
    y_bot  = 9.50
    bifurc_x = 7.25

    left_box_x1, left_box_x2   = 9.25, 16.50
    right_box_x1, right_box_x2 = 20.75, 28.25

    box_gap_in  = 0.30
    box_gap_out = 0.30

    ds = f"\n$\\downarrow {scale_factor}$" if include_downsample_labels else ""
    ups = f"\n$\\uparrow {scale_factor}$" if include_upsample_labels else ""

    # Original
    capsule(ax, -2, 14.25, 4.25, 12.5, "Original Image", fontsize=12)

    # First bifurcation (T-junction)
    arrow(ax, 4.25, rail_y, bifurc_x, rail_y); dot(ax, bifurc_x, rail_y)

    # --- Top rail ---
    arrow(ax, bifurc_x, y_top, left_box_x1 - box_gap_in, y_top)
    box(ax, left_box_x1, y_top + 1.0, left_box_x2, y_top - 1.0,
        f"{top_method}{ds}", fontsize=12)
    top_ds_out_x = left_box_x2 + box_gap_out
    top_us_in_x  = right_box_x1 - box_gap_in
    arrow(ax, top_ds_out_x, y_top, top_us_in_x, y_top)
    label(ax, (top_ds_out_x + top_us_in_x) / 2.0, y_top + 0.8, "Resized", fontsize=12)
    box(ax, right_box_x1, y_top + 1.0, right_box_x2, y_top - 1.0,
        f"{top_method}{ups}", fontsize=12)

    # --- Bottom rail ---
    seg(ax, bifurc_x, rail_y, bifurc_x, y_bot); dot(ax, bifurc_x, y_bot)
    arrow(ax, bifurc_x, y_bot, left_box_x1 - box_gap_in, y_bot)
    box(ax, left_box_x1, y_bot + 1.0, left_box_x2, y_bot - 1.0,
        f"{bottom_method}{ds}", fontsize=12)
    bot_ds_out_x = left_box_x2 + box_gap_out
    bot_us_in_x  = right_box_x1 - box_gap_in
    arrow(ax, bot_ds_out_x, y_bot, bot_us_in_x, y_bot)
    label(ax, (bot_ds_out_x + bot_us_in_x) / 2.0, y_bot + 0.8, "Resized", fontsize=12)
    box(ax, right_box_x1, y_bot + 1.0, right_box_x2, y_bot - 1.0,
        f"{bottom_method}{ups}", fontsize=12)

    # Outgoing rails
    seg(ax, right_box_x2, y_top, 33.25, y_top)
    seg(ax, right_box_x2, y_bot, 33.25, y_bot)

    # Taps + labels
    tap_x_top = right_box_x2 + 1.25
    tap_x_bot = right_box_x2 + 4.25
    dot(ax, tap_x_top, y_top); dot(ax, tap_x_bot, y_bot)
    label(ax, tap_x_top + 0.9, y_top + 0.6, "Recovered", fontsize=12)
    label(ax, tap_x_bot - 1.0, y_bot + 0.6, "Recovered", fontsize=12)

    # Bottom sums (keep layout consistent with previous function)
    sum_r     = 1.0
    sum_y_top = 3.75
    sum_y_bot = 6.25

    # Original’s vertical drops + T node + elbow dots
    seg(ax, 5.0, rail_y, 5.0, sum_y_bot)
    seg(ax, 5.0, rail_y, 5.0, sum_y_top)
    dot(ax, 5.0, rail_y); dot(ax, 5.0, sum_y_top); dot(ax, 5.0, sum_y_bot)

    # Top sum
    circle(ax, tap_x_top, sum_y_top, sum_r, r"$\sum$", fontsize=18)
    label(ax, tap_x_top - sum_r - 0.7, sum_y_top + 0.6, r"$+$", fontsize=18)
    label(ax, tap_x_top - 0.7,         sum_y_top + sum_r + 0.6, r"$-$", fontsize=18)
    arrow(ax, 5.0,       sum_y_top, tap_x_top - 1.0, sum_y_top)
    arrow(ax, tap_x_top, y_top,      tap_x_top,      sum_y_top + sum_r)

    # Bottom sum
    circle(ax, tap_x_bot, sum_y_bot, sum_r, r"$\sum$", fontsize=18)
    label(ax, tap_x_bot - sum_r - 0.7, sum_y_bot + 0.6, r"$+$", fontsize=18)
    label(ax, tap_x_bot - 0.7,         sum_y_bot + sum_r + 0.6, r"$-$", fontsize=18)
    arrow(ax, 5.0,     sum_y_bot, tap_x_bot - 1.0,  sum_y_bot)
    arrow(ax, tap_x_bot, y_bot,    tap_x_bot,       sum_y_bot + sum_r)

    # Collector
    collector_left, collector_right = 23.75, 34.0
    collector_top, collector_bottom = 1.00, -0.60
    collector_gap = 0.25
    arrow(ax, tap_x_top, sum_y_top - sum_r, tap_x_top, collector_top + collector_gap)
    arrow(ax, tap_x_bot, sum_y_bot - sum_r, tap_x_bot, collector_top + collector_gap)
    capsule(ax, collector_left, collector_top, collector_right, collector_bottom,
        "Difference Images", fontsize=12)

    fig.tight_layout(pad=0.4)
    plt.show()
    return fig, ax
