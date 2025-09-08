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
    "dot",
    "seg",
    "arrow",
    "label",
    # ready-made diagrams
    "draw_standard_vs_scipy_pipeline",
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
    """Standard/SciPy pipeline diagram with extra space above the lowest sum."""
    xmin, xmax = -2.5, (46.5 if show_plus else 34.8)
    ymin, ymax = -2.5, 16.0
    fig, ax = figure_for_extents(xmin, xmax, ymin, ymax, width=width, ax=ax)

    # Left: Original
    box(ax, -2, 14.25, 4.25, 12.5, "Original Image", fontsize=12)
    arrow(ax, 4.25, 13.25, 8, 13.25)

    # Downsample circle ↓4
    circle(ax, 9, 13.25, 1.0, r"$\downarrow 4$", fontsize=18)
    arrow(ax, 10, 13.25, 14.5, 13.25)

    # Junctions and split
    dot(ax, 12, 13.25)
    seg(ax, 12, 13.25, 12, 8.25)
    arrow(ax, 12, 8.25, 14.5, 8.25)

    # Method boxes (wider: right edge = 21.75)
    ups = "\n$\\uparrow 4$" if include_upsample_labels else ""
    box(ax, 14.5, 14, 21.75, 12.25, f"Standard Interpolation{ups}", fontsize=12)
    box(ax, 14.5, 9,  21.75, 7.25,  f"SciPy Interpolation{ups}",    fontsize=12)

    # TensorSpline branch with matched Standard↔SciPy spacing (SciPy 8.25, Standard 13 ⇒ 4.75)
    ts_y = 8.25 - (13.0 - 8.25)  # 3.5
    seg(ax, 12, 8.25, 12, ts_y)
    arrow(ax, 12, ts_y, 14.5, ts_y)
    box(ax, 14.5, ts_y + 0.875, 21.75, ts_y - 0.875,
        f"TensorSpline Interpolation{ups}", fontsize=12)
    seg(ax, 21.75, ts_y, 32.25, ts_y)

    # Standard lane to the right
    seg(ax, 21.75, 13, 32.25, 13)
    dot(ax, 27.25, 13)
    dot(ax, 30, 13)
    dot(ax, 32.0, 13)              # tap point for the Standard↔TensorSpline sum
    arrow(ax, 30, 13, 30, 11.75)   # drop to the Standard↔SciPy sum

    # Standard + SciPy sum (unchanged)
    mid_cx, mid_cy, mid_r = 30.0, 10.75, 1.0
    circle(ax, mid_cx, mid_cy, mid_r, r"$\sum$", fontsize=18)
    seg(ax, mid_cx + 1.0, mid_cy, 33.5, mid_cy)
    arrow(ax, 33.5, mid_cy, 33.5, 0)
    label(ax, mid_cx - 0.7, mid_cy + mid_r + 0.6, r"$+$", fontsize=18)  # Standard side
    label(ax, mid_cx - 0.7, mid_cy - mid_r - 0.6, r"$-$", fontsize=18)  # SciPy side

    # Standard ± TensorSpline sum (centered between SciPy and TensorSpline rails)
    st_ts_cx, st_ts_cy, st_ts_r = 32.0, (8.25 + ts_y) / 2.0, 1.0  # y = 5.875
    circle(ax, st_ts_cx, st_ts_cy, st_ts_r, r"$\sum$", fontsize=18)
    label(ax, st_ts_cx - 0.7, st_ts_cy + st_ts_r + 0.2, r"$+$", fontsize=18)  # Standard side
    label(ax, st_ts_cx - 0.7, st_ts_cy - st_ts_r - 0.2, r"$-$", fontsize=18)  # TensorSpline side
    # feeds
    arrow(ax, 32.0, 13,      st_ts_cx, st_ts_cy + st_ts_r)   # from Standard
    arrow(ax, st_ts_cx, ts_y, st_ts_cx, st_ts_cy - st_ts_r)  # from TensorSpline
    # independent output column for this sum
    exit_x = 34.0
    seg(ax, st_ts_cx + st_ts_r, st_ts_cy, exit_x, st_ts_cy)
    arrow(ax, exit_x, st_ts_cy, exit_x, 0)

    # Bottom sum (Standard vs Original) — moved down for more space
    sum_cx, sum_cy, sum_r = 27.25, 1.25, 1.0  # ↓ from 2.0 to 1.25
    circle(ax, sum_cx, sum_cy, sum_r, r"$\sum$", fontsize=18)
    label(ax, sum_cx - sum_r - 0.7, sum_cy + 0.6, r"$+$", fontsize=18)   # Original side
    label(ax, sum_cx - 0.7,         sum_cy + sum_r + 0.6, r"$-$", fontsize=18)  # above the sum
    arrow(ax, sum_cx, sum_cy - sum_r, sum_cx, 0)

    # Feed from Standard branch directly into the bottom sum (unchanged)
    arrow(ax, 27.25, 13, sum_cx, sum_cy + sum_r)

    # Original lowest horizontal rail — align with new bottom sum y
    seg(ax, 5.5, 13.25, 5.5, sum_cy)
    dot(ax, 5.5, 13.25)
    arrow(ax, 5.5, sum_cy, 26.25, sum_cy)

    if show_separator:
        seg(ax, 10.75, 15.5, 10.75, 0.25, style="dashed", linewidth=1.2, zorder=1)

    box(ax, 23.75, -0.25, 34.25, -2, "Difference Images", fontsize=12)

    if show_plus:
        label(ax, 45.5, 6.25, "+", fontsize=28)

    fig.tight_layout(pad=0.4)
    plt.show()
    return fig, ax
