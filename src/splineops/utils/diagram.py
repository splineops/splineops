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

# -----------------------------------------------------------------------------#
# Ready-made: Standard vs SciPy pipeline
# -----------------------------------------------------------------------------#

def draw_standard_vs_scipy_pipeline(
    *,
    show_separator: bool = True,
    show_plus: bool = False,
    include_upsample_labels: bool = True,
    width: float = 12.0,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """Recreate the Standard/SciPy pipeline diagram (TikZ-faithful).

    Parameters
    ----------
    show_separator : bool
        Draw the vertical dashed divider at x=10.75.
    show_plus : bool
        Show the far-right "+" (also expands x-limits to include it).
    include_upsample_labels : bool
        Include “↑ 4” as a second line inside each method rectangle.
    width : float
        Figure width in inches (height auto-computed to match data aspect).
    ax : matplotlib Axes or None
        Draw onto an existing axes or create a new figure/axes.

    Returns
    -------
    fig, ax
    """
    # data extents (expand if plus requested)
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

    # Method boxes (with optional ↑4 inside)
    ups = "\n$\\uparrow 4$" if include_upsample_labels else ""
    box(ax, 14.5, 14, 20.75, 12.25, f"Standard Interpolation{ups}", fontsize=12)
    box(ax, 14.5, 9, 20.75, 7.25,   f"SciPy Interpolation{ups}",    fontsize=12)

    # Top branch to the right
    seg(ax, 20.75, 13, 32.25, 13)
    dot(ax, 27.25, 13)
    dot(ax, 30, 13)
    arrow(ax, 30, 13, 30, 11.75)

    # Middle sum node
    circle(ax, 30, 10.75, 1.0, r"$\sum$", fontsize=18)
    seg(ax, 31, 10.75, 33.5, 10.75)
    arrow(ax, 33.5, 10.75, 33.5, 0)

    # Bottom right plumbing
    circle(ax, 25, 5.5, 1.0, r"$\sum$", fontsize=18)
    circle(ax, 27.25, 2, 1.0, r"$\sum$", fontsize=18)
    arrow(ax, 25, 4.5, 25, 0)
    arrow(ax, 27.25, 1, 27.25, 0)

    # Lower minus node and feed
    circle(ax, 27.25, 5.5, 1.0, r"$-$", fontsize=20)
    arrow(ax, 27.25, 13, 27.25, 6.5)
    arrow(ax, 27.25, 4.5, 27.25, 3)

    # Middle horizontal from SciPy
    seg(ax, 23.75, 8.25, 32.25, 8.25)
    circle(ax, 22.75, 8.25, 1.0, r"$-$", fontsize=20)
    arrow(ax, 20.75, 8.25, 21.75, 8.25)
    dot(ax, 25, 8.25)
    dot(ax, 30, 8.25)
    arrow(ax, 25, 8.25, 25, 6.5)
    arrow(ax, 30, 8.25, 30, 9.75)

    # Left vertical trunk
    seg(ax, 5.5, 13.25, 5.5, 2)
    dot(ax, 5.5, 13.25)
    dot(ax, 5.5, 5.5)
    arrow(ax, 5.5, 5.5, 24, 5.5)
    arrow(ax, 5.5, 2, 26.25, 2)

    # Optional dashed separator
    if show_separator:
        seg(ax, 10.75, 15.5, 10.75, 0.25, style="dashed", linewidth=1.2, zorder=1)

    # Collector box
    box(ax, 23.75, -0.25, 34.25, -2, "Difference Images", fontsize=12)

    # Optional far-right plus
    if show_plus:
        label(ax, 45.5, 6.25, "+", fontsize=28)

    fig.tight_layout(pad=0.4)
    plt.show()
    return fig, ax
