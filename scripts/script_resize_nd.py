# splineops/scripts/script_resize_nd.py
"""
ND resize demo (3D & 4D) with cross-section plots
==================================================

This script generates synthetic 3D and 4D volumes, resizes them with
splineops, and visualizes 2D cross-sections before and after resizing.

- 3D volume: shape (Z, Y, X)
- 4D volume: shape (T, Z, Y, X)  (T = "time" or batch)

For the 3D case, we show a central Z-slice.
For the 4D case, we fix a time index T and show a central Z-slice.

You can tweak:
  - ZOOMS_3D / ZOOMS_4D for different per-axis zooms
  - METHOD_3D / METHOD_4D to use "cubic", "cubic-fast_antialiasing",
    "cubic-best_antialiasing", etc.
"""

from __future__ import annotations

import os
import sys
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# Use C++ acceleration if available
os.environ.setdefault("SPLINEOPS_ACCEL", "auto")

# Default storage dtype for synthetic volumes (change to np.float64 if desired)
DTYPE = np.float32

# Import splineops (works when run directly or as module)
try:
    from splineops.resize import resize as sp_resize
except Exception:
    # Fallback: add ../src to sys.path if running from repo root
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    src_dir   = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from splineops.resize import resize as sp_resize


# --------------------------------------------
# Configuration: methods & zooms
# --------------------------------------------

# Resize methods (see splineops.resize.METHOD_MAP)
#   - "cubic"                    → interpolation, degree 3
#   - "cubic-fast_antialiasing"  → oblique, degree 3
#   - "cubic-best_antialiasing"  → least-squares, degree 3
METHOD_3D = "cubic-best_antialiasing"
METHOD_4D = "cubic-best_antialiasing"

# Per-axis zooms:
# 3D: (Z, Y, X)
ZOOM_3D: Tuple[float, float, float] = (0.7, 1.4, 0.5)

# 4D: (T, Z, Y, X) – keep T unchanged, resize spatial axes only
ZOOM_4D: Tuple[float, float, float, float] = (1.0, 0.6, 1.3, 0.8)


# --------------------------------------------
# Helpers
# --------------------------------------------

def _make_3d_volume(shape=(48, 64, 40), seed: int = 0) -> np.ndarray:
    """
    Synthetic 3D volume (Z, Y, X). Slightly structured random field so
    slices look a bit nicer than pure white noise.
    """
    rng = np.random.default_rng(seed)
    z, y, x = shape
    base = rng.random(shape, dtype=np.float64)

    # Add a smooth bump + sinusoidal modulation
    zz, yy, xx = np.meshgrid(
        np.linspace(-1, 1, z),
        np.linspace(-1, 1, y),
        np.linspace(-1, 1, x),
        indexing="ij",
    )
    r2 = zz**2 + yy**2 + xx**2
    bump = np.exp(-3.0 * r2)
    wave = 0.2 * (np.sin(4 * np.pi * xx) + np.cos(3 * np.pi * yy))

    vol = 0.5 * base + 0.3 * bump + 0.2 * wave
    return np.asarray(vol, dtype=DTYPE)


def _make_4d_volume(shape=(8, 48, 64, 40), seed: int = 1) -> np.ndarray:
    """
    Synthetic 4D volume (T, Z, Y, X). Think of T as time or batch.
    Each time slice is a slightly different 3D field.
    """
    rng = np.random.default_rng(seed)
    t, z, y, x = shape
    vol = np.empty(shape, dtype=np.float64)

    for ti in range(t):
        # Gradually change seed / phase per time slice
        vol[ti] = _make_3d_volume((z, y, x), seed=seed + ti * 13).astype(np.float64)

    # Normalize to [0, 1]
    vmin = float(vol.min())
    vmax = float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    return vol.astype(DTYPE, copy=False)


def _show_slice(ax, img2d: np.ndarray, title: str) -> None:
    ax.imshow(img2d, cmap="viridis", interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


# --------------------------------------------
# Main demo
# --------------------------------------------

def main(argv=None) -> int:
    # ----- 3D volume -----
    vol3d = _make_3d_volume()
    vol3d_res = sp_resize(vol3d, zoom_factors=ZOOM_3D, method=METHOD_3D)

    z0 = vol3d.shape[0] // 2
    z1 = vol3d_res.shape[0] // 2
    slice3d_orig = vol3d[z0, :, :]
    slice3d_res  = vol3d_res[z1, :, :]

    # ----- 4D volume -----
    vol4d = _make_4d_volume()
    vol4d_res = sp_resize(vol4d, zoom_factors=ZOOM_4D, method=METHOD_4D)

    t0 = vol4d.shape[0] // 2          # time index (unchanged by zoom)
    z0_4d = vol4d.shape[1] // 2       # original Z
    z1_4d = vol4d_res.shape[1] // 2   # resized Z (different length)

    slice4d_orig = vol4d[t0, z0_4d, :, :]
    slice4d_res  = vol4d_res[t0, z1_4d, :, :]

    # ----------------------------------------
    # Plot cross-sections
    # ----------------------------------------
    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(8, 7),
        constrained_layout=True,
    )

    _show_slice(
        axes[0, 0],
        slice3d_orig,
        f"3D original (Z={z0}, shape={vol3d.shape})",
    )
    _show_slice(
        axes[0, 1],
        slice3d_res,
        f"3D resized (Z={z1}, zoom={ZOOM_3D}, shape={vol3d_res.shape})",
    )

    _show_slice(
        axes[1, 0],
        slice4d_orig,
        f"4D original (T={t0}, Z={z0_4d}, shape={vol4d.shape})",
    )
    _show_slice(
        axes[1, 1],
        slice4d_res,
        f"4D resized (T={t0}, Z={z1_4d}, zoom={ZOOM_4D}, shape={vol4d_res.shape})",
    )

    fig.suptitle(
        f"ND resize cross-sections\n"
        f"3D method={METHOD_3D}, 4D method={METHOD_4D}",
        fontsize=12,
    )
    plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
