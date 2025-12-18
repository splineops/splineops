# sphinx_gallery_start_ignore
# splineops/examples/02_resize/08_benchmarking_animation.py
# sphinx_gallery_end_ignore

"""
Benchmarking Animation
======================

For each Kodak test image, animate a zoom sweep comparing:

- SplineOps: cubic-antialiasing
- Competitor: PyTorch bicubic (antialias=True)   [if available]

Each frame shows (same layout as 02_resize_module_2d.py):
- Original (fixed)
- Downsampled pasted on a white canvas (per method)
- Recovered after round-trip (per method)
- Signed error map (rec - orig), normalized to [0, 1] with 0.5 = 0,
  using a GLOBAL max(|diff|) across all frames + both methods
  (benchmark-style stable contrast), plus a legend bar.

No animation export in this example (display only).
"""

# %%
# Imports
# -------

from __future__ import annotations

import os
from urllib.request import urlopen
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

from splineops.resize import resize as sp_resize

# Optional PyTorch competitor
try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    F = None      # type: ignore[assignment]


# %%
# Configuration
# -------------

DTYPE = np.float32

INTERVAL_MS = 900
TITLE_FS = 13

SPLINEOPS_LABEL = "SplineOps Antialiasing cubic"
COMPETITOR_LABEL = "PyTorch bicubic (AA)"

# Zoom factors (same idea as in 02_resize_module_2d.py)
zoom_low   = np.geomspace(0.01, 0.10, 10, endpoint=False)
zoom_dense = np.geomspace(0.10, 0.22, 15)
zoom_mid   = np.geomspace(0.22, 0.80, 10, endpoint=False)
zoom_top   = np.array([0.85, 0.90, 0.95, 1.0])
ZOOM_VALUES = np.unique(np.concatenate([zoom_low, zoom_dense, zoom_mid, zoom_top]))
ZOOM_VALUES = np.sort(ZOOM_VALUES)[::-1]  # 1.0 -> ... -> small

KODAK_BASE = "https://r0k.us/graphics/kodak/kodak"
KODAK_IMAGES = [
    ("kodim05", f"{KODAK_BASE}/kodim05.png"),
    ("kodim07", f"{KODAK_BASE}/kodim07.png"),
    ("kodim14", f"{KODAK_BASE}/kodim14.png"),
    ("kodim15", f"{KODAK_BASE}/kodim15.png"),
    ("kodim19", f"{KODAK_BASE}/kodim19.png"),
    ("kodim22", f"{KODAK_BASE}/kodim22.png"),
    ("kodim23", f"{KODAK_BASE}/kodim23.png"),
]

# Keep docs builds reasonable by default
MAX_IMAGES_DOCS = 1
if os.environ.get("SPLINEOPS_SPHINX_BUILD") == "1":
    KODAK_IMAGES = KODAK_IMAGES[:MAX_IMAGES_DOCS]


# %%
# Small helpers
# -------------

def _load_kodak_rgb01(url: str) -> np.ndarray:
    """Load Kodak image as RGB float32 in [0,1]."""
    with urlopen(url, timeout=10) as resp:
        img = Image.open(resp).convert("RGB")
    arr = np.asarray(img, dtype=np.float64) / 255.0
    return np.clip(arr, 0.0, 1.0).astype(DTYPE, copy=False)


def _to_u8(rgb01: np.ndarray) -> np.ndarray:
    return (np.clip(rgb01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _u8_to_gray01(u8_rgb: np.ndarray) -> np.ndarray:
    u = u8_rgb.astype(np.float32) / 255.0
    return (0.2989 * u[..., 0] + 0.5870 * u[..., 1] + 0.1140 * u[..., 2]).astype(np.float32)


def _paste_on_white_canvas(down_u8: np.ndarray, canvas_u8: np.ndarray) -> None:
    """In-place: paste down_u8 at top-left onto a full-size white canvas_u8."""
    canvas_u8[...] = 255
    h1, w1 = down_u8.shape[:2]
    canvas_u8[:h1, :w1, :] = down_u8


# %%
# Backends (round-trip)
# ---------------------

def _resize_rgb_splineops(img01: np.ndarray, z: float, *, method: str) -> np.ndarray:
    """Channel-wise splineops resize for RGB."""
    zoom_hw = (float(z), float(z))
    chs = [sp_resize(img01[..., c], zoom_factors=zoom_hw, method=method) for c in range(3)]
    out = np.stack(chs, axis=-1)
    return np.clip(out, 0.0, 1.0).astype(DTYPE, copy=False)


def _roundtrip_splineops_aa(orig01: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (down_u8, rec_u8) for splineops cubic-antialiasing."""
    H0, W0 = orig01.shape[:2]
    down01 = _resize_rgb_splineops(orig01, z, method="cubic-antialiasing")
    down_u8 = _to_u8(down01)

    # Recover to exact original size (channel-wise output_size)
    rec_ch = [sp_resize(down01[..., c], output_size=(H0, W0), method="cubic-antialiasing") for c in range(3)]
    rec01 = np.stack(rec_ch, axis=-1)
    rec_u8 = _to_u8(rec01)
    return down_u8, rec_u8


def _torch_resize_rgb_bicubic_aa(img01: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Torch bicubic resize with antialias=True. Returns float32 RGB in [0,1]."""
    assert _HAS_TORCH and F is not None

    x = torch.from_numpy(img01.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    y = F.interpolate(
        x,
        size=out_hw,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    out = y[0].permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _roundtrip_torch_bicubic_aa(orig01: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (down_u8, rec_u8) for torch bicubic antialias=True."""
    assert _HAS_TORCH

    H0, W0 = orig01.shape[:2]
    H1 = max(1, int(round(H0 * z)))
    W1 = max(1, int(round(W0 * z)))

    down01 = _torch_resize_rgb_bicubic_aa(orig01, (H1, W1))
    rec01  = _torch_resize_rgb_bicubic_aa(down01, (H0, W0))

    return _to_u8(down01), _to_u8(rec01)


# %%
# Animation builder
# -----------------

def make_two_method_animation(
    *,
    img_name: str,
    orig01: np.ndarray,
    zoom_values: np.ndarray,
    rt_a: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    rt_b: Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]],
    label_a: str,
    label_b: str,
    interval_ms: int = INTERVAL_MS,
    title_fs: int = TITLE_FS,
):
    """
    Build a 3x3 layout animation:

      [ Original | A down | B down ]
      [   blank  | A rec  | B rec  ]
      [  legend  | A err  | B err  ]

    where "err" is benchmark-style normalized signed diff:
        n = 0.5 + 0.5 * ( (rec_gray - orig_gray) / max_abs )
    """
    orig_u8 = _to_u8(orig01)
    H0, W0 = orig_u8.shape[:2]
    orig_gray01 = _u8_to_gray01(orig_u8)

    # --- Precompute down + recovered frames (u8) ---
    downs_a: List[np.ndarray] = []
    recs_a: List[np.ndarray] = []
    downs_b: List[np.ndarray] = []
    recs_b: List[np.ndarray] = []

    for z in zoom_values:
        d_a, r_a = rt_a(orig01, float(z))
        d_b, r_b = rt_b(orig01, float(z))
        downs_a.append(d_a)
        recs_a.append(r_a)
        downs_b.append(d_b)
        recs_b.append(r_b)

    # --- Compute global max(|diff|) across all frames + both methods ---
    max_abs = 0.0
    for r in recs_a:
        max_abs = max(max_abs, float(np.max(np.abs(_u8_to_gray01(r) - orig_gray01))))
    for r in recs_b:
        max_abs = max(max_abs, float(np.max(np.abs(_u8_to_gray01(r) - orig_gray01))))
    max_abs = max(max_abs, 1e-12)

    def diff_norm_u8(rec_u8: np.ndarray) -> np.ndarray:
        d = _u8_to_gray01(rec_u8) - orig_gray01
        n = 0.5 + 0.5 * (d / max_abs)
        return (np.clip(n, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # --- Layout ---
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    gs = fig.add_gridspec(nrows=3, ncols=3, width_ratios=[1.05, 1.0, 1.0])

    ax_orig     = fig.add_subplot(gs[0, 0])
    ax_blank    = fig.add_subplot(gs[1, 0])
    ax_leg_host = fig.add_subplot(gs[2, 0])

    ax_down_a = fig.add_subplot(gs[0, 1])
    ax_down_b = fig.add_subplot(gs[0, 2])
    ax_rec_a  = fig.add_subplot(gs[1, 1])
    ax_rec_b  = fig.add_subplot(gs[1, 2])
    ax_err_a  = fig.add_subplot(gs[2, 1])
    ax_err_b  = fig.add_subplot(gs[2, 2])

    for ax in (ax_orig, ax_blank, ax_leg_host, ax_down_a, ax_down_b, ax_rec_a, ax_rec_b, ax_err_a, ax_err_b):
        ax.axis("off")

    # Optional: one overall title per image
    fig.suptitle(f"{img_name} — Round-trip sweep (z: 1.0 → small)", fontsize=title_fs + 1)

    # --- Legend bar (benchmark-style) ---
    ax_leg_host.axis("off")
    leg = ax_leg_host.inset_axes([0.42, 0.05, 0.18, 0.90])
    leg.axis("off")
    H_leg, W_leg = 256, 16
    y = np.linspace(1.0, 0.0, H_leg, dtype=np.float32)
    legend_img = np.repeat(y[:, None], W_leg, axis=1)
    leg.imshow(legend_img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")

    ax_leg_host.text(0.62, 0.05, "-1", transform=ax_leg_host.transAxes, fontsize=9, va="bottom", ha="left")
    ax_leg_host.text(0.62, 0.50, "0",  transform=ax_leg_host.transAxes, fontsize=9, va="center", ha="left")
    ax_leg_host.text(0.62, 0.95, "+1", transform=ax_leg_host.transAxes, fontsize=9, va="top", ha="left")
    ax_leg_host.text(0.50, 1.02, "Diff legend", transform=ax_leg_host.transAxes,
                     fontsize=title_fs, va="bottom", ha="center")

    # --- Static original ---
    ax_orig.set_title("Original", fontsize=title_fs)
    im_orig = ax_orig.imshow(orig_u8)

    # --- Downsampled canvases (reuse the same canvas arrays) ---
    canvas_a = np.full_like(orig_u8, 255)
    canvas_b = np.full_like(orig_u8, 255)
    _paste_on_white_canvas(downs_a[0], canvas_a)
    _paste_on_white_canvas(downs_b[0], canvas_b)

    t_down_a = ax_down_a.set_title(f"{label_a} (z={zoom_values[0]:.3f})", fontsize=title_fs)
    t_down_b = ax_down_b.set_title(f"{label_b} (z={zoom_values[0]:.3f})", fontsize=title_fs)
    im_down_a = ax_down_a.imshow(canvas_a)
    im_down_b = ax_down_b.imshow(canvas_b)

    # --- Recovered ---
    t_rec_a = ax_rec_a.set_title(f"Recovered, {label_a}", fontsize=title_fs)
    t_rec_b = ax_rec_b.set_title(f"Recovered, {label_b}", fontsize=title_fs)
    im_rec_a = ax_rec_a.imshow(recs_a[0])
    im_rec_b = ax_rec_b.imshow(recs_b[0])

    # --- Error ---
    ax_err_a.set_title("Signed error", fontsize=title_fs)
    ax_err_b.set_title("Signed error", fontsize=title_fs)
    im_err_a = ax_err_a.imshow(diff_norm_u8(recs_a[0]), cmap="gray", vmin=0, vmax=255)
    im_err_b = ax_err_b.imshow(diff_norm_u8(recs_b[0]), cmap="gray", vmin=0, vmax=255)

    def animate(i: int):
        z = float(zoom_values[i])

        _paste_on_white_canvas(downs_a[i], canvas_a)
        _paste_on_white_canvas(downs_b[i], canvas_b)
        im_down_a.set_data(canvas_a)
        im_down_b.set_data(canvas_b)

        im_rec_a.set_data(recs_a[i])
        im_rec_b.set_data(recs_b[i])

        im_err_a.set_data(diff_norm_u8(recs_a[i]))
        im_err_b.set_data(diff_norm_u8(recs_b[i]))

        t_down_a.set_text(f"{label_a} (z={z:.3f})")
        t_down_b.set_text(f"{label_b} (z={z:.3f})")

        return (
            im_down_a, im_down_b,
            im_rec_a, im_rec_b,
            im_err_a, im_err_b,
            t_down_a, t_down_b,
            t_rec_a, t_rec_b,
        )

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(zoom_values),
        interval=interval_ms,
        blit=True,
    )
    return ani


# %%
# Run (per image)
# ---------------

animations: List[animation.FuncAnimation] = []

if not _HAS_TORCH:
    print("[info] PyTorch not available → skipping PyTorch comparison animation.")
else:
    for name, url in KODAK_IMAGES:
        orig01 = _load_kodak_rgb01(url)

        ani = make_two_method_animation(
            img_name=name,
            orig01=orig01,
            zoom_values=ZOOM_VALUES,
            rt_a=_roundtrip_splineops_aa,
            rt_b=_roundtrip_torch_bicubic_aa,
            label_a=SPLINEOPS_LABEL,
            label_b=COMPETITOR_LABEL,
            interval_ms=INTERVAL_MS,
            title_fs=TITLE_FS,
        )
        animations.append(ani)
