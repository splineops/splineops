# sphinx_gallery_start_ignore
# splineops/examples/02_resize/08_benchmarking_animation.py
# sphinx_gallery_end_ignore

"""
Benchmarking Animation
======================

This example builds a short animation for a set of images.

For a sequence of zoom factors, we do a round-trip resize: original → downsampled → recovered.

and compare two methods:

- PyTorch bicubic (if available; otherwise SciPy cubic),
- SplineOps cubic-antialiasing.

Each frame shows the original image (fixed), the downsampled image (pasted on a white canvas),
the recovered image and a normalized signed error map (recovered − original).

The animations are displayed in the docs (no files are exported here).
"""

# %%
# Imports
# -------

from __future__ import annotations

import os
from typing import Callable, Tuple
from urllib.request import urlopen

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from PIL import Image

from splineops.resize import resize as sp_resize

# Optional: runtime context (nice for docs / reproducibility)
try:
    from splineops.utils.specs import print_runtime_context
    _HAS_SPECS = True
except Exception:
    print_runtime_context = None  # type: ignore[assignment]
    _HAS_SPECS = False

# Optional SciPy (fallback competitor)
try:
    from scipy.ndimage import zoom as ndi_zoom  # type: ignore
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    ndi_zoom = None  # type: ignore[assignment]

# Optional PyTorch (preferred competitor)
try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
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
TITLE_FS = 12

SPLINEOPS_LABEL = "SplineOps Antialiasing cubic"

# --- Highlight colors ---
SPLINEOPS_AA_COLOR = "#BE185D"   # Antialiasing
SPLINEOPS_STD_COLOR = "#C2410C"  # (unused here, but kept consistent)

# Prefer torch if available; otherwise SciPy
if _HAS_TORCH:
    COMP_LABEL = "PyTorch bicubic"
else:
    COMP_LABEL = "SciPy cubic"

# Speed knob: scale images down for animation (keeps docs builds reasonable).
# Set SPLINEOPS_ANIM_SCALE=1.0 if you want full resolution.
ANIM_SCALE = float(os.environ.get("SPLINEOPS_ANIM_SCALE", "0.5"))
ANIM_SCALE = float(np.clip(ANIM_SCALE, 0.1, 1.0))

# Zoom factors (full sweep), but start the animation at z=0.30 and go downwards
Z_START = 0.30

# Original distribution (includes values up to 1.0)
zoom_low   = np.geomspace(0.01, 0.15, 6,  endpoint=False)   # < 0.15
zoom_focus = np.geomspace(0.15, 0.50, 22, endpoint=False)   # [0.15, 0.50)
zoom_mid   = np.geomspace(0.50, 0.80, 6,  endpoint=True)    # [0.50, 0.80]
zoom_top   = np.array([0.85, 0.90, 0.95, 1.0])

ZOOM_VALUES_FULL = np.sort(
    np.unique(np.concatenate([zoom_low, zoom_focus, zoom_mid, zoom_top, [Z_START]]))
)[::-1]  # 1.0 -> ... -> small

# Start at Z_START, go down to ~0.01
start_idx = int(np.where(ZOOM_VALUES_FULL <= Z_START)[0][0])
tail = ZOOM_VALUES_FULL[start_idx:]          # Z_START -> ... -> small

# Then go from 1.0 down to Z_START to close the loop
head = ZOOM_VALUES_FULL[: start_idx + 1]     # 1.0 -> ... -> Z_START

# Final sequence: 0.30 -> ... -> 0.01 -> 1.0 -> ... -> 0.30
ZOOM_VALUES = np.concatenate([tail, head]).astype(float)

KODAK_BASE = "https://r0k.us/graphics/kodak/kodak"
KODAK_IMAGES = {
    "kodim05": f"{KODAK_BASE}/kodim05.png",
    "kodim07": f"{KODAK_BASE}/kodim07.png",
    "kodim14": f"{KODAK_BASE}/kodim14.png",
    "kodim15": f"{KODAK_BASE}/kodim15.png",
    "kodim19": f"{KODAK_BASE}/kodim19.png",
    "kodim22": f"{KODAK_BASE}/kodim22.png",
    "kodim23": f"{KODAK_BASE}/kodim23.png",
}

# Optional local limiter (defaults to "all")
# e.g.  SPLINEOPS_MAX_IMAGES=2  to run only first 2 sections locally
MAX_IMAGES = int(os.environ.get("SPLINEOPS_MAX_IMAGES", "0"))


# %%
# Small Helpers
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
    canvas_u8[...] = 255
    h1, w1 = down_u8.shape[:2]
    canvas_u8[:h1, :w1, :] = down_u8


def _resize_rgb_splineops(img01: np.ndarray, z: float, *, method: str) -> np.ndarray:
    """Channel-wise splineops resize for RGB."""
    zoom_hw = (float(z), float(z))
    chs = [sp_resize(img01[..., c], zoom_factors=zoom_hw, method=method) for c in range(3)]
    out = np.stack(chs, axis=-1)
    return np.clip(out, 0.0, 1.0).astype(DTYPE, copy=False)


def _match_shape_center(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Center-crop or pad (with edge values) to match (H,W) exactly."""
    Ht, Wt = shape
    H, W = a.shape[:2]
    if (H, W) == (Ht, Wt):
        return a
    # crop
    r0 = max(0, (H - Ht) // 2)
    c0 = max(0, (W - Wt) // 2)
    a2 = a[r0:r0 + min(Ht, H), c0:c0 + min(Wt, W), ...]
    # pad if needed
    H2, W2 = a2.shape[:2]
    if (H2, W2) == (Ht, Wt):
        return a2
    pad_top = max(0, (Ht - H2) // 2)
    pad_bot = max(0, Ht - H2 - pad_top)
    pad_lft = max(0, (Wt - W2) // 2)
    pad_rgt = max(0, Wt - W2 - pad_lft)
    return np.pad(a2, ((pad_top, pad_bot), (pad_lft, pad_rgt), (0, 0)), mode="edge")


# %%
# Backends (round-trip)
# ---------------------

def rt_splineops_aa(orig01: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (down_u8, rec_u8) for splineops cubic-antialiasing."""
    H0, W0 = orig01.shape[:2]
    down01 = _resize_rgb_splineops(orig01, z, method="cubic-antialiasing")
    down_u8 = _to_u8(down01)

    rec_ch = [sp_resize(down01[..., c], output_size=(H0, W0), method="cubic-antialiasing") for c in range(3)]
    rec01 = np.stack(rec_ch, axis=-1)
    rec_u8 = _to_u8(rec01)
    return down_u8, rec_u8


def rt_torch_bicubic(orig01: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (down_u8, rec_u8) for torch bicubic with antialias=False."""
    assert _HAS_TORCH and F is not None

    H0, W0 = orig01.shape[:2]
    H1 = max(1, int(round(H0 * z)))
    W1 = max(1, int(round(W0 * z)))

    x = torch.from_numpy(orig01.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)

    # antialias=False explicitly; if not supported by the torch version, fallback without it
    try:
        y  = F.interpolate(x,  size=(H1, W1), mode="bicubic", align_corners=False, antialias=False)
        y2 = F.interpolate(y,  size=(H0, W0), mode="bicubic", align_corners=False, antialias=False)
    except TypeError:
        y  = F.interpolate(x,  size=(H1, W1), mode="bicubic", align_corners=False)
        y2 = F.interpolate(y,  size=(H0, W0), mode="bicubic", align_corners=False)

    down01 = y[0].permute(1, 2, 0).detach().cpu().numpy()
    rec01  = y2[0].permute(1, 2, 0).detach().cpu().numpy()

    return _to_u8(down01), _to_u8(rec01)


def rt_scipy_cubic(orig01: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (down_u8, rec_u8) for SciPy cubic (reflect)."""
    assert _HAS_SCIPY and ndi_zoom is not None

    H0, W0 = orig01.shape[:2]
    H1 = max(1, int(round(H0 * z)))
    W1 = max(1, int(round(W0 * z)))

    # SciPy works channel-wise
    downs = []
    for c in range(3):
        ch = ndi_zoom(orig01[..., c], zoom=(z, z), order=3, prefilter=True, mode="reflect", grid_mode=False)
        downs.append(ch)
    down01 = np.stack(downs, axis=-1)
    down01 = _match_shape_center(down01, (H1, W1))

    # back to original size
    Hz, Wz = down01.shape[:2]
    zoom_bwd = (H0 / float(Hz), W0 / float(Wz))

    recs = []
    for c in range(3):
        ch = ndi_zoom(down01[..., c], zoom=zoom_bwd, order=3, prefilter=True, mode="reflect", grid_mode=False)
        recs.append(ch)
    rec01 = np.stack(recs, axis=-1)
    rec01 = _match_shape_center(rec01, (H0, W0))

    return _to_u8(down01), _to_u8(rec01)


def get_competitor_rt() -> tuple[str, Callable[[np.ndarray, float], tuple[np.ndarray, np.ndarray]]]:
    if _HAS_TORCH:
        return COMP_LABEL, rt_torch_bicubic
    if _HAS_SCIPY:
        return COMP_LABEL, rt_scipy_cubic
    raise RuntimeError("Neither PyTorch nor SciPy is available for the competitor backend.")


# %%
# Animation Builder (Single Image)
# --------------------------------

def make_benchmark_animation(
    img_name: str,
    url: str,
    *,
    zoom_values: np.ndarray = ZOOM_VALUES,
    interval_ms: int = INTERVAL_MS,
    title_fs: int = TITLE_FS,
) -> animation.FuncAnimation:
    """
    Build the per-image animation.
    Computes a GLOBAL max(|diff|) across all frames + both methods (stable contrast).
    """
    comp_label, rt_comp = get_competitor_rt()

    # Load + (optional) downscale for speed
    orig01 = _load_kodak_rgb01(url)
    if ANIM_SCALE < 0.999:
        orig01 = _resize_rgb_splineops(orig01, ANIM_SCALE, method="cubic")

    orig_u8 = _to_u8(orig01)
    H0, W0 = orig_u8.shape[:2]
    orig_gray01 = _u8_to_gray01(orig_u8)

    # --- Prepass: compute global max_abs across frames for both methods ---
    max_abs = 0.0
    for z in zoom_values:
        _, rec_a = rt_splineops_aa(orig01, float(z))
        _, rec_b = rt_comp(orig01, float(z))
        max_abs = max(max_abs, float(np.max(np.abs(_u8_to_gray01(rec_a) - orig_gray01))))
        max_abs = max(max_abs, float(np.max(np.abs(_u8_to_gray01(rec_b) - orig_gray01))))
    max_abs = max(max_abs, 1e-12)

    # Precompute energy for SNR (grayscale)
    orig_energy = float(np.sum(orig_gray01.astype(np.float64) ** 2))

    def _snr_db(rec_gray01: np.ndarray) -> float:
        den = float(np.sum((orig_gray01.astype(np.float64) - rec_gray01.astype(np.float64)) ** 2))
        if den == 0.0:
            return float("inf")
        if orig_energy == 0.0:
            return -float("inf")
        return 10.0 * float(np.log10(orig_energy / den))

    def _fmt_snr(v: float) -> str:
        if np.isposinf(v):
            return "∞"
        if np.isneginf(v):
            return "-∞"
        return f"{v:.2f} dB"

    # Shared (both methods) error normalization using the SAME max_abs
    def diff_norm_u8_from_gray(rec_gray01: np.ndarray) -> np.ndarray:
        d = rec_gray01 - orig_gray01
        n = 0.5 + 0.5 * (d / max_abs)
        return (np.clip(n, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    # --- Layout (same as 02_resize_module_2d.py style) ---
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

    # Row label for the recovered row (keeps per-panel titles shorter)
    ax_blank.text(
        0.5, 0.5, "Recovered image",
        transform=ax_blank.transAxes,
        ha="center", va="center",
        fontsize=title_fs,
    )

    # Legend bar
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
    ax_leg_host.text(0.50, 1.02, "Signed error", transform=ax_leg_host.transAxes,
                     fontsize=title_fs, va="bottom", ha="center")

    # Static original
    ax_orig.set_title("Original image", fontsize=title_fs)
    ax_orig.imshow(orig_u8)

    # Create reusable canvases
    canvas_a = np.full_like(orig_u8, 255)
    canvas_b = np.full_like(orig_u8, 255)

    # First frame (i=0)
    z0 = float(zoom_values[0])
    down_so0,   rec_so0   = rt_splineops_aa(orig01, z0)
    down_cmp0,  rec_cmp0  = rt_comp(orig01, z0)

    # Column order: competitor first (left), splineops second (right)
    _paste_on_white_canvas(down_cmp0, canvas_a)  # left method column
    _paste_on_white_canvas(down_so0,  canvas_b)  # right method column

    t_down_a = ax_down_a.set_title(f"{comp_label} (z={z0:.3f})", fontsize=title_fs)
    t_down_b = ax_down_b.set_title(
        f"{SPLINEOPS_LABEL} (z={z0:.3f})",
        fontsize=title_fs,
        color=SPLINEOPS_AA_COLOR,
        fontweight="bold",
    )

    ax_rec_a.set_title(f"Recovered, {comp_label}", fontsize=title_fs)
    ax_rec_b.set_title(f"Recovered, {SPLINEOPS_LABEL}", fontsize=title_fs)

    # Artists
    im_down_a = ax_down_a.imshow(canvas_a)
    im_down_b = ax_down_b.imshow(canvas_b)

    im_rec_a = ax_rec_a.imshow(rec_cmp0)
    im_rec_b = ax_rec_b.imshow(rec_so0)

    # Compute grayscale once (reuse for SNR + error)
    rec_cmp_g0 = _u8_to_gray01(rec_cmp0)
    rec_so_g0  = _u8_to_gray01(rec_so0)

    snr_cmp0 = _snr_db(rec_cmp_g0)
    snr_so0  = _snr_db(rec_so_g0)

    t_rec_a = ax_rec_a.set_title(
        f"{comp_label} (SNR={_fmt_snr(snr_cmp0)})",
        fontsize=title_fs,
    )
    t_rec_b = ax_rec_b.set_title(
        f"{SPLINEOPS_LABEL} (SNR={_fmt_snr(snr_so0)})",
        fontsize=title_fs,
        color=SPLINEOPS_AA_COLOR,
        fontweight="bold",
    )

    # Error maps: same range for BOTH methods (0..255) and shared normalization
    im_err_a = ax_err_a.imshow(diff_norm_u8_from_gray(rec_cmp_g0), cmap="gray", vmin=0, vmax=255)
    im_err_b = ax_err_b.imshow(diff_norm_u8_from_gray(rec_so_g0),  cmap="gray", vmin=0, vmax=255)

    def animate(i: int):
        z = float(zoom_values[i])

        down_so,  rec_so  = rt_splineops_aa(orig01, z)
        down_cmp, rec_cmp = rt_comp(orig01, z)

        _paste_on_white_canvas(down_cmp, canvas_a)
        _paste_on_white_canvas(down_so,  canvas_b)

        im_down_a.set_data(canvas_a)
        im_down_b.set_data(canvas_b)

        im_rec_a.set_data(rec_cmp)
        im_rec_b.set_data(rec_so)

        # Grayscale once per method (reuse)
        rec_cmp_g = _u8_to_gray01(rec_cmp)
        rec_so_g  = _u8_to_gray01(rec_so)

        im_err_a.set_data(diff_norm_u8_from_gray(rec_cmp_g))
        im_err_b.set_data(diff_norm_u8_from_gray(rec_so_g))

        # Titles
        t_down_a.set_text(f"{comp_label} (z={z:.3f})")
        t_down_b.set_text(f"{SPLINEOPS_LABEL} (z={z:.3f})")

        t_rec_a.set_text(f"{comp_label} (SNR={_fmt_snr(_snr_db(rec_cmp_g))})")
        t_rec_b.set_text(f"{SPLINEOPS_LABEL} (SNR={_fmt_snr(_snr_db(rec_so_g))})")

        return (
            im_down_a, im_down_b,
            im_rec_a, im_rec_b,
            im_err_a, im_err_b,
            t_down_a, t_down_b,
            t_rec_a, t_rec_b,
        )

    # Avoid Matplotlib caching all frames in memory (important when many animations)
    try:
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(zoom_values),
            interval=interval_ms,
            blit=True,
            cache_frame_data=False,
        )
    except TypeError:
        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(zoom_values),
            interval=interval_ms,
            blit=True,
        )

    return ani


# %%
# Image: kodim05
# --------------
ani_kodim05 = make_benchmark_animation("kodim05", KODAK_IMAGES["kodim05"])

# %%
# Image: kodim07
# --------------
ani_kodim07 = make_benchmark_animation("kodim07", KODAK_IMAGES["kodim07"])

# %%
# Export: kodim07 Animation
# -------------------------
#
# Writes into: <generated static dir>/_static/animations/
# No-op when run normally by users.

from splineops.utils.sphinx import export_animation_mp4_and_html

export_animation_mp4_and_html(
    ani_kodim07,
    stem="benchmark_animation_kodim07_pytorch_vs_splineops",
    interval_ms=INTERVAL_MS,
    dpi=80,
    force=True,
)

# %%
# Image: kodim14
# --------------
ani_kodim14 = make_benchmark_animation("kodim14", KODAK_IMAGES["kodim14"])

# %%
# Image: kodim15
# --------------
ani_kodim15 = make_benchmark_animation("kodim15", KODAK_IMAGES["kodim15"])

# %%
# Image: kodim19
# --------------
ani_kodim19 = make_benchmark_animation("kodim19", KODAK_IMAGES["kodim19"])

# %%
# Image: kodim22
# --------------
ani_kodim22 = make_benchmark_animation("kodim22", KODAK_IMAGES["kodim22"])

# %%
# Image: kodim23
# --------------
ani_kodim23 = make_benchmark_animation("kodim23", KODAK_IMAGES["kodim23"])

# %%
# Runtime Context
# ---------------
#
# Print a short summary of the runtime environment and the dtype used
# for the animation computations.

if _HAS_SPECS and print_runtime_context is not None:
    print_runtime_context(include_threadpools=True)
    print()  # blank line

print(f"Animation storage dtype: {np.dtype(DTYPE).name}")
print(f"ANIM_SCALE: {ANIM_SCALE}")
print(f"Competitor backend: {COMP_LABEL}")
print(f"Number of frames per animation: {len(ZOOM_VALUES)}")
