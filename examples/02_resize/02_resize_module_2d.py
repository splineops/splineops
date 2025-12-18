# sphinx_gallery_start_ignore
# splineops/examples/02_resize/02_resize_module_2d.py
# sphinx_gallery_end_ignore

"""
Resize Module 2D
================

Shrink and re-expand a 2-D RGB image with splineops, then discuss aliasing.

We start with a minimal example that calls :func:`splineops.resize.resize`
directly on a single channel of the image, then move on to an RGB example
with ROIs:

1. Simple grayscale resize with ``resize`` on one channel.
2. Pick a zoom factor and a single ROI.
3. Compare the first-pass shrink for standard cubic interpolation and
   ``cubic-antialiasing``.

Aliasing appears when we shrink below the Nyquist limit without proper
low-pass filtering: fine details fold back into lower frequencies and
show up as Moiré or ripple patterns. The antialiasing preset adds a
matched low-pass step to suppress these artefacts.
"""

# %%
# Imports and Helpers
# -------------------

# sphinx_gallery_thumbnail_number = 2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from urllib.request import urlopen
from PIL import Image

from scipy.ndimage import zoom as ndi_zoom          # kept for reference, not used in 2D plots
from splineops.resize import resize                 # core N-D spline resizer

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
})

# Use float32 for storage / IO (resize still computes internally in float64).
DTYPE = np.float32


def resize_rgb(
    img: np.ndarray,
    zoom: float,
    *,
    method: str = "cubic",
) -> np.ndarray:
    """
    Resize an H×W×3 RGB image with splineops.resize.resize (channel-wise).

    Parameters
    ----------
    img : ndarray, shape (H, W, 3), values in [0, 1]
    zoom : float
        Isotropic zoom factor (same for H and W).
    method : str
        One of the splineops presets, e.g. "linear", "cubic",
        "cubic-antialiasing", ...

    Returns
    -------
    out : ndarray, shape (H', W', 3)
        Same float dtype as ``img`` (float32 in this example), values in [0, 1].
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("resize_rgb expects an H×W×3 RGB array")

    zoom_hw = (float(zoom), float(zoom))  # (H, W) factors

    channels = []
    for c in range(img.shape[2]):
        ch = resize(
            img[..., c],
            zoom_factors=zoom_hw,
            method=method,
        )
        channels.append(ch)

    out = np.stack(channels, axis=-1)
    return np.clip(out, 0.0, 1.0)


def _roi_rect_from_frac_color(shape, roi_size_px, center_frac):
    """
    Compute a square ROI inside a color image, centered at fractional coordinates.

    Parameters
    ----------
    shape : tuple
        (H, W, 3) shape of the color image.
    roi_size_px : int
        Target side length (clipped to fit inside the image).
    center_frac : tuple of float
        (row_frac, col_frac) in [0, 1] × [0, 1].

    Returns
    -------
    (row_top, col_left, height, width)
    """
    H, W, _ = shape
    row_frac, col_frac = center_frac

    size = int(min(roi_size_px, H, W))
    if size < 1:
        size = min(H, W)

    center_r = int(round(row_frac * H))
    center_c = int(round(col_frac * W))

    row_top = int(np.clip(center_r - size // 2, 0, H - size))
    col_left = int(np.clip(center_c - size // 2, 0, W - size))

    return row_top, col_left, size, size


def _nearest_big_color(roi: np.ndarray, target_h: int = 256) -> np.ndarray:
    """
    Enlarge a small color ROI (H×W×3) with nearest-neighbour so that its
    height is ~target_h pixels.
    """
    h, w, _ = roi.shape
    mag = max(1, int(round(target_h / max(h, 1))))
    return np.repeat(np.repeat(roi, mag, axis=0), mag, axis=1)


def show_intro_color(
    original_uint8: np.ndarray,
    shrunk_uint8: np.ndarray,
    roi_rect,
    zoom: float,
    label: str,
    degree_label: str,
) -> None:
    """
    2×2 figure with the same wording style as the benchmarking intro:

    Row 1:
      - Original image with ROI (H×W px)
      - Original ROI (h×w px, NN magnified)

    Row 2:
      - First-pass resized image on a white canvas, with mapped ROI box
      - First-pass ROI (h'×w' px, NN magnified)
    """
    H, W, _ = original_uint8.shape
    row0, col0, roi_h, roi_w = roi_rect

    # Original ROI and its NN magnification
    roi_orig = original_uint8[row0:row0 + roi_h, col0:col0 + roi_w, :]
    roi_orig_big = _nearest_big_color(roi_orig, target_h=256)

    # Shrunk image geometry
    Hs, Ws, _ = shrunk_uint8.shape
    center_r = row0 + roi_h / 2.0
    center_c = col0 + roi_w / 2.0

    roi_h_res = max(1, int(round(roi_h * zoom)))
    roi_w_res = max(1, int(round(roi_w * zoom)))

    if roi_h_res > Hs or roi_w_res > Ws:
        roi_shrunk = shrunk_uint8
        row_top_res = 0
        col_left_res = 0
        roi_h_res = Hs
        roi_w_res = Ws
    else:
        center_r_res = int(round(center_r * zoom))
        center_c_res = int(round(center_c * zoom))
        row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, Hs - roi_h_res))
        col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, Ws - roi_w_res))
        roi_shrunk = shrunk_uint8[
            row_top_res:row_top_res + roi_h_res,
            col_left_res:col_left_res + roi_w_res,
            :
        ]

    roi_shrunk_big = _nearest_big_color(roi_shrunk, target_h=256)

    # Place shrunk image on a white canvas of the same size as original
    canvas = np.full_like(original_uint8, 255)
    h_copy = min(H, Hs)
    w_copy = min(W, Ws)
    canvas[:h_copy, :w_copy, :] = shrunk_uint8[:h_copy, :w_copy, :]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Row 1, left: original with ROI box
    ax = axes[0, 0]
    ax.imshow(original_uint8)
    rect = patches.Rectangle(
        (col0, row0),
        roi_w,
        roi_h,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.set_title(
        f"Original image with ROI ({H}×{W} px)",
        fontsize=12,
    )
    ax.axis("off")

    # Row 1, right: magnified original ROI
    ax = axes[0, 1]
    ax.imshow(roi_orig_big)
    ax.set_title(
        f"Original ROI ({roi_h}×{roi_w} px, NN magnified)",
        fontsize=12,
    )
    ax.axis("off")

    # Row 2, left: first-pass resized image on canvas with mapped ROI box
    ax = axes[1, 0]
    ax.imshow(canvas)
    if row_top_res < h_copy and col_left_res < w_copy:
        box_h = min(roi_h_res, h_copy - row_top_res)
        box_w = min(roi_w_res, w_copy - col_left_res)
        rect2 = patches.Rectangle(
            (col_left_res, row_top_res),
            box_w,
            box_h,
            linewidth=2,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect2)
    ax.set_title(
        f"{label} ({degree_label}, zoom ×{zoom:g}, {Hs}×{Ws} px)",
        fontsize=12,
    )
    ax.axis("off")

    # Row 2, right: magnified resized ROI
    ax = axes[1, 1]
    ax.imshow(roi_shrunk_big)
    ax.set_title(
        f"{label} ROI ({roi_h_res}×{roi_w_res} px, NN magnified)",
        fontsize=12,
    )
    ax.axis("off")

    fig.tight_layout()
    plt.show()


# %%
# Load and Normalize an Image
# ---------------------------
#
# We now move to a real 2D example using a Kodak color image. The data is
# stored as float32 in [0, 1] for splineops.

url = "https://r0k.us/graphics/kodak/kodak/kodim19.png"
with urlopen(url, timeout=10) as resp:
    img = Image.open(resp)
data = np.asarray(img, dtype=DTYPE) / DTYPE(255.0)          # H × W × 3, range [0, 1]
data_uint8 = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)

H0, W0, _ = data_uint8.shape
print(f"Loaded kodim19: shape={H0}×{W0} px")

# %%
# Simple 2D Resize Call
# ---------------------
#
# Before using helpers and ROIs, we start with a minimal example that calls
# :func:`splineops.resize.resize` directly on a single (grayscale) channel.
# This highlights the basic API: you provide an array, per-axis zoom factors,
# and a method string.

# Take a single channel (e.g., red) to keep things simple
gray = data[..., 0]  # shape H×W, values in [0, 1]

simple_zoom = 0.3  # shrink by a factor of 0.3 in each direction

gray_cubic = resize(
    gray,
    zoom_factors=(simple_zoom, simple_zoom),
    method="cubic",
)

gray_aa = resize(
    gray,
    zoom_factors=(simple_zoom, simple_zoom),
    method="cubic-antialiasing",
)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
axes[0].set_title("Original (single channel)")
axes[1].imshow(gray_cubic, cmap="gray", vmin=0.0, vmax=1.0)
axes[1].set_title("Resized Cubic")
axes[2].imshow(gray_aa, cmap="gray", vmin=0.0, vmax=1.0)
axes[2].set_title("Resized Cubic Antialiasing")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
plt.show()

# %%
# ROI and RGB Shrink With Antialiasing
# ------------------------------------
#
# We now follow the same spirit as the benchmarking intro:
#
# 1. Pick a zoom factor.
# 2. Pick a single ROI.
# 3. Compare the first-pass shrink for standard cubic interpolation
#    and antialiasing on the full RGB image.

# Use the same ROI position as in the benchmarking example for kodim19
ROI_SIZE_PX = 256
ROI_CENTER_FRAC = (0.65, 0.35)
roi_rect = _roi_rect_from_frac_color(data_uint8.shape, ROI_SIZE_PX, ROI_CENTER_FRAC)

# Shrink factor (same spirit as in other 2D examples)
shrink_factor = 0.3

# 3) Shrink with splineops (channel-wise): standard cubic
shrunken_cubic_f = resize_rgb(
    data,
    shrink_factor,
    method="cubic",         # plain cubic interpolation (no explicit anti-aliasing)
)
shrunken_cubic = (np.clip(shrunken_cubic_f, 0.0, 1.0) * 255).astype(np.uint8)

# 4) Shrink with splineops: cubic-antialiasing
shrunken_aa_f = resize_rgb(
    data,
    shrink_factor,
    method="cubic-antialiasing",  # antialiasing shrink, degree 3
)
shrunken_aa = (np.clip(shrunken_aa_f, 0.0, 1.0) * 255).astype(np.uint8)

# %%
# Standard Cubic Shrink
# ---------------------
#
# Standard cubic interpolation gives a smooth-looking small image, but it
# does not apply an explicit low-pass before decimation. High-frequency
# content from the original folds back (aliases) into lower frequencies,
# which can be spotted as spurious ripples or Moiré patterns in the ROI.

show_intro_color(
    original_uint8=data_uint8,
    shrunk_uint8=shrunken_cubic,
    roi_rect=roi_rect,
    zoom=shrink_factor,
    label="Standard cubic",
    degree_label="Cubic",
)

# %%
# Cubic-Antialiasing Shrink
# -------------------------
#
# The "cubic-antialiasing" preset inserts a projection-based low-pass
# filter before shrinking. The zoomed ROI shows that most of the Moiré
# pattern is removed, while larger-scale edges and contrast are preserved.

show_intro_color(
    original_uint8=data_uint8,
    shrunk_uint8=shrunken_aa,
    roi_rect=roi_rect,
    zoom=shrink_factor,
    label="Antialiasing",
    degree_label="Cubic",
)

# %%
# Round-trip animation over Zoom Factors
# --------------------------------------
#
# We animate a downsample → upsample (back to original size) round-trip for
# several zoom factors z < 1.
#
# Top:    original RGB image (fixed)
# Middle: downsampled image pasted on a white canvas of the original size
# Bottom: recovered image (round-trip), showing degradation
#
# Tip: set METHOD="cubic" to see the effect without the antialiasing preset.

from matplotlib import animation

METHOD = "cubic"   # try "cubic" for standard interpolation
INTERVAL_MS = 900  # keep in sync with MP4 export

# Dense where artefacts change fast (0.01–0.2), then sparser up to 1.0
zoom_dense = np.geomspace(0.01, 0.2, 25)          # was 10
zoom_mid   = np.geomspace(0.22, 0.8, 10)          # optional mid range
zoom_top   = np.array([0.85, 0.90, 0.95, 1.0])    # near-1 “sanity” points

zoom_values = np.unique(np.concatenate([zoom_dense, zoom_mid, zoom_top]))
zoom_values = np.sort(zoom_values)[::-1]  # 1.0 -> ... -> small

orig = np.clip(data, 0.0, 1.0)  # H×W×3 float image in [0,1]
H0, W0, _ = orig.shape

# Precompute frames for a smooth/fast animation
canvases: list[np.ndarray] = []
recs: list[np.ndarray] = []

for z in zoom_values:
    # 1) Downsample (first pass)
    down = resize_rgb(orig, z, method=METHOD)              # H1×W1×3
    down = np.clip(down, 0.0, 1.0)

    # 2) Paste on white canvas of the original size
    canvas = np.ones_like(orig)
    h1, w1, _ = down.shape
    canvas[:h1, :w1, :] = down
    canvases.append(canvas)

    # 3) Recover back to original size (second pass)
    rec_channels = [
        resize(down[..., c], output_size=(H0, W0), method=METHOD)
        for c in range(3)
    ]
    rec = np.stack(rec_channels, axis=-1)
    recs.append(np.clip(rec, 0.0, 1.0))

# Figure layout: 3 rows, 1 column
fig, axes = plt.subplots(3, 1, figsize=(7, 12), constrained_layout=True)
for ax in axes:
    ax.axis("off")

axes[0].set_title("Original")
t_down = axes[1].set_title(f"Downsampled (z={zoom_values[0]:.2f}) on white canvas")
t_rec  = axes[2].set_title(f"Recovered (round-trip) — method={METHOD}, z={zoom_values[0]:.2f}")

im0 = axes[0].imshow(orig)
im1 = axes[1].imshow(canvases[0])
im2 = axes[2].imshow(recs[0])

def animate_frame(i: int):
    im1.set_data(canvases[i])
    im2.set_data(recs[i])
    t_down.set_text(f"Downsampled (z={zoom_values[i]:.2f}) on white canvas")
    t_rec.set_text(f"Recovered (round-trip) — method={METHOD}, z={zoom_values[i]:.2f}")
    return im1, im2, t_down, t_rec

ani = animation.FuncAnimation(
    fig,
    animate_frame,
    frames=len(zoom_values),
    interval=900,
    blit=True,
)

# %%
# Side-by-side animation: cubic vs cubic-antialiasing
# ---------------------------------------------------
#
# Left column:  Standard cubic
# Right column: Cubic-antialiasing
#
# Row 1: Original (fixed)
# Row 2: Downsampled (on white canvas)
# Row 3: Recovered (round-trip)

from matplotlib import animation

METHOD_STD = "cubic"
METHOD_AA  = "cubic-antialiasing"

# Use the same zoom_values from the previous cell
# (If you prefer, you can redefine zoom_values here.)
zoom_values_cmp = np.asarray(zoom_values, dtype=float)

orig_f = np.clip(data, 0.0, 1.0)
H0, W0, _ = orig_f.shape
orig_u8 = (orig_f * 255.0 + 0.5).astype(np.uint8)

# Precompute frames (store uint8 to keep memory low)
canv_std: list[np.ndarray] = []
recs_std: list[np.ndarray] = []
canv_aa:  list[np.ndarray] = []
recs_aa:  list[np.ndarray] = []

def _roundtrip_frames(method: str):
    canv: list[np.ndarray] = []
    recs: list[np.ndarray] = []
    for z in zoom_values_cmp:
        # Downsample
        down_f = resize_rgb(orig_f, z, method=method)
        down_f = np.clip(down_f, 0.0, 1.0)
        down_u8 = (down_f * 255.0 + 0.5).astype(np.uint8)

        # Canvas (white, original size)
        canvas = np.full_like(orig_u8, 255)
        h1, w1, _ = down_u8.shape
        canvas[:h1, :w1, :] = down_u8
        canv.append(canvas)

        # Recover to original size (round-trip)
        rec_channels = [
            resize(down_f[..., c], output_size=(H0, W0), method=method)
            for c in range(3)
        ]
        rec_f = np.clip(np.stack(rec_channels, axis=-1), 0.0, 1.0)
        rec_u8 = (rec_f * 255.0 + 0.5).astype(np.uint8)
        recs.append(rec_u8)

    return canv, recs

canv_std, recs_std = _roundtrip_frames(METHOD_STD)
canv_aa,  recs_aa  = _roundtrip_frames(METHOD_AA)

# --- Signed normalized error maps (benchmark style) --------------------------
# We compare rec - orig on a grayscale luminance, normalized into [0, 1]:
#   0.5 = 0 error, <0.5 negative, >0.5 positive
# We use a GLOBAL max(|diff|) across all frames + both methods (stable contrast).

def _u8_to_gray01(u8_rgb: np.ndarray) -> np.ndarray:
    u = u8_rgb.astype(np.float32) / 255.0
    return (0.2989 * u[..., 0] + 0.5870 * u[..., 1] + 0.1140 * u[..., 2]).astype(np.float32)

orig_gray01 = _u8_to_gray01(orig_u8)

def _diff01(rec_u8: np.ndarray) -> np.ndarray:
    return _u8_to_gray01(rec_u8) - orig_gray01  # signed

max_abs = 0.0
for r in (recs_std + recs_aa):
    max_abs = max(max_abs, float(np.max(np.abs(_diff01(r)))))
max_abs = max(max_abs, 1e-12)

def _diff_norm(rec_u8: np.ndarray) -> np.ndarray:
    d = _diff01(rec_u8)
    n = 0.5 + 0.5 * (d / max_abs)
    return np.clip(n, 0.0, 1.0).astype(np.float32)

diffs_std = [_diff_norm(r) for r in recs_std]
diffs_aa  = [_diff_norm(r) for r in recs_aa]

# --- Layout: 3 columns (Original | Std | AA), 3 rows (Down | Rec | Error) ----
fig = plt.figure(figsize=(13, 9), constrained_layout=True)
gs = fig.add_gridspec(
    nrows=3, ncols=3,
    width_ratios=[1.05, 1.0, 1.0],
)

# Original ONLY in the top-left cell (same height as other panels)
ax_orig     = fig.add_subplot(gs[0, 0])
ax_orig_mid = fig.add_subplot(gs[1, 0])  # blank spacer
ax_orig_err = fig.add_subplot(gs[2, 0])  # blank spacer

ax_down_std = fig.add_subplot(gs[0, 1])
ax_down_aa  = fig.add_subplot(gs[0, 2])
ax_rec_std  = fig.add_subplot(gs[1, 1])
ax_rec_aa   = fig.add_subplot(gs[1, 2])
ax_err_std  = fig.add_subplot(gs[2, 1])
ax_err_aa   = fig.add_subplot(gs[2, 2])

for ax in (ax_orig, ax_orig_mid, ax_orig_err,
           ax_down_std, ax_down_aa, ax_rec_std, ax_rec_aa, ax_err_std, ax_err_aa):
    ax.axis("off")

# --- Legend (benchmark-style) in the empty bottom-left cell -----------------
ax_leg_host = ax_orig_err
ax_leg_host.axis("off")

# Make a thick vertical bar as an inset axis inside the cell
leg = ax_leg_host.inset_axes([0.42, 0.05, 0.18, 0.90])  # [x0, y0, w, h] in axes fraction
leg.axis("off")

H_leg = 256
W_leg = 16   # thinner bar
y = np.linspace(1.0, 0.0, H_leg, dtype=np.float32)
legend_img = np.repeat(y[:, None], W_leg, axis=1)

leg.imshow(legend_img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")

# Labels next to the bar (keep them on the host axis so they don't get clipped)
ax_leg_host.text(0.62, 0.05, "-1", transform=ax_leg_host.transAxes,
                 fontsize=9, va="bottom", ha="left")
ax_leg_host.text(0.62, 0.50, "0", transform=ax_leg_host.transAxes,
                 fontsize=9, va="center", ha="left")
ax_leg_host.text(0.62, 0.95, "+1", transform=ax_leg_host.transAxes,
                 fontsize=9, va="top", ha="left")
ax_leg_host.text(0.5, 1.02, "Diff legend", transform=ax_leg_host.transAxes,
                 fontsize=10, va="bottom", ha="center")

ax_orig.set_title("Original")
im_orig = ax_orig.imshow(orig_u8)

# Row 1: Downsampled
t_down_std = ax_down_std.set_title(f"Standard cubic (z={zoom_values_cmp[0]:.3f})")
t_down_aa  = ax_down_aa.set_title (f"Cubic-antialiasing (z={zoom_values_cmp[0]:.3f})")
im_down_std = ax_down_std.imshow(canv_std[0])
im_down_aa  = ax_down_aa.imshow(canv_aa[0])

# Row 2: Recovered
t_rec_std = ax_rec_std.set_title(f"Recovered, SplineOps Cubic")
t_rec_aa  = ax_rec_aa.set_title (f"Recovered, SplineOps Cubic-antialiasing")
im_rec_std = ax_rec_std.imshow(recs_std[0])
im_rec_aa  = ax_rec_aa.imshow(recs_aa[0])

# Row 3: Signed error (benchmark-style normalization)
ax_err_std.set_title("Signed error")
ax_err_aa.set_title ("Signed error")
im_err_std = ax_err_std.imshow(diffs_std[0], cmap="gray", vmin=0.0, vmax=1.0)
im_err_aa  = ax_err_aa.imshow (diffs_aa[0],  cmap="gray", vmin=0.0, vmax=1.0)

def animate_frame(i: int):
    z = zoom_values_cmp[i]

    im_down_std.set_data(canv_std[i])
    im_down_aa.set_data(canv_aa[i])

    im_rec_std.set_data(recs_std[i])
    im_rec_aa.set_data(recs_aa[i])

    im_err_std.set_data(diffs_std[i])
    im_err_aa.set_data(diffs_aa[i])

    t_down_std.set_text(f"Standard cubic (z={z:.3f})")
    t_down_aa.set_text (f"Cubic-antialiasing (z={z:.3f})")

    t_rec_std.set_text(f"Recovered, SplineOps Cubic")
    t_rec_aa.set_text (f"Recovered, SplineOps Cubic-antialiasing")

    return (
        im_down_std, im_down_aa,
        im_rec_std, im_rec_aa,
        im_err_std, im_err_aa,
        t_down_std, t_down_aa,
        t_rec_std,  t_rec_aa,
    )

ani_cmp = animation.FuncAnimation(
    fig,
    animate_frame,
    frames=len(zoom_values_cmp),
    interval=INTERVAL_MS,   # keep consistent with your export
    blit=True,
)


# %%
# Export animation for the user-guide (build-only)
# ------------------------------------------------
#
# Writes into: <generated static dir>/_static/animations/
# No-op when run normally by users.

from splineops.utils.sphinx import export_animation_mp4_and_html

export_animation_mp4_and_html(
    ani_cmp,
    stem="resize_module_2d_cubic_vs_aa",
    interval_ms=INTERVAL_MS,  # matches FuncAnimation interval
    dpi=80,
    force=True,
)