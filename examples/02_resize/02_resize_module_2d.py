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
