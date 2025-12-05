# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_01_resize_module.py
# sphinx_gallery_end_ignore

"""
Resize Module
=============

Shrink and re-expand a 2-D RGB image with splineops, then discuss aliasing.
"""

# %%
# Imports and Helpers
# -------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from urllib.request import urlopen
from PIL import Image

from scipy.ndimage import zoom as ndi_zoom          # only for the *first* quick shrink
from splineops.utils.image import adjust_size_for_zoom    # makes dimensions compatible with the zoom factor
from splineops.resize import resize                 # core N-D spline resizer
from splineops.spline_interpolation.tensorspline import TensorSpline

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
})

# %%
# 1D Resize-Based Coarsening
# --------------------------
#
# Starting again from a 1D signal f[k] on a unit grid, we perform a 1D resize
# directly on the samples. We then compare:
#
# - a coarse spline built from `resize(..., method="cubic")`,
# - a coarse spline built from `resize(..., method="cubic-antialiasing")`,
# - the interpolating spline f(x) of the original samples for reference.

# 1) Original 1D samples f[k] (same as in 02_01 / 02_02)
number_of_samples = 27
f_support_1d = np.arange(number_of_samples, dtype=np.float64)
f_samples_1d = np.array([
    -0.657391, -0.641319, -0.613081, -0.518523, -0.453829, -0.385138,
    -0.270688, -0.179849, -0.11805, -0.0243016, 0.0130667, 0.0355389,
    0.0901577, 0.219599, 0.374669, 0.384896, 0.301386, 0.128646,
    -0.00811776, 0.0153119, 0.106126, 0.21688, 0.347629, 0.419532,
    0.50695, 0.544767, 0.555373
], dtype=np.float64)

# Fine spline f(x) on V₁ (unit grid)
plot_points_per_unit_1d = 12
base_1d = "bspline3"
mode_1d = "mirror"

f_1d = TensorSpline(
    data=f_samples_1d,
    coordinates=f_support_1d,
    bases=base_1d,
    modes=mode_1d,
)

# Dense evaluation grid for the fine spline
f_coords_1d = np.array([
    q / plot_points_per_unit_1d
    for q in range(plot_points_per_unit_1d * number_of_samples)
])
f_data_1d = f_1d(coordinates=(f_coords_1d,), grid=False)

# 2) Choose a coarse length: round(27 // π)
val_T = np.pi
K = number_of_samples
g_support_length = round(K // val_T)   # e.g., 27 // π ≈ 8

# Express this as a zoom factor for resize
zoom_1d = g_support_length / K        # e.g., 8 / 27

# 3) Coarse samples via resize: cubic and cubic-antialiasing
g_samples_cubic = resize(
    f_samples_1d,
    zoom_factors=(zoom_1d,),
    method="cubic",
).astype(np.float64)

g_samples_aa = resize(
    f_samples_1d,
    zoom_factors=(zoom_1d,),
    method="cubic-antialiasing",
).astype(np.float64)

L = g_samples_cubic.shape[0]

# 4) Reconstruct the continuous coarse grid used by resize for pure interpolation:
#
#       step = (K - 1) / (L - 1)
#       x_l  = step * l
#
step = (K - 1) / (L - 1) if L > 1 else 0.0
g_support_x = step * np.arange(L, dtype=np.float64)

# Build TensorSplines on that coarse grid
g_cubic_ts = TensorSpline(
    data=g_samples_cubic,
    coordinates=g_support_x,
    bases=base_1d,
    modes=mode_1d,
)
g_aa_ts = TensorSpline(
    data=g_samples_aa,
    coordinates=g_support_x,
    bases=base_1d,
    modes=mode_1d,
)

# Evaluate both coarse splines on the same dense grid as f(x)
g_coords_dense = f_coords_1d
g_cubic_data = g_cubic_ts(coordinates=(g_coords_dense,), grid=False)
g_aa_data    = g_aa_ts(coordinates=(g_coords_dense,), grid=False)

# 5) Optional sanity check at the coarse nodes: resize(cubic) vs fine spline sampled at x_l
f_at_xg = f_1d(coordinates=(g_support_x,), grid=False)
mse_cubic_nodes = np.mean((g_samples_cubic - f_at_xg) ** 2)
print(f"MSE at coarse nodes: resize(cubic) vs f(x_l) = {mse_cubic_nodes:.6e}")

# 6) Plot comparison
plt.figure(figsize=(10, 4))
plt.title("1D resize on f[k]: cubic vs cubic-antialiasing")

# Original samples f[k]
plt.stem(f_support_1d, f_samples_1d, basefmt=" ", label="f[k] samples")
plt.axhline(0, color="black", linewidth=1, zorder=0)

# Fine spline f(x) for reference
plt.plot(
    f_coords_1d,
    f_data_1d,
    color="gray",
    linewidth=2,
    alpha=0.5,
    label="fine f(x) (reference)",
)

# Coarse spline from resize(..., "cubic")
plt.plot(
    g_coords_dense,
    g_cubic_data,
    color="purple",
    linewidth=2,
    label="coarse spline (cubic)",
)
plt.plot(
    g_support_x,
    g_samples_cubic,
    "p",
    color="purple",
    mfc="none",
    markersize=10,
    markeredgewidth=2,
)

# Coarse spline from resize(..., "cubic-antialiasing")
plt.plot(
    g_coords_dense,
    g_aa_data,
    color="orange",
    linewidth=2,
    label="coarse spline (cubic-antialiasing)",
)
plt.plot(
    g_support_x,
    g_samples_aa,
    "o",
    color="orange",
    mfc="none",
    markersize=8,
    markeredgewidth=2,
)

plt.xlabel("x")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()


# %%
# Helpers for 2D Processing
# -------------------------

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

url = "https://r0k.us/graphics/kodak/kodak/kodim19.png"
with urlopen(url, timeout=10) as resp:
    img = Image.open(resp)
data = np.asarray(img, dtype=DTYPE) / DTYPE(255.0)          # H × W × 3, range [0, 1]

# 1) Quick down-size so the notebook images aren't huge
initial_shrink = 0.8
data_small = ndi_zoom(data, (initial_shrink, initial_shrink, 1), order=1)

# 2) Choose the demo shrink factor and make dimensions "zoom-friendly"
shrink_factor = 0.3
adjusted = adjust_size_for_zoom(data_small, shrink_factor).astype(DTYPE, copy=False)
adjusted_uint8 = (np.clip(adjusted, 0.0, 1.0) * 255).astype(np.uint8)

# Use the same ROI position as in the benchmarking example for kodim19
ROI_SIZE_PX = 256
ROI_CENTER_FRAC = (0.65, 0.35)
roi_rect = _roi_rect_from_frac_color(adjusted_uint8.shape, ROI_SIZE_PX, ROI_CENTER_FRAC)

# 3) Shrink with splineops (channel-wise): standard cubic
shrunken_cubic_f = resize_rgb(
    adjusted,
    shrink_factor,
    method="cubic",         # plain cubic interpolation (no anti-aliasing)
)
shrunken_cubic = (np.clip(shrunken_cubic_f, 0.0, 1.0) * 255).astype(np.uint8)

# 4) Shrink with splineops: cubic-antialiasing
shrunken_aa_f = resize_rgb(
    adjusted,
    shrink_factor,
    method="cubic-antialiasing",  # antialiasing shrink, degree 3
)
shrunken_aa = (np.clip(shrunken_aa_f, 0.0, 1.0) * 255).astype(np.uint8)

# %%
# Standard cubic shrink: original vs standard interpolation
# --------------------------------------------------------

show_intro_color(
    original_uint8=adjusted_uint8,
    shrunk_uint8=shrunken_cubic,
    roi_rect=roi_rect,
    zoom=shrink_factor,
    label="Standard cubic",
    degree_label="Cubic",
)

# %%
# Cubic-antialiasing shrink: original vs antialiased interpolation
# ----------------------------------------------------------------

show_intro_color(
    original_uint8=adjusted_uint8,
    shrunk_uint8=shrunken_aa,
    roi_rect=roi_rect,
    zoom=shrink_factor,
    label="Antialiasing",
    degree_label="Cubic",
)
