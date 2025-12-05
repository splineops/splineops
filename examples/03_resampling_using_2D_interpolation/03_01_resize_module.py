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
# 1D Warm-Up: Standard vs Antialiasing
# ------------------------------------
#
# Before working with 2D images, we revisit the 1D spline setting. We start
# from the same samples :math:`f[k]` on a unit grid as in
# :ref:`sphx_glr_auto_examples_02_resampling_using_1d_interpolation_02_02_resample_a_1d_spline.py`.
# We then build two coarse versions:
#
# - one with plain cubic interpolation (no projection),
# - one with cubic projection-based antialiasing.
#
# We use :class:`TensorSpline` both as the fine spline model and to evaluate
# the coarse splines at the *same* continuous positions as :func:`resize`.

# --- 1) Start from the original fine samples f[k] as in 02_02 ---

number_of_samples = 27
f_support = np.arange(number_of_samples, dtype=np.float64)
f_samples = np.array([
    -0.657391, -0.641319, -0.613081, -0.518523, -0.453829, -0.385138,
    -0.270688, -0.179849, -0.11805, -0.0243016, 0.0130667, 0.0355389,
    0.0901577, 0.219599, 0.374669, 0.384896, 0.301386, 0.128646,
    -0.00811776, 0.0153119, 0.106126, 0.21688, 0.347629, 0.419532,
    0.50695, 0.544767, 0.555373
], dtype=np.float64)

plot_points_per_unit = 12
base = "bspline3"
mode = "mirror"

# Fine spline f(x) on the unit grid V₁
f_ts = TensorSpline(data=f_samples, coordinates=f_support, bases=base, modes=mode)

# --- 2) Build coarse samples g[k] on the physical grid x = T k (as in 02_02) ---

T = np.pi
g_support_length = round(number_of_samples // T)
k = np.arange(g_support_length, dtype=np.float64)
x_g_phys = k * T

# Physical coarse samples g_phys[k] = f(T k)
g_samples_phys = f_ts(coordinates=(x_g_phys,), grid=False)

# Up to here this matches 02_02: g_samples_phys is what you plotted there.

# --- 3) Now *re-interpret* g[k] on a unit grid and compare TensorSpline vs resize(cubic) ---

# Treat g_samples_phys as samples on the unit grid j = 0..M-1
g_support_idx = np.arange(g_support_length, dtype=np.float64)

# TensorSpline on the index grid
g_ts_idx = TensorSpline(
    data=g_samples_phys,
    coordinates=g_support_idx,
    bases=base,
    modes=mode,
)

# Choose a finer index grid (e.g. upsample by a factor of 8 in index-domain)
upsample_factor = 8
M = g_support_length
fine_len = upsample_factor * M
coords_idx_fine = np.linspace(0, M - 1, fine_len, dtype=np.float64)

g_ts_idx_data = g_ts_idx(coordinates=(coords_idx_fine,), grid=False)

# Use resize(cubic) on the *same* g[k] sequence in index-domain
g_resize_samples = resize(
    g_samples_phys,
    output_size=(fine_len,),
    method="cubic",
).astype(np.float64)

# By construction, resize's internal positions for these samples are
# exactly coords_idx_fine = linspace(0, M-1, fine_len),
# so we can compare directly:

mse = np.mean((g_ts_idx_data - g_resize_samples) ** 2)
print("MSE between TensorSpline (index-domain) and resize(cubic) on g[k] =", mse)

# --- 4) Plot to visually confirm ---

plt.figure(figsize=(10, 4))
plt.title("TensorSpline vs resize(cubic) on the same coarse samples g[k] (index domain)")

# Coarse samples g[k] on the index grid
plt.stem(g_support_idx, g_samples_phys, basefmt=" ", label="g[k] samples (index grid)")
plt.axhline(0, color="black", linewidth=1, zorder=0)

# TensorSpline curve
plt.plot(
    coords_idx_fine,
    g_ts_idx_data,
    linewidth=2,
    label="TensorSpline from g[k]",
)

# resize(cubic) curve
plt.plot(
    coords_idx_fine,
    g_resize_samples,
    linestyle="--",
    linewidth=2,
    label='resize(g[k], method="cubic")',
)

plt.xlabel("index-domain coordinate (0 .. M-1)")
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

# Helper to resize RGB image
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

    # Normalize zoom to (z_h, z_w) for the 2-D resize calls
    zoom_hw = (float(zoom), float(zoom))

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

# 3) Shrink with splineops (channel-wise)
shrunken_f = resize_rgb(
    adjusted,
    shrink_factor,
    method="cubic",         # plain cubic interpolation (no anti-aliasing)
)

# Convert to uint8 for display & composition
shrunken = (np.clip(shrunken_f, 0.0, 1.0) * 255).astype(np.uint8)

# Put the shrunken image on a white canvas the size of *adjusted*
H_adj, W_adj, _ = adjusted_uint8.shape
canvas = np.full_like(adjusted_uint8, 255)
canvas[: shrunken.shape[0], : shrunken.shape[1]] = shrunken

# 4) Re-expand to the original adjusted size (back to float [0, 1])
expanded = resize_rgb(
    shrunken.astype(DTYPE) / DTYPE(255.0),
    1.0 / shrink_factor,
    method="cubic",
)
expanded = np.clip(expanded, 0.0, 1.0)

# %%
# Expanded from Downsampled
# -------------------------
#
# We first show the final expanded image at large scale. This helps Sphinx
# generate a visually useful thumbnail and lets users preview the aliasing
# artefacts up front.

plt.figure(figsize=(10, 10))  # Tune size for thumbnail quality
plt.imshow(expanded)
plt.title(f"Expanded from Downsampled Image (×{1/shrink_factor:.1f})", fontsize=18)
plt.axis("off")
plt.tight_layout()
plt.show()

# %%
# Aliasing
# --------
#
# We go through the stages of shrinking the image and then expanding it.
# Note the wave-like artefacts in the expanded image: classic **aliasing**.
# When we shrink below the Nyquist limit, high-frequency detail folds back
# into lower frequencies.  Upsampling cannot recover the lost detail, so
# those aliased components become Moiré-style patterns.  A proper workflow
# would low-pass filter before down-sampling, but here we purposely show the
# artefacts to illustrate the point.

fig, axes = plt.subplots(3, 1, figsize=(8, 18))

axes[0].imshow(adjusted_uint8)
axes[0].set_title("Adjusted Original")
axes[0].axis("off")

axes[1].imshow(canvas)
axes[1].set_title(f"Shrunken (×{shrink_factor})")
axes[1].axis("off")

axes[2].imshow(expanded)
axes[2].set_title(f"Expanded (×{1/shrink_factor:.1f})")
axes[2].axis("off")

plt.tight_layout()
plt.show()

# %%
# Antialiasing shrink/expand
# --------------------------
#
# Now we repeat the same shrink/expand pipeline, but this time we use the
# **antialiasing** variant when shrinking:
#
#   * ``"cubic-antialiasing"`` applies an oblique-projection low-pass
#     filter before down-sampling, which strongly reduces aliasing.
#   * For the expansion step, plain cubic interpolation is enough; the
#     important part is that the shrink was anti-aliased.

aa_shrunken_f = resize_rgb(
    adjusted,
    shrink_factor,
    method="cubic-antialiasing",  # antialiasing shrink, degree 3
)
aa_shrunken = (np.clip(aa_shrunken_f, 0.0, 1.0) * 255).astype(np.uint8)

aa_expanded = resize_rgb(
    aa_shrunken.astype(DTYPE) / DTYPE(255.0),
    1.0 / shrink_factor,
    method="cubic",  # standard cubic interpolation for upsampling
)
aa_expanded = np.clip(aa_expanded, 0.0, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(expanded)
axes[0].set_title("Expanded after plain cubic shrink", fontsize=16)
axes[0].axis("off")

axes[1].imshow(aa_expanded)
axes[1].set_title("Expanded after antialiased shrink", fontsize=16)
axes[1].axis("off")

plt.tight_layout()
plt.show()
