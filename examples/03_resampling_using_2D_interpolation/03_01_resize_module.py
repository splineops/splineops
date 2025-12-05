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

# Original 1D samples f[k]
number_of_samples = 27
f_support = np.arange(number_of_samples, dtype=np.float64)
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
    coordinates=f_support,
    bases=base_1d,
    modes=mode_1d,
)

# Fine evaluation grid (for reference)
f_coords_1d = np.array([
    q / plot_points_per_unit_1d
    for q in range(plot_points_per_unit_1d * number_of_samples)
])
f_data_1d = f_1d(coordinates=(f_coords_1d,), grid=False)

# Choose a zoom factor for downsampling (e.g. similar to 27 // π ≈ 8)
K = number_of_samples
zoom_1d = K // np.pi / K  # coarse length ≈ 27//π, expressed as a zoom
zoom_1d = float(zoom_1d)

# 1) Use resize to define the coarse length and coarse samples
from splineops.resize import resize

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

g_support_length = g_samples_cubic.shape[0]

# 2) Reconstruct the continuous positions that resize uses internally
#    for pure interpolation:
#
#       step = (N - 1) / (outN - 1)
#       x_l  = step * l
#
#    (see make_plan_1d in the C++ core for the pure interpolation case).
step = (K - 1) / (g_support_length - 1) if g_support_length > 1 else 0.0
g_support_x = step * np.arange(g_support_length, dtype=np.float64)

# 3) Build TensorSplines on those coarse grids and evaluate them on a dense grid
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

# Dense grid over the same physical domain as f
g_coords_dense = f_coords_1d
g_cubic_data = g_cubic_ts(coordinates=(g_coords_dense,), grid=False)
g_aa_data    = g_aa_ts(coordinates=(g_coords_dense,), grid=False)

# (Optional) sanity check at the coarse sample positions:
# evaluate the fine spline at x_l and compare to resize("cubic")
f_at_xg = f_1d(coordinates=(g_support_x,), grid=False)
print("max |resize(cubic) - f(x_l)| =", np.max(np.abs(g_samples_cubic - f_at_xg)))

# Plot comparison
plt.figure(figsize=(10, 4))
plt.title("1D warm-up: cubic vs cubic-antialiasing on a coarse grid")

# Fine samples f[k] as stems (original signal on V₁)
plt.stem(f_support, f_samples_1d, basefmt=" ", label="f[k] samples")
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

# Standard cubic coarse spline (from resize)
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

# Cubic-antialiasing coarse spline
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

plt.xlabel("x (continuous coordinate)")
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
