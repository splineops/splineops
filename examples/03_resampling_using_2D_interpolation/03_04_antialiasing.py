# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/03_04_antialiasing.py
# sphinx_gallery_end_ignore

"""
Antialiasing
============

We construct an A/B corner mix image where, in each 2x2 tile, the
top-left pixel comes from image A and the other three come from B.
Then we downsample by 0.5 using:

- Standard interpolation (cubic): a non-anti-aliased decimation.
- Least-squares projection (cubic-best anti-aliasing): an anti-aliased decimation.
"""

# %%
# Imports
# -------

import numpy as np
import requests
from io import BytesIO
from PIL import Image

from splineops.resize.resize import resize
from splineops.utils import show_roi_zoom

# sphinx_gallery_thumbnail_number = 5  # show the fifth figure (std canvas) as thumbnail

# %%
# Load and prepare base ROI
# -------------------------
# Load A and B, convert to grayscale, and define the ROI on the originals.

URL_A = "https://r0k.us/graphics/kodak/kodak/kodim14.png"
URL_B = "https://r0k.us/graphics/kodak/kodak/kodim08.png"

ROI_SIZE_PX = 64               # original ROI side (pixels)
FACE_ROW, FACE_COL = 250, 445  # ROI center (approx) in ORIGINAL coordinates

zoom = (0.5, 0.5)              # 0.5× downsampling demo

def to_gray01(img_rgb_uint8: np.ndarray) -> np.ndarray:
    g = img_rgb_uint8.astype(np.float64) / 255.0
    return 0.2989 * g[..., 0] + 0.5870 * g[..., 1] + 0.1140 * g[..., 2]

A = to_gray01(np.array(Image.open(BytesIO(requests.get(URL_A, timeout=10).content))))
B = to_gray01(np.array(Image.open(BytesIO(requests.get(URL_B, timeout=10).content))))
assert A.shape == B.shape, "Images A and B must have identical shape."

h_img, w_img = A.shape  # ORIGINAL canvas size (e.g., 512×768)

# Original ROI (face) — top-left corner (for show_roi_zoom)
row_top = int(np.clip(FACE_ROW - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col_left = int(np.clip(FACE_COL - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))

# Keep the ROI center as *relative* position for later (downsampled views)
rel_center_r = FACE_ROW / h_img
rel_center_c = FACE_COL / w_img

roi_kwargs_orig = dict(
    roi_height_frac=ROI_SIZE_PX / h_img,
    grayscale=True,
    roi_xy=(row_top, col_left),
)

# %%
# Image A (with ROI)
# ------------------

_ = show_roi_zoom(A, ax_titles=("Image A (with ROI)", None), **roi_kwargs_orig)

# %%
# Image B (with ROI)
# ------------------

_ = show_roi_zoom(B, ax_titles=("Image B (with ROI)", None), **roi_kwargs_orig)

# %%
# Construct A/B corner mix
# ------------------------
# A fills the top-left of each 2×2 block; the other three pixels come from B.

mixed = B.copy()
mixed[0::2, 0::2] = A[0::2, 0::2]

_ = show_roi_zoom(mixed, ax_titles=("A/B corner mix (A at TL of each 2x2)", None), **roi_kwargs_orig)

# %%
# Resized (standard)
# ------------------

H, W = mixed.shape
if (H % 2 == 0) or (W % 2 == 0):
    mixed_odd = mixed[:H - (H % 2 == 0), :W - (W % 2 == 0)]
else:
    mixed_odd = mixed

h_odd, w_odd = mixed_odd.shape
assert (h_odd % 2 == 1) and (w_odd % 2 == 1), "Expect odd H×W after the crop."

res_std = resize(mixed_odd, zoom_factors=zoom, method="cubic")                    # standard (cubic)
res_ls  = resize(mixed_odd, zoom_factors=zoom, method="cubic-best_antialiasing")  # least-squares (best AA)

def show_resized_on_original_canvas_same_relpos(resized: np.ndarray, title: str):
    h_res, w_res = resized.shape

    # EXACT half-size ROI on the resized image
    roi_h_res = ROI_SIZE_PX // 2  # 64 → 32
    roi_w_res = ROI_SIZE_PX // 2

    # SAME RELATIVE CENTER as in originals
    center_r_res = int(round(rel_center_r * h_res))
    center_c_res = int(round(rel_center_c * w_res))

    # ROI top-left in RESIZED coords, clipped
    row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res - roi_h_res))
    col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res - roi_w_res))

    # Build ORIGINAL-size white canvas and paste resized at (0,0)
    canvas = np.ones((h_img, w_img), dtype=resized.dtype)
    canvas[:h_res, :w_res] = resized

    # Use ORIGINAL canvas height so 32 px is respected visually (no forced shrinking)
    roi_kwargs_canvas = dict(
        roi_height_frac=(ROI_SIZE_PX // 2) / h_img,   # 32 / original height
        grayscale=True,
        roi_xy=(row_top_res, col_left_res),           # ROI within the pasted resized patch
    )

    return show_roi_zoom(canvas, ax_titles=(title, None), **roi_kwargs_canvas)

_ = show_resized_on_original_canvas_same_relpos(
    res_std, "Resized (standard cubic) — same relative ROI, 32 px"
)

# %%
# Resized (least-squares, best AA)
# --------------------------------

_ = show_resized_on_original_canvas_same_relpos(
    res_ls, "Resized (least-squares, best AA) — same relative ROI, 32 px"
)

# %%
# Discussion
# ----------
#
# In this synthetic A/B mix, each 2×2 block has A at the top-left pixel and B
# elsewhere (i.e., 25% A, 75% B per block).
#
# • Standard interpolation (cubic) does no prefiltering before decimation.
#   With our odd-size tweak, the 0.5× sampling grid lands exactly on the 2×2
#   block corners (the A pixels). So it effectively *picks* A at every step,
#   yielding the A-by-corners image. That’s not anti-aliasing; it’s just
#   point-sampling at a favorable phase for this pattern.
#
# • Least-squares projection (best AA) performs a proper low-pass
#   (anti-aliasing) filtering matched to the downsampling, then decimates.
#   On this pattern, that filter averages over each 2×2 neighborhood, so the
#   result tends toward 25% A + 75% B — visually “more B”, more like
#   the mix. This is exactly what anti-aliasing should do: remove the high-frequency
#   checkerboard content so it doesn’t fold (alias) into the downsample.
#
# In short: interpolation without AA does sample-and-aliasing (here it locks onto A
# due to phase). Least-squares proejction does low-pass-then-sample, preserving what would
# survive an ideal anti-aliased decimation.