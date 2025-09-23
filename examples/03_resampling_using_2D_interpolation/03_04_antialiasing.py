# -*- coding: utf-8 -*-
"""
A/B Antialiasing Demo — display A, B, A/B mix, and the two downsamplings

- Construct "mixed" where each 2×2 tile uses A at the top-left pixel, B elsewhere.
- Crop to ODD H×W so 0.5× *interpolation* grid lands on the (0,0) corners.
- Downsample the mixed image with:
    (1) standard/cubic interpolation
    (2) least-squares/cubic-best_antialiasing
- For each downsampled image, paste on a white canvas of original size
  and show a zoomed ROI (same style as the gallery examples).
"""

# %%
# Imports
# -------
import numpy as np
import requests
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt

from splineops.resize.resize import resize
from splineops.utils import show_roi_zoom

# %%
# Load and convert to grayscale [0..1]
# ------------------------------------

def to_gray01(img_rgb_uint8: np.ndarray) -> np.ndarray:
    g = img_rgb_uint8.astype(np.float64) / 255.0
    return 0.2989 * g[..., 0] + 0.5870 * g[..., 1] + 0.1140 * g[..., 2]

# Kodak samples (same ones you used)
URL_A = "https://r0k.us/graphics/kodak/kodak/kodim14.png"
URL_B = "https://r0k.us/graphics/kodak/kodak/kodim08.png"

A = to_gray01(np.array(Image.open(BytesIO(requests.get(URL_A, timeout=10).content))))
B = to_gray01(np.array(Image.open(BytesIO(requests.get(URL_B, timeout=10).content))))
assert A.shape == B.shape, "Images A and B must have identical shape."

h_img, w_img = A.shape

# ---
# ROI (same face ROI as your examples)
ROI_SIZE_PX = 64
FACE_ROW, FACE_COL = 250, 445  # (row, col) approx center of face

# Compute ROI top-left on the original images
row_top = int(np.clip(FACE_ROW - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col_left = int(np.clip(FACE_COL - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))

roi_kwargs_orig = dict(
    roi_height_frac=ROI_SIZE_PX / h_img,
    grayscale=True,
    roi_xy=(row_top, col_left),
)

# %%
# 1) Show A, B
# ------------
_ = show_roi_zoom(A, ax_titles=("Image A (with ROI)", None), **roi_kwargs_orig)
_ = show_roi_zoom(B, ax_titles=("Image B (with ROI)", None), **roi_kwargs_orig)

# %%
# 2) Construct A/B mix (A at each 2×2 block top-left; B elsewhere)
# ----------------------------------------------------------------
mixed = B.copy()
mixed[0::2, 0::2] = A[0::2, 0::2]

_ = show_roi_zoom(mixed, ax_titles=("A/B corner mix (A at TL of each 2×2)", None), **roi_kwargs_orig)

# %%
# 3) Prepare for 0.5× downsampling: crop to ODD H×W for phase alignment
#    (so standard interpolation hits the (0,0) block corners exactly)
# ---------------------------------------------------------------------

H, W = mixed.shape
if (H % 2 == 0) or (W % 2 == 0):
    mixed_odd = mixed[:H - (H % 2 == 0), :W - (W % 2 == 0)]
    A_odd     = A[:mixed_odd.shape[0], :mixed_odd.shape[1]]
    B_odd     = B[:mixed_odd.shape[0], :mixed_odd.shape[1]]
else:
    mixed_odd, A_odd, B_odd = mixed, A, B

h_odd, w_odd = mixed_odd.shape
assert (h_odd % 2 == 1) and (w_odd % 2 == 1), "Expect odd H×W after the crop."

# Keep a matching ROI spec for the (possibly) cropped base
roi_kwargs_on_odd = dict(
    roi_height_frac=ROI_SIZE_PX / h_odd,
    grayscale=True,
    roi_xy=(min(row_top, h_odd - ROI_SIZE_PX), min(col_left, w_odd - ROI_SIZE_PX)),
)

# %%
# 4) Downsample the mixed image in two ways
# ----------------------------------------

zoom = (0.5, 0.5)

# (a) Standard (cubic) interpolation → should lock onto A's corners
res_std = resize(mixed_odd, zoom_factors=zoom, method="cubic")

# (b) Least-squares (cubic-best anti-aliasing) → should approach 2×2 box-average
res_ls  = resize(mixed_odd, zoom_factors=zoom, method="cubic-best_antialiasing")

# %%
# 5) Display: each downsampled result pasted on a white canvas of original size,
#    plus a magnified ROI from the *resized* image (same style as your examples).
# -----------------------------------------------------------------------------

def show_resized_on_canvas(resized: np.ndarray, title: str):
    h_res, w_res = resized.shape
    z_r, z_c = zoom

    # ROI size in the resized image (e.g., 64 -> 32 px when zoom=0.5)
    roi_h_res = max(1, int(round(ROI_SIZE_PX * z_r)))
    roi_w_res = max(1, int(round(ROI_SIZE_PX * z_c)))

    # Map the ROI center from original-odd coords into resized coords
    center_r_res = int(round(FACE_ROW * z_r))
    center_c_res = int(round(FACE_COL * z_c))

    # Clip to bounds in the resized image
    row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res - roi_h_res))
    col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res - roi_w_res))

    # Build a white canvas matching the (possibly cropped) original-odd size
    canvas = np.ones((h_odd, w_odd), dtype=resized.dtype)
    canvas[:h_res, :w_res] = resized

    # ROI params are relative to the *canvas* height; coordinates are on the resized patch
    roi_kwargs_canvas = dict(
        roi_height_frac=roi_h_res / h_odd,
        grayscale=True,
        roi_xy=(row_top_res, col_left_res),
    )

    _ = show_roi_zoom(
        canvas,
        ax_titles=(title, None),
        **roi_kwargs_canvas
    )

# Show both results on canvases
show_resized_on_canvas(res_std, "Resized (standard cubic) on canvas")
show_resized_on_canvas(res_ls,  "Resized (least-squares, best AA) on canvas")
