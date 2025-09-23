# -*- coding: utf-8 -*-
"""
A/B Antialiasing Demo — A, B, A/B, and the two downsamplings (with shifted ROI on results)

- Construct "mixed" where each 2×2 tile uses A at the top-left pixel, B elsewhere.
- Crop to ODD H×W so 0.5× *interpolation* grid lands on the (0,0) corners.
- Downsample the mixed image with:
    (1) standard/cubic interpolation
    (2) least-squares/cubic-best_antialiasing
- For each downsampled image, paste on a white canvas (original-odd size)
  and show a **magnified ROI** shifted down/right to a richer area.
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

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Kodak samples
URL_A = "https://r0k.us/graphics/kodak/kodak/kodim14.png"
URL_B = "https://r0k.us/graphics/kodak/kodak/kodim08.png"

# ROI definition for ORIGINAL views (face)
ROI_SIZE_PX = 64
FACE_ROW, FACE_COL = 250, 445  # (row, col) approx center of face

# ROI shift for DOWNSAMPLED views (move detail down/right to a richer area)
# You can use absolute pixels OR uncomment the fractional version below.
SHIFT_ROW_PX = 48   # move ~48 px down (original-grid pixels)
SHIFT_COL_PX = 72   # move ~72 px right
# (fractional alternative)
# SHIFT_ROW_PX = None
# SHIFT_COL_PX = None

# Downsampling factor
zoom = (0.5, 0.5)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def to_gray01(img_rgb_uint8: np.ndarray) -> np.ndarray:
    g = img_rgb_uint8.astype(np.float64) / 255.0
    return 0.2989 * g[..., 0] + 0.5870 * g[..., 1] + 0.1140 * g[..., 2]

# --------------------------------------------------------------------------- #
# Load and prep
# --------------------------------------------------------------------------- #

A = to_gray01(np.array(Image.open(BytesIO(requests.get(URL_A, timeout=10).content))))
B = to_gray01(np.array(Image.open(BytesIO(requests.get(URL_B, timeout=10).content))))
assert A.shape == B.shape, "Images A and B must have identical shape."

h_img, w_img = A.shape

# Face ROI on originals
row_top = int(np.clip(FACE_ROW - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col_left = int(np.clip(FACE_COL - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))

roi_kwargs_orig = dict(
    roi_height_frac=ROI_SIZE_PX / h_img,
    grayscale=True,
    roi_xy=(row_top, col_left),
)

# --------------------------------------------------------------------------- #
# 1) Show A, B
# --------------------------------------------------------------------------- #

_ = show_roi_zoom(A, ax_titles=("Image A (with ROI)", None), **roi_kwargs_orig)
_ = show_roi_zoom(B, ax_titles=("Image B (with ROI)", None), **roi_kwargs_orig)

# --------------------------------------------------------------------------- #
# 2) Construct A/B corner mix
# --------------------------------------------------------------------------- #

mixed = B.copy()
mixed[0::2, 0::2] = A[0::2, 0::2]

_ = show_roi_zoom(mixed, ax_titles=("A/B corner mix (A at TL of each 2×2)", None), **roi_kwargs_orig)

# --------------------------------------------------------------------------- #
# 3) Crop to ODD size for phase alignment (so standard hits (0,0) corners)
# --------------------------------------------------------------------------- #

H, W = mixed.shape
if (H % 2 == 0) or (W % 2 == 0):
    mixed_odd = mixed[:H - (H % 2 == 0), :W - (W % 2 == 0)]
    A_odd     = A[:mixed_odd.shape[0], :mixed_odd.shape[1]]
    B_odd     = B[:mixed_odd.shape[0], :mixed_odd.shape[1]]
else:
    mixed_odd, A_odd, B_odd = mixed, A, B

h_odd, w_odd = mixed_odd.shape
assert (h_odd % 2 == 1) and (w_odd % 2 == 1), "Expect odd H×W after the crop."

# ROI for originals-but-odd (same face box, clipped if crop happened)
roi_kwargs_on_odd = dict(
    roi_height_frac=ROI_SIZE_PX / h_odd,
    grayscale=True,
    roi_xy=(min(row_top, h_odd - ROI_SIZE_PX), min(col_left, w_odd - ROI_SIZE_PX)),
)

# --------------------------------------------------------------------------- #
# 4) Downsample in two ways
# --------------------------------------------------------------------------- #

# Standard (cubic) interpolation → should lock onto A's corners
res_std = resize(mixed_odd, zoom_factors=zoom, method="cubic")

# Least-squares (cubic-best anti-aliasing) → should approach 2×2 box-average
res_ls  = resize(mixed_odd, zoom_factors=zoom, method="cubic-best_antialiasing")

# --------------------------------------------------------------------------- #
# 5) Show downsampled results on canvas (with shifted ROI)
# --------------------------------------------------------------------------- #

def show_resized_on_canvas(resized: np.ndarray, title: str):
    h_res, w_res = resized.shape
    z_r, z_c = zoom

    # ROI size (in resized space)
    roi_h_res = max(1, int(round(ROI_SIZE_PX * z_r)))
    roi_w_res = max(1, int(round(ROI_SIZE_PX * z_c)))

    # Choose base center in original-odd coords, with shift
    if SHIFT_ROW_PX is None:
        shift_r = int(0.08 * h_odd)  # ~8% down
    else:
        shift_r = SHIFT_ROW_PX
    if SHIFT_COL_PX is None:
        shift_c = int(0.10 * w_odd)  # ~10% right
    else:
        shift_c = SHIFT_COL_PX

    base_center_r = np.clip(FACE_ROW + shift_r, 0, h_odd - 1)
    base_center_c = np.clip(FACE_COL + shift_c, 0, w_odd - 1)

    # Map center into resized coords
    center_r_res = int(round(base_center_r * z_r))
    center_c_res = int(round(base_center_c * z_c))

    # ROI top-left in resized, clipped
    row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res - roi_h_res))
    col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res - roi_w_res))

    # Build canvas (white) of original-odd size and paste resized at (0,0)
    canvas = np.ones((h_odd, w_odd), dtype=resized.dtype)
    canvas[:h_res, :w_res] = resized

    # ROI params are relative to the canvas height; coords are in the resized patch
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

# Display both results
show_resized_on_canvas(res_std, "Resized (standard cubic) on canvas — shifted ROI")
show_resized_on_canvas(res_ls,  "Resized (least-squares, best AA) on canvas — shifted ROI")
