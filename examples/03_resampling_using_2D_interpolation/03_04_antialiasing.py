# -*- coding: utf-8 -*-
"""
A/B Antialiasing Demo — A, B, A/B, and the two downsamplings
(Downsampled detail box is EXACTLY half the original ROI, shown on original-size canvas.)
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

URL_A = "https://r0k.us/graphics/kodak/kodak/kodim14.png"
URL_B = "https://r0k.us/graphics/kodak/kodak/kodim08.png"

ROI_SIZE_PX = 64                 # original ROI side (pixels)
FACE_ROW, FACE_COL = 250, 445    # ROI center (approx) in ORIGINAL coordinates

# Shift the downsampled ROI a bit to a richer area (in original pixels)
SHIFT_ROW_PX = 48
SHIFT_COL_PX = 72

zoom = (0.5, 0.5)                # 0.5× downsampling demo

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def to_gray01(img_rgb_uint8: np.ndarray) -> np.ndarray:
    g = img_rgb_uint8.astype(np.float64) / 255.0
    return 0.2989 * g[..., 0] + 0.5870 * g[..., 1] + 0.1140 * g[..., 2]

# --------------------------------------------------------------------------- #
# Load & base ROI
# --------------------------------------------------------------------------- #

A = to_gray01(np.array(Image.open(BytesIO(requests.get(URL_A, timeout=10).content))))
B = to_gray01(np.array(Image.open(BytesIO(requests.get(URL_B, timeout=10).content))))
assert A.shape == B.shape, "Images A and B must have identical shape."

h_img, w_img = A.shape  # ORIGINAL canvas size (likely even, e.g., 512×768)

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
else:
    mixed_odd = mixed

h_odd, w_odd = mixed_odd.shape
assert (h_odd % 2 == 1) and (w_odd % 2 == 1), "Expect odd H×W after the crop."

# --------------------------------------------------------------------------- #
# 4) Downsample the ODD-sized mix in two ways (correct phase), but:
#    Display them on an ORIGINAL-size canvas to keep ROI = exact 32 px.
# --------------------------------------------------------------------------- #

res_std = resize(mixed_odd, zoom_factors=zoom, method="cubic")
res_ls  = resize(mixed_odd, zoom_factors=zoom, method="cubic-best_antialiasing")

# --------------------------------------------------------------------------- #
# 5) Show downsampled results on ORIGINAL-size canvas (exact half-size ROI)
# --------------------------------------------------------------------------- #

def show_resized_on_original_canvas(resized: np.ndarray, title: str):
    """
    Paste the resized (h_res×w_res) at (0,0) on an ORIGINAL-size white canvas (h_img×w_img),
    and show a magnified ROI that is EXACTLY half (32 px) of the original 64 px box.
    """
    h_res, w_res = resized.shape
    z_r, z_c = zoom

    # EXACT half-size detail box on the resized image
    roi_h_res = ROI_SIZE_PX // 2       # 64 → 32
    roi_w_res = ROI_SIZE_PX // 2

    # Choose base center in ORIGINAL coordinates, then apply shift
    base_center_r = np.clip(FACE_ROW + SHIFT_ROW_PX, 0, h_img - 1)
    base_center_c = np.clip(FACE_COL + SHIFT_COL_PX, 0, w_img - 1)

    # Map the ORIGINAL center to RESIZED coords
    center_r_res = int(round(base_center_r * z_r))
    center_c_res = int(round(base_center_c * z_c))

    # ROI top-left in RESIZED coords, clipped
    row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res - roi_h_res))
    col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res - roi_w_res))

    # Build ORIGINAL-size white canvas and paste resized at (0,0)
    canvas = np.ones((h_img, w_img), dtype=resized.dtype)
    canvas[:h_res, :w_res] = resized

    # IMPORTANT: Use ORIGINAL canvas height in roi_height_frac so 32 divides cleanly (e.g., 32 | 512).
    roi_kwargs_canvas = dict(
        roi_height_frac=(ROI_SIZE_PX // 2) / h_img,   # 32 / original height → no forced shrinking
        grayscale=True,
        roi_xy=(row_top_res, col_left_res),           # ROI coords within the pasted resized patch
    )

    _ = show_roi_zoom(
        canvas,
        ax_titles=(title, None),
        **roi_kwargs_canvas
    )

# Display both results with half-size ROI (32 px), guaranteed
show_resized_on_original_canvas(res_std, "Resized (standard cubic) on original-size canvas — 32px ROI")
show_resized_on_original_canvas(res_ls,  "Resized (least-squares, best AA) on original-size canvas — 32px ROI")
