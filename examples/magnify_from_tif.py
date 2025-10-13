# sphinx_gallery_start_ignore
# splineops/examples/03_resampling_using_2d_interpolation/magnify_from_tif.py
# sphinx_gallery_end_ignore

"""
Least-Squares Magnification (TIFF from disk)
============================================

Pick a local ``.tif/.tiff`` image via a native file dialog (or by passing a
path on the CLI / setting IMAGE_PATH), convert to grayscale in [0, 1], and
magnify it with the Least-Squares preset (``method="cubic-best_antialiasing"``).
"""

# %%
# Imports
# -------
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from splineops.resize.resize import resize
from splineops.utils import show_roi_zoom

# sphinx_gallery_thumbnail_number = 2

# %%
# Configuration + robust file selection
# -------------------------------------

# You can pass a path on the CLI:
#   python magnify_from_tif.py "C:\path\to\image.tif"
cli_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None

USE_FILE_DIALOG = bool(locals().get("USE_FILE_DIALOG", True))
# If not using CLI, you can also set IMAGE_PATH via locals() when re-running
IMAGE_PATH = Path(locals().get("IMAGE_PATH", "")).expanduser() if cli_path is None else cli_path

MAG = float(locals().get("MAG", 2.0))            # >1 = upsample
ROI_SIZE_PX = int(locals().get("ROI_SIZE_PX", 64))
FACE_ROW = locals().get("FACE_ROW", None)        # ROI center row (optional)
FACE_COL = locals().get("FACE_COL", None)        # ROI center col (optional)

def _missing_or_not_file(p: Optional[Path]) -> bool:
    if p is None:
        return True
    s = str(p).strip()
    if not s:
        return True
    return not p.exists() or not p.is_file()

def _try_open_file_dialog() -> Optional[Path]:
    """Open native file dialog; return selected file or None if cancelled/unavailable."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select a TIFF image",
            filetypes=[("TIFF images", "*.tif *.tiff"), ("All files", "*.*")]
        )
        root.destroy()
        return Path(path).expanduser() if path else None
    except Exception as e:
        print(f"[info] GUI file dialog unavailable ({e}); falling back...", file=sys.stderr)
        return None

# Use dialog if requested and we don't already have a good file
if USE_FILE_DIALOG and _missing_or_not_file(IMAGE_PATH):
    picked = _try_open_file_dialog()
    if picked is not None:
        IMAGE_PATH = picked

# Optional last-resort prompt in interactive terminals
if _missing_or_not_file(IMAGE_PATH) and sys.stdin.isatty():
    try:
        typed = input("Enter path to a .tif/.tiff image (or leave blank to cancel): ").strip()
        if typed:
            IMAGE_PATH = Path(typed).expanduser()
    except EOFError:
        pass

# Final validation: must be an actual file, not a directory like "."
if _missing_or_not_file(IMAGE_PATH):
    raise FileNotFoundError(
        "No valid file selected. Pass a path on the CLI, set IMAGE_PATH, "
        "or enable the native dialog with USE_FILE_DIALOG=True."
    )

# %%
# Load image and verify TIFF (by suffix or actual format)
# -------------------------------------------------------

# Open first so we can detect actual format even if the suffix is odd/missing
im = Image.open(str(IMAGE_PATH))
if getattr(im, "is_animated", False):
    try:
        im.seek(0)  # first frame of multi-page TIFF
    except Exception:
        pass

fmt = (im.format or "").upper()
suffix_ok = IMAGE_PATH.suffix.lower() in (".tif", ".tiff")
format_ok = (fmt == "TIFF")

if not (suffix_ok or format_ok):
    im.close()
    raise ValueError(
        f"Selected file does not look like a TIFF:\n"
        f"  path = {IMAGE_PATH}\n  suffix = '{IMAGE_PATH.suffix}'  format = '{fmt}'"
    )

print(f"[info] Using image: {IMAGE_PATH}  (format: {fmt or 'unknown'})")

arr = np.asarray(im)  # H×W or H×W×C
im.close()

def to_gray01_from_any(a: np.ndarray) -> np.ndarray:
    """Convert H×W or H×W×C array to grayscale float64 in [0, 1]."""
    a = np.asarray(a)
    orig_dtype = a.dtype

    if a.ndim == 3:
        # Drop alpha if present
        if a.shape[2] == 1:
            a = a[..., 0]
        else:
            r = a[..., 0].astype(np.float64)
            g = a[..., 1].astype(np.float64)
            b = a[..., 2].astype(np.float64)
            if np.issubdtype(orig_dtype, np.integer):
                maxv = np.iinfo(orig_dtype).max
                r, g, b = r / maxv, g / maxv, b / maxv
            else:
                lo = min(r.min(), g.min(), b.min())
                hi = max(r.max(), g.max(), b.max())
                if hi > 1.0 or lo < 0.0:
                    eps = 1e-12
                    r = (r - lo) / (hi - lo + eps)
                    g = (g - lo) / (hi - lo + eps)
                    b = (b - lo) / (hi - lo + eps)
            a = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        a = a.astype(np.float64)
        if np.issubdtype(orig_dtype, np.integer):
            a /= np.iinfo(orig_dtype).max
        else:
            amin, amax = a.min(), a.max()
            if amax > 1.0 or amin < 0.0:
                a = (a - amin) / (amax - amin + 1e-12)

    return np.clip(a.astype(np.float64), 0.0, 1.0)

img_gray01 = to_gray01_from_any(arr)
h_img, w_img = img_gray01.shape

# %%
# ROI on the original image
# -------------------------
if FACE_ROW is None or FACE_COL is None:
    center_r, center_c = h_img // 2, w_img // 2
else:
    center_r, center_c = int(FACE_ROW), int(FACE_COL)

row_top = int(np.clip(center_r - ROI_SIZE_PX // 2, 0, h_img - ROI_SIZE_PX))
col_left = int(np.clip(center_c - ROI_SIZE_PX // 2, 0, w_img - ROI_SIZE_PX))

roi_kwargs_orig = dict(
    roi_height_frac=ROI_SIZE_PX / h_img,
    grayscale=True,
    roi_xy=(row_top, col_left),
)

_ = show_roi_zoom(img_gray01, ax_titles=("Original (from TIFF)", None), **roi_kwargs_orig)

# %%
# Least-Squares magnification
# ---------------------------
zoom_factors = (MAG, MAG)
t0 = time.perf_counter()
img_mag = resize(img_gray01, zoom_factors=zoom_factors, method="cubic-best_antialiasing")
elapsed = time.perf_counter() - t0

# Map the same ROI center into the magnified image
h_res, w_res = img_mag.shape
center_r_res = int(round(center_r * MAG))
center_c_res = int(round(center_c * MAG))
roi_h_res = max(1, int(round(ROI_SIZE_PX * MAG)))
roi_w_res = roi_h_res

row_top_res = int(np.clip(center_r_res - roi_h_res // 2, 0, h_res - roi_h_res))
col_left_res = int(np.clip(center_c_res - roi_w_res // 2, 0, w_res - roi_w_res))

roi_kwargs_mag = dict(
    roi_height_frac=roi_h_res / h_res,
    grayscale=True,
    roi_xy=(row_top_res, col_left_res),
)

_ = show_roi_zoom(
    img_mag,
    ax_titles=(f"Magnified ×{MAG:.2f} (Least-Squares)\nTime: {elapsed*1000:.1f} ms", None),
    **roi_kwargs_mag
)

# %%
# (Optional) Save the magnified image
# -----------------------------------
# out_path = IMAGE_PATH.with_name(IMAGE_PATH.stem + f"_x{MAG:.2f}_ls.tif")
# img_u16 = np.clip(img_mag * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
# Image.fromarray(img_u16).save(str(out_path))
# print(f"Saved: {out_path}")
