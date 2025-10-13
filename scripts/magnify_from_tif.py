# sphinx_gallery_start_ignore
# splineops/scripts/magnify_from_tif.py
# sphinx_gallery_end_ignore

"""
Least-Squares Magnification (TIFF from disk) — Full image
=========================================================

Pick a local .tif/.tiff (via CLI arg, native dialog, or console prompt),
convert to grayscale in [0, 1], magnify with the Least-Squares preset
(method="cubic-best_antialiasing"), and display the full original and
full magnified images side-by-side.
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
import matplotlib.pyplot as plt

from splineops.resize.resize import resize

# sphinx_gallery_thumbnail_number = 1

# %%
# Configuration + robust file selection
# -------------------------------------

# Usage:
#   python magnify_from_tif.py "C:\path\to\image.tif"
cli_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None

USE_FILE_DIALOG = bool(locals().get("USE_FILE_DIALOG", True))
IMAGE_PATH = Path(locals().get("IMAGE_PATH", "")).expanduser() if cli_path is None else cli_path

MAG = float(locals().get("MAG", 4.0))  # >1 = upsample

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
    except Exception:
        return None

# Try dialog if needed
if USE_FILE_DIALOG and _missing_or_not_file(IMAGE_PATH):
    picked = _try_open_file_dialog()
    if picked is not None:
        IMAGE_PATH = picked

# Optional terminal prompt
if _missing_or_not_file(IMAGE_PATH) and sys.stdin.isatty():
    try:
        typed = input("Enter path to a .tif/.tiff image (or leave blank to cancel): ").strip()
        if typed:
            IMAGE_PATH = Path(typed).expanduser()
    except EOFError:
        pass

# Final validation
if _missing_or_not_file(IMAGE_PATH):
    raise FileNotFoundError(
        "No valid file selected. Pass a path on the CLI, set IMAGE_PATH, "
        "or enable the native dialog with USE_FILE_DIALOG=True."
    )

# %%
# Load image and verify TIFF (by suffix or actual format)
# -------------------------------------------------------
im = Image.open(str(IMAGE_PATH))
if getattr(im, "is_animated", False):
    try:
        im.seek(0)  # first frame if multi-page
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

# %%
# Least-Squares magnification (full-image)
# ----------------------------------------
t0 = time.perf_counter()
img_mag = resize(img_gray01, zoom_factors=(MAG, MAG), method="cubic-best_antialiasing")
elapsed = time.perf_counter() - t0

print(f"[info] Original shape:  {img_gray01.shape}, dtype={img_gray01.dtype}")
print(f"[info] Magnified shape: {img_mag.shape}  (×{MAG:.2f})")
print(f"[info] Resize time:     {elapsed*1000:.1f} ms")

# %%
# Display: original and magnified side-by-side
# --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

axes[0].imshow(img_gray01, cmap="gray", aspect="equal")
axes[0].set_title("Original (grayscale)")
axes[0].axis("off")

axes[1].imshow(img_mag, cmap="gray", aspect="equal")
axes[1].set_title(f"Magnified ×{MAG:.2f} (Least-Squares)\nTime: {elapsed*1000:.1f} ms")
axes[1].axis("off")

plt.tight_layout()
plt.show()

# %%
# (Optional) Save the magnified image
# -----------------------------------
# out_path = IMAGE_PATH.with_name(IMAGE_PATH.stem + f"_x{MAG:.2f}_ls.tif")
# img_u16 = np.clip(img_mag * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
# Image.fromarray(img_u16).save(str(out_path))
# print(f"Saved: {out_path}")
