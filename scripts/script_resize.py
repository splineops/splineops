# splineops/scripts/script_resize.py
# -*- coding: utf-8 -*-
"""
Interactive image resize demo — grayscale-only, two windows (original then resized).

- Prompts for an image (PNG/JPG/TIFF).
- Asks for zoom factor (>0) + method (SciPy / Standard / Least-Squares / Oblique).
- Converts to grayscale and processes that.
- Shows the original grayscale first (no text), then the resized grayscale (no text).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple
from io import BytesIO

import numpy as np
from PIL import Image

# Optional ICC → sRGB for accurate luminance (safe to skip if unavailable)
try:
    from PIL import ImageCms  # type: ignore
    _HAS_IMAGECMS = True
except Exception:
    _HAS_IMAGECMS = False

import matplotlib.pyplot as plt

# --- Tkinter UI ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:
    tk = None  # we'll error gracefully later

# Import splineops (works when run directly or as module)
try:
    from splineops.resize.resize import resize as sp_resize
except Exception:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from splineops.resize.resize import resize as sp_resize


# -------------------------------
# Image I/O → grayscale [0,1]
# -------------------------------
def _to_srgb_if_possible(im: Image.Image) -> Image.Image:
    """Convert any color image to sRGB if an ICC profile exists."""
    if not _HAS_IMAGECMS:
        return im
    icc = im.info.get("icc_profile")
    if not icc:
        return im
    try:
        src = ImageCms.ImageCmsProfile(BytesIO(icc))
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        dst = ImageCms.createProfile("sRGB")
        return ImageCms.profileToProfile(im, src, dst, outputMode="RGB")
    except Exception:
        return im


def _open_as_gray01(path: Path) -> np.ndarray:
    """
    Open image and return grayscale float64 in [0,1].
    Uses ITU-R BT.601 luminance for RGB.
    """
    im = Image.open(str(path))

    # Convert to sRGB first if not grayscale (for accurate luminance)
    if im.mode not in ("L", "I;16", "I"):
        im = _to_srgb_if_possible(im)

    # Drop alpha
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGB")

    # Grayscale paths
    if im.mode == "L":
        arr = np.asarray(im, dtype=np.float64) / 255.0
    elif im.mode == "I;16":
        arr = np.asarray(im, dtype=np.uint16).astype(np.float64) / 65535.0
    elif im.mode == "I":
        arr = np.asarray(im, dtype=np.int32).astype(np.float64)
        amin, amax = float(arr.min()), float(arr.max())
        arr = (arr - amin) / (amax - amin + 1e-12)
    else:
        # Color → luminance
        if im.mode != "RGB":
            im = im.convert("RGB")
        rgb = np.asarray(im, dtype=np.float64) / 255.0
        arr = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]

    im.close()
    return np.clip(arr, 0.0, 1.0)


# -------------------------------
# Resizing backends (grayscale)
# -------------------------------
def _scipy_zoom_gray(data01: np.ndarray, z: float) -> np.ndarray:
    try:
        from scipy.ndimage import zoom as ndi_zoom
    except Exception as e:
        raise RuntimeError("SciPy is required for the 'SciPy' method (pip install scipy).") from e
    out = ndi_zoom(data01, (z, z), order=3, prefilter=True, mode="reflect")
    return np.clip(out, 0.0, 1.0)


def _splineops_resize_gray(data01: np.ndarray, z: float, method_key: str) -> np.ndarray:
    preset_map = {
        "standard": "cubic",
        "least-squares": "cubic-best_antialiasing",
        "oblique": "cubic-fast_antialiasing",
    }
    out = sp_resize(data01, zoom_factors=(z, z), method=preset_map[method_key])
    return np.clip(out, 0.0, 1.0)


# ------------------------
# Tiny settings UI (Tkinter)
# ------------------------
class SettingsDialog:
    METHODS = [
        ("SciPy (cubic)", "scipy"),
        ("Standard (cubic)", "standard"),
        ("Least-Squares (best AA)", "least-squares"),
        ("Oblique (fast AA)", "oblique"),
    ]

    def __init__(self, parent: tk.Tk, default_zoom: float = 0.5, default_method_key: str = "least-squares"):
        self.parent = parent
        self.result: Optional[Tuple[float, str]] = None

        self.top = tk.Toplevel(parent)
        self.top.title("Resize Settings")
        self.top.resizable(False, False)
        self.top.grab_set()

        frm = ttk.Frame(self.top, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")

        ttk.Label(frm, text="Zoom factor (> 0):").grid(row=0, column=0, sticky="w")
        self.zoom_var = tk.StringVar(value=str(default_zoom))
        self.zoom_entry = ttk.Entry(frm, textvariable=self.zoom_var, width=12)
        self.zoom_entry.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ttk.Label(frm, text="Method:").grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.method_var = tk.StringVar(value=default_method_key)
        self.method_combo = ttk.Combobox(
            frm,
            textvariable=self.method_var,
            values=[label for label, _ in self.METHODS],
            state="readonly",
            width=28,
        )
        self._label_to_key = {label: key for label, key in self.METHODS}
        default_label = next(label for label, key in self.METHODS if key == default_method_key)
        self.method_combo.set(default_label)
        self.method_combo.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(10, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=2, column=0, columnspan=2, sticky="e", pady=(12, 0))
        ttk.Button(btns, text="OK", command=self._on_ok).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(btns, text="Cancel", command=self._on_cancel).grid(row=0, column=1)

        self.top.bind("<Return>", lambda e: self._on_ok())
        self.top.bind("<Escape>", lambda e: self._on_cancel())
        self.zoom_entry.focus_set()
        self.top.protocol("WM_DELETE_WINDOW", self._on_cancel)

        # Center dialog
        self.parent.update_idletasks(); self.top.update_idletasks()
        w, h = self.top.winfo_width(), self.top.winfo_height()
        x = (self.top.winfo_screenwidth() - w) // 2
        y = (self.top.winfo_screenheight() - h) // 3
        self.top.geometry(f"+{x}+{y}")

        self.parent.wait_window(self.top)

    def _on_ok(self):
        try:
            z = float(self.zoom_var.get().strip())
            if not np.isfinite(z) or z <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Invalid zoom", "Please enter a positive number for the zoom factor.")
            return
        label = self.method_combo.get()
        key = self._label_to_key.get(label)
        if key is None:
            messagebox.showerror("Invalid method", "Please choose a resize method.")
            return
        self.result = (z, key)
        self.top.destroy()

    def _on_cancel(self):
        self.result = None
        self.top.destroy()


# ------------------------
# Main flow
# ------------------------
def _select_image_with_dialog() -> Optional[Path]:
    filetypes = [
        ("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"),
        ("PNG", "*.png"),
        ("JPEG", "*.jpg;*.jpeg"),
        ("TIFF", "*.tif;*.tiff"),
        ("All files", "*.*"),
    ]
    path = filedialog.askopenfilename(title="Select an image", filetypes=filetypes)
    return Path(path).expanduser() if path else None


def _show_gray_image(img01: np.ndarray):
    """Borderless, pixel-accurate grayscale display."""
    h, w = img01.shape
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed, no margins
    ax.imshow(img01, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    ax.set_axis_off()
    plt.show()


def main(argv=None) -> int:
    if tk is None:
        print("Error: Tkinter is not available (install python3-tk).", file=sys.stderr)
        return 2

    root = tk.Tk(); root.withdraw(); root.update()

    cli_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    img_path = cli_path if (cli_path and cli_path.exists()) else _select_image_with_dialog()
    if img_path is None:
        root.destroy(); return 0  # cancelled

    dlg = SettingsDialog(root, default_zoom=0.5, default_method_key="least-squares")
    if dlg.result is None:
        root.destroy(); return 0  # cancelled
    zoom, method_key = dlg.result

    # Load + grayscale
    try:
        gray01 = _open_as_gray01(img_path)
    except Exception as e:
        messagebox.showerror("Open failed", f"Could not open image:\n{img_path}\n\n{e}")
        root.destroy(); return 1

    root.destroy()  # close Tk before showing figures

    # First window: original grayscale
    _show_gray_image(gray01)

    # Resize grayscale
    try:
        if method_key == "scipy":
            out = _scipy_zoom_gray(gray01, zoom)
        else:
            out = _splineops_resize_gray(gray01, zoom, method_key)
    except Exception as e:
        messagebox.showerror("Resize failed", f"An error occurred during resizing:\n\n{e}")
        return 1

    # Second window: resized grayscale
    _show_gray_image(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
