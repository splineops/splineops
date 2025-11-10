# splineops/scripts/script_resize.py
# -*- coding: utf-8 -*-
"""
Interactive image resize demo — grayscale-only

Flow:
  1) Pick an image (PNG/JPG/TIFF)
  2) Pick zoom (>0) + method (SciPy / Standard / Least-Squares / Oblique)
  3) Show ORIGINAL grayscale (no text)
  4) Show RESIZED grayscale (no text)
  5) Show COMPARISON figure: all 4 methods horizontally with timing

Notes:
  - SciPy is optional. If missing, the SciPy panel in the comparison figure
    says "SciPy not installed".
  - Use the Matplotlib "Save" toolbar button to export any figure.
"""

from __future__ import annotations

import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
from PIL import Image

# Optional ICC → sRGB (safe to skip if unavailable)
try:
    from PIL import ImageCms  # type: ignore
    _HAS_IMAGECMS = True
except Exception:
    _HAS_IMAGECMS = False

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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


def _fmt_time(sec: Optional[float]) -> str:
    if sec is None:
        return "n/a"
    return f"{sec*1000:.1f} ms" if sec < 1.0 else f"{sec:.3f} s"


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
# UI helpers
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
    """Borderless, pixel-accurate grayscale display (no text)."""
    h, w = img01.shape
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # full-bleed
    ax.imshow(img01, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
    ax.set_axis_off()
    plt.show()


# ------------------------
# Timing + comparison plot
# ------------------------
def _measure_all(gray01: np.ndarray, zoom: float):
    """Run all four methods once; return list of dicts with 'key','label','img','time','error'."""
    methods = [
        ("scipy",         "SciPy (cubic)"),
        ("standard",      "Standard (cubic)"),
        ("least-squares", "Least-Squares (best AA)"),
        ("oblique",       "Oblique (fast AA)"),
    ]
    results: List[Dict] = []

    for key, label in methods:
        img = None
        elapsed = None
        err = None
        try:
            t0 = time.perf_counter()
            if key == "scipy":
                img = _scipy_zoom_gray(gray01, zoom)
            else:
                img = _splineops_resize_gray(gray01, zoom, key)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            err = str(e)
        results.append({"key": key, "label": label, "img": img, "time": elapsed, "error": err})
    return results


def _comparison_figure(results: List[Dict], zoom: float, base_shape: Tuple[int, int]):
    """Build a single-row mosaic with variable-width panels and titles showing times."""
    # Compute width ratios based on each image aspect to give each panel fair space
    heights = []
    widths = []
    for r in results:
        if r["img"] is not None:
            h, w = r["img"].shape
        else:
            # fallback estimated size when missing: base_shape scaled
            h = max(1, int(round(base_shape[0] * zoom)))
            w = max(1, int(round(base_shape[1] * zoom)))
        heights.append(h); widths.append(w)

    # Normalize panel widths by aspect
    ratios = [w / max(h, 1) for w, h in zip(widths, heights)]
    # Set a fixed panel height (inches); width is proportional to each ratio
    panel_h_in = 3.4
    panel_ws_in = [max(2.2, panel_h_in * r) for r in ratios]  # min width for readability
    fig_w_in = sum(panel_ws_in)
    fig_h_in = panel_h_in + 0.7  # little room for titles

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=100)
    gs = gridspec.GridSpec(1, len(results), width_ratios=panel_ws_in, wspace=0.05, hspace=0.0)

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[0, i])
        ax.set_axis_off()
        title = f"{r['label']}\n{_fmt_time(r['time'])}"
        if r["img"] is not None:
            ax.imshow(r["img"], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
            ax.set_title(title, fontsize=10)
        else:
            ax.set_facecolor("0.92")
            ax.text(0.5, 0.55, r["label"], ha="center", va="center", fontsize=10)
            msg = "Error" if r["error"] else "Unavailable"
            detail = "SciPy not installed" if (r["key"] == "scipy" and r["error"]) else (r["error"] or "")
            ax.text(0.5, 0.40, f"{msg}", ha="center", va="center", fontsize=9)
            if detail:
                ax.text(0.5, 0.28, detail[:48] + ("…" if len(detail) > 48 else ""), ha="center", va="center", fontsize=8)
            ax.set_title(f"{r['label']}\n{_fmt_time(None)}", fontsize=10)

    fig.suptitle(f"Resize comparison @ zoom ×{zoom:g}", y=0.98, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ------------------------
# Main flow
# ------------------------
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

    h0, w0 = gray01.shape
    root.destroy()  # close Tk before showing figures

    # 1) Original grayscale (no text)
    _show_gray_image(gray01)

    # 2) Resized grayscale (no text) — chosen method
    try:
        if method_key == "scipy":
            out = _scipy_zoom_gray(gray01, zoom)
        else:
            out = _splineops_resize_gray(gray01, zoom, method_key)
    except Exception as e:
        messagebox.showerror("Resize failed", f"An error occurred during resizing:\n\n{e}")
        return 1
    _show_gray_image(out)

    # 3) Comparison figure: all 4 methods + timing
    results = _measure_all(gray01, zoom)
    _comparison_figure(results, zoom, base_shape=(h0, w0))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
