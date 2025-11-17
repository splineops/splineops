# splineops/scripts/script_resize_2d.py
# -*- coding: utf-8 -*-
"""
Interactive image resize demo — grayscale-only, degree-aware comparison

Flow:
  1) Pick an image (PNG/JPG/TIFF)
  2) Pick zoom (>0) + method:
       - SciPy Linear / Quadratic / Cubic
       - Standard Linear / Quadratic / Cubic
       - Least-Squares Linear / Quadratic / Cubic
       - Oblique Linear / Quadratic / Cubic
  3) Show ORIGINAL grayscale (no text)
  4) Show RESIZED grayscale (no text)
  5) Show COMPARISON figure: the four families at the same degree with timing

Notes:
  - Displays use RGB uint8 (no colormap) to avoid large float RGBA buffers.
  - SciPy is optional; missing SciPy shows a friendly message in the comparison panel.
  - On macOS we force Matplotlib to the native "MacOSX" backend so Tk isn't used by figures.
"""

from __future__ import annotations

import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING

# ---- Matplotlib backend (set BEFORE importing pyplot) ----
# Use native Cocoa windows on macOS to avoid TkAgg/Tkinter interactions.
if sys.platform == "darwin":
    os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")  # quiets some Tk deprecation logs
    try:
        import matplotlib as mpl
        mpl.use("MacOSX")
    except Exception:
        pass  # fallback to default; dialog-parenting still helps

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
from PIL import Image

# Optional ICC → sRGB (safe to skip if unavailable)
try:
    from PIL import ImageCms  # type: ignore
    _HAS_IMAGECMS = True
except Exception:
    _HAS_IMAGECMS = False

# Default storage dtype for the demo (change to np.float64 if desired)
DTYPE = np.float32

# --- Tkinter UI ---
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception:
    tk = None  # type: ignore[assignment]
    filedialog = None  # type: ignore[assignment]
    messagebox = None  # type: ignore[assignment]
    ttk = None  # type: ignore[assignment]

# Typing-only alias
if TYPE_CHECKING:
    import tkinter as tkt

# Import splineops (works when run directly or as module)
try:
    from splineops.resize import resize as sp_resize
except Exception:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir   = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from splineops.resize import resize as sp_resize

# -------------------------------
# Image I/O → grayscale [0,1]
# -------------------------------
def _to_srgb_if_possible(im: Image.Image) -> Image.Image:
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
    im = Image.open(str(path))

    # Convert to sRGB first if not grayscale (for accurate luminance)
    if im.mode not in ("L", "I;16", "I"):
        im = _to_srgb_if_possible(im)

    # Drop alpha
    if im.mode in ("RGBA", "LA"):
        im = im.convert("RGB")

    if im.mode == "L":
        arr = np.asarray(im, dtype=np.float64) / 255.0
    elif im.mode == "I;16":
        arr = np.asarray(im, dtype=np.uint16).astype(np.float64) / 65535.0
    elif im.mode == "I":
        arr = np.asarray(im, dtype=np.int32).astype(np.float64)
        amin, amax = float(arr.min()), float(arr.max())
        arr = (arr - amin) / (amax - amin + 1e-12)
    else:
        if im.mode != "RGB":
            im = im.convert("RGB")
        rgb = np.asarray(im, dtype=np.float64) / 255.0
        arr = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]

    im.close()
    return np.clip(arr, 0.0, 1.0).astype(DTYPE, copy=False)

# -------------------------------
# Display helpers
# -------------------------------
def _as_rgb_u8(img01: np.ndarray) -> np.ndarray:
    a = np.clip(img01, 0.0, 1.0)
    u8 = np.rint(a * 255.0).astype(np.uint8)
    return np.repeat(u8[..., None], 3, axis=2)

# -------------------------------
# Method mapping
# -------------------------------
DEGREES = ("linear", "quadratic", "cubic")
FAMILIES = (
    ("scipy",   "SciPy"),
    ("standard","Standard"),
    ("ls",      "Least-Squares"),
    ("oblique", "Oblique"),
)
METHOD_LABELS = [f"{fam_name} {deg.title()}" for fam_key, fam_name in FAMILIES for deg in DEGREES]
LABEL_TO_KEY = {f"{fam_name} {deg.title()}": f"{fam_key}-{deg}"
                for fam_key, fam_name in FAMILIES for deg in DEGREES}
KEY_TO_LABEL = {v: k for k, v in LABEL_TO_KEY.items()}

def _parse_method_key(method_key: str) -> Tuple[str, str]:
    family, degree = method_key.split("-", 1)
    return family, degree

def _avg_runtime(fn, runs: int = 10, warmup: bool = True) -> float:
    if warmup:
        fn()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    return (time.perf_counter() - t0) / runs

# -------------------------------
# Resizing backends (grayscale)
# -------------------------------
def _scipy_zoom_gray(data01: np.ndarray, z: float, degree: str) -> np.ndarray:
    from scipy.ndimage import zoom as ndi_zoom
    order_map = {"linear": 1, "quadratic": 2, "cubic": 3}
    order = order_map[degree]
    need_prefilter = (order >= 3)
    out = ndi_zoom(data01, (z, z), order=order, prefilter=need_prefilter,
                   mode="reflect", grid_mode=False)
    return np.clip(out, 0.0, 1.0)

def _splineops_resize_gray(data01: np.ndarray, z: float, family: str, degree: str) -> np.ndarray:
    if family == "standard":
        sp_method = degree
    elif family == "ls":
        sp_method = f"{degree}-best_antialiasing"
    elif family == "oblique":
        sp_method = f"{degree}-fast_antialiasing"
    else:
        raise ValueError(f"Unsupported family for splineops: {family}")
    out = sp_resize(data01, zoom_factors=(z, z), method=sp_method)
    return np.clip(out, 0.0, 1.0)

def _resize_gray(gray01: np.ndarray, method_key: str, zoom: float) -> np.ndarray:
    family, degree = _parse_method_key(method_key)
    if family == "scipy":
        return _scipy_zoom_gray(gray01, zoom, degree)
    return _splineops_resize_gray(gray01, zoom, family, degree)

def _fmt_time(sec: Optional[float]) -> str:
    if sec is None:
        return "n/a"
    return f"{sec*1000:.1f} ms" if sec < 1.0 else f"{sec:.3f} s"

# ------------------------
# Tiny settings UI (Tkinter)
# ------------------------
class SettingsDialog:
    def __init__(self, parent: "tkt.Tk", default_zoom: float = 0.5,
                 default_method_key: str = "ls-cubic") -> None:
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
        self.method_combo = ttk.Combobox(
            frm, values=METHOD_LABELS, state="readonly", width=30
        )
        default_label = KEY_TO_LABEL.get(default_method_key, "Least-Squares Cubic")
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
            messagebox.showerror("Invalid zoom", "Please enter a positive number for the zoom factor.",
                                 parent=self.top)
            return
        label = self.method_combo.get()
        key = LABEL_TO_KEY.get(label)
        if key is None:
            messagebox.showerror("Invalid method", "Please choose a resize method.", parent=self.top)
            return
        self.result = (z, key)
        self.top.destroy()

    def _on_cancel(self):
        self.result = None
        self.top.destroy()

# ------------------------
# UI helpers
# ------------------------
def _select_image_with_dialog(parent=None) -> Optional[Path]:
    filetypes = [
        ("Image files", ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")),
        ("PNG", "*.png"),
        ("JPEG", ("*.jpg", "*.jpeg")),
        ("TIFF", ("*.tif", "*.tiff")),
        ("All files", "*"),
    ]
    path = filedialog.askopenfilename(title="Select an image",
                                      filetypes=filetypes,
                                      parent=parent)
    return Path(path).expanduser() if path else None

def _show_gray_image(img01: np.ndarray):
    rgb = _as_rgb_u8(img01)
    h, w = rgb.shape[:2]
    dpi = 100.0
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, interpolation="nearest", aspect="equal")
    ax.set_axis_off()
    plt.show()

# ------------------------
# Timing + comparison plot
# ------------------------
def _measure_families_at_degree(gray01: np.ndarray, zoom: float, degree: str):
    families = [
        ("scipy",   "SciPy"),
        ("standard","Standard"),
        ("ls",      "Least-Squares"),
        ("oblique", "Oblique"),
    ]
    results: List[Dict] = []
    for fam_key, fam_name in families:
        key = f"{fam_key}-{degree}"
        label = f"{fam_name} {degree.title()}"
        img = None
        elapsed = None
        err = None
        try:
            img = _resize_gray(gray01, key, zoom)
            elapsed = _avg_runtime(lambda: _resize_gray(gray01, key, zoom),
                                   runs=10, warmup=True)
        except Exception as e:
            err = str(e)
        results.append({"key": key, "label": label,
                        "img": img, "time": elapsed, "error": err})
    return results

def _comparison_figure(results: List[Dict], zoom: float, degree: str, base_shape: Tuple[int, int]):
    heights, widths = [], []
    for r in results:
        if r["img"] is not None:
            h, w = r["img"].shape
        else:
            h = max(1, int(round(base_shape[0] * zoom)))
            w = max(1, int(round(base_shape[1] * zoom)))
        heights.append(h); widths.append(w)

    ratios = [w / max(h, 1) for w, h in zip(widths, heights)]
    panel_h_in = 3.4
    panel_ws_in = [max(2.2, panel_h_in * r) for r in ratios]
    fig_w_in = sum(panel_ws_in)
    fig_h_in = panel_h_in + 0.7

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=100, constrained_layout=True)
    gs = fig.add_gridspec(1, len(results), width_ratios=panel_ws_in)

    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[0, i])
        ax.set_axis_off()
        title = f"{r['label']}\navg(10): {_fmt_time(r['time'])}"
        if r["img"] is not None:
            ax.imshow(_as_rgb_u8(r["img"]), interpolation="nearest", aspect="equal")
            ax.set_title(title, fontsize=10)
        else:
            ax.set_facecolor("0.92")
            ax.text(0.5, 0.55, r["label"], ha="center", va="center", fontsize=10)
            msg = "Error" if r["error"] else "Unavailable"
            detail = ("SciPy not installed" if (r["key"].startswith("scipy-") and r["error"])
                      else (r["error"] or ""))
            ax.text(0.5, 0.40, f"{msg}", ha="center", va="center", fontsize=9)
            if detail:
                ax.text(0.5, 0.28, detail[:48] + ("…" if len(detail) > 48 else ""),
                        ha="center", va="center", fontsize=8)
            ax.set_title(f"{r['label']}\navg(10): {_fmt_time(None)}", fontsize=10)

    fig.suptitle(f"Resize comparison @ zoom ×{zoom:g} — Degree: {degree.title()}",
                 fontsize=12)
    plt.show()

# ------------------------
# Main flow
# ------------------------
def main(argv=None) -> int:
    if tk is None:
        print("Error: Tkinter is not available (install python3-tk).", file=sys.stderr)
        return 2

    # Create a root for dialogs, keep it alive until AFTER we’ve read the image & params.
    root = tk.Tk()
    root.withdraw()
    root.update()

    cli_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    img_path = cli_path if (cli_path and cli_path.exists()) else _select_image_with_dialog(parent=root)
    if img_path is None:
        try: root.destroy()
        except Exception: pass
        return 0  # cancelled

    dlg = SettingsDialog(root, default_zoom=0.5, default_method_key="ls-cubic")
    if dlg.result is None:
        try: root.destroy()
        except Exception: pass
        return 0  # cancelled
    zoom, method_key = dlg.result

    # Load + grayscale (report errors with the root as parent)
    try:
        gray01 = _open_as_gray01(img_path)
    except Exception as e:
        messagebox.showerror("Open failed",
                             f"Could not open image:\n{img_path}\n\n{e}",
                             parent=root)
        try: root.destroy()
        except Exception: pass
        return 1

    # IMPORTANT on macOS: destroy Tk root BEFORE opening any Matplotlib windows.
    # We’re using the "MacOSX" backend for figures, so they don’t need Tk at all.
    try:
        root.destroy()
    except Exception:
        pass

    # 1) Original grayscale
    _show_gray_image(gray01)

    # 2) Resized grayscale
    try:
        out = _resize_gray(gray01, method_key, zoom)
    except Exception as e:
        # No root here anymore: report to stderr and abort
        print(f"Resize failed: {e}", file=sys.stderr)
        return 1
    _show_gray_image(out)

    # 3) Degree-matched comparison
    _, degree = _parse_method_key(method_key)
    results = _measure_families_at_degree(gray01, zoom, degree)
    _comparison_figure(results, zoom, degree, base_shape=gray01.shape)

    try:
        plt.close('all')
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
