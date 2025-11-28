# splineops/scripts/script_resize_2d.py
# -*- coding: utf-8 -*-
"""
Interactive image resize demo — grayscale-only, degree-aware comparison

Flow:
  1) Pick an image (PNG/JPG/TIFF)
  2) Pick zoom (>0) + method:
       - SciPy Linear / Quadratic / Cubic
       - Standard Linear / Quadratic / Cubic
       - Antialiasing Linear / Quadratic / Cubic
       - Least-Squares Linear / Quadratic / Cubic
  3) Show ORIGINAL grayscale (no text)
  4) Show RESIZED grayscale (no text)
  5) Show COMPARISON figure: original + the four families at the same degree with timing

Notes:
  - Displays use RGB uint8 (no colormap) to avoid large float RGBA buffers.
  - SciPy is optional; missing SciPy shows a friendly message in the comparison panel.
  - On macOS we force Matplotlib to the Qt-based backend so Tk isn't used by figures.
"""

from __future__ import annotations

import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# --- GUI / Matplotlib backend setup ---
try:
    from PyQt5 import QtWidgets  # single GUI toolkit for dialogs
    import matplotlib as mpl
    mpl.use("QtAgg")  # Use Qt-based backend on all platforms
except Exception:
    # Fallback: no PyQt5 available, let Matplotlib pick a default backend
    import matplotlib as mpl
    QtWidgets = None  # type: ignore[assignment]

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

# Optional ICC → sRGB (safe to skip if unavailable)
try:
    from PIL import ImageCms  # type: ignore
    _HAS_IMAGECMS = True
except Exception:
    _HAS_IMAGECMS = False

# Default storage dtype for the demo (change to np.float64 if desired)
DTYPE = np.float64

from PyQt5 import QtWidgets  # type: ignore[assignment]

# Import splineops (works when run directly or as module)
try:
    from splineops.resize import resize as sp_resize, resize_degrees as sp_resize_degrees
except Exception:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir   = repo_root / "src"
    if src_dir.exists() and str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from splineops.resize import resize as sp_resize, resize_degrees as sp_resize_degrees

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
    ("aa",      "Antialiasing"),
    ("ls",      "Least-Squares"),
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
        # Pure interpolation (no antialiasing)
        sp_method = degree
        out = sp_resize(data01, zoom_factors=(z, z), method=sp_method)
    elif family == "aa":
        # Oblique antialiasing
        sp_method = f"{degree}-antialiasing"
        out = sp_resize(data01, zoom_factors=(z, z), method=sp_method)
    elif family == "ls":
        # Equal-degree projection via explicit degrees
        degree_map = {"linear": 1, "quadratic": 2, "cubic": 3}
        n = degree_map[degree]
        out = sp_resize_degrees(
            data01,
            zoom_factors=(z, z),
            interp_degree=n,
            analy_degree=n,
            synthe_degree=n,
        )
    else:
        raise ValueError(f"Unsupported family for splineops: {family}")
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
# Tiny settings UI (PyQt5)
# ------------------------
class SettingsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        default_zoom: float = 0.5,
        default_method_key: str = "ls-cubic",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resize Settings")
        self._result: Optional[Tuple[float, str]] = None

        layout = QtWidgets.QGridLayout(self)

        # Zoom row
        zoom_label = QtWidgets.QLabel("Zoom factor (> 0):")
        self.zoom_edit = QtWidgets.QLineEdit(str(default_zoom))
        self.zoom_edit.setFixedWidth(100)

        layout.addWidget(zoom_label, 0, 0)
        layout.addWidget(self.zoom_edit, 0, 1)

        # Method row
        method_label = QtWidgets.QLabel("Method:")
        self.method_combo = QtWidgets.QComboBox()
        self.method_combo.addItems(METHOD_LABELS)

        default_label = KEY_TO_LABEL.get(default_method_key, "Least-Squares Cubic")
        idx = self.method_combo.findText(default_label)
        if idx >= 0:
            self.method_combo.setCurrentIndex(idx)
        else:
            self.method_combo.setCurrentIndex(0)

        layout.addWidget(method_label, 1, 0)
        layout.addWidget(self.method_combo, 1, 1)

        # Buttons row (OK / Cancel on the right)
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box, 2, 0, 1, 2)

        self.zoom_edit.selectAll()
        self.zoom_edit.setFocus()

        self.adjustSize()
        self._center_on_screen()

    @property
    def result(self) -> Optional[Tuple[float, str]]:
        return self._result

    def _center_on_screen(self) -> None:
        """Roughly center the dialog on the primary screen."""
        screen = QtWidgets.QApplication.primaryScreen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        y = geo.y() + (geo.height() - self.height()) // 3
        self.move(x, y)

    def _on_accept(self) -> None:
        try:
            z = float(self.zoom_edit.text().strip())
            if not np.isfinite(z) or z <= 0:
                raise ValueError
        except Exception:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid zoom",
                "Please enter a positive number for the zoom factor.",
            )
            return

        label = self.method_combo.currentText()
        key = LABEL_TO_KEY.get(label)
        if key is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid method",
                "Please choose a resize method.",
            )
            return

        self._result = (z, key)
        self.accept()

# ------------------------
# UI helpers
# ------------------------
def _select_image_with_dialog(parent: Optional[QtWidgets.QWidget] = None) -> Optional[Path]:
    filters = (
        "Image files (*.png *.jpg *.jpeg *.tif *.tiff);;"
        "PNG (*.png);;"
        "JPEG (*.jpg *.jpeg);;"
        "TIFF (*.tif *.tiff);;"
        "All files (*)"
    )
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent,
        "Select an image",
        "",
        filters,
    )
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
        ("aa",      "Antialiasing"),
        ("ls",      "Least-Squares"),
    ]
    results: List[Dict] = []
    for fam_key, fam_name in families:
        key = f"{fam_key}-{degree}"
        label = f"{fam_name} {degree.title()}"
        img = None
        elapsed: Optional[float] = None
        err: Optional[str] = None
        try:
            img = _resize_gray(gray01, key, zoom)
            elapsed = _avg_runtime(lambda: _resize_gray(gray01, key, zoom),
                                   runs=10, warmup=True)
        except Exception as e:
            err = str(e)
        results.append({"key": key, "label": label,
                        "img": img, "time": elapsed, "error": err})
    return results

def _comparison_figure(orig_gray: np.ndarray,
                       results: List[Dict],
                       zoom: float,
                       degree: str,
                       base_shape: Tuple[int, int]):
    """
    Show a comparison figure with the original image plus the resized outputs.
    """
    panels: List[Dict] = []

    # First panel: original image
    panels.append({
        "label": "Original",
        "img": orig_gray,
        "time": None,
        "error": None,
    })

    # Then each family result
    panels.extend(results)

    heights, widths = [], []
    for p in panels:
        if p["img"] is not None:
            h, w = p["img"].shape
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
    gs = fig.add_gridspec(1, len(panels), width_ratios=panel_ws_in)

    for i, p in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.set_axis_off()

        # Original panel: no timing, just a label
        if i == 0:
            ax.imshow(_as_rgb_u8(p["img"]), interpolation="nearest", aspect="equal")
            ax.set_title("Original", fontsize=10)
            continue

        title = f"{p['label']}\navg(10): {_fmt_time(p['time'])}"
        if p["img"] is not None:
            ax.imshow(_as_rgb_u8(p["img"]), interpolation="nearest", aspect="equal")
            ax.set_title(title, fontsize=10)
        else:
            ax.set_facecolor("0.92")
            ax.text(0.5, 0.55, p["label"], ha="center", va="center", fontsize=10)
            msg = "Error" if p["error"] else "Unavailable"
            detail = p["error"] or ""
            if p["label"].startswith("SciPy") and p["error"]:
                msg = "SciPy error"
            ax.text(0.5, 0.40, f"{msg}", ha="center", va="center", fontsize=9)
            if detail:
                ax.text(0.5, 0.28, detail[:48] + ("…" if len(detail) > 48 else ""),
                        ha="center", va="center", fontsize=8)
            ax.set_title(f"{p['label']}\navg(10): {_fmt_time(None)}", fontsize=10)

    fig.suptitle(f"Resize comparison @ zoom ×{zoom:g} — Degree: {degree.title()}",
                 fontsize=12)
    plt.show()

# ------------------------
# Main flow
# ------------------------
def main(argv=None) -> int:
    # Ensure a Qt application exists for dialogs
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    cli_path = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
    img_path = cli_path if (cli_path and cli_path.exists()) else _select_image_with_dialog(parent=None)
    if img_path is None:
        return 0  # cancelled

    dlg = SettingsDialog(parent=None, default_zoom=0.5, default_method_key="ls-cubic")
    if dlg.exec_() != QtWidgets.QDialog.Accepted or dlg.result is None:
        return 0  # cancelled
    zoom, method_key = dlg.result

    # Load + grayscale
    try:
        gray01 = _open_as_gray01(img_path)
    except Exception as e:
        QtWidgets.QMessageBox.critical(
            None,
            "Open failed",
            f"Could not open image:\n{img_path}\n\n{e}",
        )
        return 1

    # 1) Original grayscale
    _show_gray_image(gray01)

    # 2) Resized grayscale (selected method)
    try:
        out = _resize_gray(gray01, method_key, zoom)
    except Exception as e:
        print(f"Resize failed: {e}", file=sys.stderr)
        return 1
    _show_gray_image(out)

    # 3) Degree-matched comparison: original + families
    _, degree = _parse_method_key(method_key)
    results = _measure_families_at_degree(gray01, zoom, degree)
    _comparison_figure(gray01, results, zoom, degree, base_shape=gray01.shape)

    try:
        plt.close("all")
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
