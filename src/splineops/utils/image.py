# splineops/src/splineops/utils/image.py

"""
splineops.utils.image
=====================

Light-weight image helpers that have *no* GPU or splineops dependencies,
so they can be imported even in stripped-down CPU environments.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import zoom as _ndi_zoom

__all__: list[str] = [
    "crop_to_central_region",
    "adjust_size_for_zoom",
]


# --------------------------------------------------------------------------- #
# Public utilities
# --------------------------------------------------------------------------- #

def crop_to_central_region(img: np.ndarray, frac: float) -> np.ndarray:
    """
    Return the central region of *img* after discarding ``frac`` (0–1) of the
    height **and** width on each edge.

    Works for 2-D (H × W) or multi-channel (H × W × C) arrays; channels are
    preserved untouched.
    """
    if img.ndim < 2:
        raise ValueError("Expected an array with at least two spatial dims")

    h, w = img.shape[:2]
    top, left = int(h * frac), int(w * frac)
    slicer = (slice(top, h - top), slice(left, w - left)) + (Ellipsis,)
    return img[slicer]


def adjust_size_for_zoom(img: np.ndarray, zoom: float) -> np.ndarray:
    """
    Resample *img* so that a *subsequent* zoom by ``zoom`` (or its inverse)
    will keep dimensions integer.  Handy for “shrink-then-expand” demos.

    * Uses linear (order = 1) resampling via **SciPy**’s
      :pyfunc:`scipy.ndimage.zoom`.
    * Supports 2-D or H × W × C RGB arrays.

    Notes
    -----
    Suppose the original height is *H*.  We pick a new height ``H'`` such that
    ``H' * zoom`` is an integer, by rounding *H* to the nearest multiple of
    ``1/zoom``.  Width is treated analogously.
    """
    if img.ndim not in (2, 3):
        raise ValueError("Expected H×W or H×W×C array")

    h, w = img.shape[:2]
    inv = 1.0 / zoom
    new_h = round(round(h / inv) * inv)
    new_w = round(round(w / inv) * inv)

    factors: Tuple[float, ...] = (new_h / h, new_w / w) + (() if img.ndim == 2 else (1,))
    return _ndi_zoom(img, factors, order=1)
