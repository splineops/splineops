# splineops/src/splineops/resize/_pycore/engine.py
from __future__ import annotations
import numpy as np
from typing import Sequence
from .params import LSParams
from .resizend import resize_along_axis

# Numerical epsilon for zoom comparisons
_EPS = 1e-12

def compute_zoom(
    input_img: np.ndarray,
    output_img: np.ndarray,
    analy_degree: int,
    synthe_degree: int,
    interp_degree: int,
    zoom_factors: Sequence[float],
    shifts: Sequence[float],
    inversable: bool
) -> None:
    """
    Apply per-axis resize with the same magnification policy as the C++ path:
      • If zoom > 1 (magnification) or |zoom - 1| <= eps (identity),
        disable LS/Oblique projection (analy_degree = -1) to avoid ringing
        and to skip useless projection near identity.
      • For downsampling (zoom < 1), keep the requested LS/Oblique projection.
    """
    img = np.asarray(input_img, dtype=np.float64, order="C")
    out = img
    for ax, (z, b) in enumerate(zip(zoom_factors, shifts)):
        p = LSParams(
            interp_degree=interp_degree,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            zoom=float(z),
            shift=float(b),
            inversable=inversable,
        )

        # Magnification/identity policy (matches C++ binding):
        # - If zoom > 1 (magnify) OR zoom ≈ 1, disable projection along this axis.
        if p.analy_degree >= 0 and (abs(p.zoom - 1.0) <= _EPS or p.zoom > 1.0 + _EPS):
            p.analy_degree = -1

        # Fast identity short-circuit: if no projection and no shift, skip the axis
        if p.analy_degree < 0 and abs(p.zoom - 1.0) <= _EPS and abs(p.shift) <= 1e-15:
            continue

        out = resize_along_axis(out, ax, p)

    np.copyto(output_img, out)


def python_resize(
    data: np.ndarray,
    zoom_factors: Sequence[float],
    algo: str,                 # "interpolation" | "least-squares" | "oblique"
    degree: int,
    inversable: bool = False
) -> np.ndarray:
    # degree mapping (matches _resolve_degrees_for in the public wrapper)
    interp_degree = degree
    synthe_degree = degree
    if   algo == "interpolation": analy_degree = -1
    elif algo == "least-squares": analy_degree = degree
    else:                         analy_degree = 0 if degree == 1 else 1  # oblique

    shifts = [0.0] * len(zoom_factors)
    output_shape = tuple(int(round(n * z)) for n, z in zip(data.shape, zoom_factors))
    out = np.empty(output_shape, dtype=np.float64)
    compute_zoom(
        np.asarray(data, dtype=np.float64, order="C"),
        out,
        analy_degree=analy_degree,
        synthe_degree=synthe_degree,
        interp_degree=interp_degree,
        zoom_factors=list(map(float, zoom_factors)),
        shifts=shifts,
        inversable=inversable
    )
    return out
