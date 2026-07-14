# splineops/src/splineops/resize/_pycore/utils.py
from __future__ import annotations
import math
import numpy as np


def round_half_away_from_zero(value: float) -> int:
    """Round like C++ ``std::llround`` (ties away from zero)."""
    value = float(value)
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))

def border(size: int, degree: int, tol: float = 1e-10) -> int:
    if degree <= 1: return 0
    if   degree == 2: z = np.sqrt(8.0) - 3.0
    elif degree == 3: z = np.sqrt(3.0) - 2.0
    elif degree == 4: z = np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0
    elif degree == 5: z = (np.sqrt(135.0/2.0 - np.sqrt(17745.0/4.0)) + np.sqrt(105.0/4.0) - 13.0/2.0)
    elif degree == 6: z = -0.488294589303044755130118038883789062112279161239377608394
    elif degree == 7: z = -0.5352804307964381655424037816816460718339231523426924148812
    else: raise ValueError("border: degree [0..7]")
    horiz = 2 + int(np.log(tol)/np.log(abs(z)))
    return min(horiz, size)

def calculate_output_size_1d(N: int, zoom: float) -> int:
    if N <= 0:
        raise ValueError("input length must be positive")
    if not math.isfinite(zoom) or zoom <= 0.0:
        raise ValueError("zoom must be finite and positive")

    max_native_axis = (1 << 31) - 1

    def checked_round(value: float) -> int:
        if not math.isfinite(value) or value > max_native_axis:
            raise OverflowError("resized axis length exceeds native limits")
        return round_half_away_from_zero(value)

    return max(1, checked_round(N * zoom))

def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    s = [1]*len(shape)
    for i in range(len(shape)-2, -1, -1):
        s[i] = s[i+1] * shape[i+1]
    return tuple(s)
