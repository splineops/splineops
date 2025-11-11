# splineops/src/splineops/resize/_pycore/utils.py
from __future__ import annotations
import numpy as np

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

def calculate_final_size_1d(inversable: bool, N: int, zoom: float) -> tuple[int, int]:
    if not inversable:
        return N, int(round(N * zoom))
    working = N
    s = int(round(round((working-1)*zoom)/zoom))
    while working - 1 - s != 0:
        working += 1
        s = int(round(round((working-1)*zoom)/zoom))
    final_ = int(round((working - 1) * zoom) + 1)
    return working, final_

def strides_from_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    s = [1]*len(shape)
    for i in range(len(shape)-2, -1, -1):
        s[i] = s[i+1] * shape[i+1]
    return tuple(s)
