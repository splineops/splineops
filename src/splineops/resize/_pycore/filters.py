# splineops/src/splineops/resize/_pycore/filters.py
from __future__ import annotations
import numpy as np
from typing import Sequence
from .bspline import beta

# poles and taps (Unser '93) – identical values to your current utils.py
def spline_poles(deg: int) -> np.ndarray:
    if deg <= 1: return np.array([], dtype=float)
    if   deg == 2: return np.array([np.sqrt(8.0)-3.0])
    elif deg == 3: return np.array([np.sqrt(3.0)-2.0])
    elif deg == 4: return np.array([
        np.sqrt(664.0 - np.sqrt(438976.0)) + np.sqrt(304.0) - 19.0,
        np.sqrt(664.0 + np.sqrt(438976.0)) - np.sqrt(304.0) - 19.0
    ])
    elif deg == 5: return np.array([
        np.sqrt(135.0/2.0 - np.sqrt(17745.0/4.0)) + np.sqrt(105.0/4.0) - 6.5,
        np.sqrt(135.0/2.0 + np.sqrt(17745.0/4.0)) - np.sqrt(105.0/4.0) - 6.5
    ])
    elif deg == 6: return np.array([
        -0.488294589303044755130118038883789062112279161239377608394,
        -0.081679271076237512597937765737059080653379610398148178525368,
        -0.00141415180832581775108724397655859252786416905534669851652709
    ])
    elif deg == 7: return np.array([
        -0.5352804307964381655424037816816460718339231523426924148812,
        -0.122554615192326690515272264359357343605486549427295558490763,
        -0.0091486948096082769285930216516478534156925639545994482648003
    ])
    else:
        raise ValueError("Invalid spline degree [0..7]")

def sampling_fir(deg: int) -> np.ndarray:
    if deg <= 1: return np.array([], dtype=float)
    if   deg == 2: return np.array([3.0/4.0, 1.0/8.0])
    elif deg == 3: return np.array([2.0/3.0, 1.0/6.0])
    elif deg == 4: return np.array([115.0/192.0, 19.0/96.0, 1.0/384.0])
    elif deg == 5: return np.array([11.0/20.0, 13.0/60.0, 1.0/120.0])
    elif deg == 6: return np.array([5887.0/11520.0, 10543.0/46080.0, 361.0/23040.0, 1.0/46080.0])
    elif deg == 7: return np.array([151.0/315.0, 397.0/1680.0, 1.0/42.0, 1.0/5040.0])
    else:
        raise ValueError("Invalid degree for sampling FIR [0..7]")

def initial_causal(c: np.ndarray, z: float, tol: float = 1e-10) -> float:
    N = c.size
    if N == 0: return 0.0
    zn = z**(N-1)
    horizon = min(N, int(2 + np.log(tol)/np.log(abs(z)))) if tol > 0 else N
    s = c[0] + zn*c[-1]
    if horizon > 2:
        n = np.arange(1, horizon-1)
        s += np.sum((z**n + zn/(z**n)) * c[1:horizon-1])
    return s / (1.0 - (zn*zn))

def initial_anti_causal(c: np.ndarray, z: float) -> float:
    if c.size < 2: return 0.0
    return (z*c[-2] + c[-1]) * z / (z*z - 1.0)

def get_interpolation_coefficients(c: np.ndarray, deg: int) -> None:
    if deg <= 1 or c.size <= 1: return
    poles = spline_poles(deg)
    lam = 1.0
    for z in poles: lam *= (1.0 - z) * (1.0 - 1.0/z)
    c *= lam
    for z in poles:
        c[0] = initial_causal(c, z)
        for n in range(1, c.size):
            c[n] += z * c[n-1]
        c[-1] = initial_anti_causal(c, z)
        for n in range(c.size-2, -1, -1):
            c[n] = z * (c[n+1] - c[n])

def symmetric_fir(h: Sequence[float], c: np.ndarray, s: np.ndarray) -> None:
    # identical logic to your current utils, condensed
    H = len(h); N = c.size
    if s.size != N: raise IndexError("Incompatible size")
    if H not in (2,3,4): raise ValueError("Invalid filter half-length (2..4)")
    # The long branching is correct and tested already in your repo:
    from splineops.resize.utils import symmetric_fir as ref
    ref(h, c, s)  # reuse the known-good implementation

def get_samples(c: np.ndarray, deg: int) -> None:
    if deg <= 1: return
    h = sampling_fir(deg)
    s = np.zeros_like(c)
    symmetric_fir(h, c, s)
    np.copyto(c, s)
