# splineops/src/splineops/resize/_pycore/filters.py
from __future__ import annotations
import numpy as np
from typing import Sequence
from functools import lru_cache

# ----------------------------- cached constants -----------------------------

@lru_cache(maxsize=None)
def spline_poles(deg: int) -> np.ndarray:
    if deg <= 1: return np.array([], dtype=float)
    if   deg == 2: return np.array([np.sqrt(8.0) - 3.0])
    elif deg == 3: return np.array([np.sqrt(3.0) - 2.0])
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

@lru_cache(maxsize=None)
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

# ------------------------------- 1-D (single) --------------------------------

def initial_causal(c: np.ndarray, z: float, tol: float = 1e-10) -> float:
    N = c.size
    if N == 0: return 0.0
    zn = z**(N-1)
    horizon = min(N, int(2 + np.log(tol)/np.log(abs(z)))) if tol > 0 else N
    s = c[0] + zn*c[-1]
    if horizon > 2:
        n = np.arange(1, horizon-1)
        s += np.sum((z**n + zn/(z**n)) * c[1:horizon-1])
    return s / (1.0 - (zn * zn))

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
        for n in range(c.size - 2, -1, -1):
            c[n] = z * (c[n+1] - c[n])

def symmetric_fir(h: Sequence[float], c: np.ndarray, s: np.ndarray) -> None:
    if s.size != c.size:
        raise IndexError("Incompatible size")
    H = len(h); N = c.size
    if H == 2:
        if N >= 2:
            s[0]   = h[0]*c[0] + 2.0*h[1]*c[1]
            s[1:-1]= h[0]*c[1:-1] + h[1]*(c[:-2]+c[2:])
            s[-1]  = h[0]*c[-1] + 2.0*h[1]*c[-2]
        else:
            s[0] = (h[0] + 2.0*h[1]) * c[0]
        return
    if H == 3:
        if N >= 4:
            s[0]   = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2]
            s[1]   = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3])
            s[2:-2]= h[0]*c[2:-2] + h[1]*(c[1:-3]+c[3:-1]) + h[2]*(c[0:-4]+c[4:])
            s[-2]  = h[0]*c[-2] + h[1]*(c[-3]+c[-1]) + h[2]*(c[-4]+c[-2])
            s[-1]  = h[0]*c[-1] + 2.0*h[1]*c[-2] + 2.0*h[2]*c[-3]
        elif N == 3:
            s[0] = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2]
            s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + 2.0*h[2]*c[1]
            s[2] = h[0]*c[2] + 2.0*h[1]*c[1] + 2.0*h[2]*c[0]
        elif N == 2:
            s[0] = (h[0] + 2.0*h[2]) * c[0] + 2.0*h[1]*c[1]
            s[1] = (h[0] + 2.0*h[2]) * c[1] + 2.0*h[1]*c[0]
        else:
            s[0] = (h[0] + 2.0*(h[1]+h[2])) * c[0]
        return
    if H == 4:
        if N >= 6:
            s[0]   = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2] + 2.0*h[3]*c[3]
            s[1]   = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + h[3]*(c[2]+c[4])
            s[2]   = h[0]*c[2] + h[1]*(c[1]+c[3]) + h[2]*(c[0]+c[4]) + h[3]*(c[1]+c[5])
            s[3:-3]= (h[0]*c[3:-3] + h[1]*(c[2:-4]+c[4:-2]) + h[2]*(c[1:-5]+c[5:-1]) + h[3]*(c[0:-6]+c[6:]))
            s[-3]  = h[0]*c[-3] + h[1]*(c[-4]+c[-2]) + h[2]*(c[-5]+c[-1]) + h[3]*(c[-6]+c[-2])
            s[-2]  = h[0]*c[-2] + h[1]*(c[-3]+c[-1]) + h[2]*(c[-4]+c[-2]) + h[3]*(c[-5]+c[-3])
            s[-1]  = h[0]*c[-1] + 2.0*h[1]*c[-2] + 2.0*h[2]*c[-3] + 2.0*h[3]*c[-4]
        elif N == 5:
            s[0]=h[0]*c[0]+2.0*h[1]*c[1]+2.0*h[2]*c[2]+2.0*h[3]*c[3]
            s[1]=h[0]*c[1]+h[1]*(c[0]+c[2])+h[2]*(c[1]+c[3])+h[3]*(c[2]+c[4])
            s[2]=h[0]*c[2]+(h[1]+h[3])*(c[1]+c[3])+h[2]*(c[0]+c[4])
            s[3]=h[0]*c[3]+h[1]*(c[2]+c[4])+h[2]*(c[1]+c[3])+h[3]*(c[0]+c[2])
            s[4]=h[0]*c[4]+2.0*h[1]*c[3]+2.0*h[2]*c[2]+2.0*h[3]*c[1]
        elif N == 4:
            s[0]=h[0]*c[0]+2.0*h[1]*c[1]+2.0*h[2]*c[2]+2.0*h[3]*c[3]
            s[1]=h[0]*c[1]+h[1]*(c[0]+c[2])+h[2]*(c[1]+c[3])+2.0*h[3]*c[2]
            s[2]=h[0]*c[2]+h[1]*(c[1]+c[3])+h[2]*(c[0]+c[2])+2.0*h[3]*c[1]
            s[3]=h[0]*c[3]+2.0*h[1]*c[2]+2.0*h[2]*c[1]+2.0*h[3]*c[0]
        elif N == 3:
            s[0]=h[0]*c[0]+2.0*(h[1]+h[3])*c[1]+2.0*h[2]*c[2]
            s[1]=h[0]*c[1]+(h[1]+h[3])*(c[0]+c[2])+2.0*h[2]*c[1]
            s[2]=h[0]*c[2]+2.0*(h[1]+h[3])*c[1]+2.0*h[2]*c[0]
        elif N == 2:
            s[0]=(h[0]+2.0*h[2])*c[0]+2.0*(h[1]+h[3])*c[1]
            s[1]=(h[0]+2.0*h[2])*c[1]+2.0*(h[1]+h[3])*c[0]
        else:
            s[0] = (h[0] + 2.0*(h[1]+h[2]+h[3])) * c[0]
        return
    raise ValueError("Invalid filter half-length (should be 2..4)")

def get_samples(c: np.ndarray, deg: int) -> None:
    if deg <= 1: return
    h = sampling_fir(deg)
    s = np.zeros_like(c)
    symmetric_fir(h, c, s)
    np.copyto(c, s)

# ------------------------------- batched path --------------------------------

def initial_causal_batch(C: np.ndarray, z: float, tol: float = 1e-10) -> np.ndarray:
    """C: (B, N) -> (B,)"""
    B, N = C.shape
    if N == 0: return np.zeros(B, dtype=C.dtype)
    zn = z**(N-1)
    horizon = min(N, int(2 + np.log(tol)/np.log(abs(z)))) if tol > 0 else N
    s = C[:, 0] + zn * C[:, -1]
    if horizon > 2:
        n = np.arange(1, horizon-1)
        w = (z**n + zn/(z**n))             # (h-2,)
        s += C[:, 1:horizon-1] @ w         # (B, h-2) @ (h-2,) -> (B,)
    return s / (1.0 - (zn * zn))

def initial_anti_causal_batch(C: np.ndarray, z: float) -> np.ndarray:
    """C: (B, N) -> (B,)"""
    if C.shape[1] < 2: return np.zeros(C.shape[0], dtype=C.dtype)
    return (z*C[:, -2] + C[:, -1]) * z / (z*z - 1.0)

def get_interpolation_coefficients_batch(C: np.ndarray, deg: int) -> None:
    """In-place IIR prefilter across a batch: C is (B, N)."""
    B, N = C.shape
    if deg <= 1 or N <= 1: return
    poles = spline_poles(deg)
    lam = 1.0
    for z in poles: lam *= (1.0 - z) * (1.0 - 1.0/z)
    C *= lam
    for z in poles:
        C[:, 0] = initial_causal_batch(C, z)
        for n in range(1, N):
            C[:, n] += z * C[:, n-1]
        C[:, -1] = initial_anti_causal_batch(C, z)
        for n in range(N-2, -1, -1):
            C[:, n] = z * (C[:, n+1] - C[:, n])

def symmetric_fir_batch(h: Sequence[float], C: np.ndarray) -> np.ndarray:
    """Symmetric FIR over a batch: C (B, N) -> S (B, N)."""
    H = len(h); B, N = C.shape
    S = np.zeros_like(C)
    if H == 2:
        h0, h1 = h
        if N >= 2:
            S[:, 0]    = h0*C[:, 0] + 2.0*h1*C[:, 1]
            if N > 2:
                S[:, 1:-1] = h0*C[:, 1:-1] + h1*(C[:, :-2] + C[:, 2:])
            S[:, -1]   = h0*C[:, -1] + 2.0*h1*C[:, -2]
        elif N == 1:
            S[:, 0] = (h0 + 2.0*h1) * C[:, 0]
        return S
    if H == 3:
        h0, h1, h2 = h
        if N >= 4:
            S[:, 0]    = h0*C[:, 0] + 2.0*h1*C[:, 1] + 2.0*h2*C[:, 2]
            S[:, 1]    = h0*C[:, 1] + h1*(C[:, 0]+C[:, 2]) + h2*(C[:, 1]+C[:, 3])
            if N > 4:
                S[:, 2:-2] = (h0*C[:, 2:-2] +
                              h1*(C[:, 1:-3]+C[:, 3:-1]) +
                              h2*(C[:, 0:-4]+C[:, 4:]))
            S[:, -2]   = h0*C[:, -2] + h1*(C[:, -3]+C[:, -1]) + h2*(C[:, -4]+C[:, -2])
            S[:, -1]   = h0*C[:, -1] + 2.0*h1*C[:, -2] + 2.0*h2*C[:, -3]
        elif N == 3:
            S[:, 0] = h0*C[:, 0] + 2.0*h1*C[:, 1] + 2.0*h2*C[:, 2]
            S[:, 1] = h0*C[:, 1] + h1*(C[:, 0]+C[:, 2]) + 2.0*h2*C[:, 1]
            S[:, 2] = h0*C[:, 2] + 2.0*h1*C[:, 1] + 2.0*h2*C[:, 0]
        elif N == 2:
            S[:, 0] = (h0 + 2.0*h2)*C[:, 0] + 2.0*h1*C[:, 1]
            S[:, 1] = (h0 + 2.0*h2)*C[:, 1] + 2.0*h1*C[:, 0]
        else:  # N == 1
            S[:, 0] = (h0 + 2.0*(h1+h2)) * C[:, 0]
        return S
    if H == 4:
        h0, h1, h2, h3 = h
        if N >= 6:
            S[:, 0]    = h0*C[:, 0] + 2.0*h1*C[:, 1] + 2.0*h2*C[:, 2] + 2.0*h3*C[:, 3]
            S[:, 1]    = h0*C[:, 1] + h1*(C[:, 0]+C[:, 2]) + h2*(C[:, 1]+C[:, 3]) + h3*(C[:, 2]+C[:, 4])
            S[:, 2]    = h0*C[:, 2] + h1*(C[:, 1]+C[:, 3]) + h2*(C[:, 0]+C[:, 4]) + h3*(C[:, 1]+C[:, 5])
            if N > 6:
                S[:, 3:-3] = (h0*C[:, 3:-3] +
                              h1*(C[:, 2:-4]+C[:, 4:-2]) +
                              h2*(C[:, 1:-5]+C[:, 5:-1]) +
                              h3*(C[:, 0:-6]+C[:, 6:]))
            S[:, -3]   = h0*C[:, -3] + h1*(C[:, -4]+C[:, -2]) + h2*(C[:, -5]+C[:, -1]) + h3*(C[:, -6]+C[:, -2])
            S[:, -2]   = h0*C[:, -2] + h1*(C[:, -3]+C[:, -1]) + h2*(C[:, -4]+C[:, -2]) + h3*(C[:, -5]+C[:, -3])
            S[:, -1]   = h0*C[:, -1] + 2.0*h1*C[:, -2] + 2.0*h2*C[:, -3] + 2.0*h3*C[:, -4]
        elif N == 5:
            S[:, 0]=h0*C[:,0]+2.0*h1*C[:,1]+2.0*h2*C[:,2]+2.0*h3*C[:,3]
            S[:, 1]=h0*C[:,1]+h1*(C[:,0]+C[:,2])+h2*(C[:,1]+C[:,3])+h3*(C[:,2]+C[:,4])
            S[:, 2]=h0*C[:,2]+(h1+h3)*(C[:,1]+C[:,3])+h2*(C[:,0]+C[:,4])
            S[:, 3]=h0*C[:,3]+h1*(C[:,2]+C[:,4])+h2*(C[:,1]+C[:,3])+h3*(C[:,0]+C[:,2])
            S[:, 4]=h0*C[:,4]+2.0*h1*C[:,3]+2.0*h2*C[:,2]+2.0*h3*C[:,1]
        elif N == 4:
            S[:,0]=h0*C[:,0]+2.0*h1*C[:,1]+2.0*h2*C[:,2]+2.0*h3*C[:,3]
            S[:,1]=h0*C[:,1]+h1*(C[:,0]+C[:,2])+h2*(C[:,1]+C[:,3])+2.0*h3*C[:,2]
            S[:,2]=h0*C[:,2]+h1*(C[:,1]+C[:,3])+h2*(C[:,0]+C[:,2])+2.0*h3*C[:,1]
            S[:,3]=h0*C[:,3]+2.0*h1*C[:,2]+2.0*h2*C[:,1]+2.0*h3*C[:,0]
        elif N == 3:
            S[:,0]=h0*C[:,0]+2.0*(h1+h3)*C[:,1]+2.0*h2*C[:,2]
            S[:,1]=h0*C[:,1]+(h1+h3)*(C[:,0]+C[:,2])+2.0*h2*C[:,1]
            S[:,2]=h0*C[:,2]+2.0*(h1+h3)*C[:,1]+2.0*h2*C[:,0]
        elif N == 2:
            S[:,0]=(h0+2.0*h2)*C[:,0]+2.0*(h1+h3)*C[:,1]
            S[:,1]=(h0+2.0*h2)*C[:,1]+2.0*(h1+h3)*C[:,0]
        else:
            S[:,0] = (h0 + 2.0*(h1+h2+h3)) * C[:,0]
        return S
    raise ValueError("Invalid filter half-length (should be 2..4)")

def get_samples_batch(C: np.ndarray, deg: int) -> None:
    if deg <= 1: return
    h = sampling_fir(deg)
    S = symmetric_fir_batch(h, C)
    np.copyto(C, S)
