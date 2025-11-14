# splineops/src/splineops/resize/_pycore/diff_integ.py
from __future__ import annotations
import numpy as np

# ------------------------------- 1-D (single) --------------------------------

def integ_sa(c: np.ndarray, m: float) -> None:
    c -= m
    c[0] *= 0.5
    c[1:] += np.cumsum(c[:-1])

def integ_as(c: np.ndarray, y: np.ndarray) -> None:
    y[0] = c[0]
    if y.size > 1: y[1] = 0.0
    if y.size > 2: y[2:] = -np.cumsum(c[1:-1])

def do_integ(c: np.ndarray, nb: int) -> float:
    N = c.size
    if N == 0 or nb <= 0: return 0.0
    def avg_of(x): return (2.0*np.sum(x) - x[-1] - x[0]) / (2.0*N - 2.0)
    average = 0.0
    if nb >= 1: average = avg_of(c);            integ_sa(c, average)
    if nb >= 2: tmp = c.copy();                 integ_as(tmp, c)
    if nb >= 3: m = avg_of(c);                  integ_sa(c, m)
    if nb >= 4: tmp = c.copy();                 integ_as(tmp, c)
    return average

def diff_sa(c: np.ndarray) -> None:
    if c.size < 2: return
    old = c[-2]
    c[:-1] -= c[1:]
    c[-1]  -= old

def diff_as(c: np.ndarray) -> None:
    if c.size == 0: return
    if c.size == 1: c[0] *= 2.0; return
    np.subtract(c[1:], c[:-1], out=c[1:])
    c[0] *= 2.0

def do_diff(c: np.ndarray, nb: int) -> None:
    if nb <= 0: return
    if nb == 1: diff_as(c); return
    if nb == 2: diff_sa(c); diff_as(c); return
    if nb == 3: diff_as(c); diff_sa(c); diff_as(c); return
    diff_sa(c); diff_as(c); diff_sa(c); diff_as(c)

# -------------------------------- batched ------------------------------------

def do_integ_batch(C: np.ndarray, nb: int) -> np.ndarray:
    """
    In-place integration over batch C (B, N). Returns per-line average (B,).
    """
    B, N = C.shape
    if N == 0 or nb <= 0:
        return np.zeros(B, dtype=C.dtype)

    def avg_of(X): return (2.0*np.sum(X, axis=1) - X[:, -1] - X[:, 0]) / (2.0*N - 2.0)

    avg = np.zeros(B, dtype=C.dtype)
    if nb >= 1:
        avg = avg_of(C)
        C -= avg[:, None]
        C[:, 0] *= 0.5
        if N > 1:
            C[:, 1:] += np.cumsum(C[:, :-1], axis=1)
    if nb >= 2:
        Z = C.copy()
        C[:, 0] = Z[:, 0]
        if N > 1: C[:, 1] = 0.0
        if N > 2: C[:, 2:] = -np.cumsum(Z[:, 1:-1], axis=1)
    if nb >= 3:
        m = avg_of(C)
        C -= m[:, None]
        C[:, 0] *= 0.5
        if N > 1:
            C[:, 1:] += np.cumsum(C[:, :-1], axis=1)
    if nb >= 4:
        Z = C.copy()
        C[:, 0] = Z[:, 0]
        if N > 1: C[:, 1] = 0.0
        if N > 2: C[:, 2:] = -np.cumsum(Z[:, 1:-1], axis=1)
    return avg

def do_diff_batch(C: np.ndarray, nb: int) -> None:
    """In-place differences over batch C (B, N)."""
    B, N = C.shape
    if N == 0 or nb <= 0:
        return

    def diff_sa_batch(X):
        if N < 2: return
        old = X[:, -2].copy()
        X[:, :-1] -= X[:, 1:]
        X[:, -1]  -= old

    def diff_as_batch(X):
        if N == 0: return
        if N == 1: X[:, 0] *= 2.0; return
        np.subtract(X[:, 1:], X[:, :-1], out=X[:, 1:])
        X[:, 0] *= 2.0

    if   nb == 1: diff_as_batch(C)
    elif nb == 2: diff_sa_batch(C); diff_as_batch(C)
    elif nb == 3: diff_as_batch(C); diff_sa_batch(C); diff_as_batch(C)
    else:         diff_sa_batch(C); diff_as_batch(C); diff_sa_batch(C); diff_as_batch(C)
