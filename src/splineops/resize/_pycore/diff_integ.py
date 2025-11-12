# splineops/src/splineops/resize/_pycore/diff_integ.py
import numpy as np

def integ_sa(c: np.ndarray, m: float) -> None:
    c -= m
    c[0] *= 0.5
    c[1:] += np.cumsum(c[:-1])

def integ_as(c: np.ndarray, y: np.ndarray) -> None:
    # Caller provides a copy when needed; avoid extra copy here.
    y[0] = c[0]
    if y.size > 1:
        y[1] = 0.0
    if y.size > 2:
        y[2:] = -np.cumsum(c[1:-1])

def do_integ(c: np.ndarray, nb: int) -> float:
    N = c.size
    if N == 0 or nb <= 0:
        return 0.0
    def avg_of(x): return (2.0*np.sum(x) - x[-1] - x[0]) / (2.0*N - 2.0)
    average = 0.0
    if nb >= 1:
        average = avg_of(c);            integ_sa(c, average)
    if nb >= 2:
        tmp = c.copy();                 integ_as(tmp, c)
    if nb >= 3:
        m = avg_of(c);                  integ_sa(c, m)
    if nb >= 4:
        tmp = c.copy();                 integ_as(tmp, c)
    return average

def diff_sa(c: np.ndarray) -> None:
    if c.size < 2:
        return
    old = c[-2]
    c[:-1] -= c[1:]
    c[-1]  -= old

def diff_as(c: np.ndarray) -> None:
    if c.size == 0:
        return
    if c.size == 1:
        c[0] *= 2.0
        return
    # Vectorized in-place difference
    np.subtract(c[1:], c[:-1], out=c[1:])
    c[0] *= 2.0

def do_diff(c: np.ndarray, nb: int) -> None:
    if nb <= 0:
        return
    if nb == 1:
        diff_as(c); return
    if nb == 2:
        diff_sa(c); diff_as(c); return
    if nb == 3:
        diff_as(c); diff_sa(c); diff_as(c); return
    # nb >= 4
    diff_sa(c); diff_as(c); diff_sa(c); diff_as(c)
