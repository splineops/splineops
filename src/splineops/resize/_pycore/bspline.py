# splineops/src/splineops/resize/_pycore/bspline.py
import numpy as np

def _beta_scalar(x: float, degree: int) -> float:
    # unchanged scalar reference (kept for exact parity & scalar path)
    betan = 0.0
    if degree == 0:
        if abs(x) < 0.5 or x == -0.5: betan = 1.0
    elif degree == 1:
        x = abs(x);  betan = (1.0 - x) if x < 1.0 else 0.0
    elif degree == 2:
        x = abs(x)
        if x < 0.5: betan = 3.0/4.0 - x*x
        elif x < 1.5: x -= 1.5; betan = 0.5*x*x
    elif degree == 3:
        x = abs(x)
        if x < 1.0: betan = 0.5*x*x*(x - 2.0) + 2.0/3.0
        elif x < 2.0: x -= 2.0; betan = -(x*x*x)/6.0
    elif degree == 4:
        x = abs(x)
        if x < 0.5: t = x*x; betan = t*(t*0.25 - 5.0/8.0) + 115.0/192.0
        elif x < 1.5: betan = x*(x*(x*(5.0/6.0 - x*(1.0/6.0)) - 5.0/4.0) + 5.0/24.0) + 55.0/96.0
        elif x < 2.5: x -= 2.5; x *= x; betan = x*x*(1.0/24.0)
    elif degree == 5:
        x = abs(x)
        if x < 1.0: a = x*x; betan = a*(a*(0.25 - x*(1.0/12.0)) - 0.5) + 11.0/20.0
        elif x < 2.0: betan = x*(x*(x*(x*(x*(1.0/24.0) - 3.0/8.0) + 5.0/4.0) - 7.0/4.0) + 5.0/8.0) + 17.0/40.0
        elif x < 3.0: a = 3.0 - x; x = a*a; betan = a*x*x*(1.0/120.0)
    elif degree == 6:
        x = abs(x)
        if x < 0.5: t = x*x; betan = t*(t*(7.0/48.0 - x*(1.0/36.0)) - 77.0/192.0) + 5887.0/11520.0
        elif x < 1.5: betan = x*(x*(x*(x*(x*(x*(1.0/48.0) - 7.0/48.0) + 21.0/64.0) - 35.0/288.0) - 91.0/256.0) - 7.0/768.0) + 7861.0/15360.0
        elif x < 2.5: betan = x*(x*(x*(x*(x*(7.0/60.0 - x*(1.0/120.0)) - 21.0/32.0) + 133.0/72.0) - 329.0/128.0) + 1267.0/960.0) + 1379.0/7680.0
        elif x < 3.5: x -= 3.5; x *= (x*x); betan = x*x*(1.0/720.0)
    elif degree == 7:
        x = abs(x)
        if x < 1.0: a2 = x*x; betan = a2*(a2*(a2*(x*(1.0/144.0) - 1.0/36.0) + 1.0/9.0) - 1.0/3.0) + 151.0/315.0
        elif x < 2.0: betan = x*(x*(x*(x*(x*(x*(1.0/20.0 - x*(1.0/240.0)) - 7.0/30.0) + 0.5) - 7.0/18.0) - 0.1) - 7.0/90.0) + 103.0/210.0
        elif x < 3.0: betan = x*(x*(x*(x*(x*(x*(x*(1.0/720.0) - 1.0/36.0) + 7.0/30.0) - 19.0/18.0) + 49.0/18.0) - 23.0/6.0) + 217.0/90.0) - 139.0/630.0
        elif x < 4.0: a = 4.0 - x; x = a*a*a; betan = x*x*a*(1.0/5040.0)
    return betan

def beta(x, n: int):
    """Vectorized centered cardinal B-spline β_n(x), degrees 0..7."""
    if np.isscalar(x):
        return _beta_scalar(float(x), int(n))

    x = np.asarray(x, dtype=float)
    ax = np.abs(x)

    if n == 0:
        return np.where((ax < 0.5) | (x == -0.5), 1.0, 0.0)

    if n == 1:
        return np.where(ax < 1.0, 1.0 - ax, 0.0)

    if n == 2:
        out = np.zeros_like(ax)
        m0 = ax < 0.5
        out[m0] = 0.75 - ax[m0]*ax[m0]
        m1 = (~m0) & (ax < 1.5)
        t = ax[m1] - 1.5
        out[m1] = 0.5 * t * t
        return out

    if n == 3:
        out = np.zeros_like(ax)
        m0 = ax < 1.0
        out[m0] = 0.5*ax[m0]*ax[m0]*(ax[m0] - 2.0) + 2.0/3.0
        m1 = (~m0) & (ax < 2.0)
        t = ax[m1] - 2.0
        out[m1] = -(t*t*t) / 6.0
        return out

    # --- fully vectorized 4..7 ---
    out = np.zeros_like(ax)

    if n == 4:
        m0 = ax < 0.5
        t = ax[m0]*ax[m0]
        out[m0] = t*(t*0.25 - 5.0/8.0) + 115.0/192.0

        m1 = (~m0) & (ax < 1.5)
        a = ax[m1]
        out[m1] = a*(a*(a*(5.0/6.0 - a*(1.0/6.0)) - 5.0/4.0) + 5.0/24.0) + 55.0/96.0

        m2 = (~m0) & (~m1) & (ax < 2.5)
        t = ax[m2] - 2.5
        t = t*t
        out[m2] = (t*t) * (1.0/24.0)
        return out

    if n == 5:
        m0 = ax < 1.0
        a = ax[m0]*ax[m0]
        out[m0] = a*(a*(0.25 - ax[m0]*(1.0/12.0)) - 0.5) + 11.0/20.0

        m1 = (~m0) & (ax < 2.0)
        a = ax[m1]
        out[m1] = a*(a*(a*(a*(a*(1.0/24.0) - 3.0/8.0) + 5.0/4.0) - 7.0/4.0) + 5.0/8.0) + 17.0/40.0

        m2 = (~m0) & (~m1) & (ax < 3.0)
        a = 3.0 - ax[m2]
        t = a*a
        out[m2] = a * t * t * (1.0/120.0)
        return out

    if n == 6:
        m0 = ax < 0.5
        t = ax[m0]*ax[m0]
        out[m0] = t*(t*(7.0/48.0 - ax[m0]*(1.0/36.0)) - 77.0/192.0) + 5887.0/11520.0

        m1 = (~m0) & (ax < 1.5)
        a = ax[m1]
        out[m1] = a*(a*(a*(a*(a*(a*(1.0/48.0) - 7.0/48.0) + 21.0/64.0) - 35.0/288.0) - 91.0/256.0) - 7.0/768.0) + 7861.0/15360.0

        m2 = (~m0) & (~m1) & (ax < 2.5)
        a = ax[m2]
        out[m2] = a*(a*(a*(a*(a*(7.0/60.0 - a*(1.0/120.0)) - 21.0/32.0) + 133.0/72.0) - 329.0/128.0) + 1267.0/960.0) + 1379.0/7680.0

        m3 = (~m0) & (~m1) & (~m2) & (ax < 3.5)
        t = ax[m3] - 3.5
        t = t*(t*t)
        out[m3] = t*t * (1.0/720.0)
        return out

    if n == 7:
        m0 = ax < 1.0
        a2 = ax[m0]*ax[m0]
        out[m0] = a2*(a2*(a2*(ax[m0]*(1.0/144.0) - 1.0/36.0) + 1.0/9.0) - 1.0/3.0) + 151.0/315.0

        m1 = (~m0) & (ax < 2.0)
        a = ax[m1]
        out[m1] = a*(a*(a*(a*(a*(a*(1.0/20.0 - a*(1.0/240.0)) - 7.0/30.0) + 0.5) - 7.0/18.0) - 0.1) - 7.0/90.0) + 103.0/210.0

        m2 = (~m0) & (~m1) & (ax < 3.0)
        a = ax[m2]
        out[m2] = a*(a*(a*(a*(a*(a*(a*(1.0/720.0) - 1.0/36.0) + 7.0/30.0) - 19.0/18.0) + 49.0/18.0) - 23.0/6.0) + 217.0/90.0) - 139.0/630.0

        m3 = (~m0) & (~m1) & (~m2) & (ax < 4.0)
        a = 4.0 - ax[m3]
        t = a*a*a
        out[m3] = t*t*a * (1.0/5040.0)
        return out

    raise ValueError("Invalid degree for beta [0..7]")
