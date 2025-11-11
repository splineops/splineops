# splineops/src/splineops/resize/_pycore/bspline.py
import numpy as np

def beta(x: np.ndarray | float, n: int) -> np.ndarray | float:
    # Vectorized-friendly cardinal centered B-spline (deg 0..7)
    ax = np.abs(x)
    if n == 0:
        return np.where((ax < 0.5) | (x == -0.5), 1.0, 0.0)
    if n == 1:
        return np.where(ax < 1.0, 1.0 - ax, 0.0)
    # below uses scalar-like structure but works with ndarray via np.where
    if n == 2:
        out = np.zeros_like(ax, dtype=float)
        m0 = ax < 0.5
        out[m0] = 0.75 - ax[m0]*ax[m0]
        m1 = (~m0) & (ax < 1.5)
        t = ax[m1] - 1.5
        out[m1] = 0.5 * t * t
        return out
    if n == 3:
        out = np.zeros_like(ax, dtype=float)
        m0 = ax < 1.0
        out[m0] = 0.5*ax[m0]*ax[m0]*(ax[m0] - 2.0) + 2.0/3.0
        m1 = (~m0) & (ax < 2.0)
        t = ax[m1] - 2.0
        out[m1] = -(t*t*t)/6.0
        return out
    # Degrees 4..7 (rare in current presets) – delegate to scalar reference
    # for correctness over speed in fallback.
    from splineops.resize.utils import beta as beta_ref  # existing tested version
    if np.isscalar(x):
        return beta_ref(float(x), n)
    vec = np.vectorize(beta_ref, otypes=[float])
    return vec(x, n)
