# splineops/tests/test_02_02_resize.py
import numpy as np
import pytest
from splineops.resize import resize

# --- helper to map (method, degree) -> new method preset string ---
def to_preset(method: str, degree: int) -> str:
    name = {0: "fast", 1: "linear", 2: "quadratic", 3: "cubic"}[degree]
    if method == "least-squares":
        return f"{name}-best_antialiasing"
    elif method == "oblique":
        return f"{name}-fast_antialiasing"
    elif method in {"interpolation", "standard"}:
        return name
    else:
        raise ValueError(f"Unknown method '{method}'")

# --- analysis degree & shift to mirror the algorithm’s coordinate mapping ---
def _analy_degree_of(method: str, degree: int) -> int:
    """
    Map the public method to the analysis degree used by the algorithm.
    - interpolation/standard: analy = -1
    - least-squares:         analy = degree
    - oblique:               analy = 0 for linear, 1 for quadratic/cubic
    """
    if method in {"interpolation", "standard"}:
        return -1
    if method == "least-squares":
        return degree
    if method == "oblique":
        return 0 if degree == 1 else 1
    raise ValueError(f"Unknown method '{method}'")

def _axis_shift(analy_degree: int, zoom: float) -> float:
    """
    Match the per-axis shift used in the implementation:
        x = l/zoom + ((n1+1)/2 - floor((n1+1)/2)) * (1/zoom - 1)
    For interpolation (analy = -1) the shift is 0.
    """
    if analy_degree < 0:
        return 0.0
    t = (analy_degree + 1.0) / 2.0
    return (t - np.floor(t)) * (1.0 / float(zoom) - 1.0)

def _per_axis_analy_degrees(method: str, degree: int, zoom_factors):
    base = _analy_degree_of(method, degree)
    return [base for _ in zoom_factors]

# --- central crop helpers (dimension- & zoom-aware) ---
def _central_crop_nd(arr: np.ndarray, pads):
    """
    Crop different 'pad' per axis. If an axis is too small, leave it as-is.
    """
    slices = []
    for n, p in zip(arr.shape, pads):
        if n <= 2 * p:
            slices.append(slice(0, n))
        else:
            slices.append(slice(p, n - p))
    return arr[tuple(slices)]

def _pads_for_crop(shape_out, degree: int, pattern_name: str, zoom_factors):
    """
    Base pad grows with spline degree; enlarge for patterns prone to ringing
    (checkerboard/sinusoid). For 3-D or axes with non-unity zoom, add extra
    margin to suppress mirrored-boundary mismatch.
    """
    base = max(2 * (degree + 1), 4)
    if pattern_name == "Checkerboard":
        base = max(base, 8)
    elif pattern_name == "Sinusoidal":
        base = max(base, 6)
    else:  # Polynomial or Gradient
        base = max(base, 4)

    D = len(shape_out)
    pads = []
    for n, z in zip(shape_out, zoom_factors):
        pad = base
        # In >=3D, boundary layers stack; use ~10% of the length as extra guard
        if D >= 3:
            pad = max(pad, int(0.10 * n))
        # If this axis actually changes size, add a bit more margin
        if abs(float(z) - 1.0) > 1e-12:
            pad = max(pad, 6)
        # Keep a reasonable interior: never crop more than a third of an axis
        pad = min(pad, n // 3)
        pads.append(pad)
    return pads

# --- mathematical patterns on continuous coordinates ---
def expected_gradient_value(coords, shape):
    return sum(coord / dim_len for coord, dim_len in zip(coords, shape)) / len(shape)

def expected_sinusoidal_value(coords, shape, freqs=None):
    if freqs is None:
        freqs = [5 * (i + 1) for i in range(len(shape))]
    values = [np.sin(2 * np.pi * freq * coord / dim_len)
              for coord, dim_len, freq in zip(coords, shape, freqs)]
    return (np.sum(values) / len(values)) * 0.25 + 0.5

def expected_checkerboard_value(coords, square_sizes):
    indices = [int(coord // square_size) for coord, square_size
               in zip(coords, square_sizes)]
    return (sum(indices) % 2) * 1.0  # 1.0 for white, 0.0 for black

# --- expected generation with algorithm-matched back-mapping + crop ---
def calculate_mse_with_expected(pattern_name,
                                shape,
                                zoom_factors,
                                resized_image,
                                freqs=None,
                                square_sizes=None,
                                degree=None,
                                method=None):
    """
    Build the expected field by mapping each output index back to input
    coordinates using the *same* per-axis mapping as the resampler:
        x_in = l/zoom + shift(analy_degree, zoom)
    Then compute MSE on a central crop to reduce boundary effects from mirror
    extension vs. analytic infinite-domain functions.
    """
    target_shape = resized_image.shape
    grids = np.meshgrid(*[np.arange(dim) for dim in target_shape], indexing="ij")

    analy_axes = _per_axis_analy_degrees(method, degree, zoom_factors)
    shifts = [_axis_shift(a, float(z)) for a, z in zip(analy_axes, zoom_factors)]

    def map_back(point):
        coords = []
        for i, (p, s) in enumerate(zip(point, shifts)):
            nin  = shape[i]
            nout = resized_image.shape[i]
            if nout > 1:
                step = (nin - 1) / float(nout - 1)
            else:
                step = 0.0
            coords.append(step * float(p) + s)
        return coords

    if pattern_name == "Gradient":
        expected = np.array(
            [expected_gradient_value(map_back(pt), shape)
             for pt in zip(*[g.flat for g in grids])]
        ).reshape(target_shape)
    elif pattern_name == "Sinusoidal":
        expected = np.array(
            [expected_sinusoidal_value(map_back(pt), shape, freqs)
             for pt in zip(*[g.flat for g in grids])]
        ).reshape(target_shape)
    elif pattern_name == "Checkerboard":
        expected = np.array(
            [expected_checkerboard_value(map_back(pt), square_sizes)
             for pt in zip(*[g.flat for g in grids])]
        ).reshape(target_shape)
    else:
        raise ValueError("Unknown pattern name")

    pads = _pads_for_crop(target_shape, (degree if degree is not None else 1),
                          pattern_name, zoom_factors)
    rr = _central_crop_nd(resized_image, pads)
    ee = _central_crop_nd(expected, pads)
    return float(np.mean((ee - rr) ** 2))

# --- synthetic pattern generation on the *input* grid ---
def generate_pattern(pattern_name, shape, zoom_factors, freqs=None, square_sizes=None):
    grid = np.meshgrid(*[np.linspace(0, dim_len - 1, dim_len)
                         for dim_len in shape], indexing="ij")
    if pattern_name == "Gradient":
        pattern = np.array([expected_gradient_value(coords, shape)
                            for coords in zip(*[g.flat for g in grid])]).reshape(shape)
    elif pattern_name == "Sinusoidal":
        pattern = np.array([expected_sinusoidal_value(coords, shape, freqs)
                            for coords in zip(*[g.flat for g in grid])]).reshape(shape)
    elif pattern_name == "Checkerboard":
        pattern = np.array([expected_checkerboard_value(coords, square_sizes)
                            for coords in zip(*[g.flat for g in grid])]).reshape(shape)
    else:
        raise ValueError("Unknown pattern name")
    return pattern

# --- test driver ---
def resize_pattern_and_calculate_mse(pattern_name, shape, zoom_factors, degree, method,
                                     freqs=None, square_sizes=None, dtype=np.float64):
    preset = to_preset(method, degree)
    # Generate pattern in float64, then cast once to the desired storage dtype
    pattern = generate_pattern(pattern_name, shape, zoom_factors,
                               freqs=freqs, square_sizes=square_sizes).astype(dtype)
    resized_image = resize(pattern, zoom_factors=zoom_factors, method=preset)
    # Compute error in float64
    mse = calculate_mse_with_expected(
        pattern_name,
        shape,
        zoom_factors,
        resized_image.astype(np.float64),
        freqs=freqs,
        square_sizes=square_sizes,
        degree=degree,
        method=method,
    )
    psnr = 10 * np.log10(1 / mse) if mse != 0 else float('inf')
    return mse, psnr

# --- parametrized tests (patterns) ---
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("pattern_name, shape, zoom_factors, degree, method, mse_threshold, psnr_threshold, freqs, square_sizes", [
    ("Gradient", (100,), (0.5,), 3, "least-squares", 1e-3, 40, None, None),
    ("Gradient", (100, 100), (0.75, 1.5), 1, "oblique", 1e-3, 60, None, None),
    ("Gradient", (50, 50, 50), (0.8, 2.8, 0.5), 3, "least-squares", 1e-3, 40, None, None),

    ("Sinusoidal", (100,), (0.5,), 1, "oblique", 4e-3, 23, [10], None),
    ("Sinusoidal", (100, 100), (0.314, 0.5), 3, "least-squares", 0.3, 6, [10, 5], None),
    ("Sinusoidal", (50, 50, 50), (1.8, 0.8, 0.5), 3, "least-squares", 0.3, 6, [10, 5, 3], None),

    ("Checkerboard", (100,), (0.5,), 3, "least-squares", 2e-2, 19, None, [10]),
    ("Checkerboard", (1000, 1000), (0.3, 1.6), 1, "oblique", 1e-2, 22, None, [100, 100]),
    ("Checkerboard", (50, 50, 50), (0.8, 1.2, 0.6), 1, "oblique", 0.1, 10, None, [10, 10, 10]),
])
def test_resize_n_dimensional_pattern(pattern_name, shape, zoom_factors, degree,
                                      method, mse_threshold, psnr_threshold, freqs, square_sizes, dtype):
    mse, psnr = resize_pattern_and_calculate_mse(
        pattern_name,
        shape,
        zoom_factors,
        degree,
        method,
        freqs=freqs,
        square_sizes=square_sizes,
        dtype=dtype,
    )
    assert mse < mse_threshold, f"{pattern_name} pattern MSE {mse} exceeds threshold {mse_threshold}"
    assert psnr > psnr_threshold, f"{pattern_name} pattern PSNR {psnr} dB below threshold {psnr_threshold}"

# --------------------------------------------------------------------
# Identity & polynomial-reproduction tests for Standard interpolation
# --------------------------------------------------------------------
@pytest.mark.parametrize("shape, degree", [
    ((64,), 0),
    ((64,), 1),
    ((48, 32), 1),
    ((48, 32), 2),
    ((24, 20, 16), 3),
])
def test_standard_identity_zoom_one(shape, degree):
    rng = np.random.default_rng(0)
    x = rng.random(shape, dtype=np.float64)
    zf = tuple(1.0 for _ in shape)
    method = to_preset("interpolation", degree)  # "fast/linear/quadratic/cubic"
    y = resize(x, zoom_factors=zf, method=method)
    err = np.max(np.abs(y - x))
    assert err < 1e-10, f"Identity failed (deg={degree}, shape={shape}), L∞={err}"

def _poly_expected(shape, zf, degree):
    D = len(shape)
    out_shape = tuple(int(round(n * z)) for n, z in zip(shape, zf))
    c = 0.2
    a1 = [0.3 / (i + 1) for i in range(D)]
    a2 = [0.2 / (i + 1) for i in range(D)]
    a3 = [0.1 / (i + 1) for i in range(D)]
    f = np.full(out_shape, c, dtype=np.float64)
    for i, n in enumerate(shape):
        m = out_shape[i]
        if m > 1:
            step = (n - 1) / float(m - 1)
            u_axis = (step * np.arange(m, dtype=np.float64)) / max(n - 1, 1)
        else:
            u_axis = np.zeros(m, dtype=np.float64)
        axis_shape = [1] * D
        axis_shape[i] = m
        u = u_axis.reshape(axis_shape)
        if degree >= 1:
            f += a1[i] * u
        if degree >= 2:
            f += a2[i] * (u ** 2)
        if degree >= 3:
            f += a3[i] * (u ** 3)
    return f

def _poly_tolerances(shape, deg_poly, deg_interp):
    """
    Tolerances for polynomial reproduction. For 1D/2D keep very tight limits.
    For 3D small grids with mixed zooms, allow a modest cushion to account
    for mirrored boundaries + finite-support separable filtering.
    """
    if len(shape) < 3:
        return 2e-6, 2e-7
    # 3D case (small shapes like 20×16×12 with mixed zooms)
    return 8e-4, 2e-4

@pytest.mark.parametrize("shape, zf, deg_poly, deg_interp", [
    # 1D
    ((64,), (0.7,), 1, 1),
    ((64,), (1.3,), 2, 2),
    ((64,), (0.65,), 3, 3),
    # 2D mixed zooms
    ((48, 32), (0.7, 1.2), 1, 1),
    ((48, 32), (1.3, 0.75), 2, 2),
    ((48, 32), (0.6, 0.8), 3, 3),
    # 3D mixed zooms
    ((20, 16, 12), (0.8, 1.25, 0.7), 1, 1),
    ((20, 16, 12), (1.2, 0.7, 0.9), 2, 2),
    ((20, 16, 12), (0.65, 1.4, 0.75), 3, 3),
])
def test_standard_polynomial_reproduction(shape, zf, deg_poly, deg_interp):
    """
    Standard (interpolation) of degree >= k should reproduce any polynomial
    of degree k evaluated at mapped coords u = j/z per axis. We compare
    on a central crop to avoid small boundary mismatches.
    """
    D = len(shape)
    grid = np.meshgrid(*[np.arange(n, dtype=np.float64) for n in shape], indexing="ij")
    norm = [max(n - 1, 1) for n in shape]
    c = 0.2
    a1 = [0.3 / (i + 1) for i in range(D)]
    a2 = [0.2 / (i + 1) for i in range(D)]
    a3 = [0.1 / (i + 1) for i in range(D)]
    src = np.full(shape, c, dtype=np.float64)
    for i in range(D):
        xi = grid[i] / norm[i]
        if deg_poly >= 1:
            src += a1[i] * xi
        if deg_poly >= 2:
            src += a2[i] * (xi ** 2)
        if deg_poly >= 3:
            src += a3[i] * (xi ** 3)

    method = to_preset("interpolation", deg_interp)
    y = resize(src, zoom_factors=zf, method=method)
    y_ref = _poly_expected(shape, zf, deg_poly)

    pads = _pads_for_crop(y.shape, deg_interp, "Polynomial", zf)
    y_c   = _central_crop_nd(y, pads)
    ref_c = _central_crop_nd(y_ref, pads)

    linf = float(np.max(np.abs(y_c - ref_c)))
    l1   = float(np.mean(np.abs(y_c - ref_c)))

    tol_inf, tol_l1 = _poly_tolerances(shape, deg_poly, deg_interp)
    assert linf < tol_inf and l1 < tol_l1, (
        f"Poly repro failed: L∞={linf}, L1={l1}, "
        f"shape={shape}, zf={zf}, deg_poly={deg_poly}, deg_interp={deg_interp} "
        f"(limits: L∞<{tol_inf}, L1<{tol_l1})"
    )
