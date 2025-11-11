# splineops/tests/test_02_02_resize_ls_oblique.py
import numpy as np
import pytest
from splineops.resize.resize import resize

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
    """
    Apply the magnification policy per axis: for zoom>1, disable projection
    (i.e., use interpolation) to avoid ringing. This mirrors the native C++ path.
    """
    eps = 1e-12
    base = _analy_degree_of(method, degree)
    return [(-1 if (base >= 0 and float(z) > 1.0 + eps) else base) for z in zoom_factors]

def _central_crop(arr: np.ndarray, pad: int) -> np.ndarray:
    """
    Crop 'pad' pixels from each border of every axis. If an axis is too small,
    return the array as-is (no negative/empty slices).
    """
    if any(s <= 2 * pad for s in arr.shape):
        return arr
    slicers = tuple(slice(pad, s - pad) for s in arr.shape)
    return arr[slicers]

# --- mathematical patterns on continuous coordinates ---

def expected_gradient_value(coords, shape):
    return sum(coord / dim_len for coord, dim_len in zip(coords, shape)) / len(shape)

def expected_sinusoidal_value(coords, shape, freqs=None):
    if freqs is None:
        freqs = [5 * (i + 1) for i in range(len(shape))]
    values = [np.sin(2 * np.pi * freq * coord / dim_len) for coord, dim_len, freq in zip(coords, shape, freqs)]
    return (np.sum(values) / len(values)) * 0.25 + 0.5

def expected_checkerboard_value(coords, square_sizes):
    indices = [int(coord // square_size) for coord, square_size in zip(coords, square_sizes)]
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

    # per-axis analysis degree (with magnification fallback) and shifts
    analy_axes = _per_axis_analy_degrees(method, degree, zoom_factors)
    shifts = [_axis_shift(a, float(z)) for a, z in zip(analy_axes, zoom_factors)]

    def map_back(point):
        return [p / float(z) + s for p, z, s in zip(point, zoom_factors, shifts)]

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

    # crop borders to reduce boundary-condition mismatch
    base = max(2 * (degree + 1), 4) if degree is not None else 4
    if pattern_name == "Checkerboard":
        pad = max(base, 8)
    elif pattern_name == "Sinusoidal":
        pad = max(base, 6)
    else:
        pad = base

    rr = _central_crop(resized_image, pad)
    ee = _central_crop(expected, pad)
    mse = np.mean((ee - rr) ** 2)
    return mse

# --- synthetic pattern generation on the *input* grid ---

def generate_pattern(pattern_name, shape, zoom_factors, freqs=None, square_sizes=None):
    grid = np.meshgrid(*[np.linspace(0, dim_len - 1, dim_len) for dim_len in shape], indexing="ij")
    if pattern_name == "Gradient":
        pattern = np.array([expected_gradient_value(coords, shape) for coords in zip(*[g.flat for g in grid])]).reshape(shape)
    elif pattern_name == "Sinusoidal":
        pattern = np.array([expected_sinusoidal_value(coords, shape, freqs) for coords in zip(*[g.flat for g in grid])]).reshape(shape)
    elif pattern_name == "Checkerboard":
        pattern = np.array([expected_checkerboard_value(coords, square_sizes) for coords in zip(*[g.flat for g in grid])]).reshape(shape)
    else:
        raise ValueError("Unknown pattern name")
    return pattern

# --- test driver ---

def resize_pattern_and_calculate_mse(pattern_name, shape, zoom_factors, degree, method, freqs=None, square_sizes=None):
    preset = to_preset(method, degree)

    pattern = generate_pattern(pattern_name, shape, zoom_factors, freqs=freqs, square_sizes=square_sizes)
    pattern = pattern.astype(np.float64)
    resized_image = resize(pattern, zoom_factors=zoom_factors, method=preset)

    mse = calculate_mse_with_expected(pattern_name, shape, zoom_factors, resized_image,
                                      freqs=freqs, square_sizes=square_sizes,
                                      degree=degree, method=method)
    psnr = 10 * np.log10(1 / mse) if mse != 0 else float('inf')
    return mse, psnr

# --- parametrized tests ---

@pytest.mark.parametrize("pattern_name, shape, zoom_factors, degree, method, mse_threshold, psnr_threshold, freqs, square_sizes", [
    ("Gradient", (100,), (0.5,), 3, "least-squares", 1e-3, 60, None, None),
    ("Gradient", (100, 100), (0.75, 1.5), 1, "oblique", 1e-3, 60, None, None),
    ("Gradient", (50, 50, 50), (0.8, 2.8, 0.5), 3, "least-squares", 1e-3, 60, None, None),

    ("Sinusoidal", (100,), (0.5,), 1, "oblique", 4e-3, 23, [10], None),
    ("Sinusoidal", (100, 100), (0.314, 0.5), 3, "least-squares", 0.3, 6, [10, 5], None),
    ("Sinusoidal", (50, 50, 50), (1.8, 0.8, 0.5), 3, "least-squares", 0.3, 6, [10, 5, 3], None),

    ("Checkerboard", (100,), (0.5,), 3, "least-squares", 2e-2, 19, None, [10]),
    ("Checkerboard", (1000, 1000), (0.3, 1.6), 1, "oblique", 1e-2, 23, None, [100, 100]),
    ("Checkerboard", (50, 50, 50), (0.8, 1.2, 0.6), 1, "oblique", 1e-2, 21, None, [10, 10, 10]),
])
def test_resize_n_dimensional_pattern(pattern_name, shape, zoom_factors, degree, method, mse_threshold, psnr_threshold, freqs, square_sizes):
    mse, psnr = resize_pattern_and_calculate_mse(pattern_name, shape, zoom_factors, degree, method, freqs=freqs, square_sizes=square_sizes)
    assert mse < mse_threshold, f"{pattern_name} pattern MSE {mse} exceeds threshold {mse_threshold}"
    assert psnr > psnr_threshold, f"{pattern_name} pattern PSNR {psnr} dB below threshold {psnr_threshold}"
