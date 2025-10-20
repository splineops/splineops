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

# Mathematical functions for expected values in each pattern
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

# Calculate MSE with expected values for N-dimensional patterns
def calculate_mse_with_expected(pattern_name, shape, zoom_factors, resized_image, freqs=None, square_sizes=None):
    target_shape = resized_image.shape
    grid = np.meshgrid(*[np.arange(dim) for dim in target_shape], indexing="ij")
    if pattern_name == "Gradient":
        expected_values = np.array([expected_gradient_value([coord / zoom_factor for coord, zoom_factor in zip(point, zoom_factors)], shape)
                                    for point in zip(*[g.flat for g in grid])]).reshape(target_shape)
    elif pattern_name == "Sinusoidal":
        expected_values = np.array([expected_sinusoidal_value([coord / zoom_factor for coord, zoom_factor in zip(point, zoom_factors)], shape, freqs)
                                    for point in zip(*[g.flat for g in grid])]).reshape(target_shape)
    elif pattern_name == "Checkerboard":
        expected_values = np.array([expected_checkerboard_value([coord / zoom_factor for coord, zoom_factor in zip(point, zoom_factors)], square_sizes)
                                    for point in zip(*[g.flat for g in grid])]).reshape(target_shape)
    else:
        raise ValueError("Unknown pattern name")
    mse = np.mean((expected_values - resized_image) ** 2)
    return mse

# Generate pattern for N dimensions
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

# Test function for resizing N-dimensional patterns
def resize_pattern_and_calculate_mse(pattern_name, shape, zoom_factors, degree, method, freqs=None, square_sizes=None):
    preset = to_preset(method, degree)

    pattern = generate_pattern(pattern_name, shape, zoom_factors, freqs=freqs, square_sizes=square_sizes)
    pattern = pattern.astype(np.float64)
    resized_image = resize(pattern, zoom_factors=zoom_factors, method=preset)
    mse = calculate_mse_with_expected(pattern_name, shape, zoom_factors, resized_image, freqs, square_sizes)
    psnr = 10 * np.log10(1 / mse) if mse != 0 else float('inf')
    return mse, psnr

# Parametrized test cases for different dimensions
@pytest.mark.parametrize("pattern_name, shape, zoom_factors, degree, method, mse_threshold, psnr_threshold, freqs, square_sizes", [
    ("Gradient", (100,), (0.5,), 3, "least-squares", 1e-3, 60, None, None),
    ("Gradient", (100, 100), (0.75, 1.5), 1, "oblique", 1e-3, 60, None, None),
    ("Gradient", (50, 50, 50), (0.8, 2.8, 0.5), 3, "least-squares", 1e-3, 60, None, None),
    ("Sinusoidal", (100,), (0.5,), 1, "oblique", 1e-3, 35, [10], None),
    ("Sinusoidal", (100, 100), (0.314, 0.5), 3, "least-squares", 1e-3, 40, [10, 5], None),
    ("Sinusoidal", (50, 50, 50), (1.8, 0.8, 0.5), 3, "least-squares", 1e-3, 40, [10, 5, 3], None),
    ("Checkerboard", (100,), (0.5,), 3, "least-squares", 2e-2, 19, None, [10]),
    ("Checkerboard", (1000, 1000), (0.3, 1.6), 1, "oblique", 1e-2, 23, None, [100, 100]),
    ("Checkerboard", (50, 50, 50), (0.8, 1.2, 0.6), 1, "oblique", 1e-2, 21, None, [10, 10, 10]),
])
def test_resize_n_dimensional_pattern(pattern_name, shape, zoom_factors, degree, method, mse_threshold, psnr_threshold, freqs, square_sizes):
    mse, psnr = resize_pattern_and_calculate_mse(pattern_name, shape, zoom_factors, degree, method, freqs=freqs, square_sizes=square_sizes)
    assert mse < mse_threshold, f"{pattern_name} pattern MSE {mse} exceeds threshold {mse_threshold}"
    assert psnr > psnr_threshold, f"{pattern_name} pattern PSNR {psnr} dB below threshold {psnr_threshold}"