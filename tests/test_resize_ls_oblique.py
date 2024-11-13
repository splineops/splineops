import numpy as np
import pytest
from splineops.interpolate.resize import resize

# Mathematical functions for expected values in each pattern
def expected_gradient_value(x, width):
    return x / width

def expected_sinusoidal_value(x, y, width, height, freq_x=10, freq_y=5):
    normalized_x = x / width * 2 * np.pi * freq_x
    normalized_y = y / height * 2 * np.pi * freq_y
    return (np.sin(normalized_x) + np.sin(normalized_y)) * 0.25 + 0.5

def expected_checkerboard_value(x, y, square_size):
    row, col = int(y // square_size), int(x // square_size)
    return (row + col) % 2

# Calculate MSE with expected values
def calculate_mse_with_expected(pattern_name, width, height, zoom_factors, resized_image):
    target_height, target_width = resized_image.shape
    if pattern_name == "Gradient":
        expected_values = np.array([[expected_gradient_value(x / zoom_factors[1], width)
                                     for x in range(target_width)]
                                     for y in range(target_height)])
    elif pattern_name == "Sinusoidal":
        expected_values = np.array([[expected_sinusoidal_value(x / zoom_factors[1], y / zoom_factors[0], width, height, freq_x=10, freq_y=5)
                                     for x in range(target_width)]
                                     for y in range(target_height)])
    elif pattern_name == "Checkerboard":
        expected_values = np.array([[expected_checkerboard_value(x / zoom_factors[1], y / zoom_factors[0], 100)
                                     for x in range(target_width)]
                                     for y in range(target_height)])
    else:
        raise ValueError("Unknown pattern name")
    mse = np.mean((expected_values - resized_image) ** 2)
    return mse

# Test function for resizing patterns
def resize_pattern_and_calculate_mse(pattern_name, width, height, zoom_factors, degree, method):
    if pattern_name == "Gradient":
        pattern = np.linspace(0, 1, width).reshape(1, -1).repeat(height, axis=0)
    elif pattern_name == "Sinusoidal":
        x = np.linspace(0, 2 * np.pi * 10, width)
        y = np.linspace(0, 2 * np.pi * 5, height)
        X, Y = np.meshgrid(x, y)
        pattern = (np.sin(X) + np.sin(Y)) * 0.25 + 0.5
    elif pattern_name == "Checkerboard":
        rows = (np.arange(height) // 100) % 2
        cols = (np.arange(width) // 100) % 2
        pattern = np.bitwise_xor.outer(rows, cols).astype(float)
    else:
        raise ValueError("Unknown pattern name")
    
    # Resize pattern
    resized_image = resize(pattern, zoom_factors=zoom_factors, degree=degree, method=method)
    
    # Calculate MSE and PSNR
    mse = calculate_mse_with_expected(pattern_name, width, height, zoom_factors, resized_image)
    psnr = 10 * np.log10(1 / mse) if mse != 0 else float('inf')
    
    return mse, psnr

# Parametrized test cases
@pytest.mark.parametrize("pattern_name, zoom_factors, degree, method, mse_threshold, psnr_threshold", [
    ("Gradient", (0.75, 0.5), 3, "least-squares", 1e-6, 60),
    ("Gradient", (0.2, 1.5), 1, "least-squares", 1e-6, 60),
    ("Sinusoidal", (0.5, 0.5), 3, "least-squares", 1e-4, 40),
    ("Sinusoidal", (2.33, 0.32), 1, "least-squares", 1e-4, 40),
    ("Checkerboard", (0.3, 0.6), 3, "oblique", 1e-2, 23),
    ("Checkerboard", (0.8, 3.0), 1, "oblique", 1e-2, 23),
])
def test_resize_pattern(pattern_name, zoom_factors, degree, method, mse_threshold, psnr_threshold):
    width, height = 1000, 1000
    mse, psnr = resize_pattern_and_calculate_mse(pattern_name, width, height, zoom_factors, degree, method)
    
    # Assertions for MSE and PSNR thresholds
    assert mse < mse_threshold, f"{pattern_name} pattern MSE {mse} exceeds threshold {mse_threshold}"
    assert psnr > psnr_threshold, f"{pattern_name} pattern PSNR {psnr} dB below threshold {psnr_threshold}"
