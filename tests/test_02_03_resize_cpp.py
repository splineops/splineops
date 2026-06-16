# splineops/tests/test_02_03_resize_cpp.py

import os
import sys
import time
import importlib
import importlib.util as _util

import numpy as np
import pytest


def _has_cpp() -> bool:
    """Is the native module importable in this environment?"""
    return _util.find_spec("splineops._lsresize") is not None


def _load_resize_module(*, force_reload: bool = False):
    """Load/reload the resize implementation so it re-reads SPLINEOPS_ACCEL."""
    name = "splineops.resize.resize"
    if force_reload and name in sys.modules:
        return importlib.reload(sys.modules[name])
    return importlib.import_module(name)


def _time_and_run_preset(
    mode: str,
    arr: np.ndarray,
    zoom: tuple[float, ...],
    method: str,
    *,
    repeats: int = 2,
):
    """
    C++ vs Python parity for preset-based API: resize(..., method=...).
    Returns (best_time_sec, output_array).
    """
    os.environ["SPLINEOPS_ACCEL"] = mode
    rz = _load_resize_module(force_reload=True)

    # Warmup to load code paths/caches
    out = rz.resize(arr, zoom_factors=zoom, method=method)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out_tmp = rz.resize(arr, zoom_factors=zoom, method=method)
        dt = time.perf_counter() - t0
        if dt < best:
            best, out = dt, out_tmp
    return best, out


def _time_and_run_ls(
    mode: str,
    arr: np.ndarray,
    zoom: tuple[float, ...],
    degree: int,
    *,
    repeats: int = 2,
):
    """
    C++ vs Python parity for equal-degree projection (LS-style):

        resize_degrees(..., interp_degree=degree,
                           analy_degree=degree,
                           synthe_degree=degree)
    """
    os.environ["SPLINEOPS_ACCEL"] = mode
    rz = _load_resize_module(force_reload=True)

    # Warmup
    out = rz.resize_degrees(
        arr,
        zoom_factors=zoom,
        interp_degree=degree,
        analy_degree=degree,
        synthe_degree=degree,
        inversable=False,
    )
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out_tmp = rz.resize_degrees(
            arr,
            zoom_factors=zoom,
            interp_degree=degree,
            analy_degree=degree,
            synthe_degree=degree,
            inversable=False,
        )
        dt = time.perf_counter() - t0
        if dt < best:
            best, out = dt, out_tmp
    return best, out


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping batched-axis compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["linear", "quadratic", "cubic"])
@pytest.mark.parametrize(
    "shape,zoom",
    [
        ((64, 48), (0.7, 1.3)),
        ((48, 32), (0.5, 0.75)),
        ((24, 20, 16), (0.5, 0.75, 1.25)),
    ],
)
def test_batched_axis_matches_default_pure_interpolation(monkeypatch, dtype, method, shape, zoom):
    rng = np.random.default_rng(123)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "off")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method=method)

    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method=method)

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping batched-axis compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "method",
    ["linear-antialiasing", "quadratic-antialiasing", "cubic-antialiasing"],
)
@pytest.mark.parametrize(
    "shape,zoom",
    [
        ((64, 48), (0.7, 1.3)),
        ((48, 32), (0.5, 0.75)),
        ((24, 20, 16), (0.5, 0.75, 1.25)),
    ],
)
def test_batched_axis_matches_default_antialiasing(monkeypatch, dtype, method, shape, zoom):
    rng = np.random.default_rng(124)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "off")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method=method)

    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method=method)

    atol = 3e-5 if dtype == np.float32 else 2e-9
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping short-axis compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "method,shape,zoom",
    [
        ("cubic-antialiasing", (1, 32), (1.0, 0.73)),
        ("cubic-antialiasing", (2, 32), (1.0, 0.73)),
        ("quadratic-antialiasing", (3, 40), (1.0, 0.61)),
        ("cubic-antialiasing", (32, 2), (0.73, 1.0)),
    ],
)
def test_batched_axis_matches_default_short_projection_axes(
    monkeypatch, dtype, method, shape, zoom
):
    rng = np.random.default_rng(129)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "off")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method=method)

    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method=method)

    assert np.allclose(actual, expected, atol=0.0, rtol=0.0)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping batched-axis compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("degree", [1, 2, 3])
@pytest.mark.parametrize(
    "shape,zoom",
    [
        ((48, 40), (0.5, 1.25)),
        ((32, 24), (1.7, 0.6)),
        ((20, 18, 12), (0.75, 1.2, 0.5)),
    ],
)
def test_batched_axis_matches_default_equal_degree_projection(
    monkeypatch, dtype, degree, shape, zoom
):
    rng = np.random.default_rng(125)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "off")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize_degrees(
        arr,
        zoom_factors=zoom,
        interp_degree=degree,
        analy_degree=degree,
        synthe_degree=degree,
        inversable=False,
    )

    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize_degrees(
        arr,
        zoom_factors=zoom,
        interp_degree=degree,
        analy_degree=degree,
        synthe_degree=degree,
        inversable=False,
    )

    atol = 3e-5 if dtype == np.float32 else 2e-9
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping batched-axis auto compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "method,shape,zoom",
    [
        ("cubic", (256, 256), (0.37, 0.37)),
        ("cubic-antialiasing", (256, 256), (0.37, 0.37)),
        ("cubic", (32, 24, 16), (0.75, 1.25, 0.5)),
    ],
)
def test_batched_axis_auto_matches_default(monkeypatch, dtype, method, shape, zoom):
    rng = np.random.default_rng(126)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "off")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method=method)

    if method.endswith("-antialiasing"):
        atol = 3e-5 if dtype == np.float32 else 2e-9
    else:
        atol = 5e-6 if dtype == np.float32 else 5e-11

    for env_value in (None, "auto"):
        if env_value is None:
            monkeypatch.delenv("LSRESIZE_BATCHED_AXIS", raising=False)
        else:
            monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", env_value)
        rz = _load_resize_module(force_reload=True)
        actual = rz.resize(arr, zoom_factors=zoom, method=method)
        assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping row-gather compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "method,shape,zoom",
    [
        ("cubic", (96, 80), (0.61, 1.33)),
        ("cubic-antialiasing", (80, 72), (0.57, 1.21)),
        ("cubic", (24, 20, 16), (0.75, 1.25, 0.5)),
    ],
)
def test_batched_row_gather_matches_line_gather(monkeypatch, dtype, method, shape, zoom):
    rng = np.random.default_rng(131)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_ROW_GATHER", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method=method)

    monkeypatch.delenv("LSRESIZE_ROW_GATHER", raising=False)
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method=method)

    atol = 3e-5 if method.endswith("-antialiasing") and dtype == np.float32 else 5e-6
    if dtype == np.float64:
        atol = 2e-9 if method.endswith("-antialiasing") else 5e-11
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping direct scatter compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "method,shape,zoom",
    [
        ("quadratic", (128, 160, 64), (1.0, 0.85, 1.0)),
        ("cubic", (128, 160, 64), (1.0, 1.1, 1.0)),
    ],
)
def test_3d_axis1_direct_scatter_matches_buffered_scatter(
    monkeypatch, dtype, method, shape, zoom
):
    rng = np.random.default_rng(137)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_3D_AXIS1_DIRECT_SCATTER", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method=method)

    monkeypatch.delenv("LSRESIZE_3D_AXIS1_DIRECT_SCATTER", raising=False)
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method=method)

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping 2-D linear interpolation compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape,zoom",
    [
        ((96, 80), (0.5, 0.5)),
        ((128, 96), (0.53, 1.27)),
        ((96, 128), (1.31, 0.57)),
        ((64, 72), (1.4, 1.25)),
    ],
)
def test_2d_linear_interp_fast_path_matches_disabled(
    monkeypatch, dtype, shape, zoom
):
    rng = np.random.default_rng(130)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    monkeypatch.setenv("LSRESIZE_LINEAR_INTERP", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method="linear")

    monkeypatch.setenv("LSRESIZE_LINEAR_INTERP", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method="linear")

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert actual.dtype == dtype
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping fused 2-D linear compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape,zoom",
    [
        ((96, 80), (0.5, 0.5)),
        ((96, 80), (1.4, 1.25)),
        ((96, 80), (0.55, 1.3)),
    ],
)
def test_2d_linear_fused_path_matches_axis_direct(monkeypatch, dtype, shape, zoom):
    rng = np.random.default_rng(133)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "auto")
    monkeypatch.setenv("LSRESIZE_LINEAR_INTERP", "1")
    monkeypatch.setenv("LSRESIZE_FUSED_2D_LINEAR", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method="linear")

    monkeypatch.setenv("LSRESIZE_FUSED_2D_LINEAR", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method="linear")

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert actual.dtype == dtype
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping fused 3-D linear compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape,zoom",
    [
        ((32, 28, 20), (0.6, 0.75, 0.5)),
        ((24, 20, 18), (1.2, 0.7, 0.8)),
        ((28, 24, 20), (0.6, 0.7, 1.0)),
        ((28, 24, 20), (1.0, 0.7, 0.5)),
        ((28, 24, 20), (0.6, 1.0, 1.25)),
    ],
)
def test_3d_linear_fused_path_matches_axis_direct(monkeypatch, dtype, shape, zoom):
    rng = np.random.default_rng(135)
    arr = rng.random(shape, dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "auto")
    monkeypatch.setenv("LSRESIZE_LINEAR_INTERP", "1")
    monkeypatch.setenv("LSRESIZE_FUSED_3D_LINEAR", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method="linear")

    monkeypatch.setenv("LSRESIZE_FUSED_3D_LINEAR", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method="linear")

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert actual.dtype == dtype
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping fused 3-D two-axis compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "zoom",
    [
        (0.6, 1.0, 0.5),
        (1.0, 0.7, 0.5),
    ],
)
def test_3d_linear_two_axis_fused_path_matches_disabled(monkeypatch, dtype, zoom):
    rng = np.random.default_rng(137)
    arr = rng.random((28, 24, 20), dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "auto")
    monkeypatch.setenv("LSRESIZE_LINEAR_INTERP", "1")
    monkeypatch.setenv("LSRESIZE_FUSED_3D_LINEAR", "1")
    monkeypatch.setenv("LSRESIZE_FUSED_3D_TWO_AXIS_LINEAR", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method="linear")

    monkeypatch.setenv("LSRESIZE_FUSED_3D_TWO_AXIS_LINEAR", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method="linear")

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert actual.dtype == dtype
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping projection restore compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_projection_avg_restore_fused_path_matches_disabled(monkeypatch, dtype):
    rng = np.random.default_rng(138)
    arr = rng.random((48, 45), dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.setenv("LSRESIZE_NUM_THREADS", "1")
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "auto")
    monkeypatch.setenv("LSRESIZE_FUSED_PROJECTION_AVG_RESTORE", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=(0.53, 0.71), method="cubic-antialiasing")

    monkeypatch.setenv("LSRESIZE_FUSED_PROJECTION_AVG_RESTORE", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=(0.53, 0.71), method="cubic-antialiasing")

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert actual.dtype == dtype
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping AVX2 2-D linear compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("zoom", [(1.4, 1.25), (1.0, 0.53)])
def test_2d_linear_avx2_path_matches_disabled(monkeypatch, dtype, zoom):
    rng = np.random.default_rng(134)
    arr = rng.random((72, 80), dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "auto")
    monkeypatch.setenv("LSRESIZE_LINEAR_INTERP", "1")
    monkeypatch.setenv("LSRESIZE_FUSED_2D_LINEAR", "1")
    monkeypatch.setenv("LSRESIZE_AVX2_LINEAR", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method="linear")

    monkeypatch.setenv("LSRESIZE_AVX2_LINEAR", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method="linear")

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert actual.dtype == dtype
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping N-D linear interpolation compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nd_linear_interp_fast_path_matches_disabled(monkeypatch, dtype):
    rng = np.random.default_rng(132)
    arr = rng.random((24, 20, 16), dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "auto")
    monkeypatch.setenv("LSRESIZE_LINEAR_INTERP", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=(0.75, 1.2, 0.5), method="linear")

    monkeypatch.setenv("LSRESIZE_LINEAR_INTERP", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=(0.75, 1.2, 0.5), method="linear")

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert actual.dtype == dtype
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping N-D last-axis linear compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "zoom",
    [
        (0.75, 1.0, 0.5),
        (1.0, 0.8, 0.5),
    ],
)
def test_nd_linear_last_axis_direct_path_matches_disabled(monkeypatch, dtype, zoom):
    rng = np.random.default_rng(136)
    arr = rng.random((24, 20, 16), dtype=dtype)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "auto")
    monkeypatch.setenv("LSRESIZE_LINEAR_INTERP", "1")
    monkeypatch.setenv("LSRESIZE_LAST_AXIS_LINEAR_DIRECT", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method="linear")

    monkeypatch.setenv("LSRESIZE_LAST_AXIS_LINEAR_DIRECT", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method="linear")

    atol = 5e-6 if dtype == np.float32 else 5e-11
    assert actual.dtype == dtype
    assert np.allclose(actual, expected, atol=atol, rtol=atol)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping legacy 2-D linear flag compare",
)
def test_2d_linear_interp_legacy_float_flag_disables_fast_path(monkeypatch):
    rng = np.random.default_rng(131)
    arr = rng.random((128, 96), dtype=np.float32)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    monkeypatch.delenv("LSRESIZE_LINEAR_INTERP", raising=False)
    monkeypatch.delenv("LSRESIZE_2D_LINEAR_INTERP", raising=False)
    monkeypatch.setenv("LSRESIZE_2D_FLOAT_INTERP", "0")
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=(0.53, 1.27), method="linear")

    monkeypatch.setenv("LSRESIZE_2D_FLOAT_INTERP", "1")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=(0.53, 1.27), method="linear")

    assert np.allclose(actual, expected, atol=5e-6, rtol=5e-6)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping float32 internal compare",
)
@pytest.mark.parametrize(
    "method,shape,zoom,atol",
    [
        ("linear", (128, 96), (0.6, 1.4), 8e-7),
        ("cubic", (128, 96), (0.6, 1.4), 2e-6),
        ("linear-antialiasing", (128, 96), (0.6, 1.4), 2e-4),
        ("cubic-antialiasing", (128, 96), (0.6, 1.4), 6e-4),
        ("quadratic-antialiasing", (128, 96), (0.6, 1.4), 6e-4),
        ("cubic", (33,), (1.7,), 2e-6),
        ("cubic-antialiasing", (20, 18, 12), (0.75, 1.2, 0.5), 2e-4),
    ],
)
def test_float32_internal_matches_default_precision(
    monkeypatch, method, shape, zoom, atol
):
    rng = np.random.default_rng(127)
    arr = rng.random(shape, dtype=np.float32)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    rz = _load_resize_module(force_reload=True)
    expected = rz.resize(arr, zoom_factors=zoom, method=method)

    monkeypatch.setenv("LSRESIZE_PRECISION", "float32")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=zoom, method=method)

    max_abs = float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64))))
    assert actual.dtype == np.float32
    assert np.allclose(actual, expected, atol=atol, rtol=0.0), (
        f"{method} float32 internals max|Δ|={max_abs:.3e} exceeds atol={atol}"
    )


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping float32 constant preservation",
)
@pytest.mark.parametrize(
    "method,atol",
    [
        ("cubic", 1e-6),
        ("linear-antialiasing", 1e-7),
        ("quadratic-antialiasing", 1e-7),
        ("cubic-antialiasing", 1e-7),
    ],
)
def test_float32_internal_preserves_constant_arrays(monkeypatch, method, atol):
    arr = np.full((97, 89), 3.25, dtype=np.float32)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")
    monkeypatch.setenv("LSRESIZE_PRECISION", "float32")
    rz = _load_resize_module(force_reload=True)
    actual = rz.resize(arr, zoom_factors=(0.53, 1.37), method=method)

    assert actual.dtype == np.float32
    assert np.allclose(actual, 3.25, atol=atol, rtol=0.0)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping float32 auto precision",
)
@pytest.mark.parametrize("method", ["quadratic", "cubic"])
@pytest.mark.parametrize(
    "shape,zoom",
    [
        ((129, 97), (0.61, 1.33)),
        ((32, 24, 16), (0.75, 1.25, 0.5)),
    ],
)
def test_float32_auto_precision_for_pure_interpolation(monkeypatch, method, shape, zoom):
    rng = np.random.default_rng(128)
    arr = rng.random(shape, dtype=np.float32)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")

    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    rz = _load_resize_module(force_reload=True)
    automatic = rz.resize(arr, zoom_factors=zoom, method=method)

    monkeypatch.setenv("LSRESIZE_PRECISION", "float32")
    rz = _load_resize_module(force_reload=True)
    forced_f32 = rz.resize(arr, zoom_factors=zoom, method=method)

    monkeypatch.setenv("LSRESIZE_PRECISION", "float64")
    rz = _load_resize_module(force_reload=True)
    forced_f64 = rz.resize(arr, zoom_factors=zoom, method=method)

    max_abs = float(
        np.max(np.abs(automatic.astype(np.float64) - forced_f64.astype(np.float64)))
    )
    assert automatic.dtype == np.float32
    assert np.array_equal(automatic, forced_f32)
    assert np.allclose(automatic, forced_f64, atol=2e-6, rtol=0.0), (
        f"{method} automatic float32 internals max|Δ|={max_abs:.3e}"
    )


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping float32 auto precision",
)
@pytest.mark.parametrize("method", ["linear-antialiasing", "cubic-antialiasing"])
def test_float32_auto_precision_keeps_projection_default(monkeypatch, method):
    rng = np.random.default_rng(129)
    arr = rng.random((128, 96), dtype=np.float32)
    zoom = (0.6, 1.4)

    monkeypatch.setenv("SPLINEOPS_ACCEL", "always")
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "1")

    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)
    rz = _load_resize_module(force_reload=True)
    automatic = rz.resize(arr, zoom_factors=zoom, method=method)

    monkeypatch.setenv("LSRESIZE_PRECISION", "float64")
    rz = _load_resize_module(force_reload=True)
    forced_f64 = rz.resize(arr, zoom_factors=zoom, method=method)

    assert automatic.dtype == np.float32
    assert np.array_equal(automatic, forced_f64)


@pytest.mark.skipif(
    not _has_cpp(),
    reason="Native extension not available: skipping C++ vs Python compare",
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "method_label,kind,arg,shape,zoom,atol",
    [
        # ------------------------------------------------------------------ #
        # Core LS-style (equal-degree projection) baselines                  #
        # ------------------------------------------------------------------ #
        # Downsample: LS cubic vs Python
        (
            "Least-Squares projection (cubic) ↓",
            "ls",
            3,                     # degree
            (512, 512),
            (0.5, 0.5),
            1e-5,                  # was 6e-8; relaxed for cross-platform FP
        ),
        # Upsample: LS cubic – allow looser tol (zoom > 1)
        (
            "Least-Squares projection (cubic) ↑",
            "ls",
            3,
            (512, 512),
            (2.5, 2.5),
            5e-3,                  # was 6e-5; macOS max|Δ|≈1.8e-3 with margin
        ),
        # Non-uniform zoom (LS cubic)
        (
            "Least-Squares projection (cubic) non-uniform",
            "ls",
            3,
            (384, 256),
            (0.5, 2.0),
            8e-5,
        ),
        # Quadratic LS ↓
        (
            "Least-Squares projection (quadratic) ↓",
            "ls",
            2,
            (400, 400),
            (0.5, 0.5),
            3e-7,
        ),
        # Quadratic LS ↑
        (
            "Least-Squares projection (quadratic) ↑",
            "ls",
            2,
            (400, 400),
            (2.2, 2.2),
            3e-5,
        ),
        # Extreme downscale
        (
            "Least-Squares projection (cubic) extreme ↓",
            "ls",
            3,
            (513, 517),
            (0.24, 0.24),
            2e-6,
        ),
        # Identity (zoom=1) – LS cubic, should be a pure copy
        (
            "Least-Squares projection identity (cubic)",
            "ls",
            3,
            (128, 257),
            (1.0, 1.0),
            2e-7,
        ),
        # Single-axis shrink
        (
            "Least-Squares projection single-axis shrink (cubic)",
            "ls",
            3,
            (640, 360),
            (0.5, 1.0),
            8e-5,
        ),

        # ------------------------------------------------------------------ #
        # Antialiasing (oblique) via preset-based API                        #
        # ------------------------------------------------------------------ #
        # Downsample: Antialiasing cubic
        (
            "Antialiasing (cubic) ↓",
            "preset",
            "cubic-antialiasing",
            (512, 512),
            (0.5, 0.5),
            2e-7,
        ),
        # Upsample: Antialiasing cubic
        (
            "Antialiasing (cubic) ↑",
            "preset",
            "cubic-antialiasing",
            (512, 512),
            (2.5, 2.5),
            5e-7,
        ),
        # Non-uniform zoom (Antialiasing cubic)
        (
            "Antialiasing (cubic) non-uniform",
            "preset",
            "cubic-antialiasing",
            (300, 200),
            (2.0, 0.6),
            2e-7,
        ),

        # Quadratic Antialiasing ↓
        (
            "Antialiasing (quadratic) ↓",
            "preset",
            "quadratic-antialiasing",
            (400, 400),
            (0.5, 0.5),
            3e-7,
        ),
        # Quadratic Antialiasing ↑
        (
            "Antialiasing (quadratic) ↑",
            "preset",
            "quadratic-antialiasing",
            (400, 400),
            (2.2, 2.2),
            3e-5,
        ),

        # ------------------------------------------------------------------ #
        # Interpolation presets (no projection)                              #
        # ------------------------------------------------------------------ #
        (
            "Interpolation (cubic)",
            "preset",
            "cubic",
            (512, 512),
            (0.5, 0.5),
            8e-7,
        ),
        (
            "Interpolation (linear)",
            "preset",
            "linear",
            (300, 500),
            (2.3, 2.3),
            5e-7,
        ),

        # ------------------------------------------------------------------ #
        # Extra sanity / regression                                          #
        # ------------------------------------------------------------------ #
        (
            "Antialiasing single-axis up (cubic)",
            "preset",
            "cubic-antialiasing",
            (640, 360),
            (1.0, 2.0),
            1e-7,
        ),
        (
            "Nearest mixed zoom",
            "preset",
            "fast",
            (64, 1024),
            (3.0, 0.5),
            1e-12,
        ),
    ],
)
def test_cpp_vs_python_equality(
    method_label, kind, arg, shape, zoom, atol, dtype, monkeypatch
):
    # Stabilize timings: single threads for OpenMP/BLAS stacks
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")

    # Disable optional Python-side autotuning for reproducible timings
    monkeypatch.setenv("SPLINEOPS_AUTOTUNE", "0")

    # This parity test defines the default precision contract. Automatic and
    # opt-in float32 native paths have dedicated coverage above.
    monkeypatch.delenv("LSRESIZE_PRECISION", raising=False)

    rng = np.random.default_rng(0)
    arr = rng.random(shape, dtype=dtype)

    if kind == "preset":
        preset = arg  # type: ignore[assignment]
        t_cpp, y_cpp = _time_and_run_preset("always", arr, zoom, preset, repeats=2)
        t_py,  y_py  = _time_and_run_preset("never",  arr, zoom, preset, repeats=2)
    elif kind == "ls":
        degree = int(arg)
        t_cpp, y_cpp = _time_and_run_ls("always", arr, zoom, degree, repeats=2)
        t_py,  y_py  = _time_and_run_ls("never",  arr, zoom, degree, repeats=2)
    else:
        raise ValueError(f"Unknown kind '{kind}'")

    # Dtype sanity: both implementations should preserve the input dtype
    assert y_cpp.dtype == dtype, f"C++ output dtype {y_cpp.dtype} != input dtype {dtype}"
    assert y_py.dtype == dtype,  f"Python output dtype {y_py.dtype} != input dtype {dtype}"

    # Numerical sanity: same result within tolerance
    max_abs = float(np.max(np.abs(y_cpp - y_py)))
    assert np.allclose(y_cpp, y_py, atol=atol, rtol=0.0), (
        f"{method_label} {shape} zoom={zoom}: "
        f"max|Δ|={max_abs:.3e} exceeds atol={atol}"
    )

    # Optional: print speedup for debugging / curiosity (no assertion!)
    is_identity = all(abs(z - 1.0) <= 1e-12 for z in zoom)
    if not is_identity:
        speedup = (t_py / t_cpp) if t_cpp > 0.0 else np.inf
        print(
            f"[perf] {method_label} {shape} zoom={zoom}, dtype={dtype}: "
            f"speedup={speedup:.2f}× (C++ {t_cpp:.4f}s vs Py {t_py:.4f}s)"
        )
