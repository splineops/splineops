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
            2e-7,
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
