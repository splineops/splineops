# splineops/tests/test_02_03_resize_ls_oblique_cpp.py

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


def _time_and_run(mode: str, arr: np.ndarray, zoom: tuple[float, float], method: str, *, repeats: int = 2):
    """
    Set SPLINEOPS_ACCEL, reload module, warm up once, then time best-of-N.
    Returns (best_time_sec, output_array).
    """
    os.environ["SPLINEOPS_ACCEL"] = mode
    rz = _load_resize_module(force_reload=True)
    # warmup to load code paths/caches
    out = rz.resize(arr, zoom_factors=zoom, method=method)
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out_tmp = rz.resize(arr, zoom_factors=zoom, method=method)
        dt = time.perf_counter() - t0
        if dt < best:
            best, out = dt, out_tmp
    return best, out


@pytest.mark.skipif(not _has_cpp(), reason="Native extension not available: skipping C++ vs Python compare")
@pytest.mark.parametrize(
    "method_label,preset,shape,zoom,atol,speedup_min",
    [
        # Downsample: strong speedup expected
        ("Least-Squares (best AA)", "cubic-best_antialiasing", (512, 512), (0.5, 0.5), 2e-3, 3.0),
        ("Oblique (fast AA)",       "cubic-fast_antialiasing", (512, 512), (0.5, 0.5), 2e-3, 3.0),
        # Upsample: also fast in C++; keep a slightly relaxed assertion
        ("Least-Squares (best AA)", "cubic-best_antialiasing", (512, 512), (1.7, 1.7), 3e-3, 2.0),
        ("Oblique (fast AA)",       "cubic-fast_antialiasing", (512, 512), (1.7, 1.7), 5e-8, 2.0),
    ],
)
def test_cpp_vs_python_perf_and_equality(method_label, preset, shape, zoom, atol, speedup_min, monkeypatch):
    # Stabilize timings: single threads for OpenMP/BLAS stacks
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")

    rng = np.random.default_rng(0)
    arr = rng.random(shape, dtype=np.float64)

    # C++ path
    t_cpp, y_cpp = _time_and_run("always", arr, zoom, preset, repeats=2)
    # Python fallback
    t_py,  y_py  = _time_and_run("never",  arr, zoom, preset, repeats=2)

    # Numerical sanity: same result within tight tolerance
    max_abs = float(np.max(np.abs(y_cpp - y_py)))
    assert np.allclose(y_cpp, y_py, atol=atol, rtol=0.0), (
        f"{method_label} {shape} zoom={zoom}: max|Δ|={max_abs:.3e} exceeds atol={atol}"
    )

    # Performance sanity: C++ should be noticeably faster (allow some CI noise)
    # Require a minimum speedup; if CI is unusually noisy, this still gives generous headroom.
    speedup = (t_py / t_cpp) if t_cpp > 0 else np.inf
    assert speedup >= speedup_min, (
        f"{method_label} {shape} zoom={zoom}: speedup={speedup:.2f}× < {speedup_min}× "
        f"(C++ {t_cpp:.4f}s vs Py {t_py:.4f}s)"
    )
