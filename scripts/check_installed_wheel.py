"""Exercise installed wheel numerics, native loading, resources, and CLI.

This must run in a clean environment, not against an editable source install.
It uses only the package's runtime dependencies and works outside the checkout.
"""

from __future__ import annotations

import importlib.resources
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tomllib

os.environ["SPLINEOPS_ACCEL"] = "always"
os.environ["LSRESIZE_BATCHED_AXIS"] = "auto"
os.environ["LSRESIZE_NUM_THREADS"] = "1"

import numpy as np
import scipy
import splineops
from splineops import _lsresize
from splineops.affine import AffinePlan
from splineops.resize import resize_degrees
from splineops.resize._pycore.engine import python_resize


def main():
    project = Path(__file__).resolve().parents[1]
    location = Path(splineops.__file__).resolve()
    if location.is_relative_to(project):
        raise RuntimeError(
            f"Expected an installed wheel, imported source at {location}"
        )
    expected_version = tomllib.loads(
        (project / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]["version"]
    if splineops.__version__ != expected_version:
        raise RuntimeError(f"Expected {expected_version}, got {splineops.__version__}")
    rng = np.random.default_rng(20261028)
    maximum_ls_error = 0.0
    maximum_affine_error = {"float32": 0.0, "float64": 0.0}
    for dtype in (np.float32, np.float64):
        data = rng.normal(size=(16, 17, 18)).astype(dtype)
        zoom = (0.37, 0.8, 0.61)
        degrees = dict(interp_degree=3, analy_degree=3, synthe_degree=3)
        projected = resize_degrees(data, zoom_factors=zoom, **degrees)
        expected = python_resize(data, zoom, **degrees)
        tolerance = 5e-7 if dtype == np.float32 else 3e-11
        np.testing.assert_allclose(projected, expected, rtol=tolerance, atol=tolerance)
        assert projected.dtype == data.dtype
        maximum_ls_error = max(
            maximum_ls_error, float(np.max(abs(projected - expected)))
        )
        for shape in ((9, 11), (6, 7, 8)):
            values = rng.normal(size=shape).astype(dtype)
            matrix = np.eye(len(shape)) * 0.9
            matrix[0, 1] = 0.15
            offset = np.linspace(-0.6, 0.7, len(shape))
            for mode in ("zero", "mirror", "periodic"):
                for degree in (1, 3, 5):
                    params = dict(dtype=dtype, mode=mode, degree=degree)
                    fast = AffinePlan(shape, matrix, offset, **params)
                    reference = AffinePlan(
                        shape, matrix, offset, backend="numpy", **params
                    )
                    assert fast.backend == "scipy"
                    expected = reference(values)
                    actual = fast(values)
                    tolerance = 2e-5 if dtype == np.float32 else 3e-12
                    np.testing.assert_allclose(
                        actual, expected, rtol=tolerance, atol=tolerance
                    )
                    np.testing.assert_array_equal(
                        fast.prefilter(values), reference.prefilter(values)
                    )
                    field = reference.prepare_coefficients(values)
                    np.testing.assert_array_equal(
                        fast.apply_coefficients(field), actual
                    )
                    key = np.dtype(dtype).name
                    maximum_affine_error[key] = max(
                        maximum_affine_error[key], float(np.max(abs(actual - expected)))
                    )
    resource = importlib.resources.files("splineops.demos.data").joinpath(
        "selma3d_vessels_summary.json"
    )
    assert resource.is_file()
    json.loads(resource.read_text(encoding="utf-8"))
    command = Path(sysconfig.get_path("scripts")) / (
        "splineops-selma3d-demo.exe" if os.name == "nt" else "splineops-selma3d-demo"
    )
    subprocess.run([str(command), "--help"], check=True, capture_output=True)
    print(
        json.dumps(
            {
                "splineops": splineops.__version__,
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "native_extension": _lsresize.__file__,
                "max_ls_difference": maximum_ls_error,
                "max_affine_difference": maximum_affine_error,
                "result": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
