# splineops/src/splineops/utils/interop.py

import numpy.typing as npt

def is_cupy_type(x: npt.NDArray) -> bool:
    # Note: it avoids explicit reference to CuPy
    return "cupy" in str(type(x))


def is_ndarray(x) -> bool:
    # TODO(dperdios): this might not account for all cases
    #  currently works well with NumPy and CuPy (main targets)
    #  Note: might best handled via the `array_api` module
    return "ndarray" in str(type(x))
