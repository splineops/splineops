import pytest
import numpy as np
import numpy.typing as npt
from bssp.interpolate.tensorspline import TensorSpline
from bssp.bases.utils import asbasis, basis_map
from bssp.modes.utils import mode_map


@pytest.mark.parametrize("basis", basis_map.keys())
@pytest.mark.parametrize("mode", mode_map.keys())
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_interpolate_cardinal_spline(
    basis: str, mode: str, dtype: npt.DTypeLike
) -> None:

    # Create data with a single sample (Dirac) and proper padding
    basis = asbasis(basis)
    support = basis.support
    pad_left = (support - 1) // 2
    pad_right = support // 2
    if mode == "zero":
        # Need to have an "infinite" signal to have finite coefficients for
        # B-Splines with poles
        # Note: this is not super robust but does the job
        pad_right = 100 * pad_right
    pad_right = int(np.ceil(pad_right) // 2 * 2 + 1)  # next odd
    coords_1d = np.arange(-pad_right, pad_right + 1)
    coords = (coords_1d,)
    dirac_val = 1
    data = np.zeros(len(coords_1d), dtype=dtype)
    data[pad_right] = dirac_val

    # Create the tensor spline
    ts = TensorSpline(data=data, coords=coords, basis=basis, mode=mode)

    # Re-evaluate at points including the signal extension
    pad_right_eval = 2 * pad_right
    coords_eval_1d = np.arange(-pad_right_eval, pad_right_eval + 1)
    coords_eval = (coords_eval_1d,)
    values = ts(coords=coords_eval)

    # Expected values
    if mode == "zero":
        sig_ext_val = 0
    elif mode == "mirror":
        sig_ext_val = dirac_val
    else:
        raise NotImplementedError(f"Unsupported test mode '{mode}'")

    values_exact = np.array(
        [sig_ext_val]
        + (pad_right_eval - 1) * [0]
        + [dirac_val]
        + (pad_right_eval - 1) * [0]
        + [sig_ext_val]
    )

    # Tolerances
    # TODO(dperdios): need to account for more dtypes
    # TODO(dperdios): check `abs` and `rel` parameters for `pytest.approx` and
    #  which values should be used depending on dtype
    if dtype == "float64":
        atol = 1e-8
    elif dtype == "float32":
        atol = 2e-4
    else:
        raise NotImplementedError(f"Unsupported test dtype '{dtype}'")

    assert values == pytest.approx(values_exact, abs=atol)
