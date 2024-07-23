import pytest
import numpy as np
import numpy.typing as npt
from splineops.interpolate.tensorspline import TensorSpline
from splineops.bases.utils import asbasis, basis_map
from splineops.modes.utils import mode_map


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
    real_dtype = np.array([1], dtype=dtype).real.dtype
    coords_1d = np.arange(-pad_right, pad_right + 1, dtype=real_dtype)
    coords = (coords_1d,)
    dirac_val = 1
    data = np.zeros(len(coords_1d), dtype=dtype)
    data[pad_right] = dirac_val

    # Create the tensor spline
    ts = TensorSpline(data=data, coordinates=coords, bases=basis, modes=mode)

    # Re-evaluate at points including the signal extension
    pad_right_eval = 2 * pad_right
    coords_eval_1d = np.arange(-pad_right_eval, pad_right_eval + 1)
    coords_eval = (coords_eval_1d,)
    values = ts(coordinates=coords_eval)

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


@pytest.mark.parametrize("dtype", ["complex128", "complex64", "float64", "float32"])
@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_interpolate_ndim_dtype(ndim: int, dtype: npt.DTypeLike) -> None:

    # Data type
    flag_complex_data = np.issubdtype(np.dtype(dtype).type, np.complexfloating)
    real_dtype = np.array([1], dtype=dtype).real.dtype

    # N-D (arbitrary choices)
    if ndim not in (1, 2, 3, 4):
        raise ValueError("Test only designed for 1, 2, 3, or 4 dimensions.")

    base_sample_number_seq = (6, 5, 4, 3)
    base_bounds_seq = (-3.1, +1), (2, 6.5), (-6.75, 0), (1.1, 2.7)
    base_bases = ("bspline3", "bspline5", "bspline4", "linear")
    base_modes = ("zero", "mirror", "mirror", "zero")

    # Batch-processing settings (here with a batch size of 3)
    base_batch_offsets_seq = (
        (0, -0.15, 0.3),
        (0, -1 / 3, 0.25),
        (0.5, 0.55, 0.65),
        (-0.75, -2 / 3, -0.6),
    )
    batch_offsets_seq = base_batch_offsets_seq[:ndim]

    # Data and coordinates
    sample_number_seq = base_sample_number_seq[:ndim]
    bounds_seq = base_bounds_seq[:ndim]
    coords_seq = tuple(
        np.linspace(b[0], b[1], n, dtype=real_dtype)
        for b, n in zip(bounds_seq, sample_number_seq)
    )
    prng = np.random.default_rng(seed=5250)
    data = prng.normal(size=tuple(c.size for c in coords_seq))
    if flag_complex_data:
        data = np.asarray(data + 1j * data, dtype=dtype)
    else:
        data = np.asarray(data, dtype=dtype)

    # Evaluation coordinates
    step_seq = tuple(
        (c[-1] - c[0]) / (n - 1) for c, n in zip(coords_seq, sample_number_seq)
    )
    pad_fct = 2.5
    over_sampling_fct = 3
    len_ext_seq = tuple((b[1] - b[0]) * pad_fct for b in bounds_seq)
    eval_coords_seq = tuple(
        np.linspace(
            start=(c[-1] + c[0]) / 2 - l / 2,
            stop=(c[-1] + c[0]) / 2 + l / 2,
            num=over_sampling_fct * n,
        )
        for c, l, n in zip(coords_seq, len_ext_seq, sample_number_seq)
    )

    # Tensor Spline
    bases = base_bases[:ndim]
    modes = base_modes[:ndim]
    tensor_spline = TensorSpline(
        data=data, coordinates=coords_seq, bases=bases, modes=modes
    )

    # Default evaluation (grid=True): tensor product of evaluation coordinates
    data_eval_tp = tensor_spline(coordinates=eval_coords_seq, grid=True)
    if flag_complex_data:
        # Add a check since both real and imag party are identical
        np.testing.assert_equal(data_eval_tp.real, data_eval_tp.imag)

    # Meshgrid evaluation
    eval_coords_mg = np.meshgrid(*eval_coords_seq, indexing="ij")
    data_eval_mg = tensor_spline(coordinates=eval_coords_mg, grid=False)
    np.testing.assert_equal(data_eval_tp, data_eval_mg)

    # Reshaped meshgrid evaluation
    eval_coords_mg_rs = np.reshape(eval_coords_mg, newshape=(ndim, -1))
    data_eval_mg_rs = tensor_spline(coordinates=eval_coords_mg_rs, grid=False)
    np.testing.assert_equal(
        data_eval_tp, np.reshape(data_eval_mg_rs, data_eval_mg.shape)
    )

    # Batch-processing: tensor product
    eval_coords_tp_batch = []
    for ii, (coords, batch_offsets) in enumerate(
        zip(eval_coords_seq, batch_offsets_seq)
    ):
        coords_batch = np.stack([coords + b for b in batch_offsets])
        # Add another batch dimension (2, 3, ...)
        coords_batch = np.stack([coords_batch, coords_batch])
        eval_coords_tp_batch.append(coords_batch)
    eval_coords_tp_batch = tuple(eval_coords_tp_batch)
    data_eval_tp_batch = tensor_spline(coordinates=eval_coords_tp_batch, grid=True)

    # Batch-processing: Meshgrid
    eval_coords_mg_batch = []
    for coords, batch_offsets in zip(eval_coords_mg, batch_offsets_seq):
        # Add another batch dimension (2, 3, ...)
        coords_mg_batch = np.stack([coords + b for b in batch_offsets])
        coords_mg_batch = np.stack([coords_mg_batch, coords_mg_batch])
        eval_coords_mg_batch.append(coords_mg_batch)
    eval_coords_mg_batch = tuple(eval_coords_mg_batch)
    data_eval_mg_batch = tensor_spline(coordinates=eval_coords_mg_batch, grid=False)
    np.testing.assert_equal(data_eval_tp_batch, data_eval_mg_batch)
