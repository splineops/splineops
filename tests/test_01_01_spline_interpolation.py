# splineops/tests/test_01_01_spline_interpolation.py

import pytest
import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

from splineops.spline_interpolation.tensor_spline import TensorSpline
from splineops.spline_interpolation.bases.utils import asbasis, basis_map
from splineops.spline_interpolation.modes.utils import asmode, mode_map
from splineops.spline_interpolation._prefilter import (
    prefilter_interpolation_coefficients,
)


def test_tensorspline_stable_public_import() -> None:
    from splineops import TensorSpline as PublicTensorSpline
    from splineops.spline_interpolation import TensorSpline as ModuleTensorSpline

    assert PublicTensorSpline is TensorSpline
    assert ModuleTensorSpline is TensorSpline


@pytest.mark.parametrize("length", [1, 2, 37])
@pytest.mark.parametrize("axis", [0, 1])
def test_internal_prefilter_handles_short_long_and_strided_axes(length, axis):
    rng = np.random.default_rng(20260715 + length + axis)
    full = rng.standard_normal((length, 14) if axis == 0 else (9, 2 * length))
    data = full if axis == 0 else full[:, ::2]
    basis = asbasis("bspline3")
    mode = asmode("mirror")
    moved = np.ascontiguousarray(np.moveaxis(data, axis, -1))
    expected = np.moveaxis(mode.compute_coefficients(moved, basis), -1, axis)

    result = prefilter_interpolation_coefficients(
        data,
        bases=(basis,),
        modes=(mode,),
        axes=(axis,),
        dtype=np.float64,
        backend="numpy",
    )

    assert result.flags.c_contiguous
    assert not np.shares_memory(result, data)
    np.testing.assert_equal(result, expected)


def test_internal_prefilter_backend_contract_is_explicit():
    with pytest.raises(TypeError, match="backend"):
        prefilter_interpolation_coefficients(
            np.arange(6.0),
            bases=(asbasis("bspline3"),),
            modes=(asmode("mirror"),),
            backend="cupy",
        )


def test_tensorspline_rejects_nonuniform_construction_grid() -> None:
    data = np.arange(4.0)
    coordinates = np.array([0.0, 1.0, 2.1, 3.0])

    with pytest.raises(ValueError, match="uniform construction grid"):
        TensorSpline(data, coordinates, bases="linear", modes="mirror")


def test_tensorspline_rejects_nonfloating_construction_grid() -> None:
    with pytest.raises(TypeError, match="floating dtype"):
        TensorSpline(
            np.arange(4.0),
            np.arange(4),
            bases="linear",
            modes="mirror",
        )


def test_tensorspline_point_queries_accept_stacked_coordinates() -> None:
    data = np.arange(12.0).reshape(4, 3)
    coordinates = (np.arange(4.0), np.arange(3.0))
    spline = TensorSpline(data, coordinates, bases="linear", modes="mirror")
    points = np.array([[2.0, 0.5], [1.0, 0.5]])

    stacked = spline(points, grid=False)
    separate = spline(tuple(points), grid=False)

    np.testing.assert_equal(stacked, separate)


def test_tensorspline_grid_queries_need_matching_batch_shapes() -> None:
    data = np.arange(12.0).reshape(4, 3)
    coordinates = (np.arange(4.0), np.arange(3.0))
    spline = TensorSpline(data, coordinates, bases="linear", modes="mirror")

    with pytest.raises(ValueError, match="leading batch dimensions"):
        spline((np.zeros((2, 4)), np.zeros((3, 3))), grid=True)


@pytest.mark.parametrize("basis", ["bspline3", "bspline5", "bspline9", "omoms5"])
@pytest.mark.parametrize("length", [1, 2, 3])
def test_periodic_short_signal_reproduces_samples(basis: str, length: int) -> None:
    coordinates = np.arange(length, dtype=np.float64)
    data = np.arange(1, length + 1, dtype=np.float64)
    spline = TensorSpline(data, coordinates, bases=basis, modes="periodic")

    np.testing.assert_allclose(spline(coordinates), data, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("basis", basis_map.keys())
def test_mirror_singleton_is_constant(basis: str) -> None:
    data = np.array([2.0])
    coordinates = np.array([0.0])
    spline = TensorSpline(data, coordinates, bases=basis, modes="mirror")

    np.testing.assert_allclose(
        spline(np.array([-100.0, 0.0, 100.0])),
        2.0,
        rtol=0.0,
        atol=2e-12,
    )


@pytest.mark.parametrize("grid", [False, True])
def test_tiled_evaluation_is_tile_size_invariant(grid: bool) -> None:
    rng = np.random.default_rng(7)
    data = rng.normal(size=(7, 6))
    construction = (np.arange(7.0), np.arange(6.0))
    spline = TensorSpline(data, construction, bases="bspline3", modes="mirror")
    if grid:
        query = (np.linspace(-1.0, 7.0, 13), np.linspace(-2.0, 6.0, 11))
    else:
        query = (rng.uniform(-1.0, 7.0, 143), rng.uniform(-2.0, 6.0, 143))

    spline._EVALUATION_TILE_SIZE = 10_000
    untiled = spline(query, grid=grid)
    spline._EVALUATION_TILE_SIZE = 7
    tiled = spline(query, grid=grid)

    np.testing.assert_equal(tiled, untiled)


def test_batched_grid_matches_independent_grids_when_tiled() -> None:
    data = np.arange(30.0).reshape(6, 5)
    spline = TensorSpline(
        data,
        (np.arange(6.0), np.arange(5.0)),
        bases="linear",
        modes="mirror",
    )
    x = np.stack([np.linspace(0.0, 5.0, 9), np.linspace(0.25, 4.75, 9)])
    y = np.stack([np.linspace(0.0, 4.0, 7), np.linspace(0.5, 3.5, 7)])
    spline._EVALUATION_TILE_SIZE = 5

    batched = spline((x, y), grid=True)

    assert batched.shape == (2, 9, 7)
    np.testing.assert_equal(batched[0], spline((x[0], y[0]), grid=True))
    np.testing.assert_equal(batched[1], spline((x[1], y[1]), grid=True))


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
@pytest.mark.parametrize("degree", range(6))
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mirror_bspline_matches_scipy_at_equivalent_points(
    ndim: int, degree: int, dtype: npt.DTypeLike
) -> None:
    """Compare only the contract shared with scipy.ndimage.map_coordinates."""
    rng = np.random.default_rng(20260715 + 10 * ndim + degree)
    shape = (13, 12, 11, 10)[:ndim]
    data = rng.standard_normal(shape).astype(dtype)
    construction = tuple(np.arange(length, dtype=dtype) for length in shape)
    query = tuple(
        rng.uniform(1.0, length - 2.0, size=37).astype(dtype) for length in shape
    )
    spline = TensorSpline(
        data,
        construction,
        bases=f"bspline{degree}",
        modes="mirror",
    )

    actual = spline(query, grid=False)
    expected = map_coordinates(
        data,
        np.stack(query),
        order=degree,
        mode="mirror",
        prefilter=True,
    )

    tolerance = 8e-5 if np.dtype(dtype) == np.float32 else 2e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_complex_3d_mirror_bspline_matches_scipy(dtype: npt.DTypeLike) -> None:
    rng = np.random.default_rng(20260715)
    shape = (9, 8, 7)
    data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dtype)
    real_dtype = data.real.dtype
    construction = tuple(np.arange(length, dtype=real_dtype) for length in shape)
    query = tuple(
        rng.uniform(1.0, length - 2.0, size=31).astype(real_dtype) for length in shape
    )
    spline = TensorSpline(data, construction, bases="bspline3", modes="mirror")

    actual = spline(query, grid=False)
    expected = map_coordinates(
        data,
        np.stack(query),
        order=3,
        mode="mirror",
        prefilter=True,
    )

    tolerance = 2e-4 if np.dtype(dtype) == np.complex64 else 2e-12
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def test_linear_4d_spline_reproduces_an_affine_field() -> None:
    shape = (5, 6, 4, 7)
    construction = tuple(np.arange(length, dtype=np.float64) for length in shape)
    mesh = np.meshgrid(*construction, indexing="ij")
    data = 1.5 + 2.0 * mesh[0] - 0.5 * mesh[1] + 3.0 * mesh[2] - mesh[3]
    spline = TensorSpline(data, construction, bases="linear", modes="mirror")
    rng = np.random.default_rng(20260715)
    query = tuple(rng.uniform(0.0, length - 1.0, size=101) for length in shape)

    actual = spline(query, grid=False)
    expected = 1.5 + 2.0 * query[0] - 0.5 * query[1] + 3.0 * query[2] - query[3]

    np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=2e-14)


def test_high_dimensional_singleton_axes_reproduce_samples() -> None:
    data = np.arange(6, dtype=np.float64).reshape(1, 2, 1, 3)
    construction = tuple(np.arange(length, dtype=np.float64) for length in data.shape)
    spline = TensorSpline(data, construction, bases="bspline3", modes="mirror")

    result = spline(construction, grid=True)

    np.testing.assert_allclose(result, data, rtol=0.0, atol=2e-12)


@pytest.mark.parametrize("grid", [False, True])
def test_query_plan_matches_ordinary_evaluation(grid: bool) -> None:
    rng = np.random.default_rng(20260715)
    data = rng.standard_normal((9, 8))
    construction = (np.arange(9.0), np.arange(8.0))
    spline = TensorSpline(data, construction, bases="bspline3", modes="mirror")
    if grid:
        query = (np.linspace(-1.0, 9.0, 17), np.linspace(-2.0, 8.0, 13))
    else:
        query = (rng.uniform(-1.0, 9.0, 221), rng.uniform(-2.0, 8.0, 221))

    plan = spline.query_plan(query, grid=grid)

    assert plan.output_shape == spline(query, grid=grid).shape
    assert plan.retained_bytes > 0
    np.testing.assert_allclose(plan(), spline(query, grid=grid), rtol=2e-15, atol=2e-15)
    np.testing.assert_equal(plan.apply(), plan())


def test_query_plan_is_independent_of_coordinate_mutation() -> None:
    data = np.arange(30.0).reshape(6, 5)
    spline = TensorSpline(
        data,
        (np.arange(6.0), np.arange(5.0)),
        bases="linear",
        modes="mirror",
    )
    query = [np.linspace(0.0, 5.0, 11), np.linspace(0.0, 4.0, 11)]
    expected = spline(tuple(query), grid=False)
    plan = spline.query_plan(tuple(query), grid=False)

    query[0][:] = -1000.0
    query[1][:] = 1000.0

    np.testing.assert_equal(plan(), expected)


def test_geometry_plan_reuses_coordinates_across_compatible_splines() -> None:
    construction = (np.arange(7.0), np.arange(6.0))
    query = (np.linspace(-1.0, 7.0, 31), np.linspace(-2.0, 6.0, 31))
    first = TensorSpline(
        np.arange(42.0).reshape(7, 6),
        construction,
        bases="bspline3",
        modes="mirror",
    )
    second = TensorSpline(
        np.arange(42.0, 84.0).reshape(7, 6),
        construction,
        bases="bspline3",
        modes="mirror",
    )

    plan = first.query_plan(query, grid=False)

    np.testing.assert_allclose(
        plan.apply(second), second(query, grid=False), rtol=2e-15, atol=2e-15
    )
    # Historical no-argument behavior remains available.
    np.testing.assert_allclose(
        plan.apply(), first(query, grid=False), rtol=2e-15, atol=2e-15
    )


def test_template_refits_data_and_accepts_precomputed_coefficients() -> None:
    rng = np.random.default_rng(20260716)
    construction = (np.arange(8.0), np.arange(7.0))
    original = rng.standard_normal((8, 7))
    replacement = rng.standard_normal((8, 7))
    template = TensorSpline(original, construction, bases="bspline3", modes="mirror")
    expected = TensorSpline(replacement, construction, bases="bspline3", modes="mirror")
    query = (rng.uniform(-1.0, 8.0, 101), rng.uniform(-1.0, 7.0, 101))

    coefficients = template.coefficients_from_data(replacement)
    coefficient_out = np.empty_like(replacement)
    assert (
        template.coefficients_from_data(replacement, out=coefficient_out)
        is coefficient_out
    )
    refitted = template.with_data(replacement)
    prepared = template.with_coefficients(coefficients)

    np.testing.assert_equal(coefficients, expected.coefficients)
    np.testing.assert_equal(coefficient_out, coefficients)
    np.testing.assert_allclose(
        refitted(query, grid=False), expected(query, grid=False), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        prepared(query, grid=False), expected(query, grid=False), rtol=0.0, atol=0.0
    )
    # The template remains bound to its original samples.
    np.testing.assert_allclose(
        template(query, grid=False),
        TensorSpline(original, construction, "bspline3", "mirror")(query, grid=False),
        rtol=0.0,
        atol=0.0,
    )
    with pytest.raises(ValueError, match="shape"):
        template.with_data(replacement[:, :-1])
    with pytest.raises(TypeError, match="dtype"):
        template.with_coefficients(coefficients.astype(np.float32))


def test_precomputed_coefficients_copy_by_default_and_plan_introspection() -> None:
    data = np.arange(30.0).reshape(6, 5)
    spline = TensorSpline(
        data,
        (np.arange(6.0), np.arange(5.0)),
        bases="linear",
        modes="mirror",
    )
    coefficients = spline.coefficients
    prepared = spline.with_coefficients(coefficients)
    plan = spline.query_plan((np.arange(6.0), np.arange(5.0)), grid=True)

    coefficients[:] = -1.0

    np.testing.assert_equal(prepared.coefficients, data)
    assert plan.grid
    assert plan.attached
    assert plan.is_compatible(prepared)
    assert plan.incompatibility_reason(prepared) is None
    assert not plan.is_compatible(object())
    assert plan.incompatibility_reason(object()) == (
        "The supplied object is not a TensorSpline."
    )
    plan.detach()
    assert not plan.attached


def test_geometry_plan_rejects_incompatible_spline() -> None:
    first = TensorSpline(
        np.ones((5, 4)),
        (np.arange(5.0), np.arange(4.0)),
        bases="linear",
        modes="mirror",
    )
    incompatible = TensorSpline(
        np.ones((5, 4)),
        (np.arange(5.0), np.arange(4.0)),
        bases="bspline3",
        modes="mirror",
    )
    plan = first.query_plan((np.arange(5.0), np.arange(4.0)), grid=True)

    with pytest.raises(ValueError, match="incompatible"):
        plan.apply(incompatible)


@pytest.mark.parametrize("grid", [False, True])
def test_tensorspline_output_buffer(grid: bool) -> None:
    data = np.arange(30.0).reshape(6, 5)
    spline = TensorSpline(
        data,
        (np.arange(6.0), np.arange(5.0)),
        bases="linear",
        modes="mirror",
    )
    query = (
        (np.linspace(0.0, 5.0, 9), np.linspace(0.0, 4.0, 7))
        if grid
        else (np.linspace(0.0, 5.0, 13), np.linspace(0.0, 4.0, 13))
    )
    expected = spline(query, grid=grid)
    output = np.empty_like(expected)

    returned = spline(query, grid=grid, out=output)

    assert returned is output
    np.testing.assert_equal(output, expected)


def test_geometry_plan_output_buffer() -> None:
    spline = TensorSpline(
        np.arange(20.0).reshape(5, 4),
        (np.arange(5.0), np.arange(4.0)),
        bases="linear",
        modes="mirror",
    )
    query = (np.linspace(0.0, 4.0, 9), np.linspace(0.0, 3.0, 7))
    plan = spline.query_plan(query, grid=True)
    output = np.empty(plan.output_shape, dtype=spline.coefficients.dtype)

    returned = plan.apply(out=output)

    assert returned is output
    np.testing.assert_equal(output, spline(query, grid=True))


def test_query_plan_enforces_retained_memory_limit() -> None:
    spline = TensorSpline(
        np.ones((8, 8)),
        (np.arange(8.0), np.arange(8.0)),
        bases="bspline3",
        modes="mirror",
    )
    query = (np.linspace(0.0, 7.0, 100), np.linspace(0.0, 7.0, 100))

    with pytest.raises(MemoryError, match="max_retained_bytes"):
        spline.query_plan(query, grid=False, max_retained_bytes=64)


def test_query_plan_rejects_batched_grids_explicitly() -> None:
    spline = TensorSpline(
        np.ones((4, 4)),
        (np.arange(4.0), np.arange(4.0)),
        bases="linear",
        modes="mirror",
    )
    query = (np.ones((2, 4)), np.ones((2, 4)))

    with pytest.raises(ValueError, match="unbatched grid"):
        spline.query_plan(query, grid=True)


# --------------------------------------------------------------------------- #
# 1) Dirac-impulse sanity check on a cardinal grid
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("basis", basis_map.keys())
@pytest.mark.parametrize("mode", mode_map.keys())  # "periodic" is in map
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
        # Need a very long signal so poles produce finite coeffs
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
    elif mode == "periodic":
        sig_ext_val = 0  # impulse repeats every full period, not at edges here
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
    if dtype == "float64":
        atol = 1e-8
    elif dtype == "float32":
        atol = 2e-4
    else:
        raise NotImplementedError(f"Unsupported test dtype '{dtype}'")

    assert values == pytest.approx(values_exact, abs=atol)


# --------------------------------------------------------------------------- #
# 2) N-D / dtype regression test (unchanged)
# --------------------------------------------------------------------------- #
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
        np.testing.assert_equal(data_eval_tp.real, data_eval_tp.imag)

    # Meshgrid evaluation
    eval_coords_mg = np.meshgrid(*eval_coords_seq, indexing="ij")
    data_eval_mg = tensor_spline(coordinates=eval_coords_mg, grid=False)
    evaluation_atol = 8e-5 if real_dtype == np.float32 else 2e-12
    np.testing.assert_allclose(
        data_eval_tp, data_eval_mg, rtol=evaluation_atol, atol=evaluation_atol
    )

    # Reshaped meshgrid evaluation
    eval_coords_mg_rs = np.reshape(eval_coords_mg, (ndim, -1))
    data_eval_mg_rs = tensor_spline(coordinates=eval_coords_mg_rs, grid=False)
    np.testing.assert_allclose(
        data_eval_tp,
        np.reshape(data_eval_mg_rs, data_eval_mg.shape),
        rtol=evaluation_atol,
        atol=evaluation_atol,
    )

    # Batch-processing: tensor product
    eval_coords_tp_batch = []
    for coords, batch_offsets in zip(eval_coords_seq, batch_offsets_seq):
        coords_batch = np.stack([coords + b for b in batch_offsets])
        coords_batch = np.stack([coords_batch, coords_batch])  # add one more batch dim
        eval_coords_tp_batch.append(coords_batch)
    eval_coords_tp_batch = tuple(eval_coords_tp_batch)
    data_eval_tp_batch = tensor_spline(coordinates=eval_coords_tp_batch, grid=True)

    # Batch-processing: meshgrid
    eval_coords_mg_batch = []
    for coords, batch_offsets in zip(eval_coords_mg, batch_offsets_seq):
        coords_mg_batch = np.stack([coords + b for b in batch_offsets])
        coords_mg_batch = np.stack([coords_mg_batch, coords_mg_batch])
        eval_coords_mg_batch.append(coords_mg_batch)
    eval_coords_mg_batch = tuple(eval_coords_mg_batch)
    data_eval_mg_batch = tensor_spline(coordinates=eval_coords_mg_batch, grid=False)
    np.testing.assert_allclose(
        data_eval_tp_batch,
        data_eval_mg_batch,
        rtol=evaluation_atol,
        atol=evaluation_atol,
    )


# --------------------------------------------------------------------------- #
# 3) Periodic analytic-function test
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("basis", ["linear", "bspline3", "bspline5"])
@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_periodic_padding_analytic(basis: str, dtype: str) -> None:

    # Create a periodic analytical function
    L = 1.0
    nsamp = 32
    x = np.linspace(0, L, nsamp, endpoint=False, dtype=dtype)

    def f(xx):
        return np.sin(2 * np.pi * xx / L) + 0.3 * np.cos(4 * np.pi * xx / L)

    data = f(x).astype(dtype, copy=False)

    # Create the tensor spline with periodic mode
    ts = TensorSpline(data=data, coordinates=(x,), bases=basis, modes="periodic")

    # Query outside the base interval
    x_query = np.concatenate([x - L, x + 0.5 * L, x + 2 * L]).astype(dtype)
    y_pred = ts(coordinates=(x_query,), grid=False)
    y_true = f(np.mod(x_query, L))

    # Tolerances
    if dtype == "float64":
        atol = 1e-8
    else:
        atol = 3e-4

    np.testing.assert_allclose(y_pred, y_true, atol=atol, rtol=0)
