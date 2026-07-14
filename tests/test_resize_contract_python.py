import importlib
import inspect

import numpy as np
import pytest


@pytest.fixture
def resize_module(monkeypatch):
    """Exercise the public API with the Python reference backend."""
    module = importlib.import_module("splineops.resize.resize")
    monkeypatch.setattr(module, "_HAS_CPP", False)
    monkeypatch.setattr(module, "_resize_nd_cpp", None)
    monkeypatch.setattr(module, "_ResizePlanCpp", None)
    return module


@pytest.mark.parametrize(
    "method",
    [
        "fast",
        "linear",
        "quadratic",
        "cubic",
        "linear-antialiasing",
        "quadratic-antialiasing",
        "cubic-antialiasing",
    ],
)
def test_same_grid_is_exact_identity(resize_module, method):
    rng = np.random.default_rng(10)
    data = rng.standard_normal((9, 7))

    # 0.96 and 1.04 both request the original integer shape. The nominal
    # zoom must have no effect after that shape has been resolved.
    for zoom in ((1.0, 1.0), (0.96, 1.04)):
        actual = resize_module.resize(data, zoom_factors=zoom, method=method)
        np.testing.assert_array_equal(actual, data)


@pytest.mark.parametrize(
    "method",
    ["linear", "cubic", "linear-antialiasing", "cubic-antialiasing"],
)
def test_equal_output_shape_has_equal_endpoint_geometry(resize_module, method):
    data = np.linspace(-1.0, 2.0, 10)

    first = resize_module.resize(data, zoom_factors=0.50, method=method)
    second = resize_module.resize(data, zoom_factors=0.51, method=method)

    assert first.shape == second.shape == (5,)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "method",
    [
        "fast",
        "linear",
        "quadratic",
        "cubic",
        "linear-antialiasing",
        "quadratic-antialiasing",
        "cubic-antialiasing",
    ],
)
@pytest.mark.parametrize(
    "shape,output_size",
    [((7,), (4,)), ((4,), (9,)), ((1,), (7,)), ((7,), (1,)), ((1, 5), (3, 1))],
)
def test_constants_are_preserved_for_regular_and_degenerate_grids(
    resize_module, method, shape, output_size
):
    data = np.full(shape, 2.75)

    actual = resize_module.resize(
        data, output_size=output_size, method=method
    )

    assert actual.shape == output_size
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, 2.75, atol=5e-10, rtol=0.0)


@pytest.mark.parametrize("method", ["fast", "cubic", "cubic-antialiasing"])
def test_singleton_input_axis_replicates(resize_module, method):
    data = np.array([[-2.0], [3.5]])

    actual = resize_module.resize(
        data, output_size=(6,), axes=(1,), method=method
    )

    np.testing.assert_array_equal(actual, np.repeat(data, 6, axis=1))


def test_single_output_interpolation_uses_symmetric_center(resize_module):
    data = np.array([0.0, 1.0, 4.0, 9.0])

    linear = resize_module.resize(data, output_size=(1,), method="linear")
    nearest = resize_module.resize(data, output_size=(1,), method="fast")

    # x=(N-1)/2=1.5: linear interpolation and the symmetric nearest-neighbour
    # tie rule both average the two equidistant central samples.
    np.testing.assert_array_equal(linear, np.array([2.5]))
    np.testing.assert_array_equal(nearest, np.array([2.5]))


@pytest.mark.parametrize(
    "method",
    [
        "linear-antialiasing",
        "quadratic-antialiasing",
        "cubic-antialiasing",
    ],
)
def test_single_output_projection_is_line_mean(resize_module, method):
    data = np.array([[0.0, 1.0, 4.0, 9.0], [2.0, 8.0, -1.0, 3.0]])

    actual = resize_module.resize(
        data, output_size=(1,), axes=(1,), method=method
    )

    np.testing.assert_array_equal(actual[:, 0], np.mean(data, axis=1))


@pytest.mark.parametrize("length", [1, 3, 5])
def test_positive_zoom_uses_half_away_rounding_and_never_emits_zero(
    resize_module, length
):
    data = np.arange(length, dtype=np.float64)

    actual = resize_module.resize(data, zoom_factors=0.5, method="linear")

    assert actual.shape == (max(1, int(np.floor(length * 0.5 + 0.5))),)


def test_axes_limit_geometry_and_normalize_negative_indices(resize_module):
    data = np.arange(2 * 5 * 3, dtype=np.float64).reshape(2, 5, 3)

    actual = resize_module.resize(
        data, output_size=(7,), axes=(-2,), method="linear"
    )
    plan = resize_module.ResizePlan(
        data.shape, zoom_factors=(0.5, 2.0), axes=(2, 0), method="linear"
    )

    assert actual.shape == (2, 7, 3)
    assert plan.axes == (2, 0)
    assert plan.zoom_factors == (2.0, 1.0, 0.5)
    assert plan.output_shape == (4, 5, 2)


def test_scalar_zoom_broadcasts_only_to_selected_axes(resize_module):
    data = np.ones((4, 5, 6))

    actual = resize_module.resize(
        data, zoom_factors=0.5, axes=(0, 2), method="linear"
    )

    assert actual.shape == (2, 5, 3)


@pytest.mark.parametrize("target_length", [9, 6])
def test_unselected_axes_are_never_filtered_by_custom_degrees(
    resize_module, target_length
):
    # target_length=6 also proves that a selected same-grid custom-synthesis
    # pass remains active; selection and geometric identity are independent.
    data = np.random.default_rng(52).standard_normal((6, 7))
    kwargs = {
        "output_size": (target_length,),
        "axes": (0,),
        "interp_degree": 3,
        "analy_degree": 1,
        "synthe_degree": 1,
    }

    expected = np.stack(
        [
            resize_module.resize_degrees(
                data[:, column],
                output_size=(target_length,),
                interp_degree=3,
                analy_degree=1,
                synthe_degree=1,
            )
            for column in range(data.shape[1])
        ],
        axis=1,
    )
    actual = resize_module.resize_degrees(data, **kwargs)
    planned = resize_module.ResizePlan.from_degrees(data.shape, **kwargs)(data)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(planned, expected, rtol=0.0, atol=2e-12)


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"zoom_factors": (0.5,)}, ValueError),
        ({"zoom_factors": (0.5, 1.0, 2.0)}, ValueError),
        ({"zoom_factors": (0.0, 1.0)}, ValueError),
        ({"zoom_factors": (-0.5, 1.0)}, ValueError),
        ({"zoom_factors": (np.nan, 1.0)}, ValueError),
        ({"zoom_factors": (np.inf, 1.0)}, ValueError),
        ({"zoom_factors": (True, 1.0)}, TypeError),
        ({"zoom_factors": (1.0 + 1.0j, 1.0)}, TypeError),
        ({"output_size": (4,)}, ValueError),
        ({"output_size": (4, 0)}, ValueError),
        ({"output_size": (4, -1)}, ValueError),
        ({"output_size": (4, 2.5)}, TypeError),
        ({"output_size": (4, True)}, TypeError),
        ({"zoom_factors": (1.0,), "axes": (0, 0)}, ValueError),
        ({"zoom_factors": (1.0,), "axes": (2,)}, ValueError),
        ({"zoom_factors": (1.0,), "axes": (0.5,)}, TypeError),
    ],
)
def test_invalid_geometry_is_rejected(resize_module, kwargs, exception):
    with pytest.raises(exception):
        resize_module.resize(np.ones((4, 5)), method="linear", **kwargs)


@pytest.mark.parametrize(
    "data",
    [
        np.ones(4, dtype=np.bool_),
        np.ones(4, dtype=np.complex128),
        np.array([object(), object()], dtype=object),
        np.array(["1", "2"]),
    ],
)
def test_non_real_or_non_numeric_inputs_are_rejected(resize_module, data):
    with pytest.raises(TypeError, match="real integer or floating dtype"):
        resize_module.resize(data, zoom_factors=1.0, method="linear")


@pytest.mark.parametrize(
    "dtype,expected_dtype",
    [
        (np.int16, np.float64),
        (np.float16, np.float64),
        (np.float32, np.float32),
        (np.float64, np.float64),
    ],
)
def test_supported_input_dtype_policy(resize_module, dtype, expected_dtype):
    data = np.arange(6, dtype=dtype)

    actual = resize_module.resize(data, output_size=(9,), method="linear")

    assert actual.dtype == np.dtype(expected_dtype)


@pytest.mark.parametrize("output", [np.complex64, np.bool_])
def test_non_real_output_dtype_is_rejected(resize_module, output):
    with pytest.raises(TypeError, match="real integer or floating dtype"):
        resize_module.resize(
            np.arange(5.0), zoom_factors=1.0, output=output, method="linear"
        )


def test_invalid_output_array_and_empty_input_are_rejected(resize_module):
    with pytest.raises(TypeError, match="real integer or floating dtype"):
        resize_module.resize(
            np.arange(5.0),
            zoom_factors=1.0,
            output=np.empty(5, dtype=np.complex128),
            method="linear",
        )
    with pytest.raises(ValueError, match="non-empty"):
        resize_module.resize(np.empty((0, 3)), zoom_factors=1.0, method="linear")


def test_plan_applies_same_validation_and_axes_contract(resize_module):
    plan = resize_module.ResizePlan(
        (3, 4), output_size=(7,), axes=(1,), method="cubic-antialiasing"
    )

    assert plan.output_shape == (3, 7)
    with pytest.raises(TypeError, match="real integer or floating dtype"):
        plan.apply(np.ones((3, 4), dtype=np.complex128))
    with pytest.raises(TypeError, match="real integer or floating dtype"):
        plan.apply(
            np.ones((3, 4)), output=np.empty((3, 7), dtype=np.complex128)
        )


def test_tiny_positive_zoom_uses_canonical_singleton_shape(resize_module):
    plan = resize_module.ResizePlan((8,), zoom_factors=0.01, method="linear")

    assert plan.output_shape == (1,)


def test_retired_size_policy_is_absent_from_public_signatures(resize_module):
    assert tuple(inspect.signature(resize_module.resize_degrees).parameters) == (
        "data", "zoom_factors", "output", "output_size", "axes",
        "interp_degree", "analy_degree", "synthe_degree",
    )
    assert tuple(inspect.signature(resize_module.ResizePlan).parameters) == (
        "input_shape", "zoom_factors", "output_size", "axes", "method",
    )
    assert tuple(
        inspect.signature(resize_module.ResizePlan.from_degrees).parameters
    ) == (
        "input_shape", "zoom_factors", "output_size", "axes",
        "interp_degree", "analy_degree", "synthe_degree",
    )
