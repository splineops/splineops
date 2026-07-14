"""Strict contract tests for the native endpoint-aligned resize backend."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("splineops._lsresize") is None,
    reason="native resize extension is not available",
)


def _resize_native(
    values: np.ndarray,
    zoom: tuple[float, ...],
    degrees: tuple[int, int, int],
    axes=None,
) -> np.ndarray:
    from splineops import _lsresize

    interp, analy, synthe = degrees
    return _lsresize.resize_nd(
        values,
        zoom,
        interp,
        analy,
        synthe,
        axes,
    )


PROJECTION_DEGREES = [
    pytest.param((1, 0, 1), id="linear-oblique"),
    pytest.param((2, 1, 2), id="quadratic-oblique"),
    pytest.param((3, 1, 3), id="cubic-oblique"),
    pytest.param((3, 3, 3), id="cubic-ls"),
]


@pytest.mark.parametrize("degrees", PROJECTION_DEGREES)
def test_projection_depends_on_realized_shape_not_nominal_zoom(degrees):
    rng = np.random.default_rng(20260714)
    values = rng.standard_normal(10)

    # Both requested factors realize the same five-point endpoint grid.
    first = _resize_native(values, (0.50,), degrees)
    second = _resize_native(values, (0.51,), degrees)

    assert first.shape == second.shape == (5,)
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize(
    "degrees",
    [
        pytest.param((1, -1, 1), id="linear"),
        pytest.param((3, -1, 3), id="cubic"),
        *PROJECTION_DEGREES,
    ],
)
@pytest.mark.parametrize(
    "shape,zoom",
    [
        pytest.param((9,), (0.56,), id="one-dimensional"),
        pytest.param((9, 7), (0.56, 1.29), id="two-dimensional"),
        pytest.param((7, 5, 3), (0.57, 1.4, 1.0), id="three-dimensional"),
    ],
)
def test_native_resize_preserves_constants_strictly(degrees, shape, zoom):
    values = np.full(shape, 3.25, dtype=np.float64)

    actual = _resize_native(values, zoom, degrees)

    np.testing.assert_allclose(actual, 3.25, rtol=0.0, atol=2e-10)


@pytest.mark.parametrize(
    "degrees",
    [
        pytest.param((1, -1, 1), id="linear"),
        pytest.param((3, -1, 3), id="cubic"),
        pytest.param((3, 1, 3), id="cubic-oblique"),
        pytest.param((3, 3, 3), id="cubic-ls"),
    ],
)
def test_singleton_input_axis_is_replicated_exactly(degrees):
    values = np.array([[2.0, -5.0, 11.0]], dtype=np.float64)

    actual = _resize_native(values, (4.0, 1.0), degrees)

    expected = np.repeat(values, 4, axis=0)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "degrees",
    [
        pytest.param((0, -1, 0), id="nearest"),
        pytest.param((1, -1, 1), id="linear"),
        pytest.param((2, -1, 2), id="quadratic"),
        pytest.param((3, -1, 3), id="cubic"),
    ],
)
def test_singleton_interpolation_output_evaluates_symmetric_center(degrees):
    values = np.arange(8, dtype=np.float64)

    actual = _resize_native(values, (0.01,), degrees)

    assert actual.shape == (1,)
    np.testing.assert_allclose(actual, [3.5], rtol=0.0, atol=2e-10)


@pytest.mark.parametrize("degrees", PROJECTION_DEGREES)
def test_singleton_projection_output_is_line_mean(degrees):
    values = np.array([0.0, 2.0, 8.0, 10.0], dtype=np.float64)

    actual = _resize_native(values, (0.01,), degrees)

    assert actual.shape == (1,)
    np.testing.assert_array_equal(actual, np.array([values.mean()]))


@pytest.mark.parametrize("degrees", PROJECTION_DEGREES)
def test_realized_identity_projection_is_exact(degrees):
    rng = np.random.default_rng(11)
    values = rng.standard_normal((17, 3))

    # These are not nominally one, but round to the input shape. The realized
    # endpoint grid is therefore the identity grid.
    actual = _resize_native(values, (0.999, 1.001), degrees)

    np.testing.assert_array_equal(actual, values)


@pytest.mark.parametrize("degrees", PROJECTION_DEGREES)
def test_identity_channel_axis_matches_independent_channel_resizes(degrees):
    rng = np.random.default_rng(71)
    values = rng.standard_normal((19, 3))

    actual = _resize_native(values, (0.47, 1.0), degrees)
    expected = np.stack(
        [_resize_native(values[:, channel], (0.47,), degrees)
         for channel in range(values.shape[1])],
        axis=1,
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-12)


def test_custom_synthesis_space_is_not_treated_as_identity():
    rng = np.random.default_rng(123)
    values = rng.standard_normal(15)

    actual = _resize_native(values, (1.0,), (3, 1, 2))

    assert actual.shape == values.shape
    assert not np.array_equal(actual, values)


@pytest.mark.parametrize("zoom", [(0.0,), (-1.0,), (np.nan,), (np.inf,)])
def test_native_rejects_invalid_zoom(zoom):
    values = np.arange(8, dtype=np.float64)

    with pytest.raises((ValueError, RuntimeError)):
        _resize_native(values, zoom, (1, -1, 1))


def test_native_rejects_zoom_rank_mismatch():
    values = np.ones((4, 5), dtype=np.float64)

    with pytest.raises(RuntimeError, match="length must match"):
        _resize_native(values, (1.0,), (1, -1, 1))


def test_tiny_positive_zoom_clamps_to_one_output_sample():
    from splineops import _lsresize

    plan = _lsresize.ResizePlan((8,), (1e-12,), 1, -1, 1)

    assert plan.output_shape == (1,)


def test_native_axes_none_means_all_and_empty_means_none():
    from splineops import _lsresize

    values = np.arange(20.0).reshape(4, 5)
    default = _lsresize.resize_nd(values, (0.5, 0.6), 1, -1, 1)
    explicit_all = _lsresize.resize_nd(
        values, (0.5, 0.6), 1, -1, 1, (0, 1)
    )
    no_axes = _lsresize.resize_nd(values, (0.5, 0.6), 3, 1, 1, [])

    np.testing.assert_array_equal(default, explicit_all)
    np.testing.assert_array_equal(no_axes, values)


def test_native_plan_normalizes_and_exposes_axes():
    from splineops import _lsresize

    plan = _lsresize.ResizePlan(
        (4, 5), (1.0, 2.0), 1, -1, 1, np.array([-1], dtype=np.int64)
    )

    assert plan.axes == (1,)
    assert plan.output_shape == (4, 10)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_native_fused_linear_axes_ignore_unselected_zoom(dtype):
    from splineops import _lsresize

    values = np.random.default_rng(20260714).standard_normal((17, 19, 7)).astype(
        dtype
    )
    identity_zoom = (0.53, 0.61, 1.0)
    ignored_zoom = (0.53, 0.61, 4.25)
    expected = _lsresize.resize_nd(
        values, identity_zoom, 1, -1, 1, (0, 1)
    )
    actual = _lsresize.resize_nd(
        values, ignored_zoom, 1, -1, 1, (0, 1)
    )

    np.testing.assert_array_equal(actual, expected)

    plan = _lsresize.ResizePlan(
        values.shape, ignored_zoom, 1, -1, 1, (0, 1)
    )
    assert plan.zoom_factors == identity_zoom
    np.testing.assert_array_equal(plan.apply(values), expected)

    output = np.empty(expected.shape, dtype=dtype)
    assert plan.apply_into(values, output) is output
    np.testing.assert_array_equal(output, expected)


@pytest.mark.parametrize(
    "axes,exception",
    [((0, 0), ValueError), ((2,), ValueError), ((0.5,), TypeError), ((True,), TypeError)],
)
def test_native_rejects_invalid_axes(axes, exception):
    from splineops import _lsresize

    with pytest.raises(exception):
        _lsresize.resize_nd(np.ones((4, 5)), (1.0, 1.0), 1, -1, 1, axes)
