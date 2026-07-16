"""Randomized cross-module contract checks for the stability soak."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings, strategies as st

from splineops import TensorSpline
from splineops.affine import AffinePlan
from splineops.differentials import DifferentialPlan, DifferentialResult
from splineops.multiscale.wavelets.haar import HaarWavelets
from splineops.multiscale.wavelets.spline_wavelets import Spline3Wavelets

PROPERTY_SETTINGS = settings(max_examples=18, deadline=None, derandomize=True)


def _shape_with_spatial_axes(spatial_shape, spatial_axes):
    shape = [2, 2, 2, 2]
    for axis, length in zip(spatial_axes, spatial_shape):
        shape[axis] = length
    return tuple(shape)


def _plane_slices(shape, spatial_axes):
    nonspatial_axes = tuple(
        axis for axis in range(len(shape)) if axis not in spatial_axes
    )
    for index in np.ndindex(*(shape[axis] for axis in nonspatial_axes)):
        selection = [slice(None)] * len(shape)
        for axis, item in zip(nonspatial_axes, index):
            selection[axis] = item
        yield tuple(selection)


@PROPERTY_SETTINGS
@given(
    height=st.integers(4, 9),
    width=st.integers(4, 9),
    spatial_axes=st.sampled_from(((0, 1), (1, 2), (2, 3))),
    degree=st.sampled_from((0, 1, 3)),
    dtype=st.sampled_from((np.float32, np.float64)),
    angle=st.integers(-25, 25),
    seed=st.integers(0, 2**32 - 1),
)
def test_affine_explicit_axes_and_strided_output_match_scalar_dispatch(
    height, width, spatial_axes, degree, dtype, angle, seed
):
    spatial_shape = (height, width)
    shape = _shape_with_spatial_axes(spatial_shape, spatial_axes)
    data = np.random.default_rng(seed).standard_normal(shape).astype(dtype)
    source = data.copy()
    radians = np.radians(-angle)
    matrix = np.asarray(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ],
        dtype=dtype,
    )
    center = (np.asarray(spatial_shape, dtype=dtype) - 1) / 2
    plan = AffinePlan(
        spatial_shape,
        matrix,
        center - matrix @ center,
        degree=degree,
        mode="mirror",
        dtype=dtype,
    )
    backing = np.empty(shape + (2,), dtype=dtype)
    output = backing[..., 0]

    returned = plan(data, spatial_axes=spatial_axes, out=output)

    assert returned is output
    assert not output.flags.c_contiguous
    np.testing.assert_equal(data, source)
    for selection in _plane_slices(shape, spatial_axes):
        np.testing.assert_equal(output[selection], plan(data[selection]))


@PROPERTY_SETTINGS
@given(
    height=st.integers(4, 8),
    width=st.integers(4, 8),
    spatial_axes=st.sampled_from(((0, 1), (1, 2), (2, 3))),
    outputs=st.sampled_from(
        (
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, False, True),
            (True, True, True),
        )
    ),
    dtype=st.sampled_from((np.float32, np.float64)),
    seed=st.integers(0, 2**32 - 1),
)
def test_differential_explicit_axes_buffers_match_scalar_dispatch(
    height, width, spatial_axes, outputs, dtype, seed
):
    spatial_shape = (height, width)
    shape = _shape_with_spatial_axes(spatial_shape, spatial_axes)
    data = np.random.default_rng(seed).standard_normal(shape).astype(dtype)
    source = data.copy()
    gradient, hessian, laplacian = outputs
    gradient_backing = np.empty((2,) + shape + (2,), dtype=dtype)
    hessian_backing = np.empty((3,) + shape + (2,), dtype=dtype)
    laplacian_backing = np.empty(shape + (2,), dtype=dtype)
    destination = DifferentialResult(
        (
            tuple(gradient_backing[index, ..., 0] for index in range(2))
            if gradient
            else None
        ),
        (
            tuple(hessian_backing[index, ..., 0] for index in range(3))
            if hessian
            else None
        ),
        laplacian_backing[..., 0] if laplacian else None,
    )
    plan = DifferentialPlan(spatial_shape, spacing=(0.75, 1.25))

    returned = plan(
        data,
        gradient=gradient,
        hessian=hessian,
        laplacian=laplacian,
        spatial_axes=spatial_axes,
        out=destination,
    )

    assert returned is destination
    np.testing.assert_equal(data, source)
    for selection in _plane_slices(shape, spatial_axes):
        expected = plan(
            data[selection],
            gradient=gradient,
            hessian=hessian,
            laplacian=laplacian,
        )
        if gradient:
            for actual, reference in zip(returned.gradient, expected.gradient):
                np.testing.assert_equal(actual[selection], reference)
        if hessian:
            for actual, reference in zip(returned.hessian, expected.hessian):
                np.testing.assert_equal(actual[selection], reference)
        if laplacian:
            np.testing.assert_equal(returned.laplacian[selection], expected.laplacian)


@st.composite
def _tensorspline_cases(draw):
    ndim = draw(st.integers(1, 3))
    shape = tuple(draw(st.integers(2, 7)) for _ in range(ndim))
    query_lengths = tuple(draw(st.integers(2, 8)) for _ in range(ndim))
    return ndim, shape, query_lengths


@PROPERTY_SETTINGS
@given(
    case=_tensorspline_cases(),
    basis=st.sampled_from(("linear", "keys", "bspline3", "omoms3")),
    mode=st.sampled_from(("mirror", "periodic")),
    grid=st.booleans(),
    dtype=st.sampled_from((np.float32, np.float64)),
    seed=st.integers(0, 2**32 - 1),
)
def test_tensorspline_geometry_plan_matches_changing_splines_and_strided_output(
    case, basis, mode, grid, dtype, seed
):
    _, shape, query_lengths = case
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(shape).astype(dtype)
    coordinates = tuple(np.arange(length, dtype=dtype) for length in shape)
    spline = TensorSpline(data, coordinates, bases=basis, modes=mode)
    if grid:
        query = tuple(
            np.linspace(-0.5, length - 0.5, count, dtype=dtype)
            for length, count in zip(shape, query_lengths)
        )
    else:
        count = max(query_lengths)
        query = tuple(
            rng.uniform(-0.5, length - 0.5, count).astype(dtype) for length in shape
        )
    plan = spline.query_plan(query, grid=grid)
    backing = np.empty(plan.output_shape + (2,), dtype=dtype)
    output = backing[..., 0]

    assert plan(spline, out=output) is output
    np.testing.assert_equal(output, spline(query, grid=grid))

    changed = spline.with_data(rng.standard_normal(shape).astype(dtype))
    np.testing.assert_equal(plan(changed), changed(query, grid=grid))


@st.composite
def _reversible_wavelet_cases(draw):
    scales = draw(st.integers(1, 3))
    divisor = 2**scales
    shape = tuple(divisor * draw(st.integers(1, 4)) for _ in range(2))
    return scales, shape


@PROPERTY_SETTINGS
@given(
    case=_reversible_wavelet_cases(),
    wavelet_class=st.sampled_from((HaarWavelets, Spline3Wavelets)),
    dtype=st.sampled_from((np.float32, np.float64)),
    seed=st.integers(0, 2**32 - 1),
)
def test_exact_wavelet_families_reconstruct_random_reversible_shapes(
    case, wavelet_class, dtype, seed
):
    scales, shape = case
    samples = np.random.default_rng(seed).standard_normal(shape).astype(dtype)
    wavelet = wavelet_class(scales=scales)

    reconstructed = wavelet.synthesis(wavelet.analysis(samples))

    tolerance = 2e-6 if dtype is np.float32 else 1e-10
    np.testing.assert_allclose(reconstructed, samples, rtol=0.0, atol=tolerance)
