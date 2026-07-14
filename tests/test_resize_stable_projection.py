"""Accuracy and plan-shape guards for stable endpoint-grid projection."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from splineops.resize._pycore.engine import python_resize
from splineops.resize._pycore.params import LSParams, Work1D
from splineops.resize._pycore.plan_1d import make_plan_1d
from splineops.resize._pycore.resize_1d import resize_1d_ws


PROJECTION_DEGREES = [
    pytest.param((1, 0, 1), id="linear-oblique"),
    pytest.param((2, 1, 2), id="quadratic-oblique"),
    pytest.param((3, 1, 3), id="cubic-oblique"),
    pytest.param((1, 1, 1), id="linear-ls"),
    pytest.param((2, 2, 2), id="quadratic-ls"),
    pytest.param((3, 3, 3), id="cubic-ls"),
]

ALL_PROJECTION_DEGREES = [
    (interp, analy, synthe)
    for interp in range(4)
    for analy in range(interp + 1)
    for synthe in range(interp + 1)
]
DIRECT_PROJECTION_DEGREES = [
    degrees for degrees in ALL_PROJECTION_DEGREES if degrees[1] >= 1
]
FD_PROJECTION_DEGREES = [
    degrees for degrees in ALL_PROJECTION_DEGREES if degrees[1] == 0
]


def _python_projection(
    values: np.ndarray,
    zoom: float,
    degrees: tuple[int, int, int],
) -> np.ndarray:
    interp, analy, synthe = degrees
    return python_resize(
        values,
        (zoom,),
        interp_degree=interp,
        analy_degree=analy,
        synthe_degree=synthe,
    )


def _native_projection(
    values: np.ndarray,
    zoom: float,
    degrees: tuple[int, int, int],
) -> np.ndarray:
    if importlib.util.find_spec("splineops._lsresize") is None:
        pytest.skip("native resize extension is not available")
    from splineops import _lsresize

    interp, analy, synthe = degrees
    return _lsresize.resize_nd(
        values,
        (zoom,),
        interp,
        analy,
        synthe,
    )


@pytest.fixture(params=["python", "native"])
def projection_backend(request):
    return _python_projection if request.param == "python" else _native_projection


# Generated from the direct B-spline inner-product definition with 100-digit
# mpmath arithmetic. These cover all 20 generalized direct-projection triples
# plus an analysis-degree-zero FD control without making tests depend on mpmath.
_ORACLE_INPUT = np.array(
    [
        0.0,
        1.0,
        -2.0,
        3.5,
        0.25,
        -1.5,
        2.25,
        4.0,
        -3.0,
        0.75,
        1.25,
        -0.5,
        2.75,
        -4.0,
        0.5,
        3.0,
    ],
    dtype=np.float64,
)

_ORACLE_OUTPUTS = {
    (1, 0, 1): np.array(
        [
            0.3949397252593216,
            0.481847490888702,
            1.7973086627417998,
            -0.3490328006728343,
            -0.6197785253714606,
            1.4843706195682647,
        ]
    ),
    (2, 1, 2): np.array(
        [
            -0.056776504981861084,
            0.6700611541403193,
            1.1805727411890545,
            0.19139428652559062,
            0.0025774160637808525,
            0.4675653091443706,
        ]
    ),
    (3, 1, 3): np.array(
        [
            -0.043560720957544645,
            0.6643179407790158,
            1.1765541359320693,
            0.1943953399178741,
            0.0035595578800364557,
            0.4659067719395534,
        ]
    ),
    (1, 1, 1): np.array(
        [
            -0.13769271664008506,
            0.664274322169059,
            1.2861509835194045,
            0.16334396597554493,
            -0.05063795853269538,
            0.5114300903774588,
        ]
    ),
    (2, 2, 2): np.array(
        [
            0.16328230637393204,
            0.5056653926364288,
            1.1778627696534054,
            0.3243797782594477,
            -0.0372817006181492,
            0.3954652137638026,
        ]
    ),
    (3, 3, 3): np.array(
        [
            0.15572271379551994,
            0.5357732224118412,
            1.0982049990253486,
            0.38225439338872347,
            0.02763734778059501,
            0.2565373609914633,
        ]
    ),
    (1, 1, 0): np.array(
        [
            -0.04603152353362614,
            0.656613089119397,
            1.1804270628913185,
            0.2237874964956546,
            -0.00463352334672772,
            0.4336432732143413,
        ]
    ),
    (2, 1, 0): np.array(
        [
            -0.09295607489002765,
            0.6795305219842753,
            1.2077862229224379,
            0.1843536736720159,
            -0.011506598175836675,
            0.472628434084243,
        ]
    ),
    (2, 1, 1): np.array(
        [
            -0.19624067224571268,
            0.6929780674770696,
            1.3183383622909806,
            0.11661963325219628,
            -0.05801564521574336,
            0.5563998366367063,
        ]
    ),
    (2, 2, 0): np.array(
        [
            0.10976294305148547,
            0.5256890880181256,
            1.201582495565557,
            0.3224334619845628,
            -0.05368618931352202,
            0.3981993444390676,
        ]
    ),
    (2, 2, 1): np.array(
        [
            0.08519855441274589,
            0.4812151746108237,
            1.3235713158618037,
            0.2752991281648447,
            -0.11672614862373223,
            0.48808250555977434,
        ]
    ),
    (3, 1, 0): np.array(
        [
            -0.09037500724143718,
            0.6768177520564527,
            1.2112242954091599,
            0.1820382773783807,
            -0.014370125121774652,
            0.47895460779699994,
        ]
    ),
    (3, 1, 1): np.array(
        [
            -0.19232635608975873,
            0.6889222599286233,
            1.3229541667550946,
            0.11391242446834317,
            -0.06179098871021516,
            0.5643306312060671,
        ]
    ),
    (3, 1, 2): np.array(
        [
            -0.053707562922733876,
            0.6669429530781992,
            1.1841783971669704,
            0.18912682993025245,
            -0.0003246157056034975,
            0.4738604339830967,
        ]
    ),
    (3, 2, 0): np.array(
        [
            0.1080673340620106,
            0.5268360908696486,
            1.201982388029305,
            0.3212992319830592,
            -0.05370702728253262,
            0.39911129873902884,
        ]
    ),
    (3, 2, 1): np.array(
        [
            0.08292824030390683,
            0.4827556862174402,
            1.3239487881201266,
            0.27400380001900704,
            -0.11671846382763566,
            0.48909213863821654,
        ]
    ),
    (3, 2, 2): np.array(
        [
            0.161557104337333,
            0.5068655465753145,
            1.1781579447396566,
            0.32334436939401817,
            -0.037300212186100926,
            0.39630759861689013,
        ]
    ),
    (3, 2, 3): np.array(
        [
            0.17078446265388517,
            0.5047163780887078,
            1.1716174690703058,
            0.32586543222243025,
            -0.030620811508359515,
            0.3860586015999459,
        ]
    ),
    (3, 3, 0): np.array(
        [
            0.09833721791378157,
            0.5497191482686946,
            1.136668159425238,
            0.3772771759494398,
            0.00032725278189244694,
            0.27367930923568895,
        ]
    ),
    (3, 3, 1): np.array(
        [
            0.062323943694396436,
            0.5253328181747996,
            1.2296889255185364,
            0.3424944276844783,
            -0.040282114296951436,
            0.32320794214387805,
        ]
    ),
    (3, 3, 2): np.array(
        [
            0.14748172173849072,
            0.5372317662709379,
            1.1047746400840157,
            0.37985678638471587,
            0.021686164412338716,
            0.2654195639574929,
        ]
    ),
}


@pytest.mark.parametrize("degrees", list(_ORACLE_OUTPUTS))
def test_projection_matches_arbitrary_precision_fixture(
    projection_backend,
    degrees,
):
    actual = projection_backend(_ORACLE_INPUT, 0.37, degrees)

    np.testing.assert_allclose(
        actual,
        _ORACLE_OUTPUTS[degrees],
        rtol=0.0,
        atol=2e-10,
    )


_STRONG_REDUCTION_INPUT = np.array(
    [
        0.0, 1.0, -2.0, 3.5, 0.25, -1.5, 2.25, 4.0,
        -3.0, 0.75, 1.25, -0.5, 2.75, -4.0, 0.5, 3.0,
        -1.25, 2.5, 0.125, -2.75, 4.5, -0.25, 1.75, -3.5,
        2.0, 0.625, -1.0, 3.25, -2.25, 4.25, 0.375, -0.75,
    ],
    dtype=np.float64,
)

_STRONG_REDUCTION_ORACLE = {
    (2, 2, 2): np.array([0.4691470498997869, 0.5792400468744067]),
    (3, 3, 3): np.array([0.4659567904376345, 0.5824303063365590]),
}


@pytest.mark.parametrize("degrees", [(2, 2, 2), (3, 3, 3)])
def test_two_sample_projection_spans_full_mirror_periods(
    projection_backend,
    degrees,
):
    actual = projection_backend(_STRONG_REDUCTION_INPUT, 0.05, degrees)

    assert actual.shape == (2,)
    np.testing.assert_allclose(
        actual,
        _STRONG_REDUCTION_ORACLE[degrees],
        rtol=0.0,
        atol=2e-10,
    )


@pytest.mark.parametrize("degrees", PROJECTION_DEGREES)
def test_projection_commutes_with_endpoint_reflection(
    projection_backend,
    degrees,
):
    impulse = np.zeros(32, dtype=np.float64)
    impulse[0] = 1.0

    left = projection_backend(impulse, 0.37, degrees)
    right = projection_backend(impulse[::-1], 0.37, degrees)

    np.testing.assert_allclose(left, right[::-1], rtol=0.0, atol=2e-10)


@pytest.mark.parametrize("degrees", DIRECT_PROJECTION_DEGREES)
def test_direct_plan_is_tail_and_padding_free_for_strong_reduction(degrees):
    size = _STRONG_REDUCTION_INPUT.size
    params = LSParams(*degrees, zoom=0.05, shift=0.0)
    plan = make_plan_1d(size, params)

    assert plan.direct_projection
    assert plan.outN == plan.out_total == 2
    assert plan.length_total == size
    assert plan.left_pad == plan.right_pad == 0
    assert plan.weights2d.size < 9 * size
    np.testing.assert_allclose(
        plan.weights2d.sum(axis=1), 1.0, rtol=0.0, atol=4e-15
    )

    unwrapped = plan.kmin[:, None] + np.arange(
        plan.win_len_max, dtype=np.int64
    )
    period = 2 * size - 2
    expected = np.mod(unwrapped, period)
    expected = np.where(expected >= size, period - expected, expected)
    np.testing.assert_array_equal(plan.idx2d, expected)


@pytest.mark.parametrize(
    "degrees",
    [(1, 1, 0), (2, 1, 2), (3, 2, 0), (3, 3, 3)],
)
def test_python_scalar_direct_projection_skips_extension_copy(degrees):
    params = LSParams(*degrees, zoom=0.37, shift=0.0)
    plan = make_plan_1d(_ORACLE_INPUT.size, params)
    workspace = Work1D()

    actual = resize_1d_ws(_ORACLE_INPUT, params, plan, workspace)

    assert workspace.ext.size == 0
    assert workspace.ext_full.size == 0
    np.testing.assert_allclose(
        actual,
        _ORACLE_OUTPUTS[degrees],
        rtol=0.0,
        atol=2e-10,
    )


@pytest.mark.parametrize("degrees", ALL_PROJECTION_DEGREES)
def test_zero_shift_projection_has_no_hidden_output_tail(degrees):
    params = LSParams(*degrees, zoom=0.37, shift=0.0)
    plan = make_plan_1d(127, params)

    assert plan.out_total == plan.outN
    assert plan.direct_projection == (degrees[1] >= 1)


@pytest.mark.parametrize("degrees", DIRECT_PROJECTION_DEGREES)
def test_nonzero_shift_retains_conservative_projection_tail(degrees):
    params = LSParams(*degrees, zoom=0.37, shift=0.125)
    plan = make_plan_1d(127, params)

    assert not plan.direct_projection
    assert plan.out_total > plan.outN


@pytest.mark.parametrize("degrees", FD_PROJECTION_DEGREES)
def test_analysis_degree_zero_keeps_finite_difference_plan(degrees):
    params = LSParams(*degrees, zoom=0.37, shift=0.0)

    assert not make_plan_1d(127, params).direct_projection


def test_direct_plan_rejects_unrepresentable_support_before_allocation():
    size = 1 << 30
    params = LSParams(3, 3, 3, zoom=2.0 / size, shift=0.0)

    with pytest.raises(OverflowError, match="native limits"):
        make_plan_1d(size, params)


@pytest.mark.parametrize("degrees", DIRECT_PROJECTION_DEGREES)
def test_native_and_python_direct_projection_agree(degrees):
    if importlib.util.find_spec("splineops._lsresize") is None:
        pytest.skip("native resize extension is not available")
    values = np.random.default_rng(20260714).standard_normal(257)

    expected = _python_projection(values, 0.37, degrees)
    actual = _native_projection(values, 0.37, degrees)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-10)


@pytest.mark.parametrize("degrees", DIRECT_PROJECTION_DEGREES)
def test_native_batched_and_python_direct_projection_agree(degrees, monkeypatch):
    if importlib.util.find_spec("splineops._lsresize") is None:
        pytest.skip("native resize extension is not available")
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "on")
    values = np.random.default_rng(42).standard_normal((32, 129))
    zoom = (1.0, 0.37)
    interp, analy, synthe = degrees

    expected = python_resize(
        values,
        zoom,
        interp_degree=interp,
        analy_degree=analy,
        synthe_degree=synthe,
        axes=(1,),
    )
    from splineops import _lsresize

    actual = _lsresize.resize_nd(
        values, zoom, interp, analy, synthe, (1,)
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-10)


@pytest.mark.parametrize(
    "degrees",
    [(1, 1, 0), (2, 1, 2), (3, 2, 1), (3, 3, 2)],
)
def test_native_forced_float32_direct_projection_agrees_with_python(
    degrees,
    monkeypatch,
):
    if importlib.util.find_spec("splineops._lsresize") is None:
        pytest.skip("native resize extension is not available")
    monkeypatch.setenv("LSRESIZE_BATCHED_AXIS", "on")
    monkeypatch.setenv("LSRESIZE_PRECISION", "float32")
    values = np.random.default_rng(7).standard_normal((24, 129)).astype(
        np.float32
    )
    zoom = (1.0, 0.37)
    interp, analy, synthe = degrees

    expected = python_resize(
        values,
        zoom,
        interp_degree=interp,
        analy_degree=analy,
        synthe_degree=synthe,
        axes=(1,),
    )
    from splineops import _lsresize

    actual = _lsresize.resize_nd(
        values, zoom, interp, analy, synthe, (1,)
    )

    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize(
    "degrees",
    [(2, 2, 2), (3, 1, 3), (3, 2, 0), (3, 3, 1), (3, 3, 3)],
)
def test_long_direct_projection_stays_bounded(degrees):
    if importlib.util.find_spec("splineops._lsresize") is None:
        pytest.skip("native resize extension is not available")
    values = np.linspace(0.0, 1.0, 65536, dtype=np.float64)

    actual = _native_projection(values, 0.37, degrees)

    assert np.isfinite(actual).all()
    assert actual.min() > -1e-3
    assert actual.max() < 1.001
    assert abs(float(actual.mean()) - 0.5) < 2e-10
    assert np.max(np.abs(np.diff(actual))) < 1e-3
