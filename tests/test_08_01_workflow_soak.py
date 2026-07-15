from concurrent.futures import ThreadPoolExecutor

import numpy as np

from splineops.affine import AffinePlan
from splineops.differentials import DifferentialPlan, DifferentialResult


def _rotation(shape, angle):
    radians = np.radians(-angle)
    matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    center = (np.asarray(shape) - 1.0) / 2.0
    return matrix, center - matrix @ center


def test_persisted_registration_field_fans_out_across_threads_and_geometries(
    tmp_path,
):
    shape = (32, 40)
    frame = np.random.default_rng(20260719).standard_normal(shape)
    plans = []
    for angle in (7.0, -11.0, 18.0):
        matrix, offset = _rotation(shape, angle)
        plans.append(
            AffinePlan(
                shape,
                matrix,
                offset,
                degree=3,
                mode="mirror",
                max_retained_bytes=2 * 1024**2,
            )
        )
    field = plans[0].prepare_coefficients(frame)
    archive = tmp_path / "registration-field.npz"
    field.save(archive)
    restored = plans[-1].load_coefficients(archive)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(lambda plan: plan.apply_coefficients(restored), plans)
        )

    for plan, result in zip(plans, results):
        np.testing.assert_equal(result, plan(frame))
        assert plan.retained_bytes <= 2 * 1024**2


def test_batched_volume_warp_and_buffered_feature_pipeline_matches_scalar_calls():
    shape = (10, 12, 14)
    volumes = (
        np.random.default_rng(20260719).standard_normal((2,) + shape).astype(np.float32)
    )
    radians = np.radians(-5.0)
    matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians), 0.0],
            [np.sin(radians), np.cos(radians), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    center = (np.asarray(shape, dtype=np.float32) - 1.0) / 2.0
    affine = AffinePlan(
        shape,
        matrix,
        center - matrix @ center,
        degree=3,
        mode="mirror",
        dtype=np.float32,
        max_retained_bytes=4 * 1024**2,
    )
    warped = np.empty_like(volumes)
    assert affine(volumes, spatial_axes=(1, 2, 3), out=warped) is warped

    plan = DifferentialPlan(shape, spacing=(0.8, 0.8, 1.5))
    output = DifferentialResult(
        tuple(np.empty_like(warped) for _ in range(3)),
        None,
        np.empty_like(warped),
    )
    returned = plan(
        warped,
        gradient=True,
        hessian=False,
        laplacian=True,
        spatial_axes=(1, 2, 3),
        out=output,
    )

    assert returned is output
    for batch in range(volumes.shape[0]):
        expected_warp = affine(volumes[batch])
        expected = plan(expected_warp, gradient=True, hessian=False, laplacian=True)
        np.testing.assert_equal(warped[batch], expected_warp)
        for component, reference in zip(returned.gradient, expected.gradient):
            np.testing.assert_equal(component[batch], reference)
        np.testing.assert_equal(returned.laplacian[batch], expected.laplacian)
