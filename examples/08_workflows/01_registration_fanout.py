"""
Persist one frame and evaluate several affine geometries
=========================================================

An affine coefficient field depends on the sampled frame, spline degree,
boundary mode, precision, and input grid.  It does not depend on the output
geometry, so registration or augmentation pipelines can prepare it once and
evaluate several compatible transforms concurrently.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from splineops.affine import AffinePlan


def rotation(shape, angle):
    radians = np.radians(-angle)
    matrix = np.array(
        [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    )
    center = (np.asarray(shape) - 1.0) / 2.0
    return matrix, center - matrix @ center


shape = (96, 112)
frame = np.random.default_rng(20260719).standard_normal(shape)
matrix_a, offset_a = rotation(shape, 8.0)
matrix_b, offset_b = rotation(shape, -13.0)
plans = (
    AffinePlan(shape, matrix_a, offset_a, degree=3, mode="mirror"),
    AffinePlan(shape, matrix_b, offset_b, degree=3, mode="mirror"),
)

field = plans[0].prepare_coefficients(frame)
with TemporaryDirectory() as directory:
    archive = Path(directory) / "frame-coefficients.npz"
    field.save(archive)
    restored = plans[1].load_coefficients(archive)

    with ThreadPoolExecutor(max_workers=2) as executor:
        transformed = tuple(
            executor.map(lambda plan: plan.apply_coefficients(restored), plans)
        )

references = tuple(plan(frame) for plan in plans)
for actual, reference in zip(transformed, references):
    np.testing.assert_equal(actual, reference)

print("shared coefficient bytes:", restored.nbytes)
print("geometry outputs:", tuple(result.shape for result in transformed))
