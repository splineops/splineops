"""
Persist one frame and evaluate several affine geometries
=========================================================

Save one frame's spline coefficients and reuse them for several affine geometries.

An affine coefficient field depends on the sampled frame, spline degree,
boundary mode, precision, and input grid.  It does not depend on the output
geometry, so registration or augmentation pipelines can prepare it once and
evaluate several compatible transforms concurrently.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import matplotlib.pyplot as plt

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
rows, columns = np.mgrid[-1 : 1 : complex(shape[0]), -1 : 1 : complex(shape[1])]
# A deterministic synthetic frame makes the two rotations easy to compare.
frame = np.exp(-18 * ((columns + 0.35) ** 2 + (rows + 0.2) ** 2))
frame += 0.7 * np.exp(-35 * ((columns - 0.3) ** 2 + (rows - 0.35) ** 2))
frame += 0.25 * (np.abs(rows + 0.4 * columns) < 0.06)
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

# %%
# One prepared frame, two output geometries
# -------------------------------------------
# The same saved coefficient field produces both rotated views. Their values
# match the direct evaluation checked above.

fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
for axis, image, title in zip(
    axes, (frame, *transformed), ("Input frame", "Rotation +8°", "Rotation −13°")
):
    axis.imshow(image, cmap="viridis", vmin=0, vmax=frame.max())
    axis.set_title(title)
    axis.set_axis_off()
plt.show()
