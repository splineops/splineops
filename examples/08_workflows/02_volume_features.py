"""
Warp a volume and write selected differential features
=======================================================

This pipeline applies one 3-D affine transform and then requests only gradient
and Laplacian outputs.  Exact structured buffers make allocation ownership
explicit while Affine and Differentials remain independent modules.
"""

import numpy as np

from splineops.affine import AffinePlan
from splineops.differentials import DifferentialPlan, DifferentialResult

shape = (24, 28, 32)
volume = np.random.default_rng(20260719).standard_normal(shape).astype(np.float32)
radians = np.radians(-6.0)
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
)
warped = affine(volume)

differentials = DifferentialPlan(shape, spacing=(0.8, 0.8, 1.5))
output = DifferentialResult(
    tuple(np.empty_like(warped) for _ in range(3)),
    None,
    np.empty_like(warped),
)
features = differentials(
    warped,
    gradient=True,
    hessian=False,
    laplacian=True,
    out=output,
)

print("warped volume:", warped.shape, warped.dtype)
print("gradient components:", len(features.gradient))
print(
    "Laplacian range:", float(features.laplacian.min()), float(features.laplacian.max())
)
