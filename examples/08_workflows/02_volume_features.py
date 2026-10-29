"""
Warp a volume and write selected differential features
=======================================================

Warp a synthetic 3-D volume and compute selected differential features into reusable
output buffers.

This pipeline applies one 3-D affine transform and then requests only gradient
and Laplacian outputs.  Exact structured buffers make allocation ownership
explicit while Affine and Differentials remain independent modules.
"""

import numpy as np
import matplotlib.pyplot as plt

from splineops.affine import AffinePlan
from splineops.differentials import DifferentialPlan, DifferentialResult

shape = (24, 28, 32)
z, y, x = np.meshgrid(
    *(np.linspace(-1, 1, size, dtype=np.float32) for size in shape), indexing="ij"
)
volume = np.exp(-9 * ((x + 0.25) ** 2 + (y - 0.15) ** 2 + z**2))
volume += 0.7 * np.exp(-20 * ((x - 0.4) ** 2 + (y + 0.35) ** 2 + z**2))
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

# %%
# Follow the volume through the workflow
# ---------------------------------------------
# A central slice shows the synthetic input, affine result, and the two
# requested feature types. All feature arrays come from the supplied buffers.

middle = shape[0] // 2
gradient_magnitude = np.sqrt(sum(component**2 for component in features.gradient))
laplacian_limit = np.max(np.abs(features.laplacian[middle]))
fig, axes = plt.subplots(1, 4, figsize=(12, 3.5), layout="constrained")
for axis, image, title, cmap in zip(
    axes,
    (
        volume[middle],
        warped[middle],
        gradient_magnitude[middle],
        features.laplacian[middle],
    ),
    ("Input slice", "Warped slice", "Gradient magnitude", "Laplacian"),
    ("viridis", "viridis", "magma", "RdBu_r"),
):
    limits = (
        {"vmin": -laplacian_limit, "vmax": laplacian_limit}
        if title == "Laplacian"
        else {}
    )
    axis.imshow(image, cmap=cmap, **limits)
    axis.set_title(title)
    axis.set_axis_off()
plt.show()
