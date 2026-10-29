"""Render the homepage's synthetic-field illustration from public SplineOps APIs.

Run from any directory with the package and Matplotlib installed:
    python scripts/render_landing_illustration.py

The checked-in lossless WebP images keep documentation builds independent of rendering time.
No external data, downloads, or randomized inputs are needed. Heights and colours
in the derivative panel are normalized separately for this qualitative illustration.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import Bbox
import numpy as np

from splineops import TensorSpline, resize
from splineops.affine import AffinePlan
from splineops.differentials import Differentials

DESTINATION = Path(__file__).resolve().parents[1] / "docs" / "_static" / "landing"
BACKGROUND = "#0c1729"
SURFACE = LinearSegmentedColormap.from_list(
    "splineops_surface", ["#173c65", "#247d9f", "#50d5c0", "#dcf7bd", "#ffcc75"]
)
DERIVATIVE = LinearSegmentedColormap.from_list(
    "splineops_derivative", ["#173c65", "#5158a8", "#bf78c7", "#f8bbad", "#ffe1a1"]
)


def make_fields():
    """Keep each stage explicit so the illustration is reproducible and inspectable."""
    coordinates = np.linspace(-2.8, 2.8, 21, dtype=np.float64)
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    samples = np.exp(-0.24 * (x**2 + y**2)) * (
        0.6 + 0.4 * np.cos(2.2 * x) * np.cos(2.2 * y)
    )
    samples += 0.65 * np.exp(-3 * ((x + 0.85) ** 2 + (y - 0.65) ** 2))
    # Taper smoothly to zero so rotation does not introduce a hard image edge.
    window = np.cos(np.pi * coordinates / 5.6) ** 2
    samples *= np.outer(window, window)
    samples /= samples.max()

    spline = TensorSpline(
        data=samples,
        coordinates=(coordinates, coordinates),
        bases="bspline3",
        modes="mirror",
    )
    fine_coordinates = np.linspace(coordinates[0], coordinates[-1], 129)
    fine = spline(coordinates=(fine_coordinates, fine_coordinates))
    coarse = resize(fine, output_size=(49, 49), method="cubic-antialiasing")

    angle = np.deg2rad(24)
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    center = (np.asarray(coarse.shape) - 1) / 2
    transform = AffinePlan(
        coarse.shape, matrix, center - matrix @ center, degree=3, mode="zero"
    )
    transformed = transform(coarse)
    spacing = (coordinates[-1] - coordinates[0]) / (coarse.shape[0] - 1)
    components = Differentials(
        transformed, spacing=(spacing, spacing)
    ).gradient_components()
    magnitude = np.sqrt(sum(component**2 for component in components))

    # Check the underlying outputs before drawing; rendering must not disguise
    # a failed operator or a mismatched construction grid.
    np.testing.assert_allclose(
        spline(coordinates=(coordinates, coordinates)), samples, rtol=0, atol=1e-12
    )
    for field in (fine, transformed, magnitude):
        assert np.isfinite(field).all()
    return coordinates, samples, fine, transformed, magnitude


def render_surface(name, field, *, cmap=SURFACE, samples=None):
    """Draw a height map with a consistent camera, footprint, and quiet base grid."""
    fig = plt.figure(figsize=(5, 4), dpi=180, facecolor=BACKGROUND)
    axis = fig.add_axes((0, 0, 1, 1), projection="3d", computed_zorder=False)
    axis.set_facecolor(BACKGROUND)
    coordinates = np.linspace(-2.8, 2.8, field.shape[0])
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    axis.plot_surface(
        x,
        y,
        field,
        cmap=cmap,
        vmin=0,
        vmax=1.1,
        rcount=65,
        ccount=65,
        linewidth=0,
        antialiased=True,
        shade=False,
        zorder=3,
    )
    axis.plot_wireframe(
        x,
        y,
        field,
        rcount=17,
        ccount=17,
        linewidth=0.32,
        color=(0.65, 0.95, 0.98, 0.23),
        zorder=4,
    )
    for position in np.linspace(-2.8, 2.8, 9):
        axis.plot(
            [-2.8, 2.8],
            [position, position],
            [-0.12, -0.12],
            color="#263d58",
            linewidth=0.45,
            zorder=1,
        )
        axis.plot(
            [position, position],
            [-2.8, 2.8],
            [-0.12, -0.12],
            color="#263d58",
            linewidth=0.45,
            zorder=1,
        )
    if samples is not None:
        sample_coordinates, values = samples
        sample_y, sample_x = np.meshgrid(
            sample_coordinates, sample_coordinates, indexing="ij"
        )
        # A subset of the construction samples keeps the surface readable.
        axis.scatter(
            sample_x[::2, ::2],
            sample_y[::2, ::2],
            values[::2, ::2],
            s=5,
            c="#ffffff",
            alpha=0.85,
            linewidths=0,
            depthshade=False,
            zorder=5,
        )
    axis.set(xlim=(-2.8, 2.8), ylim=(-2.8, 2.8), zlim=(-0.15, 1.2))
    axis.set_box_aspect((1, 1, 0.65), zoom=1.18)
    axis.set_proj_type("ortho")
    axis.view_init(elev=32, azim=-53)
    axis.set_axis_off()
    fig.savefig(
        DESTINATION / name,
        facecolor=BACKGROUND,
        pil_kwargs={"lossless": True, "method": 6},
        # Trim the unused space above the camera while keeping the full surface.
        bbox_inches=Bbox.from_bounds(0, 0, 5, 3.4),
    )
    plt.close(fig)


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    coordinates, samples, fine, transformed, magnitude = make_fields()
    render_surface("reconstruct.webp", fine, samples=(coordinates, samples))
    render_surface("resample.webp", transformed)
    render_surface("differentiate.webp", magnitude / magnitude.max(), cmap=DERIVATIVE)
    print(f"Rendered three SplineOps illustration panels in {DESTINATION}")


if __name__ == "__main__":
    main()
