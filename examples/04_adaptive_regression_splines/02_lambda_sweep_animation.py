# sphinx_gallery_start_ignore
# splineops/examples/04_adaptive_regression_splines/02_lambda_sweep_animation.py
# sphinx_gallery_end_ignore

"""
Lambda Sweep Animation
======================

This example animates the fidelity–sparsity trade-off of the adaptive regression
splines module by sweeping the regularization parameter :math:`\\lambda`.

Each frame shows:
    - Original samples (fixed).
    - TV-denoised samples (changes with :math:`\\lambda`).
    - Sparsest piecewise-linear spline (changes with :math:`\\lambda`).
    - Detected knots (changes with :math:`\\lambda`).
"""

# %%
# Imports
# -------

import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# sphinx_gallery_thumbnail_number = 2 # show second figure as thumbnail
from splineops.adaptive_regression_splines.denoising import denoise_y
from splineops.adaptive_regression_splines.sparsification import (
    sparsest_interpolant,
    linear_spline,
)

# %%
# Data
# ----
#
# Embedded (x, y) dataset.

data = np.array([
    [0.0107766212868331, 0.260227935166001],
    [0.0310564395737153, 0.124128829346261],
    [0.0568406178471921, -0.0319625924377939],
    [0.0624834663023982, -0.305487158118621],
    [0.0855836735802228, 0.0198584921896104],
    [0.111715185429166, 0.132374842819488],
    [0.139391914966393, 0.0346881909310438],
    [0.151220604385114, 0.225044726396834],
    [0.160372945787459, -0.0839333693634482],
    [0.196012653453612, 0.100524891786437],
    [0.20465948547682, 0.286553206119747],
    [0.236142103912376, -0.0969023194982265],
    [0.247757212881283, 0.344416030734225],
    [0.277270837091189, 0.322338105021903],
    [0.294942432854744, 0.628493233708394],
    [0.311124804679808, 0.238685146788896],
    [0.322729104513214, 0.0314182350548619],
    [0.341198353790244, 0.554001442697049],
    [0.362426869114815, 0.658491185386012],
    [0.380891037570895, 0.622866466731061],
    [0.402149882582122, 0.832680133314763],
    [0.424514186772157, 0.282871329344068],
    [0.454259779607654, 0.418961645149398],
    [0.471194339641083, 0.765569673816136],
    [0.480251119603182, 0.936269519087734],
    [0.50143948559379, 1.21690697362292],
    [0.539345526600005, 0.856785149480669],
    [0.551362009238399, 0.918563536364133],
    [0.56406586469322, 1.04369945227154],
    [0.585046514891406, 0.891659244596406],
    [0.614876517081502, 1.01020862285029],
    [0.623908589622186, 1.05646068606692],
    [0.651627178545465, 1.13455056187785],
    [0.679400399781766, 1.56682321923577],
    [0.696936576029801, 1.47238622944877],
    [0.704796955182952, 1.20492985044235],
    [0.729875394285375, 1.45329058102288],
    [0.752399114367628, 1.26394538858847],
    [0.776579617991004, 1.4052754431186],
    [0.783135827892922, 1.2534612824523],
    [0.800371524043548, 1.58782975330571],
    [0.821400442874384, 1.3740621261335],
    [0.849726902218741, 2.27247168443063],
    [0.872126589233067, 1.98304748773148],
    [0.89137702874173, 1.59274379691406],
    [0.906347248186443, 1.82991582958117],
    [0.939772323088249, 1.9344157693364],
    [0.951594904384916, 1.71570985938051],
    [0.967602823452471, 2.32573940626424],
    [0.991018964382358, 2.11540602201059],
])

x, y = data[:, 0], data[:, 1]

fig, ax = plt.subplots()
ax.plot(x, y, "x", label="Original data", markersize=8)
ax.set_title("Original data")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True, alpha=0.3)
ax.legend()
plt.show()

# %%
# Animation Helper
# ----------------

def _solve_for_lambda(lamb: float):
    """
    Precompute all quantities needed to render one frame.
    Keeping the animation step light makes the HTML export fast and stable.
    """
    y_d = denoise_y(x, y, lamb, rho=lamb)
    knots, amplitudes, polynomial = sparsest_interpolant(x, y_d)

    margin = (x[-1] - x[0]) / 10.0
    t = np.linspace(x[0] - margin, x[-1] + margin, 400)
    y_s = linear_spline(t, knots, amplitudes, polynomial)

    if len(knots) > 0:
        y_k = linear_spline(knots, knots, amplitudes, polynomial)
        knot_xy = np.c_[knots, y_k]
    else:
        knot_xy = np.empty((0, 2))

    return {
        "lamb": float(lamb),
        "y_d": y_d,
        "t": t,
        "y_s": y_s,
        "knot_xy": knot_xy,
        "K": int(len(knots)),
    }


def create_lambda_sweep_animation(lambdas: np.ndarray, interval: int = 700):
    sols = [_solve_for_lambda(lmb) for lmb in lambdas]

    fig, ax = plt.subplots()
    ax.plot(x, y, "x", label="Original", markersize=8)

    # Initialize with first frame
    den_line, = ax.plot(
        x, sols[0]["y_d"], "x",
        label="Denoised", markersize=8,
        zorder=2,
    )

    spline_line, = ax.plot(
        sols[0]["t"], sols[0]["y_s"],
        label="Sparsest",
        zorder=1,
    )

    knot_scatter = ax.scatter(
        sols[0]["knot_xy"][:, 0],
        sols[0]["knot_xy"][:, 1],
        s=36,
        color="C3",
        marker="o",
        label="Knots",
        zorder=5,   # <- on top of everything
    )

    title = ax.set_title(f"λ = {sols[0]['lamb']:.2e}   |   K = {sols[0]['K']}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Fix axis limits to avoid flicker while animating
    margin = (x[-1] - x[0]) / 10.0
    ax.set_xlim(x[0] - margin, x[-1] + margin)

    y_all = np.concatenate([y] + [s["y_d"] for s in sols] + [s["y_s"] for s in sols])
    pad = 0.05 * (y_all.max() - y_all.min() + 1e-12)
    ax.set_ylim(y_all.min() - pad, y_all.max() + pad)

    def animate(i: int):
        s = sols[i]
        den_line.set_ydata(s["y_d"])
        spline_line.set_data(s["t"], s["y_s"])
        knot_scatter.set_offsets(s["knot_xy"])
        title.set_text(f"λ = {s['lamb']:.2e}   |   K = {s['K']}")
        return spline_line, den_line, knot_scatter, title

    return animation.FuncAnimation(
        fig,
        animate,
        frames=len(sols),
        interval=interval,
        blit=True,
    )


# %%
# Build the Animation
# -------------------
#
# Geometric spacing gives a nice “slow-to-fast” transition.

lambda_values = np.geomspace(1e-4, 2e-1, 12)

INTERVAL_MS = 750
ani = create_lambda_sweep_animation(lambda_values, interval=INTERVAL_MS)

# %%
# Export the Animation
# --------------------

from splineops.utils.sphinx import export_animation_mp4_and_html

export_animation_mp4_and_html(
    ani,
    stem="lambda_sweep_animation",
    interval_ms=INTERVAL_MS,  # e.g. 750
    dpi=80,
    force=True,
)