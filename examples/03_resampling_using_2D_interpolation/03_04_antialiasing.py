# -*- coding: utf-8 -*-
"""
Antialiasing A/B demo (Option A: trim to common even area for targets/metrics)

- Build "mixed" image where each 2×2 block has A at the top-left pixel and B elsewhere.
- Crop to ODD H×W so a 0.5× interpolation grid lands exactly on (0,0) block corners.
- Downsample the mixed image with:
    (1) standard/cubic interpolation  → should yield A[0::2, 0::2]
    (2) least-squares/cubic-best_antialiasing → should yield ~ 2×2 box-average
- Compute SNR/MSE on the largest interior EVEN area (common shape for 2×2 slices),
  and crop the downsampled results to match those target sizes.

This script is self-contained except for `splineops`. It only needs NumPy, PIL, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

from splineops.resize.resize import resize  # your library


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def to_gray01(img_rgb_uint8: np.ndarray) -> np.ndarray:
    """RGB uint8 → grayscale float64 in [0,1]."""
    g = img_rgb_uint8.astype(np.float64) / 255.0
    return 0.2989 * g[..., 0] + 0.5870 * g[..., 1] + 0.1140 * g[..., 2]


def snr_mse(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Return (SNR[dB], MSE) for arrays of same shape."""
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    signal = np.mean(x ** 2)
    noise = np.mean((x - y) ** 2)
    snr = float("inf") if noise <= 1e-30 else 10.0 * np.log10(signal / noise)
    return snr, noise


def imshow_row(imgs, titles, *, vmin=0.0, vmax=1.0, cmap="gray", h=3.6):
    """Quick side-by-side viewer."""
    n = len(imgs)
    H, W = imgs[0].shape
    fig, axes = plt.subplots(1, n, figsize=(h * n, h * H / W))
    if n == 1:
        axes = [axes]
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(t)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# 1) Load A and B, build mixed image (A corners, B elsewhere)
# --------------------------------------------------------------------------- #

url_a = "https://r0k.us/graphics/kodak/kodak/kodim14.png"
url_b = "https://r0k.us/graphics/kodak/kodak/kodim08.png"

A = to_gray01(np.array(Image.open(BytesIO(requests.get(url_a, timeout=10).content))))
B = to_gray01(np.array(Image.open(BytesIO(requests.get(url_b, timeout=10).content))))

assert A.shape == B.shape, "A and B must have identical shape."
H, W = A.shape

mixed = B.copy()
mixed[0::2, 0::2] = A[0::2, 0::2]

imshow_row([A, B, mixed], ["A (gray)", "B (gray)", "A/B corner mix"])


# --------------------------------------------------------------------------- #
# 2) Crop to ODD size so 0.5× interpolation lands on (0,0) corners
# --------------------------------------------------------------------------- #

if (H % 2 == 0) or (W % 2 == 0):
    mixed_odd = mixed[:H - (H % 2 == 0), :W - (W % 2 == 0)]
    A_odd = A[:mixed_odd.shape[0], :mixed_odd.shape[1]]
    B_odd = B[:mixed_odd.shape[0], :mixed_odd.shape[1]]
else:
    mixed_odd = mixed
    A_odd = A
    B_odd = B

h, w = mixed_odd.shape
assert (h % 2 == 1) and (w % 2 == 1), "Expect odd H×W after the crop."


# --------------------------------------------------------------------------- #
# 3) Define targets on the LARGEST INTERIOR EVEN AREA (Option A)
#    This guarantees all 2×2 strata have the same shape.
# --------------------------------------------------------------------------- #

h2 = (h // 2) * 2  # largest even <= h
w2 = (w // 2) * 2  # largest even <= w

A_ev = A_odd[:h2, :w2]
B_ev = B_odd[:h2, :w2]
mixed_ev = mixed_odd[:h2, :w2]

# Targets:
# - Standard interpolation should match A corners at 0.5×
A_corners = A_ev[0::2, 0::2]

# - Least-squares (best AA) should approach a 2×2 box-average of the mixed tile:
#   (A at TL + B at TR + B at BL + B at BR) / 4
B_avg2x2 = 0.25 * (
    A_ev[0::2, 0::2] +
    B_ev[0::2, 1::2] +
    B_ev[1::2, 0::2] +
    B_ev[1::2, 1::2]
)

# These target sizes will be (h2//2, w2//2)
H_t, W_t = A_corners.shape


# --------------------------------------------------------------------------- #
# 4) Downsample the mixed image in two ways (from the ODD-sized image)
# --------------------------------------------------------------------------- #

zoom = (0.5, 0.5)

# (a) Standard cubic interpolation → should lock onto A corners
mixed_std_full = resize(mixed_odd, zoom_factors=zoom, method="cubic")
# (b) Least-squares cubic-best AA → should approximate the 2×2 box-average
mixed_ls_full = resize(mixed_odd, zoom_factors=zoom, method="cubic-best_antialiasing")

# Crop outputs to the target shape for fair comparison/metrics
mixed_std = mixed_std_full[:H_t, :W_t]
mixed_ls  = mixed_ls_full[:H_t,  :W_t]


# --------------------------------------------------------------------------- #
# 5) Metrics and visual checks
# --------------------------------------------------------------------------- #

snr_std, mse_std = snr_mse(mixed_std, A_corners)
snr_ls,  mse_ls  = snr_mse(mixed_ls,  B_avg2x2)

print(f"[STANDARD → A corners]     shape={mixed_std.shape}   SNR: {snr_std:6.2f} dB   MSE: {mse_std:.3e}")
print(f"[LEAST-SQUARES → box avg]  shape={mixed_ls.shape}    SNR: {snr_ls:6.2f} dB   MSE: {mse_ls:.3e}")

imshow_row(
    [A_corners, mixed_std],
    ["Target: A corners (0.5×)", "Result: Standard (cubic)"]
)

imshow_row(
    [B_avg2x2, mixed_ls],
    ["Target: 2×2 box-average", "Result: Least-squares (best AA)"]
)


# --------------------------------------------------------------------------- #
# 6) Optional: side-by-side of both downsampled outputs
# --------------------------------------------------------------------------- #

imshow_row(
    [mixed_std, mixed_ls],
    ["Standard (cubic) 0.5×", "Least-squares (cubic-best AA) 0.5×"]
)
