"""
Illustrative speed-vs-quality trade-off for splineops resize methods
--------------------------------------------------------------------

The dotted line is a quadratic that passes *exactly* through the three
method points:

    • Standard interpolation   – fastest, lowest quality
    • Oblique projection       – mid-speed, mid-quality
    • Least-Squares projection – slowest, best quality
"""

import numpy as np
import matplotlib.pyplot as plt

# ── raw data points ────────────────────────────────────────────────────
speed_pts   = np.array([0.90, 0.50, 0.10])   # Standard, Oblique, LS
quality_pts = np.array([0.20, 0.60, 0.90])

method_info = {
    "Standard":      dict(s=0.90, q=0.20, color="#2ca02c"),
    "Oblique":       dict(s=0.50, q=0.60, color="#ff7f0e"),
    "Least-Squares": dict(s=0.10, q=0.90, color="#1f77b4"),
}

# ── fit a quadratic y = ax² + bx + c that goes through the three points ─
coeff = np.polyfit(speed_pts, quality_pts, deg=2)   # exact for 3 points
speed_curve = np.linspace(0, 1, 300)
quality_curve = np.polyval(coeff, speed_curve)

# ── plot ───────────────────────────────────────────────────────────────
plt.figure(figsize=(7, 5))

plt.plot(speed_curve, quality_curve, linestyle="--", color="0.6",
         label="Illustrative trade-off")

for name, d in method_info.items():
    plt.scatter(d["s"], d["q"], s=140, color=d["color"],
                edgecolor="k", zorder=3)
    plt.text(d["s"] + 0.03, d["q"] + 0.02, name,
             fontsize=11, weight="bold")

plt.xlim(0, 1); plt.ylim(0, 1)
plt.xlabel("Speed   (1 = fastest)")
plt.ylabel("Quality (1 = best)")
plt.title("Resize Method Trade-off in splineops")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
