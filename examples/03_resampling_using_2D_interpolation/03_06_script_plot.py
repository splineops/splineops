import numpy as np
import matplotlib.pyplot as plt

points = [
    ("Least-Squares", 0.10, 0.90, "#1f77b4"),
    ("Oblique",       0.50, 0.60, "#ff7f0e"),
    ("Standard",      0.90, 0.20, "#2ca02c"),
]

fig, ax = plt.subplots(figsize=(7, 5))

for name, spd, qlt, clr in points:
    ax.scatter(spd, qlt, s=140, color=clr, edgecolor="k", zorder=3)
    ax.text(spd + 0.03, qlt + 0.03, name,
            fontsize=13, weight="bold")        # method label size

    # guide lines
    ax.plot([spd, spd], [0, qlt], ls="--", color=clr, alpha=0.7)
    ax.plot([0, spd], [qlt, qlt], ls="--", color=clr, alpha=0.7)

# Bigger, bold axis labels
ax.set_xlabel("Speed →", fontsize=14, fontweight="bold")
ax.set_ylabel("Quality ↑", fontsize=14, fontweight="bold")

# symbolic axes (no numbers)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks([]); ax.set_yticks([])
ax.grid(False)

ax.set_title("Trade-off Among splineops Resize Modes")
plt.tight_layout()
plt.show()
