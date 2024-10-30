import numpy as np
import matplotlib.pyplot as plt
from splineops.interpolate.resize import resize
from splineops.interpolate.ls_oblique.ls_oblique_resize import LS_Oblique_Resize
from splineops.interpolate.tensorspline import TensorSpline

# Generate a sample 2D image (e.g., simple gradient or synthetic data)
image = np.linspace(0, 1, 100).reshape(10, 10)

# Parameters for resizing
zoom_factors = (2, 2)  # Scale by 2 in both dimensions
degrees = [1, 3]       # Degrees to compare

# Set up subplots
fig, axes = plt.subplots(3, len(degrees), figsize=(10, 6))
fig.suptitle("Comparison of Resizing Methods")

for i, degree in enumerate(degrees):
    # Tensor Spline Interpolation
    ts_resized = resize(image, zoom_factors=zoom_factors, degree=degree)
    axes[0, i].imshow(ts_resized, cmap="viridis")
    axes[0, i].set_title(f"Tensor Spline Degree {degree}")
    axes[0, i].axis("off")
    
    # LS Interpolation
    ls_resizer = LS_Oblique_Resize()
    ls_resized = np.zeros((image.shape[0] * zoom_factors[0], image.shape[1] * zoom_factors[1]))
    ls_resizer.compute_zoom(image, ls_resized, analy_degree=degree, synthe_degree=degree,
                            interp_degree=degree, zoom_factors=zoom_factors, shifts=(0, 0), inversable=False)
    axes[1, i].imshow(ls_resized, cmap="viridis")
    axes[1, i].set_title(f"LS Resize Degree {degree}")
    axes[1, i].axis("off")
    
    # Oblique Interpolation
    oblique_resizer = LS_Oblique_Resize()
    oblique_resized = np.zeros((image.shape[0] * zoom_factors[0], image.shape[1] * zoom_factors[1]))
    oblique_resizer.compute_zoom(image, oblique_resized, analy_degree=degree+1, synthe_degree=degree-1,
                                 interp_degree=degree, zoom_factors=zoom_factors, shifts=(0, 0), inversable=False)
    axes[2, i].imshow(oblique_resized, cmap="viridis")
    axes[2, i].set_title(f"Oblique Resize Degree {degree}")
    axes[2, i].axis("off")

# Display the comparison plot
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()
