import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
from PIL import Image

from scipy.ndimage import zoom as ndi_zoom
from splineops.interpolate.resize import resize  # or your actual import path

# ------------------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------------------

def adjust_image_size_for_shrink(data, shrink_factor):
    """
    Adjust the image size so that after shrinking and re-expanding,
    the final dimensions match this 'adjusted image' exactly.

    Steps:
      1) We want H' * shrink_factor to be integer (same for W').
      2) Choose H' as nearest multiple of 1/shrink_factor to original H, similarly for W'.
      3) Use ndimage.zoom to resample to (H', W').
    """
    h, w = data.shape[:2]
    c = data.shape[2] if data.ndim == 3 else 1

    inv_s = 1.0 / shrink_factor

    # Determine new height
    new_h = round(round(h / inv_s) * inv_s)
    # Determine new width
    new_w = round(round(w / inv_s) * inv_s)

    new_h = max(new_h, 1)
    new_w = max(new_w, 1)

    # Scale factors for ndimage.zoom
    scale_factor_h = new_h / h
    scale_factor_w = new_w / w

    # Zoom the image (assume color image has shape (H, W, 3))
    if c == 1:
        data_zoomed = ndi_zoom(data, (scale_factor_h, scale_factor_w), order=1)
    else:
        data_zoomed = ndi_zoom(data, (scale_factor_h, scale_factor_w, 1), order=1)

    return data_zoomed

def resize_image_splineops(data, zoom_factor, degree=3, extension_mode="mirror"):
    """
    Wrapper around splineops' resize function for a 3-channel (RGB) image.
    Assumes 'data' is float64 in [0, 1].
    """
    resized_channels = []
    for ch in range(data.shape[2]):
        resized_ch = resize(
            data[:, :, ch],
            zoom_factors=zoom_factor,
            degree=degree,
            modes=extension_mode,
            method="interpolation"
        )
        resized_channels.append(resized_ch)

    resized_image = np.stack(resized_channels, axis=-1)
    # Clip to [0,1], then convert to uint8
    resized_image = np.clip(resized_image, 0.0, 1.0)
    return (resized_image * 255.0).astype(np.uint8)

# ------------------------------------------------------------------------------
# Main code
# ------------------------------------------------------------------------------

# 1) Load and normalize the original image
url = 'https://r0k.us/graphics/kodak/kodak/kodim19.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
data = np.array(img, dtype=np.float64)  # shape: (H, W, 3)
data_normalized = data / 255.0          # Convert to [0,1]

# 2) Adjust image so that shrinking and then re-expanding yields the same final shape
shrink_factor = 0.23
adjusted_data = adjust_image_size_for_shrink(data_normalized, shrink_factor)
adjusted_data_uint8 = (adjusted_data * 255).astype(np.uint8)

# 3) Shrink the adjusted image
shrunken_image = resize_image_splineops(
    adjusted_data,
    zoom_factor=shrink_factor,
    degree=3,
    extension_mode="mirror"
)

# 4) Place the shrunken image onto a white canvas matching adjusted image size (so it appears smaller)
H_adj, W_adj, _ = adjusted_data_uint8.shape
canvas_shrunken = np.ones((H_adj, W_adj, 3), dtype=np.uint8) * 255  # white background
H_shr, W_shr, _ = shrunken_image.shape
canvas_shrunken[:H_shr, :W_shr, :] = shrunken_image

# 5) Expand the shrunken image back to original (adjusted) dimensions
expanded_image = resize_image_splineops(
    shrunken_image.astype(np.float64) / 255.0,  # re-normalize to [0,1]
    zoom_factor=1.0 / shrink_factor,
    degree=3,
    extension_mode="mirror"
)

# 6) Plot the three images in one figure: 
#    (Left) Adjusted original, (Center) Shrunken-on-canvas, (Right) Expanded
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].imshow(adjusted_data_uint8)
axes[0].set_title("Adjusted Original")
axes[0].axis("off")

axes[1].imshow(canvas_shrunken)
axes[1].set_title(f"Shrunken Image (x{shrink_factor})")
axes[1].axis("off")

axes[2].imshow(expanded_image)
axes[2].set_title(f"Expanded Image (x{1/shrink_factor:.2f})")
axes[2].axis("off")

plt.tight_layout()
plt.show()
