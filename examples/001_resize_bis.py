import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom  # For comparison if needed
from splineops.interpolate.resize import resize  # Unified resize function
import requests
from io import BytesIO
from PIL import Image

# Load the 'kodim19.png' image
url = 'https://r0k.us/graphics/kodak/kodak/kodim19.png'
response = requests.get(url)
img = Image.open(BytesIO(response.content))
data = np.array(img, dtype=np.float32)

# Normalize the image to [0,1]
data_normalized = data / 255.0

# Example parameters
output_size = (data_normalized.shape[0] * 2, data_normalized.shape[1] * 2)  # Double the size
extension_mode = "mirror"   # Try zero mode
degree = 3                # Cubic spline

# Resize each channel separately using our custom resize function
resized_channels = []
for channel in range(data_normalized.shape[2]):
    resized_channel = resize(
        data_normalized[:, :, channel],
        output_size=output_size,
        degree=degree,
        modes=extension_mode,
        method="interpolation"
    )
    resized_channels.append(resized_channel)

resized_image_float = np.stack(resized_channels, axis=-1)

# Check for NaNs or Infs before converting back
if np.isnan(resized_image_float).any():
    print("Warning: NaNs detected in the resized image.")
if np.isinf(resized_image_float).any():
    print("Warning: Infs detected in the resized image.")

# Ensure the values are in [0,1] range before scaling back
resized_image_float = np.clip(resized_image_float, 0.0, 1.0)

# Convert back to [0,255] and then to uint8
resized_image = (resized_image_float * 255.0).astype(np.uint8)

# Visualize the resized image
plt.figure(figsize=(5, 5))
plt.imshow(resized_image)
plt.title("Resized RGB Image (Doubled)")
plt.axis('off')
plt.show()


# -------------------------------------------------------------------------
# Additional Example: Downscale with a zoom factor of 0.3 and degree=3
# -------------------------------------------------------------------------
zoom_factor_2 = 0.3
resized_channels_zoom_2 = []
for channel in range(data_normalized.shape[2]):
    resized_channel = resize(
        data_normalized[:, :, channel],
        zoom_factors=zoom_factor_2,
        degree=degree,
        modes=extension_mode,
        method="interpolation"
    )
    resized_channels_zoom_2.append(resized_channel)

resized_zoom_2_float = np.stack(resized_channels_zoom_2, axis=-1)

if np.isnan(resized_zoom_2_float).any():
    print("Warning: NaNs detected in the zoom=0.3 resized image.")
if np.isinf(resized_zoom_2_float).any():
    print("Warning: Infs detected in the zoom=0.3 resized image.")

resized_zoom_2_float = np.clip(resized_zoom_2_float, 0.0, 1.0)
resized_zoom_2 = (resized_zoom_2_float * 255.0).astype(np.uint8)

plt.figure(figsize=(5, 5))
plt.imshow(resized_zoom_2)
plt.title("Degree 3, Zoom Factor 0.3")
plt.axis('off')
plt.show()


# -------------------------------------------------------------------------
# Additional Example: Upscale with a zoom factor of 2.5 and degree=3
# -------------------------------------------------------------------------
zoom_factor_3 = 2.5
resized_channels_zoom_3 = []
for channel in range(data_normalized.shape[2]):
    resized_channel = resize(
        data_normalized[:, :, channel],
        zoom_factors=zoom_factor_3,
        degree=degree,
        modes=extension_mode,
        method="interpolation"
    )
    resized_channels_zoom_3.append(resized_channel)

resized_zoom_3_float = np.stack(resized_channels_zoom_3, axis=-1)

if np.isnan(resized_zoom_3_float).any():
    print("Warning: NaNs detected in the zoom=2.5 resized image.")
if np.isinf(resized_zoom_3_float).any():
    print("Warning: Infs detected in the zoom=2.5 resized image.")

resized_zoom_3_float = np.clip(resized_zoom_3_float, 0.0, 1.0)
resized_zoom_3 = (resized_zoom_3_float * 255.0).astype(np.uint8)

plt.figure(figsize=(5, 5))
plt.imshow(resized_zoom_3)
plt.title("Degree 3, Zoom Factor 2.5")
plt.axis('off')
plt.show()
