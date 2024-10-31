# splineops/utils/image_loader.py
import importlib.resources as pkg_resources
import numpy as np
from PIL import Image

def load_head_mri_image():
    """Load the head MRI image as a numpy array."""
    # Load the image as a binary stream from the `data` directory
    with pkg_resources.open_binary('splineops.data', 'headMRI_256x256.png') as file:
        image = Image.open(file)
        return np.array(image)
