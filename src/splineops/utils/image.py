import numpy as np

__all__ = ["crop_to_central_region"]

def crop_to_central_region(img: np.ndarray, frac: float) -> np.ndarray:
    """Crop away `frac` of width/height on every side."""
    h, w = img.shape[:2]
    top, left = int(h * frac), int(w * frac)
    return img[top : h - top, left : w - left]
