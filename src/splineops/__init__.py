# splineops/src/splineops/__init__.py

from .resize.resize import ResizePlan, resize, resize_degrees
from .spline_interpolation.tensor_spline import TensorSpline

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("splineops")
except PackageNotFoundError:
    __version__ = (
        "0.0.0+local"  # used only if someone imports from source without installing
    )

__all__ = [
    "ResizePlan",
    "TensorSpline",
    "resize",
    "resize_degrees",
    "__version__",
]
