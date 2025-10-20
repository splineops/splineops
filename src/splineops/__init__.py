# splineops/src/splineops/__init__.py

import importlib.metadata
from .resize.resize import resize

__all__ = ["resize", "__version__"]
__version__ = importlib.metadata.version("splineops")