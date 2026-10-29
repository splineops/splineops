# splineops/src/splineops/utils/__init__.py

"""
splineops.utils
===============

Utility subpackages used primarily by the example gallery and tutorials.

To keep the API clear and avoid a giant flat namespace, prefer importing
from the concrete submodules directly, for example:

    from splineops.utils.image import adjust_size_for_zoom
    from splineops.utils.metrics import compute_snr_and_mse_region
    from splineops.utils.plotting import show_roi_zoom
    from splineops.utils.diagram import draw_standard_vs_scipy_pipeline
    from splineops.utils.specs import print_runtime_context

Only the submodules themselves are exported from ``splineops.utils``.
"""

from __future__ import annotations

from importlib import import_module

__all__ = ["image", "metrics", "plotting", "diagram", "specs", "sphinx"]


def __getattr__(name: str):
    # Metrics and array helpers must not initialize Matplotlib or import gallery
    # dependencies. Preserve package-level submodule access by loading on demand.
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
