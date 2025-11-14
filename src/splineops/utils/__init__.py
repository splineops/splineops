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

from . import image, metrics, plotting, diagram, specs

__all__ = ["image", "metrics", "plotting", "diagram", "specs"]
