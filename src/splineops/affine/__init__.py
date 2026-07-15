# splineops/src/splineops/affine/__init__.py

"""
Affine geometric transforms and reusable fixed-geometry plans.

Main entry point:
    - :class:`splineops.affine.AffinePlan`
    - :func:`splineops.affine.affine_transform`
    - :func:`splineops.affine.rotate`
"""

from .affine import AffinePlan, affine_transform, rotate

__all__ = ["AffinePlan", "affine_transform", "rotate"]
