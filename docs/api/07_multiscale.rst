.. splineops/docs/api/07_multiscale.rst

.. _api-multiscale:

Multiscale
==========

This module provides functionality for **pyramid decomposition** (downsampling/up-sampling) and **wavelet transforms** (analysis and synthesis) based on spline models.

Pyramid Module
--------------

The :mod:`splineops.multiscale.pyramid` module implements 1-D and 2-D
REDUCE/EXPAND operations with mirror boundary handling.  Two-dimensional axis
passes operate on whole selected axes rather than dispatching a Python call for
each row or column.  Explicit ``spatial_axes`` preserve and independently
process any remaining batch or channel dimensions.  These operations form the
foundation for many wavelet constructions.

Reduction uses ``floor(n / 2)`` for odd lengths, so pyramid reduce/expand is
not a reversible shape operation in that case.  Wavelet analysis is stricter:
both dimensions must be divisible by ``2**scales``.

Perfect reconstruction is established for Haar and the cubic spline filter on
the tested even rectangular shapes.  Order 1 has small truncation error and
order 5 is approximate because its inherited taps have limited precision; see
the multiscale user guide for the measured bounds.

Inputs must be finite, real, and non-empty.  Floating inputs preserve their
precision; integer inputs promote to float64.  Scalar images need no axis
argument.  Higher-rank arrays require two explicit spatial axes; dimensions are
never inferred as batch or channel merely from rank.

Key functionalities:

- **get_pyramid_filter**  
  Retrieve filter coefficients (g for REDUCE, h for EXPAND) plus a “centered” flag.

- **reduce_1d**, **expand_1d**  
  Down/upsample a 1D signal using the specified filters and mirror reflection.

- **reduce_2d**, **expand_2d**  
  Down/upsample two selected image axes with vectorized separable passes.

Example (pyramid usage)
-----------------------

.. code-block:: python

    from splineops.multiscale.pyramid import get_pyramid_filter, reduce_1d, expand_1d
    import numpy as np

    # Retrieve filter for a spline of order 3
    g, h, centered = get_pyramid_filter("Spline", 3)

    # 1D reduce/expand
    x = np.array([0,1,2,3,2,1,0,-2,-4,-6], dtype=float)
    x_reduced = reduce_1d(x, g, centered)
    x_expanded = expand_1d(x_reduced, h, centered)

    print("Original:", x)
    print("Reduced: ", x_reduced)
    print("Expanded:", x_expanded)

    # 2D reduce/expand
    arr = np.random.rand(8,8).astype(np.float32)
    arr_reduced = reduce_2d(arr, g, centered)
    arr_expanded = expand_2d(arr_reduced, h, centered)

API Reference: Pyramid
----------------------

.. automodule:: splineops.multiscale.pyramid
   :members:
   :member-order: bysource

Wavelet Modules
---------------

The :mod:`splineops.multiscale.wavelets` subpackage provides Haar and
spline-based wavelet transforms using whole-axis separable passes.  Their
multi-scale methods also accept two explicit ``spatial_axes`` for batched
arrays.  Small planes use cache-sized vectorized groups; large planes avoid
whole-array transpose copies through direct independent dispatch.  Classes
typically define:

- **analysis** (multi-scale forward transform)
- **synthesis** (multi-scale inverse transform)

Submodules:

- :mod:`splineops.multiscale.wavelets.abstract_wavelets` - Base `AbstractWavelets` class
- :mod:`splineops.multiscale.wavelets.haar` - Pure 2D Haar wavelets (`HaarWavelets`)
- :mod:`splineops.multiscale.wavelets.spline_wavelets` - Spline wavelets, e.g. `Spline1Wavelets`, `Spline3Wavelets`, etc.
- :mod:`splineops.multiscale.wavelets.spline_wavelets_tool` - Low-level “splitMirror” & “mergeMirror” routines, plus `SplineWaveletsTool` class

API Reference: Wavelets
-----------------------
.. automodule:: splineops.multiscale.wavelets.abstract_wavelets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: splineops.multiscale.wavelets.haar
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: splineops.multiscale.wavelets.spline_wavelets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: splineops.multiscale.wavelets.spline_wavelets_tool
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: analysis1, synthesis1
