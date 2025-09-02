.. splineops/docs/api/07_decompose.rst

.. _api-decompose:

Decompose
=========

The :mod:`splineops.decompose` package provides functionality for **pyramid decomposition** (downsampling/up-sampling) and **wavelet transforms** (analysis and synthesis) based on spline models.

Pyramid Module
--------------
The :mod:`splineops.decompose.pyramid` module implements 1D and 2D REDUCE/EXPAND operations with mirror boundary handling. These form the foundation for many wavelet constructions.

Key functionalities:

- **get_pyramid_filter**  
  Retrieve filter coefficients (g for REDUCE, h for EXPAND) plus a “centered” flag.

- **reduce_1d**, **expand_1d**  
  Down/upsample a 1D signal using the specified filters and mirror reflection.

- **reduce_2d**, **expand_2d**  
  Down/upsample a 2D image row-by-row and column-by-column.

Example (pyramid usage)
-----------------------
.. code-block:: python

    from splineops.decompose.pyramid import get_pyramid_filter, reduce_1d, expand_1d
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
.. automodule:: splineops.decompose.pyramid
   :members:
   :member-order: bysource

Wavelet Modules
---------------
The :mod:`splineops.decompose.wavelets` subpackage provides various wavelet transforms (Haar, spline-based) using row-column (or column-row) passes. Classes typically define:

- **analysis** (multi-scale forward transform)
- **synthesis** (multi-scale inverse transform)

Submodules:

- :mod:`splineops.decompose.wavelets.abstractwavelets` – Base `AbstractWavelets` class
- :mod:`splineops.decompose.wavelets.haar` – Pure 2D Haar wavelets (`HaarWavelets`)
- :mod:`splineops.decompose.wavelets.splinewavelets` – Spline wavelets, e.g. `Spline1Wavelets`, `Spline3Wavelets`, etc.
- :mod:`splineops.decompose.wavelets.splinewaveletstool` – Low-level “splitMirror” & “mergeMirror” routines, plus `SplineWaveletsTool` class

API Reference: Wavelets
-----------------------
.. automodule:: splineops.decompose.wavelets.abstractwavelets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: splineops.decompose.wavelets.haar
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: splineops.decompose.wavelets.splinewavelets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: splineops.decompose.wavelets.splinewaveletstool
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: analysis1, synthesis1

