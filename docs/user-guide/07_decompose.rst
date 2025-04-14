Decompose
=========

.. currentmodule:: splineops

Overview
--------
This module provides spline-based multiresolution
decomposition for signals and images, to reduce (downsample)
and expand (upsample) data via spline interpolation, forming the basis for
pyramid and wavelet transforms [1]_. [2]_. [3]_, [4]_, [5]_.

Spline Representation
---------------------

A 1D discrete signal :math:`\{f[k]\}` can be modeled as a continuous function:

.. math::

   f(x) \;=\; \sum_{k} c[k]\, \phi\bigl(x - k\bigr),

where :math:`\phi(x)` is a polynomial spline basis function (e.g. B-spline of degree 3),
and :math:`c[k]` are the spline coefficients determined from the samples :math:`f[k]`.
This representation allows both downsampling and upsampling filters to be derived
directly from the spline model.

Pyramid Decomposition
---------------------

From this spline framework, two key operators are derived:

- REDUCE: filters the signal (or image) and downsamples by factor 2,
  producing a coarser approximation.
- EXPAND: upsamples and interpolates the coarser approximation back to
  the original resolution.

When applied iteratively, these operations create a pyramid structure
(approximation at multiple scales). In 2D, the same concept applies along rows
and columns, often with optional “centered” filtering to handle half-sample offsets.

Wavelet Decomposition
---------------------

Wavelet transforms extend the idea of the pyramid by also tracking the
detail that is lost at each reduction step. At each scale:

- An approximation is obtained (the reduced signal or image).
- A corresponding detail or wavelet sub-band is formed (the difference or “error” relative to the expanded approximation).

Repeating this decomposition over multiple scales yields a full wavelet
representation, where reconstruction (synthesis) uses the stored approximation
plus the detail coefficients.

Implementation Details
----------------------

- Reduce and expand features perform the core downsampling and upsampling based on spline filters.
- HaarWavelets, SplineWavelets, etc. implement wavelet transforms (analysis
  and synthesis) by combining pyramid steps with detail sub-bands.
- Various spline degrees (e.g. degree 3) are supported for smoother or sharper
  approximations.

Example
-------

- :ref:`sphx_glr_auto_examples_009_using_decompose_module.py`

References
----------

.. [1] Unser, M.,
  Splines: A Perfect Fit for Signal and Image Processing.  
  IEEE Signal Processing Magazine, 16 (6): 22–38, 1999.

.. [2] Unser, M., Aldroubi, A., & Eden, M.,
  B-Spline Signal Processing: Part II – Efficient Design and Applications*. IEEE Transactions on Signal  
  Processing, 41 (2): 834–848, 1993

.. [3] Unser, M., Aldroubi, A., & Eden, M.,
  The L2-Polynomial Spline Pyramid,
  IEEE Transactions on Pattern Analysis and Machine Intelligence, 15 (4): 364–379, 1993.

.. [4] Brigger, P., Müller, F., Illgner, K., & Unser, M. 
  Centered Pyramids,
  IEEE Transactions on Image Processing, 8 (9): 1254–1264, 1999.

.. [5] Burt, P. J., & Adelson, E. H.,
  The Laplacian Pyramid as a Compact Code,
  IEEE Transactions on Communication, 31 (4): 337–345, 1983
