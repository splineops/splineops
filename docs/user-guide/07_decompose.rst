.. splineops/docs/user-guide/07_decompose.rst

Decompose
=========

.. currentmodule:: splineops

Overview
--------
This module provides spline-based multiresolution
decomposition of signals and images. It allows one to reduce (downsample)
and expand (upsample) data via spline interpolation and provides an important building 
block for pyramid and wavelet transforms [1]_. [2]_. [3]_, [4]_.

Spline Representation
---------------------

A 1D discrete signal :math:`\{f[k]\}` can be modeled as the continuous function

.. math::

   f(x) \;=\; \sum_{k} c[k]\, \phi\bigl(x - k\bigr),

where :math:`\phi(x)` is a polynomial spline basis function (e.g., a B-spline of degree 3),
and :math:`c[k]` are the spline coefficients determined from the samples :math:`f[k]`.
This representation allows the application of downsampling and upsampling filters
directly to the spline model.

Pyramid Decomposition
---------------------

Two key operators are proposed.

- Reduce: it filters the signal (or image) and downsamples by dyadic factors,
  thus producing a coarse approximation.
- Expand: it upsamples and interpolates the coarse approximation back to
  the original resolution.

When applied iteratively, these operations create a pyramid structure
(approximation at multiple scales). In 2D, the same concept applies along rows
and columns.

Wavelet Decomposition
---------------------

Wavelet transforms extend the idea of the pyramid by also tracking the
detail that is lost at each reduction step. At each scale (analysis)

- an approximation is obtained (the reduced signal or image);
- a corresponding detail or wavelet sub-band is formed (the difference or “error” relative to the expanded approximation).

The application of this decomposition over multiple scales yields a so-called wavelet
representation, where the stored approximation plus the detail coefficients can be used 
to perfectly reconstruct the original data (synthesis).

Implementation Details
----------------------

- Reduce and expand features perform the core downsampling and upsampling based on spline filters.
- Wavelet transforms such as Haar wavelets or spline wavelets (analysis and synthesis) are implemented by the combination of pyramid steps with detail sub-bands.
- Various spline degrees (e.g., degree 3) are supported. They allow one to control how the data are dispatched in the approximation channel and the sub-bands.

Decompose Examples
------------------

* :ref:`sphx_glr_auto_examples_08_decompose_08_01_pyramid_decomposition.py`
* :ref:`sphx_glr_auto_examples_08_decompose_08_02_wavelet_decomposition.py`

References
----------

.. [1] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22–38, November 1999.

.. [2] M. Unser, A. Aldroubi, M. Eden, 
   `B-Spline Signal Processing: Part II—Efficient Design and Applications <https://doi.org/10.1109/78.193221>`_, 
   IEEE Transactions 
   on Signal Processing, vol. 41, no. 2, pp. 834–848, February 1993.

.. [3] M. Unser, A. Aldroubi, M. Eden, 
   `The L2-Polynomial Spline Pyramid <https://doi.org/10.1109/34.206956>`_, 
   IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 15,
   no. 4, pp. 364–379, April 1993.

.. [4] P. Brigger, F. Müller, K. Illgner, M. Unser, 
   `Centered Pyramids <https://doi.org/10.1109/83.784437>`_, 
   IEEE Transactions on Image Processing, vol. 8, no. 9, pp. 1254–1264,
   September 1999.
