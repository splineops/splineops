.. splineops/docs/user-guide/07_multiscale.rst

Multiscale
==========

.. currentmodule:: splineops

Overview
--------
This module models signals and images as a hierarchy of spline approximations at progressively coarser resolutions. 
On top of this model, it provides reduction (spline filtering + dyadic decimation) and expansion 
(upsampling + spline interpolation), which are the building blocks of pyramid and wavelet transforms [1]_, [2]_, [3]_, [4]_.

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

Two key operators are proposed:

- Reduce: it filters the signal (or image) and downsamples by dyadic factors,
  thus producing a coarse approximation.
- Expand: it upsamples and interpolates the coarse approximation back to
  the original resolution.

When applied iteratively, these operations create a pyramid structure
(approximation at multiple scales). In 2D, the same concept applies along rows
and columns.

Subpixel Registration
~~~~~~~~~~~~~~~~~~~~~

Method [5]_ does subpixel registration as least-squares matching of image intensities under a global transform 
(affine, optionally restricted to rigid/similarity) with an optional contrast change, and solves it with a modified 
Levenberg-Marquardt optimizer.

Its key ingredient is a spline pyramid: starting from a dyadic, least-squares fine-to-coarse 
decomposition built with cubic-spline filtering, the optimizer estimates the transform at the coarsest level and propagates
the parameters down the pyramid so that finer levels only apply small corrections. The spline model (cubic) is used consistently for
resampling and for computing exact spatial derivatives, which further stabilizes the coarse-to-fine.

A compact Python implementation following this approach is available in the GitHub repository https://github.com/glichtner/pystackreg.

Wavelet Decomposition
---------------------

We construct a spline-based multiscale basis (wavelets) by capturing the detail lost at each reduction step. 
The wavelet (detail) coefficients together with the final coarse approximation allow perfect reconstruction (synthesis).

At each scale (analysis):

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

Multiscale Examples
-------------------

* :ref:`sphx_glr_auto_examples_08_multiscale_08_01_pyramid_decomposition.py`
* :ref:`sphx_glr_auto_examples_08_multiscale_08_02_wavelet_decomposition.py`

References
----------

.. [1] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22-38, November 1999.

.. [2] M. Unser, A. Aldroubi, M. Eden, 
   `B-Spline Signal Processing: Part II—Efficient Design and Applications <https://doi.org/10.1109/78.193221>`_, 
   IEEE Transactions 
   on Signal Processing, vol. 41, no. 2, pp. 834-848, February 1993.

.. [3] M. Unser, A. Aldroubi, M. Eden, 
   `The L2-Polynomial Spline Pyramid <https://doi.org/10.1109/34.206956>`_, 
   IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 15,
   no. 4, pp. 364-379, April 1993.

.. [4] P. Brigger, F. Müller, K. Illgner, M. Unser, 
   `Centered Pyramids <https://doi.org/10.1109/83.784437>`_, 
   IEEE Transactions on Image Processing, vol. 8, no. 9, pp. 1254-1264,
   September 1999.

.. [5] P. Thevenaz, U. E. Ruttimann, M. Unser, 
   `A Pyramid Approach to Subpixel Registration Based on Intensity <https://doi.org/10.1109/83.650848>`_, 
   IEEE Transactions on Image Processing, vol. 7, no. 1, pp. 27-41,
   January 1998.
