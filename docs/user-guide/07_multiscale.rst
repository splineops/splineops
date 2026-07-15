.. splineops/docs/user-guide/07_multiscale.rst

Multiscale
==========

.. currentmodule:: splineops

Overview
--------
The :ref:`multiscale <api-multiscale>` module in :ref:`SplineOps <api-index>` models signals and images as a hierarchy of spline approximations at progressively coarser resolutions. 
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
For the explicitly supported shape and scale combinations, the wavelet
(detail) coefficients together with the final coarse approximation allow
perfect reconstruction (synthesis).

The next figure, from
:ref:`sphx_glr_auto_examples_07_multiscale_02_wavelet_decomposition.py`,
shows a three-level 2D Haar decomposition: the coarse approximation in the
top-left corner and the horizontal, vertical, and diagonal detail sub-bands
at each scale.

.. image:: /auto_examples/07_multiscale/images/sphx_glr_02_wavelet_decomposition_001.png
   :align: center
   :width: 100%

At each scale (analysis):

- an approximation is obtained (the reduced signal or image);
- a corresponding detail or wavelet sub-band is formed (the difference or “error” relative to the expanded approximation).

The application of this decomposition over multiple scales yields a so-called wavelet
representation, where the stored approximation plus the detail coefficients can be used 
to perfectly reconstruct the original data (synthesis).

Implementation Details
----------------------

- Reduce and expand perform the core downsampling and upsampling with spline
  filters.  Whole-array axis operations replace row-by-row and column-by-column
  Python dispatch.
- Haar split/merge operations are vectorized over complete scale regions.
  Spline wavelets vectorize over samples and image axes while retaining short,
  explicit loops over filter taps.
- Various spline degrees (e.g., degree 3) control how data are dispatched
  between the approximation channel and sub-bands.

Supported shapes and boundaries
-------------------------------

The pyramid functions accept finite, non-empty real 1-D signals and 2-D arrays
and use their documented mirror mappings.  Floating inputs preserve their
precision; integer inputs promote to float64, and booleans are rejected.
Reducing an odd length returns
``floor(n / 2)`` samples; expanding that result therefore does not recover the
dropped extent.  A singleton is preserved exactly.

The multiscale Haar and spline-wavelet classes currently support non-empty 2D
arrays whose two dimensions are divisible by ``2**scales``.  Unsupported odd
or too-small scale regions are rejected instead of silently losing samples.
Perfect reconstruction is tested for even square and rectangular arrays for
Haar and the cubic spline-wavelet implementation.  The order-1 spline filter
reconstructs the audited float64 cases within ``2e-7``.  The inherited order-5
filter coefficients contain only roughly five to six significant digits and
produce errors up to about ``2e-3`` in the current randomized rectangular
audit.  Order 5 is therefore an approximate research implementation, not a
perfect-reconstruction transform.  The test suite records that limitation so
it cannot silently become a stronger claim.

These APIs never infer batch or channel dimensions.  ``reduce_2d``,
``expand_2d``, and multi-scale wavelet ``analysis``/``synthesis`` accept
explicit ``spatial_axes`` and transform every remaining slice independently.
The divisibility and reconstruction contracts apply only to selected axes.
See :doc:`../performance` for the reproducible row/column-oracle comparison and
:doc:`../consolidation-recipes` for a batched example.

Multiscale Examples
-------------------

* :ref:`sphx_glr_auto_examples_07_multiscale_01_pyramid_decomposition.py`
* :ref:`sphx_glr_auto_examples_07_multiscale_02_wavelet_decomposition.py`

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
