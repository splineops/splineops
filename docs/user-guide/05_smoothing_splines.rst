.. splineops/docs/user-guide/05_smoothing_splines.rst

Smoothing Splines
=================

Overview
--------

The :ref:`smoothing splines <api-smoothing_splines>` module in :ref:`SplineOps <api-index>` provides tools 
to fit fractional smoothing splines to noisy data. Think of these splines as flexible low-pass filters whose sharpness can
be tuned continuously, making them effective for signals and images that
exhibit repeating, self-similar patterns.

You will find:

* Exact 1D routine:
  Works on a 1D array and returns the mathematically exact
  fractional-spline result.

* Isotropic N-D routine:
  Extends the idea to 2D pictures or 3D volumes through one FFT;
  internally it uses a Butterworth low-pass filter.

* Fast linear shortcut:
  A lightweight forward/backward IIR filter that implements the
  first-order (piecewise-linear) smoothing spline in linear time—handy
  for real-time streams.

* Extra helpers to generate test data
  (fractional Brownian motion) and to compute spline autocorrelations.

Core Idea in One Dimension
--------------------------

Given noisy samples :math:`y[k]` at integer positions, we look for a
smooth curve :math:`s(t)` that minimises

.. math::

   \sum_{k}\lvert y[k]-s(k)\rvert^{2}
   \;+\;
   \lambda\,\lVert\partial^{\gamma}s\rVert_{L^{2}}^{2},

where

* the first term measures closeness to the data,
* the second term penalises roughness,
* :math:`\lambda` balances the two,
* :math:`\partial^{\gamma}` is a *fractional* derivative: 
  the optimal solution is a fractional spline of degree :math:`2\gamma - 1`.

Taking the discrete Fourier transform (DFT) of both sides turns the
problem into a simple, frequency-by-frequency scaling

.. math::

   S(\omega) \;=\; H(\omega)\,Y(\omega),

where :math:`Y(\omega)` is the DFT of the data and :math:`S(\omega)` the
DFT of the solution.  
:math:`H(\omega)` has been computed and has a closed form [1]_. A convenient Butterworth-like
approximation of :math:`H(\omega)` is :math:`H_{2\gamma}(\omega)`, with

.. math::

   H_{2\gamma}(\omega)=\frac{1}{1+\lambda\,|\omega|^{2\gamma}}.

.. note::

   Here :math:`\omega` denotes the (angular) frequency variable.  This form is
   equivalent to the standard Butterworth parameterization
   :math:`1/(1+|\omega/\omega_0|^{2\gamma})` with :math:`\omega_0=\lambda^{-1/(2\gamma)}`.

The practical recipe is therefore

#. FFT the data,
#. multiply by :math:`H_{2\gamma}(\omega)`,
#. inverse FFT to obtain the smoothed samples.

A full derivation of this result can be found in [1]_, [2]_ and [3]_.

The next figure, from
:ref:`sphx_glr_auto_examples_05_smoothing_splines_01_1d_fractional_brownian_motion.py`,
illustrates these ideas on a noisy fractional Brownian motion: the original
process, noisy measurements, and the smoothed spline estimates at the
original and oversampled grids.

.. image:: /auto_examples/05_smoothing_splines/images/sphx_glr_01_1d_fractional_brownian_motion_001.png
   :align: center
   :width: 100%

Core Idea in Higher Dimensions
------------------------------

For a 2D image or a 3D volume we replace the one-dimensional
fractional derivative with the fractional Laplacian
:math:`(-\Delta)^{\gamma/2}`.  The variational cost therefore becomes

.. math::

   \sum_{\mathbf k}\bigl|\,y[\mathbf k]-s(\mathbf k)\bigr|^{2}
   \;+\;
   \lambda\,\bigl\lVert(-\Delta)^{\gamma/2}s\bigr\rVert_{L^{2}}^{2}.

In the Fourier domain the Laplacian turns into
:math:`\|\boldsymbol\omega\|^{2}`. A good approximation of the optimal filter is the
radial counterpart of the 1D Butterworth filter:

.. math::

   S(\boldsymbol\omega)
   \;=\;
   \frac{1}{1+\lambda\,\lVert\boldsymbol\omega\rVert^{2\gamma}}\,
   Y(\boldsymbol\omega).

The practical algorithm is identical to the 1D case:

#. Run an *n*-dimensional FFT to obtain :math:`Y(\boldsymbol\omega)`.  
#. Multiply by the gain above.  
#. Apply the inverse FFT to get the smoothed image or volume.

Because the filter is applied element-wise in the frequency domain, the
computation still needs just one forward FFT and one inverse FFT,
whatever the data dimension.

Fast Recursive Linear Smoother
------------------------------

When you only need the first-order case (:math:`\gamma = 1`), the
corresponding discrete frequency response simplifies to

.. math::

   H(\omega)=\frac{1}{1+4\lambda\,\sin^{2}(\omega/2)}.

This symmetric all-pole response can be factorized into a causal and an
anti-causal first-order filter with pole :math:`z_1\in(0,1)`:

.. math::

   H(\omega)=\frac{(1-z_1)^2}{1-2z_1\cos\omega+z_1^2},
   \qquad
   z_1=\frac{\sqrt{1+4\lambda}-1}{\sqrt{1+4\lambda}+1}.

With that number in hand the algorithm is

#. Causal pass (forward), with steady-state initialization:

   .. math::

      c[0]=\frac{y[0]}{1-z_1},\qquad
      c[k]=y[k]+z_1\,c[k-1].

#. Anti-causal pass (backward), with steady-state initialization:

   .. math::

      s[K-1]=\frac{c[K-1]}{1-z_1},\qquad
      s[k]=c[k]+z_1\,s[k+1].

#. Normalization:

   .. math::

      s[k]\leftarrow (1-z_1)^2\,s[k].

The two passes yield the same zero-phase result as applying :math:`H(\omega)`
in the Fourier domain, but at a cost that is strictly linear in the number
of samples and with constant memory. A detailed derivation (including
boundary handling for finite-length signals) appears in [4]_, Sections II-B
and II-D.

.. note::

   The *cubic* smoothing spline case requires a higher-order recursion
   (complex-conjugate poles) and does not reduce to a single real pole;
   see [4]_, Section IV-B.

Choosing the Parameters
-----------------------

* gamma: controls how steeply the filter rolls off  
  (larger values ⇒ steeper transition).  Typical range:
  :math:`0.5 \le \gamma \le 3`.

* lambda: moves the cut-off frequency  
  (small values keep more detail, large values smooth harder).
  For most images, :math:`10^{-3} \le \lambda \le 10^{-1}` is a
  good starting interval.

Smoothing Splines Examples
--------------------------

* :ref:`sphx_glr_auto_examples_05_smoothing_splines_01_1d_fractional_brownian_motion.py`
* :ref:`sphx_glr_auto_examples_05_smoothing_splines_02_2d_image_smoothing.py`
* :ref:`sphx_glr_auto_examples_05_smoothing_splines_03_3d_volume_smoothing.py`
* :ref:`sphx_glr_auto_examples_05_smoothing_splines_04_recursive_smoothing_spline.py`

References
----------

.. [1] M. Unser, T. Blu, `Self-Similarity: Part I—Splines and Operators <https://doi.org/10.1109/TSP.2006.890843>`_, 
   IEEE Transactions on Signal Processing, vol. 55, no. 4, pp. 1352-1363,
   April 2007.

.. [2] T. Blu, M. Unser, `Self-Similarity: Part II—Optimal Estimation of
   Fractal Processes <https://doi.org/10.1109/TSP.2006.890845>`_, 
   IEEE Transactions on Signal Processing, vol. 55, no. 4, pp. 1364-1378,
   April 2007.

.. [3] M. Unser, T. Blu, `Fractional Splines and Wavelets <https://doi.org/10.1137/S0036144598349435>`_, 
   SIAM Review, vol. 42, no. 1, pp. 43-67, March 2000.

.. [4] M. Unser, A. Aldroubi, M. Eden, *B-Spline Signal Processing:
   Part II—Efficient Design and Applications*, IEEE Transactions on Signal
   Processing, vol. 41, no. 2, pp. 834–848, Feb. 1993.
