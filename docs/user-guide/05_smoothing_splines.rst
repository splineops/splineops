.. splineops/docs/user-guide/05_smoothing_splines.rst

Smoothing Splines
=================

Overview
--------

The *smoothing splines* module provides tools to fit *fractional smoothing splines*
to noisy data. Think of these splines as flexible low-pass filters whose sharpness can
be tuned continuously, making them effective for signals and images that
exhibit repeating, self-similar patterns.

You will find:

* Exact 1D routine:
  Works on a 1D array and returns the mathematically exact
  fractional-spline result.

* Isotropic N-D routine:
  Extends the idea to 2D pictures or 3D volumes through one FFT;
  internally it uses a Butterworth low-pass filter.

* Fast cubic shortcut:
  A lightweight forward/backward IIR filter that approximates the cubic
  case and runs in a single pass—handy for real-time
  streams.

* Extra helpers to generate test data
  (fractional Brownian motion) and to compute spline autocorrelations.

Core idea in one dimension
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
* :math:`\partial^{\gamma}` is a *fractional* derivative
  (:math:`\gamma=1` gives the classic cubic penalty).

Taking the discrete Fourier transform (DFT) of both sides turns the
problem into a simple, frequency-by-frequency scaling

.. math::

   S(\omega) \;=\; H(\omega)\,Y(\omega),\qquad
   H(\omega)=\frac{1}{1+\lambda\,|\omega|^{2\gamma}},

where :math:`Y(\omega)` is the DFT of the data and :math:`S(\omega)` the
DFT of the solution.  The practical recipe is therefore

#. FFT the data,
#. multiply by :math:`H(\omega)`,
#. inverse FFT to obtain the smoothed samples.

A full derivation of this result can be found in [1]_, [2]_ and [3]_.

Core idea in higher dimensions
------------------------------

For a 2D image or a 3D volume we replace the one-dimensional
fractional derivative with the fractional Laplacian
:math:`(-\Delta)^{\gamma/2}`.  The variational cost therefore becomes

.. math::

   \sum_{\mathbf k}\bigl|\,y[\mathbf k]-s(\mathbf k)\bigr|^{2}
   \;+\;
   \lambda\,\bigl\lVert(-\Delta)^{\gamma/2}s\bigr\rVert_{L^{2}}^{2}.

In the Fourier domain the Laplacian turns into
:math:`\|\boldsymbol\omega\|^{2}`, so the optimal filter is the *radial*
version of the 1D one:

.. math::

   S(\boldsymbol\omega)
   \;=\;
   \frac{1}{1+\lambda\,\lVert\boldsymbol\omega\rVert^{2\gamma}}\,
   Y(\boldsymbol\omega).

This a Butterworth low-pass filter of order :math:`2\gamma`. The practical
algorithm is identical to the 1D case:

#. Run an *n*-dimensional FFT to obtain :math:`Y(\boldsymbol\omega)`.  
#. Multiply by the gain above.  
#. Apply the inverse FFT to get the smoothed image or volume.

Because the filter is applied element-wise in the frequency domain, the
computation still needs just one forward FFT and one inverse FFT,
whatever the data dimension.

Fast recursive cubic smoother
-----------------------------

When you only need the cubic case (:math:`\gamma = 1`) the frequency
response above simplifies so much that it can be implemented with two
tiny first-order filters—one run forward, the other backward.  The key
quantity is the *pole*  

.. math::

   z_1 \;=\;-\frac{\lambda}{1+\sqrt{\,1+4\lambda\,}}.

With that number in hand the algorithm is

#. Causal pass:
   start at :math:`k = 0` and accumulate  
   :math:`c[k] = y[k] + z_1\,c[k-1]`.

#. Anti-causal pass:
   start at the last sample and run backwards  
   :math:`s[k] = c[k] + z_1\,s[k+1]`.

The two passes give the same zero-phase result you would obtain from the
FFT method but at a cost that is strictly linear in the number of
samples and with virtually no memory footprint.  A detailed derivation
appears in [1]_, Section IV-B.

Choosing the parameters
-----------------------

* gamma: controls how steeply the filter rolls off  
  (larger values ⇒ steeper transition).  Typical range:
  :math:`0.5 \le \gamma \le 3`.

* lambda: moves the cut-off frequency  
  (small values keep more detail, large values smooth harder).
  For most images, :math:`10^{-3} \le \lambda \le 10^{-1}` is a
  good starting interval.

Smoothing splines examples
--------------------------

* :ref:`sphx_glr_auto_examples_06_smoothing_splines_06_01_1d_fractional_brownian_motion.py`
* :ref:`sphx_glr_auto_examples_06_smoothing_splines_06_02_2d_image_smoothing.py`
* :ref:`sphx_glr_auto_examples_06_smoothing_splines_06_04_recursive_smoothing_spline.py`

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
