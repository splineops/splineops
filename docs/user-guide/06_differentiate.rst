Differentiate
=============

.. currentmodule:: splineops

Overview
--------
The `differentiate` module in `splineops` provides a collection of algorithms for computing image differentials based on cubic B‑spline interpolation [1]_, [2]_. 
By modeling a grayscale image as a continuous function reconstructed from its discrete samples, the module enables the accurate computation of derivatives. 
It offers several operations including:

- Gradient Magnitude – the local rate of intensity change,
- Gradient Direction – the orientation of maximum intensity change,
- Laplacian – the sum of second-order derivatives,
- Largest Hessian Eigenvalue – the maximum curvature,
- Smallest Hessian Eigenvalue – the minimum curvature, and
- Hessian Orientation – the principal direction of curvature.

Key features include the use of precise spline interpolation coefficients, the implementation of both anti-symmetric and symmetric FIR filters for first and 
second derivatives, and a tunable tolerance parameter to balance speed and accuracy.

Mathematical Background
-------------------------
Image Representation
~~~~~~~~~~~~~~~~~~~~
A grayscale image is modeled as a continuous function

.. math::

    f(x, y) = \sum_{k,l} c[k,l] \cdot \phi(x-k, y-l),

where :math:`\phi(x,y)` is defined as the tensor-product cubic B‑spline

.. math::

    \phi(x,y) = \beta_3(x) \cdot \beta_3(y).

This formulation allows one to compute exact derivatives of the image by first determining the spline coefficients :math:`c[k,l]`.

Differentiation Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~
Based on the spline representation, the module computes various differential operators:

- Gradient Magnitude  
  Computed as the Euclidean norm of the first derivatives:

  .. math::

      \|\nabla f(x,y)\| = \sqrt{\left(\frac{\partial f}{\partial x}\right)^2 + \left(\frac{\partial f}{\partial y}\right)^2}.

- Gradient Direction  
  The orientation of the gradient is given by:

  .. math::

      \theta(x,y) = \arctan\left(\frac{\partial f/\partial y}{\partial f/\partial x}\right).

- Laplacian  
  A second-order operator that highlights regions of rapid intensity change:

  .. math::

      \Delta f(x,y) = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}.

- Largest Hessian Eigenvalue  
  The maximum eigenvalue of the Hessian matrix is computed as:

  .. math::

      \lambda_{\text{max}} = \frac{1}{2}\Bigl(f_{xx} + f_{yy} + \sqrt{4f_{xy}^2 + \left(f_{xx} - f_{yy}\right)^2}\Bigr).

- Smallest Hessian Eigenvalue  
  The minimum eigenvalue of the Hessian matrix is given by:

  .. math::

      \lambda_{\text{min}} = \frac{1}{2}\Bigl(f_{xx} + f_{yy} - \sqrt{4f_{xy}^2 + \left(f_{xx} - f_{yy}\right)^2}\Bigr).

- Hessian Orientation  
  This operation returns the orientation corresponding to the maximum second derivative:

  .. math::

      \theta_H(x,y) = \pm \frac{1}{2}\arccos\Bigl(\frac{f_{xx} - f_{yy}}{\sqrt{4f_{xy}^2 + \left(f_{xx} - f_{yy}\right)^2}}\Bigr),

  where the sign is determined by the sign of the cross derivative :math:`f_{xy}`.

Implementation Details
----------------------
The `Differentials` class implements these operations in two main stages:

1. Spline Interpolation:  
   The image is first converted to a continuously defined function by computing the cubic B‑spline interpolation coefficients. This step ensures that 
   derivatives are computed on a smooth approximation of the original data.

2. Differentiation Filtering:  
   Using the precomputed spline coefficients, the module applies finite impulse response (FIR) filters:
   
   - Anti-symmetric filters are used to compute first-order derivatives (gradients).
   - Symmetric filters are applied for second-order derivatives (Hessians).

A tolerance parameter is provided to control the trade-off between precision and computational efficiency (with the recommended value being near the machine 
epsilon for single-precision floats). Progress during processing is displayed via a simple progress bar mechanism.

Differentiate Example
----------------------
* :ref:`sphx_glr_auto_examples_008_using_differentiate_module.py`

References
----------

.. [1] Unser, M.,
  Splines: A Perfect Fit for Signal and Image Processing, 
  IEEE Signal Processing Magazine, vol. 16, no. 6, 1999.

.. [2] Unser, M. & Blu, T., 
  Fractional Splines and Wavelets,
  SIAM Review, vol. 42, no. 1, pp. 43–67, 2000.

.. note::
    The `differentiate` module is optimized for grayscale images of type ``GRAY32``. For best results, ensure that input images are normalized (typically 
    in the range [0, 1]) and that the tolerance parameter is set appropriately.