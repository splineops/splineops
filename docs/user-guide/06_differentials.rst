.. splineops/docs/user-guide/06_differentials.rst

Differentials
=============

.. currentmodule:: splineops

Overview
--------
The *differentials* module in *splineops* provides a collection of algorithms for the computation of image differentials based on cubic B-spline interpolation [1]_, [2]_. 
By modeling a grayscale image as a continuous function reconstructed from its discrete samples, the module enables the accurate computation of derivatives. 
It offers several operations such as

- Gradient Magnitude: the local rate of change of the intensity;
- Gradient Direction: the direction along which the intensity changes most;
- Laplacian:the sum of second-order derivatives;
- Largest Hessian Eigenvalue: the maximal curvature;
- Smallest Hessian Eigenvalue: the minimal curvature; and
- Hessian Orientation: the principal orientation of the curvature.

Image Representation
--------------------

A grayscale image is modeled as the continuous function

.. math::

    f(x_1, x_2) = \sum_{k_1,k_2} c[k_1,k_2] \cdot \phi(x_1-k_1, x_2-k_2),

where :math:`\phi(x_1,x_2)` is defined as the tensor-product cubic B‑spline

.. math::

    \phi(x_1,x_2) = \beta^{3}(x_1) \cdot \beta^{3}(x_2).

This formulation allows one to compute exact derivatives of the image by first determining the spline coefficients :math:`c[k_1,k_2]`.

Differentiation Operations
--------------------------

Based on the spline representation, the module computes several differential operators. We define

.. math::

     f_1   \equiv \frac{\partial f}{\partial x_1},\\
     f_2   \equiv \frac{\partial f}{\partial x_2},\\
     f_{11}\equiv \frac{\partial^2 f}{\partial x_1^2},\\
     f_{22}\equiv \frac{\partial^2 f}{\partial x_2^2},\\
     f_{12}=f_{21} \equiv \frac{\partial^2 f}{\partial x_1 \partial x_2}.  

- Gradient Magnitude:  
  Computed as the Euclidean norm of the first derivatives

  .. math::

      \|\pmb{\nabla}f(x,y)\| = \sqrt{\bigl(f_1\bigr)^2 + \bigl(f_2\bigr)^2}.

- Gradient Direction:  
  The direction of the gradient is given by

  .. math::

      \theta(x,y) = \arctan\!\Bigl(\tfrac{f_2}{f_1}\Bigr).

- Laplacian:  
  A second-order operator that highlights regions of rapid intensity change as

  .. math::

      \Delta f(x,y) = f_{11} + f_{22}.

- Largest Hessian Eigenvalue:  
  The maximal eigenvalue of the Hessian matrix is given by

  .. math::

      \lambda_{\text{max}} = \tfrac12\Bigl(f_{11} + f_{22}
      + \sqrt{4f_{12}^2 + (f_{11} - f_{22})^2}\Bigr).

- Smallest Hessian Eigenvalue:  
  The minimal eigenvalue of the Hessian matrix is given by

  .. math::

      \lambda_{\text{min}} = \tfrac12\Bigl(f_{11} + f_{22}
      - \sqrt{4f_{12}^2 + (f_{11} - f_{22})^2}\Bigr).

- Hessian Orientation:  
  This operation returns the orientation that corresponds to the maximal second derivative, as

  .. math::

      \theta_H(x,y) = \pm \tfrac12
      \arccos\!\Bigl(\tfrac{f_{11} - f_{22}}{\sqrt{4f_{12}^2 + (f_{11}-f_{22})^2}}\Bigr),

  where the sign is determined by the sign of the cross derivative :math:`f_{12}`.

Implementation Details
----------------------

The *Differentials* class implements these operations as follows: the input image is provided by its samples. We assume it to be a cubic B-spline and first determine 
its interpolation coefficients. The differential-based computations that we perform are then perfectly consistent with this continuously defined function.
We finally build the output image by sampling the ideal, continuously defined intermediate result.

Differentiate Examples
----------------------

* :ref:`sphx_glr_auto_examples_07_differentials_07_01_differentials_module.py`

References
----------

.. [1] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22-38, November 1999.

.. [2] M. Unser, T. Blu, `Fractional Splines and Wavelets <https://doi.org/10.1137/S0036144598349435>`_, 
   SIAM Review, vol. 42, no. 1, pp. 43-67, March 2000.
