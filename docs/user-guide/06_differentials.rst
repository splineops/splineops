.. splineops/docs/user-guide/06_differentials.rst

Differentials
=============

.. currentmodule:: splineops

Overview
--------

The :ref:`differentials <api-differentials>` module in :ref:`SplineOps <api-index>` provides a collection of algorithms for the 
computation of image differentials based on cubic B-spline interpolation [1]_, [2]_. By modeling a grayscale image as a continuous function 
reconstructed from its discrete samples, the module enables the accurate computation of derivatives. 
It offers several operations such as

- Gradient Magnitude: the local rate of change of the intensity;
- Gradient Direction: the direction along which the intensity changes most;
- Laplacian:the sum of second-order derivatives;
- Largest Hessian Eigenvalue: the maximal curvature;
- Smallest Hessian Eigenvalue: the minimal curvature; and
- Hessian Orientation: the principal orientation of the curvature.

The numerical behavior is deliberately separate from visualization:
``Differentials.run`` returns raw values, does not replace the source image,
and performs no console output.  ``normalize=True`` is an explicit convenience
for non-angular display maps.  ``spacing`` gives one physical sample distance
per array axis; all values default to one.

Image Representation
--------------------

A grayscale image is modeled as the continuous function

.. math::

    f(x_1, x_2) = \sum_{k_1,k_2} c[k_1,k_2] \cdot \varphi(x_1-k_1, x_2-k_2),

where :math:`\varphi(x_1,x_2)` is defined as the tensor-product cubic B‑spline

.. math::

    \varphi(x_1,x_2) = \beta^{3}(x_1) \cdot \beta^{3}(x_2).

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

      \theta(x,y) = \operatorname{atan2}\!\bigl(f_\mathrm{row},
      f_\mathrm{column}\bigr).

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

The preferred public class is ``Differentials``; the historical lowercase
``differentials`` alias remains available.  Direct component methods avoid
reconstructing quantities from composite maps:

.. code-block:: python

   from splineops.differentials import Differentials

   operator = Differentials(image, spacing=(0.7, 1.3))
   vertical, horizontal = operator.gradient_components()
   vertical2, cross, horizontal2 = operator.hessian_components()

``vertical`` follows increasing row coordinates and ``horizontal`` follows
increasing column coordinates.  Mirror-boundary derivatives are zero at the
outermost sample for the antisymmetric first-derivative filter.  Polynomial
and trigonometric fields are used as analytical tests away from that boundary.

Volumes and multi-output plans
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same component interface accepts one scalar 3-D volume.  Components are
returned in increasing axis order, while Hessians use packed
upper-triangular order ``(00, 01, 02, 11, 12, 22)``.  Gradient magnitude,
Laplacian, and minimum/maximum Hessian eigenvalue maps generalize to 3-D;
gradient direction and Hessian orientation remain explicitly 2-D quantities.

Use :class:`splineops.differentials.DifferentialPlan` when several derivative
families are needed together, or when changing volumes share one shape and
spacing:

.. code-block:: python

   from splineops.differentials import DifferentialPlan

   plan = DifferentialPlan(volume.shape, spacing=(0.7, 0.7, 1.5))
   result = plan.apply(volume, gradient=True, hessian=True)
   gx, gy, gz = result.gradient
   hxx, hxy, hxz, hyy, hyz, hzz = result.hessian
   laplacian = result.laplacian

Output families can be selected independently.  ``laplacian=None`` preserves
the original behavior and follows ``hessian``; request it explicitly to avoid
building gradients, packed mixed Hessians, or their output arrays:

.. code-block:: python

   laplacian = plan.apply(
       volume,
       gradient=False,
       hessian=False,
       laplacian=True,
   ).laplacian

For repeated allocation-sensitive calls, pass a
:class:`splineops.differentials.DifferentialResult` containing exact-shape and
exact-dtype destination arrays through ``out=``.  Fields corresponding to
unrequested families must be ``None``; the returned object is the supplied
buffer container.

One batched per-call workspace is used, so shared coefficient and derivative
intermediates are not recomputed for the requested outputs.
``DifferentialPlan`` also accepts explicit ``spatial_axes`` for batch and
channel arrays; components retain the full input shape and follow the selected
axis order.  The legacy ``Differentials`` object continues to model one scalar
image or volume, keeping its historical component helpers uncomplicated.

The following figure, taken from
:ref:`sphx_glr_auto_examples_06_differentials_01_differentials_module.py`,
shows some of these differential maps (gradient magnitude, gradient direction,
Laplacian and Hessian-based quantities).

.. image:: /auto_examples/06_differentials/images/sphx_glr_01_differentials_module_002.png
   :align: center
   :width: 100%

.. image:: /auto_examples/06_differentials/images/sphx_glr_01_differentials_module_003.png
   :align: center
   :width: 100%

.. image:: /auto_examples/06_differentials/images/sphx_glr_01_differentials_module_004.png
   :align: center
   :width: 100%

.. image:: /auto_examples/06_differentials/images/sphx_glr_01_differentials_module_005.png
   :align: center
   :width: 100%

Differentiate Examples
----------------------

* :ref:`sphx_glr_auto_examples_06_differentials_01_differentials_module.py`

References
----------

.. [1] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22-38, November 1999.

.. [2] M. Unser, T. Blu, `Fractional Splines and Wavelets <https://doi.org/10.1137/S0036144598349435>`_, 
   SIAM Review, vol. 42, no. 1, pp. 43-67, March 2000.
