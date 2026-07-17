Modules
=======

SplineOps keeps its spline tools as independent public modules.  Their
maturity labels describe API and validation stability, not the importance of
the underlying methods.

Stable
------

.. grid:: 1

   .. grid-item-card:: Resize and ResizePlan
      :link: user-guide/02_resize
      :link-type: doc

      Native N-D interpolation and projection antialiasing, with explicit
      grids, axes, boundaries, output buffers, and a Python reference path.

Stabilizing
-----------

.. grid:: 1

   .. grid-item-card:: TensorSpline
      :link: user-guide/01_spline_interpolation
      :link-type: doc

      Continuous tensor-product spline models evaluated at arbitrary
      coordinates.

Research modules
----------------

These modules are experimental.  Their APIs or validation contracts may
change while their individual graduation audits remain open.

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Affine
      :link: user-guide/03_affine
      :link-type: doc

      Spline-evaluated 2-D and 3-D geometric transforms.

   .. grid-item-card:: Differentials
      :link: user-guide/06_differentials
      :link-type: doc

      Gradients, Hessians, Laplacians, and derived features.

   .. grid-item-card:: Smoothing splines
      :link: user-guide/05_smoothing_splines
      :link-type: doc

      Fractional and regularized smoothing methods.

   .. grid-item-card:: Adaptive regression
      :link: user-guide/04_adaptive_regression_splines
      :link-type: doc

      Sparse one-dimensional piecewise-linear models.

   .. grid-item-card:: Multiscale
      :link: user-guide/07_multiscale
      :link-type: doc

      Spline pyramids and wavelet transforms.

   .. grid-item-card:: Module maturity
      :link: project-status
      :link-type: doc

      Exact support boundaries, evidence, and remaining graduation work.

.. toctree::
   :hidden:
   :maxdepth: 2

   user-guide/index
   user-guide/02_resize
   user-guide/01_spline_interpolation
   user-guide/03_affine
   user-guide/04_adaptive_regression_splines
   user-guide/05_smoothing_splines
   user-guide/06_differentials
   user-guide/07_multiscale
   consolidation-recipes

