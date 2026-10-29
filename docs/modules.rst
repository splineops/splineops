Modules
=======

Choose a tool for your task. All seven modules have stable NumPy/CPU APIs in
SplineOps 2.3, with explicit numerical contracts. See :doc:`project-status`
for supported precision, dimensions, and boundaries.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: TensorSpline
      :link: user-guide/01_spline_interpolation
      :link-type: doc

      Build a continuous model from uniform samples and evaluate it at
      arbitrary points or grids. Reuse geometry across changing data.

   .. grid-item-card:: Resize
      :link: user-guide/02_resize
      :link-type: doc

      Resize N-D arrays with interpolation or projection antialiasing.
      Reuse fixed geometry with ResizePlan.

   .. grid-item-card:: Affine
      :link: user-guide/03_affine
      :link-type: doc

      Rotate, shear, scale, and translate 2-D images or 3-D volumes.
      Share spline coefficients across transforms.

   .. grid-item-card:: Differentials
      :link: user-guide/06_differentials
      :link-type: doc

      Compute cubic-spline gradients, Hessians, Laplacians, and derived
      features with explicit physical spacing.

   .. grid-item-card:: Smoothing splines
      :link: user-guide/05_smoothing_splines
      :link-type: doc

      Smooth real signals and arrays with fractional, radial FFT, or
      recursive methods. Choose the formulation and boundaries explicitly.

   .. grid-item-card:: Adaptive regression
      :link: user-guide/04_adaptive_regression_splines
      :link-type: doc

      Fit sparse piecewise-linear models to 1-D observations and inspect
      convergence along a regularization path.

   .. grid-item-card:: Multiscale
      :link: user-guide/07_multiscale
      :link-type: doc

      Build spline pyramids and decompose images with Haar or spline
      wavelets. Order-5 wavelets use documented approximate taps.

See the :doc:`examples` for runnable illustrations and
:doc:`consolidation-recipes` for workflows combining several modules.

.. toctree::
   :hidden:
   :maxdepth: 2

   user-guide/index
   user-guide/01_spline_interpolation
   user-guide/02_resize
   user-guide/03_affine
   user-guide/04_adaptive_regression_splines
   user-guide/05_smoothing_splines
   user-guide/06_differentials
   user-guide/07_multiscale
   consolidation-recipes
