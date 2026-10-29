User guide
==========

Start with resize
-----------------

Regular-grid N-D resizing provides explicit
coordinates and projection antialiasing.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Resize
      :link: 02_resize
      :link-type: doc

      Choose methods, geometry, axes, boundaries, and reusable plans.

   .. grid-item-card:: Five-minute quickstart
      :link: ../quickstart
      :link-type: doc

      Downsample an array and select spatial axes safely.

Continuous models
-----------------

``TensorSpline`` is a stable NumPy/CPU abstraction for evaluating a
continuous tensor-product spline at arbitrary coordinates.

.. grid:: 1

   .. grid-item-card:: TensorSpline
      :link: 01_spline_interpolation
      :link-type: doc

      Construct and query continuous spline models with multiple bases and
      extension modes.

More spline tools
-----------------

These stable CPU modules keep their own numerical operators and support
contracts. See :doc:`../project-status` for the complete support matrix.

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Affine
      :link: 03_affine
      :link-type: doc

      Spline-evaluated 2-D and 3-D affine transforms.

   .. grid-item-card:: Differentials
      :link: 06_differentials
      :link-type: doc

      Gradients, Laplacians, and Hessian features.

   .. grid-item-card:: Smoothing
      :link: 05_smoothing_splines
      :link-type: doc

      Fractional smoothing and related research methods.

   .. grid-item-card:: Adaptive regression
      :link: 04_adaptive_regression_splines
      :link-type: doc

      Sparse one-dimensional piecewise-linear models.

   .. grid-item-card:: Multiscale
      :link: 07_multiscale
      :link-type: doc

      Spline pyramids and wavelets.
