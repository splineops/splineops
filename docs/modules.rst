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
      :class-card: example-preview
      :img-top: /auto_examples/01_spline_interpolation/images/thumb/sphx_glr_01_tensorspline_class_thumb.png
      :img-alt: Discrete samples and the continuous two-dimensional spline fitted through them.

      Build a continuous model from uniform samples and evaluate it at
      arbitrary points or grids. Reuse geometry across changing data.

   .. grid-item-card:: Resize
      :link: user-guide/02_resize
      :link-type: doc
      :class-card: example-preview
      :img-top: /auto_examples/02_resize/images/thumb/sphx_glr_04_antialiasing_thumb.png
      :img-alt: Image detail compared with and without antialiasing during resizing.

      Resize N-D arrays with interpolation or projection antialiasing.
      Reuse fixed geometry with ResizePlan.

   .. grid-item-card:: Affine
      :link: user-guide/03_affine
      :link-type: doc
      :class-card: example-preview
      :img-top: /auto_examples/03_affine/images/thumb/sphx_glr_01_rotate_image_thumb.png
      :img-alt: An image rotated using spline interpolation.

      Rotate, shear, scale, and translate 2-D images or 3-D volumes.
      Share spline coefficients across transforms.

   .. grid-item-card:: Differentials
      :link: user-guide/06_differentials
      :link-type: doc
      :class-card: example-preview
      :img-top: /auto_examples/06_differentials/images/thumb/sphx_glr_01_differentials_module_thumb.png
      :img-alt: Gradient magnitude reveals image edges and fine structures.

      Compute cubic-spline gradients, Hessians, Laplacians, and derived
      features with explicit physical spacing.

   .. grid-item-card:: Smoothing splines
      :link: user-guide/05_smoothing_splines
      :link-type: doc
      :class-card: example-preview
      :img-top: /auto_examples/05_smoothing_splines/images/thumb/sphx_glr_02_2d_image_smoothing_thumb.png
      :img-alt: An original image, its noisy version, and the smoothed result.

      Smooth real signals and arrays with fractional, radial FFT, or
      recursive methods. Choose the formulation and boundaries explicitly.

   .. grid-item-card:: Adaptive regression
      :link: user-guide/04_adaptive_regression_splines
      :link-type: doc
      :class-card: example-preview
      :img-top: /auto_examples/04_adaptive_regression_splines/images/thumb/sphx_glr_01_adaptive_regression_splines_module_thumb.png
      :img-alt: A sparse spline regression fit through noisy observations.

      Fit sparse piecewise-linear models to 1-D observations and inspect
      convergence along a regularization path.

   .. grid-item-card:: Multiscale
      :link: user-guide/07_multiscale
      :link-type: doc
      :class-card: example-preview
      :img-top: /auto_examples/07_multiscale/images/thumb/sphx_glr_02_wavelet_decomposition_thumb.png
      :img-alt: Wavelet approximation and detail bands at multiple scales.

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
