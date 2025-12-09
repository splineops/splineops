.. splineops/docs/user-guide/index.rst

User Guide
==========

This guide provides detailed explanations, tutorials, and examples to use the modules available in `splineops`.

.. toctree::
   :maxdepth: 2
   :caption: Modules
   :titlesonly:
   :hidden:

   01_spline_interpolation
   02_resize
   03_rotate
   04_adaptive_regression_splines
   05_smoothing_splines
   06_differentials
   07_multiscale

Module overview
---------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Spline interpolation
      :link: 01_spline_interpolation
      :link-type: doc
      :img-top: /auto_examples/01_quick-start/images/sphx_glr_01_02_spline_bases_004.png
      :img-alt: Cubic B-spline basis
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Build continuous splines from discrete samples and evaluate them at arbitrary coordinates.

   .. grid-item-card:: Resize
      :link: 02_resize
      :link-type: doc
      :img-top: /auto_examples/02_resampling_using_1d_samples/images/sphx_glr_02_02_resample_a_1d_spline_001.png
      :img-alt: 1D spline resampling
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      High-quality N-D resizing on uniform grids with optional antialiasing.

   .. grid-item-card:: Rotate
      :link: 03_rotate
      :link-type: doc
      :img-top: /auto_examples/04_rotate_04_01_rotate_image/images/sphx_glr_04_01_rotate_image_001.png
      :img-alt: Rotated image example
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      2D and 3D rotations around arbitrary centers and axes using spline interpolation.

   .. grid-item-card:: Adaptive regression splines
      :link: 04_adaptive_regression_splines
      :link-type: doc
      :img-top: /auto_examples/05_adaptive_regression_splines/images/sphx_glr_05_01_adaptive_regression_splines_module_001.png
      :img-alt: Piecewise-linear regression
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      TV-regularised 1D regression with guaranteed piecewise-linear, few-knot solutions.

   .. grid-item-card:: Smoothing splines
      :link: 05_smoothing_splines
      :link-type: doc
      :img-top: /auto_examples/06_smoothing_splines/images/sphx_glr_06_02_2d_image_smoothing_001.png
      :img-alt: Smoothing spline on an image
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Fractional smoothing splines for tunable low-pass filtering in 1D and N-D.

   .. grid-item-card:: Differentials
      :link: 06_differentials
      :link-type: doc
      :img-top: /auto_examples/07_differentials/images/sphx_glr_07_01_differentials_module_001.png
      :img-alt: Gradient/Laplacian visualisation
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Exact spline-based gradients, Laplacians, and Hessian-derived features.

   .. grid-item-card:: Multiscale
      :link: 07_multiscale
      :link-type: doc
      :img-top: /auto_examples/08_multiscale/images/sphx_glr_08_02_wavelet_decomposition_001.png
      :img-alt: Multiscale / wavelet decomposition
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Spline pyramids and wavelet decompositions for multiscale analysis.
