.. splineops/docs/index.rst

SplineOps: Spline Operations
============================

**SplineOps** is an open-source software library written in Python. It provides ground-breaking signal-processing tools based on splines. 
Adapted and built on the algorithms developed through the years by the `Biomedical Imaging Group at EPFL <https://bigwww.epfl.ch/>`_ (Lausanne, Switzerland), 
SplineOps is in active development and supports modern computational demands.

.. figure:: _static/waveletbird_full.jpeg
   :alt: Main Feature of SplineOps
   :align: center
   :scale: 40%

   A medley of spline functions and their derivatives

Key Features and Capabilities
=============================

- **Optimized Performance**: Leveraging of CPU and GPU architectures to handle large-scale signal datasets effectively.

- **Precision and Flexibility**: High-degree spline interpolation across multiple dimensions.

- **Scalability and Extensibility**: Incorporation of new functionalities tailored to specific applications.

.. figure:: _static/feature_01.jpg
   :alt: Key Feature Illustration
   :align: center
   :scale: 60%

   B-Splines

Modules at a Glance
===================

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Spline Interpolation
      :link: user-guide/01_spline_interpolation
      :link-type: doc
      :img-top: /auto_examples/01_spline_interpolation/images/sphx_glr_04_interpolate_1d_samples_003.png
      :img-alt: Cubic B-spline basis
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Continuous spline models from discrete data, in any dimension.

   .. grid-item-card:: Resize
      :link: user-guide/02_resize
      :link-type: doc
      :img-top: /auto_examples/02_resize/images/sphx_glr_06_benchmarking_055.png
      :img-alt: 1D spline resampling
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Projection-based resampler for highest-quality spline resizing and antialiasing.

   .. grid-item-card:: Affine
      :link: user-guide/03_affine
      :link-type: doc
      :img-top: /auto_examples/03_affine/images/sphx_glr_01_rotate_image_001.png
      :img-alt: Rotated image example
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Geometric transformations on images and volumes.

   .. grid-item-card:: Adaptive Regression Splines
      :link: user-guide/04_adaptive_regression_splines
      :link-type: doc
      :img-top: /auto_examples/04_adaptive_regression_splines/images/sphx_glr_01_adaptive_regression_splines_module_004.png
      :img-alt: Piecewise-linear regression
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Sparsest 1D piecewise-linear fits: data in, knots out.

   .. grid-item-card:: Smoothing Splines
      :link: user-guide/05_smoothing_splines
      :link-type: doc
      :img-top: /auto_examples/05_smoothing_splines/images/sphx_glr_01_1d_fractional_brownian_motion_001.png
      :img-alt: Smoothing spline on a 1D process
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Fractional spline filters for principled, tunable smoothing of signals and images.

   .. grid-item-card:: Differentials
      :link: user-guide/06_differentials
      :link-type: doc
      :img-top: /auto_examples/06_differentials/images/sphx_glr_01_differentials_module_002.png
      :img-alt: Gradient/Laplacian visualisation
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Precise gradients, Laplacians and Hessian features from a spline representation.

   .. grid-item-card:: Multiscale
      :link: user-guide/07_multiscale
      :link-type: doc
      :img-top: /auto_examples/07_multiscale/images/sphx_glr_02_wavelet_decomposition_001.png
      :img-alt: Multiscale / wavelet decomposition
      :shadow: md
      :class-card: sd-rounded-2 sd-border

      Spline pyramids and wavelets for multiscale analysis and processing.

Contents
========

.. toctree::
   :maxdepth: 1
   :caption: Contents:
   :titlesonly:

   installation/index
   user-guide/index
   auto_examples/index
   api/index