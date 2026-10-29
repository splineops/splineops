.. splineops/docs/api/index.rst

.. _api-index:

API Reference
=============

The documented NumPy/CPU APIs below are stable in SplineOps 2.3. See
:doc:`../project-status` for precision, shape, boundary, and compatibility
contracts. Experimental CuPy interoperability is described separately in
:doc:`../backend-support`.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - **Module**
     - **Description**
   * - :ref:`Spline Interpolation <api-spline_interpolation>`
     - Extensible Python-based interpolation class for multidimensional data.
   * - :ref:`Resize <api-resize>`
     - N-D interpolation and projection antialiasing with reusable native plans.
   * - :ref:`Affine <api-affine>`
     - Affine transforms on 2D/3D data using spline interpolation.
   * - :ref:`Adaptive Regression Splines <api-adaptive_regression_splines>`
     - Perform sparsest linear regression on 1D data.
   * - :ref:`Smoothing Splines <api-smoothing_splines>`
     - Fit splines to noisy data.
   * - :ref:`Differentials <api-differentials>`
     - Compute first and second-order spline derivatives on 2-D/3-D arrays.
   * - :ref:`Multiscale <api-multiscale>`
     - Analyze 1D signals and 2D images at multiple scales.

.. toctree::
   :maxdepth: 1
   :hidden:

   01_spline_interpolation
   02_resize
   03_affine
   04_adaptive_regression_splines
   05_smoothing_splines
   06_differentials
   07_multiscale
