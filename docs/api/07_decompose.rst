.. _api-decompose:

Decompose
=========

The :mod:`splineops.decompose.pyramid` module implements pyramid decomposition functionality based on spline expansions. It provides a set of functions to perform multiresolution analysis of 1D signals and 2D images using carefully designed REDUCE (downsampling) and EXPAND (upsampling) operators with mirror boundary handling. These operators are fundamental in constructing spline pyramids and establish a connection with wavelet transforms.

Key functionalities include:

- **get_pyramid_filter** – Retrieve the filter coefficients for the REDUCE (g) and EXPAND (h) operations along with an indicator of whether the filter is centered.
- **reduce_1d** – Downsample a 1D signal by a factor of two using the appropriate filter and mirror reflection for boundary conditions.
- **expand_1d** – Upsample a 1D signal by a factor of two using spline interpolation derived from the EXPAND filter.
- **reduce_2d** – Apply the 1D reduction operator along rows and columns to downsample a 2D image.
- **expand_2d** – Apply the 1D expansion operator along rows and columns to upsample a 2D image.

Additional internal routines manage robust mirror boundary reflection and the centered variants of the reduction and expansion operations.

.. automodule:: splineops.decompose.pyramid
   :members:
   :member-order: bysource
