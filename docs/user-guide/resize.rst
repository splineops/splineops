Resize Module
=============

Overview
--------

The `resize` function in the `splineops` library enables resizing (scaling) of N-dimensional data arrays using advanced spline-based methods. It supports various interpolation techniques, including standard interpolation, least-squares projection, and oblique projection. This flexibility allows for high-quality resizing operations suitable for image processing, scientific computing, and data analysis.

Key Features:
- Supports arbitrary scaling factors and output sizes.
- Offers multiple interpolation methods:
  - **Standard Interpolation**: Uses spline interpolation for smooth scaling.
  - **Least-Squares Projection**: Minimizes the least-squares error during resizing.
  - **Oblique Projection**: Provides an alternative projection method for resizing.
- Handles multi-dimensional data (2D, 3D, and higher).
- Customizable spline degree and boundary extension modes.

Mathematical Background
-----------------------

**Standard Interpolation**

The standard interpolation method uses spline interpolation to estimate the values of the resized data. Given an input data array :math:`f(\mathbf{x})`, where :math:`\mathbf{x}` represents the coordinates, the goal is to compute a continuous representation :math:`s(\mathbf{x})` such that:

.. math::

   s(\mathbf{x}) = \sum_{\mathbf{k}} c_{\mathbf{k}} \phi(\mathbf{x} - \mathbf{k})

Here, :math:`\phi` is the B-spline basis function of a specified degree, and :math:`c_{\mathbf{k}}` are the spline coefficients computed from the input data.

The resized data is then obtained by evaluating :math:`s(\mathbf{x})` at the new scaled coordinates.

**Least-Squares Projection**

The least-squares method aims to find a resized data array that best approximates the original data in the least-squares sense. It minimizes the squared difference between the original data and the resized data after scaling. Mathematically, it solves:

.. math::

   \min_{\tilde{f}} \left\| f - \tilde{f} \circ \mathbf{T}^{-1} \right\|^2

where:

- :math:`f` is the original data.
- :math:`\tilde{f}` is the resized data.
- :math:`\mathbf{T}` is the transformation matrix representing scaling.

**Oblique Projection**

The oblique projection method is an alternative to the least-squares approach. It projects the original data onto the space spanned by the scaling functions, but unlike the least-squares method, the projection is not necessarily orthogonal. This can be beneficial in certain scenarios where preserving specific data characteristics is important.

Implementation Details
----------------------

1. **Input Parameters**

   - **Data**: The input N-dimensional array to be resized.
   - **Zoom Factors**: Scaling factors for each axis. Alternatively, an explicit `output_size` can be specified.
   - **Degree**: The degree of the B-spline basis functions used for interpolation (typically between 0 and 9).
   - **Modes**: Boundary extension modes to handle data beyond the original domain (e.g., "mirror", "zero").
   - **Method**: The interpolation method to use ("interpolation", "least-squares", or "oblique").

2. **Coordinate Transformation**

   The function computes new coordinates for the resized data based on the scaling factors or the desired output size. For each dimension, the new coordinates are calculated as:

   .. math::

      x'_i = \frac{(N'_i - 1)}{(N_i - 1)} x_i

   where:

   - :math:`x_i` are the original coordinates.
   - :math:`x'_i` are the new coordinates.
   - :math:`N_i` and :math:`N'_i` are the original and new sizes along dimension :math:`i`.

3. **Spline Coefficient Computation**

   For standard interpolation, the spline coefficients are computed using the `TensorSpline` class, which performs multidimensional spline interpolation.

4. **Resampling**

   The resized data is obtained by evaluating the continuous spline representation at the new coordinates.

5. **Least-Squares and Oblique Methods**

   When using "least-squares" or "oblique" methods with degrees 1, 2, or 3, the function employs specialized algorithms to compute the resized data. These methods aim to minimize the reconstruction error during resizing.

Examples
--------

**Example 1: Resize a 2D Array Using Standard Interpolation**

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from splineops import resize

   # Create a sample 2D data array (e.g., an image)
   data = np.random.rand(100, 100)

   # Resize the data by a factor of 1.5
   resized_data = resize(data, zoom_factors=1.5, degree=3, method="interpolation")

   # Plot the original and resized data
   fig, axes = plt.subplots(1, 2, figsize=(10, 5))
   axes[0].imshow(data, cmap='gray')
   axes[0].set_title('Original Data')
   axes[1].imshow(resized_data, cmap='gray')
   axes[1].set_title('Resized Data (Interpolation)')
   plt.show()

**Example 2: Resize a 3D Array Using Least-Squares Projection**

.. code-block:: python

   import numpy as np
   from splineops import resize

   # Create a sample 3D data array (e.g., volumetric data)
   data_3d = np.random.rand(50, 50, 50)

   # Resize the data to a new size
   new_size = (100, 100, 100)
   resized_data_3d = resize(data_3d, output_size=new_size, degree=3, method="least-squares")

   # Check the shape of the resized data
   print('Original shape:', data_3d.shape)
   print('Resized shape:', resized_data_3d.shape)

**Example 3: Resize with Different Scaling Factors Along Each Axis**

.. code-block:: python

   import numpy as np
   from splineops import resize

   # Create a sample 2D data array
   data = np.random.rand(100, 200)

   # Resize with different zoom factors along each axis
   zoom_factors = (0.5, 2.0)
   resized_data = resize(data, zoom_factors=zoom_factors, degree=3)

   # Output the new shape
   print('Original shape:', data.shape)
   print('Resized shape:', resized_data.shape)

API Reference
-------------

For detailed information on the function parameters and usage, see the :ref:`Resize API documentation <api-resize>`.

