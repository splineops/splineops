Resize Module
=============

Overview
--------

The `Resize` module provides tools for scaling spline geometries. It supports uniform and non-uniform resizing operations.

Key Features:
- Uniform resizing of splines.
- Axis-specific resizing for custom transformations.
- Seamless integration with the Rotate module.

Examples
--------

Basic Usage
~~~~~~~~~~~

Here's how to perform a uniform resize:

.. code-block:: python

   from splineops import resize

   # Resize a spline object by a factor of 2
   resized_spline = resize.uniform(spline, factor=2)

Axis-Specific Resize
~~~~~~~~~~~~~~~~~~~~~

To resize only along specific axes:

.. code-block:: python

   resized_spline = resize.axis_specific(spline, x_factor=2, y_factor=1, z_factor=0.5)

API Reference
-------------

For more details, see the :ref:`Resize API documentation <api-resize>`.
