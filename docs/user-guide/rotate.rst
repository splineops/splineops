Rotate Module
=============

Overview
--------

The `Rotate` module enables transformations for spline geometries around specified axes or angles.

Key Features:
- Arbitrary axis rotation.
- Predefined axis rotation (X, Y, Z).
- Integration with Resize for combined transformations.

Examples
--------

Basic Rotation
~~~~~~~~~~~~~~

Rotate a spline 90 degrees around the Z-axis:

.. code-block:: python

   from splineops import rotate

   # Rotate a spline
   rotated_spline = rotate.around_axis(spline, axis='z', angle=90)

Arbitrary Axis Rotation
~~~~~~~~~~~~~~~~~~~~~~~

To rotate around an arbitrary axis:

.. code-block:: python

   rotated_spline = rotate.arbitrary(spline, axis_vector=[1, 1, 0], angle=45)

API Reference
-------------

For more details, see the :ref:`Rotate API documentation <api-rotate>`.
