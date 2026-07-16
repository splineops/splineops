Volume downsampling
===================

This tutorial downsamples continuous-valued 3-D volumes with an explicit
spatial-axis contract.  SplineOps operates on array grids; it does not manage
medical-image origin, direction, or physical-spacing metadata in version 2.1.

One volume
----------

.. code-block:: python

   import numpy as np
   from splineops import resize

   volume = np.random.default_rng(0).random((80, 256, 256), dtype=np.float32)
   coarse = resize(
       volume,
       output_size=(40, 128, 128),
       axes=(0, 1, 2),
       method="cubic-antialiasing",
   )

Channels or time points
-----------------------

For an array shaped ``T x C x Z x Y x X``, resize only the last three axes:

.. code-block:: python

   coarse_series = resize(
       series,
       output_size=(40, 128, 128),
       axes=(-3, -2, -1),
       method="cubic-antialiasing",
   )

The time and channel dimensions remain unchanged and are not filtered.

Repeated geometry
-----------------

Use ``ResizePlan`` when volumes arrive separately but share a shape:

.. code-block:: python

   from splineops import ResizePlan

   plan = ResizePlan(
       input_shape=(80, 256, 256),
       output_size=(40, 128, 128),
       method="cubic-antialiasing",
   )

   for volume in volumes:
       coarse = plan(volume)
       consume(coarse)

Data semantics
--------------

Projection antialiasing is intended for continuous-valued samples such as
intensity fields.  Categorical segmentation labels normally require nearest-
neighbour resampling.  Probability channels may require normalization after
resizing, depending on the application.

If an input comes from NIfTI, DICOM, or another physical-space format, update
and validate its metadata in the library that owns that metadata.  SplineOps
2.1 returns an array, not a physical-space image object.

See :doc:`user-guide/02_resize` for the endpoint-aligned coordinate contract
and :doc:`performance` for measured performance and its limitations.
