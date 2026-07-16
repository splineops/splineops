Five-minute quickstart
======================

Downsample continuous data
--------------------------

Use ``cubic-antialiasing`` as the default starting point for continuous-valued
data that is being made smaller:

.. code-block:: python

   import numpy as np
   from splineops import resize

   image = np.random.default_rng(0).random((512, 512), dtype=np.float32)
   small = resize(
       image,
       output_size=(256, 256),
       method="cubic-antialiasing",
   )

Select spatial axes
-------------------

When an array includes batch or channel dimensions, identify only the spatial
axes.  Unselected axes are preserved exactly:

.. code-block:: python

   # H x W x C
   small_rgb = resize(
       rgb,
       output_size=(256, 256),
       axes=(0, 1),
       method="cubic-antialiasing",
   )

Choose a method
---------------

.. list-table:: Resize method selection
   :header-rows: 1
   :widths: 32 30 38

   * - Goal
     - Method
     - Note
   * - Continuous-data downsampling
     - ``cubic-antialiasing``
     - Recommended starting point.
   * - Faster downsampling
     - ``linear-antialiasing``
     - Lower-order model.
   * - Upsampling
     - ``cubic``
     - Interpolation without projection.
   * - Categorical labels
     - ``fast``
     - Nearest-neighbour semantics; validate the intended grid convention.
   * - Research configurations
     - ``resize_degrees``
     - Exposes analysis and synthesis degrees directly.

Reuse fixed geometry
--------------------

``ResizePlan`` avoids repeating geometry setup when many arrays have the same
shape:

.. code-block:: python

   from splineops import ResizePlan

   plan = ResizePlan(
       input_shape=(64, 192, 192),
       output_size=(32, 96, 96),
       method="cubic-antialiasing",
   )
   coarse_frame = plan(frame)

Important semantics
-------------------

SplineOps resize uses one endpoint-aligned output grid.  Calls that produce the
same integer output shape therefore produce the same grid and result.  Review
the :doc:`user-guide/02_resize` guide before comparing another library: output
size, coordinates, boundaries, and antialiasing methods must all match.

Next, see :doc:`volume-downsampling` for a complete 3-D workflow.
