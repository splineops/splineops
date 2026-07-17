SplineOps
=========

**Projection-based antialiased resizing for N-D scientific arrays.**

SplineOps is designed for precise, repeatable downsampling of regular-grid
2-D images and 3-D volumes.  It combines explicit coordinate semantics, a
readable Python reference implementation, a native CPU backend, and reusable
plans for fixed-geometry workloads.

Start here
----------

.. code-block:: python

   import numpy as np
   from splineops import resize

   volume = np.random.default_rng(0).random((64, 192, 192), dtype=np.float32)
   coarse = resize(
       volume,
       output_size=(32, 96, 96),
       method="cubic-antialiasing",
   )

The antialiasing methods project the input spline onto a coarser spline space
instead of treating downsampling as interpolation alone.  Continue with the
:doc:`quickstart`, then use the :doc:`volume-downsampling` tutorial for a
batched 3-D workflow.

Choose SplineOps when
---------------------

* continuous-valued 2-D or 3-D NumPy data must be downsampled;
* coordinate, boundary, and output-shape semantics must be explicit;
* aliasing matters; or
* many arrays share one resize geometry.

Choose another tool when differentiable GPU execution or automatic
physical-space image metadata is required.  For ordinary display-image scaling,
OpenCV or Pillow may be simpler.  Categorical labels normally require
nearest-neighbour rather than spline-projection semantics.

Release status
--------------

.. list-table:: SplineOps 2.2 public maturity
   :header-rows: 1
   :widths: 24 16 60

   * - Module
     - Status
     - Public position
   * - Resize and ``ResizePlan``
     - Stable
     - Native N-D interpolation and projection antialiasing with a Python
       reference path.
   * - ``TensorSpline``
     - Stabilizing
     - Continuous tensor-product models evaluated at arbitrary coordinates.
   * - Other spline modules
     - Experimental
     - Available for research while their contracts and independent reference
       coverage mature.

SplineOps does not claim to be the fastest generic 2-D resizer.  Its strongest
position is explicit spline semantics, N-D projection antialiasing,
native/reference parity, and reusable fixed-geometry execution.  See
:doc:`performance` for measured wins, losses, and limitations.

The current flagship is the :doc:`selma3d-vessels-study`: on 18 held-out
expert-labelled 3-D microscopy patches, SplineOps passed a predeclared narrow
quality-and-speed superiority rule against named SciPy, scikit-image, and
PyTorch pipelines.  It is a preprocessing result, not segmentation or
universal resampling superiority.

Explore the package
-------------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Modules
      :link: modules
      :link-type: doc

      See every stable, stabilizing, and research module in one place.

   .. grid-item-card:: Examples
      :link: examples
      :link-type: doc

      Browse the executable gallery by module or complete workflow.

   .. grid-item-card:: Claims and evidence
      :link: evidence
      :link-type: doc

      Read the approved wording, limitations, studies, and interactive demo.

Research lineage
----------------

SplineOps modernizes spline methods developed across the `Biomedical Imaging
Group at EPFL <https://bigwww.epfl.ch/>`_ and its collaborators.  Method
citations, implementation history, and source provenance are recorded in
:doc:`provenance`.

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started
   modules
   examples
   evidence
   project
   api/index
