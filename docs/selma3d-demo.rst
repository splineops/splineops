Interactive SELMA3D comparison
==============================

The napari demo turns the flagship held-out microvessel study into an
interactive comparison without redistributing source microscopy images.  A
curtain slider reveals SplineOps from the left while one selected method
remains on the right.

Install and launch
------------------

Install the optional demo dependencies and run the installed command:

.. code-block:: shell

   python -m pip install 'splineops[selma3d-demo]'
   splineops-selma3d-demo

The default installation compares SplineOps with SciPy, scikit-image, and
SplineOps cubic interpolation without antialiasing.  Add the optional PyTorch
area comparison with:

.. code-block:: shell

   python -m pip install 'splineops[selma3d-demo-all]'
   splineops-selma3d-demo --comparisons torch_area scipy_gaussian skimage_resize

From a repository checkout, the compatibility launcher is also available:

.. code-block:: shell

   python demos/selma3d_napari.py

The first run downloads only the selected WGA channel and expert vessel mask
from BioImage Archive accession ``S-BIAD1196``.  It verifies pinned SHA-256
digests and caches the files under ``~/.cache/splineops/selma3d``.  Use
``--help`` to select another held-out patch, restrict comparison methods, or
provide an existing ``VessAP_vessel`` directory.

What the viewer shows
---------------------

* **Reveal SplineOps** moves from comparison-only at 0% to SplineOps-only at
  100%.
* The method selector includes SciPy Gaussian+cubic, scikit-image cubic
  antialiasing, and SplineOps cubic without antialiasing.  PyTorch area is
  available with the ``selma3d-demo-all`` installation.
* The expert mask is composited on each method's documented endpoint or
  half-pixel sampling grid.
* Local single-patch timings and ROC AUC are displayed separately from the
  frozen 18-patch evidence.
* Napari's first dimension slider browses all 50 retained planes.

Patch 013 is the default because it gives clear visual separation.  It is a
post-study presentation choice, not a newly held-out selection.  The
18-patch aggregate result remains the primary evidence.

Scope and data terms
--------------------

ROC AUC measures expert-labelled vessel/background voxel ranking; it is not a
segmentation metric.  The result covers one channel, one lateral twofold
geometry, named methods, and the recorded machine.  Read :doc:`claims` and
:doc:`selma3d-vessels-study` before quoting the result.

The official source records contain conflicting CC BY 4.0 and CC BY-NC
metadata.  SplineOps follows the stricter CC BY-NC interpretation and
redistributes no source images.  The packaged demo contains only checksums and
a non-image aggregate evidence summary.
