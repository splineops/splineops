Competitive position and post-2.2 plan
=======================================

SplineOps 2.2 turns the stable resize path into a focused product: native,
projection-antialiased resizing for N-D scientific NumPy arrays.  The next
cycle should make that capability easier to adopt and independently reproduce
before starting another broad optimization or feature wave.

This page interprets the published evidence for product and development
planning.  It does not expand the approved claims in :doc:`claims`.

How much better is SplineOps?
-----------------------------

There is no honest library-wide multiplier.  The strongest demonstrated
advantage is for one important workload: CPU coarsening of continuous 3-D
scientific data when antialiasing quality and throughput matter together.

The held-out SELMA3D microvessel study measured a twofold lateral coarsening
from ``500 x 500 x 50`` to ``250 x 250 x 50``:

.. list-table:: Frozen SELMA3D quality-at-speed result
   :header-rows: 1
   :widths: 29 19 25 27

   * - Pipeline
     - Mean ROC AUC
     - SplineOps AUC difference
     - Recorded speedup
   * - SplineOps cubic projection
     - 0.927370
     - --
     - --
   * - SciPy Gaussian + cubic
     - 0.926648
     - +0.000722
     - 12.49x
   * - scikit-image cubic AA
     - 0.922168
     - +0.005202
     - 15.17x
   * - PyTorch area
     - 0.922184
     - +0.005186
     - 1.36x

The AUC difference from SciPy is small and should be presented that way.  The
aggregate paired result nevertheless passed the frozen family-wise quality
rule, as did the larger differences from scikit-image and PyTorch.  One patch
favoured SciPy.  The speed ratios are from the recorded one-thread CPU run,
not portable promises.  The endpoint is voxel-ranking ROC AUC, not
segmentation or a biological outcome.  See :doc:`selma3d-vessels-study` for
the protocol, confidence intervals, environment, and per-patch evidence.

The controlled 3-D field study supplies complementary evidence.  Across 72
smooth-field cases, SplineOps projection reduced analytical-target NRMSE by
54.7% relative to the closest frozen generic baseline.  It was about 11.56x
faster than the SciPy Gaussian pipeline and 12.85x faster than scikit-image by
geometric mean on that recorded workflow.  The post-hoc SciPy polyphase FIR
pipeline was far more accurate and about 16.14x slower.  This is evidence of a
useful quality--speed trade-off, not universal scientific-resampling
superiority.  See :doc:`wavefield3d-study`.

Where the product is strongest
------------------------------

SplineOps is differentiated by the combination of projection antialiasing,
native N-D CPU execution, explicit axes and output geometry, documented
boundary behavior, reusable plans, caller-owned output buffers, and a
maintained Python reference path.  The strongest fit is therefore:

* continuous-valued 2-D and 3-D NumPy data;
* anisotropic or non-integer scientific downsampling;
* repeated fixed-geometry processing of many arrays or frames;
* workflows where sampling grids and boundary semantics must be reviewable;
  and
* users who need reproducible quality and performance evidence rather than an
  opaque resize primitive.

Promising adoption areas include microscopy volume pyramids, scientific field
coarsening, tomography or medical-imaging preprocessing, and repeated
multiscale feature generation.  These are target workloads for evaluation,
not claims that every such application has already been validated.

Competitor boundaries
---------------------

The practical comparison is about fit, not a universal leaderboard:

.. list-table:: Choose by workload
   :header-rows: 1
   :widths: 30 35 35

   * - Need
     - SplineOps position
     - Likely alternative
   * - N-D NumPy downsampling with explicit spline projection
     - The stable core and strongest evidence-backed use case.
     - SciPy_ and scikit-image_ remain broad, familiar scientific tools with
       different filtering and grid contracts.
   * - Differentiable or accelerator-first resizing
     - General GPU superiority is not established; CuPy support is limited.
     - PyTorch_ or JAX_ when automatic differentiation, JIT compilation, or
       accelerator integration dominates.
   * - Ordinary display-image resizing
     - Not the primary market and not claimed fastest.
     - OpenCV_ or Pillow for established 2-D image pipelines.
   * - Maximum stopband rejection
     - Projection offers a fast compromise with a defined spline model.
     - A well-designed polyphase/FIR pipeline can be substantially more
       accurate when its additional runtime is acceptable.
   * - Automatic physical-space metadata management
     - Geometry is explicit, but image metadata is not managed automatically.
     - An ITK/SimpleITK-style imaging workflow may be a better fit.
   * - Categorical masks or labels
     - Spline projection is normally the wrong semantic choice.
     - Use nearest-neighbour or another label-preserving operation.

The short market position is:

   **High-quality, high-throughput downsampling for N-D scientific NumPy
   arrays where sampling semantics matter.**

It must not be shortened to “the fastest image resizer” or “the most accurate
scientific resampler.”

Post-2.2 execution plan
-----------------------

The immediate objective is to convert the validated technical advantage into
adoption and independent evidence.  Optimization continues only when profiles
from representative complete workflows identify a material bottleneck.

Phase 1: publish and observe
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Publish a concise technical article anchored on the reusable short claim in
  :doc:`claims`, with one installation command and the interactive
  :doc:`selma3d-demo`.
* Produce a short curtain-view capture and one static evidence graphic using
  the frozen artifacts; keep local demo measurements visually separate from
  the published 18-patch result.
* Add an issue template for reproduction reports that captures package
  versions, machine, thread settings, geometry, input checksums, timings, and
  numerical endpoints.
* Invite at least two scientific-imaging groups to run the demo or frozen
  benchmark and report installation or semantic friction.

Completion means a fresh environment can move from installation to an
interpretable comparison without consulting repository internals, and every
public number links back to its protocol and machine-readable artifact.

Phase 2: establish portability and independent replication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Run the frozen SELMA3D and controlled-field workflows on at least three
  distinct CPU/machine combinations across at least two operating-system
  families.
* Record throughput, setup cost, peak memory, thread settings, and plan
  amortization.  Report distributions and losses rather than only winning
  rows.
* Test a preregistered dataset outside the current microvessel evidence,
  selected before examining method outcomes.
* Keep the historical frozen comparisons intact when adding newer competitor
  versions; publish a new dated artifact rather than silently replacing the
  original evidence.

Completion means the project can distinguish portable conclusions from the
original development-machine timings and has at least one credible external
reproduction report.

Phase 3: integrate with scientific volume workflows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Prototype optional OME-Zarr, Dask, or xarray-facing examples without adding
  them as required dependencies of the stable core.
* Define chunk-boundary, halo, physical-spacing, and memory contracts before
  presenting chunked resizing as supported.
* Benchmark a representative volume workflow under an explicit memory limit,
  including storage and conversion overhead rather than only the resize
  kernel.
* Document round trips to metadata-owning imaging libraries instead of making
  SplineOps silently own metadata it does not understand.

Completion requires numerical agreement with the non-chunked reference on the
documented domain, a bounded-memory demonstration, and a clear statement of
which layer owns physical-space metadata.

Phase 4: stabilize selectively
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Consider ``TensorSpline`` for stable status only after dedicated CuPy CI and
  the remaining independent references close its graduation audit.
* Let affine and differentials complete a real compatibility soak before
  changing their experimental status.
* Optimize experimental modules only from complete-workflow profiles; do not
  trade away numerical contracts for isolated microbenchmark wins.
* Keep 2.2.x releases focused on corrections and compatibility.  Use 2.3 for
  adoption, portable evidence, and volume-workflow integration rather than a
  broad new algorithm wave.

Phase 5: make the evidence citable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Archive the protocols, scripts, frozen results, and environment manifests in
  a DOI-backed release.
* Add citation metadata and prepare a focused methods paper describing the
  projection model, implementation contract, and positive and negative
  validations.
* Seek an external reproduction or co-authored application note before making
  any broader application claim.

Proposed 2.3 boundary
---------------------

.. list-table:: Scope discipline for the next feature release
   :header-rows: 1
   :widths: 50 50

   * - In scope
     - Out of scope without new evidence
   * - Multi-machine resize evidence
     - Universal speed or accuracy claims
   * - One preregistered external-domain validation
     - Segmentation, biological, or clinical outcome claims
   * - Optional chunked-volume integration and memory contracts
     - Required heavyweight imaging dependencies
   * - Reproduction tooling and publication assets
     - A broad collection of new experimental algorithms
   * - ``TensorSpline`` graduation work
     - Premature promotion of affine, differentials, or CuPy support

Success should be measured by reproducible external runs, complete demo
workflows, documented integrations, and citations—not download counts alone.

.. _SciPy: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html
.. _scikit-image: https://scikit-image.org/docs/stable/api/skimage.transform.html#skimage.transform.resize
.. _PyTorch: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
.. _JAX: https://docs.jax.dev/en/latest/_autosummary/jax.image.resize.html
.. _OpenCV: https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html
