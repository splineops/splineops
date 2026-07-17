Claims and evidence
===================

This page is the canonical registry for public SplineOps claims.  A benchmark
result may be quoted only with the scope stated here or a narrower scope.
Machine-specific timings are evidence from recorded runs, not portable
guarantees.

Primary product claim
---------------------

The general product statement is:

   **Projection-based antialiased resizing for N-D scientific arrays.**

SplineOps provides native CPU N-D interpolation and projection antialiasing
for NumPy arrays, with explicit axes, output geometry, boundary behavior,
output buffers, reusable plans, and a maintained Python reference path.  This
is a capability description rather than a claim to beat every resize library.

Validated competitive claim
---------------------------

The strongest approved competitive statement is:

   For the tested twofold lateral coarsening of 18 held-out SELMA3D WGA
   microvessel patches, SplineOps cubic projection had higher mean
   vessel-ranking ROC AUC than the named SciPy, scikit-image, and PyTorch
   pipelines
   under a predeclared family-wise rule.  On the recorded one-thread CPU run it
   was 12.49 times, 15.17 times, and 1.36 times faster, respectively.

The exact geometry was ``500 x 500 x 50`` to ``250 x 250 x 50``.  The primary
endpoint was expert-labelled vessel-versus-background voxel-ranking ROC AUC,
not segmentation.  Endpoint and half-pixel methods were each scored against
labels on their documented grid.  One patch favoured SciPy; the decision is an
aggregate paired result.  See :doc:`selma3d-vessels-study` and the
:doc:`selma3d-demo`.

Supporting evidence
-------------------

Native acceleration
~~~~~~~~~~~~~~~~~~~

On the dated formal resize suite, the native backend was 53.63 times faster
than the maintained Python reference at the median of 46 overlapping cases.
The oblique-antialiasing subset measured a 47.42-times median over 13 overlaps.
These are internal acceleration results, not competitive superiority.

Controlled 3-D fields
~~~~~~~~~~~~~~~~~~~~~

Across 72 controlled smooth-field cases with meaningful above-output-Nyquist
content, SplineOps projection reduced analytical-target NRMSE by 54.7% versus
the closest frozen generic baseline.  A post-hoc SciPy polyphase FIR pipeline
was much more accurate and about 16 times slower, so the study does not support
broad scientific-resampling superiority.  See :doc:`wavefield3d-study`.

MiniVess vessel overviews
~~~~~~~~~~~~~~~~~~~~~~~~~

For the tested 8-times lateral MiniVess overview, all five paired quality
margins favoured SplineOps.  The predeclared joint quality-at-speed claim
failed because the PyTorch-area and SciPy-polyphase speed conditions failed.
This is supporting quality evidence, not a passed joint superiority claim.

Explicit nonclaims
------------------

Current evidence does not establish that SplineOps:

* is the fastest generic 2-D image resizer;
* is universally the most accurate scientific resampler;
* improves segmentation, biological measurements, or clinical outcomes;
* wins every microscopy patch, dataset, geometry, or boundary condition;
* outperforms OpenCV for ordinary display resizing;
* has general GPU performance superiority; or
* is more accurate than a well-designed polyphase/FIR method.

Reusable short form
-------------------

For space-constrained material, use:

   On 18 held-out expert-labelled SELMA3D microvessel patches, SplineOps
   demonstrated narrow quality-at-speed superiority over named SciPy,
   scikit-image, and PyTorch pipelines for the tested twofold lateral
   coarsening.

Link this sentence back to the full claim above.  Do not shorten it to
“SplineOps is the best image resizer” or “SplineOps segments vessels better.”
