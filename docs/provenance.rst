Method and source provenance
============================

SplineOps stands on a distinguished body of spline research and scientific
software.  Preserving that lineage means recording who developed a method,
which source informed an implementation, what license accompanied it, and how
the modern code was validated.

This inventory is an engineering record, not a legal opinion.  On 2026-07-15,
the project maintainer confirmed that the repository has the necessary rights
and permissions for the implementations distributed here.  Historical notices
remain recorded because attribution and method lineage are valuable even when
they are not release blockers.

Maintainer decision
-------------------

The provenance decision for the current SplineOps distribution is **cleared**.
The archived ``legacy_code`` collection may contain mixed historical notices,
but the maintainer has confirmed that those notices do not create a licensing
or redistribution issue for the implementations shipped by SplineOps.  Future
ports should still record their exact source and authorship rather than treating
the whole archive as one undifferentiated codebase.

Current inventory
-----------------

.. list-table:: Shipped Python and native modules
   :header-rows: 1
   :widths: 18 29 20 33

   * - Module
     - Method or source lineage
     - Implementation history
     - Provenance status
   * - ``spline_interpolation``
     - Cardinal tensor-product splines, B-splines, O-MOMS, and extension-mode
       machinery developed in the EPFL spline-processing lineage.
     - Present in the earlier ``bssp`` package and renamed to SplineOps in
       2024; principal repository authors Dimitris Perdios and Pablo
       Garcia-Amorena.
     - Covered by the repository BSD notices; basis-by-basis publication
       citations and contributor records still need consolidation.
   * - ``resize`` and native ``lsresize``
     - Muñoz Barrutia/Blu/Unser spline resizing, with production projection
       choices informed by oblique-projection work and the historical
       ``Resize.java`` implementation.
     - Python and C++ implementations developed and extensively revised in
       this repository, including the v2 direct compact cross-Gram pipeline.
     - Maintainer-cleared for distribution; method citations and historical
       Java lineage are documented.
   * - ``affine``
     - Rotation through continuous tensor-spline evaluation.
     - Python implementation added in this repository in 2025.
     - Repository BSD applies; mathematical citations and authorship detail
       should be added to the module documentation.
   * - ``differentials``
     - Cubic-spline gradients, Laplacians, and Hessian features corresponding
       to EPFL's historical ImageJ ``Differentials_`` plugin.
     - Python port added in this repository in 2025.
     - Maintainer-cleared for distribution.  The historical Java notice is
       retained in the archive and acknowledged as part of the lineage.
   * - ``smoothing_splines``
     - Fractional spline autocorrelation and smoothing methods associated with
       Unser and Blu; historical MATLAB files exist in ``legacy_code``.
     - Python implementations assembled in this repository in 2025.
     - Maintainer-cleared for distribution; method publications are cited in
       the user guide.  File-level derivation notes can still be enriched.
   * - ``adaptive_regression_splines``
     - Sparse piecewise-linear regression and total-variation denoising,
       attributed in source to Thomas Debarre and related published work.
     - Python implementation added in this repository in 2025.
     - Maintainer-cleared for distribution; author and publication are
       identified, with room for a more detailed implementation history.
   * - ``multiscale.pyramid``
     - Polynomial spline pyramids and centered pyramids; historical C sources
       are archived in ``legacy_code/pyramids``.
     - Python implementation added in this repository in 2025.
     - Maintainer-cleared for distribution; historical C sources remain
       identified for attribution.
   * - ``multiscale.wavelets``
     - Haar and spline wavelet implementations corresponding to archived Java
       sources.  The limited-precision order-5 table matches
       `DeconvolutionLab2's SplineFilter at revision e9af0ab
       <https://github.com/Biomedical-Imaging-Group/DeconvolutionLab2/blob/e9af0aba493ba137d70877154648e5583e376a81/src/main/java/wavelets/spline/SplineFilter.java#L141-L185>`_.
     - Python implementation added and revised in this repository in 2025.
     - Maintainer-cleared for distribution.  Mixed notices in archived source
       remain documented and are not represented as the license of the current
       Python implementation.  Order 5 is explicitly classified as bounded
       approximate rather than perfect reconstruction.

Historical archive
------------------

The separate ``legacy_code`` repository is a source archive, not a uniform
previous SplineOps release.  It contains Java, C, and MATLAB material from
different authors and eras, including GPL and research-use notices.  The
maintainer decision above clears the current distribution; the archive's mixed
history is still described accurately rather than flattened into one license.

Required record for future ports
--------------------------------

Before importing or translating additional legacy material, record:

#. Original file, upstream location, and immutable revision or retrieval date.
#. Original authors and copyright holders.
#. Every file-level and repository-level license notice.
#. Publications defining the method.
#. Whether the new implementation is a translation, behavioral reimplementation,
   or independent implementation from the paper.
#. The tests or fixtures used to validate equivalence.
#. The maintainer responsible for explaining and reviewing the implementation.
#. Any automated or AI-assisted development provenance required by the target
   project or publication venue.

Ongoing record improvements
---------------------------

* Trace smoothing and pyramid functions at file level rather than relying on
  directory-level similarity.
* Add publication citations and explicit contributor acknowledgements to each
  module instead of relying only on the package author list.
* Preserve AI/LLM disclosure in any future upstream contribution where project
  policy requires it.

These are documentation-quality improvements, not licensing blockers.  Module
maturity remains governed by numerical contracts, validation, maintenance, and
performance evidence described in :doc:`project-status`.
