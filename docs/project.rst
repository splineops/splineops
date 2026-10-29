About
=====

SplineOps brings spline methods from EPFL's Biomedical Imaging Group and its
collaborators into a Python library for scientific arrays. The project is
distributed under the BSD 3-Clause license.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: API stability and support
      :link: project-status
      :link-type: doc

      Supported modules, precision and boundary contracts, and compatibility
      commitments for the 2.x series.

   .. grid-item-card:: Research and attribution
      :link: provenance
      :link-type: doc

      Original methods, authors, implementation sources, and references
      for citing the work behind SplineOps.

   .. grid-item-card:: Release notes
      :link: https://github.com/splineops/splineops/blob/main/CHANGELOG.md

      New features, numerical corrections, compatibility notes, and
      changes between versions.

Get help or contribute
----------------------

Report a problem or propose an improvement in the
`GitHub issue tracker <https://github.com/splineops/splineops/issues>`_. For a
numerical issue, include a small reproducible array, the call you made, the
expected result, and your Python, NumPy, SciPy, and SplineOps versions.

To work on the library, clone the
`repository <https://github.com/splineops/splineops>`_ and install its
development dependencies:

.. code-block:: shell

   python -m pip install -e ".[dev,docs-core]"
   python -m pytest -q
   python -m tox -e format-check,type

Numerical changes should include a regression against an independent formula,
reference implementation, or invariant. See :doc:`internal-contracts` before
sharing numerical machinery across modules, and :doc:`releasing` for the
full documentation and distribution checks.

Maintainer records
------------------

The :doc:`graduation-audits` explains the current stability decision.
:doc:`stability-soak` records repeated public-API workflows. July's
:doc:`roadmap` and :doc:`progress` remain available as historical engineering
records; the current support contract is :doc:`project-status`.

.. toctree::
   :hidden:
   :maxdepth: 1

   project-status
   backend-support
   provenance
   releasing
   graduation-audits
   internal-contracts
   stability-soak
   roadmap
   progress
