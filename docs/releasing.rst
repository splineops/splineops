Release validation
==================

Release candidates use a prerelease version such as ``2.3.0rc1`` in
``pyproject.toml`` and a matching changelog entry. Module maturity labels and
performance claims remain separate from the package version.

Pull-request checks
-------------------

Relevant pull requests run the Python 3.11--3.13 test matrix on Linux, macOS,
and Windows, minimum runtime dependencies, quality checks, numerical benchmark
comparisons, and repeated API stability checks. The wheel workflow builds
Linux x86_64, Windows AMD64, macOS Intel, and macOS arm64 packages for the three
Python versions. Hardware-dependent timing ratios are recorded as evidence;
the new LS/affine comparisons fail on numerical disagreement rather than
requiring a universal speedup.

Combined workflow timings use calibrated batches and alternating paired
samples; their artifacts include raw measurements and variability. Check
these diagnostics when a timing threshold fails. Numerical failures remain
blocking, and a passing timing rerun alone does not resolve the original
failure. The measurement protocol is described in :doc:`performance`.

Every wheel is installed outside the checkout before
``scripts/check_installed_wheel.py`` runs. This imports the actual native
extension, compares cubic LS with the independent Python projection,
compares affine backends and coefficient preparation across modes and
precisions, and checks packaged data and the demo entry point.

Validation without publication
------------------------------

In GitHub Actions, manually run **Validate and publish release** on the desired
branch or revision. Manual dispatch runs all validation and builds artifacts;
it never enters the publishing job. Once this workflow is on the default
branch, the Actions interface exposes its manual-run control.

The release workflow calls the test, quality, stability, and wheel workflows
using local references, so they run from the same commit. This is the
`GitHub local reusable-workflow contract
<https://github.blog/changelog/2022-01-25-github-actions-reusable-workflows-can-be-referenced-locally/>`_.
The artifact audit requires all twelve distinct Python/platform combinations,
matching filename and package metadata versions, a native binary and packaged
data in each wheel, and a source archive containing the wheel checks without
Python cache files.

Local validation can supplement the hosted checks:

.. code-block:: shell

   python -m pytest -q
   python -m build
   python scripts/audit_release_artifacts.py --version 2.3.0

Install the resulting wheel into a fresh virtual environment and run
``scripts/check_installed_wheel.py`` with that environment's Python. The
script rejects source imports and verifies the version against this checkout.
An audit without ``--complete`` validates a local build; it does not claim
coverage of the full release wheel matrix.

Publishing
----------

After review and successful checks, finalize the version and changelog, then
create a matching ``v<version>`` tag on the reviewed commit. A tag such as
``v2.3.0rc1`` publishes a prerelease; ``v2.3.0`` publishes the stable release.
Tagging starts validation again. The publishing job requires the complete
tests, quality checks, standard stability soak, and artifact audit to succeed.
Only this publishing job has permission to request the PyPI identity token.

Do not bypass failed checks with an unvalidated upload. Diagnose numerical
failures before changing tolerances; record legitimate precision-dependent
contracts and keep independent accuracy checks. Preserve benchmark results
with their environment and revision so release claims remain reproducible.
