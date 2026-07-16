Installation
============

SplineOps requires Python 3.11 or newer.  Install the current release from
PyPI:

.. code-block:: shell

   python -m pip install splineops

Published wheels include the native resize extension on supported platforms.
The Python resize reference remains available for parity checks and build
environments without the extension.

Verify the installation
-----------------------

.. code-block:: python

   import importlib.util
   import splineops

   print(splineops.__version__)
   print(importlib.util.find_spec("splineops._lsresize") is not None)

The second line reports whether the native CPU resize backend is present.

Backend control
---------------

``SPLINEOPS_ACCEL=never`` forces the Python resize reference.
``SPLINEOPS_ACCEL=always`` requires the native extension and raises an error if
it cannot be imported.  The default, ``auto``, uses the native backend when it
is available.

Development installation
------------------------

.. code-block:: shell

   git clone https://github.com/splineops/splineops.git
   cd splineops
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e '.[dev]'
   python -m pytest -q

Optional dependencies and exact backend limitations are listed in
:doc:`../backend-support`.
