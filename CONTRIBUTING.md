# Contributing

SplineOps welcomes reproducible bug reports, numerical improvements, examples,
and documentation fixes. For a numerical issue, include a small input, the call,
the expected result, and your SplineOps, NumPy, SciPy, and Python versions.

## Development setup

Clone the repository and run the following commands from its root. Use a
supported Python version, currently 3.11 through 3.13. Building an editable
installation also requires a C++17 compiler; on Windows, use Visual Studio
Build Tools with the C++ workload.

Create an environment with either standard Python or
[uv](https://docs.astral.sh/uv/getting-started/installation/):

```shell
python -m venv .venv
```

Or, with uv:

```shell
uv venv --python 3.12 .venv
```

Activate the environment on Linux or macOS:

```shell
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the development dependencies using the installer you chose:

```shell
python -m pip install -e ".[dev]"
```

Or, with uv:

```shell
uv pip install -e ".[dev]"
```

Both workflows use the same dependencies and build configuration in
`pyproject.toml`. Mamba is not required. uv is optional; CI continues to use pip
and tox. See the [uv environment guide](https://docs.astral.sh/uv/pip/environments/)
for selecting an interpreter without activating an environment.

## Checks before a pull request

Run the tests relevant to your change, then the formatting and type checks:

```shell
python -m pytest -q
python -m tox -e format-check,type
```

For numerical fixes, add a regression based on an independent formula,
reference implementation, or invariant. Exercise the affected precisions,
boundaries, and shapes. Check accuracy before comparing performance, and
record the workload and environment alongside timing results.

Public behavior must follow the
[API stability policy](https://splineops.github.io/project-status.html).
Document intentional numerical corrections in the changelog. Keep changes
focused so a reviewer can connect the reported problem, implementation, and
validation.

## Docstring standard

Use the [NumPy docstring format](https://numpydoc.readthedocs.io/en/latest/format.html)
for public modules, classes, functions, and methods. Sphinx renders this format
through its Napoleon extension.

Start with a short summary, then use `Parameters`, `Returns` or `Yields`,
`Raises`, `Notes`, `References`, and `Examples` where applicable. Omit sections
that add no information. Use `Examples`, not `Example`, as the section heading.

Type annotations describe Python types; docstrings must also explain array
shapes, precision rules, coordinate units, boundaries, and whether inputs can
be modified or outputs can share memory. Describe actual behavior rather than
inferring these contracts from annotations. Keep examples small, deterministic,
and based on public imports. Prefer lines around 75 characters for prose.

This illustrative evaluation-method docstring shows the format. Adapt its
contract and example to the method being documented:

```python
"""Evaluate a one-dimensional spline at query coordinates.

Parameters
----------
coordinates : ndarray of shape (n_points,)
    Finite query coordinates in the construction grid's physical units.

Returns
-------
values : ndarray of shape (n_points,)
    Interpolated values. The input coordinates are not modified.

Raises
------
ValueError
    If a query coordinate is not finite.

Examples
--------
>>> import numpy as np
>>> from splineops import TensorSpline
>>> spline = TensorSpline(
...     np.array([0.0, 2.0, 4.0]), np.arange(3.0),
...     bases="linear", modes="mirror",
... )
>>> spline(np.array([0.5, 1.5]))
array([1., 3.])
"""
```

## Documentation and examples

Install the documentation dependencies with your chosen installer:

```shell
python -m pip install -e ".[docs]"
```

Or:

```shell
uv pip install -e ".[docs]"
```

Build the documentation, execute the gallery, and validate sharing metadata:

```shell
python -m sphinx -b html docs docs/_build/html -W --keep-going
python scripts/check_docs_social_metadata.py docs/_build/html
```

Open `docs/_build/html/index.html` to inspect the result. For gallery or layout
changes, check the affected pages in both themes and at a narrow width. A
parse-only build does not validate executed examples or their thumbnails.

Gallery pages provide downloadable source files and notebooks for local use.
JupyterLite execution in the browser is not supported. The documentation does
not enable a JupyterLite extension or advertise a launch button. Adding it
would require a separately validated browser environment and compatible native
dependencies; ordinary local notebooks do not depend on JupyterLite.

See [release validation](https://splineops.github.io/releasing.html) for the
complete platform, wheel, stability, and publication checks.
