Backend support contract
========================

SplineOps uses the word *supported* only when dispatch, numerical tests,
packaging, and continuous integration cover the same configuration.  Merely
accepting an array type is interoperability evidence, not a package-wide
backend promise.

Current matrix
--------------

.. list-table:: Tested array and execution backends
   :header-rows: 1
   :widths: 18 20 22 40

   * - Backend
     - Public scope
     - Status
     - Exact contract
   * - NumPy
     - All public modules
     - Supported
     - The Python/OS test matrix exercises NumPy arrays, documented dtypes,
       output buffers, and module-specific boundaries.
   * - Native C++
     - Resize only
     - Supported CPU accelerator
     - Implements the resize contract, with Python-reference parity tests and
       automatic or explicitly controlled dispatch.  It is not a generic
       ``TensorSpline`` or GPU backend.
   * - CuPy
     - Portions of ``TensorSpline`` only
     - Experimental interoperability
     - No CuPy configuration is currently advertised as supported because no
       reliable GPU runner exercises it in CI.  Other modules require NumPy.

CuPy boundary
-------------

``TensorSpline`` recognizes an actual ``cupy.ndarray`` through explicit type
dispatch.  Construction data, construction coordinates, query coordinates,
output buffers, and reusable geometry must remain on one array backend.  An
object is not accepted merely because it exposes ``__cuda_array_interface__``;
this prevents accidental duck-typed dispatch into an untested implementation.

The present implementation has materially different paths:

* periodic coefficient filtering has a CuPy FFT path;
* zero-boundary finite-support coefficient filtering currently transfers data
  to a SciPy CPU solve and returns it to CuPy; and
* mirror filtering and the full basis/mode/query matrix do not yet have
  dedicated GPU CI evidence.

Consequently, SplineOps does not promise device residency, GPU acceleration,
or complete basis/mode support for CuPy.  The optional ``dev_cupy`` dependency
group exists for maintainers evaluating that experimental surface; installing
it does not change the support status.

Graduation decision
-------------------

CuPy will be promoted only after a reliable GPU runner covers construction,
sample reproduction, point and grid queries, dtypes, boundaries, reusable
geometry, buffers, and host-transfer checks.  Until then the honest choice is
to retain the useful experimental code, keep the dispatch narrow, and avoid
general GPU performance claims.  This decision does not affect
``TensorSpline``'s independent public identity or the supported NumPy path.
