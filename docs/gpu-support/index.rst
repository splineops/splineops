GPU Support
===========

This guide demonstrates how to use SplineOps with CuPy for GPU interoperability.

GPU Interoperability Using CuPy
-------------------------------

This example demonstrates GPU interoperability using CuPy and tensor spline interpolation. The following steps are performed:

1. Generate random data and coordinates.

2. Create tensor splines using NumPy and CuPy.

3. Evaluate the splines on a set of coordinates.

4. Compute the absolute difference and mean square error between the NumPy and CuPy evaluations.

5. Plot the results.

Imports
-------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 24-27

Data type
---------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 35-35

Create random data samples and corresponding coordinates
--------------------------------------------------------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 41-49

Tensor spline bases and modes
-----------------------------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 57-58

Create tensor spline from NumPy data
------------------------------------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 64-68

Create tensor spline from CuPy data for GPU computations
--------------------------------------------------------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 76-80

Create evaluation coordinates (extended and oversampled in this case)
---------------------------------------------------------------------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 86-92

Evaluate using NumPy
--------------------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 98-99

Evaluate using CuPy
-------------------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 107-108

Compute difference
------------------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 114-118

The results on terminal:

.. code-block:: none

   Maximum absolute difference: 1.6689300537109375e-06
   Mean square error: 1.372329577449885e-13

Plot results
------------

.. literalinclude:: GPU_Interoperability_Using_Cupy.py
   :language: python
   :lines: 124-140

.. image:: ../_static/GPU_Interoperability_Using_Cupy_Plot.png
   :alt: Plot showing results of GPU Interoperability using CuPy

Download the full script
------------------------

You can download the full Python script here:

:download:`Download full Python script <GPU_Interoperability_Using_Cupy.py>`
