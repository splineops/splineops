Decompose
=========

.. currentmodule:: splineops

Overview
--------
The `decompose` module in `splineops` provides a suite of algorithms for generating multiresolution representations of signals and images using spline pyramid decomposition. 
By modeling a signal as a continuously defined function reconstructed from its discrete samples via polynomial spline interpolation, the module enables both reduction (downsampling) 
and expansion (upsampling) operations. These operations form the basis for constructing error pyramids and establishing a connection with wavelet transforms.

Mathematical Background
-------------------------
Spline Pyramid Decomposition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A signal is represented as a continuous function based on its discrete samples:

.. math::

    f(x) = \sum_{k} c[k] \cdot \phi(x-k),

where :math:`\phi(x)` denotes a polynomial spline basis function. The spline model is interpolating and completely defined by the sample values. This representation facilitates 
the derivation of the REDUCE and EXPAND operators, which are essential for the pyramid construction.

REDUCE and EXPAND Operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **REDUCE Operator:**  
  The REDUCE operator filters and downsamples a signal by a factor of two. It minimizes the approximation error in a least squares sense, thereby ensuring that the reduced signal 
  maintains an optimal representation of the original. For centered pyramids, a variant named `ReduceCentered_1D` is used to accommodate grid shifts.

- **EXPAND Operator:**  
  The EXPAND operator up-samples a signal by a factor of two. It fills in the missing samples by applying an interpolation filter derived from the underlying spline model. In 
  the case of centered pyramids, the `ExpandCentered_1D` function is employed to correctly re-align the finer grid points.

Wavelets and the Error Pyramid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A close relationship exists between spline pyramids and wavelet transforms. The error pyramid is computed as:

.. math::

    \text{error} = f - \text{EXPAND}(\text{REDUCE}(f)),

which represents the loss of information during the reduction process. This error signal is analogous to the wavelet coefficients, capturing the details that are discarded in 
each level of the decomposition.

Implementation Details
----------------------
The `Decompose` class implements these operations in two major stages:

1. **Spline Interpolation:**  
   The input signal is converted into a continuous function by computing its spline interpolation coefficients. This step is crucial to ensure that subsequent operations are 
   performed on a smooth representation.

2. **Multiresolution Decomposition:**  
   - **REDUCE Operation:** The function `Reduce_1D` (or `ReduceCentered_1D` for centered grids) filters and downsamples the signal, constructing a coarser approximation.
   - **EXPAND Operation:** The function `Expand_1D` (or `ExpandCentered_1D` for centered grids) up-samples the reduced signal, interpolating the missing values.
   - **Error Pyramid:** The difference between the original signal and its reconstructed version (via EXPAND(REDUCE(signal))) yields the error pyramid, which is essential for 
   understanding the details removed during the decomposition and is tightly linked to wavelet analysis.

Users can customize the underlying spline degree (typically n=3) and choose between different error measures (discrete or continuous norms). This flexibility allows the module 
to be adapted for various applications in signal and image processing.

Decompose Example
------------------
* :ref:`sphx_glr_auto_examples_009_using_decompose_module.py`

References
----------
- Unser, M. (1999). *Splines: A Perfect Fit for Signal and Image Processing*, IEEE Signal Processing Magazine, vol. 16, no. 6, pp. 22-38.
- Unser, M., Aldroubi, A., & Eden, M. (1993). *B-Spline Signal Processing: Part II--Efficient Design and Applications*, IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 834-848.
- Unser, M., Aldroubi, A., & Eden, M. (1993). *The L2-Polynomial Spline Pyramid*, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 15, no. 4, pp. 364-379.
- Brigger, P., Müller, F., Illgner, K., & Unser, M. (1999). *Centered Pyramids*, IEEE Transactions on Image Processing, vol. 8, no. 9, pp. 1254-1264.
- Burt, P. J., & Adelson, E. H. (1983). *The Laplacian Pyramid as a Compact Code*, IEEE Transactions on Communication, vol. COM-31, no. 4, pp. 337-345.

.. note::
   The `decompose` module is optimized for signals and images reconstructed as continuous functions from discrete samples. For optimal performance, ensure that input data is 
   properly normalized and that spline parameters are appropriately selected.