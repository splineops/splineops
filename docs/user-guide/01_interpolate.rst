.. splineops/docs/user-guide/01_interpolate.rst

Spline Interpolation
====================

.. currentmodule:: splineops

Spline Processing
-----------------

Splines are :math:`d`-dimensional real functions :math:`{\mathbb{R}}^{d}\rightarrow{\mathbb{R}}` that are continuously defined while being parameterized by discrete data :math:`{\mathbb{Z}}^{d}\rightarrow{\mathbb{R}}`. This gives access to many interesting operations that would otherwise be illegitimate with discrete data since these operations are truly valid in the continuum only, for instance:

* differentiation—gradients are often relied upon to detect edges in images, or to minimize some cost function, or in the handling of the differential equations of physical models;
* arbitrary geometric transformations—the spline can be evaluated at any desired coordinate;
* specific geometric transformations such as resizing—the aliasing inherent with the downsizing of data can be handled much more safely in the continuous domain than it can with the discrete data.

There are many brands of splines. In the graphics world, one often relies on splines to represent curves, for instance with nonuniform rational B-splines. The SplineOps library is not meant to be used for such applications; in return, it is well-suited to the handling of data defined on a uniform grid and offers highly successful tradeoffs between quality of representation and computational efficiency [1]_, [2]_, [3]_,  [4]_, [5]_.

**Pros**

* Bridge between the discrete world and the continuous world.
* Tunable tradeoff between speed and quality.
* Efficient continuously defined representation of uniform data in multiple dimensions.

**Cons**

* The spline may overshoot/undershoot the data samples.
* In 1D, the spline may not be monotonous in regions where the data samples are.
* The spline is nonlocal, by which we mean that the update of just one data sample still requires the update of the whole spline.

B-Splines
---------

Here is the plot of a *B-spline*.

..  image:: interpolatefig01.png
    :width: 288pt
    :align: center

A B-spline is a member of a family of real functions that are indexed by their *degree* (*e.g.*, linear, quadratic, cubic, quartic, quintic). For instance, the degree above is three.

Now, let us shift this B-spline horizontally by one third.

..  image:: interpolatefig02.png
    :width: 288pt
    :align: center

Moreover, let us shrink it by 60%.

..  image:: interpolatefig03.png
    :width: 288pt
    :align: center

Finally, let us multiply it by one fourth. This multiplicative step is called a *weighting* of the B-spline.

..  image:: interpolatefig04.png
    :width: 288pt
    :align: center

Likewise, we could play with any other combination of (shift, shrink, weight) to obtain a zoo of other functions, including some with negative weight. In the present case, all of them would be said to be cubic B-splines, up to their individual (shift, shrink, weight). Here are some.

..  image:: interpolatefig05.png
    :width: 288pt
    :align: center

Splines
-------

Now, we are going to do something bold. Let us sum these functions together.

..  image:: interpolatefig06.png
    :width: 288pt
    :align: center

We were able to create some combined function that seems to be kind of arbitrary. This combined function somehow retains the characteristics of B-splines, but it is no more a B-spline (the letter B stands for Basis); instead it is called a *spline* (without the B).

Spline Interpolation
--------------------

We are going to use splines to *interpolate* data, which is an operation whose purpose is to build a continuously defined function out of arbitrary discrete samples, in such a way that the samples of the built function are identical to the provided ones. To make our life simple, from now on we are going to consider only integer-valued shifts (the spline is then said to be a *regular* spline). Also, we are not going to either shrink or expand B-splines anymore, nor are we ever going to consider splines made of a mix of degrees. Yet, we want to maintain our free choice of the weights of the B-splines; this will give us sufficient freedom to build splines that can be shaped any way we want.

Here is some uniform spline (thick curve), along with its additive constituents (arbitrarily weighted and integer-shifted B-splines of same degree, thin curves).

..  image:: interpolatefig07.png
    :width: 288pt
    :align: center

We now mark with dots the samples at the integers of this particular spline.

..  image:: interpolatefig08.png
    :width: 288pt
    :align: center

These samples make for a discrete list of values (*i.e.*, the data samples). Since we want to interpolate these data, a natural question that arises is as follows: is there a way to reverse the process and to first impose a list of arbitrary sample values, then only to determine which B-spline weights are appropriate to build the uniform spline that happens to go through these samples? Here is the succession of operations we have in mind.

..  image:: interpolatefig09.png
    :width: 928pt
    :align: center

The answer is yes, we can go from discrete samples to continuously defined curve, but one needs to do it right. For instance, the weighting process is not trivial; the center panel of the figure above clearly illustrates the fact that the value of a weight is usually not equal to the value of a sample (for a clear case, do inspect abscissa at 2). The TensorSpline class solves the difficulties for you in an efficient way and in multiple dimensions, for many degrees of splines.

Interpolation Examples
----------------------

* :ref:`sphx_glr_auto_examples_01_quick-start_01_01_tensorspline_class.py`
* :ref:`sphx_glr_auto_examples_01_quick-start_01_02_spline_bases.py`
* :ref:`sphx_glr_auto_examples_01_quick-start_01_03_extension_modes.py`
* :ref:`sphx_glr_auto_examples_02_resampling_using_1d_interpolation_02_01_interpolate_1d_samples.py`
* :ref:`sphx_glr_auto_examples_02_resampling_using_1d_interpolation_02_02_resample_a_1d_spline.py`
* :ref:`sphx_glr_auto_examples_02_resampling_using_1d_interpolation_02_03_compare_different_splines.py`

References
----------

.. [1] M. Unser, A. Aldroubi, M. Eden, 
   `B-Spline Signal Processing: Part I—Theory <https://doi.org/10.1109/78.193220>`_, 
   IEEE-SPS best paper award, IEEE Transactions on 
   Signal Processing, vol. 41, no. 2, pp. 821–833, February 1993.

.. [2] M. Unser, A. Aldroubi, M. Eden, 
   `B-Spline Signal Processing: Part II—Efficient Design and Applications <https://doi.org/10.1109/78.193221>`_, 
   IEEE Transactions 
   on Signal Processing, vol. 41, no. 2, pp. 834–848, February 1993.

.. [3] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22–38, November 1999.

.. [4] M. Unser, J. Zerubia, 
   `A Generalized Sampling Theory Without Band-Limiting Constraints <https://doi.org/10.1109/82.718806>`_, 
   IEEE Transactions on Circuits and 
   Systems—II: Analog and Digital Signal Processing, vol. 45, no. 8, pp. 959–969, August 1998.

.. [5] M. Unser, `Sampling—50 Years After Shannon <https://doi.org/10.1109/5.843002>`_, 
   Proceedings of the IEEE, vol. 88, no. 4, pp. 569–587, April 2000.