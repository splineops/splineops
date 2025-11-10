.. splineops/docs/user-guide/01_spline_interpolation.rst

Spline Interpolation
====================

.. currentmodule:: splineops

Spline Processing
-----------------

Splines are real functions that are *continuously* defined while being *parameterized* by discrete data. This gives access to many interesting operations that would otherwise be illegitimate with purely discrete data since these operations are truly valid in the continuum only, for instance:

* differentiation—gradients are often relied upon to detect edges in images, or to minimize some cost function, or in the handling of the differential equations of physical models, while discrete data give access to finite differences, not to gradients;
* arbitrary geometric transformations—it is very much desirable to be able to evaluate a function at any desired coordinate, while discrete data can be evaluated at the samples only;
* specific geometric transformations such as resizing—the aliasing inherent with the downsizing of data can be handled much more safely in the continuous domain than it can with discrete data.

There are many brands of splines. In the graphics world, one often relies on splines to represent curves, for instance with nonuniform rational B-splines. The SplineOps library is not meant to be used for such applications; in return, it is well-suited to the handling of data defined on a uniform Cartesian grid and offers highly successful tradeoffs between quality of representation and computational efficiency [1]_, [2]_, [3]_, [4]_, [5]_.

**Pros**

* Bridge between the discrete world and the continuous world.
* Tunable tradeoff between speed and quality.
* Efficient continuously defined representation of uniform data in multiple dimensions.

**Cons**

* The spline may overshoot/undershoot the data samples.
* Along a path, the spline may not be monotonous in regions where the data samples are.
* The spline is nonlocal, by which we mean that the update of just one data sample requires the update of the whole spline.

**Roadmap**

* In what follows, we first introduce *B-splines* as convenient building blocks of splines. We give a productive formula that defines polynomial B-splines and illustrate them.
* We then build a *spline* as an arbitrary combination of B-splines. At first, we do so with exuberance.
* Finally, we rein in splines and limit ourselves to *regular* splines. They allow one to efficiently interpolate discrete data.

B-Splines
---------

A one-dimensional polynomial B-spline is a member of a family of real functions :math:`\beta^{n}:{\mathbb{R}}\rightarrow{\mathbb{R}},x\mapsto\beta^{n}(x)` that are characterized by their degree :math:`n\in{\mathbb{N}}` (*e.g.*, linear, cubic). There, the degree :math:`n` is a superscript---as opposed to a power. Several equivalent explicit formulations of :math:`\beta^{n}` exist. One of them, valid for :math:`n\in{\mathbb{N}}_{>0}`, is

..  math::
    
    \beta^{n}(x)=\frac{1}{n!}\,\sum_{k=0}^{n+1}\,\left(-1\right)^{k}\,{n\choose k}\,{\mathrm{ReLu}}^{n}(x+\frac{n+1}{2}-k),

where :math:`x\in{\mathbb{R}}` is the argument of the B-spline and where one recognizes an interplay between signed binomial coefficients and the :math:`n`-th power of the celebrated :math:`{\mathrm{ReLu}}` function of artificial-intelligence fame, with :math:`{\mathrm{ReLu}}(x)=\max(x,0)` for all :math:`x\in{\mathbb{R}}.`

Here is the plot of a cubic B-spline.

..  image:: interpolatefig01.png
    :width: 288pt
    :align: center

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

B-splines have many relevant properties. Among them, the (technical) fact that they have an optimal *order of approximation* explains why these functions are so good at representing discrete data. In nearly every case of relevance, their most important (practical) property is that their *support* is finite.

Splines
-------

Now, we are going to do something bold. Let us sum together the functions of the previous figure.

..  image:: interpolatefig06.png
    :width: 288pt
    :align: center

We were able to create some combined function that seems to be kind of arbitrary. This combined function somehow retains the characteristics of B-splines, but it is no more a B-spline (the letter B stands for Basis); instead it is called a *spline* (without the B).

Interpolation
-------------

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

The answer is yes, we can go from discrete samples to continuously defined curve, but one needs to do it right. For instance, the weighting process is not trivial; the center panel of the figure above illustrates the fact that the value of a weight is usually not equal to the value of a sample—for a clear case, do inspect abscissa at 2.

Multidimensional Splines
------------------------

The class ``TensorSpline`` solves the difficulties for you in an efficient way and in multiple dimensions, for many degrees of splines. Internally, it considers the continuously defined :math:`d`-dimensional real function

..  math::

    f:{\mathbb{R}}^{d}\rightarrow{\mathbb{R}},{\mathbf{x}}\mapsto f({\mathbf{x}})=\sum_{{\mathbf{k}}\in{\mathbb{Z}}^{d}}\,c[{\mathbf{k}}]\,\prod_{p=1}^{d}\,\beta^{n}(x_{p}-k_{p}),

where :math:`{\mathbf{x}}` is the function argument in :math:`d` dimensions and :math:`c` is an infinite list of real coefficients with indices in :math:`d` dimensions, too. These coefficients are carefully tuned in such a way that

..  math::

    \forall{\mathbf{q}}\in\Omega:f({\mathbf{q}})=s[{\mathbf{q}}],

where the function :math:`f` is the spline and where the list :math:`s` contains the samples that we want to interpolate, as provided over a set :math:`\Omega\subset{\mathbb{N}}^{d}` of indices. Since this set is finite in practice while the coefficients :math:`c` must be defined for infinitely many indices, one has to invent values for those coefficients that are far away from :math:`\Omega`. Arbitrary recipes are followed to that effect. For instance, the argument ``modes`` of the class ``TensorSpline`` of this library allows one to choose from among a few recipes.

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
   Signal Processing, vol. 41, no. 2, pp. 821-833, February 1993.

.. [2] M. Unser, A. Aldroubi, M. Eden, 
   `B-Spline Signal Processing: Part II—Efficient Design and Applications <https://doi.org/10.1109/78.193221>`_, 
   IEEE Transactions 
   on Signal Processing, vol. 41, no. 2, pp. 834-848, February 1993.

.. [3] M. Unser, J. Zerubia, 
   `A Generalized Sampling Theory Without Band-Limiting Constraints <https://doi.org/10.1109/82.718806>`_, 
   IEEE Transactions on Circuits and 
   Systems—II: Analog and Digital Signal Processing, vol. 45, no. 8, pp. 959-969, August 1998.

.. [4] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22–38, November 1999.

.. [5] M. Unser, `Sampling—50 Years After Shannon <https://doi.org/10.1109/5.843002>`_, 
   Proceedings of the IEEE, vol. 88, no. 4, pp. 569-587, April 2000.