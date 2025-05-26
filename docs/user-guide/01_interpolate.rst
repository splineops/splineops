Interpolate
===========

.. currentmodule:: splineops


Tedious Construction of Polynomial Splines
------------------------------------------

A polynomial spline is a continuously defined function made of polynomial pieces of nonnegative integer degree :math:`N`. 
What makes such a spline special is that all pieces connect smoothly when :math:`N\geq1`. In one dimension, let some piece be defined over :math:`x_{-1}\leq x\leq x_{0}` 
by the polynomial

.. math::
   
   a(x)=a_{0}+\sum_{n=1}^{N}\,a_{n}\,x^{n}

and let an adjacent piece be defined over :math:`x_{0}\leq x\leq x_{1}` as

.. math::

   b(x)=b_{0}+\sum_{n=1}^{N}\,b_{n}\,x^{n}.

Then, to be a legitimate part of a polynomial spline, the two pieces :math:`a` and :math:`b` must join continuously at :math:`x_{0}`. The first derivatives must 
also be in agreement; likewise, the next derivatives are in agreement, too, up to the :math:`\left(N-1\right)` th derivative. (However, the derivatives of 
order :math:`N` of :math:`a` and :math:`b` are allowed to disagree at :math:`x_{0}`.)

.. math::

   a(x_{0})=b(x_{0})

.. math::

   \dot{a}(x_{0})=\dot{b}(x_{0})

.. math::

   \ddot{a}(x_{0})=\ddot{b}(x_{0})

.. math::

   \vdots

.. math::

   {\mathrm{d}}^{N-1}a(x_{0})/{\mathrm{d}}x^{N-1}={\mathrm{d}}^{N-1}b(x_{0})/{\mathrm{d}}x^{N-1}.


In one dimension, each polynomial piece has a left neighbor and a right neighbor. With many pieces and a large polynomial degree, the explicit construction of 
a spline would make for a tedious task that must honor many constraints of continuity. In the interpolation context, one additionally asks that the spline, which 
we now call :math:`f`, reproduces 
the list :math:`\{y[k]\}_{k=0}^{K-1}` of :math:`K` samples at the set :math:`\{x[k]\}_{k=0}^{K-1}` of :math:`K` sampling locations, with

.. math::

   f(x[k])=y[k].


Practical Construction of Polynomial Splines
--------------------------------------------

The construction of a polynomial spline is greatly simplified under some appropriate assumptions, namely, that the sampling locations are regularly spaced 
exactly one unit apart and coincide with the integers; it is also especially convenient to assume that the polynomial pieces are of unit length, too, and that 
their extremities coincide either with the integers (odd :math:`N`) or with the half integers (even :math:`N`). Under these simplifying assumptions, algorithms 
were championed in [1]_, [2]_, [3]_.
to honor all constraints (continuity and interpolation), with the computational effort :math:`{\mathcal{O}}(K\,\left\lfloor N/2\right\rfloor)` 
to build a polynomial spline of degree :math:`N` out of :math:`K` samples.


Once the spline is built, it can be interrogated at any continuous argument :math:`x\in{\mathbb{R}}` to yield the value :math:`f(x)`, with the per-point 
computational effort :math:`{\mathcal{O}}(N^{2})`.


In the SplineOps implementation, the polynomial pieces are never expressed explicitly. Instead, so-called spline coefficients :math:`c` are used to represent 
the spline. Moreover, the polynomial pieces that make the spline are decomposed in a basis of integer translates of B-splines, which are functions :math:`\beta^{N}` 
that are uniquely defined as even-symmetric splines of unit integral and least support. There, :math:`N` is a superscript (not an exponent) that gives the degree of the B-spline.


Tensor Product of Splines
-------------------------

In one dimension, we write a generic spline in terms of :math:`x\in{\mathbb{R}}` as

.. math::

   f(x)=\sum_{k\in{\mathbb{Z}}}\,c[k]\,\beta^{N}(x-k).

In multiple dimensions, we consider tensor-product splines. For instance, a continuously defined grayscale spline image that interpolates the 
array :math:`y[k_{1},k_{2}]` of width :math:`W` and height :math:`H` writes

.. math::

   f(x_{1},x_{2})=\sum_{q_{2}\in{\mathbb{Z}}}\,\left(\sum_{q_{1}\in{\mathbb{Z}}}\,c[q_{1},q_{2}]\,\beta^{N}(x_{1}-q_{1})\right)\,\beta^{N}(x_{2}-q_{2})

and satisfies that :math:`f(k_{1},k_{2})=y[k_{1},k_{2}]`. Pay attention that :math:`(x_{1},x_{2})\in{\mathbb{R}}^{2}` is a continuously defined two-component vector, 
while :math:`k_{1}\in[0\ldots W-1]` and :math:`k_{2}\in[0\ldots H-1]` are integers. Similarly, a spline volume would write

.. math::

   f({\mathbf{x}})=\sum_{q_{3}\in{\mathbb{Z}}}\,\left(\sum_{q_{2}\in{\mathbb{Z}}}\,\left(\sum_{q_{1}\in{\mathbb{Z}}}\,c[{\mathbf{q}}]\,\beta^{N}(x_{1}-q_{1})\right)\,\beta^{N}(x_{2}-q_{2})\right)\,\beta^{N}(x_{3}-q_{3}).


Extension of Splines Beyond Their Known Support
-----------------------------------------------

It is remarkable that splines can be interrogated at any coordinate, not only at a non-integer one, but also at one that would be far remote from the support of 
the samples that define, say, an image. This is achieved by imposing some structural organization to the spline coefficients. Typically, this organization takes 
some form of periodicity, possibly combined with local reversals of chunks of samples. It allows us to predict the value of a spline coefficient :math:`c[{\mathbf{k}}]` 
for any :math:`{\mathbf{k}}\in{\mathbb{Z}}^{2}` from the sole knowledge 
of :math:`c[{\mathbf{k}}]` for :math:`k_{1}\in[0\ldots W-1]` and :math:`k_{2}\in[0\ldots H-1]`.


Quality of Approximation
------------------------

The computational burden increases with the degree of a spline, but so does the quality of the representation. Indeed, consider a one-dimensional ground-truth generic 
function (not necessarily a spline) that is differentiable sufficiently many times. Now, it is customary to know this function only through its samples and to let a 
spline of degree :math:`N` interpolate them. The higher the degree, the closer to the local Taylor expansion of the ground truth the spline gets the chance to be 
because :math:`N+1` terms can potentially match. This argument is mere hand-waiving; its merit is that it relies on no more than calculus to make plausible that 
the continuously defined spline represents the continuously defined ground truth better when the degree rises.

To understand the true mechanism that links degree and quality, one needs to master the theory of approximation, which in turn relies on the Fourier theory, the theory 
of distributions, the measure theory, the theory of sampling, and the theory of integration, among others, all of them being advanced mathematical 
topics [4]_, [5]_.

While one-dimensional splines are well-understood, the approximation properties of tensor-product splines are less so. Yet, numerous practical experiments have led to 
the conclusion that a tensor-product cubic spline offers a good tradeoff between computational effort and quality, being significantly more accurate than what linear 
interpolation offers.


Interpolation Examples
----------------------

* :ref:`sphx_glr_auto_examples_001_usage_of_the_class_tensorspline.py`
* :ref:`sphx_glr_auto_examples_002_interpolate_1D_samples.py`
* :ref:`sphx_glr_auto_examples_003_interpolate_2D_images.py`

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
