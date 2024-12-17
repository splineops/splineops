Resize
======

.. currentmodule:: splineops

Overview
--------

The `resize` function in the `splineops` library enables resizing (scaling) of N-dimensional data arrays using advanced spline-based methods. It
- supports arbitrary scaling factors and output sizes;

- handles multi-dimensional data (2D, 3D, and higher);

- has customizable spline degree and boundary extension modes.


Three resizing methods can be called:

- **Standard Interpolation**: Smooth, continuous interpolation.
- **Least-Squares Projection**: Optimal resizing with minimal approximation error. Slower than standard interpolation but with better interpolation quality. It requires float64 precision.
- **Oblique Projection**: Similar to Least-Squares projection, but faster and lower quality. It works well with with float32 precision

In general, it is recommended to use standard interpolation for most cases; in case higher accuracy is required and float64 precision is used, least-squares projection is recommended.
Oblique projection provides better balance of performance, speed and accuracy.

The `resize` module is suitable for a wide range of applications, including Image processing, Scientific visualization and Medical imaging.

Mathematical Details
--------------------

Standard Interpolation
~~~~~~~~~~~~~~~~~~~~~~

A polynomial spline is a continuously defined function made of polynomial pieces of degree :math:`N`. What makes such a spline special is that the pieces connect smoothly when :math:`N\geq1`. In one dimension, let some piece be defined over :math:`x\leq x_{0}` by the polynomial :math:`a(x)=a_{0}+\sum_{n=1}^{N}\,a_{n}\,x^{n}` and let an adjacent piece be :math:`b(x)=b_{0}+\sum_{n=1}^{N}\,b_{n}\,x^{n}`, with :math:`x\geq x_{0}`. Then, the two pieces must join continuously, with :math:`a(x_{0})=b(x_{0})`. The first derivatives are also in agreement, with :math:`\dot{a}(x_{0})=\dot{b}(x_{0})`. Likewise, :math:`\ddot{a}(x_{0})=\ddot{b}(x_{0})`, and so on until :math:`{\mathrm{d}}^{N-1}a(x_{0})/{\mathrm{d}}x^{N-1}={\mathrm{d}}^{N-1}b(x_{0})/{\mathrm{d}}x^{N-1}`. (However, the derivatives of order :math:`N` of :math:`a` and :math:`b` are allowed to disagree at :math:`x_{0}`.)

In one dimension, each polynomial piece has a left neighbor and a right neighbor. With many pieces and a large polynomial degree, the construction of a spline makes for a complicated task that must honor many constraints of continuity. In the interpolation context, one additionally asks that the spline, which we now call :math:`f`, reproduces the list :math:`\{y[k]\}_{k=0}^{K-1}` of :math:`K` samples at the set :math:`\{x[k]\}_{k=0}^{K-1}` of :math:`K` sampling locations, with :math:`f(x[k])=y[k]`. All of this is greatly simplified under some appropriate assumptions, namely, that the sampling locations are regularly spaced exactly one unit apart and coincide with the integers; it is also especially convenient to assume that the polynomial pieces are of unit length, too, and that their extremities coincide either with the integers (odd :math:`N`) or with the half integers (even :math:`N`). Under these simplifying assumptions, algorithms were championed in [`1 <https://ieeexplore.ieee.org/document/193220>`_, `2 <https://ieeexplore.ieee.org/document/193221>`_, `3 <https://ieeexplore.ieee.org/document/799930>`_] to honor all constraints (continuity and interpolation), with the computational effort :math:`{\mathcal{O}}(K\,\left\lfloor N/2\right\rfloor)` to build a spline out of :math:`K` samples. Once the spline is built, it can be interrogated at any continuous argument :math:`x\in{\mathbb{R}}` to yield the value :math:`f(x)`, with the per-point computational effort :math:`{\mathcal{O}}(N^{2})`.

In the SplineOps implementation, the polynomial pieces are never expressed explicitly. Instead, so-called spline coefficients :math:`c` are used to represent the spline. Moreover, the polynomials that make the spline are decomposed in a basis of integer translates of B-splines, which are functions :math:`\beta^{N}` that are uniquely defined as even-symmetric splines of unit integral and least support. There, :math:`N` is a superscript (not an exponent) that gives the degree of the B-spline.

In one dimension, we thus write a generic spline in terms of :math:`x\in{\mathbb{R}}` as :math:`f(x)=\sum_{k\in{\mathbb{Z}}}\,c[k]\,\beta^{N}(x-k)`. In multiple dimensions, we consider tensor-product splines. For instance, a continuously defined grayscale spline image that interpolates the array :math:`y[k_{1},k_{2}]` of width :math:`W` and height :math:`H` writes :math:`f(x_{1},x_{2})=\sum_{q_{2}\in{\mathbb{Z}}}\,\left(\sum_{q_{1}\in{\mathbb{Z}}}\,c[q_{1},q_{2}]\,\beta^{N}(x_{1}-q_{1})\right)\,\beta^{N}(x_{2}-q_{2})` and satisfies that :math:`f(k_{1},k_{2})=y[k_{1},k_{2}]`. Pay attention that :math:`(x_{1},x_{2})\in{\mathbb{R}}^{2}` is a continuously defined two-component vector, while :math:`k_{1}\in[0\ldots W-1]` and :math:`k_{2}\in[0\ldots H-1]` are integers. Similarly, a spline volume would write

.. math::
   f({\mathbf{x}})=\sum_{q_{3}\in{\mathbb{Z}}}\,\left(\sum_{q_{2}\in{\mathbb{Z}}}\,\left(\sum_{q_{1}\in{\mathbb{Z}}}\,c[{\mathbf{q}}]\,\beta^{N}(x_{1}-q_{1})\right)\,\beta^{N}(x_{2}-q_{2})\right)\,\beta^{N}(x_{3}-q_{3}).

It is remarkable that splines can be interrogated at any coordinate, not only at a non-integer one, but also at one that would be far remote from the support of the samples that define, say, an image. This is achieved by imposing a structure to the spline coefficients. Typically, this structure takes some form of periodicity, possibly combined with local reversals of chunks of samples. It allows us to predict :math:`c[{\mathbf{k}}]` for any :math:`{\mathbf{k}}\in{\mathbb{Z}}^{2}` from the sole knowledge of :math:`c[{\mathbf{k}}]` for :math:`k_{1}\in[0\ldots W-1]` and :math:`k_{2}\in[0\ldots H-1]`.

The computational burden increases with the degree of a spline, but so does the quality of the representation. Indeed, consider a one-dimensional ground-truth function that is differentiable sufficiently many times. Now, it is customary to know this function only through its samples and to let a spline of degree :math:`N` interpolate them. The higher the degree, the closer to the local Taylor expansion of the ground truth the spline gets the chance to be because :math:`N+1` terms can potentially match. This argument is mere hand-waiving; its merit is that it relies on no more than calculus to make plausible that the continuously defined spline represents the continuously defined ground truth better when the degree rises. To understand the true mechanism that links degree and quality, one needs to master the theory of approximation, which in turn relies on the Fourier theory, the theory of distributions, the measure theory, the theory of sampling, and the theory of integration, among others, all of them being advanced mathematical topics [`4 <https://ieeexplore.ieee.org/document/718806>`_, `5 <https://ieeexplore.ieee.org/document/843002>`_].

While one-dimensional splines are well-understood, the approximation properties of tensor-product splines are not. Yet, numerous practical experiments have led to the conclusion that a tensor-product cubic spline offers a good tradeoff between computational effort and quality, being significantly more accurate than what linear interpolation offers.

[1] M. Unser, A. Aldroubi, M. Eden, `B-Spline Signal Processing: Part I—Theory <https://ieeexplore.ieee.org/document/193220>`_, IEEE-SPS best paper award, IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 821–833, February 1993.

[2] M. Unser, A. Aldroubi, M. Eden, `B-Spline Signal Processing: Part II—Efficient Design and Applications <https://ieeexplore.ieee.org/document/193221>`_, IEEE Transactions on Signal Processing, vol. 41, no. 2, pp. 834–848, February 1993.

[3] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://ieeexplore.ieee.org/document/799930>`_, IEEE-SPS best paper award, IEEE Signal Processing Magazine, vol. 16, no. 6, pp. 22–38, November 1999.

[4] M. Unser, J. Zerubia, `A Generalized Sampling Theory Without Band-Limiting Constraints <https://ieeexplore.ieee.org/document/718806>`_, IEEE Transactions on Circuits and Systems—II: Analog and Digital Signal Processing, vol. 45, no. 8, pp. 959–969, August 1998.

[5] M. Unser, `Sampling—50 Years After Shannon <https://ieeexplore.ieee.org/document/843002>`_, Proceedings of the IEEE, vol. 88, no. 4, pp. 569–587, April 2000.


..
  B-Splines and Interpolation
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~

  B-splines are piecewise polynomial functions with compact support, often used 
  for interpolation. A B-spline of degree :math:`n`, denoted as :math:`\beta_n(x)`, is defined recursively:

  .. math::

     \beta_0(x) =
     \begin{cases}
     1, & 0 \leq x < 1, \\
     0, & \text{otherwise},
     \end{cases}

  and for :math:`n > 0`,

  .. math::

     \beta_n(x) = \frac{x}{n} \beta_{n-1}(x) + \frac{n+1-x}{n} \beta_{n-1}(x-1).

  Alternatively, the :math:`n`-degree B-spline can be expressed using truncated power functions and binomial coefficients:

  .. math::

     \beta_n(x) = \frac{1}{n!} \sum_{k=0}^{n+1} \binom{n+1}{k} (-1)^k (x-k)_+^n,

  where :math:`(x-k)_+^n` is the truncated power function defined as:

  .. math::

     (x-k)_+^n =
     \begin{cases}
     (x-k)^n, & x \geq k, \\
     0, & \text{otherwise}.
     \end{cases}

  The spline interpolation process approximates a function :math:`f(\mathbf{x})` using a linear combination of shifted B-splines:

  .. math::

     s(\mathbf{x}) = \sum_{\mathbf{k}} c_{\mathbf{k}} \beta_n(\mathbf{x} - \mathbf{k}),

  where :math:`c_{\mathbf{k}}` are the spline coefficients determined from the input data.

Least-Squares Projection
~~~~~~~~~~~~~~~~~~~~~~~~

Least-squares resizing minimizes the squared error between the original data :math:`f(\mathbf{x})` and the resized data :math:`\tilde{f}(\mathbf{T}^{-1}\mathbf{x})`, where :math:`\mathbf{T}` is the transformation matrix defining the scaling:

.. math::

   \min_{\tilde{f}} \int_{\Omega} \|f(\mathbf{x}) - \tilde{f}(\mathbf{T}^{-1} \mathbf{x})\|^2 \, d\mathbf{x}.

Key Insights:

- The projection is computed using finite differences to solve for the spline coefficients in a way that preserves the :math:`L_2`-norm.

- Pre-filtering is necessary to compute coefficients :math:`c_{\mathbf{k}}` accurately, which involves solving a linear system or applying recursive filtering. These filters minimize aliasing and improve stability.

- This approach is slower than direct interpolation but provides superior quality by minimizing artifacts and preserving fine details.

Oblique Projection
~~~~~~~~~~~~~~~~~~

The oblique projection generalizes the least-squares method by relaxing orthogonality constraints. It maps the input data onto a biorthogonal basis formed by the scaling and wavelet functions:

.. math::

   c_{\mathbf{k}} = \int_{\Omega} f(\mathbf{x}) \phi(\mathbf{x} - \mathbf{k}) \, d\mathbf{x},

where :math:`\phi` represents the biorthogonal dual basis. Unlike least-squares, the oblique projection:

- Reduces computational cost by approximating the exact solution.

- Results in slightly lower interpolation quality compared to least-squares.

This method is particularly effective for scenarios requiring fast computation with acceptable trade-offs in accuracy.

Resize Example
--------------

* :ref:`sphx_glr_auto_examples_001_resize.py`

References
----------

- Unser, M. (1999). `Splines: A perfect fit for signal/image processing <https://ieeexplore.ieee.org/document/7075842>`_. IEEE Signal Processing Magazine.
- Muñoz, A., Blu, T., & Unser, M. (2001). `Least-squares image resizing using finite differences <https://ieeexplore.ieee.org/document/941860>`_. IEEE Transactions on Image Processing, 10(9), 1365–1378.
- Thévenaz, P., Blu, T., & Unser, M. (2000). `Interpolation revisited <https://ieeexplore.ieee.org/document/875199>`_. IEEE Transactions on Medical Imaging, 19(7), 739–758.
