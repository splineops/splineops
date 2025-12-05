.. splineops/docs/user-guide/02_resize.rst

Resize
======

.. currentmodule:: splineops

Overview
--------

The :func:`resize` family in :mod:`splineops` provides high-performance,
high-fidelity resizing for N-dimensional arrays built on the spline model
introduced in :doc:`01_spline_interpolation`.

Conceptually, resizing means:

- starting from a spline :math:`f` defined on an **input grid** 
  (typically the integers),
- choosing a new **output grid**, obtained by scaling the grid by a factor
  :math:`T` (e.g., :math:`0, T, 2T, 3T, \dots`),
- and constructing a new spline :math:`g` that “lives” on that new grid and
  best represents the same underlying continuous function.

We illustrate this with the 1D example
:ref:`sphx_glr_auto_examples_02_resampling_using_1d_interpolation_02_02_resample_a_1d_spline.py`,
which starts from a spline :math:`f` and samples it more coarsely at positions
:math:`x = T k`:

.. image:: /auto_examples/02_resampling_using_1d_interpolation/images/sphx_glr_02_02_resample_a_1d_spline_001.png
   :align: center
   :width: 100%

The red stems and markers correspond to the new samples :math:`f(Tk)` on the
coarser grid.

A 1D spline-space view
----------------------

In the interpolation chapter we introduced a 1D spline model of the form

.. math::

    f(x) = \sum_{k \in \mathbb{Z}} c[k]\,\varphi(x - k),

where :math:`\varphi` is a fixed basis function (typically a B-spline of some
degree :math:`n`, e.g. :math:`\varphi = \beta^{n}`) and :math:`c[k]` are the
spline coefficients.

We will work with coefficient sequences that are square-summable:

.. math::

    \ell_2(\mathbb{Z})
    = \Bigl\{ (c[k])_{k \in \mathbb{Z}} \;\Big|\;
               \sum_{k \in \mathbb{Z}} |c[k]|^2 < \infty \Bigr\}.

The set of all such splines then forms a spline space, which we denote by

.. math::

    V_1
    = \Bigl\{
        f : \mathbb{R} \to \mathbb{R}
        \;\Big|\;
        f(x) = \sum_{k \in \mathbb{Z}} c[k]\,\varphi(x - k),
        \ (c[k])_{k \in \mathbb{Z}} \in \ell_2(\mathbb{Z})
      \Bigr\}.

We call this space :math:`V_1` because it corresponds to a **unit** sampling
step along the integer grid :math:`\{0, 1, 2, \dots\}`.

Now fix a scale factor :math:`T > 0` and consider the **scaled grid**

.. math::

    \Gamma_T = \{ T k \mid k \in \mathbb{Z} \}.
    
We can build a similar spline space adapted to this new grid by defining
basis functions

.. math::

    \varphi_{k,T}(x) = \varphi\!\left(\frac{x}{T} - k\right),

and setting

.. math::

    V_T
    = \Bigl\{
        g_T : \mathbb{R} \to \mathbb{R}
        \;\Big|\;
        g_T(x) = \sum_{k \in \mathbb{Z}} c_T[k]\,\varphi_{k,T}(x),
        \ (c_T[k])_{k \in \mathbb{Z}} \in \ell_2(\mathbb{Z})
      \Bigr\}.

In other words, :math:`V_T` is the spline space associated with the grid
:math:`\Gamma_T`. A generic element :math:`g \in V_T` can be written as

.. math::

    g_T(x) = \sum_{k \in \mathbb{Z}} c_T[k] \,\varphi\!\left(\frac{x}{T} - k\right),

for some coefficient sequence :math:`(c_T[k])_{k \in \mathbb{Z}}` in
:math:`\ell_2(\mathbb{Z})`.

As a visual example, the following figure from
:ref:`sphx_glr_auto_examples_02_resampling_using_1d_interpolation_02_02_resample_a_1d_spline.py`
has two rows. The top row shows the fine-grid samples :math:`f[k]` together
with their shifted basis functions :math:`\varphi(x - k)` scaled by the
coefficients :math:`c[k]`, illustrating the spline space :math:`V_1`. The
bottom row shows the coarse samples :math:`g[k]` together with their shifted
basis functions :math:`\varphi(x/T - k)` scaled by :math:`c_T[k]`, illustrating
the spline space :math:`V_T`. In that example, :math:`\varphi = \beta^{3}` is
the cubic B-spline, and the coefficients :math:`c[k]` and :math:`c_T[k]`
implement the standard interpolation scheme described in
:doc:`01_spline_interpolation`:

.. image:: /auto_examples/02_resampling_using_1d_interpolation/images/sphx_glr_02_02_resample_a_1d_spline_002.png
   :align: center
   :width: 100%

From samples to a new spline
----------------------------

Suppose that the original signal :math:`f` belongs to :math:`V_1`. For a given
scale factor :math:`T`, we can form **new samples** on the scaled grid
:math:`\Gamma_T`:

.. math::

    f_T[k] = f(Tk), \qquad k \in \mathbb{Z}.

These samples tell us how the original continuous spline :math:`f` behaves at
the new grid locations. The goal of resizing is to construct a new spline
:math:`g_T \in V_T` that is consistent with these samples and remains a good
approximation of :math:`f` in the continuous domain.

A natural way to define :math:`g_T` is as an orthogonal projection of :math:`f`
onto :math:`V_T` in :math:`L_2(\mathbb{R})`:

.. math::

    g_T
    = \underset{g \in V_T}{\arg\min}
      \,\|f - g\|_{L_2(\mathbb{R})}^2,

where the :math:`L_2(\mathbb{R})` norm is given by

.. math::

    \|h\|_{L_2(\mathbb{R})}^2
    = \int_{\mathbb{R}} \bigl|h(x)\bigr|^2 \,\mathrm{d}x.

This is the **least-squares projection** point of view: among all splines that
live in :math:`V_T` (on the grid :math:`\Gamma_T`), we pick the one that is as
close as possible to :math:`f` in the :math:`L_2` sense.

In this language:

- :math:`V_1` is the input spline space (grid step 1),
- :math:`V_T` is the output spline space (grid step :math:`T`),
- and **resizing** is the operation :math:`V_1 \to V_T` that maps the
  coefficients (or samples) of :math:`f` to the coefficients :math:`c_T[k]` of
  :math:`g_T`.

Least-squares projection and dual basis
---------------------------------------

To characterize the least-squares solution more explicitly, we introduce a
family of **dual functions** :math:`\{\tilde{\varphi}_{k,T}\}_{k\in\mathbb{Z}}`
in :math:`V_T` such that they are biorthonormal to the basis
:math:`\{\varphi_{k,T}\}_{k\in\mathbb{Z}}`:

.. math::

    \bigl\langle \varphi_{k,T}, \tilde{\varphi}_{m,T} \bigr\rangle_{L_2(\mathbb{R})}
    = \delta_{km},
    \qquad k,m \in \mathbb{Z},

where :math:`\delta_{km}` is the Kronecker delta. Under mild conditions on
:math:`\varphi`, this dual family exists and is unique, and the orthogonal
projection :math:`g_T` of :math:`f` onto :math:`V_T` admits the expansion

.. math::

    g_T(x)
    = \sum_{k \in \mathbb{Z}}
      \bigl\langle f, \tilde{\varphi}_{k,T} \bigr\rangle_{L_2(\mathbb{R})}
      \,\varphi_{k,T}(x).

In other words, the least-squares coefficients :math:`c_T[k]` are obtained
by taking inner products of :math:`f` with the dual functions:

.. math::

    c_T[k]
    = \bigl\langle f, \tilde{\varphi}_{k,T} \bigr\rangle_{L_2(\mathbb{R})},
    \qquad k \in \mathbb{Z},

and the resized spline :math:`g_T` is reconstructed by combining these
coefficients with the shifted basis functions :math:`\varphi_{k,T}(x)`.
This is precisely the **least-squares projection** of :math:`f` onto
the spline space :math:`V_T` [1]_.

Oblique projection
------------------

While the least-squares scheme uses the dual functions
:math:`\tilde{\varphi}_{k,T}` that are uniquely determined by the
synthesis basis :math:`\varphi_{k,T}`, their continuous-domain prefilters
can become complicated and expensive to implement for higher spline
degrees (e.g., cubic and above). To alleviate this, one can replace the
orthogonal projection by an **oblique projection** [2]_.

The idea is to introduce a simpler **analysis family**
:math:`\{\psi_{k,T}\}_{k\in\mathbb{Z}}` in a *different* spline space,
typically of lower degree, and to define the approximation as

.. math::

    g_T^{\mathrm{obl}}(x)
    = \sum_{k \in \mathbb{Z}} d[k]\,\varphi_{k,T}(x),

where the coefficients :math:`d[k]` are obtained from inner products
with the analysis functions :math:`\psi_{k,T}` followed by a discrete
correction filter. In contrast to the least-squares case, the projection
error is orthogonal to the analysis space spanned by :math:`\psi_{k,T}`,
not to :math:`V_T` itself; this is why the operator is called "oblique."

A key point is that the **approximation space** :math:`V_T` (the space
spanned by :math:`\varphi_{k,T}`) is kept the same as for the
least-squares projection. Under this condition, Lee et al. show that:

* the error of the oblique projection remains very close to that of the
  least-squares (orthogonal) projection, with a provable worst-case
  bound depending on the angle between the analysis and synthesis
  spaces [2]_, their Table I and inequality (10);

* both methods have the **same asymptotic approximation order** as the
  sampling step tends to zero, provided the analysis functions satisfy a
  partition-of-unity condition [2]_, equation (11);

* in practice, oblique projection allows the use of **higher order
  spline models** (e.g., cubic and above) with only a modest increase in
  computation, while delivering almost the same quality as the optimal
  least-squares solution.

In SplineOps, the "antialiasing" presets of :func:`resize` follow this
philosophy: they use a higher-degree spline space as synthesis model,
but a lower-degree spline space for analysis (continuous prefiltering).
This yields an efficient projection-based resize operator with strong
antialiasing properties and quality close to the full least-squares
approach, especially for downsampling by noninteger factors.

Resize Examples
---------------

* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_01_resize_module.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_02_standard_interpolation.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_03_antialiasing.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_04_how_bad_aliasing_can_be.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_05_benchmarking.py`

References
----------

.. [1] A. Muñoz Barrutia, T. Blu, M. Unser, 
   `Least-Squares Image Resizing Using Finite Differences <https://doi.org/10.1109/83.941860>`_,
   IEEE Transactions on Image Processing, vol. 10, no. 9, pp. 1365-1378,
   September 2001.

.. [2] C. Lee, M. Eden, M. Unser,
   `High-Quality Image Resizing Using Oblique Projection Operators <https://doi.org/10.1109/83.668025>`_,
   IEEE Transactions on Image Processing, vol. 7, no. 5,
   pp. 679–692, May 1998.

.. [3] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22-38, November 1999.

.. [4] P. Thévenaz, T. Blu, M. Unser,
   `Interpolation Revisited <https://doi.org/10.1109/42.875199>`_,
   IEEE Transactions on Medical Imaging, vol. 19, no. 7, pp. 739-758,
   July 2000.
