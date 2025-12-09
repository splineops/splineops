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

- starting from a spline :math:`f` defined on an input grid
  (typically the integers),
- choosing a new output grid, obtained by scaling the grid by a factor
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

Scaled grids and basis functions
--------------------------------

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

We call this space :math:`V_1` because it corresponds to a unit sampling
step along the integer grid :math:`\{0, 1, 2, \dots\}`.

Now fix a scale factor :math:`T > 0` and consider the scaled grid

.. math::

    \Gamma_T = \{ T k \mid k \in \mathbb{Z} \}.
    
We can picture the two grids schematically as

.. math::

    \begin{aligned}
      \Gamma_1 \text{ (input grid)} &: \quad
      \begin{array}{cccccc}
        \cdots & \bullet & \bullet & \bullet & \bullet & \cdots \\
               & 0       & 1       & 2       & 3       &
      \end{array} \\[0.75em]
      \Gamma_T \text{ (scaled grid)} &: \quad
      \begin{array}{cccccccccc}
        \cdots & \bullet &        & \bullet &        & \bullet &        & \bullet &        & \cdots \\
               & 0       &        & T       &        & 2T      &        & 3T      &        &
      \end{array}
    \end{aligned}

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

Resizing
--------

Suppose that the original signal :math:`f` belongs to :math:`V_1`. For a given
scale factor :math:`T`, we can form new samples on the scaled grid
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

This is the least-squares projection point of view: among all splines that
live in :math:`V_T` (on the grid :math:`\Gamma_T`), we pick the one that is as
close as possible to :math:`f` in the :math:`L_2` sense.

In this language:

- :math:`V_1` is the input spline space (grid step 1),
- :math:`V_T` is the output spline space (grid step :math:`T`),
- and resizing is the operation :math:`V_1 \to V_T` that maps the
  coefficients (or samples) of :math:`f` to the coefficients :math:`c_T[k]` of
  :math:`g_T`.

Least-squares projection
------------------------

So far we have described elements of :math:`V_T` by expanding them in terms of
the shifted basis functions :math:`\varphi_{k,T}`:

.. math::

    g_T(x) = \sum_{k \in \mathbb{Z}} c_T[k]\,\varphi_{k,T}(x).

These functions :math:`\varphi_{k,T}` play a synthesis role: they tell us
how to reconstruct :math:`g_T` once the coefficients :math:`c_T[k]` are known.
What remains is to explain how these coefficients are obtained from the input
signal :math:`f`.

In the least-squares setting, this is done using a second family of functions
:math:`\{\tilde{\varphi}_{k,T}\}_{k\in\mathbb{Z}}`, often called the
analysis functions. They are chosen to be dual to the synthesis functions,
in the sense of the biorthonormality relation

.. math::

    \bigl\langle \varphi_{k,T}, \tilde{\varphi}_{m,T} \bigr\rangle_{L_2(\mathbb{R})}
    = \delta_{km},
    \qquad k,m \in \mathbb{Z},

where :math:`\delta_{km}` is the Kronecker delta. Under mild conditions on
:math:`\varphi`, this dual family exists and is unique. The least-squares
projection :math:`g_T` of :math:`f` onto :math:`V_T` can then be written as

.. math::

    g_T(x)
    = \sum_{k \in \mathbb{Z}}
      \bigl\langle f, \tilde{\varphi}_{k,T} \bigr\rangle_{L_2(\mathbb{R})}
      \,\varphi_{k,T}(x).

In other words, the least-squares coefficients :math:`c_T[k]` are obtained
by first analyzing :math:`f` with the functions :math:`\tilde{\varphi}_{k,T}`
and then synthesizing with :math:`\varphi_{k,T}`:

.. math::

    c_T[k]
    = \bigl\langle f, \tilde{\varphi}_{k,T} \bigr\rangle_{L_2(\mathbb{R})},
    \qquad k \in \mathbb{Z}.

The resized spline :math:`g_T` is thus the least-squares (orthogonal)
projection of :math:`f` onto the spline space :math:`V_T` [1]_.

Oblique projection
------------------

For higher spline orders, the continuous-domain prefilters associated with the
dual functions :math:`\tilde{\varphi}_{k,T}` can become expensive to implement.
A practical alternative is to replace the orthogonal (least-squares) projection
by an oblique projection [2]_.

The idea is to keep the synthesis space :math:`V_T` unchanged, i.e. the
approximation is still written as

.. math::

    g_T^{\mathrm{obl}}(x)
    = \sum_{k \in \mathbb{Z}} d[k]\,\varphi_{k,T}(x),

but to compute the coefficients :math:`d[k]` using a simpler analysis family
:math:`\{\psi_{k,T}\}_{k\in\mathbb{Z}}` that typically belongs to a lower-degree
spline space. In this case, the projection error is orthogonal to the analysis
space spanned by :math:`\psi_{k,T}`, rather than to :math:`V_T` itself, hence
the term “oblique” projection.

When the analysis and synthesis spaces satisfy mild compatibility conditions,
oblique projection retains the same approximation order as the least-squares
projection and yields very similar quality in practice, while significantly
reducing computational cost. 

Implementation
--------------

In SplineOps, the antialiasing presets of :func:`resize` follow this pattern: they
use a higher-degree spline model for the resized data, combined with a lower-degree
analysis spline to implement an efficient projection-based (low-pass) prefilter.

Concretely, the presets correspond to the following degree triples
(interpolation, analysis, synthesis):

.. list-table:: Spline degree configuration for oblique projection in ``resize``
   :header-rows: 1

   * - Method
     - Interpolation degree
     - Analysis degree
     - Synthesis degree
   * - ``"linear-antialiasing"``
     - 1
     - 0
     - 1
   * - ``"quadratic-antialiasing"``
     - 2
     - 1
     - 2
   * - ``"cubic-antialiasing"``
     - 3
     - 1
     - 3

Here, the synthesis degree matches the interpolation degree, defining the spline
model used for the resized data, while the analysis degree is chosen lower to
simplify the continuous prefilter and make the oblique projection more efficient,
with only a small loss compared to the full least-squares projection.

Resize Examples
---------------

* :ref:`sphx_glr_auto_examples_03_resampling_using_nd_samples_03_01_resize_module_1d.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_nd_samples_03_02_resize_module_2d.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_nd_samples_03_03_standard_interpolation.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_nd_samples_03_04_antialiasing.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_nd_samples_03_05_how_bad_aliasing_can_be.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_nd_samples_03_06_benchmarking.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_nd_samples_03_07_benchmarking_plot.py`

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