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
      \int_{\mathbb{R}} \bigl|f(x) - g(x)\bigr|^2 \,\mathrm{d}x.

This is the **least-squares projection** point of view: among all splines that
live in :math:`V_T` (on the grid :math:`\Gamma_T`), we pick the one that is as
close as possible to :math:`f` in the :math:`L_2` sense.

In this language:

- :math:`V_1` is the input spline space (grid step 1),
- :math:`V_T` is the output spline space (grid step :math:`T`),
- and **resizing** is the operation :math:`V_1 \to V_T` that maps the
  coefficients (or samples) of :math:`f` to the coefficients :math:`c_T[k]` of
  :math:`g_T`.

Resize Examples
---------------

* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_01_resize_module.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_02_standard_interpolation.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_03_antialiasing.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_04_how_bad_aliasing_can_be.py`
* :ref:`sphx_glr_auto_examples_03_resampling_using_2d_interpolation_03_05_benchmarking.py`

References
----------

.. [1] C. Lee, M. Eden, M. Unser,
   `High-Quality Image Resizing Using Oblique Projection Operators <https://doi.org/10.1109/83.668025>`_,
   IEEE Transactions on Image Processing, vol. 7, no. 5,
   pp. 679–692, May 1998.

.. [2] A. Muñoz Barrutia, T. Blu, M. Unser, 
   `Least-Squares Image Resizing Using Finite Differences <https://doi.org/10.1109/83.941860>`_,
   IEEE Transactions on Image Processing, vol. 10, no. 9, pp. 1365-1378,
   September 2001.

.. [3] M. Unser, `Splines: A Perfect Fit for Signal and Image Processing <https://doi.org/10.1109/79.799930>`_, 
   IEEE-SPS best paper award, IEEE Signal Processing Magazine, 
   vol. 16, no. 6, pp. 22-38, November 1999.

.. [4] P. Thévenaz, T. Blu, M. Unser,
   `Interpolation Revisited <https://doi.org/10.1109/42.875199>`_,
   IEEE Transactions on Medical Imaging, vol. 19, no. 7, pp. 739-758,
   July 2000.
