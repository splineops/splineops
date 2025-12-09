.. splineops/docs/user-guide/02_resize.rst

Resize
======

.. currentmodule:: splineops

Overview
--------

The :ref:`resize <api-resize>` module in :ref:`Splineops <api-index>` provides high-performance,
high-fidelity resizing for N-dimensional arrays. 

Conceptually, resizing means:

- starting from a spline :math:`f` defined on an input grid
  (typically the integers),
- choosing a new output grid, obtained by scaling the grid by a factor
  :math:`T` (e.g., :math:`0, T, 2T, 3T, \dots`),
- and constructing a new spline :math:`g` that “lives” on that new grid and
  best represents the same underlying continuous function.

We illustrate this with the 1D example
:ref:`sphx_glr_auto_examples_02_resampling_using_1d_samples_02_02_resample_a_1d_spline.py`,
which starts from a spline :math:`f` and samples it more coarsely at positions
:math:`x = T k`:

.. image:: /auto_examples/02_resampling_using_1d_samples/images/sphx_glr_02_02_resample_a_1d_spline_001.png
   :align: center
   :width: 100%

The red stems and markers correspond to the new samples :math:`f(Tk)` on the
coarser grid.

Resized Grids and Basis Functions
---------------------------------

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
:ref:`sphx_glr_auto_examples_02_resampling_using_1d_samples_02_02_resample_a_1d_spline.py`
has two rows. The top row shows the fine-grid samples :math:`f[k]` together
with their shifted basis functions :math:`\varphi(x - k)` scaled by the
coefficients :math:`c[k]`, illustrating the spline space :math:`V_1`. The
bottom row shows the coarse samples :math:`g[k]` together with their shifted
basis functions :math:`\varphi(x/T - k)` scaled by :math:`c_T[k]`, illustrating
the spline space :math:`V_T`. In that example, :math:`\varphi = \beta^{3}` is
the cubic B-spline, and the coefficients :math:`c[k]` and :math:`c_T[k]`
implement the standard interpolation scheme described in
:doc:`01_spline_interpolation`:

.. image:: /auto_examples/02_resampling_using_1d_samples/images/sphx_glr_02_02_resample_a_1d_spline_002.png
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

Least-Squares Projection
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

Oblique Projection
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

The Algorithm
-------------

At implementation level, :ref:`resize <api-resize>` follows the
projection framework described above, but organized as a simple sequence of
1D operations applied axis by axis.

For a single axis, the algorithm works on one 1D line at a time:

1. **Spline prefilter.**  
   The input samples along the line are first converted into spline
   coefficients using a stable recursive filter. After this step, the line
   represents a continuous spline in the sense of the previous sections,
   rather than just raw samples.

2. **Optional projection prefilter.**  
   In antialiasing modes, a small number of discrete integrations and
   differences are applied to these coefficients. This realizes the
   least-squares / oblique projection prefilter from [1]_ and [2]_, and acts
   as a controlled low-pass filter when down-sampling. For pure
   interpolation this stage is skipped.

3. **Boundary handling.**  
   Because real data are finite, each line is extended beyond its endpoints
   by symmetric or antisymmetric mirroring, depending on the spline degree.
   This produces a slightly longer “virtual” line on which the spline is
   evaluated, without introducing visible edge artefacts.

4. **Resampling on the new grid.**  
   For the chosen zoom, the algorithm precomputes, once per axis, how every
   output position maps back to the original grid: which input coefficients
   contribute, and with which spline weights. During execution, each output
   sample is then obtained as a short weighted sum over that local window.
   This precomputation is what makes the method both accurate and efficient.

5. **Projection tail (if enabled).**  
   When using antialiasing, the intermediate result is brought back from the
   analysis space to the desired spline model by applying the corresponding
   discrete differences and a final spline reconstruction on the output grid.

For N-dimensional data, this 1D scheme is applied separately along each axis
in turn (a separable algorithm). All other axes are treated as batch
dimensions, so the same 1D logic is reused for many lines, with a single
precomputed plan per axis and zoom configuration.

Implementation
--------------

Internally, :ref:`resize <api-resize>` uses two cooperating backends:

* A compiled C++ core, wrapped as a small extension module. For each axis, it
  builds a reusable 1D plan that encodes the mapping from output samples back
  to the input grid (support window, spline weights, boundary handling). The
  actual evaluation is done in double precision in a tight inner loop and is
  parallelized over independent 1D lines when the workload is large enough.

* A pure-NumPy fallback that mirrors the same 1D scheme. It reshapes the data
  so that each line along the resized axis is contiguous, processes lines in
  batches, and uses vectorized gathers and reductions to apply the same
  precomputed weights. Plans are cached and reused when the same configuration
  is requested again.

Both backends perform all spline computations in 64-bit floating point; input
and output arrays keep their original dtype (or a user-specified dtype), with
casting only at the boundary of each axis pass.

Conceptually, :ref:`resize <api-resize>` is configured by three spline degrees:

* the **interpolation degree**, which sets the underlying spline model,
* the **analysis degree**, which controls the projection-based prefilter
  (for antialiasing; ``-1`` means “no projection”),
* and the **synthesis degree**, which sets the spline model on the resized
  grid.

These degrees are exposed through the :ref:`resize <api-resize>` API via the
``method`` argument. There are two main families of presets.

**Standard interpolation presets** use no projection at all (analysis degree
``-1``) and perform plain spline interpolation:

.. list-table:: Spline degree standard interpolation presets in :ref:`resize <api-resize>`
   :header-rows: 1

   * - Method
     - Interpolation degree
     - Analysis degree
     - Synthesis degree
   * - ``"fast"``
     - 0
     - -1
     - 0
   * - ``"linear"``
     - 1
     - -1
     - 1
   * - ``"quadratic"``
     - 2
     - -1
     - 2
   * - ``"cubic"``
     - 3
     - -1
     - 3

These are appropriate when you mainly want smooth interpolation and are not
aggressively downsampling.

**Antialiasing presets** use an oblique projection with a lower analysis
degree and a higher synthesis degree, and are designed for downsampling
(and its inverse round-trip):

.. list-table:: Spline degree antialiasing configuration in :ref:`resize <api-resize>`
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

Here, the synthesis degree matches the interpolation degree, defining the
output spline model, while the lower analysis degree keeps the projection
prefilter short, robust, and efficient, yet still very close to the ideal
least-squares solution in [1]_ and [2]_.

.. warning::
   **Exact least-squares configurations**, where the analysis and synthesis
   degrees are equal (for example, a cubic–cubic combination), require
   high-order discrete integration in this framework. For cubic splines,
   the theory calls for fourth-order integration: a running-sum operator
   applied four times in a row to implement the continuous prefilter. Each
   pass is stable in exact arithmetic, but in double precision they amplify
   tiny rounding errors, especially on long lines, which can lead to slow
   drift in the mean level and other visible artefacts. For this reason such
   configurations **are not exposed as presets and are not recommended**
   in routine use. **The oblique antialiasing presets above avoid the
   problematic high-order integration while remaining very close in quality**
   to the ideal least-squares projection.

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