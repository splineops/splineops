Smooth
======

.. currentmodule:: splineops

Overview
--------

The `smooth` module in `splineops` implements **fractional smoothing splines**, which are optimal estimators for smooth function approximation and interpolation. Unlike standard polynomial splines, these splines are derived from **fractional differential operators**, making them highly adaptable for **self-similar** and **fractal-like** signals.

Key features:

- Supports **1D and N-dimensional** smoothing splines.
- Provides a **recursive filtering** implementation for fast computation.
- Uses **fractional-order derivatives** for enhanced smoothness control.
- Implements **fast Fourier transform (FFT)-based methods** for large-scale smoothing.
- Provides direct **interpolation** and **denoising** functionality.

These methods are particularly useful in **signal processing**, **image reconstruction**, and **time-series modeling**, especially for **noisy or fractal-like data**.

Mathematical Background
------------------------

Problem Formulation
~~~~~~~~~~~~~~~~~~~

Smoothing splines solve a **regularized variational problem** where the objective is to fit a function :math:`f(x)` to given data points :math:`(x_m, y_m)`, while penalizing roughness:

.. math::

    \arg\min_{f} \sum_{m=1}^{M} E(f(x_m), y_m) + \lambda \| D^\gamma f \|_M,

where:

- :math:`E(f(x_m), y_m)` is a **data-fidelity** term, typically quadratic: :math:`(f(x_m) - y_m)^2/2`.
- :math:`\lambda` is the **regularization parameter**, controlling the smoothness.
- :math:`D^\gamma f` is the **fractional derivative of order** :math:`\gamma = H + 0.5`.
- :math:`\| \cdot \|_M` represents the **total-variation norm**, enforcing smoothness.

This formulation ensures that the **smoothing spline solution is a fractional B-spline**.

Fractional B-Splines
~~~~~~~~~~~~~~~~~~~~

Fractional splines generalize classical polynomial splines by allowing **non-integer derivatives**. The smoothing spline **minimizes an energy functional** of the form:

.. math::

    \| D^\gamma f \|^2_M,

which is equivalent to **applying a Butterworth-like low-pass filter**.

For a discrete signal :math:`y[n]`, the solution is given by:

.. math::

    y_{\text{smooth}} = \mathcal{F}^{-1} \left( H(\omega) \mathcal{F}(y) \right),

where the **smoothing filter** is:

.. math::

    H(\omega) = \frac{1}{1 + \lambda |\omega|^{2\gamma}}.

This filter **attenuates high frequencies**, leading to **optimal smoothing**.

Types of Smoothing Splines
--------------------------

1D Smoothing Splines
~~~~~~~~~~~~~~~~~~~~

The **fractional smoothing spline** in 1D is implemented using **recursive filtering**, following:

.. code-block:: python

    def recursive_smoothing_spline(signal, lam=1.0):
        """
        Applies recursive smoothing spline filtering using a causal and anticausal IIR filter.

        Parameters:
        - signal: 1D array of data points
        - lam: Regularization parameter controlling the smoothness

        Returns:
        - smoothed_signal: 1D array of smoothed data
        """
        z1 = -lam / (1 + np.sqrt(1 + 4 * lam))
        K = len(signal)
        
        # Forward pass (causal filter)
        y_causal = np.zeros(K)
        y_causal[0] = signal[0]
        for k in range(1, K):
            y_causal[k] = signal[k] + z1 * y_causal[k - 1]

        # Backward pass (anticausal filter)
        smoothed_signal = np.zeros(K)
        smoothed_signal[-1] = y_causal[-1]
        for k in range(K - 2, -1, -1):
            smoothed_signal[k] = y_causal[k] + z1 * smoothed_signal[k + 1]

        return smoothed_signal

This provides **fast filtering with linear time complexity**.

Multi-Dimensional Splines
~~~~~~~~~~~~~~~~~~~~~~~~~

The **multi-dimensional smoothing spline** applies the smoothing operation **independently across all dimensions**:

.. code-block:: python

    def smoothing_spline_nd(data, lambda_, gamma):
        """
        Applies multi-dimensional fractional smoothing spline to input data.

        Parameters:
        - data (ndarray): Input image or volume.
        - lambda_ (float): Regularization parameter.
        - gamma (float): Spline order (gamma = H + 0.5).

        Returns:
        - data_smooth (ndarray): Smoothed data.
        """
        data = np.asarray(data)
        freq_grids = np.meshgrid(*[np.fft.fftfreq(n) for n in data.shape], indexing='ij')
        omega_squared = np.sum((2 * np.pi * np.stack(freq_grids)) ** 2, axis=0)

        # Butterworth-like smoothing filter
        H = 1 / (1 + lambda_ * omega_squared ** gamma)

        # Apply in Fourier domain
        data_fft = np.fft.fftn(data)
        data_smooth_fft = H * data_fft
        data_smooth = np.real(np.fft.ifftn(data_smooth_fft))

        return data_smooth

This **leverages FFT-based filtering**, enabling **fast processing of images and 3D volumes**.

Example Usage
-------------

Here’s how to apply **1D and N-dimensional smoothing splines**:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from splineops.interpolate.smoothing_spline import smoothing_spline, smoothing_spline_nd

    # 1D example
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + 0.2 * np.random.randn(len(x))
    y_smooth = smoothing_spline(y, lambda_=0.1, m=1, gamma=1.5)

    plt.plot(x, y, label="Noisy Signal")
    plt.plot(x, y_smooth, label="Smoothed Signal", linewidth=2)
    plt.legend()
    plt.show()

    # 2D example (image)
    from skimage import data

    img = data.camera().astype(np.float64) / 255.0  # Normalize
    img_noisy = img + 0.1 * np.random.randn(*img.shape)
    img_smooth = smoothing_spline_nd(img_noisy, lambda_=0.05, gamma=2.0)

    plt.subplot(1, 2, 1)
    plt.imshow(img_noisy, cmap='gray')
    plt.title("Noisy Image")

    plt.subplot(1, 2, 2)
    plt.imshow(img_smooth, cmap='gray')
    plt.title("Smoothed Image")
    plt.show()

This example demonstrates:

- **Denoising a 1D signal using recursive smoothing splines**.
- **Applying 2D smoothing splines to image data**.

Regularization Parameter
------------------------

The regularization parameter :math:`\lambda` balances **data fidelity vs. smoothness**:

- **Small** :math:`\lambda` → Preserves more details but may leave noise.
- **Large** :math:`\lambda` → Produces a smoother function but may oversmooth.

For **images and high-dimensional data**, a typical choice is :math:`\lambda \approx 0.05 - 0.2`.

Smooth Example
--------------

* :ref:`sphx_glr_auto_examples_008_using_smooth_module.py`

References
----------

- **Unser, M., & Blu, T.** (2007). *Self-Similarity: Part I — Splines and Operators*. IEEE Transactions on Signal Processing, vol. 55, no. 4, pp. 1352-1363.
- **Blu, T. & M. Unser.** (2007). *Self-Similarity: Part II — Optimal Estimation of Fractal Processes*. IEEE Transactions on Signal Processing, vol. 55, no. 4, pp. 1364-1378.
- **Unser, M., & Blu, T.** (2000). *Fractional Splines and Wavelets*. SIAM Review, vol. 42, no. 1, pp. 43-67.

.. note::
    Smoothing splines are widely used for **fractal process estimation**, **medical imaging**, and **machine learning**.