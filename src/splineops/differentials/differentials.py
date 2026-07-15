# splineops/src/splineops/differentials/differentials.py

import numpy as np

from splineops.spline_interpolation.utils import _data_to_coeffs


class Differentials:
    """
    Class for computing image differentials using cubic B-spline interpolation.

    This class provides methods to compute first- and second-order derivatives
    of a grayscale image by reconstructing the image as a continuous function
    using cubic B-spline interpolation. Supported operations include gradient
    magnitude, gradient direction, Laplacian, largest and smallest Hessian
    eigenvalues, and Hessian orientation.

    Attributes
    ----------
    GRADIENT_DIRECTION : int
        The gradient direction operation.
    GRADIENT_MAGNITUDE : int
        The gradient magnitude operation.
    HESSIAN_ORIENTATION : int
        The Hessian orientation operation.
    LAPLACIAN : int
        The Laplacian operation.
    LARGEST_HESSIAN : int
        The largest Hessian eigenvalue operation.
    SMALLEST_HESSIAN : int
        The smallest Hessian eigenvalue operation.
    FLT_EPSILON : float
        Constant for single precision floats.
    """

    GRADIENT_DIRECTION = 1
    GRADIENT_MAGNITUDE = 0
    HESSIAN_ORIENTATION = 5
    LAPLACIAN = 2
    LARGEST_HESSIAN = 3
    SMALLEST_HESSIAN = 4

    FLT_EPSILON = np.finfo(np.float32).eps

    def __init__(self, image, spacing=(1.0, 1.0)):
        """
        Initialize a new differentials instance.

        Parameters
        ----------
        image : ndarray
            Input grayscale image as a 2D numpy array.
        spacing : tuple of float, optional
            Physical sample spacing as ``(row_spacing, column_spacing)``.
            Both values must be finite and positive.  The default is unit
            pixel spacing.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("'image' must be a NumPy array.")
        if image.ndim != 2:
            raise ValueError("'image' must be a two-dimensional grayscale array.")
        if any(length == 0 for length in image.shape):
            raise ValueError("'image' dimensions must be non-empty.")
        if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
            raise TypeError("'image' must have a real numeric dtype.")
        if not np.all(np.isfinite(image)):
            raise ValueError("'image' must contain only finite values.")
        try:
            spacing = tuple(spacing)
        except TypeError as exc:
            raise TypeError("'spacing' must contain two positive real values.") from exc
        if len(spacing) != 2:
            raise ValueError("'spacing' must contain row and column spacing.")
        if any(
            isinstance(value, (bool, np.bool_))
            or not np.isscalar(value)
            or not np.isreal(value)
            for value in spacing
        ):
            raise TypeError("'spacing' must contain two positive real values.")
        spacing = tuple(float(value) for value in spacing)
        if not all(np.isfinite(value) and value > 0 for value in spacing):
            raise ValueError("'spacing' values must be finite and positive.")
        work_dtype = (
            image.dtype
            if np.issubdtype(image.dtype, np.floating)
            else np.dtype(np.float64)
        )
        self.image = image.astype(work_dtype, copy=True)
        self.spacing = spacing
        self.height, self.width = image.shape
        self.operation = self.LAPLACIAN

    def run(self, operation=None, *, normalize=False):
        """
        Execute the selected differential operation on the image.

        Parameters
        ----------
        operation : int, optional
            Operation to perform. If None, the default operation (Laplacian) is used.
        normalize : bool, optional
            If true, scale non-angular results to the interval [0, 1].  The
            default is false so the numerical API returns raw derivatives.

        Returns
        -------
        ndarray
            A newly computed differential image.  The source stored in
            ``image`` is not replaced, so repeated calls are independent.
        """
        if operation is not None:
            self.operation = operation

        if self.operation == self.GRADIENT_MAGNITUDE:
            result = self.gradient_magnitude()
        elif self.operation == self.GRADIENT_DIRECTION:
            result = self.gradient_direction()
        elif self.operation == self.LAPLACIAN:
            result = self.laplacian()
        elif self.operation == self.LARGEST_HESSIAN:
            result = self.largest_hessian()
        elif self.operation == self.SMALLEST_HESSIAN:
            result = self.smallest_hessian()
        elif self.operation == self.HESSIAN_ORIENTATION:
            result = self.hessian_orientation()
        else:
            raise ValueError(f"Unknown differential operation {self.operation!r}.")

        if normalize:
            if self.operation in [self.GRADIENT_DIRECTION, self.HESSIAN_ORIENTATION]:
                raise ValueError(
                    "Angular differential results cannot be normalized as intensities."
                )
            result = result - result.min()
            maximum = result.max()
            if maximum != 0:
                result = result / maximum

        return result

    def _coefficients_along_axis(self, image, axis):
        """Return cubic B-spline coefficients along one image axis."""
        # ``_data_to_coeffs`` works in place.  ``ascontiguousarray`` alone may
        # return ``image`` itself when the selected axis is already last,
        # which would silently mutate the source and make repeated calls
        # order-dependent.
        coefficients = np.array(np.moveaxis(image, axis, -1), copy=True, order="C")
        if coefficients.shape[-1] > 1:
            pole = np.array([np.sqrt(3.0) - 2.0])
            _data_to_coeffs(
                coefficients,
                poles=pole,
                boundary="mirror",
                tol=self.FLT_EPSILON,
            )
        return np.moveaxis(coefficients, -1, axis)

    def _gradient_along_axis(self, image, axis):
        coefficients = self._coefficients_along_axis(image, axis)
        result = np.zeros_like(coefficients)
        middle = [slice(None)] * 2
        previous = [slice(None)] * 2
        following = [slice(None)] * 2
        middle[axis] = slice(1, -1)
        previous[axis] = slice(None, -2)
        following[axis] = slice(2, None)
        result[tuple(middle)] = (
            0.5
            * (coefficients[tuple(following)] - coefficients[tuple(previous)])
            / self.spacing[axis]
        )
        return result

    def horizontal_gradient(self):
        """Return the derivative along increasing column coordinates."""
        return self._gradient_along_axis(self.image, axis=1)

    def vertical_gradient(self):
        """Return the derivative along increasing row coordinates."""
        return self._gradient_along_axis(self.image, axis=0)

    def gradient_components(self):
        """Return ``(vertical, horizontal)`` first-derivative components."""
        return self.vertical_gradient(), self.horizontal_gradient()

    def horizontal_hessian(self):
        """Return the second derivative along column coordinates."""
        return self._hessian_along_axis(self.image, axis=1)

    def vertical_hessian(self):
        """Return the second derivative along row coordinates."""
        return self._hessian_along_axis(self.image, axis=0)

    def cross_hessian(self):
        """Return the mixed row-column second derivative."""
        return self._gradient_along_axis(self.horizontal_gradient(), axis=0)

    def hessian_components(self):
        """Return ``(vertical, cross, horizontal)`` Hessian components."""
        return self.vertical_hessian(), self.cross_hessian(), self.horizontal_hessian()

    def _hessian_along_axis(self, image, axis):
        coefficients = self._coefficients_along_axis(image, axis)
        if coefficients.shape[axis] < 2:
            return np.zeros_like(coefficients)
        result = -2.0 * coefficients
        middle = [slice(None)] * 2
        previous = [slice(None)] * 2
        following = [slice(None)] * 2
        middle[axis] = slice(1, -1)
        previous[axis] = slice(None, -2)
        following[axis] = slice(2, None)
        result[tuple(middle)] += (
            coefficients[tuple(previous)] + coefficients[tuple(following)]
        )
        first = [slice(None)] * 2
        second = [slice(None)] * 2
        last = [slice(None)] * 2
        penultimate = [slice(None)] * 2
        first[axis] = 0
        second[axis] = 1
        last[axis] = -1
        penultimate[axis] = -2
        result[tuple(first)] += 2.0 * coefficients[tuple(second)]
        result[tuple(last)] += 2.0 * coefficients[tuple(penultimate)]
        return result / self.spacing[axis] ** 2

    def get_cross_hessian(self, image, tolerance):
        """
        Compute the cross (mixed) Hessian term of the image.

        Parameters
        ----------
        image : ndarray
            Input image array.
        tolerance : float
            Tolerance parameter for spline coefficient computation.

        Returns
        -------
        ndarray
            Element-wise cross hessian.
        """
        # 1) partial f / partial x
        intermediate = self.get_horizontal_gradient(image, tolerance)
        # 2) partial/partial y of that
        f_xy = self.get_vertical_gradient(intermediate, tolerance)
        return f_xy

    def get_horizontal_gradient(self, image, tolerance):
        """
        Compute the horizontal gradient of the image.

        Parameters
        ----------
        image : ndarray
            Input image array.
        tolerance : float
            Tolerance parameter for spline coefficient computation.

        Returns
        -------
        ndarray
            Horizontal gradient of the image.
        """
        return self._gradient_along_axis(image, axis=1)

    def get_horizontal_hessian(self, image, tolerance):
        """
        Compute the horizontal second derivative (Hessian) of the image.

        Parameters
        ----------
        image : ndarray
            Input image array.
        tolerance : float
            Tolerance parameter for spline coefficient computation.

        Returns
        -------
        ndarray
            Horizontal Hessian of the image.
        """
        return self._hessian_along_axis(image, axis=1)

    def get_vertical_gradient(self, image, tolerance):
        """
        Compute the vertical gradient of the image.

        Parameters
        ----------
        image : ndarray
            Input image array.
        tolerance : float
            Tolerance parameter for spline coefficient computation.

        Returns
        -------
        ndarray
            Vertical gradient of the image.
        """
        return self._gradient_along_axis(image, axis=0)

    def get_vertical_hessian(self, image, tolerance):
        """
        Compute the vertical second derivative (Hessian) of the image.

        Parameters
        ----------
        image : ndarray
            Input image array.
        tolerance : float
            Tolerance parameter for spline coefficient computation.

        Returns
        -------
        ndarray
            Vertical Hessian of the image.
        """
        return self._hessian_along_axis(image, axis=0)

    def anti_symmetric_fir_mirror_on_bounds(self, h, c):
        """
        Apply an anti-symmetric FIR filter with mirror boundary extension.

        Parameters
        ----------
        h : ndarray
            Filter coefficients (expected length 2, with h[0] == 0.0).
        c : ndarray
            Signal (or coefficient array) to be filtered.

        Returns
        -------
        ndarray
            Filtered signal.
        """
        if len(h) != 2:
            raise IndexError("The half-length filter size should be 2")
        if h[0] != 0.0:
            raise ValueError("Antisymmetry violation (should have h[0]=0.0)")
        if len(c) < 2:
            return np.zeros_like(c)
        s = np.zeros_like(c)
        for i in range(1, len(c) - 1):
            s[i] = h[1] * (c[i + 1] - c[i - 1])
        return s

    def symmetric_fir_mirror_on_bounds(self, h, c):
        """
        Apply a symmetric FIR filter with mirror boundary extension.

        Parameters
        ----------
        h : ndarray
            Filter coefficients (expected length 2).
        c : ndarray
            Signal (or coefficient array) to be filtered.

        Returns
        -------
        ndarray
            Filtered signal.
        """
        if len(h) != 2:
            raise IndexError("The half-length filter size should be 2")
        if len(c) < 2:
            return c * (h[0] + 2.0 * h[1])
        s = np.zeros_like(c)
        s[0] = h[0] * c[0] + 2.0 * h[1] * c[1]
        for i in range(1, len(c) - 1):
            s[i] = h[0] * c[i] + h[1] * (c[i - 1] + c[i + 1])
        s[-1] = h[0] * c[-1] + 2.0 * h[1] * c[-2]
        return s

    def get_gradient(self, c):
        """
        Compute the first derivative (gradient) of a 1D signal using an anti-symmetric filter.

        Parameters
        ----------
        c : ndarray
            1D array of spline coefficients.

        Returns
        -------
        ndarray
            Computed gradient of the input signal.
        """
        h = np.array([0.0, 1.0 / 2.0])
        return self.anti_symmetric_fir_mirror_on_bounds(h, c)

    def get_hessian(self, c):
        """
        Compute the second derivative (Hessian) of a 1D signal using a symmetric filter.

        Parameters
        ----------
        c : ndarray
            1D array of spline coefficients.

        Returns
        -------
        ndarray
            Computed Hessian of the input signal.
        """
        h = np.array([-2.0, 1.0])
        return self.symmetric_fir_mirror_on_bounds(h, c)

    def get_spline_interpolation_coefficients(self, c, tolerance):
        """
        Compute the cubic B-spline interpolation coefficients for a 1D signal.

        This method adjusts the input signal `c` in place using a recursive scheme
        based on a cubic B-spline and a specified tolerance.

        Parameters
        ----------
        c : ndarray
            1D array representing the signal to be interpolated.
        tolerance : float
            Tolerance parameter controlling the trade-off between speed and accuracy.
        """
        # If the signal has less than 2 elements, no interpolation is needed.
        if len(c) < 2:
            return

        z = [np.sqrt(3.0) - 2.0]
        lambda_ = 1.0
        for zk in z:
            lambda_ *= (1.0 - zk) * (1.0 - 1.0 / zk)
        c *= lambda_
        for zk in z:
            c[0] = self.get_initial_causal_coefficient_mirror_on_bounds(
                c, zk, tolerance
            )
            for n in range(1, len(c)):
                c[n] += zk * c[n - 1]
            c[-1] = self.get_initial_anti_causal_coefficient_mirror_on_bounds(
                c, zk, tolerance
            )
            for n in range(len(c) - 2, -1, -1):
                c[n] = zk * (c[n + 1] - c[n])

    def get_initial_causal_coefficient_mirror_on_bounds(self, c, z, tolerance):
        """
        Compute the initial causal coefficient for spline interpolation with mirror boundary conditions.

        Parameters
        ----------
        c : ndarray
            1D array of spline coefficients.
        z : float
            Pole of the filter.
        tolerance : float
            Tolerance parameter to limit the recursion depth.

        Returns
        -------
        float
            The initial causal coefficient.
        """
        z1 = z
        zn = z ** (len(c) - 1)
        sum_ = c[0] + zn * c[-1]
        horizon = len(c)
        if tolerance > 0:
            horizon = min(horizon, 2 + int(np.log(tolerance) / np.log(np.abs(z))))
        zn *= zn
        for n in range(1, horizon - 1):
            zn /= z
            sum_ += (z1 + zn) * c[n]
            z1 *= z
        return sum_ / (1.0 - z ** (2 * len(c) - 2))

    def get_initial_anti_causal_coefficient_mirror_on_bounds(self, c, z, tolerance):
        """
        Compute the initial anti-causal coefficient for spline interpolation with mirror boundary conditions.

        Parameters
        ----------
        c : ndarray
            1D array of spline coefficients.
        z : float
            Pole of the filter.
        tolerance : float
            Tolerance parameter.

        Returns
        -------
        float
            The initial anti-causal coefficient.
        """
        return (z * c[-2] + c[-1]) * z / (z**2 - 1.0)

    def gradient_magnitude(self):
        """
        Compute the gradient magnitude of the image.

        Returns
        -------
        ndarray
            Image representing the gradient magnitude.
        """
        v_grad, h_grad = self.gradient_components()
        return np.sqrt(h_grad**2 + v_grad**2)

    def gradient_direction(self):
        """
        Compute the gradient direction of the image.

        Returns
        -------
        ndarray
            Image representing the gradient direction (in radians).
        """
        v_grad, h_grad = self.gradient_components()
        return np.arctan2(v_grad, h_grad)

    def laplacian(self):
        """
        Compute the Laplacian of the image.

        Returns
        -------
        ndarray
            Image representing the Laplacian.
        """
        v_hess, _, h_hess = self.hessian_components()
        return h_hess + v_hess

    def largest_hessian(self):
        """
        Compute the largest eigenvalue of the Hessian matrix of the image.

        Returns
        -------
        ndarray
            Image representing the largest Hessian eigenvalue.
        """
        v_hess, hv_hess, h_hess = self.hessian_components()
        return 0.5 * (
            h_hess + v_hess + np.sqrt(4.0 * hv_hess**2 + (h_hess - v_hess) ** 2)
        )

    def smallest_hessian(self):
        """
        Compute the smallest eigenvalue of the Hessian matrix of the image.

        Returns
        -------
        ndarray
            Image representing the smallest Hessian eigenvalue.
        """
        v_hess, hv_hess, h_hess = self.hessian_components()
        return 0.5 * (
            h_hess + v_hess - np.sqrt(4.0 * hv_hess**2 + (h_hess - v_hess) ** 2)
        )

    def hessian_orientation(self):
        """
        Compute the orientation of the Hessian of the image.

        Returns
        -------
        ndarray
            Image representing the Hessian orientation (in radians).
        """
        v_hess, hv_hess, h_hess = self.hessian_components()

        denominator = np.sqrt(4.0 * hv_hess**2 + (h_hess - v_hess) ** 2)
        # Avoid division by zero by setting denominator to a small value where it is zero
        denominator[denominator == 0] = self.FLT_EPSILON

        orientation = np.arccos((h_hess - v_hess) / denominator)
        return np.where(hv_hess < 0, -0.5 * orientation, 0.5 * orientation)


# Backward-compatible historical class name.  New code should use
# ``Differentials`` from ``splineops.differentials``.
differentials = Differentials
