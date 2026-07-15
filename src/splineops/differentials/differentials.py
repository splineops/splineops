# splineops/src/splineops/differentials/differentials.py

from dataclasses import dataclass
import operator

import numpy as np

from splineops.spline_interpolation._prefilter import (
    prefilter_interpolation_coefficients,
)
from splineops.spline_interpolation.bases.utils import asbasis
from splineops.spline_interpolation.modes.utils import asmode

_CUBIC_BASIS = asbasis("bspline3")
_MIRROR_MODE = asmode("mirror")


def _validate_spacing(spacing, ndim):
    if spacing is None:
        return (1.0,) * ndim
    try:
        spacing = tuple(spacing)
    except TypeError as exc:
        raise TypeError(f"'spacing' must contain {ndim} positive real values.") from exc
    if len(spacing) != ndim:
        raise ValueError(f"'spacing' must contain exactly {ndim} values.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not np.isscalar(value)
        or not np.isreal(value)
        for value in spacing
    ):
        raise TypeError(f"'spacing' must contain {ndim} positive real values.")
    spacing = tuple(float(value) for value in spacing)
    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError("'spacing' values must be finite and positive.")
    return spacing


def _validate_shape(shape):
    try:
        shape = tuple(operator.index(length) for length in shape)
    except TypeError as exc:
        raise TypeError("'shape' must contain two or three integer lengths.") from exc
    if len(shape) not in (2, 3) or any(length <= 0 for length in shape):
        raise ValueError("'shape' must contain two or three positive lengths.")
    return shape


def _axis_coefficients(image, axis):
    return prefilter_interpolation_coefficients(
        image,
        bases=(_CUBIC_BASIS,),
        modes=(_MIRROR_MODE,),
        axes=(axis,),
        dtype=image.dtype,
        backend="numpy",
    )


def _derivative_stencil(coefficients, axis, order, spacing):
    """Apply a cubic-spline derivative sample stencil along one axis."""

    result = np.zeros_like(coefficients)
    length = coefficients.shape[axis]
    if length < 2:
        return result
    middle = [slice(None)] * coefficients.ndim
    previous = [slice(None)] * coefficients.ndim
    following = [slice(None)] * coefficients.ndim
    middle[axis] = slice(1, -1)
    previous[axis] = slice(None, -2)
    following[axis] = slice(2, None)
    if order == 1:
        result[tuple(middle)] = (
            0.5
            * (coefficients[tuple(following)] - coefficients[tuple(previous)])
            / spacing
        )
        return result
    if order != 2:
        raise ValueError("Derivative order must be one or two.")

    result = -2.0 * coefficients
    result[tuple(middle)] += (
        coefficients[tuple(previous)] + coefficients[tuple(following)]
    )
    first = [slice(None)] * coefficients.ndim
    second = [slice(None)] * coefficients.ndim
    last = [slice(None)] * coefficients.ndim
    penultimate = [slice(None)] * coefficients.ndim
    first[axis] = 0
    second[axis] = 1
    last[axis] = -1
    penultimate[axis] = -2
    result[tuple(first)] += 2.0 * coefficients[tuple(second)]
    result[tuple(last)] += 2.0 * coefficients[tuple(penultimate)]
    return result / spacing**2


class _DifferentialWorkspace:
    """Lazy per-array cache shared by all derivative outputs."""

    def __init__(self, image, spacing):
        self.image = image
        self.spacing = spacing
        self._coefficients = {}
        self._gradients = {}
        self._diagonal_hessians = {}
        self._mixed_hessians = {}

    def coefficients(self, axis):
        if axis not in self._coefficients:
            self._coefficients[axis] = _axis_coefficients(self.image, axis)
        return self._coefficients[axis]

    def gradient(self, axis):
        if axis not in self._gradients:
            self._gradients[axis] = _derivative_stencil(
                self.coefficients(axis), axis, 1, self.spacing[axis]
            )
        return self._gradients[axis]

    def diagonal_hessian(self, axis):
        if axis not in self._diagonal_hessians:
            self._diagonal_hessians[axis] = _derivative_stencil(
                self.coefficients(axis), axis, 2, self.spacing[axis]
            )
        return self._diagonal_hessians[axis]

    def mixed_hessian(self, first_axis, second_axis):
        axes = tuple(sorted((first_axis, second_axis)))
        if axes not in self._mixed_hessians:
            first_derivative = self.gradient(axes[1])
            coefficients = _axis_coefficients(first_derivative, axes[0])
            self._mixed_hessians[axes] = _derivative_stencil(
                coefficients, axes[0], 1, self.spacing[axes[0]]
            )
        return self._mixed_hessians[axes]

    def gradient_components(self):
        return tuple(self.gradient(axis) for axis in range(self.image.ndim))

    def hessian_components(self):
        return tuple(
            (
                self.diagonal_hessian(first)
                if first == second
                else self.mixed_hessian(first, second)
            )
            for first in range(self.image.ndim)
            for second in range(first, self.image.ndim)
        )


@dataclass(frozen=True)
class DifferentialResult:
    """Multi-output result returned by :class:`DifferentialPlan`.

    Hessian entries use packed upper-triangular order: ``(00, 01, 11)`` in
    2-D and ``(00, 01, 02, 11, 12, 22)`` in 3-D.
    """

    gradient: tuple[np.ndarray, ...] | None
    hessian: tuple[np.ndarray, ...] | None
    laplacian: np.ndarray | None


class DifferentialPlan:
    """Reusable shape/spacing contract for 2-D and 3-D spline derivatives."""

    def __init__(self, shape, spacing=None):
        self.shape = _validate_shape(shape)
        self.spacing = _validate_spacing(spacing, len(self.shape))

    def apply(self, image, *, gradient=True, hessian=True):
        """Compute requested derivative families through one cached workspace."""

        if not isinstance(gradient, (bool, np.bool_)) or not isinstance(
            hessian, (bool, np.bool_)
        ):
            raise TypeError("'gradient' and 'hessian' must be booleans.")
        image = np.asarray(image)
        if image.shape != self.shape:
            raise ValueError(f"'image' must have shape {self.shape}.")
        if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
            raise TypeError("'image' must have a real numeric dtype.")
        if not np.all(np.isfinite(image)):
            raise ValueError("'image' must contain only finite values.")
        work_dtype = (
            image.dtype
            if np.issubdtype(image.dtype, np.floating)
            else np.dtype(np.float64)
        )
        workspace = _DifferentialWorkspace(
            image.astype(work_dtype, copy=True), self.spacing
        )
        gradient_result = workspace.gradient_components() if gradient else None
        hessian_result = workspace.hessian_components() if hessian else None
        laplacian = None
        if hessian_result is not None:
            diagonal_indices = (0, 2) if len(self.shape) == 2 else (0, 3, 5)
            laplacian = sum(hessian_result[index] for index in diagonal_indices)
        return DifferentialResult(gradient_result, hessian_result, laplacian)

    __call__ = apply


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

    def __init__(self, image, spacing=None):
        """
        Initialize a new differentials instance.

        Parameters
        ----------
        image : ndarray
            Input scalar image or volume as a 2D or 3D NumPy array.
        spacing : tuple of float, optional
            Positive physical sample spacing for every axis.  The default is
            unit spacing.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError("'image' must be a NumPy array.")
        if image.ndim not in (2, 3):
            raise ValueError("'image' must be a two- or three-dimensional array.")
        if any(length == 0 for length in image.shape):
            raise ValueError("'image' dimensions must be non-empty.")
        if not np.issubdtype(image.dtype, np.number) or np.iscomplexobj(image):
            raise TypeError("'image' must have a real numeric dtype.")
        if not np.all(np.isfinite(image)):
            raise ValueError("'image' must contain only finite values.")
        spacing = _validate_spacing(spacing, image.ndim)
        work_dtype = (
            image.dtype
            if np.issubdtype(image.dtype, np.floating)
            else np.dtype(np.float64)
        )
        self.image = image.astype(work_dtype, copy=True)
        self.spacing = spacing
        self.shape = image.shape
        self.height, self.width = image.shape[-2:]
        self.operation = self.LAPLACIAN
        self._workspace = _DifferentialWorkspace(self.image, self.spacing)

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
        if image is self.image:
            return self._workspace.coefficients(axis)
        return _axis_coefficients(image, axis)

    def _gradient_along_axis(self, image, axis):
        if image is self.image:
            return self._workspace.gradient(axis)
        coefficients = self._coefficients_along_axis(image, axis)
        return _derivative_stencil(coefficients, axis, 1, self.spacing[axis])

    def horizontal_gradient(self):
        """Return the derivative along increasing column coordinates."""
        return self._gradient_along_axis(self.image, axis=self.image.ndim - 1)

    def vertical_gradient(self):
        """Return the derivative along increasing row coordinates."""
        return self._gradient_along_axis(self.image, axis=self.image.ndim - 2)

    def gradient_components(self):
        """Return first derivatives in increasing axis order."""
        return self._workspace.gradient_components()

    def horizontal_hessian(self):
        """Return the second derivative along column coordinates."""
        return self._hessian_along_axis(self.image, axis=self.image.ndim - 1)

    def vertical_hessian(self):
        """Return the second derivative along row coordinates."""
        return self._hessian_along_axis(self.image, axis=self.image.ndim - 2)

    def cross_hessian(self):
        """Return the mixed row-column second derivative."""
        return self._workspace.mixed_hessian(self.image.ndim - 2, self.image.ndim - 1)

    def hessian_components(self):
        """Return packed upper-triangular Hessian components."""
        return self._workspace.hessian_components()

    def _hessian_along_axis(self, image, axis):
        if image is self.image:
            return self._workspace.diagonal_hessian(axis)
        coefficients = self._coefficients_along_axis(image, axis)
        return _derivative_stencil(coefficients, axis, 2, self.spacing[axis])

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
        components = self.gradient_components()
        squared = components[0] ** 2
        for component in components[1:]:
            squared = squared + component**2
        return np.sqrt(squared)

    def gradient_direction(self):
        """
        Compute the gradient direction of the image.

        Returns
        -------
        ndarray
            Image representing the gradient direction (in radians).
        """
        if self.image.ndim != 2:
            raise ValueError("Gradient direction is defined only for 2-D images.")
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
        components = self.hessian_components()
        diagonal_indices = (0, 2) if self.image.ndim == 2 else (0, 3, 5)
        return sum(components[index] for index in diagonal_indices)

    def _hessian_eigenvalues(self):
        components = self.hessian_components()
        ndim = self.image.ndim
        matrices = np.empty(self.image.shape + (ndim, ndim), dtype=self.image.dtype)
        index = 0
        for first in range(ndim):
            for second in range(first, ndim):
                matrices[..., first, second] = components[index]
                matrices[..., second, first] = components[index]
                index += 1
        return np.linalg.eigvalsh(matrices)

    def largest_hessian(self):
        """
        Compute the largest eigenvalue of the Hessian matrix of the image.

        Returns
        -------
        ndarray
            Image representing the largest Hessian eigenvalue.
        """
        if self.image.ndim == 3:
            return self._hessian_eigenvalues()[..., -1]
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
        if self.image.ndim == 3:
            return self._hessian_eigenvalues()[..., 0]
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
        if self.image.ndim != 2:
            raise ValueError("Hessian orientation is defined only for 2-D images.")
        v_hess, hv_hess, h_hess = self.hessian_components()

        denominator = np.sqrt(4.0 * hv_hess**2 + (h_hess - v_hess) ** 2)
        # Avoid division by zero by setting denominator to a small value where it is zero
        denominator[denominator == 0] = self.FLT_EPSILON

        orientation = np.arccos((h_hess - v_hess) / denominator)
        return np.where(hv_hess < 0, -0.5 * orientation, 0.5 * orientation)


# Backward-compatible historical class name.  New code should use
# ``Differentials`` from ``splineops.differentials``.
differentials = Differentials
