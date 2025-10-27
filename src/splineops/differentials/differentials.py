# splineops/src/splineops/differentials/differentials.py

import numpy as np
import time

class differentials:
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

    def __init__(self, image):
        """
        Initialize a new differentials instance.

        Parameters
        ----------
        image : ndarray
            Input grayscale image as a 2D numpy array.
        """
        self.image = image.astype(np.float32)
        self.height, self.width = image.shape
        self.operation = self.LAPLACIAN
        self.completed = 1
        self.process_duration = 1
        self.stack_size = 1
        self.last_time = time.time()

    def run(self, operation=None):
        """
        Execute the selected differential operation on the image.

        Parameters
        ----------
        operation : int, optional
            Operation to perform. If None, the default operation (Laplacian) is used.
        """
        if operation is not None:
            self.operation = operation

        start_time = time.time()

        if self.operation == self.GRADIENT_MAGNITUDE:
            self.image = self.gradient_magnitude()
        elif self.operation == self.GRADIENT_DIRECTION:
            self.image = self.gradient_direction()
        elif self.operation == self.LAPLACIAN:
            self.image = self.laplacian()
        elif self.operation == self.LARGEST_HESSIAN:
            self.image = self.largest_hessian()
        elif self.operation == self.SMALLEST_HESSIAN:
            self.image = self.smallest_hessian()
        elif self.operation == self.HESSIAN_ORIENTATION:
            self.image = self.hessian_orientation()

        if self.operation not in [self.GRADIENT_DIRECTION, self.HESSIAN_ORIENTATION]:
            self.image -= self.image.min()
            self.image /= self.image.max()

        print(f"Completed in {time.time() - start_time:.2f} seconds")

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
        output = np.zeros_like(image)
        for y in range(self.height):
            line = image[y, :]
            self.get_spline_interpolation_coefficients(line, tolerance)
            gradient = self.get_gradient(line)
            output[y, :] = gradient
        return output

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
        output = np.zeros_like(image)
        for y in range(self.height):
            line = image[y, :]
            self.get_spline_interpolation_coefficients(line, tolerance)
            hessian = self.get_hessian(line)
            output[y, :] = hessian
        return output

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
        output = np.zeros_like(image)
        for x in range(self.width):
            line = image[:, x]
            self.get_spline_interpolation_coefficients(line, tolerance)
            gradient = self.get_gradient(line)
            output[:, x] = gradient
        return output

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
        output = np.zeros_like(image)
        for x in range(self.width):
            line = image[:, x]
            self.get_spline_interpolation_coefficients(line, tolerance)
            hessian = self.get_hessian(line)
            output[:, x] = hessian
        return output

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
        h = np.array([0.0, -1.0 / 2.0])
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
            c[0] = self.get_initial_causal_coefficient_mirror_on_bounds(c, zk, tolerance)
            for n in range(1, len(c)):
                c[n] += zk * c[n - 1]
            c[-1] = self.get_initial_anti_causal_coefficient_mirror_on_bounds(c, zk, tolerance)
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
        return (z * c[-2] + c[-1]) * z / (z ** 2 - 1.0)

    def gradient_magnitude(self):
        """
        Compute the gradient magnitude of the image.

        Returns
        -------
        ndarray
            Image representing the gradient magnitude.
        """
        h_grad = self.get_horizontal_gradient(self.image.copy(), self.FLT_EPSILON)
        v_grad = self.get_vertical_gradient(self.image.copy(), self.FLT_EPSILON)
        return np.sqrt(h_grad ** 2 + v_grad ** 2)

    def gradient_direction(self):
        """
        Compute the gradient direction of the image.

        Returns
        -------
        ndarray
            Image representing the gradient direction (in radians).
        """
        h_grad = self.get_horizontal_gradient(self.image.copy(), self.FLT_EPSILON)
        v_grad = self.get_vertical_gradient(self.image.copy(), self.FLT_EPSILON)
        return np.arctan2(v_grad, h_grad)

    def laplacian(self):
        """
        Compute the Laplacian of the image.

        Returns
        -------
        ndarray
            Image representing the Laplacian.
        """
        h_hess = self.get_horizontal_hessian(self.image.copy(), self.FLT_EPSILON)
        v_hess = self.get_vertical_hessian(self.image.copy(), self.FLT_EPSILON)
        return h_hess + v_hess

    def largest_hessian(self):
        """
        Compute the largest eigenvalue of the Hessian matrix of the image.

        Returns
        -------
        ndarray
            Image representing the largest Hessian eigenvalue.
        """
        h_hess = self.get_horizontal_hessian(self.image.copy(), self.FLT_EPSILON)
        v_hess = self.get_vertical_hessian(self.image.copy(), self.FLT_EPSILON)
        hv_hess = self.get_cross_hessian(self.image.copy(), self.FLT_EPSILON)
        return 0.5 * (h_hess + v_hess + np.sqrt(4.0 * hv_hess**2 + (h_hess - v_hess)**2))

    def smallest_hessian(self):
        """
        Compute the smallest eigenvalue of the Hessian matrix of the image.

        Returns
        -------
        ndarray
            Image representing the smallest Hessian eigenvalue.
        """
        h_hess = self.get_horizontal_hessian(self.image.copy(), self.FLT_EPSILON)
        v_hess = self.get_vertical_hessian(self.image.copy(), self.FLT_EPSILON)
        hv_hess = self.get_cross_hessian(self.image.copy(), self.FLT_EPSILON)
        return 0.5 * (h_hess + v_hess - np.sqrt(4.0 * hv_hess ** 2 + (h_hess - v_hess) ** 2))

    def hessian_orientation(self):
        """
        Compute the orientation of the Hessian of the image.

        Returns
        -------
        ndarray
            Image representing the Hessian orientation (in radians).
        """
        h_hess = self.get_horizontal_hessian(self.image.copy(), self.FLT_EPSILON)
        v_hess = self.get_vertical_hessian(self.image.copy(), self.FLT_EPSILON)
        hv_hess = self.get_cross_hessian(self.image.copy(), self.FLT_EPSILON)
        
        denominator = np.sqrt(4.0 * hv_hess ** 2 + (h_hess - v_hess) ** 2)
        # Avoid division by zero by setting denominator to a small value where it is zero
        denominator[denominator == 0] = self.FLT_EPSILON
        
        orientation = np.arccos((h_hess - v_hess) / denominator)
        return np.where(hv_hess < 0, -0.5 * orientation, 0.5 * orientation)
