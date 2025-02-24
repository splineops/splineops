import numpy as np
import time

class Differentials:
    GRADIENT_DIRECTION = 1
    GRADIENT_MAGNITUDE = 0
    HESSIAN_ORIENTATION = 5
    LAPLACIAN = 2
    LARGEST_HESSIAN = 3
    SMALLEST_HESSIAN = 4

    FLT_EPSILON = np.finfo(np.float32).eps

    def __init__(self, image):
        self.image = image.astype(np.float32)
        self.height, self.width = image.shape
        self.operation = self.LAPLACIAN
        self.completed = 1
        self.process_duration = 1
        self.stack_size = 1
        self.last_time = time.time()

    def run(self, operation=None):
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

        self.image -= self.image.min()
        self.image /= self.image.max()
        print(f"Completed in {time.time() - start_time:.2f} seconds")

    def get_cross_hessian(self, image, tolerance):
        h_grad = self.get_horizontal_gradient(image, tolerance)
        v_grad = self.get_vertical_gradient(image, tolerance)
        return h_grad * v_grad

    def get_horizontal_gradient(self, image, tolerance):
        output = np.zeros_like(image)
        for y in range(self.height):
            line = image[y, :]
            self.get_spline_interpolation_coefficients(line, tolerance)
            gradient = self.get_gradient(line)
            output[y, :] = gradient
            self.step_progress_bar()
        return output

    def get_horizontal_hessian(self, image, tolerance):
        output = np.zeros_like(image)
        for y in range(self.height):
            line = image[y, :]
            self.get_spline_interpolation_coefficients(line, tolerance)
            hessian = self.get_hessian(line)
            output[y, :] = hessian
            self.step_progress_bar()
        return output

    def get_vertical_gradient(self, image, tolerance):
        output = np.zeros_like(image)
        for x in range(self.width):
            line = image[:, x]
            self.get_spline_interpolation_coefficients(line, tolerance)
            gradient = self.get_gradient(line)
            output[:, x] = gradient
            self.step_progress_bar()
        return output

    def get_vertical_hessian(self, image, tolerance):
        output = np.zeros_like(image)
        for x in range(self.width):
            line = image[:, x]
            self.get_spline_interpolation_coefficients(line, tolerance)
            hessian = self.get_hessian(line)
            output[:, x] = hessian
            self.step_progress_bar()
        return output

    def anti_symmetric_fir_mirror_on_bounds(self, h, c):
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

    def clean_up_progress_bar(self):
        self.completed = 0
        self.show_progress(1.0)

    def show_progress(self, progress):
        print(f"Progress: {progress:.2%}")

    def step_progress_bar(self):
        self.completed += 1
        current_time = time.time()
        if current_time - self.last_time > 0.05:
            self.last_time = current_time
            self.show_progress(self.completed / self.process_duration)

    def symmetric_fir_mirror_on_bounds(self, h, c):
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
        h = np.array([0.0, -1.0 / 2.0])
        return self.anti_symmetric_fir_mirror_on_bounds(h, c)

    def get_hessian(self, c):
        h = np.array([-2.0, 1.0])
        return self.symmetric_fir_mirror_on_bounds(h, c)

    def get_spline_interpolation_coefficients(self, c, tolerance):
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
        return (z * c[-2] + c[-1]) * z / (z ** 2 - 1.0)

    def gradient_magnitude(self):
        h_grad = self.get_horizontal_gradient(self.image.copy(), self.FLT_EPSILON)
        v_grad = self.get_vertical_gradient(self.image.copy(), self.FLT_EPSILON)
        return np.sqrt(h_grad ** 2 + v_grad ** 2)

    def gradient_direction(self):
        h_grad = self.get_horizontal_gradient(self.image.copy(), self.FLT_EPSILON)
        v_grad = self.get_vertical_gradient(self.image.copy(), self.FLT_EPSILON)
        return np.arctan2(v_grad, h_grad)

    def laplacian(self):
        h_hess = self.get_horizontal_hessian(self.image.copy(), self.FLT_EPSILON)
        v_hess = self.get_vertical_hessian(self.image.copy(), self.FLT_EPSILON)
        return h_hess + v_hess

    def largest_hessian(self):
        h_hess = self.get_horizontal_hessian(self.image.copy(), self.FLT_EPSILON)
        v_hess = self.get_vertical_hessian(self.image.copy(), self.FLT_EPSILON)
        hv_hess = self.get_cross_hessian(self.image.copy(), self.FLT_EPSILON)
        return 0.5 * (h_hess + v_hess + np.sqrt(4.0 * hv_hess ** 2 + (h_hess - v_hess) ** 2))

    def smallest_hessian(self):
        h_hess = self.get_horizontal_hessian(self.image.copy(), self.FLT_EPSILON)
        v_hess = self.get_vertical_hessian(self.image.copy(), self.FLT_EPSILON)
        hv_hess = self.get_cross_hessian(self.image.copy(), self.FLT_EPSILON)
        return 0.5 * (h_hess + v_hess - np.sqrt(4.0 * hv_hess ** 2 + (h_hess - v_hess) ** 2))

    def hessian_orientation(self):
        h_hess = self.get_horizontal_hessian(self.image.copy(), self.FLT_EPSILON)
        v_hess = self.get_vertical_hessian(self.image.copy(), self.FLT_EPSILON)
        hv_hess = self.get_cross_hessian(self.image.copy(), self.FLT_EPSILON)
        
        denominator = np.sqrt(4.0 * hv_hess ** 2 + (h_hess - v_hess) ** 2)
        # Avoid division by zero by setting denominator to a small value where it is zero
        denominator[denominator == 0] = self.FLT_EPSILON
        
        orientation = np.arccos((h_hess - v_hess) / denominator)
        return np.where(hv_hess < 0, -0.5 * orientation, 0.5 * orientation)
