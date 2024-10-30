import numpy as np
import numpy.typing as npt
from typing import Sequence
from splineops.interpolate.ls_oblique.utils import (
    beta, get_interpolation_coefficients, get_samples,
    do_integ, do_diff, calculate_final_size, border
)

class LS_Oblique_Resize:
    def __init__(self) -> None:
        # Initialization of parameters
        self.interp_degree: int = None       # Degree of the interpolation spline
        self.analy_degree: int = None        # Degree of the analysis spline
        self.synthe_degree: int = None       # Degree of the synthesis spline
        self.zoom_factors: Sequence[float] = None  # Zoom factors per dimension
        self.shifts: Sequence[float] = None         # Shift per dimension (usually zero)
        self.inversable: bool = None         # Whether to adjust sizes to ensure invertibility
        self.analy_even: int = 0             # Indicates symmetry of the analysis function (1 if even)
        self.corr_degree: int = None         # Degree used for correlation (analysis + synthesis + 1)
        self.half_support: float = None      # Half of the support size of the spline function
        self.spline_arrays: list[npt.NDArray] = []   # Precomputed spline values per dimension
        self.index_min_list: list[npt.NDArray] = []  # Minimum indices for spline evaluation per dimension
        self.index_max_list: list[npt.NDArray] = []  # Maximum indices for spline evaluation per dimension
        self.add_vector_list: list[npt.NDArray] = []         # Auxiliary vectors for resampling per dimension
        self.add_output_vector_list: list[npt.NDArray] = []  # Auxiliary output vectors for resampling per dimension
        self.period_sym_list: list[int] = []   # Periods for symmetric boundary conditions per dimension
        self.period_asym_list: list[int] = []  # Periods for antisymmetric boundary conditions per dimension
        self.length_totals: list[int] = []             # Total lengths of the extended signal per dimension
        self.length_output_totals: list[int] = []      # Total lengths of the output signal per dimension

    def compute_zoom(
        self, 
        input_img: npt.NDArray, 
        output_img: npt.NDArray, 
        analy_degree: int, 
        synthe_degree: int, 
        interp_degree: int, 
        zoom_factors: Sequence[float], 
        shifts: Sequence[float], 
        inversable: bool
    ) -> None:
        """
        Main function to compute the zoomed (resized) image.

        Parameters:
        - input_img: Input image to be resized.
        - output_img: Output image (preallocated) to store the resized image.
        - analy_degree: Degree of the analysis spline.
        - synthe_degree: Degree of the synthesis spline.
        - interp_degree: Degree of the interpolation spline.
        - zoom_factors: Zoom factors per dimension.
        - shifts: Shifts per dimension (usually zero).
        - inversable: Whether to adjust sizes to ensure invertibility.
        """
        self.interp_degree = interp_degree
        self.analy_degree = analy_degree
        self.synthe_degree = synthe_degree
        self.zoom_factors = zoom_factors
        self.shifts = shifts
        self.inversable = inversable

        n_dims = input_img.ndim
        input_shape = input_img.shape

        # Determine if the analysis degree is even or odd
        # This affects the boundary conditions (symmetric or antisymmetric)
        self.analy_even = ((analy_degree + 1) % 2 == 0)

        # Compute total degree and half of the support size
        total_degree = interp_degree + analy_degree + 1
        self.corr_degree = analy_degree + synthe_degree + 1
        self.half_support = (total_degree + 1) / 2.0

        # Calculate working and final sizes per dimension
        self.working_sizes, self.final_sizes = calculate_final_size(
            inversable, input_shape, zoom_factors)

        # Initialize lists to store per-dimension variables
        self.index_min_list = []
        self.index_max_list = []
        self.spline_arrays = []
        self.add_vector_list = []
        self.add_output_vector_list = []
        self.period_sym_list = []
        self.period_asym_list = []
        self.length_totals = []
        self.length_output_totals = []

        # Precompute spline coefficients and indices for each dimension
        for dim in range(n_dims):
            ny = self.working_sizes[dim]
            zoom = self.zoom_factors[dim]
            shift = self.shifts[dim]
            final_size = self.final_sizes[dim]

            # Compute the additional border required
            add_border = max(border(final_size, self.corr_degree), total_degree)
            final_total_size = final_size + add_border
            length_total = ny + int(np.ceil(add_border / zoom))
            self.length_totals.append(length_total)
            self.length_output_totals.append(final_total_size)

            # Shift adjustments to align the sampling grids
            shift += ((analy_degree + 1.0) / 2.0 - np.floor((analy_degree + 1.0) / 2.0)) * (1.0 / zoom - 1.0)
            # Scaling factor for the spline coefficients
            fact = np.power(zoom, analy_degree + 1)

            # Compute the affine indices (positions in the input image corresponding to output positions)
            l_range = np.arange(final_total_size)
            affine_indices = l_range / zoom + shift
            # Determine the range of indices over which the spline function is non-zero
            index_min = np.ceil(affine_indices - self.half_support).astype(int)
            index_max = np.floor(affine_indices + self.half_support).astype(int)

            # Initialize spline array to store precomputed spline values
            length_spline_array = final_total_size * (2 + total_degree)
            spline_array = np.zeros(length_spline_array)

            # Compute the spline coefficients for each output position
            i = 0
            for l in range(final_total_size):
                for k in range(index_min[l], index_max[l] + 1):
                    spline_array[i] = fact * beta(affine_indices[l] - k, total_degree)
                    i += 1

            # Store per-dimension variables
            self.index_min_list.append(index_min)
            self.index_max_list.append(index_max)
            self.spline_arrays.append(spline_array)

            # Periods for boundary conditions (used in signal extension)
            period_sym = 2 * ny - 2        # For symmetric extension
            period_asym = 2 * ny - 3       # For antisymmetric extension
            self.period_sym_list.append(period_sym)
            self.period_asym_list.append(period_asym)

            # Initialize auxiliary vectors for resampling
            add_vector = np.zeros(length_total)
            add_output_vector = np.zeros(final_total_size)
            self.add_vector_list.append(add_vector)
            self.add_output_vector_list.append(add_output_vector)

        # Begin resizing process
        image = input_img.copy()

        # Process each dimension separately
        for dim in range(n_dims):
            # Move current dimension to axis 0 for processing
            image = np.moveaxis(image, dim, 0)

            # Prepare per-dimension variables
            index_min = self.index_min_list[dim]
            index_max = self.index_max_list[dim]
            spline_array = self.spline_arrays[dim]
            period_sym = self.period_sym_list[dim]
            period_asym = self.period_asym_list[dim]
            add_vector = self.add_vector_list[dim]
            add_output_vector = self.add_output_vector_list[dim]
            length_total = self.length_totals[dim]
            length_output_total = self.length_output_totals[dim]
            output_size = self.final_sizes[dim]

            # Reshape image for processing along current dimension
            shape = image.shape
            reshaped_image = image.reshape(shape[0], -1)
            output_shape = (output_size,) + shape[1:]
            output_image = np.zeros(output_shape, dtype=image.dtype)
            reshaped_output = output_image.reshape(output_shape[0], -1)

            # Process each line (1D signal along the current dimension)
            for idx in range(reshaped_image.shape[1]):
                input_vector = reshaped_image[:, idx]
                output_vector = np.zeros(output_size)

                # Get interpolation coefficients for the input vector
                get_interpolation_coefficients(input_vector, interp_degree)

                # Resampling step: compute the output vector
                self.resampling(
                    input_vector=input_vector,
                    output_vector=output_vector,
                    add_vector=add_vector,
                    add_output_vector=add_output_vector,
                    max_sym_boundary=period_sym,
                    max_asym_boundary=period_asym,
                    index_min=index_min,
                    index_max=index_max,
                    spline_array=spline_array
                )

                # Store the output vector
                reshaped_output[:, idx] = output_vector

            # Reshape back to original dimensions and move axis back
            image = output_image.reshape(output_shape)
            image = np.moveaxis(image, 0, dim)

        # Copy the final resized image to output_img
        np.copyto(output_img, image)

    def resampling(
        self, 
        input_vector: npt.NDArray, 
        output_vector: npt.NDArray, 
        add_vector: npt.NDArray, 
        add_output_vector: npt.NDArray, 
        max_sym_boundary: int, 
        max_asym_boundary: int, 
        index_min: npt.NDArray, 
        index_max: npt.NDArray, 
        spline_array: npt.NDArray
    ) -> None:
        """
        Perform resampling of a 1D signal (vector).

        Parameters:
        - input_vector: The input signal to resample.
        - output_vector: The output resampled signal.
        - add_vector: Auxiliary vector for extended signal.
        - add_output_vector: Auxiliary vector for output computation.
        - max_sym_boundary: Period for symmetric boundary extension.
        - max_asym_boundary: Period for antisymmetric boundary extension.
        - index_min: Minimum indices for spline evaluation.
        - index_max: Maximum indices for spline evaluation.
        - spline_array: Precomputed spline coefficients.
        """
        length_input = len(input_vector)
        length_output = len(output_vector)
        length_total = len(add_vector)
        length_output_total = len(add_output_vector)
        average = 0

        # Projection Method: If analy_degree != -1, perform the projection step
        if self.analy_degree != -1:
            # Integrate the input vector analy_degree + 1 times
            average = do_integ(input_vector, self.analy_degree + 1)

        # Copy the input vector into the beginning of add_vector
        add_vector[:length_input] = input_vector

        # Extend the signal beyond its original length using symmetric or antisymmetric extension
        l = np.arange(length_input, length_total)
        if self.analy_even == 1:
            # Symmetric extension
            l2 = np.where(l >= max_sym_boundary, np.abs(l % max_sym_boundary), l)
            l2 = np.where(l2 >= length_input, max_sym_boundary - l2, l2)
            add_vector[length_input:length_total] = input_vector[l2]
        else:
            # Antisymmetric extension
            l2 = np.where(l >= max_asym_boundary, np.abs(l % max_asym_boundary), l)
            l2 = np.where(l2 >= length_input, max_asym_boundary - l2, l2)
            add_vector[length_input:length_total] = -input_vector[l2]

        # Initialize the output vector
        add_output_vector.fill(0.0)

        # Perform the convolution with the spline coefficients
        i = 0
        for l in range(length_output_total):
            for k in range(index_min[l], index_max[l] + 1):
                index = k
                sign = 1
                if k < 0:
                    # Handle negative indices (mirror at zero)
                    index = -k
                    if self.analy_even == 0:
                        index -= 1
                        sign = -1
                if k >= length_total:
                    # Handle indices beyond the extended signal
                    index = length_total - 1
                # Accumulate the weighted contributions
                add_output_vector[l] += sign * add_vector[index] * spline_array[i]
                i += 1

        # Projection Method: Differentiation and filtering steps
        if self.analy_degree != -1:
            # Differentiate the signal analy_degree + 1 times
            do_diff(add_output_vector, self.analy_degree + 1)
            # Add the average value back to the signal
            add_output_vector[:length_output_total] += average
            # Apply IIR filtering to obtain interpolation coefficients
            get_interpolation_coefficients(add_output_vector, self.corr_degree)
            # Extract samples from the continuous representation
            get_samples(add_output_vector, self.synthe_degree)

        # Copy the computed values to the output vector
        output_vector[:length_output] = add_output_vector[:length_output]

def ls_oblique_resize(
    input_img_normalized: npt.NDArray, 
    output_size: Sequence[int] = None, 
    zoom_factors: Sequence[float] = None, 
    method: str = 'Least-Squares', 
    interpolation: str = 'Linear', 
    inversable: bool = False
) -> npt.NDArray:
    """
    Resize an image using spline interpolation.

    Parameters:
    - input_img_normalized: The input image to be resized.
    - output_size: Desired output image size per dimension.
    - zoom_factors: Zoom factors per dimension (used if output_size is not provided).
    - method: Interpolation method ('Interpolation', 'Least-Squares', 'Oblique projection').
    - interpolation: Type of interpolation ('Linear', 'Quadratic', 'Cubic').
    - inversable: If True, adjust sizes to ensure invertibility (output size may change slightly).

    Returns:
    - output_image: The resized image.
    """
    n_dims = input_img_normalized.ndim
    input_shape = input_img_normalized.shape

    # Determine the zoom factors if output_size is provided
    if output_size is not None:
        zoom_factors = [output_size[i] / input_shape[i] for i in range(n_dims)]
    elif zoom_factors is not None:
        if len(zoom_factors) != n_dims:
            raise ValueError(f"zoom_factors must have {n_dims} elements.")
    else:
        raise ValueError("Either output_size or zoom_factors must be provided.")

    # Initialize shifts per dimension (usually zero)
    shifts = [0.0] * n_dims

    # Set degrees based on interpolation method
    if interpolation == "Linear":
        interp_degree = 1
        synthe_degree = 1
        analy_degree = 1
    elif interpolation == "Quadratic":
        interp_degree = 2
        synthe_degree = 2
        analy_degree = 2
    else:  # Cubic
        interp_degree = 3
        synthe_degree = 3
        analy_degree = 3

    # Adjust degrees based on method
    if method == "Interpolation":
        analy_degree = -1  # No analysis degree needed for interpolation
    elif method == "Oblique projection":
        # For oblique projection, the analysis degree may be different
        if interpolation == "Linear":
            analy_degree = 0
        elif interpolation == "Quadratic":
            analy_degree = 1
        else:  # Cubic
            analy_degree = 1

    # Compute output image size based on inversable parameter
    if inversable:
        # Adjust sizes to ensure invertibility
        working_sizes, final_sizes = calculate_final_size(
            inversable, input_shape, zoom_factors)
        output_shape = tuple(final_sizes)
        # Inform the user if the output size has changed
        if output_size is not None and output_shape != tuple(output_size):
            print(f"Note: Output size adjusted to {output_shape} to ensure invertibility.")
    else:
        output_shape = tuple([int(np.round(input_shape[i] * zoom_factors[i])) for i in range(n_dims)])

    # Create the output image with the correct size
    output_image = np.zeros(output_shape, dtype=np.float64)

    # Create an instance of LS_Oblique_Resize class
    resizer = LS_Oblique_Resize()

    # Perform resizing with a copy of the input image
    input_image_copy = input_img_normalized.copy()

    # Perform the resizing operation
    resizer.compute_zoom(
        input_image_copy,
        output_image,
        analy_degree,
        synthe_degree,
        interp_degree,
        zoom_factors,
        shifts,
        inversable=inversable
    )

    return output_image
