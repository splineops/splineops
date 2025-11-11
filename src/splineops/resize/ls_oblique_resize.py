# splineops/src/splineops/resize/ls_oblique_resize.py

# LS Oblique Resize
# =================

# This Python implementation is inspired by the Resize plugin for ImageJ, which performs image resizing
# using least-squares oblique image resizing with spline interpolation.

# Author: Arrate Munoz
#         Swiss Federal Institute of Technology Lausanne
#         Biomedical Imaging Group
#         BM-Ecublens
#         CH-1015 Lausanne EPFL, Switzerland

# Original Java version: July 11, 2001

# This Python class implements similar functionality with adjustments for Python's numpy-based ecosystem.


import numpy as np
import numpy.typing as npt
from typing import Sequence, Dict, Any, List
from splineops.resize.utils import (
    beta, get_interpolation_coefficients, get_samples,
    do_integ, do_diff, calculate_final_size, border
)

class LS_Oblique_Resize:
    """
    A class to perform least-squares oblique image resizing using spline interpolation.

    This class implements image resizing algorithms based on splines, allowing for
    interpolation, least-squares, and oblique projection methods with various spline degrees.

    Attributes
    ----------
    interp_degree : int
        Degree of the interpolation spline.
    analy_degree : int
        Degree of the analysis spline.
    synthe_degree : int
        Degree of the synthesis spline.
    zoom_factors : Sequence[float]
        Zoom factors per dimension.
    shifts : Sequence[float]
        Shifts per dimension (usually zero).
    inversable : bool
        Indicates if the resizing should be inversable.
    analy_even : int
        Indicates if the analysis function is even (1) or odd (0).
    corr_degree : int
        Degree used for correlation (analy_degree + synthe_degree + 1).
    half_support : float
        Half of the support size of the spline function.
    spline_arrays : list of np.ndarray
        Precomputed spline values per dimension.
    index_min_list : list of np.ndarray
        Minimum indices for spline evaluation per dimension.
    index_max_list : list of np.ndarray
        Maximum indices for spline evaluation per dimension.
    add_vector_list : list of np.ndarray
        Auxiliary vectors for resampling per dimension.
    add_output_vector_list : list of np.ndarray
        Auxiliary output vectors for resampling per dimension.
    period_sym_list : list of int
        Periods for symmetric boundary conditions per dimension.
    period_asym_list : list of int
        Periods for antisymmetric boundary conditions per dimension.
    length_totals : list of int
        Total lengths of the extended signal per dimension.
    length_output_totals : list of int
        Total lengths of the output signal per dimension.
    """

    def __init__(self) -> None:
        """
        Initialize the LS_Oblique_Resize object with default parameters.
        """
        # Initialization of parameters
        self.interp_degree: int = -1
        self.analy_degree: int = -1
        self.synthe_degree: int = -1
        self.zoom_factors: Sequence[float] = ()
        self.shifts: Sequence[float] = ()
        self.inversable: bool = False
        self.plans: List[Dict[str, Any]] = []   # per-dimension plans

    # ---------------------------
    # Plan builder (once per axis)
    # ---------------------------
    def _build_plan_for_axis(
        self,
        ny: int,
        zoom: float,
        shift: float,
        output_size: int,
        interp_degree: int,
        analy_degree: int,
        synthe_degree: int
    ) -> Dict[str, Any]:
        # degrees/support
        total_degree = interp_degree + analy_degree + 1
        corr_degree  = (interp_degree if analy_degree < 0 else analy_degree + synthe_degree + 1)
        half_support = 0.5 * (total_degree + 1)

        # shift used by analysis stage (matches C++ & tests)
        if analy_degree >= 0:
            t = (analy_degree + 1.0) / 2.0
            shift = shift + (t - np.floor(t)) * (1.0 / zoom - 1.0)

        # output tail sizing
        add_border   = max(border(output_size, corr_degree), total_degree)
        out_total    = output_size + add_border
        length_total = ny + int(np.ceil(add_border / zoom))

        # window metadata
        l = np.arange(out_total, dtype=np.float64)
        x = l / zoom + shift
        kmin = np.ceil(x - half_support).astype(np.int32)
        kmax = np.floor(x + half_support).astype(np.int32)
        wlen = (kmax - kmin + 1).astype(np.int32)
        win_len_max = int(wlen.max())

        # Build weights 2D (out_total, win_len_max); fill variable-length rows
        weights = np.zeros((out_total, win_len_max), dtype=np.float64)
        fact = (zoom ** (analy_degree + 1)) if analy_degree >= 0 else 1.0
        # compute per-row weights once (this happens once per axis)
        for i in range(out_total):
            m = int(wlen[i])
            if m <= 0:
                continue
            ks = kmin[i] + np.arange(m, dtype=np.int32)
            # beta is scalar; loop once per row is OK (done once/axis)
            row = np.empty(m, dtype=np.float64)
            dx = x[i] - ks.astype(np.float64)
            for t in range(m):
                row[t] = fact * beta(dx[t], total_degree)
            weights[i, :m] = row

        # left/right pad sizes for a unified buffer [LP | ext | RP]
        min_kmin = int(kmin.min())
        max_kmax = int(kmax.max())
        LP = max(0, -min_kmin)
        RP = max(0, max_kmax - (length_total - 1))

        # Precompute 2-D indices into ext_full for gathers: LP + (kmin + t)
        tgrid = np.arange(win_len_max, dtype=np.int32)[None, :]   # shape (1, win_len_max)
        idx2d = (LP + (kmin[:, None] + tgrid)).astype(np.int64)   # shape (out_total, win_len_max)

        # Save plan
        plan = dict(
            ny=ny,
            zoom=zoom,
            analy_degree=analy_degree,
            synthe_degree=synthe_degree,
            interp_degree=interp_degree,
            total_degree=total_degree,
            corr_degree=corr_degree,
            out_size=output_size,
            out_total=out_total,
            length_total=length_total,
            LP=LP, RP=RP,
            kmin=kmin, wlen=wlen,
            weights=weights,          # (out_total, win_len_max)
            idx2d=idx2d,              # (out_total, win_len_max)
            win_len_max=win_len_max,
            analy_even=int((analy_degree + 1) % 2 == 0),
            period_sym=2 * ny - 2,
            period_asym=2 * ny - 3,
        )
        return plan

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
        self.interp_degree = interp_degree
        self.analy_degree  = analy_degree
        self.synthe_degree = synthe_degree
        self.zoom_factors  = zoom_factors
        self.shifts        = shifts
        self.inversable    = inversable

        n_dims = input_img.ndim
        in_shape = input_img.shape

        # final sizes per axis
        working_sizes, final_sizes = calculate_final_size(inversable, in_shape, zoom_factors)

        # Build all plans once per dimension
        self.plans = []
        for ax in range(n_dims):
            plan = self._build_plan_for_axis(
                ny=working_sizes[ax],
                zoom=float(zoom_factors[ax]),
                shift=float(shifts[ax]),
                output_size=final_sizes[ax],
                interp_degree=interp_degree,
                analy_degree=analy_degree,
                synthe_degree=synthe_degree
            )
            self.plans.append(plan)

        # Resample axis by axis
        image = np.asarray(input_img, dtype=np.float64, order="C")
        for dim, plan in enumerate(self.plans):
            # move dim to front -> shape (N, rest)
            image = np.moveaxis(image, dim, 0)
            N = image.shape[0]
            cols = int(np.prod(image.shape[1:] or (1,)))
            X = image.reshape(N, cols)

            # prepare outputs
            Y = np.empty((plan["out_size"], cols), dtype=np.float64)

            # one set of reusable buffers per column
            coeff  = np.empty(N, dtype=np.float64)
            ext    = np.empty(plan["length_total"], dtype=np.float64)
            ext_full = np.empty(plan["LP"] + plan["length_total"] + plan["RP"], dtype=np.float64)
            add_out = np.empty(plan["out_total"], dtype=np.float64)  # for tail ops

            # process each 1-D line
            for j in range(cols):
                coeff[:] = X[:, j]
                # 1) interpolation coefficients
                get_interpolation_coefficients(coeff, plan["interp_degree"])

                # 2) optional integration
                average = 0.0
                if plan["analy_degree"] >= 0:
                    average = do_integ(coeff, plan["analy_degree"] + 1)

                # 3) build the finite extension ext[0:len_total]
                ext[:N] = coeff
                if plan["length_total"] > N:
                    l = np.arange(N, plan["length_total"])
                    if plan["analy_even"] == 1:  # symmetric
                        if plan["period_sym"] > 0:
                            lk = np.where(l >= plan["period_sym"], l % plan["period_sym"], l)
                        else:
                            lk = l
                        lk = np.where(lk >= N, plan["period_sym"] - lk, lk)
                        lk = np.clip(lk, 0, N - 1)
                        ext[N:] = coeff[lk]
                    else:  # antisymmetric
                        if plan["period_asym"] > 0:
                            lk = np.where(l >= plan["period_asym"], l % plan["period_asym"], l)
                        else:
                            lk = l
                        lk = np.where(lk >= N, plan["period_asym"] - lk, lk)
                        lk = np.clip(lk, 0, N - 1)
                        ext[N:] = -coeff[lk]

                # 3b) ext_full = [LP | ext | RP]
                if plan["LP"] > 0:
                    # precompute left pad once per line (mirror around zero)
                    # symmetric: -t -> +coeff[t]; antisym: -t -> -coeff[t-1]
                    t = np.arange(1, plan["LP"] + 1)
                    if plan["analy_even"] == 1:
                        src = np.clip(t, 0, N - 1)
                        ext_full[plan["LP"] - t] = coeff[src]
                    else:
                        src = np.clip(t - 1, 0, N - 1)
                        ext_full[plan["LP"] - t] = -coeff[src]
                ext_full[plan["LP"]:plan["LP"] + plan["length_total"]] = ext
                if plan["RP"] > 0:
                    ext_full[plan["LP"] + plan["length_total"]:] = ext[-1]

                # 4) accumulate with vectorized gather + dot
                # Gather all needed samples: (out_total, win_len_max)
                gather = np.take(ext_full, plan["idx2d"], mode="clip")
                # Zero out padded slots beyond each row length
                if plan["win_len_max"] > 1:
                    mask = (np.arange(plan["win_len_max"])[None, :] >= plan["wlen"][:, None])
                    if mask.any():
                        gather[mask] = 0.0

                add_out[:] = (plan["weights"] * gather).sum(axis=1)

                # 5) projection tail
                if plan["analy_degree"] >= 0:
                    do_diff(add_out, plan["analy_degree"] + 1)
                    add_out[:plan["out_total"]] += average
                    get_interpolation_coefficients(add_out, plan["corr_degree"])
                    get_samples(add_out, plan["synthe_degree"])

                # 6) crop to output size
                Y[:, j] = add_out[:plan["out_size"]]

            image = Y.reshape((plan["out_size"],) + image.shape[1:])
            image = np.moveaxis(image, 0, dim)

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
        Perform resampling of a 1D signal (vector) using spline interpolation.

        Parameters
        ----------
        input_vector : np.ndarray
            The input signal to resample.
        output_vector : np.ndarray
            The output resampled signal.
        add_vector : np.ndarray
            Auxiliary vector for extended signal.
        add_output_vector : np.ndarray
            Auxiliary vector for output computation.
        max_sym_boundary : int
            Period for symmetric boundary extension.
        max_asym_boundary : int
            Period for antisymmetric boundary extension.
        index_min : np.ndarray
            Minimum indices for spline evaluation.
        index_max : np.ndarray
            Maximum indices for spline evaluation.
        spline_array : np.ndarray
            Precomputed spline coefficients.
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
    method: str = 'least-squares',
    interpolation: str = 'linear',
    inversable: bool = False
) -> npt.NDArray:
    """
    Resize an image using spline interpolation.

    Parameters
    ----------
    input_img_normalized : np.ndarray
        The input image to be resized.
    output_size : tuple of ints, optional
        Desired output image size per dimension. If provided, zoom factors are computed from it.
    zoom_factors : tuple of floats, optional
        Zoom factors per dimension. Used if output_size is not provided.
    method : str, optional
        Interpolation method ('interpolation', 'least-squares', 'oblique').
    interpolation : str, optional
        Type of interpolation ('linear', 'quadratic', 'cubic').
    inversable : bool, optional
        If True, adjust sizes to ensure invertibility. Output size may change slightly.

    Returns
    -------
    output_image : np.ndarray
        The resized image.

    Raises
    ------
    ValueError
        If neither output_size nor zoom_factors are provided.
        If zoom_factors length does not match the number of dimensions.
    """
    n_dims = input_img_normalized.ndim
    input_shape = input_img_normalized.shape

    # Determine the zoom factors
    if output_size is not None:
        zoom_factors = [output_size[i] / input_shape[i] for i in range(n_dims)]
    elif zoom_factors is not None:
        if len(zoom_factors) != n_dims:
            raise ValueError(f"zoom_factors must have {n_dims} elements.")
    else:
        raise ValueError("Either output_size or zoom_factors must be provided.")

    shifts = [0.0] * n_dims  # Initialize shifts per dimension

    # Set degrees based on interpolation method
    if interpolation == "linear":
        interp_degree = 1
        synthe_degree = 1
        analy_degree = 1
    elif interpolation == "quadratic":
        interp_degree = 2
        synthe_degree = 2
        analy_degree = 2
    else:  # Cubic
        interp_degree = 3
        synthe_degree = 3
        analy_degree = 3

    # Adjust degrees based on method
    if method == "interpolation":
        analy_degree = -1  # No analysis degree needed for interpolation
    elif method == "oblique":
        # For oblique projection, the analysis degree may differ
        if interpolation == "linear":
            analy_degree = 0
        elif interpolation == "quadratic":
            analy_degree = 1
        else:  # cubic
            analy_degree = 1

    # Compute output image size based on inversable parameter
    if inversable:
        # Use calculate_final_size to get the correct output sizes per dimension
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
