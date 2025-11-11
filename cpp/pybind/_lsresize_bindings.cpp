// splineops/cpp/pybind/_lsresize_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <numeric>
#include <cstdint>

#include "../lsresize/src/resizend.h"
#include "../lsresize/src/utils.h"

namespace py = pybind11;
using int64 = std::int64_t;

static std::vector<int64> shape_to_vec_i64(const py::array &a) {
    std::vector<int64> s(static_cast<size_t>(a.ndim()));
    for (py::ssize_t i = 0; i < a.ndim(); ++i) {
        s[static_cast<size_t>(i)] = static_cast<int64>(a.shape(i));
    }
    return s;
}

/**
 * @brief Per-axis policy for magnification: use Standard interpolation.
 *
 * For upsampling (zoom > 1) the LS/Oblique projections tend to behave like
 * a deconvolution (due to the analysis stage and the scale a^(n1+1)), which
 * can amplify high-frequency content and introduce ringing/overshoot near
 * sharp edges. For magnification there is no aliasing to suppress; the
 * recommended behavior is to reconstruct with plain spline interpolation.
 *
 * This function enforces that policy by disabling the projection stage
 * on a per-axis basis: if zoom > 1 and the analysis degree is non-negative
 * (i.e., LS/Oblique), set analy_degree = -1 to select Standard interpolation.
 *
 * Shrinking axes (zoom < 1) are left unchanged so they still benefit from
 * LS/Oblique anti-aliasing.
 */
static inline void normalize_params_for_magnification(lsresize::LSParams& p) {
    const double eps = 1e-12;
    // Identity safety: never run a projection at unity zoom
    if (std::abs(p.zoom - 1.0) <= eps) {
        p.analy_degree = -1;
        return;
    }
    // Magnification policy: projection can ring, use Standard
    if (p.zoom > 1.0 + eps && p.analy_degree >= 0) {
        p.analy_degree = -1;
    }
}

py::array_t<double> resize_nd(py::array input,
                              std::vector<double> zoom_factors,
                              int interp_degree,
                              int analy_degree,
                              int synthe_degree,
                              bool inversable)
{
    // Force float64, C-ordered (no extra copy if already f64/C)
    py::array_t<double, py::array::c_style | py::array::forcecast> in_f64(input);
    if (in_f64.ndim() <= 0)
        throw std::runtime_error("resize_nd: input must be at least 1-D");

    const int D = static_cast<int>(in_f64.ndim());
    if (static_cast<int>(zoom_factors.size()) != D)
        throw std::runtime_error("resize_nd: zoom_factors length must match ndim");

    // Shapes (int64 for C++ core)
    std::vector<int64> in_shape  = shape_to_vec_i64(in_f64);
    std::vector<int64> out_shape = in_shape;

    // Compute per-axis output size (same as Python)
    for (int ax = 0; ax < D; ++ax) {
        int workN = 0, outN = 0;
        lsresize::calculate_final_size_1d(
            inversable, static_cast<int>(in_shape[ax]), zoom_factors[ax], workN, outN);
        out_shape[ax] = outN;
    }

    // Allocate final output (written on the last axis pass)
    std::vector<py::ssize_t> out_shape_ssize(out_shape.begin(), out_shape.end());
    py::array_t<double> out(out_shape_ssize);

    // Ping-pong buffers to avoid aliasing the input of a pass with the output allocation
    std::vector<double> bufA(static_cast<size_t>(in_f64.size()));
    std::memcpy(bufA.data(), in_f64.data(), bufA.size() * sizeof(double));

    std::vector<int64> cur_shape = in_shape;
    std::vector<double> bufB;

    for (int ax = 0; ax < D; ++ax) {
        std::vector<int64> next_shape = cur_shape;
        next_shape[ax] = out_shape[ax];

        int64 total_next = std::accumulate(
            next_shape.begin(), next_shape.end(), (int64)1, std::multiplies<int64>());

        bufB.assign(static_cast<size_t>(total_next), 0.0);

        lsresize::LSParams p;
        p.interp_degree = interp_degree;
        p.analy_degree  = analy_degree;   // -1 allowed (Standard when -1)
        p.synthe_degree = synthe_degree;
        p.zoom          = zoom_factors[ax];
        p.shift         = 0.0;
        p.inversable    = inversable;

        normalize_params_for_magnification(p);

        const double eps = 1e-12;
        if (std::abs(p.zoom - 1.0) <= eps && next_shape[ax] == cur_shape[ax]) {
            // Short-circuit true identity along this axis
            bufB = bufA; // pure copy
        } else {
            lsresize::resize_along_axis(bufA.data(), bufB.data(),
                                        cur_shape, next_shape, ax, p);
        }

        bufA.swap(bufB);
        cur_shape.swap(next_shape);
    }

    // Final copy into the Python array
    std::memcpy(out.mutable_data(), bufA.data(), bufA.size() * sizeof(double));
    return out;
}

PYBIND11_MODULE(_lsresize, m) {
    m.doc() = "splineops: fast LS/oblique resize (C++ core)";
    m.def("resize_nd", &resize_nd,
          py::arg("input"),
          py::arg("zoom_factors"),
          py::arg("interp_degree"),
          py::arg("analy_degree"),
          py::arg("synthe_degree"),
          py::arg("inversable"));
}
