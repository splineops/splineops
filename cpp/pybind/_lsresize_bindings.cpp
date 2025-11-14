// splineops/cpp/pybind/_lsresize_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <numeric>
#include <cstdint>
#include <algorithm>   // std::accumulate

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

// If the per-axis zoom is effectively 1.0, disable projection and fall back
// to standard interpolation (analy_degree = -1). For all other zoom values
// the parameters are left unchanged.
static inline void normalize_params_for_magnification(lsresize::LSParams& p) {
    const double eps = 1e-12;
    // Identity safety: never run a projection at unity zoom
    if (std::abs(p.zoom - 1.0) <= eps) {
        p.analy_degree = -1;
        return;
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

    // Compute final per-axis output size (same policy as Python)
    for (int ax = 0; ax < D; ++ax) {
        int workN = 0, outN = 0;
        lsresize::calculate_final_size_1d(
            inversable, static_cast<int>(in_shape[static_cast<size_t>(ax)]),
            zoom_factors[static_cast<size_t>(ax)], workN, outN);
        out_shape[static_cast<size_t>(ax)] = outN;
    }

    // Allocate final output (will be written on the LAST axis pass)
    std::vector<py::ssize_t> out_shape_ssize(out_shape.begin(), out_shape.end());
    py::array_t<double> out(out_shape_ssize);

    // Ping-pong plan:
    //  - First pass reads directly from in_f64.data()  (no initial memcpy)
    //  - Middle passes use a single transient vector `prev` (reused)
    //  - Last pass writes directly into `out.mutable_data()` (no final memcpy)
    std::vector<double> prev;     // holds current intermediate result
    std::vector<double> scratch;  // temporary buffer for next pass
    std::vector<int64>  cur_shape = in_shape;

    for (int ax = 0; ax < D; ++ax) {
        std::vector<int64> next_shape = cur_shape;
        next_shape[static_cast<size_t>(ax)] = out_shape[static_cast<size_t>(ax)];

        int64 total_next = std::accumulate(
            next_shape.begin(), next_shape.end(),
            static_cast<int64>(1), std::multiplies<int64>()
        );

        lsresize::LSParams p;
        p.interp_degree = interp_degree;
        p.analy_degree  = analy_degree;
        p.synthe_degree = synthe_degree;
        p.zoom          = zoom_factors[static_cast<size_t>(ax)];
        p.shift         = 0.0;
        p.inversable    = inversable;
        normalize_params_for_magnification(p);

        const bool first_pass = (ax == 0);
        const bool last_pass  = (ax == D - 1);

        const double* in_ptr  = nullptr;
        double*       out_ptr = nullptr;

        if (first_pass) {
            in_ptr = static_cast<const double*>(in_f64.data());
        } else {
            in_ptr = prev.data();
        }

        if (last_pass) {
            out_ptr = static_cast<double*>(out.mutable_data());
        } else {
            scratch.resize(static_cast<size_t>(total_next));
            out_ptr = scratch.data();
        }

        lsresize::resize_along_axis(in_ptr, out_ptr, cur_shape, next_shape, ax, p);

        if (!last_pass) {
            // Now 'scratch' holds the latest result; keep it in 'prev'
            prev.swap(scratch);
        }

        cur_shape.swap(next_shape);
    }

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
