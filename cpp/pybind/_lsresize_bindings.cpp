// splineops/cpp/pybind/_lsresize_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <numeric>
#include <cstdint>
#include <cstring>

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

py::array_t<double> resize_nd(py::array input,
                              std::vector<double> zoom_factors,
                              int interp_degree,
                              int analy_degree,
                              int synthe_degree,
                              bool inversable)
{
    // Force float64, C-ordered
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

    // Construct output array with py::ssize_t shape
    std::vector<py::ssize_t> out_shape_ssize(out_shape.begin(), out_shape.end());
    py::array_t<double> out(out_shape_ssize);

    // Work buffers
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
        p.analy_degree  = analy_degree;   // -1 allowed
        p.synthe_degree = synthe_degree;
        p.zoom          = zoom_factors[ax];
        p.shift         = 0.0;
        p.inversable    = inversable;

        lsresize::resize_along_axis(bufA.data(), bufB.data(),
                                    cur_shape, next_shape, ax, p);
        bufA.swap(bufB);
        cur_shape.swap(next_shape);
    }

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
