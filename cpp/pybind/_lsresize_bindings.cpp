// splineops/cpp/pybind/_lsresize_bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <vector>
#include <numeric>
#include <cstdint>
#include <algorithm>
#include <cmath>       // std::abs
#include <functional>  // std::multiplies

#include "../lsresize/src/resize_nd.h"
#include "../lsresize/src/utils.h"

namespace py = pybind11;
using int64 = std::int64_t;

// Convert NumPy shape to std::vector<int64>
static std::vector<int64> shape_to_vec_i64(const py::array &a) 
{
    std::vector<int64> s(static_cast<size_t>(a.ndim()));
    for (py::ssize_t i = 0; i < a.ndim(); ++i) {
        s[static_cast<size_t>(i)] =
            static_cast<int64>(a.shape(i));
    }
    return s;
}

// Restrict and sanity-check the degrees coming from Python.
// We treat 3 as the hard ceiling for robustness:
//
//   - 0 <= interp_degree <= 3
//   - -1 <= analy_degree <= 3   (-1 = no projection / pure interpolation)
//   - 0 <= synthe_degree <= 3
//   - if analy_degree >= 0: analy_degree <= interp_degree
//   - synthe_degree <= interp_degree
//
static void validate_degrees(
    int interp_degree, 
    int analy_degree, 
    int synthe_degree)
{
    const int MAX_DEG = 3;

    if (interp_degree < 0 || interp_degree > MAX_DEG) {
        throw std::runtime_error("interp_degree must be in [0, 3]");
    }
    if (analy_degree < -1 || analy_degree > MAX_DEG) {
        throw std::runtime_error("analy_degree must be in [-1, 3]");
    }
    if (synthe_degree < 0 || synthe_degree > MAX_DEG) {
        throw std::runtime_error("synthe_degree must be in [0, 3]");
    }

    if (analy_degree >= 0 && analy_degree > interp_degree) {
        throw std::runtime_error(
            "analy_degree must be <= interp_degree when analy_degree >= 0");
    }
    if (synthe_degree > interp_degree) {
        throw std::runtime_error(
            "synthe_degree must be <= interp_degree");
    }
}

// Axis-level dispatch: choose float32 vs float64 kernel
template <typename T> struct AxisDispatch;

template <>
struct AxisDispatch<double> {
    static inline void apply(
        const double* in, double* out,
        const std::vector<int64>& in_shape,
        const std::vector<int64>& out_shape,
        int axis,
        const lsresize::LSParams& p)
    {
        lsresize::resize_along_axis(in, out, in_shape, out_shape, axis, p);
    }
};

template <>
struct AxisDispatch<float> {
    static inline void apply(
        const float* in, float* out,
        const std::vector<int64>& in_shape,
        const std::vector<int64>& out_shape,
        int axis,
        const lsresize::LSParams& p)
    {
        lsresize::resize_along_axis_f32(in, out, in_shape, out_shape, axis, p);
    }
};

// Templated ND resize over storage scalar T (float or double).
// Internal math remains in double (handled inside lsresize::resize_along_axis_*).
template <typename T>
py::array_t<T> resize_nd_impl(
    py::array input,
    std::vector<double> zoom_factors,
    int interp_degree,
    int analy_degree,
    int synthe_degree,
    bool inversable)
{
    // Force T, C-ordered (no copy if already T/C).
    py::array_t<T, py::array::c_style | py::array::forcecast> in_arr(input);
    if (in_arr.ndim() <= 0) 
    {
        throw std::runtime_error(
            "resize_nd: input must be at least 1-D");
    }

    const int D = static_cast<int>(in_arr.ndim());
    if (static_cast<int>(zoom_factors.size()) != D) {
        throw std::runtime_error(
            "resize_nd: zoom_factors length must match ndim");
    }

    // Shapes (int64 for C++ core)
    std::vector<int64> in_shape  = shape_to_vec_i64(in_arr);
    std::vector<int64> out_shape = in_shape;

    // Compute final per-axis output size
    for (int ax = 0; ax < D; ++ax) {
        int workN = 0;
        int outN  = 0;
        lsresize::calculate_final_size_1d(
            inversable,
            static_cast<int>(in_shape[static_cast<size_t>(ax)]),
            zoom_factors[static_cast<size_t>(ax)],
            workN,
            outN);
        out_shape[static_cast<size_t>(ax)] = outN;
    }

    // Allocate final output (written on the LAST axis pass)
    std::vector<py::ssize_t> out_shape_ssize(
        out_shape.begin(), out_shape.end());
    py::array_t<T> out(out_shape_ssize);

    // Ping-pong plan:
    //  - First pass reads directly from in_arr.data()
    //  - Middle passes use a single transient vector<T> `prev` (reused)
    //  - Last pass writes directly into out.mutable_data()
    std::vector<T> prev;     // holds current intermediate result
    std::vector<T> scratch;  // temporary buffer for next pass
    std::vector<int64> cur_shape = in_shape;

    for (int ax = 0; ax < D; ++ax) {
        std::vector<int64> next_shape = cur_shape;
        next_shape[static_cast<size_t>(ax)] =
            out_shape[static_cast<size_t>(ax)];

        int64 total_next = std::accumulate(
            next_shape.begin(), next_shape.end(),
            static_cast<int64>(1),
            std::multiplies<int64>());

        lsresize::LSParams p;
        p.interp_degree = interp_degree;
        p.analy_degree  = analy_degree;
        p.synthe_degree = synthe_degree;
        p.zoom          = zoom_factors[static_cast<size_t>(ax)];
        p.shift         = 0.0;
        p.inversable    = inversable;

        const bool first_pass = (ax == 0);
        const bool last_pass  = (ax == D - 1);

        const T* in_ptr  = nullptr;
        T*       out_ptr = nullptr;

        if (first_pass) {
            in_ptr = static_cast<const T*>(in_arr.data());
        } else {
            in_ptr = prev.data();
        }

        if (last_pass) {
            out_ptr = static_cast<T*>(out.mutable_data());
        } else {
            scratch.resize(static_cast<size_t>(total_next));
            out_ptr = scratch.data();
        }

        AxisDispatch<T>::apply(
            in_ptr,
            out_ptr,
            cur_shape,
            next_shape,
            ax,
            p);

        if (!last_pass) {
            // Now 'scratch' holds the latest result; keep it in 'prev'
            prev.swap(scratch);
        }

        cur_shape.swap(next_shape);
    }

    return out;
}

// Python-visible dispatcher: chooses float32 vs float64 pipeline
static py::array resize_nd(
    py::array input,
    std::vector<double> zoom_factors,
    int interp_degree,
    int analy_degree,
    int synthe_degree,
    bool inversable)
{
    // Safety net: ensure degrees/combos are within the supported regime.
    validate_degrees(interp_degree, analy_degree, synthe_degree);

    py::dtype dt = input.dtype();

    // Keep float32 as float32 storage, double-internal.
    if (dt.is(py::dtype::of<float>())) {
        return resize_nd_impl<float>(
            input,
            std::move(zoom_factors),
            interp_degree,
            analy_degree,
            synthe_degree,
            inversable
        );
    }

    // Default: float64 storage (and internal).
    return resize_nd_impl<double>(
        input,
        std::move(zoom_factors),
        interp_degree,
        analy_degree,
        synthe_degree,
        inversable
    );
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
