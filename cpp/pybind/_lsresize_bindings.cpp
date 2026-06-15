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
#include <stdexcept>

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

static std::vector<int> choose_axis_order(
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape)
{
    const int D = static_cast<int>(in_shape.size());
    std::vector<int> axes(static_cast<size_t>(D));
    std::iota(axes.begin(), axes.end(), 0);

    constexpr double eps = 1e-12;

    auto scale = [&](int ax) -> double {
        const int64 in_len = in_shape[static_cast<size_t>(ax)];
        if (in_len <= 0) {
            return 1.0;
        }
        return static_cast<double>(out_shape[static_cast<size_t>(ax)]) /
               static_cast<double>(in_len);
    };

    std::stable_sort(
        axes.begin(),
        axes.end(),
        [&](int a, int b) {
            const double sa = scale(a);
            const double sb = scale(b);
            const bool shrink_a = sa < 1.0 - eps;
            const bool shrink_b = sb < 1.0 - eps;
            const bool grow_a = sa > 1.0 + eps;
            const bool grow_b = sb > 1.0 + eps;

            if (shrink_a != shrink_b) {
                return shrink_a;
            }

            if (shrink_a && shrink_b) {
                if (std::abs(sa - sb) > eps) {
                    return sa < sb;
                }
                return a > b;
            }

            if (grow_a != grow_b) {
                return !grow_a && grow_b;
            }

            if (grow_a && grow_b) {
                if (std::abs(sa - sb) > eps) {
                    return sa < sb;
                }
                return a < b;
            }

            return a < b;
        });

    return axes;
}

static std::vector<int64> compute_output_shape(
    const std::vector<int64>& in_shape,
    const std::vector<double>& zoom_factors,
    bool inversable)
{
    if (zoom_factors.size() != in_shape.size()) {
        throw std::runtime_error(
            "zoom_factors length must match input_shape length");
    }

    std::vector<int64> out_shape = in_shape;
    for (int ax = 0; ax < static_cast<int>(in_shape.size()); ++ax) {
        int workN = 0;
        int outN = 0;
        lsresize::calculate_final_size_1d(
            inversable,
            static_cast<int>(in_shape[static_cast<size_t>(ax)]),
            zoom_factors[static_cast<size_t>(ax)],
            workN,
            outN);
        out_shape[static_cast<size_t>(ax)] = outN;
    }
    return out_shape;
}

static std::vector<int> choose_active_axes(
    const std::vector<int>& axis_order,
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<double>& zoom_factors,
    int analy_degree)
{
    std::vector<int> active_axes;
    active_axes.reserve(axis_order.size());
    for (int ax : axis_order) {
        const bool identity_axis =
            (analy_degree < 0) &&
            (out_shape[static_cast<size_t>(ax)] ==
             in_shape[static_cast<size_t>(ax)]) &&
            (std::abs(zoom_factors[static_cast<size_t>(ax)] - 1.0) <= 1e-12);
        if (!identity_axis) {
            active_axes.push_back(ax);
        }
    }
    return active_axes;
}

static py::tuple vec_i64_to_tuple(const std::vector<int64>& values)
{
    py::tuple out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = py::int_(values[i]);
    }
    return out;
}

static py::tuple vec_double_to_tuple(const std::vector<double>& values)
{
    py::tuple out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = py::float_(values[i]);
    }
    return out;
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
py::array_t<T> resize_nd_impl_planned(
    py::array input,
    const std::vector<int64>& expected_in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<double>& zoom_factors,
    const std::vector<int>& active_axes,
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
    if (static_cast<int>(expected_in_shape.size()) != D) {
        throw std::runtime_error(
            "resize_nd: input ndim does not match plan ndim");
    }

    for (int ax = 0; ax < D; ++ax) {
        const int64 got = static_cast<int64>(in_arr.shape(ax));
        const int64 expected = expected_in_shape[static_cast<size_t>(ax)];
        if (got != expected) {
            throw std::runtime_error(
                "resize_nd: input shape does not match plan input_shape");
        }
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
    std::vector<int64> cur_shape = expected_in_shape;

    if (active_axes.empty()) {
        const int64 total = std::accumulate(
            expected_in_shape.begin(), expected_in_shape.end(),
            static_cast<int64>(1),
            std::multiplies<int64>());
        std::copy(
            static_cast<const T*>(in_arr.data()),
            static_cast<const T*>(in_arr.data()) + total,
            static_cast<T*>(out.mutable_data()));
        return out;
    }

    const int n_passes = static_cast<int>(active_axes.size());
    for (int pass = 0; pass < n_passes; ++pass) {
        const int ax = active_axes[static_cast<size_t>(pass)];
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

        const bool first_pass = (pass == 0);
        const bool last_pass  = (pass == n_passes - 1);

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

template <typename T>
void resize_nd_impl_preplanned_to(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& in_arr,
    T* final_out,
    const std::vector<int64>& expected_in_shape,
    const std::vector<int>& active_axes,
    const std::vector<std::vector<int64>>& pass_in_shapes,
    const std::vector<std::vector<int64>>& pass_out_shapes,
    const std::vector<int64>& pass_output_elems,
    const std::vector<lsresize::LSParams>& pass_params,
    std::vector<T>& prev,
    std::vector<T>& scratch)
{
    if (active_axes.empty()) {
        const int64 total = std::accumulate(
            expected_in_shape.begin(), expected_in_shape.end(),
            static_cast<int64>(1),
            std::multiplies<int64>());
        std::copy(
            static_cast<const T*>(in_arr.data()),
            static_cast<const T*>(in_arr.data()) + total,
            final_out);
        return;
    }

    const int n_passes = static_cast<int>(active_axes.size());
    for (int pass = 0; pass < n_passes; ++pass) {
        const bool first_pass = (pass == 0);
        const bool last_pass = (pass == n_passes - 1);

        const T* in_ptr = first_pass
            ? static_cast<const T*>(in_arr.data())
            : prev.data();
        T* out_ptr = nullptr;

        if (last_pass) {
            out_ptr = final_out;
        } else {
            scratch.resize(static_cast<size_t>(
                pass_output_elems[static_cast<size_t>(pass)]));
            out_ptr = scratch.data();
        }

        AxisDispatch<T>::apply(
            in_ptr,
            out_ptr,
            pass_in_shapes[static_cast<size_t>(pass)],
            pass_out_shapes[static_cast<size_t>(pass)],
            active_axes[static_cast<size_t>(pass)],
            pass_params[static_cast<size_t>(pass)]);

        if (!last_pass) {
            prev.swap(scratch);
        }
    }
}

template <typename T>
static py::array_t<T, py::array::c_style | py::array::forcecast>
checked_preplanned_input(
    py::array input,
    const std::vector<int64>& expected_in_shape)
{
    py::array_t<T, py::array::c_style | py::array::forcecast> in_arr(input);
    if (in_arr.ndim() <= 0)
    {
        throw std::runtime_error(
            "resize_nd: input must be at least 1-D");
    }

    const int D = static_cast<int>(in_arr.ndim());
    if (static_cast<int>(expected_in_shape.size()) != D) {
        throw std::runtime_error(
            "resize_nd: input ndim does not match plan ndim");
    }

    for (int ax = 0; ax < D; ++ax) {
        const int64 got = static_cast<int64>(in_arr.shape(ax));
        const int64 expected = expected_in_shape[static_cast<size_t>(ax)];
        if (got != expected) {
            throw std::runtime_error(
                "resize_nd: input shape does not match plan input_shape");
        }
    }

    return in_arr;
}

template <typename T>
py::array_t<T> resize_nd_impl_preplanned(
    py::array input,
    const std::vector<int64>& expected_in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<int>& active_axes,
    const std::vector<std::vector<int64>>& pass_in_shapes,
    const std::vector<std::vector<int64>>& pass_out_shapes,
    const std::vector<int64>& pass_output_elems,
    const std::vector<lsresize::LSParams>& pass_params,
    std::vector<T>& prev,
    std::vector<T>& scratch)
{
    auto in_arr = checked_preplanned_input<T>(input, expected_in_shape);
    std::vector<py::ssize_t> out_shape_ssize(
        out_shape.begin(), out_shape.end());
    py::array_t<T> out(out_shape_ssize);

    resize_nd_impl_preplanned_to<T>(
        in_arr,
        static_cast<T*>(out.mutable_data()),
        expected_in_shape,
        active_axes,
        pass_in_shapes,
        pass_out_shapes,
        pass_output_elems,
        pass_params,
        prev,
        scratch);
    return out;
}

template <typename T>
py::array resize_nd_impl_preplanned_into(
    py::array input,
    py::array output,
    const std::vector<int64>& expected_in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<int>& active_axes,
    const std::vector<std::vector<int64>>& pass_in_shapes,
    const std::vector<std::vector<int64>>& pass_out_shapes,
    const std::vector<int64>& pass_output_elems,
    const std::vector<lsresize::LSParams>& pass_params,
    std::vector<T>& prev,
    std::vector<T>& scratch)
{
    auto in_arr = checked_preplanned_input<T>(input, expected_in_shape);
    py::array_t<T, py::array::c_style> out_arr(output);
    if (!out_arr.writeable()) {
        throw std::runtime_error(
            "resize_nd: output must be writeable");
    }
    if (out_arr.ndim() != static_cast<py::ssize_t>(out_shape.size())) {
        throw std::runtime_error(
            "resize_nd: output ndim does not match plan output ndim");
    }
    for (int ax = 0; ax < out_arr.ndim(); ++ax) {
        const int64 got = static_cast<int64>(out_arr.shape(ax));
        const int64 expected = out_shape[static_cast<size_t>(ax)];
        if (got != expected) {
            throw std::runtime_error(
                "resize_nd: output shape does not match plan output_shape");
        }
    }

    resize_nd_impl_preplanned_to<T>(
        in_arr,
        static_cast<T*>(out_arr.mutable_data()),
        expected_in_shape,
        active_axes,
        pass_in_shapes,
        pass_out_shapes,
        pass_output_elems,
        pass_params,
        prev,
        scratch);
    return output;
}

template <typename T>
py::array_t<T> resize_nd_impl(
    py::array input,
    std::vector<double> zoom_factors,
    int interp_degree,
    int analy_degree,
    int synthe_degree,
    bool inversable)
{
    py::array_t<T, py::array::c_style | py::array::forcecast> in_arr(input);
    if (in_arr.ndim() <= 0) {
        throw std::runtime_error(
            "resize_nd: input must be at least 1-D");
    }

    const std::vector<int64> in_shape = shape_to_vec_i64(in_arr);
    const std::vector<int64> out_shape =
        compute_output_shape(in_shape, zoom_factors, inversable);
    const std::vector<int> axis_order =
        choose_axis_order(in_shape, out_shape);
    const std::vector<int> active_axes =
        choose_active_axes(
            axis_order,
            in_shape,
            out_shape,
            zoom_factors,
            analy_degree);

    return resize_nd_impl_planned<T>(
        in_arr,
        in_shape,
        out_shape,
        zoom_factors,
        active_axes,
        interp_degree,
        analy_degree,
        synthe_degree,
        inversable);
}

class ResizePlanNative {
public:
    ResizePlanNative(
        std::vector<int64> input_shape,
        std::vector<double> zoom_factors,
        int interp_degree,
        int analy_degree,
        int synthe_degree,
        bool inversable)
        : input_shape_(std::move(input_shape)),
          zoom_factors_(std::move(zoom_factors)),
          interp_degree_(interp_degree),
          analy_degree_(analy_degree),
          synthe_degree_(synthe_degree),
          inversable_(inversable)
    {
        validate_degrees(interp_degree_, analy_degree_, synthe_degree_);
        if (input_shape_.empty()) {
            throw std::runtime_error(
                "ResizePlan: input_shape must be at least 1-D");
        }
        for (int64 n : input_shape_) {
            if (n <= 0) {
                throw std::runtime_error(
                    "ResizePlan: input_shape entries must be positive");
            }
        }
        output_shape_ =
            compute_output_shape(input_shape_, zoom_factors_, inversable_);
        axis_order_ = choose_axis_order(input_shape_, output_shape_);
        active_axes_ = choose_active_axes(
            axis_order_,
            input_shape_,
            output_shape_,
            zoom_factors_,
            analy_degree_);

        std::vector<int64> cur_shape = input_shape_;
        pass_in_shapes_.reserve(active_axes_.size());
        pass_out_shapes_.reserve(active_axes_.size());
        pass_output_elems_.reserve(active_axes_.size());
        pass_params_.reserve(active_axes_.size());
        for (int ax : active_axes_) {
            std::vector<int64> next_shape = cur_shape;
            next_shape[static_cast<size_t>(ax)] =
                output_shape_[static_cast<size_t>(ax)];
            pass_in_shapes_.push_back(cur_shape);
            pass_out_shapes_.push_back(next_shape);
            pass_output_elems_.push_back(std::accumulate(
                next_shape.begin(), next_shape.end(),
                static_cast<int64>(1),
                std::multiplies<int64>()));

            lsresize::LSParams p;
            p.interp_degree = interp_degree_;
            p.analy_degree = analy_degree_;
            p.synthe_degree = synthe_degree_;
            p.zoom = zoom_factors_[static_cast<size_t>(ax)];
            p.shift = 0.0;
            p.inversable = inversable_;
            pass_params_.push_back(p);

            cur_shape.swap(next_shape);
        }
    }

    py::array apply(py::array input)
    {
        py::dtype dt = input.dtype();
        if (dt.is(py::dtype::of<float>())) {
            return resize_nd_impl_preplanned<float>(
                input,
                input_shape_,
                output_shape_,
                active_axes_,
                pass_in_shapes_,
                pass_out_shapes_,
                pass_output_elems_,
                pass_params_,
                prev_f32_,
                scratch_f32_);
        }
        return resize_nd_impl_preplanned<double>(
            input,
            input_shape_,
            output_shape_,
            active_axes_,
            pass_in_shapes_,
            pass_out_shapes_,
            pass_output_elems_,
            pass_params_,
            prev_f64_,
            scratch_f64_);
    }

    py::array apply_into(py::array input, py::array output)
    {
        py::dtype dt = input.dtype();
        if (dt.is(py::dtype::of<float>())) {
            return resize_nd_impl_preplanned_into<float>(
                input,
                output,
                input_shape_,
                output_shape_,
                active_axes_,
                pass_in_shapes_,
                pass_out_shapes_,
                pass_output_elems_,
                pass_params_,
                prev_f32_,
                scratch_f32_);
        }
        return resize_nd_impl_preplanned_into<double>(
            input,
            output,
            input_shape_,
            output_shape_,
            active_axes_,
            pass_in_shapes_,
            pass_out_shapes_,
            pass_output_elems_,
            pass_params_,
            prev_f64_,
            scratch_f64_);
    }

    py::tuple input_shape() const { return vec_i64_to_tuple(input_shape_); }
    py::tuple output_shape() const { return vec_i64_to_tuple(output_shape_); }
    py::tuple zoom_factors() const { return vec_double_to_tuple(zoom_factors_); }
    int interp_degree() const { return interp_degree_; }
    int analy_degree() const { return analy_degree_; }
    int synthe_degree() const { return synthe_degree_; }
    bool inversable() const { return inversable_; }

private:
    std::vector<int64> input_shape_;
    std::vector<int64> output_shape_;
    std::vector<double> zoom_factors_;
    std::vector<int> axis_order_;
    std::vector<int> active_axes_;
    std::vector<std::vector<int64>> pass_in_shapes_;
    std::vector<std::vector<int64>> pass_out_shapes_;
    std::vector<int64> pass_output_elems_;
    std::vector<lsresize::LSParams> pass_params_;
    std::vector<float> prev_f32_;
    std::vector<float> scratch_f32_;
    std::vector<double> prev_f64_;
    std::vector<double> scratch_f64_;
    int interp_degree_;
    int analy_degree_;
    int synthe_degree_;
    bool inversable_;
};

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

    py::class_<ResizePlanNative>(m, "ResizePlan")
        .def(py::init<
             std::vector<int64>,
             std::vector<double>,
             int,
             int,
             int,
             bool>(),
             py::arg("input_shape"),
             py::arg("zoom_factors"),
             py::arg("interp_degree"),
             py::arg("analy_degree"),
             py::arg("synthe_degree"),
             py::arg("inversable"))
        .def("apply", &ResizePlanNative::apply, py::arg("input"))
        .def("apply_into",
             &ResizePlanNative::apply_into,
             py::arg("input"),
             py::arg("output"))
        .def_property_readonly("input_shape", &ResizePlanNative::input_shape)
        .def_property_readonly("output_shape", &ResizePlanNative::output_shape)
        .def_property_readonly("zoom_factors", &ResizePlanNative::zoom_factors)
        .def_property_readonly("interp_degree", &ResizePlanNative::interp_degree)
        .def_property_readonly("analy_degree", &ResizePlanNative::analy_degree)
        .def_property_readonly("synthe_degree", &ResizePlanNative::synthe_degree)
        .def_property_readonly("inversable", &ResizePlanNative::inversable);
}
