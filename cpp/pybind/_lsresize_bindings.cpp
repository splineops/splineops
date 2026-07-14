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
#include <cerrno>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

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

static std::vector<int> normalize_selected_axes(
    const py::object& axes,
    int ndim)
{
    if (axes.is_none()) {
        std::vector<int> all(static_cast<size_t>(ndim));
        std::iota(all.begin(), all.end(), 0);
        return all;
    }
    if (!PySequence_Check(axes.ptr())) {
        throw py::type_error("axes must be None or a sequence of integers");
    }

    // Materialize a tuple before iterating.  In particular, treating a NumPy
    // array as a borrowed ``py::sequence`` leaves its scalar accessors tied to
    // temporary proxy objects on some pybind11/NumPy combinations.
    const py::tuple sequence = py::tuple(axes);
    std::vector<int> normalized;
    normalized.reserve(static_cast<size_t>(py::len(sequence)));
    std::vector<char> seen(static_cast<size_t>(ndim), 0);
    for (const py::handle item : sequence) {
        const bool numpy_bool =
            py::hasattr(item, "dtype") &&
            py::str(item.attr("dtype")).equal(py::str("bool"));
        if (PyBool_Check(item.ptr()) || numpy_bool) {
            throw py::type_error("axes entries must be integers");
        }
        PyObject* index_ptr = PyNumber_Index(item.ptr());
        if (index_ptr == nullptr) {
            PyErr_Clear();
            throw py::type_error("axes entries must be integers");
        }
        const py::object index = py::reinterpret_steal<py::object>(index_ptr);
        long long axis_value = PyLong_AsLongLong(index.ptr());
        if (PyErr_Occurred()) {
            throw py::error_already_set();
        }
        if (axis_value < 0) {
            axis_value += ndim;
        }
        if (axis_value < 0 || axis_value >= ndim) {
            throw py::value_error(
                "axes entry is out of range for the input rank");
        }
        const int axis = static_cast<int>(axis_value);
        if (seen[static_cast<size_t>(axis)] != 0) {
            throw py::value_error("axes entries must be unique");
        }
        seen[static_cast<size_t>(axis)] = 1;
        normalized.push_back(axis);
    }
    return normalized;
}

static std::vector<int64> compute_output_shape(
    const std::vector<int64>& in_shape,
    const std::vector<double>& zoom_factors,
    const std::vector<int>& selected_axes)
{
    if (in_shape.empty()) {
        throw std::runtime_error(
            "input_shape must describe at least one dimension");
    }
    if (zoom_factors.size() != in_shape.size()) {
        throw std::runtime_error(
            "zoom_factors length must match input_shape length");
    }

    const int ndim = static_cast<int>(in_shape.size());
    std::vector<char> selected(static_cast<size_t>(ndim), 0);
    for (int axis : selected_axes) {
        if (axis < 0 || axis >= ndim || selected[static_cast<size_t>(axis)] != 0) {
            throw std::runtime_error("selected axes must be unique and in range");
        }
        selected[static_cast<size_t>(axis)] = 1;
    }

    std::vector<int64> out_shape = in_shape;
    int64 in_elements = 1;
    int64 out_elements = 1;
    for (int ax = 0; ax < static_cast<int>(in_shape.size()); ++ax) {
        const int64 in_len = in_shape[static_cast<size_t>(ax)];
        const double zoom = zoom_factors[static_cast<size_t>(ax)];
        if (in_len <= 0 ||
            in_len > static_cast<int64>(std::numeric_limits<int>::max())) {
            throw std::runtime_error(
                "input_shape entries must be positive native-axis lengths");
        }
        if (!std::isfinite(zoom) || zoom <= 0.0) {
            throw std::runtime_error(
                "zoom_factors entries must be finite and positive");
        }
        if (in_elements > std::numeric_limits<int64>::max() / in_len) {
            throw std::overflow_error("input array size exceeds native limits");
        }
        in_elements *= in_len;

        const int outN = selected[static_cast<size_t>(ax)] != 0
            ? lsresize::calculate_output_size_1d(
                  static_cast<int>(in_len), zoom)
            : static_cast<int>(in_len);
        out_shape[static_cast<size_t>(ax)] = outN;
        if (out_elements > std::numeric_limits<int64>::max() / outN) {
            throw std::overflow_error("output array size exceeds native limits");
        }
        out_elements *= outN;
    }
    return out_shape;
}

static void canonicalize_unselected_zoom_factors(
    std::vector<double>& zoom_factors,
    const std::vector<int>& selected_axes)
{
    std::vector<char> selected(zoom_factors.size(), 0);
    for (int axis : selected_axes) {
        selected[static_cast<size_t>(axis)] = 1;
    }
    for (size_t axis = 0; axis < zoom_factors.size(); ++axis) {
        if (selected[axis] == 0) {
            zoom_factors[axis] = 1.0;
        }
    }
}

static std::vector<int> choose_active_axes(
    const std::vector<int>& axis_order,
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<int>& selected_axes,
    int interp_degree,
    int synthe_degree)
{
    std::vector<char> selected(in_shape.size(), 0);
    for (int axis : selected_axes) {
        selected[static_cast<size_t>(axis)] = 1;
    }
    std::vector<int> active_axes;
    active_axes.reserve(axis_order.size());
    for (int ax : axis_order) {
        if (selected[static_cast<size_t>(ax)] == 0) {
            continue;
        }
        const bool identity_axis =
            (out_shape[static_cast<size_t>(ax)] ==
             in_shape[static_cast<size_t>(ax)]) &&
            (synthe_degree == interp_degree);
        if (!identity_axis) {
            active_axes.push_back(ax);
        }
    }
    return active_axes;
}

static bool has_degenerate_active_axis(
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<int>& active_axes)
{
    for (int ax : active_axes) {
        if (in_shape[static_cast<size_t>(ax)] == 1 ||
            out_shape[static_cast<size_t>(ax)] == 1) {
            return true;
        }
    }
    return false;
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

static py::tuple vec_int_to_tuple(const std::vector<int>& values)
{
    py::tuple out(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        out[i] = py::int_(values[i]);
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

static inline char ascii_lower(char c)
{
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

static inline bool env_equals_ci(const char* value, const char* token)
{
    if (value == nullptr) {
        return false;
    }
    size_t i = 0;
    for (; token[i] != '\0'; ++i) {
        if (ascii_lower(value[i]) != token[i]) {
            return false;
        }
    }
    return value[i] == '\0';
}

static inline bool env_flag_enabled_default_true(const char* name)
{
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    if (value[0] == '0' ||
        env_equals_ci(value, "off") ||
        env_equals_ci(value, "false") ||
        env_equals_ci(value, "no")) {
        return false;
    }
    return true;
}

static inline bool linear_interp_enabled()
{
    const char* value = std::getenv("LSRESIZE_LINEAR_INTERP");
    if (value != nullptr && value[0] != '\0') {
        return env_flag_enabled_default_true("LSRESIZE_LINEAR_INTERP");
    }
    value = std::getenv("LSRESIZE_2D_LINEAR_INTERP");
    if (value != nullptr && value[0] != '\0') {
        return env_flag_enabled_default_true("LSRESIZE_2D_LINEAR_INTERP");
    }
    return env_flag_enabled_default_true("LSRESIZE_2D_FLOAT_INTERP");
}

static inline bool fused_2d_linear_enabled()
{
    return env_flag_enabled_default_true("LSRESIZE_FUSED_2D_LINEAR");
}

static inline bool fused_3d_linear_enabled()
{
    return env_flag_enabled_default_true("LSRESIZE_FUSED_3D_LINEAR");
}

static inline bool fused_3d_two_axis_linear_enabled()
{
    return env_flag_enabled_default_true("LSRESIZE_FUSED_3D_TWO_AXIS_LINEAR");
}

static inline bool batched_axis_not_off()
{
    const char* value = std::getenv("LSRESIZE_BATCHED_AXIS");
    if (value == nullptr || value[0] == '\0') {
        return true;
    }
    return !(value[0] == '0' ||
             env_equals_ci(value, "off") ||
             env_equals_ci(value, "false") ||
             env_equals_ci(value, "no"));
}

static inline bool float32_internal_enabled()
{
    const char* value = std::getenv("LSRESIZE_PRECISION");
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    return env_equals_ci(value, "float32") ||
           env_equals_ci(value, "single") ||
           env_equals_ci(value, "f32");
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

    static inline void apply_preplanned(
        const double* in, double* out,
        const std::vector<int64>& in_shape,
        const std::vector<int64>& out_shape,
        int axis,
        const lsresize::LSParams& p,
        const lsresize::Plan1D& plan)
    {
        lsresize::resize_along_axis_preplanned(
            in, out, in_shape, out_shape, axis, p, plan);
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

    static inline void apply_preplanned(
        const float* in, float* out,
        const std::vector<int64>& in_shape,
        const std::vector<int64>& out_shape,
        int axis,
        const lsresize::LSParams& p,
        const lsresize::Plan1D& plan)
    {
        lsresize::resize_along_axis_preplanned_f32(
            in, out, in_shape, out_shape, axis, p, plan);
    }
};

template <typename T>
static inline bool can_use_fused_2d_linear(
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<int>& active_axes,
    int interp_degree,
    int analy_degree,
    int synthe_degree)
{
    if (!linear_interp_enabled() ||
        !fused_2d_linear_enabled() ||
        !batched_axis_not_off() ||
        has_degenerate_active_axis(in_shape, out_shape, active_axes)) {
        return false;
    }
    if constexpr (std::is_same_v<T, float>) {
        if (float32_internal_enabled()) {
            return false;
        }
    }
    return in_shape.size() == 2 &&
           out_shape.size() == 2 &&
           active_axes.size() == 2 &&
           interp_degree == 1 &&
           analy_degree < 0 &&
           synthe_degree == interp_degree;
}

template <typename T>
static inline bool can_use_fused_3d_linear(
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<int>& active_axes,
    int interp_degree,
    int analy_degree,
    int synthe_degree)
{
    if (!linear_interp_enabled() ||
        !fused_3d_linear_enabled() ||
        !batched_axis_not_off() ||
        has_degenerate_active_axis(in_shape, out_shape, active_axes)) {
        return false;
    }
    if constexpr (std::is_same_v<T, float>) {
        if (float32_internal_enabled()) {
            return false;
        }
    }
    bool active0 = false;
    bool active1 = false;
    bool active2 = false;
    for (int ax : active_axes) {
        active0 = active0 || (ax == 0);
        active1 = active1 || (ax == 1);
        active2 = active2 || (ax == 2);
    }
    const bool supported_active_axes =
        (active0 && active1 && active2) ||
        (active_axes.size() == 2 && active0 && active1);

    return in_shape.size() == 3 &&
           out_shape.size() == 3 &&
           supported_active_axes &&
           interp_degree == 1 &&
           analy_degree < 0 &&
           synthe_degree == interp_degree;
}

template <typename T>
static inline int fused_3d_linear_two_axis_kind(
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<int>& active_axes,
    int interp_degree,
    int analy_degree,
    int synthe_degree)
{
    if (!linear_interp_enabled() ||
        !fused_3d_linear_enabled() ||
        !fused_3d_two_axis_linear_enabled() ||
        !batched_axis_not_off() ||
        has_degenerate_active_axis(in_shape, out_shape, active_axes)) {
        return 0;
    }
    if constexpr (std::is_same_v<T, float>) {
        if (float32_internal_enabled()) {
            return 0;
        }
    }
    if (in_shape.size() != 3 ||
        out_shape.size() != 3 ||
        active_axes.size() != 2 ||
        interp_degree != 1 ||
        analy_degree >= 0 ||
        synthe_degree != interp_degree) {
        return 0;
    }

    bool active0 = false;
    bool active1 = false;
    bool active2 = false;
    for (int ax : active_axes) {
        active0 = active0 || (ax == 0);
        active1 = active1 || (ax == 1);
        active2 = active2 || (ax == 2);
    }

    if (active0 && active2 && !active1) {
        return 2;   // axes (0, 2)
    }
    if (active1 && active2 && !active0) {
        return 12;  // axes (1, 2)
    }
    return 0;
}

template <typename T>
static py::array_t<T, py::array::c_style | py::array::forcecast>
checked_preplanned_input(
    py::array input,
    const std::vector<int64>& expected_in_shape);

template <typename T>
static py::array_t<T, py::array::c_style> checked_preplanned_output(
    py::array output,
    const std::vector<int64>& expected_out_shape);

template <typename T, int InputFlags, int OutputFlags>
static void reject_overlapping_arrays(
    const py::array_t<T, InputFlags>& input,
    const py::array_t<T, OutputFlags>& output);

template <typename T>
using NativeInputArray =
    py::array_t<T, py::array::c_style | py::array::forcecast>;

template <typename T>
static NativeInputArray<T> aligned_native_input(py::array input)
{
    NativeInputArray<T> in_arr(input);
    if ((reinterpret_cast<std::uintptr_t>(in_arr.data()) % alignof(T)) == 0) {
        return in_arr;
    }

    std::vector<py::ssize_t> shape(static_cast<size_t>(in_arr.ndim()));
    for (py::ssize_t axis = 0; axis < in_arr.ndim(); ++axis) {
        shape[static_cast<size_t>(axis)] = in_arr.shape(axis);
    }
    NativeInputArray<T> aligned(shape);
    if (in_arr.nbytes() > 0) {
        std::memcpy(
            aligned.mutable_data(),
            in_arr.data(),
            static_cast<size_t>(in_arr.nbytes()));
    }
    return aligned;
}

struct PreparedLinearAxis {
    lsresize::LSParams params;
    std::shared_ptr<const lsresize::Plan1D> plan;
};

static PreparedLinearAxis prepare_linear_axis(
    int64 input_length,
    double zoom)
{
    // Keep cache lookup/build on the GIL-held side of every binding.  Besides
    // making the released region compute-only, this prevents a Python fork
    // from inheriting the process-wide plan-cache mutex while it is locked.
    lsresize::LSParams params;
    params.interp_degree = 1;
    params.analy_degree = -1;
    params.synthe_degree = 1;
    params.zoom = zoom;
    params.shift = 0.0;
    return PreparedLinearAxis{
        params,
        lsresize::get_plan_1d_cached(
            static_cast<int>(input_length), params)};
}

template <typename T>
void resize_nd_impl_fused_2d_linear_to(
    const T* input_data,
    T* out_data,
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const PreparedLinearAxis& axis0,
    const PreparedLinearAxis& axis1)
{
    if constexpr (std::is_same_v<T, float>) {
        lsresize::resize_2d_linear_preplanned_f32(
            static_cast<const float*>(input_data),
            static_cast<float*>(out_data),
            in_shape,
            out_shape,
            axis0.params,
            axis1.params,
            *axis0.plan,
            *axis1.plan);
    } else {
        lsresize::resize_2d_linear_preplanned(
            static_cast<const double*>(input_data),
            static_cast<double*>(out_data),
            in_shape,
            out_shape,
            axis0.params,
            axis1.params,
            *axis0.plan,
            *axis1.plan);
    }
}

template <typename T>
py::array_t<T> resize_nd_impl_fused_2d_linear(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& in_arr,
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<double>& zoom_factors)
{
    const auto axis0 = prepare_linear_axis(
        in_shape[0], zoom_factors[0]);
    const auto axis1 = prepare_linear_axis(
        in_shape[1], zoom_factors[1]);
    std::vector<py::ssize_t> out_shape_ssize(
        out_shape.begin(), out_shape.end());
    py::array_t<T> out(out_shape_ssize);
    const T* input_data = static_cast<const T*>(in_arr.data());
    T* output_data = static_cast<T*>(out.mutable_data());

    {
        py::gil_scoped_release release;
        resize_nd_impl_fused_2d_linear_to<T>(
            input_data,
            output_data,
            in_shape,
            out_shape,
            axis0,
            axis1);
    }

    return out;
}

template <typename T>
py::array resize_nd_impl_fused_2d_linear_into(
    py::array input,
    py::array output,
    const std::vector<int64>& expected_in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<double>& zoom_factors)
{
    auto in_arr = checked_preplanned_input<T>(input, expected_in_shape);
    auto out_arr = checked_preplanned_output<T>(output, out_shape);
    reject_overlapping_arrays(in_arr, out_arr);
    const auto axis0 = prepare_linear_axis(
        expected_in_shape[0], zoom_factors[0]);
    const auto axis1 = prepare_linear_axis(
        expected_in_shape[1], zoom_factors[1]);
    const T* input_data = static_cast<const T*>(in_arr.data());
    T* output_data = static_cast<T*>(out_arr.mutable_data());

    {
        py::gil_scoped_release release;
        resize_nd_impl_fused_2d_linear_to<T>(
            input_data,
            output_data,
            expected_in_shape,
            out_shape,
            axis0,
            axis1);
    }
    return output;
}

template <typename T>
void resize_nd_impl_fused_3d_linear_to(
    const T* input_data,
    T* out_data,
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const PreparedLinearAxis& axis0,
    const PreparedLinearAxis& axis1,
    const PreparedLinearAxis& axis2)
{
    if constexpr (std::is_same_v<T, float>) {
        lsresize::resize_3d_linear_preplanned_f32(
            static_cast<const float*>(input_data),
            static_cast<float*>(out_data),
            in_shape,
            out_shape,
            axis0.params,
            axis1.params,
            axis2.params,
            *axis0.plan,
            *axis1.plan,
            *axis2.plan);
    } else {
        lsresize::resize_3d_linear_preplanned(
            static_cast<const double*>(input_data),
            static_cast<double*>(out_data),
            in_shape,
            out_shape,
            axis0.params,
            axis1.params,
            axis2.params,
            *axis0.plan,
            *axis1.plan,
            *axis2.plan);
    }
}

template <typename T>
py::array_t<T> resize_nd_impl_fused_3d_linear(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& in_arr,
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<double>& zoom_factors)
{
    const auto axis0 = prepare_linear_axis(
        in_shape[0], zoom_factors[0]);
    const auto axis1 = prepare_linear_axis(
        in_shape[1], zoom_factors[1]);
    const auto axis2 = prepare_linear_axis(
        in_shape[2], zoom_factors[2]);
    std::vector<py::ssize_t> out_shape_ssize(
        out_shape.begin(), out_shape.end());
    py::array_t<T> out(out_shape_ssize);
    const T* input_data = static_cast<const T*>(in_arr.data());
    T* output_data = static_cast<T*>(out.mutable_data());

    {
        py::gil_scoped_release release;
        resize_nd_impl_fused_3d_linear_to<T>(
            input_data,
            output_data,
            in_shape,
            out_shape,
            axis0,
            axis1,
            axis2);
    }

    return out;
}

template <typename T>
void resize_nd_impl_fused_3d_linear_two_axis_to(
    const T* input_data,
    T* out_data,
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const PreparedLinearAxis& first_axis,
    const PreparedLinearAxis& second_axis,
    int axis_kind)
{
    if (axis_kind == 2) {
        if constexpr (std::is_same_v<T, float>) {
            lsresize::resize_3d_linear_axis02_preplanned_f32(
                static_cast<const float*>(input_data),
                static_cast<float*>(out_data),
                in_shape,
                out_shape,
                first_axis.params,
                second_axis.params,
                *first_axis.plan,
                *second_axis.plan);
        } else {
            lsresize::resize_3d_linear_axis02_preplanned(
                static_cast<const double*>(input_data),
                static_cast<double*>(out_data),
                in_shape,
                out_shape,
                first_axis.params,
                second_axis.params,
                *first_axis.plan,
                *second_axis.plan);
        }
        return;
    }

    if (axis_kind == 12) {
        if constexpr (std::is_same_v<T, float>) {
            lsresize::resize_3d_linear_axis12_preplanned_f32(
                static_cast<const float*>(input_data),
                static_cast<float*>(out_data),
                in_shape,
                out_shape,
                first_axis.params,
                second_axis.params,
                *first_axis.plan,
                *second_axis.plan);
        } else {
            lsresize::resize_3d_linear_axis12_preplanned(
                static_cast<const double*>(input_data),
                static_cast<double*>(out_data),
                in_shape,
                out_shape,
                first_axis.params,
                second_axis.params,
                *first_axis.plan,
                *second_axis.plan);
        }
        return;
    }

    throw std::runtime_error("resize_nd: unsupported fused 3-D two-axis kind");
}

template <typename T>
py::array_t<T> resize_nd_impl_fused_3d_linear_two_axis(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& in_arr,
    const std::vector<int64>& in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<double>& zoom_factors,
    int axis_kind)
{
    const int first_axis_index = (axis_kind == 2) ? 0 : 1;
    const auto first_axis = prepare_linear_axis(
        in_shape[static_cast<size_t>(first_axis_index)],
        zoom_factors[static_cast<size_t>(first_axis_index)]);
    const auto second_axis = prepare_linear_axis(
        in_shape[2], zoom_factors[2]);
    std::vector<py::ssize_t> out_shape_ssize(
        out_shape.begin(), out_shape.end());
    py::array_t<T> out(out_shape_ssize);
    const T* input_data = static_cast<const T*>(in_arr.data());
    T* output_data = static_cast<T*>(out.mutable_data());

    {
        py::gil_scoped_release release;
        resize_nd_impl_fused_3d_linear_two_axis_to<T>(
            input_data,
            output_data,
            in_shape,
            out_shape,
            first_axis,
            second_axis,
            axis_kind);
    }

    return out;
}

template <typename T>
py::array resize_nd_impl_fused_3d_linear_into(
    py::array input,
    py::array output,
    const std::vector<int64>& expected_in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<double>& zoom_factors)
{
    auto in_arr = checked_preplanned_input<T>(input, expected_in_shape);
    auto out_arr = checked_preplanned_output<T>(output, out_shape);
    reject_overlapping_arrays(in_arr, out_arr);
    const auto axis0 = prepare_linear_axis(
        expected_in_shape[0], zoom_factors[0]);
    const auto axis1 = prepare_linear_axis(
        expected_in_shape[1], zoom_factors[1]);
    const auto axis2 = prepare_linear_axis(
        expected_in_shape[2], zoom_factors[2]);
    const T* input_data = static_cast<const T*>(in_arr.data());
    T* output_data = static_cast<T*>(out_arr.mutable_data());

    {
        py::gil_scoped_release release;
        resize_nd_impl_fused_3d_linear_to<T>(
            input_data,
            output_data,
            expected_in_shape,
            out_shape,
            axis0,
            axis1,
            axis2);
    }
    return output;
}

template <typename T>
py::array resize_nd_impl_fused_3d_linear_two_axis_into(
    py::array input,
    py::array output,
    const std::vector<int64>& expected_in_shape,
    const std::vector<int64>& out_shape,
    const std::vector<double>& zoom_factors,
    int axis_kind)
{
    auto in_arr = checked_preplanned_input<T>(input, expected_in_shape);
    auto out_arr = checked_preplanned_output<T>(output, out_shape);
    reject_overlapping_arrays(in_arr, out_arr);
    const int first_axis_index = (axis_kind == 2) ? 0 : 1;
    const auto first_axis = prepare_linear_axis(
        expected_in_shape[static_cast<size_t>(first_axis_index)],
        zoom_factors[static_cast<size_t>(first_axis_index)]);
    const auto second_axis = prepare_linear_axis(
        expected_in_shape[2], zoom_factors[2]);
    const T* input_data = static_cast<const T*>(in_arr.data());
    T* output_data = static_cast<T*>(out_arr.mutable_data());

    {
        py::gil_scoped_release release;
        resize_nd_impl_fused_3d_linear_two_axis_to<T>(
            input_data,
            output_data,
            expected_in_shape,
            out_shape,
            first_axis,
            second_axis,
            axis_kind);
    }
    return output;
}

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
    int synthe_degree)
{
    // Force T, C-ordered (no copy if already T/C).
    auto in_arr = aligned_native_input<T>(input);
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
    const T* input_data = static_cast<const T*>(in_arr.data());
    T* output_data = static_cast<T*>(out.mutable_data());

    if (active_axes.empty()) {
        const int64 total = std::accumulate(
            expected_in_shape.begin(), expected_in_shape.end(),
            static_cast<int64>(1),
            std::multiplies<int64>());
        {
            py::gil_scoped_release release;
            std::copy(input_data, input_data + total, output_data);
        }
        return out;
    }

    const int n_passes = static_cast<int>(active_axes.size());
    // Resolve every immutable Plan1D before releasing the GIL.  The execution
    // loop below must not enter the process-wide plan cache.
    std::vector<lsresize::LSParams> pass_params;
    std::vector<std::shared_ptr<const lsresize::Plan1D>> pass_plans;
    pass_params.reserve(active_axes.size());
    pass_plans.reserve(active_axes.size());
    for (int ax : active_axes) {
        std::vector<int64> next_shape = cur_shape;
        next_shape[static_cast<size_t>(ax)] =
            out_shape[static_cast<size_t>(ax)];

        lsresize::LSParams params;
        params.interp_degree = interp_degree;
        params.analy_degree = analy_degree;
        params.synthe_degree = synthe_degree;
        params.zoom = zoom_factors[static_cast<size_t>(ax)];
        params.shift = 0.0;
        pass_params.push_back(params);

        const int64 input_length = cur_shape[static_cast<size_t>(ax)];
        const int64 output_length = next_shape[static_cast<size_t>(ax)];
        if (input_length == 1 || output_length == 1) {
            // Degenerate endpoint grids return before the axis kernel reaches
            // its cache lookup and therefore need no Plan1D.
            pass_plans.push_back(nullptr);
        } else {
            pass_plans.push_back(lsresize::get_plan_1d_cached(
                static_cast<int>(input_length), params));
        }
        cur_shape.swap(next_shape);
    }
    cur_shape = expected_in_shape;

    {
      py::gil_scoped_release release;
      for (int pass = 0; pass < n_passes; ++pass) {
        const int ax = active_axes[static_cast<size_t>(pass)];
        std::vector<int64> next_shape = cur_shape;
        next_shape[static_cast<size_t>(ax)] =
            out_shape[static_cast<size_t>(ax)];

        int64 total_next = std::accumulate(
            next_shape.begin(), next_shape.end(),
            static_cast<int64>(1),
            std::multiplies<int64>());

        const bool first_pass = (pass == 0);
        const bool last_pass  = (pass == n_passes - 1);

        const T* in_ptr  = nullptr;
        T*       out_ptr = nullptr;

        if (first_pass) {
            in_ptr = input_data;
        } else {
            in_ptr = prev.data();
        }

        if (last_pass) {
            out_ptr = output_data;
        } else {
            scratch.resize(static_cast<size_t>(total_next));
            out_ptr = scratch.data();
        }

        const auto& plan = pass_plans[static_cast<size_t>(pass)];
        const auto& params = pass_params[static_cast<size_t>(pass)];
        if (plan) {
            AxisDispatch<T>::apply_preplanned(
                in_ptr,
                out_ptr,
                cur_shape,
                next_shape,
                ax,
                params,
                *plan);
        } else {
            AxisDispatch<T>::apply(
                in_ptr,
                out_ptr,
                cur_shape,
                next_shape,
                ax,
                params);
        }

        if (!last_pass) {
            // Now 'scratch' holds the latest result; keep it in 'prev'
            prev.swap(scratch);
        }

        cur_shape.swap(next_shape);
      }
    }

    return out;
}

template <typename T>
struct ResizeWorkspace {
    std::vector<T> prev;
    std::vector<T> scratch;
};

static constexpr size_t default_workspace_cache_bytes =
    static_cast<size_t>(128) * 1024 * 1024;
static constexpr size_t max_retained_plan_workspaces = 4;

static size_t workspace_cache_limit_bytes() noexcept
{
    const char* raw = std::getenv("LSRESIZE_WORKSPACE_CACHE_BYTES");
    if (raw == nullptr) {
        return default_workspace_cache_bytes;
    }

    while (*raw != '\0' &&
           std::isspace(static_cast<unsigned char>(*raw)) != 0) {
        ++raw;
    }
    if (*raw == '\0' || *raw == '-' || *raw == '+') {
        return default_workspace_cache_bytes;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (errno == ERANGE || end == raw) {
        return default_workspace_cache_bytes;
    }
    while (*end != '\0' &&
           std::isspace(static_cast<unsigned char>(*end)) != 0) {
        ++end;
    }
    if (*end != '\0' ||
        parsed > static_cast<unsigned long long>(
            std::numeric_limits<size_t>::max())) {
        return default_workspace_cache_bytes;
    }
    return static_cast<size_t>(parsed);
}

template <typename T>
static size_t workspace_capacity_bytes(
    const ResizeWorkspace<T>& workspace) noexcept
{
    const size_t prev_capacity = workspace.prev.capacity();
    const size_t scratch_capacity = workspace.scratch.capacity();
    if (prev_capacity > std::numeric_limits<size_t>::max() - scratch_capacity) {
        return std::numeric_limits<size_t>::max();
    }
    const size_t total_capacity = prev_capacity + scratch_capacity;
    if (total_capacity > std::numeric_limits<size_t>::max() / sizeof(T)) {
        return std::numeric_limits<size_t>::max();
    }
    return total_capacity * sizeof(T);
}

class ResizeWorkspaceCacheBudget {
public:
    struct Snapshot {
        size_t limit_bytes = 0;
        size_t retained_bytes = 0;
        size_t retained_count = 0;
    };

    explicit ResizeWorkspaceCacheBudget(size_t limit_bytes) noexcept
        : limit_bytes_(limit_bytes)
    {
    }

    bool try_admit(size_t bytes) noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            // Always keep one useful primary allocation for repeated calls,
            // even when the configured extra-workspace budget is zero or the
            // primary alone exceeds it.  The byte limit governs only entries
            // admitted alongside that primary.
            if (retained_count_ == 0) {
                retained_bytes_ = bytes;
                retained_count_ = 1;
                return true;
            }
            if (retained_count_ >= max_retained_plan_workspaces ||
                retained_bytes_ >= limit_bytes_ ||
                bytes > limit_bytes_ ||
                bytes > limit_bytes_ - retained_bytes_) {
                return false;
            }
            retained_bytes_ += bytes;
            ++retained_count_;
            return true;
        } catch (...) {
            return false;
        }
    }

    void remove(size_t bytes) noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            retained_bytes_ =
                (bytes <= retained_bytes_) ? retained_bytes_ - bytes : 0;
            if (retained_count_ > 0) {
                --retained_count_;
            }
        } catch (...) {
            // Conservative stale accounting can only reduce later admission.
        }
    }

    Snapshot snapshot() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return Snapshot{limit_bytes_, retained_bytes_, retained_count_};
    }

private:
    const size_t limit_bytes_;
    mutable std::mutex mutex_;
    size_t retained_bytes_ = 0;
    size_t retained_count_ = 0;
};

struct ResizeWorkspacePoolSnapshot {
    size_t retained_count = 0;
    size_t retained_bytes = 0;
    size_t prev_bytes = 0;
    size_t scratch_bytes = 0;
};

// ResizePlan instances are immutable execution descriptions.  Transient
// ping-pong storage is leased per call, so concurrent applications of the same
// plan never share mutable buffers.  Both dtype-specific pools share a
// byte/count coordinator owned by the plan, so a one-off burst cannot leave
// an unbounded amount of transient storage behind.
template <typename T>
class ResizeWorkspacePool {
public:
    explicit ResizeWorkspacePool(ResizeWorkspaceCacheBudget* budget)
        : budget_(budget)
    {
        available_.reserve(max_retained_plan_workspaces);
    }

    class Lease {
    public:
        Lease() = default;
        Lease(
            ResizeWorkspacePool* owner,
            std::unique_ptr<ResizeWorkspace<T>> workspace)
            : owner_(owner), workspace_(std::move(workspace))
        {
        }

        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;

        Lease(Lease&& other) noexcept
            : owner_(other.owner_), workspace_(std::move(other.workspace_))
        {
            other.owner_ = nullptr;
        }

        Lease& operator=(Lease&& other) noexcept
        {
            if (this != &other) {
                release();
                owner_ = other.owner_;
                workspace_ = std::move(other.workspace_);
                other.owner_ = nullptr;
            }
            return *this;
        }

        ~Lease() noexcept { release(); }

        ResizeWorkspace<T>& get() noexcept { return *workspace_; }

    private:
        void release() noexcept
        {
            if (owner_ != nullptr && workspace_ != nullptr) {
                owner_->put(std::move(workspace_));
            }
            owner_ = nullptr;
        }

        ResizeWorkspacePool* owner_ = nullptr;
        std::unique_ptr<ResizeWorkspace<T>> workspace_;
    };

    Lease acquire()
    {
        std::unique_ptr<ResizeWorkspace<T>> workspace;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!available_.empty()) {
                workspace = std::move(available_.back());
                available_.pop_back();
                budget_->remove(workspace_capacity_bytes(*workspace));
            }
        }
        if (!workspace) {
            workspace = std::make_unique<ResizeWorkspace<T>>();
        }
        return Lease(this, std::move(workspace));
    }

    ResizeWorkspacePoolSnapshot snapshot() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ResizeWorkspacePoolSnapshot result;
        result.retained_count = available_.size();
        for (const auto& workspace : available_) {
            const size_t prev_bytes = workspace->prev.capacity() * sizeof(T);
            const size_t scratch_bytes =
                workspace->scratch.capacity() * sizeof(T);
            result.prev_bytes += prev_bytes;
            result.scratch_bytes += scratch_bytes;
            result.retained_bytes += prev_bytes + scratch_bytes;
        }
        return result;
    }

private:
    void put(std::unique_ptr<ResizeWorkspace<T>> workspace) noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            const size_t bytes = workspace_capacity_bytes(*workspace);
            if (!budget_->try_admit(bytes)) {
                return;
            }
            try {
                available_.push_back(std::move(workspace));
            } catch (...) {
                budget_->remove(bytes);
                throw;
            }
        } catch (...) {
            // A lease destructor must not throw.  Discarding a workspace only
            // forfeits a future allocation optimization.
        }
    }

    ResizeWorkspaceCacheBudget* budget_;
    mutable std::mutex mutex_;
    std::vector<std::unique_ptr<ResizeWorkspace<T>>> available_;
};

template <typename T>
void resize_nd_impl_preplanned_to(
    const T* input_data,
    T* final_out,
    const std::vector<int64>& expected_in_shape,
    const std::vector<int>& active_axes,
    const std::vector<std::vector<int64>>& pass_in_shapes,
    const std::vector<std::vector<int64>>& pass_out_shapes,
    const std::vector<int64>& pass_output_elems,
    const std::vector<lsresize::LSParams>& pass_params,
    const std::vector<std::shared_ptr<const lsresize::Plan1D>>& pass_plans,
    const PreparedLinearAxis* fused_identity_axis2,
    std::vector<T>& prev,
    std::vector<T>& scratch)
{
    if (active_axes.empty()) {
        const int64 total = std::accumulate(
            expected_in_shape.begin(), expected_in_shape.end(),
            static_cast<int64>(1),
            std::multiplies<int64>());
        std::copy(
            input_data,
            input_data + total,
            final_out);
        return;
    }

    const std::vector<int64>& final_shape = pass_out_shapes.back();
    auto pass_for_axis = [&](int axis) -> size_t {
        const auto found = std::find(
            active_axes.begin(), active_axes.end(), axis);
        if (found == active_axes.end()) {
            return active_axes.size();
        }
        return static_cast<size_t>(found - active_axes.begin());
    };

    const lsresize::LSParams& method = pass_params.front();
    if (can_use_fused_2d_linear<T>(
            expected_in_shape,
            final_shape,
            active_axes,
            method.interp_degree,
            method.analy_degree,
            method.synthe_degree)) {
        const size_t i0 = pass_for_axis(0);
        const size_t i1 = pass_for_axis(1);
        if (i0 < pass_plans.size() && i1 < pass_plans.size() &&
            pass_plans[i0] && pass_plans[i1]) {
            if constexpr (std::is_same_v<T, float>) {
                lsresize::resize_2d_linear_preplanned_f32(
                    input_data,
                    final_out,
                    expected_in_shape,
                    final_shape,
                    pass_params[i0],
                    pass_params[i1],
                    *pass_plans[i0],
                    *pass_plans[i1]);
            } else {
                lsresize::resize_2d_linear_preplanned(
                    input_data,
                    final_out,
                    expected_in_shape,
                    final_shape,
                    pass_params[i0],
                    pass_params[i1],
                    *pass_plans[i0],
                    *pass_plans[i1]);
            }
            return;
        }
    }

    if (can_use_fused_3d_linear<T>(
            expected_in_shape,
            final_shape,
            active_axes,
            method.interp_degree,
            method.analy_degree,
            method.synthe_degree)) {
        const size_t i0 = pass_for_axis(0);
        const size_t i1 = pass_for_axis(1);
        const lsresize::LSParams* params2 = nullptr;
        const lsresize::Plan1D* plan2 = nullptr;
        if (active_axes.size() == 3) {
            const size_t i2 = pass_for_axis(2);
            if (i2 < pass_plans.size() && pass_plans[i2]) {
                params2 = &pass_params[i2];
                plan2 = pass_plans[i2].get();
            }
        } else if (fused_identity_axis2 != nullptr &&
                   fused_identity_axis2->plan) {
            params2 = &fused_identity_axis2->params;
            plan2 = fused_identity_axis2->plan.get();
        }
        if (i0 < pass_plans.size() && i1 < pass_plans.size() &&
            pass_plans[i0] && pass_plans[i1] &&
            params2 != nullptr && plan2 != nullptr) {
            if constexpr (std::is_same_v<T, float>) {
                lsresize::resize_3d_linear_preplanned_f32(
                    input_data,
                    final_out,
                    expected_in_shape,
                    final_shape,
                    pass_params[i0],
                    pass_params[i1],
                    *params2,
                    *pass_plans[i0],
                    *pass_plans[i1],
                    *plan2);
            } else {
                lsresize::resize_3d_linear_preplanned(
                    input_data,
                    final_out,
                    expected_in_shape,
                    final_shape,
                    pass_params[i0],
                    pass_params[i1],
                    *params2,
                    *pass_plans[i0],
                    *pass_plans[i1],
                    *plan2);
            }
            return;
        }
    }

    const int two_axis_kind = fused_3d_linear_two_axis_kind<T>(
        expected_in_shape,
        final_shape,
        active_axes,
        method.interp_degree,
        method.analy_degree,
        method.synthe_degree);
    if (two_axis_kind == 2 || two_axis_kind == 12) {
        const int first_axis = (two_axis_kind == 2) ? 0 : 1;
        const size_t first = pass_for_axis(first_axis);
        const size_t second = pass_for_axis(2);
        if (first < pass_plans.size() && second < pass_plans.size() &&
            pass_plans[first] && pass_plans[second]) {
            if constexpr (std::is_same_v<T, float>) {
                if (two_axis_kind == 2) {
                    lsresize::resize_3d_linear_axis02_preplanned_f32(
                        input_data, final_out, expected_in_shape, final_shape,
                        pass_params[first], pass_params[second],
                        *pass_plans[first], *pass_plans[second]);
                } else {
                    lsresize::resize_3d_linear_axis12_preplanned_f32(
                        input_data, final_out, expected_in_shape, final_shape,
                        pass_params[first], pass_params[second],
                        *pass_plans[first], *pass_plans[second]);
                }
            } else {
                if (two_axis_kind == 2) {
                    lsresize::resize_3d_linear_axis02_preplanned(
                        input_data, final_out, expected_in_shape, final_shape,
                        pass_params[first], pass_params[second],
                        *pass_plans[first], *pass_plans[second]);
                } else {
                    lsresize::resize_3d_linear_axis12_preplanned(
                        input_data, final_out, expected_in_shape, final_shape,
                        pass_params[first], pass_params[second],
                        *pass_plans[first], *pass_plans[second]);
                }
            }
            return;
        }
    }

    const int n_passes = static_cast<int>(active_axes.size());
    for (int pass = 0; pass < n_passes; ++pass) {
        const bool first_pass = (pass == 0);
        const bool last_pass = (pass == n_passes - 1);

        const T* in_ptr = first_pass
            ? input_data
            : prev.data();
        T* out_ptr = nullptr;

        if (last_pass) {
            out_ptr = final_out;
        } else if (n_passes == 2) {
            // A two-axis plan needs exactly one intermediate.  Always reuse
            // `prev` for it; swapping `prev` and `scratch` on repeated calls
            // would eventually retain two equally large allocations.
            prev.resize(static_cast<size_t>(
                pass_output_elems[static_cast<size_t>(pass)]));
            out_ptr = prev.data();
        } else {
            scratch.resize(static_cast<size_t>(
                pass_output_elems[static_cast<size_t>(pass)]));
            out_ptr = scratch.data();
        }

        const auto& plan = pass_plans[static_cast<size_t>(pass)];
        if (plan) {
            AxisDispatch<T>::apply_preplanned(
                in_ptr,
                out_ptr,
                pass_in_shapes[static_cast<size_t>(pass)],
                pass_out_shapes[static_cast<size_t>(pass)],
                active_axes[static_cast<size_t>(pass)],
                pass_params[static_cast<size_t>(pass)],
                *plan);
        } else {
            // Degenerate axes return before the ordinary axis entry point
            // reaches its cache lookup, so a Plan1D is neither needed nor
            // constructible for this pass.
            AxisDispatch<T>::apply(
                in_ptr,
                out_ptr,
                pass_in_shapes[static_cast<size_t>(pass)],
                pass_out_shapes[static_cast<size_t>(pass)],
                active_axes[static_cast<size_t>(pass)],
                pass_params[static_cast<size_t>(pass)]);
        }

        if (!last_pass && n_passes != 2) {
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
    auto in_arr = aligned_native_input<T>(input);
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
static py::array_t<T, py::array::c_style> checked_preplanned_output(
    py::array output,
    const std::vector<int64>& expected_out_shape)
{
    // Constructing array_t<T, c_style> directly is allowed to create a
    // converted/contiguous temporary.  That is never valid for an ``into``
    // API: the compute kernel must write the exact caller-supplied buffer.
    if (!output.dtype().is(py::dtype::of<T>())) {
        throw py::type_error(
            "resize_nd: output dtype must match the native compute dtype");
    }
    if ((output.flags() & py::array::c_style) == 0) {
        throw py::value_error(
            "resize_nd: output must be C-contiguous");
    }
    if ((reinterpret_cast<std::uintptr_t>(output.data()) % alignof(T)) != 0) {
        throw py::value_error(
            "resize_nd: output must be naturally aligned");
    }
    auto out_arr =
        py::reinterpret_borrow<py::array_t<T, py::array::c_style>>(output);
    if (!out_arr.writeable()) {
        throw py::value_error("resize_nd: output must be writeable");
    }
    if (out_arr.ndim() !=
        static_cast<py::ssize_t>(expected_out_shape.size())) {
        throw py::value_error(
            "resize_nd: output ndim does not match output ndim");
    }
    for (int ax = 0; ax < out_arr.ndim(); ++ax) {
        const int64 got = static_cast<int64>(out_arr.shape(ax));
        const int64 expected = expected_out_shape[static_cast<size_t>(ax)];
        if (got != expected) {
            throw py::value_error(
                "resize_nd: output shape does not match output_shape");
        }
    }
    return out_arr;
}

static bool byte_ranges_overlap(
    const void* first,
    size_t first_size,
    const void* second,
    size_t second_size) noexcept
{
    if (first_size == 0 || second_size == 0) {
        return false;
    }
    const auto a = reinterpret_cast<std::uintptr_t>(first);
    const auto b = reinterpret_cast<std::uintptr_t>(second);
    if (a <= b) {
        return (b - a) < first_size;
    }
    return (a - b) < second_size;
}

template <typename T, int InputFlags, int OutputFlags>
static void reject_overlapping_arrays(
    const py::array_t<T, InputFlags>& input,
    const py::array_t<T, OutputFlags>& output)
{
    if (byte_ranges_overlap(
            input.data(),
            static_cast<size_t>(input.nbytes()),
            output.data(),
            static_cast<size_t>(output.nbytes()))) {
        throw py::value_error(
            "resize_nd: input and output arrays must not overlap");
    }
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
    const std::vector<std::shared_ptr<const lsresize::Plan1D>>& pass_plans,
    const PreparedLinearAxis* fused_identity_axis2,
    std::vector<T>& prev,
    std::vector<T>& scratch)
{
    auto in_arr = checked_preplanned_input<T>(input, expected_in_shape);
    std::vector<py::ssize_t> out_shape_ssize(
        out_shape.begin(), out_shape.end());
    py::array_t<T> out(out_shape_ssize);

    const T* input_data = static_cast<const T*>(in_arr.data());
    T* output_data = static_cast<T*>(out.mutable_data());

    {
        py::gil_scoped_release release;
        resize_nd_impl_preplanned_to<T>(
            input_data,
            output_data,
            expected_in_shape,
            active_axes,
            pass_in_shapes,
            pass_out_shapes,
            pass_output_elems,
            pass_params,
            pass_plans,
            fused_identity_axis2,
            prev,
            scratch);
    }
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
    const std::vector<std::shared_ptr<const lsresize::Plan1D>>& pass_plans,
    const PreparedLinearAxis* fused_identity_axis2,
    std::vector<T>& prev,
    std::vector<T>& scratch)
{
    auto in_arr = checked_preplanned_input<T>(input, expected_in_shape);
    auto out_arr = checked_preplanned_output<T>(output, out_shape);
    reject_overlapping_arrays(in_arr, out_arr);

    const T* input_data = static_cast<const T*>(in_arr.data());
    T* output_data = static_cast<T*>(out_arr.mutable_data());

    {
        py::gil_scoped_release release;
        resize_nd_impl_preplanned_to<T>(
            input_data,
            output_data,
            expected_in_shape,
            active_axes,
            pass_in_shapes,
            pass_out_shapes,
            pass_output_elems,
            pass_params,
            pass_plans,
            fused_identity_axis2,
            prev,
            scratch);
    }
    return output;
}

template <typename T>
py::array_t<T> resize_nd_impl(
    py::array input,
    std::vector<double> zoom_factors,
    const std::vector<int>& selected_axes,
    int interp_degree,
    int analy_degree,
    int synthe_degree)
{
    auto in_arr = aligned_native_input<T>(input);
    if (in_arr.ndim() <= 0) {
        throw std::runtime_error(
            "resize_nd: input must be at least 1-D");
    }

    const std::vector<int64> in_shape = shape_to_vec_i64(in_arr);
    const std::vector<int64> out_shape =
        compute_output_shape(in_shape, zoom_factors, selected_axes);
    canonicalize_unselected_zoom_factors(zoom_factors, selected_axes);
    const std::vector<int> axis_order =
        choose_axis_order(in_shape, out_shape);
    const std::vector<int> active_axes =
        choose_active_axes(
            axis_order,
            in_shape,
            out_shape,
            selected_axes,
            interp_degree,
            synthe_degree);

    if (can_use_fused_2d_linear<T>(
            in_shape,
            out_shape,
            active_axes,
            interp_degree,
            analy_degree,
            synthe_degree)) {
        return resize_nd_impl_fused_2d_linear<T>(
            in_arr,
            in_shape,
            out_shape,
            zoom_factors);
    }

    if (can_use_fused_3d_linear<T>(
            in_shape,
            out_shape,
            active_axes,
            interp_degree,
            analy_degree,
            synthe_degree)) {
        return resize_nd_impl_fused_3d_linear<T>(
            in_arr,
            in_shape,
            out_shape,
            zoom_factors);
    }

    const int fused_3d_two_axis_kind =
        fused_3d_linear_two_axis_kind<T>(
            in_shape,
            out_shape,
            active_axes,
            interp_degree,
            analy_degree,
            synthe_degree);
    if (fused_3d_two_axis_kind != 0) {
        return resize_nd_impl_fused_3d_linear_two_axis<T>(
            in_arr,
            in_shape,
            out_shape,
            zoom_factors,
            fused_3d_two_axis_kind);
    }

    return resize_nd_impl_planned<T>(
        in_arr,
        in_shape,
        out_shape,
        zoom_factors,
        active_axes,
        interp_degree,
        analy_degree,
        synthe_degree);
}

class ResizePlanNative {
public:
    ResizePlanNative(
        std::vector<int64> input_shape,
        std::vector<double> zoom_factors,
        int interp_degree,
        int analy_degree,
        int synthe_degree,
        std::vector<int> selected_axes)
        : input_shape_(std::move(input_shape)),
          zoom_factors_(std::move(zoom_factors)),
          selected_axes_(std::move(selected_axes)),
          workspace_cache_budget_(workspace_cache_limit_bytes()),
          workspace_pool_f32_(&workspace_cache_budget_),
          workspace_pool_f64_(&workspace_cache_budget_),
          interp_degree_(interp_degree),
          analy_degree_(analy_degree),
          synthe_degree_(synthe_degree)
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
            compute_output_shape(input_shape_, zoom_factors_, selected_axes_);
        canonicalize_unselected_zoom_factors(
            zoom_factors_, selected_axes_);
        axis_order_ = choose_axis_order(input_shape_, output_shape_);
        active_axes_ = choose_active_axes(
            axis_order_,
            input_shape_,
            output_shape_,
            selected_axes_,
            interp_degree_,
            synthe_degree_);

        std::vector<int64> cur_shape = input_shape_;
        pass_in_shapes_.reserve(active_axes_.size());
        pass_out_shapes_.reserve(active_axes_.size());
        pass_output_elems_.reserve(active_axes_.size());
        pass_params_.reserve(active_axes_.size());
        pass_plans_.reserve(active_axes_.size());
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
            pass_params_.push_back(p);

            const int64 in_length = cur_shape[static_cast<size_t>(ax)];
            const int64 out_length = next_shape[static_cast<size_t>(ax)];
            if (in_length == 1 || out_length == 1) {
                // The native axis kernel handles endpoint-degenerate grids
                // explicitly before consulting a Plan1D.
                pass_plans_.push_back(nullptr);
            } else {
                pass_plans_.push_back(lsresize::get_plan_1d_cached(
                    static_cast<int>(in_length), p));
            }

            cur_shape.swap(next_shape);
        }

        if (input_shape_.size() == 3 && active_axes_.size() == 2 &&
            interp_degree_ == 1 && analy_degree_ < 0 && synthe_degree_ == 1) {
            const bool has_axis0 = std::find(
                active_axes_.begin(), active_axes_.end(), 0) != active_axes_.end();
            const bool has_axis1 = std::find(
                active_axes_.begin(), active_axes_.end(), 1) != active_axes_.end();
            if (has_axis0 && has_axis1) {
                fused_identity_axis2_ = prepare_linear_axis(input_shape_[2], 1.0);
            }
        }
    }

    py::array apply(py::array input)
    {
        py::dtype dt = input.dtype();
        if (dt.is(py::dtype::of<float>())) {
            auto lease = workspace_pool_f32_.acquire();
            auto& workspace = lease.get();
            return resize_nd_impl_preplanned<float>(
                input,
                input_shape_,
                output_shape_,
                active_axes_,
                pass_in_shapes_,
                pass_out_shapes_,
                pass_output_elems_,
                pass_params_,
                pass_plans_,
                fused_identity_axis2_.plan ? &fused_identity_axis2_ : nullptr,
                workspace.prev,
                workspace.scratch);
        }
        auto lease = workspace_pool_f64_.acquire();
        auto& workspace = lease.get();
        return resize_nd_impl_preplanned<double>(
            input,
            input_shape_,
            output_shape_,
            active_axes_,
            pass_in_shapes_,
            pass_out_shapes_,
            pass_output_elems_,
            pass_params_,
            pass_plans_,
            fused_identity_axis2_.plan ? &fused_identity_axis2_ : nullptr,
            workspace.prev,
            workspace.scratch);
    }

    py::array apply_into(py::array input, py::array output)
    {
        py::dtype dt = input.dtype();
        if (dt.is(py::dtype::of<float>())) {
            auto lease = workspace_pool_f32_.acquire();
            auto& workspace = lease.get();
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
                pass_plans_,
                fused_identity_axis2_.plan ? &fused_identity_axis2_ : nullptr,
                workspace.prev,
                workspace.scratch);
        }
        auto lease = workspace_pool_f64_.acquire();
        auto& workspace = lease.get();
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
            pass_plans_,
            fused_identity_axis2_.plan ? &fused_identity_axis2_ : nullptr,
            workspace.prev,
            workspace.scratch);
    }

    py::tuple input_shape() const { return vec_i64_to_tuple(input_shape_); }
    py::tuple output_shape() const { return vec_i64_to_tuple(output_shape_); }
    py::tuple zoom_factors() const { return vec_double_to_tuple(zoom_factors_); }
    py::tuple axes() const { return vec_int_to_tuple(selected_axes_); }
    int interp_degree() const { return interp_degree_; }
    int analy_degree() const { return analy_degree_; }
    int synthe_degree() const { return synthe_degree_; }

    py::dict workspace_cache_info() const
    {
        // Snapshot the pools without nesting their mutexes with the shared
        // budget mutex.  This property is a private testing/diagnostic hook;
        // callers should query it only when no plan application is active.
        const auto f32 = workspace_pool_f32_.snapshot();
        const auto f64 = workspace_pool_f64_.snapshot();
        const auto total = workspace_cache_budget_.snapshot();
        py::dict info;
        info["limit_bytes"] = py::int_(total.limit_bytes);
        info["retained_bytes"] = py::int_(total.retained_bytes);
        info["retained_count"] = py::int_(total.retained_count);
        info["max_retained_count"] =
            py::int_(max_retained_plan_workspaces);
        info["float32_count"] = py::int_(f32.retained_count);
        info["float32_bytes"] = py::int_(f32.retained_bytes);
        info["float32_prev_bytes"] = py::int_(f32.prev_bytes);
        info["float32_scratch_bytes"] = py::int_(f32.scratch_bytes);
        info["float64_count"] = py::int_(f64.retained_count);
        info["float64_bytes"] = py::int_(f64.retained_bytes);
        info["float64_prev_bytes"] = py::int_(f64.prev_bytes);
        info["float64_scratch_bytes"] = py::int_(f64.scratch_bytes);
        return info;
    }

private:
    std::vector<int64> input_shape_;
    std::vector<int64> output_shape_;
    std::vector<double> zoom_factors_;
    std::vector<int> selected_axes_;
    std::vector<int> axis_order_;
    std::vector<int> active_axes_;
    std::vector<std::vector<int64>> pass_in_shapes_;
    std::vector<std::vector<int64>> pass_out_shapes_;
    std::vector<int64> pass_output_elems_;
    std::vector<lsresize::LSParams> pass_params_;
    std::vector<std::shared_ptr<const lsresize::Plan1D>> pass_plans_;
    PreparedLinearAxis fused_identity_axis2_{};
    ResizeWorkspaceCacheBudget workspace_cache_budget_;
    ResizeWorkspacePool<float> workspace_pool_f32_;
    ResizeWorkspacePool<double> workspace_pool_f64_;
    int interp_degree_;
    int analy_degree_;
    int synthe_degree_;
};

// Python-visible dispatcher: chooses float32 vs float64 pipeline
static py::array resize_nd(
    py::array input,
    std::vector<double> zoom_factors,
    int interp_degree,
    int analy_degree,
    int synthe_degree,
    py::object axes)
{
    // Safety net: ensure degrees/combos are within the supported regime.
    validate_degrees(interp_degree, analy_degree, synthe_degree);
    if (input.ndim() <= 0) {
        throw py::value_error("resize_nd: input must be at least 1-D");
    }
    const std::vector<int> selected_axes =
        normalize_selected_axes(axes, static_cast<int>(input.ndim()));

    py::dtype dt = input.dtype();

    // Keep float32 as float32 storage, double-internal.
    if (dt.is(py::dtype::of<float>())) {
        return resize_nd_impl<float>(
            input,
            std::move(zoom_factors),
            selected_axes,
            interp_degree,
            analy_degree,
            synthe_degree
        );
    }

    // Default: float64 storage (and internal).
    return resize_nd_impl<double>(
        input,
        std::move(zoom_factors),
        selected_axes,
        interp_degree,
        analy_degree,
        synthe_degree
    );
}

template <typename T>
static py::array resize_nd_into_t(
    py::array input,
    py::array output,
    std::vector<double> zoom_factors,
    const std::vector<int>& selected_axes,
    int interp_degree,
    int analy_degree,
    int synthe_degree)
{
    const std::vector<int64> in_shape = shape_to_vec_i64(input);
    const std::vector<int64> out_shape =
        compute_output_shape(in_shape, zoom_factors, selected_axes);
    canonicalize_unselected_zoom_factors(zoom_factors, selected_axes);
    const std::vector<int> axis_order =
        choose_axis_order(in_shape, out_shape);
    const std::vector<int> active_axes = choose_active_axes(
        axis_order,
        in_shape,
        out_shape,
        selected_axes,
        interp_degree,
        synthe_degree);

    if (can_use_fused_2d_linear<T>(
            in_shape,
            out_shape,
            active_axes,
            interp_degree,
            analy_degree,
            synthe_degree)) {
        return resize_nd_impl_fused_2d_linear_into<T>(
            input,
            output,
            in_shape,
            out_shape,
            zoom_factors);
    }

    if (can_use_fused_3d_linear<T>(
            in_shape,
            out_shape,
            active_axes,
            interp_degree,
            analy_degree,
            synthe_degree)) {
        return resize_nd_impl_fused_3d_linear_into<T>(
            input,
            output,
            in_shape,
            out_shape,
            zoom_factors);
    }

    const int two_axis_kind = fused_3d_linear_two_axis_kind<T>(
        in_shape,
        out_shape,
        active_axes,
        interp_degree,
        analy_degree,
        synthe_degree);
    if (two_axis_kind != 0) {
        return resize_nd_impl_fused_3d_linear_two_axis_into<T>(
            input,
            output,
            in_shape,
            out_shape,
            zoom_factors,
            two_axis_kind);
    }

    // The generic direct-output route uses the same immutable execution plan
    // as repeated workloads.  Its final pass writes to `output`; no temporary
    // NumPy result is allocated.
    ResizePlanNative plan(
        in_shape,
        zoom_factors,
        interp_degree,
        analy_degree,
        synthe_degree,
        selected_axes);
    return plan.apply_into(input, output);
}

static py::array resize_nd_into(
    py::array input,
    py::array output,
    std::vector<double> zoom_factors,
    int interp_degree,
    int analy_degree,
    int synthe_degree,
    py::object axes)
{
    validate_degrees(interp_degree, analy_degree, synthe_degree);
    if (input.ndim() <= 0) {
        throw py::value_error("resize_nd: input must be at least 1-D");
    }
    const std::vector<int> selected_axes =
        normalize_selected_axes(axes, static_cast<int>(input.ndim()));

    if (input.dtype().is(py::dtype::of<float>())) {
        return resize_nd_into_t<float>(
            input,
            output,
            zoom_factors,
            selected_axes,
            interp_degree,
            analy_degree,
            synthe_degree);
    }
    return resize_nd_into_t<double>(
        input,
        output,
        zoom_factors,
        selected_axes,
        interp_degree,
        analy_degree,
        synthe_degree);
}

PYBIND11_MODULE(_lsresize, m) {
    m.doc() = "splineops: fast LS/oblique resize (C++ core)";

    m.def("resize_nd", &resize_nd,
          py::arg("input"),
          py::arg("zoom_factors"),
          py::arg("interp_degree"),
          py::arg("analy_degree"),
          py::arg("synthe_degree"),
          py::arg("axes") = py::none());

    m.def("resize_nd_into", &resize_nd_into,
          py::arg("input"),
          py::arg("output"),
          py::arg("zoom_factors"),
          py::arg("interp_degree"),
          py::arg("analy_degree"),
          py::arg("synthe_degree"),
          py::arg("axes") = py::none());

    py::class_<ResizePlanNative>(m, "ResizePlan")
        .def(py::init([](
                 std::vector<int64> input_shape,
                 std::vector<double> zoom_factors,
                 int interp_degree,
                 int analy_degree,
                 int synthe_degree,
                 py::object axes) {
             std::vector<int> selected_axes = normalize_selected_axes(
                 axes, static_cast<int>(input_shape.size()));
             return std::make_unique<ResizePlanNative>(
                 std::move(input_shape),
                 std::move(zoom_factors),
                 interp_degree,
                 analy_degree,
                 synthe_degree,
                 std::move(selected_axes));
             }),
             py::arg("input_shape"),
             py::arg("zoom_factors"),
             py::arg("interp_degree"),
             py::arg("analy_degree"),
             py::arg("synthe_degree"),
             py::arg("axes") = py::none())
        .def("apply", &ResizePlanNative::apply, py::arg("input"))
        .def("apply_into",
             &ResizePlanNative::apply_into,
             py::arg("input"),
             py::arg("output"))
        .def_property_readonly("input_shape", &ResizePlanNative::input_shape)
        .def_property_readonly("output_shape", &ResizePlanNative::output_shape)
        .def_property_readonly("zoom_factors", &ResizePlanNative::zoom_factors)
        .def_property_readonly("axes", &ResizePlanNative::axes)
        .def_property_readonly("interp_degree", &ResizePlanNative::interp_degree)
        .def_property_readonly("analy_degree", &ResizePlanNative::analy_degree)
        .def_property_readonly("synthe_degree", &ResizePlanNative::synthe_degree)
        .def_property_readonly(
            "_workspace_cache_info",
            &ResizePlanNative::workspace_cache_info);
}
