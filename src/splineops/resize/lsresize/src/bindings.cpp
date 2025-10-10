// splineops/src/splineops/resize/lsresize/src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cstdint>
#include <cstring>
#include "lsresize/resize1d.h"
#include "lsresize/resizend.h"

namespace py = pybind11;
using namespace lsresize;

static std::vector<int64_t> to_vec64(const py::tuple& t) {
  std::vector<int64_t> v; v.reserve(t.size());
  for (auto it : t) v.push_back(it.cast<int64_t>());
  return v;
}
static std::vector<double> to_vecd(const py::sequence& s) {
  std::vector<double> v; v.reserve(s.size());
  for (auto it : s) v.push_back(py::cast<double>(it));
  return v;
}
static int64_t prod(const std::vector<int64_t>& shape) {
  int64_t p = 1; for (auto x : shape) p *= x; return p;
}

py::array_t<double> resizend(py::array_t<double, py::array::c_style | py::array::forcecast> data,
                             py::object zoom_factors_obj,
                             py::object output_size_obj,
                             std::string algo,  // "interpolation" | "oblique" | "least-squares"
                             int degree,       // 1..3
                             bool inversable,
                             py::object channel_axis_obj)
{
  if (!channel_axis_obj.is_none())
    throw std::runtime_error("channel_axis is not supported in this minimal binding yet.");

  py::buffer_info info = data.request();
  const int D = (int)info.ndim;

  // shapes
  std::vector<int64_t> in_shape(info.shape.begin(), info.shape.end());
  std::vector<int64_t> out_shape = in_shape;
  std::vector<double> zoom(D, 1.0);

  if (!output_size_obj.is_none()) {
    auto tup = output_size_obj.cast<py::tuple>();
    auto outsz = to_vec64(tup);
    if ((int)outsz.size() != D) throw std::runtime_error("output_size has wrong length");
    out_shape = outsz;
    for (int i=0;i<D;++i) zoom[i] = (double)out_shape[i] / (double)in_shape[i];
  } else {
    auto z = to_vecd(zoom_factors_obj);
    if (z.size() == 1) z.assign(D, z[0]);
    if ((int)z.size() != D) throw std::runtime_error("zoom_factors has wrong length");
    zoom = z;
    for (int i=0;i<D;++i) out_shape[i] = (int64_t)std::llround(in_shape[i]*zoom[i]);
  }

  // copy input into contiguous double buffer
  const double* src = static_cast<const double*>(info.ptr);
  std::vector<double> tmp_in((size_t)prod(in_shape));
  std::memcpy(tmp_in.data(), src, tmp_in.size() * sizeof(double));

  // degrees/preset
  int interp_degree = degree, analy_degree = degree, synthe_degree = degree;
  if (algo == "interpolation") analy_degree = -1;
  else if (algo == "oblique") analy_degree = (degree==1 ? 0 : 1);

  // separable pass over axes
  std::vector<int64_t> cur_shape = in_shape;
  for (int ax = 0; ax < D; ++ax) {
    std::vector<int64_t> next_shape = cur_shape; next_shape[ax] = out_shape[ax];
    std::vector<double> tmp_out((size_t)prod(next_shape), 0.0);

    LSParams params{interp_degree, analy_degree, synthe_degree, zoom[ax], 0.0, inversable};
    resize_along_axis(tmp_in.data(), tmp_out.data(), cur_shape, next_shape, ax, params);

    tmp_in.swap(tmp_out);
    cur_shape.swap(next_shape);
  }

  // return numpy array
  py::array_t<double> out_py(out_shape);
  std::memcpy(out_py.mutable_data(), tmp_in.data(), tmp_in.size()*sizeof(double));
  return out_py;
}

PYBIND11_MODULE(lsresize, m) {
  m.doc() = "Least-squares / oblique B-spline resize (C++ core)";
  m.def("resize", &resizend,
        py::arg("data"),
        py::arg("zoom_factors") = py::none(),
        py::arg("output_size")  = py::none(),
        py::arg("algo")         = "least-squares",
        py::arg("degree")       = 3,
        py::arg("inversable")   = false,
        py::arg("channel_axis") = py::none(),
        "Resize N-D arrays using the Muñoz–Blu–Unser projection pipeline.\n"
        "algo: 'interpolation' | 'oblique' | 'least-squares'\n"
        "degree: 1..3 (internal correlation degree up to 7).\n"
        "Note: 'channel_axis' is not yet supported in this minimal binding.");
}
