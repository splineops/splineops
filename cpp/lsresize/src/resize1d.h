// splineops/cpp/lsresize/src/resize1d.h
#pragma once
#include <vector>

namespace lsresize {

struct LSParams {
  int interp_degree;   // n
  int analy_degree;    // n1  (=-1 for pure interpolation)
  int synthe_degree;   // n2  (usually = n)
  double zoom;         // a
  double shift;        // b
  bool inversable;     // size adjustment
};

void resize_1d(const std::vector<double>& in,
               std::vector<double>& out,
               const LSParams& p);

} // namespace lsresize
