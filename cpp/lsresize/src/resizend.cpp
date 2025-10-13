// splineops/cpp/lsresize/src/resizend.cpp
#include "resizend.h"
#include "utils.h"
#include <vector>
#include <numeric>
#include <cstdint>

namespace lsresize {

static std::vector<int64_t> strides_from_shape(const std::vector<int64_t>& shape) {
  std::vector<int64_t> s(shape.size(), 1);
  for (int i = (int)shape.size() - 2; i >= 0; --i) s[i] = s[i+1] * shape[i+1];
  return s;
}

void resize_along_axis(const double* in, double* out,
                       const std::vector<int64_t>& in_shape,
                       const std::vector<int64_t>& out_shape,
                       int axis,
                       const LSParams& p)
{
  const int D = (int)in_shape.size();
  const auto in_strides  = strides_from_shape(in_shape);
  const auto out_strides = strides_from_shape(out_shape);

  int64_t nlines = 1;
  for (int d = 0; d < D; ++d) if (d != axis) nlines *= in_shape[d];

  std::vector<int64_t> idx(D, 0);
  std::vector<int> bases;
  bases.reserve(D);
  for (int d = D-1; d >= 0; --d) if (d != axis) bases.push_back(d);

  #pragma omp parallel for if(nlines > 64)
  for (int64_t line = 0; line < nlines; ++line) {
    int64_t t = line;
    for (int bi = 0; bi < (int)bases.size(); ++bi) {
      int d = bases[bi];
      const int64_t q = t % in_shape[d];
      idx[d] = q;
      t /= in_shape[d];
    }

    int64_t in_off = 0, out_off = 0;
    for (int d = 0; d < D; ++d) if (d != axis) {
      in_off  += idx[d] * in_strides[d];
      out_off += idx[d] * out_strides[d];
    }

    std::vector<double> line_in((size_t)in_shape[axis]);
    for (int64_t i = 0; i < in_shape[axis]; ++i) line_in[i] = in[in_off + i*in_strides[axis]];

    std::vector<double> line_out;
    resize_1d(line_in, line_out, p);

    for (int64_t i = 0; i < (int64_t)line_out.size(); ++i) out[out_off + i*out_strides[axis]] = line_out[i];
  }
}

} // namespace lsresize
