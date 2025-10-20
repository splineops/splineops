// splineops/cpp/lsresize/src/resizend.cpp
#include "resizend.h"
#include "utils.h"

#include <vector>
#include <numeric>
#include <cstdint>
#include <algorithm>

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

  // total number of independent 1-D lines (all dims except 'axis')
  int64_t nlines = 1;
  for (int d = 0; d < D; ++d) if (d != axis) nlines *= in_shape[d];

  // list non-axis dimensions (rightmost fastest)
  std::vector<int> bases;
  bases.reserve(D);
  for (int d = D - 1; d >= 0; --d) {
    if (d != axis) bases.push_back(d);
  }

  // Parallelize over lines if problem is big enough
  // (Pragmas are guarded so this compiles fine without OpenMP too.)
  #if defined(_OPENMP)
  #pragma omp parallel for if(nlines > 64) schedule(static)
  #endif
  for (int64_t line = 0; line < nlines; ++line) {
    // thread-local index buffer (prevents races)
    std::vector<int64_t> idx(D, 0);

    // unravel 'line' into coordinates for all dims except 'axis'
    int64_t t = line;
    for (int bi = 0; bi < (int)bases.size(); ++bi) {
      const int d = bases[bi];
      idx[d] = t % in_shape[d];
      t     /= in_shape[d];
    }

    // offsets at the start of this line
    int64_t in_off = 0, out_off = 0;
    for (int d = 0; d < D; ++d) if (d != axis) {
      in_off  += idx[d] * in_strides[d];
      out_off += idx[d] * out_strides[d];
    }

    // gather 1-D input line
    std::vector<double> line_in((size_t)in_shape[axis]);
    for (int64_t i = 0; i < in_shape[axis]; ++i) {
      line_in[i] = in[in_off + i * in_strides[axis]];
    }

    // resize the 1-D line
    std::vector<double> line_out;
    resize_1d(line_in, line_out, p);

    // scatter to output
    for (int64_t i = 0; i < (int64_t)line_out.size(); ++i) {
      out[out_off + i * out_strides[axis]] = line_out[i];
    }
  }
}

} // namespace lsresize
