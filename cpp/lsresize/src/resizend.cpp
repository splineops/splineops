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
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    s[static_cast<size_t>(i)] = s[static_cast<size_t>(i+1)] * shape[static_cast<size_t>(i+1)];
  }
  return s;
}

void resize_along_axis(const double* LS_RESTRICT in, double* LS_RESTRICT out,
                       const std::vector<int64_t>& in_shape,
                       const std::vector<int64_t>& out_shape,
                       int axis,
                       const LSParams& p)
{
  const int D = static_cast<int>(in_shape.size());
  const auto in_strides  = strides_from_shape(in_shape);
  const auto out_strides = strides_from_shape(out_shape);

  // total number of independent 1-D lines (all dims except 'axis')
  int64_t nlines = 1;
  for (int d = 0; d < D; ++d) if (d != axis) nlines *= in_shape[static_cast<size_t>(d)];

  // list non-axis dimensions (rightmost fastest)
  std::vector<int> bases;
  bases.reserve(D);
  for (int d = D - 1; d >= 0; --d) {
    if (d != axis) bases.push_back(d);
  }

#if defined(_OPENMP)
  if (nlines > 64) {
    // Parallel path
    #pragma omp parallel
    {
      // per-thread reusable buffers
      std::vector<int64_t> idx(D, 0);
      std::vector<double>  line_in;  line_in.reserve(static_cast<size_t>(in_shape[static_cast<size_t>(axis)]));
      std::vector<double>  line_out; line_out.reserve(static_cast<size_t>(out_shape[static_cast<size_t>(axis)]));

      #pragma omp for schedule(static, 32)
      for (int64_t line = 0; line < nlines; ++line) {
        std::fill(idx.begin(), idx.end(), 0);

        // unravel 'line' into coordinates for all dims except 'axis'
        int64_t t = line;
        for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
          const int d = bases[static_cast<size_t>(bi)];
          idx[static_cast<size_t>(d)] = t % in_shape[static_cast<size_t>(d)];
          t                           /= in_shape[static_cast<size_t>(d)];
        }

        // offsets at the start of this line
        int64_t in_off = 0, out_off = 0;
        for (int d = 0; d < D; ++d) if (d != axis) {
          in_off  += idx[static_cast<size_t>(d)] * in_strides[static_cast<size_t>(d)];
          out_off += idx[static_cast<size_t>(d)] * out_strides[static_cast<size_t>(d)];
        }

        // gather 1-D input line
        line_in.resize(static_cast<size_t>(in_shape[static_cast<size_t>(axis)]));
        for (int64_t i = 0; i < in_shape[static_cast<size_t>(axis)]; ++i) {
          line_in[static_cast<size_t>(i)] = in[in_off + i * in_strides[static_cast<size_t>(axis)]];
        }

        // resize the 1-D line
        resize_1d(line_in, line_out, p);

        // scatter to output
        for (int64_t i = 0; i < static_cast<int64_t>(line_out.size()); ++i) {
          out[out_off + i * out_strides[static_cast<size_t>(axis)]] = line_out[static_cast<size_t>(i)];
        }
      }
    }
    return;
  }
#endif

  // Serial path (or parallel disabled / small problem)
  {
    std::vector<int64_t> idx(D, 0);
    std::vector<double>  line_in;  line_in.reserve(static_cast<size_t>(in_shape[static_cast<size_t>(axis)]));
    std::vector<double>  line_out; line_out.reserve(static_cast<size_t>(out_shape[static_cast<size_t>(axis)]));

    for (int64_t line = 0; line < nlines; ++line) {
      std::fill(idx.begin(), idx.end(), 0);

      // unravel 'line' into coordinates for all dims except 'axis'
      int64_t t = line;
      for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
        const int d = bases[static_cast<size_t>(bi)];
        idx[static_cast<size_t>(d)] = t % in_shape[static_cast<size_t>(d)];
        t                           /= in_shape[static_cast<size_t>(d)];
      }

      // offsets at the start of this line
      int64_t in_off = 0, out_off = 0;
      for (int d = 0; d < D; ++d) if (d != axis) {
        in_off  += idx[static_cast<size_t>(d)] * in_strides[static_cast<size_t>(d)];
        out_off += idx[static_cast<size_t>(d)] * out_strides[static_cast<size_t>(d)];
      }

      // gather 1-D input line
      line_in.resize(static_cast<size_t>(in_shape[static_cast<size_t>(axis)]));
      for (int64_t i = 0; i < in_shape[static_cast<size_t>(axis)]; ++i) {
        line_in[static_cast<size_t>(i)] = in[in_off + i * in_strides[static_cast<size_t>(axis)]];
      }

      // resize the 1-D line
      resize_1d(line_in, line_out, p);

      // scatter to output
      for (int64_t i = 0; i < static_cast<int64_t>(line_out.size()); ++i) {
        out[out_off + i * out_strides[static_cast<size_t>(axis)]] = line_out[static_cast<size_t>(i)];
      }
    }
  }
}

} // namespace lsresize
