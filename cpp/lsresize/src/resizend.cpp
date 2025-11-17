// splineops/cpp/lsresize/src/resizend.cpp
#include "resizend.h"
#include "utils.h"
#include "resize1d.h"

#include <vector>
#include <numeric>
#include <cstdint>
#include <algorithm>
#include <cmath>     // std::abs
#include <cstdlib>   // std::getenv, std::atof, std::atoi
#include <cstring>   // std::memcpy
#include <type_traits>

namespace lsresize {

static std::vector<int64_t> strides_from_shape(const std::vector<int64_t>& shape) {
  std::vector<int64_t> s(shape.size(), 1);
  if (shape.empty()) return s;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    s[static_cast<size_t>(i)] =
        s[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return s;
}

static inline int64_t prod_elems(const std::vector<int64_t>& shape) {
  int64_t p = 1;
  for (int64_t v : shape) p *= v;
  return p;
}

// Simple heuristic to decide when to use OpenMP
static inline bool use_parallel(std::int64_t nlines, const lsresize::Plan1D& plan) {
  const double L    = static_cast<double>(plan.out_total);
  const double nnz  = plan.row_ptr.empty()
                    ? 0.0
                    : static_cast<double>(plan.row_ptr.back());
  const double wavg = (L > 0.0) ? (nnz / L) : 0.0;
  const double flops = 2.0 * static_cast<double>(nlines) * L * wavg;

  double thr = 1e6; // ~1M FLOPs by default
  if (const char* env = std::getenv("LSRESIZE_OMP_THRESHOLD")) {
    if (double t = std::atof(env); t > 0.0) thr = t;
  }
  // Use either the classic guard or the FLOPs-based one
  return (nlines > 64) || (flops > thr);
}

// -----------------------------------------------------------------------------
// Templated ND axis kernel over storage scalar (float or double).
// All internal computation (Plan1D, Work1D, filters) remains in double.
// -----------------------------------------------------------------------------
template <typename Scalar>
static void resize_along_axis_t(const Scalar* LS_RESTRICT in,
                                Scalar* LS_RESTRICT out,
                                const std::vector<int64_t>& in_shape,
                                const std::vector<int64_t>& out_shape,
                                int axis,
                                const LSParams& p)
{
  const int D = static_cast<int>(in_shape.size());
  const auto in_strides  = strides_from_shape(in_shape);
  const auto out_strides = strides_from_shape(out_shape);

  // Early identity short-circuit on this axis:
  {
    const double eps = 1e-12;
    const bool identity_axis =
        (out_shape[static_cast<size_t>(axis)] ==
         in_shape[static_cast<size_t>(axis)]) &&
        (std::abs(p.zoom - 1.0) <= eps) &&
        (p.analy_degree < 0); // Standard interpolation (no projection)
    if (identity_axis) {
      const int64_t total = prod_elems(in_shape); // in_shape == out_shape in this pass
      std::copy(in, in + total, out);             // Scalar -> Scalar
      return;
    }
  }

  // Total number of independent 1-D lines (all dims except 'axis')
  int64_t nlines = 1;
  for (int d = 0; d < D; ++d) {
    if (d != axis) nlines *= in_shape[static_cast<size_t>(d)];
  }

  // List non-axis dimensions (rightmost fastest)
  std::vector<int> bases;
  bases.reserve(D);
  for (int d = D - 1; d >= 0; --d) {
    if (d != axis) bases.push_back(d);
  }

  // Build the per-axis plan ONCE (shared read-only across threads)
  const int N_line = static_cast<int>(in_shape[static_cast<size_t>(axis)]);
  const Plan1D plan = make_plan_1d(N_line, p);

#if defined(_OPENMP)
  if (use_parallel(nlines, plan)) {
    // Parallel path
    #pragma omp parallel
    {
      // per-thread reusable workspace + helpers
      Work1D ws; // double internal
      std::vector<int64_t> idx(D, 0);
      std::vector<double>  line_in;
      std::vector<double>  line_out;
      line_in .reserve(static_cast<size_t>(N_line));
      line_out.reserve(static_cast<size_t>(plan.outN));

      // configurable chunk size (env: LSRESIZE_OMP_CHUNK)
      int chunk = 32;
      if (const char* e = std::getenv("LSRESIZE_OMP_CHUNK")) {
        if (int c = std::atoi(e); c > 0) chunk = c;
      }

      #pragma omp for schedule(static, chunk)
      for (int64_t line = 0; line < nlines; ++line) {
        std::fill(idx.begin(), idx.end(), 0);

        // unravel 'line' into coordinates for all dims except 'axis'
        int64_t t = line;
        for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
          const int d = bases[static_cast<size_t>(bi)];
          idx[static_cast<size_t>(d)] =
              t % in_shape[static_cast<size_t>(d)];
          t /= in_shape[static_cast<size_t>(d)];
        }

        // offsets at the start of this line
        int64_t in_off = 0, out_off = 0;
        for (int d = 0; d < D; ++d) {
          if (d != axis) {
            in_off  += idx[static_cast<size_t>(d)] *
                       in_strides[static_cast<size_t>(d)];
            out_off += idx[static_cast<size_t>(d)] *
                       out_strides[static_cast<size_t>(d)];
          }
        }

        // gather 1-D input line into double
        const bool contig_in =
            (in_strides[static_cast<size_t>(axis)] == 1);
        line_in.resize(static_cast<size_t>(N_line));
        if (contig_in) {
          // vector<double>::assign handles Scalar -> double conversion
          line_in.assign(in + in_off, in + in_off + N_line);
        } else {
          for (int64_t i = 0;
               i < in_shape[static_cast<size_t>(axis)];
               ++i) {
            line_in[static_cast<size_t>(i)] =
                static_cast<double>(
                    in[in_off +
                       i * in_strides[static_cast<size_t>(axis)]]);
          }
        }

        // fast planned path with workspace reuse (double internal)
        resize_1d_ws(line_in, line_out, p, plan, ws);

        // scatter to output (Scalar storage)
        const bool contig_out =
            (out_strides[static_cast<size_t>(axis)] == 1);
        if (contig_out) {
          if constexpr (std::is_same_v<Scalar, double>) {
            // One-shot block write when the axis is contiguous
            std::memcpy(out + out_off,
                        line_out.data(),
                        line_out.size() * sizeof(double));
          } else {
            for (size_t i = 0; i < line_out.size(); ++i) {
              out[out_off + static_cast<int64_t>(i)] =
                  static_cast<Scalar>(line_out[i]);
            }
          }
        } else {
          for (int64_t i = 0;
               i < static_cast<int64_t>(line_out.size());
               ++i) {
            out[out_off +
                i * out_strides[static_cast<size_t>(axis)]] =
                static_cast<Scalar>(
                    line_out[static_cast<size_t>(i)]);
          }
        }
      }
    }
    return;
  }
#endif

  // Serial path (or parallel disabled / small problem)
  {
    Work1D ws; // double internal
    std::vector<int64_t> idx(D, 0);
    std::vector<double>  line_in;
    std::vector<double>  line_out;
    line_in .reserve(static_cast<size_t>(N_line));
    line_out.reserve(static_cast<size_t>(plan.outN));

    for (int64_t line = 0; line < nlines; ++line) {
      std::fill(idx.begin(), idx.end(), 0);

      // unravel 'line' into coordinates for all dims except 'axis'
      int64_t t = line;
      for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
        const int d = bases[static_cast<size_t>(bi)];
        idx[static_cast<size_t>(d)] =
            t % in_shape[static_cast<size_t>(d)];
        t /= in_shape[static_cast<size_t>(d)];
      }

      // offsets at the start of this line
      int64_t in_off = 0, out_off = 0;
      for (int d = 0; d < D; ++d) {
        if (d != axis) {
          in_off  += idx[static_cast<size_t>(d)] *
                     in_strides[static_cast<size_t>(d)];
          out_off += idx[static_cast<size_t>(d)] *
                     out_strides[static_cast<size_t>(d)];
        }
      }

      // gather 1-D input line into double
      const bool contig_in =
          (in_strides[static_cast<size_t>(axis)] == 1);
      line_in.resize(static_cast<size_t>(N_line));
      if (contig_in) {
        line_in.assign(in + in_off, in + in_off + N_line);
      } else {
        for (int64_t i = 0;
             i < in_shape[static_cast<size_t>(axis)];
             ++i) {
          line_in[static_cast<size_t>(i)] =
              static_cast<double>(
                  in[in_off +
                     i * in_strides[static_cast<size_t>(axis)]]);
        }
      }

      // fast planned path with workspace reuse (double internal)
      resize_1d_ws(line_in, line_out, p, plan, ws);

      // scatter to output (Scalar storage)
      const bool contig_out =
          (out_strides[static_cast<size_t>(axis)] == 1);
      if (contig_out) {
        if constexpr (std::is_same_v<Scalar, double>) {
          std::memcpy(out + out_off,
                      line_out.data(),
                      line_out.size() * sizeof(double));
        } else {
          for (size_t i = 0; i < line_out.size(); ++i) {
            out[out_off + static_cast<int64_t>(i)] =
                static_cast<Scalar>(line_out[i]);
          }
        }
      } else {
        for (int64_t i = 0;
             i < static_cast<int64_t>(line_out.size());
             ++i) {
          out[out_off +
              i * out_strides[static_cast<size_t>(axis)]] =
              static_cast<Scalar>(
                  line_out[static_cast<size_t>(i)]);
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Public entry points
// -----------------------------------------------------------------------------

void resize_along_axis(const double* LS_RESTRICT in,
                       double* LS_RESTRICT out,
                       const std::vector<int64_t>& in_shape,
                       const std::vector<int64_t>& out_shape,
                       int axis,
                       const LSParams& p)
{
  resize_along_axis_t<double>(in, out, in_shape, out_shape, axis, p);
}

void resize_along_axis_f32(const float* LS_RESTRICT in,
                           float* LS_RESTRICT out,
                           const std::vector<int64_t>& in_shape,
                           const std::vector<int64_t>& out_shape,
                           int axis,
                           const LSParams& p)
{
  resize_along_axis_t<float>(in, out, in_shape, out_shape, axis, p);
}

} // namespace lsresize
