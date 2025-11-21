// splineops/cpp/lsresize/src/resize_nd.cpp
#include "resize_nd.h"
#include "utils.h"
#include "parallel_utils.h"
#include "resize_1d.h"

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

static inline bool is_least_squares(const LSParams& p) {
  // LS: analy_degree >= 1 and interp == analy == synthe
  return (p.analy_degree >= 1) &&
         (p.analy_degree == p.interp_degree) &&
         (p.analy_degree == p.synthe_degree);
}

// LS path for float32 storage: internal math in double (Plan1D/Work1D).
static void resize_along_axis_ls_f32(const float* LS_RESTRICT in,
                                     float* LS_RESTRICT out,
                                     const std::vector<int64_t>& in_shape,
                                     const std::vector<int64_t>& out_shape,
                                     int axis,
                                     const LSParams& p)
{
  const int D = static_cast<int>(in_shape.size());
  const auto in_strides  = strides_from_shape(in_shape);
  const auto out_strides = strides_from_shape(out_shape);

  // Identity short-circuit (same as in resize_along_axis_t)
  {
    const double eps = 1e-12;
    const bool identity_axis =
        (out_shape[static_cast<size_t>(axis)] ==
         in_shape[static_cast<size_t>(axis)]) &&
        (std::abs(p.zoom - 1.0) <= eps) &&
        (p.analy_degree < 0);
    if (identity_axis) {
      const int64_t total = prod_elems(in_shape);
      std::copy(in, in + total, out);
      return;
    }
  }

  // Number of independent lines
  int64_t nlines = 1;
  for (int d = 0; d < D; ++d) {
    if (d != axis) {
      nlines *= in_shape[static_cast<size_t>(d)];
    }
  }

  // Non-axis dimensions, rightmost fastest
  std::vector<int> bases;
  bases.reserve(D);
  for (int d = D - 1; d >= 0; --d) {
    if (d != axis) {
      bases.push_back(d);
    }
  }

  const int N_line = static_cast<int>(in_shape[static_cast<size_t>(axis)]);

  // Double-precision plan / workspace
  Plan1D plan = make_plan_1d(N_line, p);
  auto worker = [&](int64_t start, int64_t end) {
    Work1D ws;
    std::vector<int64_t> idx(D, 0);
    std::vector<double>  line_in;
    std::vector<double>  line_out;
    line_in .reserve(static_cast<size_t>(N_line));
    line_out.reserve(static_cast<size_t>(plan.outN));

    const bool axis_contig_in  =
        (in_strides[static_cast<size_t>(axis)] == 1);
    const bool axis_contig_out =
        (out_strides[static_cast<size_t>(axis)] == 1);

    for (int64_t line = start; line < end; ++line) {
      std::fill(idx.begin(), idx.end(), 0);

      int64_t t = line;
      for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
        const int d = bases[static_cast<size_t>(bi)];
        idx[static_cast<size_t>(d)] =
            t % in_shape[static_cast<size_t>(d)];
        t /= in_shape[static_cast<size_t>(d)];
      }

      int64_t in_off  = 0;
      int64_t out_off = 0;
      for (int d = 0; d < D; ++d) {
        if (d != axis) {
          in_off  += idx[static_cast<size_t>(d)] *
                     in_strides[static_cast<size_t>(d)];
          out_off += idx[static_cast<size_t>(d)] *
                     out_strides[static_cast<size_t>(d)];
        }
      }

      // --- Gather float32 -> double ---
      line_in.resize(static_cast<size_t>(N_line));
      for (int64_t i = 0;
           i < in_shape[static_cast<size_t>(axis)];
           ++i) {
        line_in[static_cast<size_t>(i)] =
            static_cast<double>(
                in[in_off +
                   i * in_strides[static_cast<size_t>(axis)]]);
      }

      // Double-precision 1D resize
      resize_1d_ws(line_in, line_out, p, plan, ws);

      // --- Scatter double -> float32 ---
      if (axis_contig_out) {
        for (int64_t i = 0;
             i < static_cast<int64_t>(line_out.size());
             ++i) {
          out[out_off + i] =
              static_cast<float>(line_out[static_cast<size_t>(i)]);
        }
      } else {
        for (int64_t i = 0;
             i < static_cast<int64_t>(line_out.size());
             ++i) {
          out[out_off +
              i * out_strides[static_cast<size_t>(axis)]] =
              static_cast<float>(line_out[static_cast<size_t>(i)]);
        }
      }
    }
  };

  // Use double as the "Real" type for parallel scheduling
  run_parallel_or_serial<double>(nlines, plan, worker);
}

// -----------------------------------------------------------------------------
// Templated ND axis kernel over storage scalar (float or double).
// Internal math uses the same scalar type as storage (float32 or float64).
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

  {
    const double eps = 1e-12;
    const bool identity_axis =
        (out_shape[static_cast<size_t>(axis)] ==
         in_shape[static_cast<size_t>(axis)]) &&
        (std::abs(p.zoom - 1.0) <= eps) &&
        (p.analy_degree < 0);
    if (identity_axis) {
      const int64_t total = prod_elems(in_shape);
      std::copy(in, in + total, out);
      return;
    }
  }

  int64_t nlines = 1;
  for (int d = 0; d < D; ++d) {
    if (d != axis) {
      nlines *= in_shape[static_cast<size_t>(d)];
    }
  }

  std::vector<int> bases;
  bases.reserve(D);
  for (int d = D - 1; d >= 0; --d) {
    if (d != axis) {
      bases.push_back(d);
    }
  }

  const int N_line = static_cast<int>(in_shape[static_cast<size_t>(axis)]);

  // Choose plan & workspace type based on Scalar
  using PlanT = std::conditional_t<
      std::is_same_v<Scalar, float>,
      Plan1Df,
      Plan1D
  >;
  using WorkT = std::conditional_t<
      std::is_same_v<Scalar, float>,
      Work1Df,
      Work1D
  >;

  PlanT plan = [&]() {
    if constexpr (std::is_same_v<Scalar, float>) {
      return make_plan_1d_f32(N_line, p);
    } else {
      return make_plan_1d(N_line, p);
    }
  }();

  auto worker = [&](int64_t start, int64_t end) {
    WorkT ws;
    std::vector<int64_t> idx(D, 0);
    std::vector<Scalar>  line_in;
    std::vector<Scalar>  line_out;
    line_in .reserve(static_cast<size_t>(N_line));
    line_out.reserve(static_cast<size_t>(plan.outN));

    const bool axis_contig_in  =
        (in_strides[static_cast<size_t>(axis)] == 1);
    const bool axis_contig_out =
        (out_strides[static_cast<size_t>(axis)] == 1);

    for (int64_t line = start; line < end; ++line) {
      std::fill(idx.begin(), idx.end(), 0);

      int64_t t = line;
      for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
        const int d = bases[static_cast<size_t>(bi)];
        idx[static_cast<size_t>(d)] =
            t % in_shape[static_cast<size_t>(d)];
        t /= in_shape[static_cast<size_t>(d)];
      }

      int64_t in_off  = 0;
      int64_t out_off = 0;
      for (int d = 0; d < D; ++d) {
        if (d != axis) {
          in_off  += idx[static_cast<size_t>(d)] *
                     in_strides[static_cast<size_t>(d)];
          out_off += idx[static_cast<size_t>(d)] *
                     out_strides[static_cast<size_t>(d)];
        }
      }

      // --- Fast path: axis contiguous in both in & out ---
      if (axis_contig_in && axis_contig_out) {
        if constexpr (std::is_same_v<Scalar, double>) {
          resize_1d_ws_raw(in + in_off,
                           out + out_off,
                           p, plan, ws);
          continue;
        } else {
          resize_1d_ws_raw_f32(in + in_off,
                               out + out_off,
                               p, plan, ws);
          continue;
        }
      }

      // --- Fallback: gather into line_in, use vector API ---
      line_in.resize(static_cast<size_t>(N_line));
      if (axis_contig_in) {
        line_in.assign(in + in_off, in + in_off + N_line);
      } else {
        for (int64_t i = 0;
             i < in_shape[static_cast<size_t>(axis)];
             ++i) {
          line_in[static_cast<size_t>(i)] =
              in[in_off +
                 i * in_strides[static_cast<size_t>(axis)]];
        }
      }

      if constexpr (std::is_same_v<Scalar, double>) {
        resize_1d_ws(line_in, line_out, p, plan, ws);
      } else {
        resize_1d_ws_f32(line_in, line_out, p, plan, ws);
      }

      if (axis_contig_out) {
        std::memcpy(out + out_off,
                    line_out.data(),
                    line_out.size() * sizeof(Scalar));
      } else {
        for (int64_t i = 0;
             i < static_cast<int64_t>(line_out.size());
             ++i) {
          out[out_off +
              i * out_strides[static_cast<size_t>(axis)]] =
              line_out[static_cast<size_t>(i)];
        }
      }
    }
  };

  run_parallel_or_serial<Scalar>(nlines, plan, worker);
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
  if (is_least_squares(p)) {
    // LS: always use double internal math
    resize_along_axis_ls_f32(in, out, in_shape, out_shape, axis, p);
  } else {
    // Standard / Oblique: float32 internal math is fine
    resize_along_axis_t<float>(in, out, in_shape, out_shape, axis, p);
  }
}

} // namespace lsresize
