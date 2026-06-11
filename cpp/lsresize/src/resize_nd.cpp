// splineops/cpp/lsresize/src/resize_nd.cpp
#include "resize_nd.h"
#include "utils.h"
#include "parallel_utils.h"
#include "resize_1d.h"
#include "filters.h"

#include <vector>
#include <numeric>
#include <cstdint>
#include <algorithm>
#include <cmath>     // std::abs
#include <cstdlib>   // std::getenv, std::atof, std::atoi
#include <cstring>   // std::memcpy
#include <type_traits>

namespace lsresize {

static std::vector<int64_t> strides_from_shape(
  const std::vector<int64_t>& shape) 
{
  std::vector<int64_t> s(shape.size(), 1);
  if (shape.empty()) return s;
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
    s[static_cast<size_t>(i)] =
        s[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
  }
  return s;
}

static inline int64_t prod_elems(
  const std::vector<int64_t>& shape) 
{
  int64_t p = 1;
  for (int64_t v : shape) p *= v;
  return p;
}

static inline bool env_flag_enabled(const char* name)
{
  const char* value = std::getenv(name);
  return value != nullptr &&
         (value[0] == '1' || value[0] == 't' || value[0] == 'T' ||
          value[0] == 'y' || value[0] == 'Y');
}

static inline int env_int_or_default(const char* name, int fallback)
{
  if (const char* value = std::getenv(name)) {
    if (int parsed = std::atoi(value); parsed > 0) {
      return parsed;
    }
  }
  return fallback;
}

static inline void line_offsets(
  int64_t line,
  int axis,
  const std::vector<int>& bases,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& in_strides,
  const std::vector<int64_t>& out_strides,
  std::vector<int64_t>& idx,
  int64_t& in_off,
  int64_t& out_off)
{
  std::fill(idx.begin(), idx.end(), 0);

  int64_t t = line;
  for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
    const int d = bases[static_cast<size_t>(bi)];
    idx[static_cast<size_t>(d)] = t % in_shape[static_cast<size_t>(d)];
    t /= in_shape[static_cast<size_t>(d)];
  }

  in_off = 0;
  out_off = 0;
  for (int d = 0; d < static_cast<int>(idx.size()); ++d) {
    if (d != axis) {
      in_off  += idx[static_cast<size_t>(d)] *
                 in_strides[static_cast<size_t>(d)];
      out_off += idx[static_cast<size_t>(d)] *
                 out_strides[static_cast<size_t>(d)];
    }
  }
}

static double initial_causal_colmajor(
  const std::vector<double>& coeff,
  int B,
  int N,
  int b,
  double z,
  double tol = 1e-10)
{
  if (N == 0) return 0.0;
  if (N == 1) return coeff[static_cast<size_t>(b)];

  size_t horizon = static_cast<size_t>(N);
  if (tol > 0.0) {
    horizon = std::min(
        static_cast<size_t>(N),
        static_cast<size_t>(2 + std::log(tol) / std::log(std::abs(z))));
  }

  if (horizon < static_cast<size_t>(N)) {
    double sum = coeff[static_cast<size_t>(b)];
    double p = z;
    for (size_t n = 1; n < horizon; ++n) {
      sum += p * coeff[n * static_cast<size_t>(B) + static_cast<size_t>(b)];
      p *= z;
    }
    return sum;
  }

  const double zn = std::pow(z, double(N - 1));
  double sum = coeff[static_cast<size_t>(b)] +
               zn * coeff[static_cast<size_t>(N - 1) *
                          static_cast<size_t>(B) +
                          static_cast<size_t>(b)];
  double p1 = z;
  double p2 = (zn * zn) / z;
  for (int n = 1; n + 1 < N; ++n) {
    sum += (p1 + p2) *
           coeff[static_cast<size_t>(n) * static_cast<size_t>(B) +
                 static_cast<size_t>(b)];
    p1 *= z;
    p2 /= z;
  }

  return sum / (1.0 - (zn * zn));
}

static void get_interpolation_coefficients_colmajor(
  std::vector<double>& coeff,
  int B,
  int N,
  int deg)
{
  if (deg <= 1 || N <= 1 || B <= 0) return;

  const auto& poles = spline_poles(deg);
  double lambda = 1.0;
  for (double z : poles) {
    lambda *= (1.0 - z) * (1.0 - 1.0 / z);
  }
  for (double& v : coeff) {
    v *= lambda;
  }

  const size_t Bs = static_cast<size_t>(B);
  for (double z : poles) {
    for (int b = 0; b < B; ++b) {
      coeff[static_cast<size_t>(b)] =
          initial_causal_colmajor(coeff, B, N, b, z);
    }

    for (int n = 1; n < N; ++n) {
      double* cur = coeff.data() + static_cast<size_t>(n) * Bs;
      const double* prev = coeff.data() + static_cast<size_t>(n - 1) * Bs;
      for (int b = 0; b < B; ++b) {
        cur[static_cast<size_t>(b)] += z * prev[static_cast<size_t>(b)];
      }
    }

    double* last = coeff.data() + static_cast<size_t>(N - 1) * Bs;
    const double* before_last = coeff.data() + static_cast<size_t>(N - 2) * Bs;
    const double denom = z * z - 1.0;
    for (int b = 0; b < B; ++b) {
      last[static_cast<size_t>(b)] =
          (z * before_last[static_cast<size_t>(b)] +
           last[static_cast<size_t>(b)]) * z / denom;
    }

    for (int n = N - 2; n >= 0; --n) {
      double* cur = coeff.data() + static_cast<size_t>(n) * Bs;
      const double* next = coeff.data() + static_cast<size_t>(n + 1) * Bs;
      for (int b = 0; b < B; ++b) {
        cur[static_cast<size_t>(b)] =
            z * (next[static_cast<size_t>(b)] - cur[static_cast<size_t>(b)]);
      }
    }
  }
}

template <typename Scalar>
static void resize_along_axis_batched_interp_t(
  const Scalar* LS_RESTRICT in,
  Scalar* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  const std::vector<int64_t>& in_strides,
  const std::vector<int64_t>& out_strides,
  int axis,
  const LSParams& p,
  const Plan1D& plan,
  const std::vector<int>& bases,
  int64_t nlines)
{
  const int D = static_cast<int>(in_shape.size());
  const int N = plan.N;
  const int outN = plan.outN;
  const int length = plan.length_total;
  const int LP = plan.left_pad;
  const int RP = plan.right_pad;
  const int full_len = LP + length + RP;
  const int batch_lines = std::max(1, env_int_or_default("LSRESIZE_BATCH_LINES", 32));
  const bool axis_contig_in = (in_strides[static_cast<size_t>(axis)] == 1);
  const bool axis_contig_out = (out_strides[static_cast<size_t>(axis)] == 1);

  auto worker = [&](int64_t start, int64_t end) {
    std::vector<int64_t> idx(D, 0);
    std::vector<int64_t> in_offsets(static_cast<size_t>(batch_lines));
    std::vector<int64_t> out_offsets(static_cast<size_t>(batch_lines));
    std::vector<double> coeff;
    std::vector<double> ext_full;
    std::vector<double> y;
    coeff.reserve(static_cast<size_t>(N) * static_cast<size_t>(batch_lines));
    ext_full.reserve(static_cast<size_t>(full_len) * static_cast<size_t>(batch_lines));
    y.reserve(static_cast<size_t>(outN) * static_cast<size_t>(batch_lines));

    for (int64_t block = start; block < end; block += batch_lines) {
      const int B = static_cast<int>(std::min<int64_t>(batch_lines, end - block));
      const size_t Bs = static_cast<size_t>(B);
      coeff.assign(static_cast<size_t>(N) * Bs, 0.0);
      ext_full.assign(static_cast<size_t>(full_len) * Bs, 0.0);
      y.assign(static_cast<size_t>(outN) * Bs, 0.0);

      for (int b = 0; b < B; ++b) {
        int64_t in_off = 0;
        int64_t out_off = 0;
        line_offsets(
            block + b,
            axis,
            bases,
            in_shape,
            in_strides,
            out_strides,
            idx,
            in_off,
            out_off);
        in_offsets[static_cast<size_t>(b)] = in_off;
        out_offsets[static_cast<size_t>(b)] = out_off;

        if (axis_contig_in) {
          const Scalar* src = in + in_off;
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                static_cast<double>(src[static_cast<size_t>(n)]);
          }
        } else {
          const int64_t stride = in_strides[static_cast<size_t>(axis)];
          for (int n = 0; n < N; ++n) {
            coeff[static_cast<size_t>(n) * Bs + static_cast<size_t>(b)] =
                static_cast<double>(in[in_off + static_cast<int64_t>(n) * stride]);
          }
        }
      }

      get_interpolation_coefficients_colmajor(coeff, B, N, p.interp_degree);

      for (int i = 0; i < LP; ++i) {
        const int src = std::min(
            std::max(plan.pad_src_idx[static_cast<size_t>(i)], 0),
            std::max(0, N - 1));
        const double sgn = static_cast<int>(plan.pad_src_sgn[static_cast<size_t>(i)]);
        double* dst = ext_full.data() + static_cast<size_t>(i) * Bs;
        const double* src_col = coeff.data() + static_cast<size_t>(src) * Bs;
        for (int b = 0; b < B; ++b) {
          dst[static_cast<size_t>(b)] = sgn * src_col[static_cast<size_t>(b)];
        }
      }

      for (int n = 0; n < N; ++n) {
        double* dst = ext_full.data() + static_cast<size_t>(LP + n) * Bs;
        const double* src = coeff.data() + static_cast<size_t>(n) * Bs;
        std::copy(src, src + B, dst);
      }

      const int rem = length - N;
      if (rem > 0 && !plan.rp_src.empty()) {
        const double sgn = static_cast<int>(plan.rp_sign);
        for (int i = 0; i < rem; ++i) {
          const int src = plan.rp_src[static_cast<size_t>(i)];
          double* dst = ext_full.data() + static_cast<size_t>(LP + N + i) * Bs;
          const double* src_col = coeff.data() + static_cast<size_t>(src) * Bs;
          for (int b = 0; b < B; ++b) {
            dst[static_cast<size_t>(b)] = sgn * src_col[static_cast<size_t>(b)];
          }
        }
      }

      if (RP > 0) {
        const double* last = ext_full.data() + static_cast<size_t>(LP + length - 1) * Bs;
        for (int i = 0; i < RP; ++i) {
          double* dst = ext_full.data() + static_cast<size_t>(LP + length + i) * Bs;
          std::copy(last, last + B, dst);
        }
      }

      const int* rp = plan.row_ptr.data();
      const double* ww = plan.weights.data();
      for (int l = 0; l < outN; ++l) {
        const int begin = rp[static_cast<size_t>(l)];
        const int endw = rp[static_cast<size_t>(l) + 1];
        const int k0 = plan.kmin[static_cast<size_t>(l)];
        double* y_col = y.data() + static_cast<size_t>(l) * Bs;

        for (int t = begin; t < endw; ++t) {
          const double w = ww[static_cast<size_t>(t)];
          const int kt = k0 + (t - begin);
          const double* v = ext_full.data() + static_cast<size_t>(LP + kt) * Bs;
          for (int b = 0; b < B; ++b) {
            y_col[static_cast<size_t>(b)] += w * v[static_cast<size_t>(b)];
          }
        }
      }

      for (int b = 0; b < B; ++b) {
        const int64_t out_off = out_offsets[static_cast<size_t>(b)];
        if (axis_contig_out) {
          Scalar* dst = out + out_off;
          for (int l = 0; l < outN; ++l) {
            dst[static_cast<size_t>(l)] =
                static_cast<Scalar>(y[static_cast<size_t>(l) * Bs +
                                      static_cast<size_t>(b)]);
          }
        } else {
          const int64_t stride = out_strides[static_cast<size_t>(axis)];
          for (int l = 0; l < outN; ++l) {
            out[out_off + static_cast<int64_t>(l) * stride] =
                static_cast<Scalar>(y[static_cast<size_t>(l) * Bs +
                                      static_cast<size_t>(b)]);
          }
        }
      }
    }
  };

  run_parallel_or_serial(nlines, plan, worker);
}

// -----------------------------------------------------------------------------
// Templated ND axis kernel over storage scalar (float or double).
// All internal computation (Plan1D, Work1D, filters) remains in double.
// -----------------------------------------------------------------------------
template <typename Scalar>
static void resize_along_axis_t(
  const Scalar* LS_RESTRICT in,
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
    if (d != axis) {
      nlines *= in_shape[static_cast<size_t>(d)];
    }
  }

  // List non-axis dimensions (rightmost fastest)
  std::vector<int> bases;
  bases.reserve(D);
  for (int d = D - 1; d >= 0; --d) {
    if (d != axis) {
      bases.push_back(d);
    }
  }

  // Build the per-axis plan ONCE (shared read-only across threads)
  const int N_line = static_cast<int>(in_shape[static_cast<size_t>(axis)]);
  const Plan1D plan = make_plan_1d(N_line, p);

  if (env_flag_enabled("LSRESIZE_BATCHED_AXIS") && p.analy_degree < 0) {
    resize_along_axis_batched_interp_t(
        in,
        out,
        in_shape,
        out_shape,
        in_strides,
        out_strides,
        axis,
        p,
        plan,
        bases,
        nlines);
    return;
  }

  // Worker that processes a range of 1-D lines [start, end)
  auto worker = [&](int64_t start, int64_t end) {
    Work1D workspace; // per-thread reusable workspace (double internal)
    std::vector<int64_t> idx(D, 0);
    std::vector<double>  line_out;
    workspace.line.reserve(static_cast<size_t>(N_line));
    workspace.ext_full.reserve(static_cast<size_t>(plan.left_pad +
                                            plan.length_total +
                                            plan.right_pad));
    workspace.y.reserve(static_cast<size_t>(plan.out_total));
    line_out.reserve(static_cast<size_t>(plan.outN));

    const bool axis_contig_in  =
        (in_strides[static_cast<size_t>(axis)] == 1);
    const bool axis_contig_out =
        (out_strides[static_cast<size_t>(axis)] == 1);

    for (int64_t line = start; line < end; ++line) {
      std::fill(idx.begin(), idx.end(), 0);

      // Unravel 'line' into coordinates for all dims except 'axis'
      int64_t t = line;
      for (int bi = 0; bi < static_cast<int>(bases.size()); ++bi) {
        const int d = bases[static_cast<size_t>(bi)];
        idx[static_cast<size_t>(d)] =
            t % in_shape[static_cast<size_t>(d)];
        t /= in_shape[static_cast<size_t>(d)];
      }

      // Offsets at the start of this line
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

      // --- Fast path: axis contiguous in both in & out and Scalar == double ---
      if constexpr (std::is_same_v<Scalar, double>) {
        if (axis_contig_in && axis_contig_out) {
          // Direct 1-D resize on raw buffers, no gather/scatter via vectors.
          resize_1d_line_contiguous(in + in_off, out + out_off, p, plan, workspace);
          continue;
        }
      }

      // --- Fallback: gather into workspace.line (double), run 1-D core from line ---

      workspace.line.resize(static_cast<size_t>(N_line));

      if (axis_contig_in) {
        // contiguous axis: simple block copy with cast
        const Scalar* in_line = in + in_off;
        for (int i = 0; i < N_line; ++i) {
          workspace.line[static_cast<size_t>(i)] =
            static_cast<double>(in_line[static_cast<size_t>(i)]);
        }
      } else {
        // non-contiguous axis: strided gather with cast
        const int64_t stride = in_strides[static_cast<size_t>(axis)];
        for (int i = 0; i < N_line; ++i) {
          workspace.line[static_cast<size_t>(i)] =
              static_cast<double>(
                  in[in_off +
                     static_cast<int64_t>(i) * stride]);
        }
      }

      // Fast planned path with workspace reuse (double internal)
      resize_1d_line_buffered(workspace.line, line_out, p, plan, workspace);

      // Scatter to output (Scalar storage)
      const bool contig_out = axis_contig_out;
      if (contig_out) {
        if constexpr (std::is_same_v<Scalar, double>) {
          // One-shot block write when the axis is contiguous
          std::memcpy(out + out_off,
                      line_out.data(),
                      line_out.size() * sizeof(double));
        } else {
          for (size_t i = 0; i < line_out.size(); ++i) {
            out[out_off + static_cast<int64_t>(i)] =
                static_cast<Scalar>(line_out[static_cast<size_t>(i)]);
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
  };

  // Centralized scheduling: OpenMP, std::thread, or serial
  run_parallel_or_serial(nlines, plan, worker);
}

// -----------------------------------------------------------------------------
// Public entry points
// -----------------------------------------------------------------------------

void resize_along_axis(
  const double* LS_RESTRICT in,
  double* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const LSParams& p)
{
  resize_along_axis_t<double>(in, out, in_shape, out_shape, axis, p);
}

void resize_along_axis_f32(
  const float* LS_RESTRICT in,
  float* LS_RESTRICT out,
  const std::vector<int64_t>& in_shape,
  const std::vector<int64_t>& out_shape,
  int axis,
  const LSParams& p)
{
  resize_along_axis_t<float>(in, out, in_shape, out_shape, axis, p);
}

} // namespace lsresize
