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
