// splineops/cpp/lsresize/src/resize_1d.cpp
#include "resize_1d.h"
#include "bspline.h"
#include "filters.h"
#include "utils.h"
#include "dot_kernels.h"

#include <algorithm>
#include <cmath>
#include <vector>

// 1D resizing pipeline layout
//
//   Public entry points (see resize_1d.h):
//
//     - resize_1d_workspace(in_vec, out_vec, params, plan, workspace)
//     - resize_1d_line_contiguous(in_ptr, out_ptr, params, plan, workspace)
//     - resize_1d_line_buffered(line_buf, out_vec, params, plan, workspace)
//
// All three delegate to a single internal pipeline:
//
//   run_pipeline_from_line(out_ptr, params, plan,
//                          line_buffer, ext_full, y)
//
// The only difference between the public entry points is how the input
// samples are fed into the line buffer:
//
//   - resize_1d_line_contiguous: copies from a raw contiguous double* line
//   - resize_1d_workspace:       copies from a std::vector<double>
//   - resize_1d_line_buffered:   caller has already filled the line buffer

namespace lsresize {

// Build the reusable 1-D plan (window metadata + contiguous weights + pad map)
Plan1D make_plan_1d(int N, const LSParams& p)
{
  Plan1D plan{};
  plan.N = N;

  // Output size (same as before)
  int workN = 0, outN = 0;
  calculate_final_size_1d(p.inversable, N, p.zoom, workN, outN);
  plan.outN = outN;

  const bool pure_interp = (p.analy_degree < 0);

  // total_degree controls the spline support used in the windows
  const int total_degree = p.interp_degree + p.analy_degree + 1;

  // Correction degree for LS / oblique projection
  const int corr_degree = pure_interp
                        ? p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);

  // Tail length / out_total
  //  - Pure interpolation: no projection tail, only outN samples
  //  - LS / oblique: keep original border-based tail
  int add_border = 0;
  if (!pure_interp) {
    add_border = std::max(border(outN, corr_degree), total_degree);
  }
  plan.out_total = outN + add_border;

  // Shift:
  //  - Interpolation uses p.shift as-is
  //  - Projection adds the Muñoz correction
  double shift = p.shift;
  if (!pure_interp) {
    const double t = (p.analy_degree + 1.0) / 2.0;
    shift += (t - std::floor(t)) * (1.0 / p.zoom - 1.0);
  }

  // Symmetric (even) vs antisymmetric (odd) boundary
  plan.symmetric_ext = ((p.analy_degree + 1) % 2 == 0);

  const double half_support = 0.5 * (total_degree + 1);

  // Zoom exponent for LS / oblique (Unser–Muñoz step 3 factor)
  const double fact = std::pow(
      p.zoom,
      (p.analy_degree >= 0) ? (p.analy_degree + 1) : 0
  );

  // Extended input length:
  //  - Interpolation: only need a small mirror tail up to the spline support.
  //  - LS / oblique: original LS sizing using add_border/zoom.
  if (pure_interp) {
    const int right_ext = static_cast<int>(std::ceil(half_support));
    plan.length_total   = N + right_ext;
  } else {
    plan.length_total   = N + static_cast<int>(std::ceil(add_border / p.zoom));
  }

  // CSR-style window metadata
  plan.row_ptr.resize(static_cast<size_t>(plan.out_total) + 1);
  plan.kmin   .resize(static_cast<size_t>(plan.out_total));
  plan.win_len.resize(static_cast<size_t>(plan.out_total));

  int nnz      = 0;
  int min_kmin =  0;
  int max_kmax = -1;

  // Unified TensorSpline-style geometry for ALL methods:
  //
  //   - Input samples at k = 0 .. N-1
  //   - Visible outputs (0 .. outN-1) span [0, N-1]
  //     => step = (N-1)/(outN-1) when outN > 1
  //   - Tail samples (l >= outN) simply continue with the same step.
  const double step = (plan.outN > 1)
                    ? (static_cast<double>(N - 1) /
                       static_cast<double>(plan.outN - 1))
                    : 0.0;

  // First pass: compute (kmin, kmax) per row, nnz, global min/max
  for (int l = 0; l < plan.out_total; ++l) {
    const double x = step * static_cast<double>(l) + shift;

    const int kmin = static_cast<int>(std::ceil (x - half_support));
    const int kmax = static_cast<int>(std::floor(x + half_support));
    const int wlen = kmax - kmin + 1;

    plan.kmin   [static_cast<size_t>(l)] = kmin;
    plan.win_len[static_cast<size_t>(l)] = wlen;
    nnz += wlen;

    if (kmin < min_kmin) min_kmin = kmin;
    if (kmax > max_kmax) max_kmax = kmax;
  }

  // Global pads to build a single contiguous extended buffer: [LP | ext | RP]
  plan.left_pad  = std::max(0, -min_kmin);
  plan.right_pad = std::max(0,  max_kmax - (plan.length_total - 1));

  // Precompute left-pad mapping for negative indices: -t -> sign * line[src]
  plan.pad_src_idx.resize(static_cast<size_t>(plan.left_pad));
  plan.pad_src_sgn.resize(static_cast<size_t>(plan.left_pad), 1);
  for (int t = 1; t <= plan.left_pad; ++t) {
    const int pos = plan.left_pad - t; // 0 .. left_pad-1
    if (plan.symmetric_ext) {
      // symmetric: -t -> +line[t]
      plan.pad_src_idx[static_cast<size_t>(pos)] = t;   // clamped later to [0, N-1]
      plan.pad_src_sgn[static_cast<size_t>(pos)] =  1;
    } else {
      // antisymmetric: -t -> -line[t-1]
      plan.pad_src_idx[static_cast<size_t>(pos)] = t - 1;
      plan.pad_src_sgn[static_cast<size_t>(pos)] = -1;
    }
  }

  // Allocate contiguous weights (sign handled via extension, not weights)
  plan.weights.resize(static_cast<size_t>(nnz));

  // Second pass: fill row_ptr and weights
  int cursor = 0;
  for (int l = 0; l < plan.out_total; ++l) {
    plan.row_ptr[static_cast<size_t>(l)] = cursor;

    const double x = step * static_cast<double>(l) + shift;
    const int    k0   = plan.kmin   [static_cast<size_t>(l)];
    const int    wlen = plan.win_len[static_cast<size_t>(l)];

    for (int t = 0; t < wlen; ++t) {
      const int k = k0 + t;
      const double w = fact * beta(x - k, total_degree);
      plan.weights[static_cast<size_t>(cursor++)] = w;
    }
  }
  plan.row_ptr.back() = cursor;

  // --- Precompute right extension mapping (mirrored indices) ---
  {
    const int rem = plan.length_total - N;
    plan.rp_src.clear();
    plan.rp_sign = plan.symmetric_ext ?  1 : -1;

    if (rem > 0) {
      plan.rp_src.resize(static_cast<size_t>(rem));
      if (plan.symmetric_ext) {
        const int period = 2 * N - 2;
        for (int l = N; l < plan.length_total; ++l) {
          int t = l;
          if (period > 0 && t >= period) t %= period;
          if (t >= N) t = period - t;
          if (t < 0) t = 0; else if (t >= N) t = N - 1;
          plan.rp_src[static_cast<size_t>(l - N)] = t;
        }
      } else { // antisymmetric
        const int period = 2 * N - 3;
        for (int l = N; l < plan.length_total; ++l) {
          int t = l;
          if (period > 0 && t >= period) t %= period;
          if (t >= N) t = period - t;
          if (t < 0) t = 0; else if (t >= N) t = N - 1;
          plan.rp_src[static_cast<size_t>(l - N)] = t;
        }
      }
    }
  }

  return plan;
}

// -----------------------------------------------------------------------------
// Internal 1-D cores
// -----------------------------------------------------------------------------

// Core that assumes `line[0..N-1]` is already filled with input samples.
// This lets callers (like the ND kernel) avoid an extra copy into line.
static inline void run_pipeline_from_line(
  double* out,
  const LSParams& p,
  const Plan1D& plan,
  std::vector<double>& line,
  std::vector<double>& ext_full,
  std::vector<double>& y)
{
  const int N = plan.N;
  if (N == 0) {
    return;
  }

  const int corr_degree = (p.analy_degree < 0)
                        ? p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);

  // 1) Interpolation coefficients (causal/anti-causal IIR on input), in-place.
  get_interpolation_coefficients(line, p.interp_degree);

  // 2) Optional projection integration
  double average = 0.0;
  if (p.analy_degree >= 0) {
    average = do_integ(line, p.analy_degree + 1);
  }

  // 3) Single padded buffer [LP | ext | RP], built directly from line
  const int LP     = plan.left_pad;
  const int length = plan.length_total;
  const int RP     = plan.right_pad;

  ext_full.resize(static_cast<size_t>(LP + length + RP));
  double* dst = ext_full.data();

  // 3a) Left pad using the precomputed mapping
  if (LP > 0) {
    for (int i = 0; i < LP; ++i) {
      const int src = std::min(
          std::max(plan.pad_src_idx[static_cast<size_t>(i)], 0),
          std::max(0, N - 1));
      const int sgn = static_cast<int>(plan.pad_src_sgn[static_cast<size_t>(i)]);
      dst[static_cast<size_t>(i)] = sgn * line[static_cast<size_t>(src)];
    }
  }

  // 3b) Main input samples
  std::copy(line.begin(), line.end(), dst + LP);

  // 3c) Right extension into the middle block
  const int rem = length - N;
  if (rem > 0 && !plan.rp_src.empty()) {
    const double sgn = static_cast<int>(plan.rp_sign);
    double* tail = dst + LP + N;
    for (int i = 0; i < rem; ++i) {
      const int src = plan.rp_src[static_cast<size_t>(i)];
      tail[static_cast<size_t>(i)] = sgn * line[static_cast<size_t>(src)];
    }
  }

  // 3d) Right pad (clamp to last sample)
  if (RP > 0) {
    const double last = dst[LP + length - 1];
    std::fill(dst + LP + length, dst + LP + length + RP, last);
  }

  // 4) Accumulate using the plan (contiguous weights & samples)
  y.resize(static_cast<size_t>(plan.out_total));
  {
    const int*    rp = plan.row_ptr.data();
    const double* ww = plan.weights.data();
    const double* vf = ext_full.data();

    for (int l = 0; l < plan.out_total; ++l) {
      const int begin = rp[static_cast<size_t>(l)];
      const int end   = rp[static_cast<size_t>(l) + 1];
      const int M     = end - begin;
      const int k0    = plan.kmin[static_cast<size_t>(l)];

      const double* w = ww + begin;
      const double* v = vf + (LP + k0);

      y[static_cast<size_t>(l)] = dot_small(w, v, M);
    }
  }

  // 5) Projection tail (unchanged)
  if (p.analy_degree >= 0) {
    do_diff(y, p.analy_degree + 1);
    for (int i = 0; i < plan.out_total; ++i) {
      y[static_cast<size_t>(i)] += average;
    }
    get_interpolation_coefficients(y, corr_degree);
    get_samples(y, p.synthe_degree);
  }

  // 6) Copy to true output size
  std::copy(y.begin(), y.begin() + plan.outN, out);
}

// Raw-pointer core: operates directly on in/out buffers using workspace vectors.
// This version still accepts `in` and copies it into `line` once, then
// delegates to run_pipeline_from_line.
static inline void run_pipeline_from_raw(
  const double* in_samples,
  double* out_samples,
  const LSParams& p,
  const Plan1D& plan,
  std::vector<double>& line,
  std::vector<double>& ext_full,
  std::vector<double>& y)
{
  const int N = plan.N;
  if (N == 0) return;

  line.resize(static_cast<size_t>(N));
  std::copy(in_samples, in_samples + N, line.begin());

  run_pipeline_from_line(out_samples, p, plan,
                         line, ext_full, y);
}

// Old vector API now just wraps the raw core.
static inline void resize_1d_core(
  const std::vector<double>& in,
  std::vector<double>& out,
  const LSParams& p,
  const Plan1D& plan,
  std::vector<double>& line,
  std::vector<double>& ext_full,
  std::vector<double>& y)
{
  const int N = plan.N;
  if (N == 0) {
    out.clear();
    return;
  }
  out.resize(static_cast<size_t>(plan.outN));
  run_pipeline_from_raw(in.data(), out.data(), p, plan, line, ext_full, y);
}

// -----------------------------------------------------------------------------
// Public, allocation-free wrappers
// -----------------------------------------------------------------------------

void resize_1d_workspace(
  const std::vector<double>& in,
  std::vector<double>& out,
  const LSParams& p,
  const Plan1D& plan,
  Work1D& workspace)
{
  resize_1d_core(in, out, p, plan, workspace.line, workspace.ext_full, workspace.y);
}

// Raw-pointer wrapper for contiguous lines (no std::vector in/out).
void resize_1d_line_contiguous(
  const double* in,
  double* out,
  const LSParams& p,
  const Plan1D& plan,
  Work1D& workspace)
{
  run_pipeline_from_raw(in, out, p, plan, workspace.line, workspace.ext_full, workspace.y);
}

// Wrapper used by the ND kernel when it has already filled `line`.
// Avoids an extra copy from a temporary into workspace.line.
void resize_1d_line_buffered(
  std::vector<double>& line,
  std::vector<double>& out,
  const LSParams& p,
  const Plan1D& plan,
  Work1D& workspace)
{
  const int N = plan.N;
  if (N == 0) {
    out.clear();
    return;
  }
  out.resize(static_cast<size_t>(plan.outN));
  run_pipeline_from_line(out.data(), p, plan, line, workspace.ext_full, workspace.y);
}

} // namespace lsresize
