// splineops/cpp/lsresize/src/resize1d.cpp
#include "resize1d.h"
#include "bspline.h"
#include "filters.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace lsresize {

// Build the reusable 1-D plan (window metadata + contiguous weights)
Plan1D make_plan_1d(int N, const LSParams& p)
{
  Plan1D plan{};
  plan.N = N;

  // Output and tail sizing
  int workN = 0, outN = 0;
  calculate_final_size_1d(p.inversable, N, p.zoom, workN, outN);
  plan.outN = outN;

  const int total_degree = p.interp_degree + p.analy_degree + 1;
  const int corr_degree  = (p.analy_degree < 0)
                         ?  p.interp_degree
                         : (p.analy_degree + p.synthe_degree + 1);

  const int add_border   = std::max(border(outN, corr_degree), total_degree);
  plan.out_total         = outN + add_border;

  // Center shift (matches Python path)
  double shift = p.shift;
  if (p.analy_degree >= 0) {
    const double t = (p.analy_degree + 1.0) / 2.0;
    shift += (t - std::floor(t)) * (1.0 / p.zoom - 1.0);
  }

  plan.symmetric_ext = ((p.analy_degree + 1) % 2 == 0);
  plan.length_total  = N + static_cast<int>(std::ceil(add_border / p.zoom));

  // Precompute window metadata and contiguous weights (CSR-like)
  const double half_support = 0.5 * (total_degree + 1);
  const double fact         = std::pow(p.zoom, (p.analy_degree >= 0) ? (p.analy_degree + 1) : 0);

  plan.row_ptr.resize(static_cast<size_t>(plan.out_total) + 1);
  plan.kmin   .resize(static_cast<size_t>(plan.out_total));
  plan.win_len.resize(static_cast<size_t>(plan.out_total));

  int nnz = 0;
  int min_kmin =  0;
  int max_kmax = -1;

  // First pass: determine (kmin, kmax) per row, count nnz, track global min/max
  for (int l = 0; l < plan.out_total; ++l) {
    const double x    = l / p.zoom + shift;
    const int    kmin = static_cast<int>(std::ceil (x - half_support));
    const int    kmax = static_cast<int>(std::floor(x + half_support));
    const int    wlen = kmax - kmin + 1;

    plan.kmin   [static_cast<size_t>(l)] = kmin;
    plan.win_len[static_cast<size_t>(l)] = wlen;
    nnz += wlen;

    if (kmin < min_kmin) min_kmin = kmin;
    if (kmax > max_kmax) max_kmax = kmax;
  }

  // Global pads for a single extended buffer: [ left_pad | ext | right_pad ]
  plan.left_pad  = std::max(0, -min_kmin);
  plan.right_pad = std::max(0,  max_kmax - (plan.length_total - 1));

  // Allocate contiguous weights; fold antisymmetric sign into weights
  plan.weights.resize(static_cast<size_t>(nnz));

  // Second pass: fill row_ptr and weights
  int cursor = 0;
  for (int l = 0; l < plan.out_total; ++l) {
    plan.row_ptr[static_cast<size_t>(l)] = cursor;

    const double x    = l / p.zoom + shift;
    const int    k0   = plan.kmin   [static_cast<size_t>(l)];
    const int    wlen = plan.win_len[static_cast<size_t>(l)];

    for (int t = 0; t < wlen; ++t) {
      const int k = k0 + t;

      // Sign from antisymmetric boundary for negative k only
      int sign = 1;
      if (k < 0 && !plan.symmetric_ext) sign = -1;

      double w = fact * beta(x - k, total_degree);
      if (sign != 1) w = -w;

      plan.weights[static_cast<size_t>(cursor++)] = w;
    }
  }
  plan.row_ptr.back() = cursor;

  return plan;
}

// Planned version — uses precomputed windows/weights; only data-dependent work remains.
void resize_1d_planned(const std::vector<double>& in,
                       std::vector<double>& out,
                       const LSParams& p,
                       const Plan1D& plan)
{
  const int N = plan.N;
  if (N == 0) { out.clear(); return; }

  const int corr_degree = (p.analy_degree < 0)
                        ?  p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);

  // 1) Interpolation coefficients (causal/anti-causal IIR on input)
  std::vector<double> coeff = in;
  get_interpolation_coefficients(coeff, p.interp_degree);

  // 2) Optional projection integration
  double average = 0.0;
  if (p.analy_degree >= 0) {
    average = do_integ(coeff, p.analy_degree + 1);
  }

  // 3) Build the finite extended buffer once (right tail only)
  std::vector<double> ext(static_cast<size_t>(plan.length_total));
  std::copy(coeff.begin(), coeff.end(), ext.begin());
  if (plan.length_total > N) {
    if (plan.symmetric_ext) {
      const int period = 2 * N - 2;
      for (int l = N; l < plan.length_total; ++l) {
        int t = l;
        if (period > 0 && t >= period) t %= period;
        if (t >= N) t = period - t;
        t = std::clamp(t, 0, N - 1);
        ext[static_cast<size_t>(l)] = coeff[static_cast<size_t>(t)];
      }
    } else {
      const int period = 2 * N - 3;
      for (int l = N; l < plan.length_total; ++l) {
        int t = l;
        if (period > 0 && t >= period) t %= period;
        if (t >= N) t = period - t;
        t = std::clamp(t, 0, N - 1);
        ext[static_cast<size_t>(l)] = -coeff[static_cast<size_t>(t)];
      }
    }
  }

  // 3b) Single padded buffer for contiguous window access: [LP | ext | RP]
  const int LP = plan.left_pad;
  const int RP = plan.right_pad;
  std::vector<double> ext_full(static_cast<size_t>(LP + plan.length_total + RP));

  // Left pad for negative k: mirror around 0 with correct sign rule
  if (LP > 0) {
    if (plan.symmetric_ext) {
      // symmetric: k = -t -> +coeff[t]
      for (int t = 1; t <= LP; ++t) {
        const int pos = LP - t;                  // target in ext_full
        const int src = std::min(t, std::max(0, N - 1));
        ext_full[static_cast<size_t>(pos)] = coeff[static_cast<size_t>(src)];
      }
    } else {
      // antisymmetric: k = -t -> -coeff[t-1]
      for (int t = 1; t <= LP; ++t) {
        const int pos = LP - t;
        const int src = std::min(t - 1, std::max(0, N - 1));
        ext_full[static_cast<size_t>(pos)] = -coeff[static_cast<size_t>(src)];
      }
    }
  }

  // Copy main ext block
  std::copy(ext.begin(), ext.end(), ext_full.begin() + LP);

  // Right pad (clamp)
  if (RP > 0) {
    const double last = ext.back();
    std::fill(ext_full.begin() + LP + plan.length_total, ext_full.end(), last);
  }

  // 4) Accumulate using the plan (contiguous weights & samples)
  std::vector<double> y(static_cast<size_t>(plan.out_total), 0.0);
  {
    const int*    __restrict rp = plan.row_ptr.data();
    const double* __restrict ww = plan.weights.data();
    const double* __restrict vf = ext_full.data();

    for (int l = 0; l < plan.out_total; ++l) {
      const int begin = rp[static_cast<size_t>(l)];
      const int end   = rp[static_cast<size_t>(l) + 1];
      const int M     = end - begin;
      const int k0    = plan.kmin[static_cast<size_t>(l)];

      const double* __restrict w = ww + begin;
      const double* __restrict v = vf + (LP + k0);

      double acc = 0.0;
#if defined(_OPENMP) && !defined(_MSC_VER)
      #pragma omp simd reduction(+:acc)
#endif
#if defined(_MSC_VER)
      #pragma loop(ivdep)
#endif
      for (int t = 0; t < M; ++t) acc += w[t] * v[t];

      y[static_cast<size_t>(l)] = acc;
    }
  }

  // 5) Projection tail: differentiate, add average, IIR + symmetric FIR sampling
  if (p.analy_degree >= 0) {
    do_diff(y, p.analy_degree + 1);
    for (int i = 0; i < plan.out_total; ++i) y[static_cast<size_t>(i)] += average;
    get_interpolation_coefficients(y, corr_degree);
    get_samples(y, p.synthe_degree);
  }

  // 6) Crop to true output size
  out.assign(y.begin(), y.begin() + plan.outN);
}

// Keep legacy symbol; if used directly, it will still work (builds a one-off plan).
void resize_1d(const std::vector<double>& in,
               std::vector<double>& out,
               const LSParams& p)
{
  Plan1D plan = make_plan_1d(static_cast<int>(in.size()), p);
  resize_1d_planned(in, out, p, plan);
}

} // namespace lsresize
