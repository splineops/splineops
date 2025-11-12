// splineops/cpp/lsresize/src/resize1d.cpp
#include "resize1d.h"
#include "bspline.h"
#include "filters.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

#if defined(__AVX2__)
  #include <immintrin.h>
#endif

namespace lsresize {

// Optional tiny AVX2 FMA dot kernel (falls back to scalar when not available)
static inline double dot_small(const double* w, const double* v, int M) {
#if defined(__AVX2__)
  __m256d acc0 = _mm256_setzero_pd();
  int t = 0;
  for (; t + 4 <= M; t += 4) {
    __m256d ww = _mm256_loadu_pd(w + t);
    __m256d vv = _mm256_loadu_pd(v + t);
    acc0 = _mm256_fmadd_pd(ww, vv, acc0);
  }
  alignas(32) double tmp[4];
  _mm256_store_pd(tmp, acc0);
  double acc = tmp[0] + tmp[1] + tmp[2] + tmp[3];
  for (; t < M; ++t) acc += w[t] * v[t];
  return acc;
#else
  double acc = 0.0;
  for (int t = 0; t < M; ++t) acc += w[t] * v[t];
  return acc;
#endif
}

// Build the reusable 1-D plan (window metadata + contiguous weights + pad map)
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

  // Center shift (matches Python/native path)
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

  // First pass: (kmin, kmax) per row, nnz, global min/max
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

  // Global pads to build a single contiguous extended buffer: [LP | ext | RP]
  plan.left_pad  = std::max(0, -min_kmin);
  plan.right_pad = std::max(0,  max_kmax - (plan.length_total - 1));

  // Precompute left pad mapping for negative indices (removes per-line mirror math)
  plan.pad_src_idx.resize(static_cast<size_t>(plan.left_pad));
  plan.pad_src_sgn.resize(static_cast<size_t>(plan.left_pad), 1);
  for (int t = 1; t <= plan.left_pad; ++t) {
    const int pos = plan.left_pad - t; // 0 .. left_pad-1
    if (plan.symmetric_ext) {
      // symmetric: -t -> +coeff[t]
      plan.pad_src_idx[static_cast<size_t>(pos)] = t;   // clamped later to [0, N-1]
      plan.pad_src_sgn[static_cast<size_t>(pos)] =  1;
    } else {
      // antisymmetric: -t -> -coeff[t-1]
      plan.pad_src_idx[static_cast<size_t>(pos)] = t - 1;
      plan.pad_src_sgn[static_cast<size_t>(pos)] = -1;
    }
  }

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
      double w = fact * beta(x - k, total_degree);
      plan.weights[static_cast<size_t>(cursor++)] = w;
    }
  }
  plan.row_ptr.back() = cursor;

  return plan;
}

static inline void resize_1d_core(const std::vector<double>& in,
                                  std::vector<double>& out,
                                  const LSParams& p,
                                  const Plan1D& plan,
                                  std::vector<double>& coeff,
                                  std::vector<double>& ext,
                                  std::vector<double>& ext_full,
                                  std::vector<double>& y)
{
  const int N = plan.N;
  if (N == 0) { out.clear(); return; }

  const int corr_degree = (p.analy_degree < 0)
                        ?  p.interp_degree
                        : (p.analy_degree + p.synthe_degree + 1);

  // 1) Interpolation coefficients (causal/anti-causal IIR on input)
  coeff.assign(in.begin(), in.end());
  get_interpolation_coefficients(coeff, p.interp_degree);

  // 2) Optional projection integration
  double average = 0.0;
  if (p.analy_degree >= 0) {
    average = do_integ(coeff, p.analy_degree + 1);
  }

  // 3) Build the finite extended buffer once (right tail only)
  ext.resize(static_cast<size_t>(plan.length_total));
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
  ext_full.resize(static_cast<size_t>(LP + plan.length_total + RP));

  // Left pad using the precomputed mapping
  if (LP > 0) {
#if defined(_OPENMP) && !defined(_MSC_VER)
    #pragma omp simd
#endif
    for (int i = 0; i < LP; ++i) {
      const int src = std::min(std::max(plan.pad_src_idx[static_cast<size_t>(i)], 0), std::max(0, N - 1));
      const int sgn = static_cast<int>(plan.pad_src_sgn[static_cast<size_t>(i)]);
      ext_full[static_cast<size_t>(i)] = sgn * coeff[static_cast<size_t>(src)];
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
  y.assign(static_cast<size_t>(plan.out_total), 0.0);
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

      const double acc = dot_small(w, v, M);
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

// Public, allocation-free wrapper
void resize_1d_ws(const std::vector<double>& in,
                  std::vector<double>& out,
                  const LSParams& p,
                  const Plan1D& plan,
                  Work1D& ws)
{
  resize_1d_core(in, out, p, plan, ws.coeff, ws.ext, ws.ext_full, ws.y);
}

} // namespace lsresize
