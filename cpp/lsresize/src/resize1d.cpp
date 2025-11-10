// splineops/cpp/lsresize/src/resize1d.cpp
#include "resize1d.h"
#include "bspline.h"
#include "filters.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace lsresize {

// Build the reusable 1-D plan (index/sign maps + weights)
Plan1D make_plan_1d(int N, const LSParams& p)
{
  Plan1D plan{};
  plan.N = N;

  // Output and tail sizing
  int workN = 0, outN = 0;
  calculate_final_size_1d(p.inversable, N, p.zoom, workN, outN);
  plan.outN = outN;

  const int total_degree = p.interp_degree + p.analy_degree + 1;
  const int corr_degree  = (p.analy_degree < 0) ? p.interp_degree
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

  // Precompute sparse windows and weights in CSR-like form
  const double half_support = 0.5 * (total_degree + 1);
  const double fact         = std::pow(p.zoom, (p.analy_degree >= 0) ? (p.analy_degree + 1) : 0);

  plan.row_ptr.resize(static_cast<size_t>(plan.out_total) + 1);
  plan.row_ptr[0] = 0;

  // First pass: count nnz to reserve
  int nnz = 0;
  for (int l = 0; l < plan.out_total; ++l) {
    const double x = l / p.zoom + shift;
    const int kmin = static_cast<int>(std::ceil (x - half_support));
    const int kmax = static_cast<int>(std::floor(x + half_support));
    nnz += (kmax - kmin + 1);
  }

  plan.ext_index.resize(nnz);
  plan.ext_sign .resize(nnz);
  plan.weights  .resize(nnz);

  // Second pass: fill row_ptr / indices / weights
  int cursor = 0;
  for (int l = 0; l < plan.out_total; ++l) {
    plan.row_ptr[static_cast<size_t>(l)] = cursor;

    const double x = l / p.zoom + shift;
    const int kmin = static_cast<int>(std::ceil (x - half_support));
    const int kmax = static_cast<int>(std::floor(x + half_support));

    for (int k = kmin; k <= kmax; ++k) {
      int idx;
      int sign = 1;

      if (k < 0) {
        idx = -k;
        if (!plan.symmetric_ext) { idx -= 1; sign = -1; }
      } else if (k >= plan.length_total) {
        idx = plan.length_total - 1; // clamp on the right (matches Python)
      } else {
        idx = k;
      }

      idx = std::clamp(idx, 0, plan.length_total - 1);

      plan.ext_index[static_cast<size_t>(cursor)] = idx;
      plan.ext_sign [static_cast<size_t>(cursor)] = static_cast<signed char>(sign);
      plan.weights  [static_cast<size_t>(cursor)] = fact * beta(x - k, total_degree);
      ++cursor;
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

  const int corr_degree  = (p.analy_degree < 0) ? p.interp_degree
                                                : (p.analy_degree + p.synthe_degree + 1);

  // 1) Interpolation coefficients (causal/anti-causal IIR on input)
  std::vector<double> coeff = in;
  get_interpolation_coefficients(coeff, p.interp_degree);

  // 2) Optional projection integration
  double average = 0.0;
  if (p.analy_degree >= 0) {
    average = do_integ(coeff, p.analy_degree + 1);
  }

  // 3) Build the finite extended buffer once (fast)
  std::vector<double> ext(static_cast<size_t>(plan.length_total));
  std::copy(coeff.begin(), coeff.end(), ext.begin());
  if (plan.length_total > N) {
    if (plan.symmetric_ext) {
      const int period = 2 * N - 2;
      for (int l = N; l < plan.length_total; ++l) {
        int t = l;
        if (period > 0 && t >= period) t = t % period;
        if (t >= N) t = period - t;
        t = std::clamp(t, 0, N - 1);
        ext[static_cast<size_t>(l)] = coeff[static_cast<size_t>(t)];
      }
    } else {
      const int period = 2 * N - 3;
      for (int l = N; l < plan.length_total; ++l) {
        int t = l;
        if (period > 0 && t >= period) t = t % period;
        if (t >= N) t = period - t;
        t = std::clamp(t, 0, N - 1);
        ext[static_cast<size_t>(l)] = -coeff[static_cast<size_t>(t)];
      }
    }
  }

  // 4) Accumulate using the plan (no branches, contiguous weights)
  std::vector<double> y(static_cast<size_t>(plan.out_total), 0.0);
  for (int l = 0; l < plan.out_total; ++l) {
    const int begin = plan.row_ptr[static_cast<size_t>(l)];
    const int end   = plan.row_ptr[static_cast<size_t>(l)+1];

    double acc = 0.0;
#if defined(_OPENMP) && !defined(_MSC_VER)
    #pragma omp simd reduction(+:acc)
#endif
#if defined(_MSC_VER)
    #pragma loop(ivdep)
#endif
    for (int e = begin; e < end; ++e) {
      const double w   = plan.weights [static_cast<size_t>(e)];
      const int    idx = plan.ext_index[static_cast<size_t>(e)];
      const int    sgn = static_cast<int>(plan.ext_sign[static_cast<size_t>(e)]);
      acc += w * (sgn * ext[static_cast<size_t>(idx)]);
    }
    y[static_cast<size_t>(l)] = acc;
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
