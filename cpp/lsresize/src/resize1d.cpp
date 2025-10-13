// splineops/cpp/lsresize/src/resize1d.cpp
#include "resize1d.h"
#include "bspline.h"
#include "filters.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace lsresize {

void resize_1d(const std::vector<double>& in,
               std::vector<double>& out,
               const LSParams& p)
{
  const int inputN = static_cast<int>(in.size());
  if (inputN == 0) { out.clear(); return; }

  // Final (output) length and "working" length (used only to define periods)
  int workN = 0, outN = 0;
  calculate_final_size_1d(p.inversable, inputN, p.zoom, workN, outN);

  // Shift adjustment (matches Python: analy-dependent center correction)
  double shift = p.shift;
  if (p.analy_degree >= 0) {
    const double t = (p.analy_degree + 1.0) / 2.0;
    shift += (t - std::floor(t)) * (1.0 / p.zoom - 1.0);
  }

  const int total_degree = p.interp_degree + p.analy_degree + 1;
  const int corr_degree  = (p.analy_degree < 0) ? p.interp_degree
                                               : (p.analy_degree + p.synthe_degree + 1);

  // Additional tail to guarantee tolerance of the recursive filter
  const int add_border = std::max(border(outN, corr_degree), total_degree);
  const int out_total  = outN + add_border;

  // 1) Interpolation coefficients of input line (causal/anti-causal IIR)
  std::vector<double> coeff = in;
  get_interpolation_coefficients(coeff, p.interp_degree);

  // 2) (Projection variant) integrate analy_degree+1 times and remember average
  double average = 0.0;
  if (p.analy_degree >= 0) {
    average = do_integ(coeff, p.analy_degree + 1);
  }

  // 3) Precompute support windows and weights
  const double half_support = 0.5 * (total_degree + 1);
  // Remaining a^(n1+1) factor (see paper Fig. 3(e) note)
  const double fact = std::pow(p.zoom, (p.analy_degree >= 0) ? (p.analy_degree + 1) : 0);

  std::vector<int>    idx_min(out_total), idx_max(out_total);
  std::vector<double> w; w.reserve(static_cast<size_t>(out_total) * (2 + total_degree));
  for (int l = 0; l < out_total; ++l) {
    const double x = l / p.zoom + shift;
    const int kmin = static_cast<int>(std::ceil (x - half_support));
    const int kmax = static_cast<int>(std::floor(x + half_support));
    idx_min[l] = kmin;
    idx_max[l] = kmax;
    for (int k = kmin; k <= kmax; ++k) {
      w.push_back(fact * beta(x - k, total_degree));
    }
  }

  // 4) Build the finite extended buffer (Python-compatible)
  //    length_total = ny + ceil(add_border / zoom)
  const bool symmetric_ext = ((p.analy_degree + 1) % 2 == 0);
  const int  N = static_cast<int>(coeff.size());

  // Python: length_total = length_input + ceil(add_border / zoom)
  const int  length_total = N + static_cast<int>(std::ceil(add_border / p.zoom));

  std::vector<double> ext(length_total);
  std::copy(coeff.begin(), coeff.end(), ext.begin());

  if (length_total > N) {
    if (symmetric_ext) {
      // Symmetric extension with period based on *N*
      const int period = 2 * N - 2;
      for (int l = N; l < length_total; ++l) {
        int t = l;
        if (period > 0 && t >= period) t = t % period;
        if (t >= N) t = period - t;          // reflect
        t = std::clamp(t, 0, N - 1);
        ext[l] = coeff[t];
      }
    } else {
      // Antisymmetric extension with period based on *N*
      const int period = 2 * N - 3;
      for (int l = N; l < length_total; ++l) {
        int t = l;
        if (period > 0 && t >= period) t = t % period;
        if (t >= N) t = period - t;          // reflect
        t = std::clamp(t, 0, N - 1);
        ext[l] = -coeff[t];
      }
    }
  }

  // 5) Accumulate with Python's k-handling:
  //    - if k < 0: reflect at 0; antisym flips sign and shifts by one
  //    - if k >= length_total: clamp to length_total-1
  std::vector<double> y(out_total, 0.0);
  size_t wi = 0;
  for (int l = 0; l < out_total; ++l) {
    double acc = 0.0;
    for (int k = idx_min[l]; k <= idx_max[l]; ++k, ++wi) {
      int index = k;
      int sign  = 1;

      if (k < 0) {
        index = -k;
        if (!symmetric_ext) { index -= 1; sign = -1; }
      } else if (k >= length_total) {
        index = length_total - 1; // clamp on the right
      }

      index = std::clamp(index, 0, length_total - 1);
      acc += static_cast<double>(sign) * ext[index] * w[wi];
    }
    y[l] = acc;
  }

  // 6) Projection tail (differentiate, add average, IIR + symmetric FIR sampling)
  if (p.analy_degree >= 0) {
    do_diff(y, p.analy_degree + 1);
    for (int i = 0; i < out_total; ++i) y[i] += average;
    get_interpolation_coefficients(y, corr_degree);
    get_samples(y, p.synthe_degree);
  }

  // 7) Crop to the true output size
  out.assign(y.begin(), y.begin() + outN);
}

} // namespace lsresize
