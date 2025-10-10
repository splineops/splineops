// splineops/src/splineops/resize/lsresize/src/resize1d.cpp
#include "lsresize/resize1d.h"
#include "lsresize/bspline.h"
#include "lsresize/filters.h"
#include "lsresize/utils.h"

#include <cmath>
#include <vector>
#include <algorithm>

namespace lsresize {

void resize_1d(const std::vector<double>& in,
               std::vector<double>& out,
               const LSParams& p)
{
  const int N = (int)in.size();
  if (N == 0) { out.clear(); return; }

  int workN, outN;
  calculate_final_size_1d(p.inversable, N, p.zoom, workN, outN);

  double shift = p.shift;
  if (p.analy_degree >= 0) {
    const double t = (p.analy_degree + 1.0) / 2.0;
    shift += (t - std::floor(t)) * (1.0 / p.zoom - 1.0);
  }

  const int total_degree = p.interp_degree + std::max(p.analy_degree, 0) + 1;
  const int corr_degree  = (p.analy_degree < 0) ? p.interp_degree : (p.analy_degree + p.synthe_degree + 1);

  const int add_border = std::max(border(outN, corr_degree), total_degree);
  const int out_total  = outN + add_border;

  std::vector<double> coeff = in;
  get_interpolation_coefficients(coeff, p.interp_degree);

  double average = 0.0;
  if (p.analy_degree >= 0) average = do_integ(coeff, p.analy_degree + 1);

  const double half_support = (total_degree + 1) * 0.5;
  // See note in paper’s Fig. 3(e) about the remaining 'a' factor with the kernel.
  const double fact = std::pow(p.zoom, (p.analy_degree >= 0) ? (p.analy_degree + 1) : 0);

  std::vector<int>    idx_min(out_total), idx_max(out_total);
  std::vector<double> w; w.reserve((size_t)out_total * (2 + total_degree));
  for (int l = 0; l < out_total; ++l) {
    const double x = l / p.zoom + shift;
    const int kmin = (int)std::ceil(x - half_support);
    const int kmax = (int)std::floor(x + half_support);
    idx_min[l] = kmin; idx_max[l] = kmax;
    for (int k = kmin; k <= kmax; ++k) w.push_back( fact * beta(x - k, total_degree) );
  }

  std::vector<double> y(out_total, 0.0);
  const bool symmetric_ext = ((p.analy_degree + 1) % 2 == 0);
  size_t wi = 0;
  for (int l = 0; l < out_total; ++l) {
    double acc = 0.0;
    for (int k = idx_min[l]; k <= idx_max[l]; ++k, ++wi) {
      MirrorIndex m = mirror_index(k, N, symmetric_ext);
      acc += (double)m.sign * coeff[m.idx] * w[wi];
    }
    y[l] = acc;
  }

  if (p.analy_degree >= 0) {
    do_diff(y, p.analy_degree + 1);
    for (int i = 0; i < out_total; ++i) y[i] += average;
    get_interpolation_coefficients(y, corr_degree);
    get_samples(y, p.synthe_degree);
  }

  out.assign(y.begin(), y.begin() + outN);
}

} // namespace lsresize
