// splineops/cpp/lsresize/src/filters.cpp
#include "filters.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace lsresize {

double initial_causal(const std::vector<double>& c, double z, double tol) {
  const size_t N = c.size();
  if (N == 0) return 0.0;

  const double zn = std::pow(z, double(N - 1));
  size_t horizon = N;
  if (tol > 0.0) {
    horizon = std::min(N, size_t(2 + std::log(tol) / std::log(std::abs(z))));
  }

  double sum = c[0] + zn * c[N - 1];
  // z^n and z^(N-1-n) without pow:
  double p1 = z;          // z^1
  double p2 = (N >= 2) ? (zn / z) : 1.0;  // z^(N-2) if N>=2 else 1
  for (size_t n = 1; n + 1 < horizon; ++n) {
    sum += (p1 + p2) * c[n];
    p1 *= z;            // z^n -> z^(n+1)
    p2 /= z;            // z^(N-1-n) -> z^(N-2-n)
  }

  const double denom = 1.0 - (zn * zn);   // 1 - z^(2N-2)
  return sum / denom;
}

double initial_anti_causal(const std::vector<double>& c, double z) {
  if (c.size() < 2) return 0.0;
  return (z * c[c.size()-2] + c.back()) * z / (z*z - 1.0);
}

void symmetric_fir(const std::vector<double>& h, const std::vector<double>& c, std::vector<double>& s) {
  const size_t N = c.size();
  if (s.size() != N) s.assign(N, 0.0);
  const size_t L = h.size();
  if (L == 0 || N == 0) return;

  if (L == 2) {
    if (N >= 2) {
      s[0]   = h[0]*c[0] + 2.0*h[1]*c[1];
      for (size_t i=1; i+1<N; ++i) s[i] = h[0]*c[i] + h[1]*(c[i-1]+c[i+1]);
      s[N-1] = h[0]*c[N-1] + 2.0*h[1]*c[N-2];
    } else {
      s[0] = (h[0] + 2.0*h[1]) * c[0];
    }
    return;
  }

  if (L == 3) {
    if (N >= 4) {
      s[0]   = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2];
      s[1]   = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]);
      for (size_t i=2; i+2<N; ++i)
        s[i] = h[0]*c[i] + h[1]*(c[i-1]+c[i+1]) + h[2]*(c[i-2]+c[i+2]);
      s[N-2] = h[0]*c[N-2] + h[1]*(c[N-3]+c[N-1]) + h[2]*(c[N-4]+c[N-2]);
      s[N-1] = h[0]*c[N-1] + 2.0*h[1]*c[N-2] + 2.0*h[2]*c[N-3];
    } else if (N == 3) {
      s[0] = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + 2.0*h[2]*c[1];
      s[2] = h[0]*c[2] + 2.0*h[1]*c[1] + 2.0*h[2]*c[0];
    } else if (N == 2) {
      s[0] = (h[0] + 2.0*h[2]) * c[0] + 2.0*h[1]*c[1];
      s[1] = (h[0] + 2.0*h[2]) * c[1] + 2.0*h[1]*c[0];
    } else { // N==1
      s[0] = (h[0] + 2.0*(h[1]+h[2])) * c[0];
    }
    return;
  }

  if (L == 4) {
    if (N >= 6) {
      s[0] = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2] + 2.0*h[3]*c[3];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + h[3]*(c[2]+c[4]);
      s[2] = h[0]*c[2] + h[1]*(c[1]+c[3]) + h[2]*(c[0]+c[4]) + h[3]*(c[1]+c[5]);
      for (size_t i=3; i+3<N; ++i)
        s[i] = h[0]*c[i] + h[1]*(c[i-1]+c[i+1]) + h[2]*(c[i-2]+c[i+2]) + h[3]*(c[i-3]+c[i+3]);
      s[N-3] = h[0]*c[N-3] + h[1]*(c[N-4]+c[N-2]) + h[2]*(c[N-5]+c[N-1]) + h[3]*(c[N-6]+c[N-2]);
      s[N-2] = h[0]*c[N-2] + h[1]*(c[N-3]+c[N-1]) + h[2]*(c[N-4]+c[N-2]) + h[3]*(c[N-5]+c[N-3]);
      s[N-1] = h[0]*c[N-1] + 2.0*h[1]*c[N-2] + 2.0*h[2]*c[N-3] + 2.0*h[3]*c[N-4];
    } else if (N == 5) {
      s[0] = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2] + 2.0*h[3]*c[3];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + h[3]*(c[2]+c[4]);
      s[2] = h[0]*c[2] + (h[1]+h[3])*(c[1]+c[3]) + h[2]*(c[0]+c[4]);
      s[3] = h[0]*c[3] + h[1]*(c[2]+c[4]) + h[2]*(c[1]+c[3]) + h[3]*(c[0]+c[2]);
      s[4] = h[0]*c[4] + 2.0*h[1]*c[3] + 2.0*h[2]*c[2] + 2.0*h[3]*c[1];
    } else if (N == 4) {
      s[0] = h[0]*c[0] + 2.0*h[1]*c[1] + 2.0*h[2]*c[2] + 2.0*h[3]*c[3];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + 2.0*h[3]*c[2];
      s[2] = h[0]*c[2] + h[1]*(c[1]+c[3]) + h[2]*(c[0]+c[2]) + 2.0*h[3]*c[1];
      s[3] = h[0]*c[3] + 2.0*h[1]*c[2] + 2.0*h[2]*c[1] + 2.0*h[3]*c[0];
    } else if (N == 3) {
      s[0] = h[0]*c[0] + 2.0*(h[1]+h[3])*c[1] + 2.0*h[2]*c[2];
      s[1] = h[0]*c[1] + (h[1]+h[3])*(c[0]+c[2]) + 2.0*h[2]*c[1];
      s[2] = h[0]*c[2] + 2.0*(h[1]+h[3])*c[1] + 2.0*h[2]*c[0];
    } else if (N == 2) {
      s[0] = (h[0]+2.0*h[2])*c[0] + 2.0*(h[1]+h[3])*c[1];
      s[1] = (h[0]+2.0*h[2])*c[1] + 2.0*(h[1]+h[3])*c[0];
    } else { // N==1
      s[0] = (h[0] + 2.0*(h[1]+h[2]+h[3])) * c[0];
    }
    return;
  }

  throw std::invalid_argument("Invalid filter half-length (should be 2..4)");
}

double do_integ(std::vector<double>& c, int nb) {
  const size_t N = c.size();
  if (N == 0 || nb <= 0) return 0.0;

  auto avg_of = [&](const std::vector<double>& x)->double {
    const double sum = std::accumulate(x.begin(), x.end(), 0.0);
    return (2.0*sum - x.back() - x.front()) / (2.0*N - 2.0);
  };

  double m = 0.0, average = 0.0;
  if (nb >= 1) { average = avg_of(c); integ_sa(c, average); }
  if (nb >= 2) { std::vector<double> tmp=c; integ_as(tmp, c); }
  if (nb >= 3) { m = avg_of(c); integ_sa(c, m); }
  if (nb >= 4) { std::vector<double> tmp=c; integ_as(tmp, c); }
  return average;
}

void integ_sa(std::vector<double>& c, double m) {
  c[0] -= m; c[0] *= 0.5;
  for (size_t i = 1; i < c.size(); ++i) { c[i] -= m; c[i] += c[i-1]; }
}

void integ_as(const std::vector<double>& c, std::vector<double>& y) {
  const size_t N = c.size();
  y.resize(N);
  std::vector<double> z = c;
  y[0] = z[0];
  if (N > 1) y[1] = 0.0;
  for (size_t i = 2; i < N; ++i) y[i] = y[i-1] - z[i-1];
}

void do_diff(std::vector<double>& c, int nb) {
  const size_t N = c.size();
  if (N == 0 || nb <= 0) return;
  if (nb == 1) { diff_as(c); return; }
  if (nb == 2) { diff_sa(c); diff_as(c); return; }
  if (nb == 3) { diff_as(c); diff_sa(c); diff_as(c); return; }
  if (nb >= 4) { diff_sa(c); diff_as(c); diff_sa(c); diff_as(c); return; }
}

void diff_sa(std::vector<double>& c) {
  if (c.size() < 2) return;
  double old = c[c.size()-2];
  for (size_t i = 0; i + 1 < c.size(); ++i) c[i] = c[i] - c[i+1];
  c.back() -= old;
}

void diff_as(std::vector<double>& c) {
  if (c.size() < 2) { if (!c.empty()) c[0] *= 2.0; return; }
  for (size_t i = c.size()-1; i > 0; --i) c[i] = c[i] - c[i-1];
  c[0] *= 2.0;
}

} // namespace lsresize
