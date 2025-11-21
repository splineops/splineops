// splineops/cpp/lsresize/src/filters.h
#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace lsresize {

// -----------------------------------------------------------------------------
// Poles z_k for degrees 2..7 (Unser '93), templated on Real
// -----------------------------------------------------------------------------
template <typename Real>
inline const std::vector<Real>& spline_poles(int deg) {
  using std::sqrt;

  static const std::vector<Real> z2 = {
    Real(sqrt(8.0) - 3.0)
  };
  static const std::vector<Real> z3 = {
    Real(sqrt(3.0) - 2.0)
  };
  static const std::vector<Real> z4 = {
    Real(sqrt(664.0 - std::sqrt(438976.0)) + sqrt(304.0) - 19.0),
    Real(sqrt(664.0 + std::sqrt(438976.0)) - sqrt(304.0) - 19.0)
  };
  static const std::vector<Real> z5 = {
    Real(sqrt(135.0/2.0 - std::sqrt(17745.0/4.0)) + sqrt(105.0/4.0) - 6.5),
    Real(sqrt(135.0/2.0 + std::sqrt(17745.0/4.0)) - sqrt(105.0/4.0) - 6.5)
  };
  static const std::vector<Real> z6 = {
    Real(-0.488294589303044755130118038883789062112279161239377608394),
    Real(-0.081679271076237512597937765737059080653379610398148178525368),
    Real(-0.00141415180832581775108724397655859252786416905534669851652709)
  };
  static const std::vector<Real> z7 = {
    Real(-0.5352804307964381655424037816816460718339231523426924148812),
    Real(-0.122554615192326690515272264359357343605486549427295558490763),
    Real(-0.0091486948096082769285930216516478534156925639545994482648003)
  };

  switch (deg) {
    case 0: case 1: {
      static const std::vector<Real> none;
      return none;
    }
    case 2: return z2;
    case 3: return z3;
    case 4: return z4;
    case 5: return z5;
    case 6: return z6;
    case 7: return z7;
    default:
      throw std::invalid_argument("spline_poles: degree must be in [0..7]");
  }
}

// -----------------------------------------------------------------------------
// Symmetric FIR taps for sampling (Step 5)
// -----------------------------------------------------------------------------
template <typename Real>
inline const std::vector<Real>& sampling_fir(int deg) {
  static const std::vector<Real> h2 = {
    Real(3.0/4.0), Real(1.0/8.0)
  };
  static const std::vector<Real> h3 = {
    Real(2.0/3.0), Real(1.0/6.0)
  };
  static const std::vector<Real> h4 = {
    Real(115.0/192.0), Real(19.0/96.0), Real(1.0/384.0)
  };
  static const std::vector<Real> h5 = {
    Real(11.0/20.0), Real(13.0/60.0), Real(1.0/120.0)
  };
  static const std::vector<Real> h6 = {
    Real(5887.0/11520.0),
    Real(10543.0/46080.0),
    Real(361.0/23040.0),
    Real(1.0/46080.0)
  };
  static const std::vector<Real> h7 = {
    Real(151.0/315.0),
    Real(397.0/1680.0),
    Real(1.0/42.0),
    Real(1.0/5040.0)
  };

  switch (deg) {
    case 0: case 1: {
      static const std::vector<Real> none;
      return none;
    }
    case 2: return h2;
    case 3: return h3;
    case 4: return h4;
    case 5: return h5;
    case 6: return h6;
    case 7: return h7;
    default:
      throw std::invalid_argument("sampling_fir: degree must be in [0..7]");
  }
}

// -----------------------------------------------------------------------------
// Spline IIR prefilter (Unser '93), templated on Real
// -----------------------------------------------------------------------------
template <typename Real>
inline Real initial_causal(const std::vector<Real>& c, Real z, Real tol = Real(1e-10)) {
  const size_t N = c.size();
  if (N == 0) return Real(0);

  const Real zn = std::pow(z, Real(N - 1));
  size_t horizon = N;
  if (tol > Real(0)) {
    horizon = std::min(
      N,
      size_t(2 + std::log(tol) / std::log(std::abs(z)))
    );
  }

  Real sum = c[0] + zn * c[N - 1];
  Real p1 = z;                      // z^1
  Real p2 = (N >= 2) ? (zn / z) : Real(1);  // z^(N-2) if N>=2 else 1

  for (size_t n = 1; n + 1 < horizon; ++n) {
    sum += (p1 + p2) * c[n];
    p1 *= z;  // z^n -> z^(n+1)
    p2 /= z;  // z^(N-1-n) -> z^(N-2-n)
  }

  const Real denom = Real(1) - (zn * zn);   // 1 - z^(2N-2)
  return sum / denom;
}

template <typename Real>
inline Real initial_anti_causal(const std::vector<Real>& c, Real z) {
  const size_t N = c.size();
  if (N < 2) return Real(0);
  return (z * c[N - 2] + c[N - 1]) * z / (z*z - Real(1));
}

// -----------------------------------------------------------------------------
// Interpolation coefficients (causal/anti-causal IIR on input)
// -----------------------------------------------------------------------------
template <typename Real>
inline void get_interpolation_coefficients(std::vector<Real>& c, int deg) {
  const size_t N = c.size();
  if (deg <= 1 || N <= 1) return;

  const auto& poles = spline_poles<Real>(deg);

  Real lambda = Real(1);
  for (Real z : poles) {
    lambda *= (Real(1) - z) * (Real(1) - Real(1) / z);
  }
  for (Real& v : c) {
    v *= lambda;
  }

  for (Real z : poles) {
    // forward (causal)
    c[0] = initial_causal<Real>(c, z);
    for (size_t n = 1; n < N; ++n) {
      c[n] += z * c[n - 1];
    }

    // backward (anti-causal) — must include n == 0
    c[N - 1] = initial_anti_causal<Real>(c, z);
    for (int n = static_cast<int>(N) - 2; n >= 0; --n) {
      c[static_cast<size_t>(n)] = z * (c[static_cast<size_t>(n + 1)] - c[static_cast<size_t>(n)]);
    }
  }
}

// -----------------------------------------------------------------------------
// Symmetric FIR sampling (Step 5)
// -----------------------------------------------------------------------------
template <typename Real>
inline void symmetric_fir(const std::vector<Real>& h,
                          const std::vector<Real>& c,
                          std::vector<Real>& s)
{
  const size_t N = c.size();
  if (s.size() != N) s.assign(N, Real(0));
  const size_t L = h.size();
  if (L == 0 || N == 0) return;

  if (L == 2) {
    if (N >= 2) {
      s[0]   = h[0]*c[0] + Real(2)*h[1]*c[1];
      for (size_t i = 1; i + 1 < N; ++i) {
        s[i] = h[0]*c[i] + h[1]*(c[i-1] + c[i+1]);
      }
      s[N-1] = h[0]*c[N-1] + Real(2)*h[1]*c[N-2];
    } else {
      s[0] = (h[0] + Real(2)*h[1]) * c[0];
    }
    return;
  }

  if (L == 3) {
    if (N >= 4) {
      s[0]   = h[0]*c[0] + Real(2)*h[1]*c[1] + Real(2)*h[2]*c[2];
      s[1]   = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]);
      for (size_t i = 2; i + 2 < N; ++i) {
        s[i] = h[0]*c[i] + h[1]*(c[i-1]+c[i+1]) + h[2]*(c[i-2]+c[i+2]);
      }
      s[N-2] = h[0]*c[N-2] + h[1]*(c[N-3]+c[N-1]) + h[2]*(c[N-4]+c[N-2]);
      s[N-1] = h[0]*c[N-1] + Real(2)*h[1]*c[N-2] + Real(2)*h[2]*c[N-3];
    } else if (N == 3) {
      s[0] = h[0]*c[0] + Real(2)*h[1]*c[1] + Real(2)*h[2]*c[2];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + Real(2)*h[2]*c[1];
      s[2] = h[0]*c[2] + Real(2)*h[1]*c[1] + Real(2)*h[2]*c[0];
    } else if (N == 2) {
      s[0] = (h[0] + Real(2)*h[2]) * c[0] + Real(2)*h[1]*c[1];
      s[1] = (h[0] + Real(2)*h[2]) * c[1] + Real(2)*h[1]*c[0];
    } else { // N==1
      s[0] = (h[0] + Real(2)*(h[1]+h[2])) * c[0];
    }
    return;
  }

  if (L == 4) {
    if (N >= 6) {
      s[0] = h[0]*c[0] + Real(2)*h[1]*c[1] + Real(2)*h[2]*c[2] + Real(2)*h[3]*c[3];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + h[3]*(c[2]+c[4]);
      s[2] = h[0]*c[2] + h[1]*(c[1]+c[3]) + h[2]*(c[0]+c[4]) + h[3]*(c[1]+c[5]);
      for (size_t i = 3; i + 3 < N; ++i) {
        s[i] = h[0]*c[i] +
               h[1]*(c[i-1]+c[i+1]) +
               h[2]*(c[i-2]+c[i+2]) +
               h[3]*(c[i-3]+c[i+3]);
      }
      s[N-3] = h[0]*c[N-3] + h[1]*(c[N-4]+c[N-2]) + h[2]*(c[N-5]+c[N-1]) + h[3]*(c[N-6]+c[N-2]);
      s[N-2] = h[0]*c[N-2] + h[1]*(c[N-3]+c[N-1]) + h[2]*(c[N-4]+c[N-2]) + h[3]*(c[N-5]+c[N-3]);
      s[N-1] = h[0]*c[N-1] + Real(2)*h[1]*c[N-2] + Real(2)*h[2]*c[N-3] + Real(2)*h[3]*c[N-4];
    } else if (N == 5) {
      s[0] = h[0]*c[0] + Real(2)*h[1]*c[1] + Real(2)*h[2]*c[2] + Real(2)*h[3]*c[3];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + h[3]*(c[2]+c[4]);
      s[2] = h[0]*c[2] + (h[1]+h[3])*(c[1]+c[3]) + h[2]*(c[0]+c[4]);
      s[3] = h[0]*c[3] + h[1]*(c[2]+c[4]) + h[2]*(c[1]+c[3]) + h[3]*(c[0]+c[2]);
      s[4] = h[0]*c[4] + Real(2)*h[1]*c[3] + Real(2)*h[2]*c[2] + Real(2)*h[3]*c[1];
    } else if (N == 4) {
      s[0] = h[0]*c[0] + Real(2)*h[1]*c[1] + Real(2)*h[2]*c[2] + Real(2)*h[3]*c[3];
      s[1] = h[0]*c[1] + h[1]*(c[0]+c[2]) + h[2]*(c[1]+c[3]) + Real(2)*h[3]*c[2];
      s[2] = h[0]*c[2] + h[1]*(c[1]+c[3]) + h[2]*(c[0]+c[2]) + Real(2)*h[3]*c[1];
      s[3] = h[0]*c[3] + Real(2)*h[1]*c[2] + Real(2)*h[2]*c[1] + Real(2)*h[3]*c[0];
    } else if (N == 3) {
      s[0] = h[0]*c[0] + Real(2)*(h[1]+h[3])*c[1] + Real(2)*h[2]*c[2];
      s[1] = h[0]*c[1] + (h[1]+h[3])*(c[0]+c[2]) + Real(2)*h[2]*c[1];
      s[2] = h[0]*c[2] + Real(2)*(h[1]+h[3])*c[1] + Real(2)*h[2]*c[0];
    } else if (N == 2) {
      s[0] = (h[0]+Real(2)*h[2])*c[0] + Real(2)*(h[1]+h[3])*c[1];
      s[1] = (h[0]+Real(2)*h[2])*c[1] + Real(2)*(h[1]+h[3])*c[0];
    } else { // N==1
      s[0] = (h[0] + Real(2)*(h[1]+h[2]+h[3])) * c[0];
    }
    return;
  }

  throw std::invalid_argument("symmetric_fir: invalid filter half-length (should be 2..4)");
}

// High-level get_samples: symmetric FIR sampling
template <typename Real>
inline void get_samples(std::vector<Real>& c, int deg) {
  if (deg <= 1) return;
  const auto& h = sampling_fir<Real>(deg);
  std::vector<Real> s(c.size(), Real(0));
  symmetric_fir(h, c, s);
  c.swap(s);
}

// -----------------------------------------------------------------------------
// Running sums Δ^{-1} and centered differences Δ with alternation (Fig. 8)
// -----------------------------------------------------------------------------
template <typename Real>
inline void integ_sa(std::vector<Real>& c, Real m) {
  if (c.empty()) return;
  c[0] -= m;
  c[0] *= Real(0.5);
  for (size_t i = 1; i < c.size(); ++i) {
    c[i] -= m;
    c[i] += c[i-1];
  }
}

template <typename Real>
inline void integ_as(const std::vector<Real>& c, std::vector<Real>& y) {
  const size_t N = c.size();
  y.resize(N);
  std::vector<Real> z = c;
  if (N == 0) return;
  y[0] = z[0];
  if (N > 1) y[1] = Real(0);
  for (size_t i = 2; i < N; ++i) {
    y[i] = y[i-1] - z[i-1];
  }
}

template <typename Real>
inline Real do_integ(std::vector<Real>& c, int nb) {
  const size_t N = c.size();
  if (N == 0 || nb <= 0) return Real(0);

  auto avg_of = [&](const std::vector<Real>& x)->Real {
    Real sum = Real(0);
    for (Real v : x) sum += v;
    return (Real(2)*sum - x.back() - x.front()) /
           (Real(2)*Real(N) - Real(2));
  };

  Real m = Real(0);
  Real average = Real(0);

  if (nb >= 1) {
    average = avg_of(c);
    integ_sa(c, average);
  }
  if (nb >= 2) {
    std::vector<Real> tmp = c;
    integ_as(tmp, c);
  }
  if (nb >= 3) {
    m = avg_of(c);
    integ_sa(c, m);
  }
  if (nb >= 4) {
    std::vector<Real> tmp = c;
    integ_as(tmp, c);
  }
  return average;
}

template <typename Real>
inline void diff_sa(std::vector<Real>& c) {
  const size_t N = c.size();
  if (N < 2) return;
  Real old = c[N-2];
  for (size_t i = 0; i + 1 < N; ++i) {
    c[i] = c[i] - c[i+1];
  }
  c[N-1] -= old;
}

template <typename Real>
inline void diff_as(std::vector<Real>& c) {
  const size_t N = c.size();
  if (N < 2) {
    if (N == 1) c[0] *= Real(2);
    return;
  }
  for (size_t i = N - 1; i > 0; --i) {
    c[i] = c[i] - c[i-1];
  }
  c[0] *= Real(2);
}

template <typename Real>
inline void do_diff(std::vector<Real>& c, int nb) {
  const size_t N = c.size();
  if (N == 0 || nb <= 0) return;
  if (nb == 1) { diff_as(c); return; }
  if (nb == 2) { diff_sa(c); diff_as(c); return; }
  if (nb == 3) { diff_as(c); diff_sa(c); diff_as(c); return; }
  if (nb >= 4) { diff_sa(c); diff_as(c); diff_sa(c); diff_as(c); return; }
}

} // namespace lsresize
