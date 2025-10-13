// splineops/cpp/lsresize/src/bspline.h
#pragma once
#include <cmath>
#include <stdexcept>
#include <array>

namespace lsresize {

// Centered cardinal B-spline of integer degree n (0..7)
inline double beta(double x, int n) {
  if (n < 0 || n > 7) {
    throw std::invalid_argument("beta: degree must be in [0..7]");
  }

  // Degree 0: special-case to match Python/Java edge convention
  if (n == 0) {
    const double ax = std::abs(x);
    return (ax < 0.5 || x == -0.5) ? 1.0 : 0.0;
  }

  // Shift to the non-centered support [0, n+1]
  const double t = x + 0.5 * (n + 1);

  // Outside compact support (open on the right)
  if (t <= 0.0 || t >= (n + 1.0)) return 0.0;

  // small factorial table up to 8!
  static constexpr double FACT[] = {
    1.0,   // 0!
    1.0,   // 1!
    2.0,   // 2!
    6.0,   // 3!
    24.0,  // 4!
    120.0, // 5!
    720.0, // 6!
    5040.0,// 7!
    40320.0// 8!
  };

  auto binom = [](int nn, int kk) -> double {
    // exact for small nn using factorials
    return FACT[nn] / (FACT[kk] * FACT[nn - kk]);
  };

  double sum = 0.0;
  const int m = n + 1; // upper index in the sum and support length

  for (int k = 0; k <= m; ++k) {
    const double u = t - static_cast<double>(k);
    if (u <= 0.0) continue;               // truncated power (u)_+^n
    const double term = std::pow(u, n);
    const double sign = (k & 1) ? -1.0 : 1.0;
    sum += sign * binom(m, k) * term;
  }
  return sum / FACT[n];
}

} // namespace lsresize
