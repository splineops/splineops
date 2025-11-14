// splineops/cpp/lsresize/src/bspline.h
#pragma once
#include <cmath>
#include <stdexcept>
#include <array>

namespace lsresize {

// Centered cardinal B-spline of integer degree n (0..7)
inline double beta(double x, int n) {
  if (n < 0 || n > 7) throw std::invalid_argument("beta: degree must be in [0..7]");

  if (n == 0) { const double ax = std::abs(x); return (ax < 0.5 || x == -0.5) ? 1.0 : 0.0; }
  if (n == 1) { x = std::abs(x); return (x < 1.0) ? (1.0 - x) : 0.0; }

  x = std::abs(x);
  switch (n) {
    case 2:
      if (x < 0.5) return 3.0/4.0 - x*x;
      if (x < 1.5) { double t = x - 1.5; return 0.5 * t*t; }
      return 0.0;

    case 3:
      if (x < 1.0)  return 0.5*x*x*(x - 2.0) + 2.0/3.0;
      if (x < 2.0)  { double t = x - 2.0; return -t*t*t / 6.0; }
      return 0.0;

    case 4:
      if (x < 0.5)  { double t = x*x; return t*(t*0.25 - 0.625) + 115.0/192.0; }
      if (x < 1.5)  return x*(x*(x*(5.0/6.0 - x*(1.0/6.0)) - 1.25) + 5.0/24.0) + 55.0/96.0;
      if (x < 2.5)  { double t = x - 2.5; t *= t; return t*t * (1.0/24.0); }
      return 0.0;

    case 5:
      if (x < 1.0)  { double a = x*x; return a*(a*(0.25 - x*(1.0/12.0)) - 0.5) + 11.0/20.0; }
      if (x < 2.0)  return x*(x*(x*(x*(x*(1.0/24.0) - 3.0/8.0) + 1.25) - 1.75) + 0.625) + 17.0/40.0;
      if (x < 3.0)  { double a = 3.0 - x; double t = a*a; return a * t * t * (1.0/120.0); }
      return 0.0;

    case 6:
      if (x < 0.5)  { double t = x*x; return t*(t*(7.0/48.0 - x*(1.0/36.0)) - 77.0/192.0) + 5887.0/11520.0; }
      if (x < 1.5)  return x*(x*(x*(x*(x*(x*(1.0/48.0) - 7.0/48.0) + 21.0/64.0) - 35.0/288.0) - 91.0/256.0) - 7.0/768.0) + 7861.0/15360.0;
      if (x < 2.5)  return x*(x*(x*(x*(x*(7.0/60.0 - x*(1.0/120.0)) - 21.0/32.0) + 133.0/72.0) - 329.0/128.0) + 1267.0/960.0) + 1379.0/7680.0;
      if (x < 3.5)  { double t = x - 3.5; t *= (t*t); return t*t * (1.0/720.0); }
      return 0.0;

    case 7:
      if (x < 1.0)  { double a = x*x; return a*(a*(a*(x*(1.0/144.0) - 1.0/36.0) + 1.0/9.0) - 1.0/3.0) + 151.0/315.0; }
      if (x < 2.0)  return x*(x*(x*(x*(x*(x*(1.0/20.0 - x*(1.0/240.0)) - 7.0/30.0) + 0.5) - 7.0/18.0) - 0.1) - 7.0/90.0) + 103.0/210.0;
      if (x < 3.0)  return x*(x*(x*(x*(x*(x*(x*(1.0/720.0) - 1.0/36.0) + 7.0/30.0) - 19.0/18.0) + 49.0/18.0) - 23.0/6.0) + 217.0/90.0) - 139.0/630.0;
      if (x < 4.0)  { double a = 4.0 - x; double t = a*a*a; return t*t*a * (1.0/5040.0); }
      return 0.0;
  }
}

} // namespace lsresize
