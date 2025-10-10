// splineops/src/splineops/resize/lsresize/src/bspline.h
#pragma once
#include <cmath>
#include <stdexcept>

namespace lsresize {

// Centered B-spline of integer degree n (0..7)
inline double beta(double x, int n) {
  using std::abs;
  using std::sqrt;

  double b = 0.0;
  switch (n) {
    case 0: { if (abs(x) < 0.5 || x == -0.5) b = 1.0; break; }
    case 1: { x = abs(x); if (x < 1.0) b = 1.0 - x; break; }
    case 2: {
      x = abs(x);
      if (x < 0.5) b = 0.75 - x*x;
      else if (x < 1.5) { x -= 1.5; b = 0.5 * x*x; }
      break;
    }
    case 3: {
      x = abs(x);
      if (x < 1.0) b = 0.5*x*x*(x - 2.0) + 2.0/3.0;
      else if (x < 2.0) { x -= 2.0; b = - (1.0/6.0)*x*x*x; }
      break;
    }
    case 4: {
      x = abs(x);
      if (x < 0.5) { double xx = x*x; b = xx*(0.25*xx - 5.0/8.0) + 115.0/192.0; }
      else if (x < 1.5) b = x*(x*(x*(5.0/6.0 - x*(1.0/6.0)) - 5.0/4.0) + 5.0/24.0) + 55.0/96.0;
      else if (x < 2.5) { x -= 2.5; double xx = x*x; b =  (1.0/24.0)*xx*xx; }
      break;
    }
    case 5: {
      x = abs(x);
      if (x < 1.0) { double a = x*x; b = a*(a*(0.25 - x*(1.0/12.0)) - 0.5) + 11.0/20.0; }
      else if (x < 2.0)
        b = x*(x*(x*(x*(x*(1.0/24.0) - 3.0/8.0) + 5.0/4.0) - 7.0/4.0) + 5.0/8.0) + 17.0/40.0;
      else if (x < 3.0) { double a = 3.0 - x; double xx = a*a; b = (1.0/120.0)*a*xx*xx; }
      break;
    }
    case 6: {
      x = abs(x);
      if (x < 0.5) { double xx = x*x; b = xx*(xx*(7.0/48.0 - xx*(1.0/36.0)) - 77.0/192.0) + 5887.0/11520.0; }
      else if (x < 1.5)
        b = x*(x*(x*(x*(x*(x*(1.0/48.0) - 7.0/48.0) + 21.0/64.0) - 35.0/288.0) - 91.0/256.0) - 7.0/768.0) + 7861.0/15360.0;
      else if (x < 2.5)
        b = x*(x*(x*(x*(x*(7.0/60.0 - x*(1.0/120.0)) - 21.0/32.0) + 133.0/72.0) - 329.0/128.0) + 1267.0/960.0) + 1379.0/7680.0;
      else if (x < 3.5) { x -= 3.5; double xxx = x*x*x; b = (1.0/720.0)*xxx*xxx; }
      break;
    }
    case 7: {
      x = abs(x);
      if (x < 1.0) { double a = x*x; b = a*(a*(a*(x*(1.0/144.0) - 1.0/36.0) + 1.0/9.0) - 1.0/3.0) + 151.0/315.0; }
      else if (x < 2.0)
        b = x*(x*(x*(x*(x*(x*(1.0/20.0 - x*(1.0/240.0)) - 7.0/30.0) + 1.0/2.0) - 7.0/18.0) - 1.0/10.0) - 7.0/90.0) + 103.0/210.0;
      else if (x < 3.0)
        b = x*(x*(x*(x*(x*(x*(x*(1.0/720.0) - 1.0/36.0) + 7.0/30.0) - 19.0/18.0) + 49.0/18.0) - 23.0/6.0) + 217.0/90.0) - 139.0/630.0;
      else if (x < 4.0) { double a = 4.0 - x; double xxx = a*a*a; b = (1.0/5040.0)*xxx*xxx*a; }
      break;
    }
    default: throw std::invalid_argument("beta: degree must be in [0..7]");
  }
  return b;
}

} // namespace lsresize
