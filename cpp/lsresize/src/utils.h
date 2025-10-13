// splineops/cpp/lsresize/src/utils.h
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

namespace lsresize {

inline int border(int size, int degree, double tol = 1e-10) {
  if (degree <= 1) return 0;
  double z;
  switch (degree) {
    case 2: z = std::sqrt(8.0) - 3.0; break;
    case 3: z = std::sqrt(3.0) - 2.0; break;
    case 4: z = std::sqrt(664.0 - std::sqrt(438976.0)) + std::sqrt(304.0) - 19.0; break;
    case 5: z = std::sqrt(135.0/2.0 - std::sqrt(17745.0/4.0)) + std::sqrt(105.0/4.0) - 6.5; break;
    case 6: z = -0.488294589303044755130118038883789062112279161239377608394; break;
    case 7: z = -0.5352804307964381655424037816816460718339231523426924148812; break;
    default: throw std::invalid_argument("border: degree [0..7]");
  }
  int horiz = 2 + static_cast<int>(std::log(tol)/std::log(std::abs(z)));
  return std::min(horiz, size);
}

inline void calculate_final_size_1d(bool inversable, int input, double zoom, int& working, int& final_) {
  if (!inversable) { working = input; final_ = (int)std::llround(working * zoom); return; }
  working = input;
  int s = (int)std::llround(std::llround((working - 1) * zoom) / zoom);
  while (working - 1 - s != 0) {
    ++working;
    s = (int)std::llround(std::llround((working - 1) * zoom) / zoom);
  }
  final_ = (int)std::llround((working - 1) * zoom) + 1;
}

// periodic mirror mapping with sign for antisymmetric boundaries
struct MirrorIndex { int idx; int sign; };
inline MirrorIndex mirror_index(int k, int N, bool symmetric) {
  if (N == 1) return {0, 1};
  const int period_sym  = 2*N - 2;
  const int period_asym = 2*N - 3;
  const int P = symmetric ? period_sym : period_asym;

  auto mod = [&](int a, int m){ int r = a % m; return (r < 0) ? r + m : r; };
  int t = mod(k, P);

  if (symmetric) {
    int j = (t >= N) ? (2*N - 2 - t) : t;
    return { j, 1 };
  } else {
    if (t < N)       return { t,  1 };
    if (t == N)      return { N-2, -1 };
    int j = 2*N - 3 - t;
    int sign = ( ((t - N) & 1) ? -1 : 1 );
    if (j < 0) j = 0;
    if (j >= N) j = N-1;
    return { j, sign };
  }
}

} // namespace lsresize
