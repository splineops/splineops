// splineops/cpp/lsresize/src/utils.h
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <algorithm>
#include <limits>

namespace lsresize {

inline int border(
  int size, 
  int degree, 
  double tol = 1e-10) 
{
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

inline int calculate_output_size_1d(
  int input,
  double zoom)
{
  if (input <= 0) {
    throw std::invalid_argument("input length must be positive");
  }
  if (!std::isfinite(zoom) || zoom <= 0.0) {
    throw std::invalid_argument("zoom must be finite and positive");
  }

  const auto checked_round = [](long double value) -> int {
    constexpr long double max_int =
        static_cast<long double>(std::numeric_limits<int>::max());
    if (!std::isfinite(value) || value > max_int) {
      throw std::overflow_error("resized axis length exceeds native limits");
    }
    if (value < 0.5L) {
      return 0;
    }
    return static_cast<int>(std::llround(value));
  };

  return std::max(
      1,
      checked_round(
          static_cast<long double>(input) *
          static_cast<long double>(zoom)));
}

// Canonical endpoint-aligned scale after the discrete input/output lengths are
// known. Degenerate grids have no finite align-corners scale and must be
// handled explicitly by the caller.
inline double endpoint_aligned_scale(int input, int output)
{
  if (input <= 1 || output <= 1) {
    throw std::invalid_argument(
        "endpoint-aligned scale is undefined for singleton axes");
  }
  return static_cast<double>(output - 1) /
         static_cast<double>(input - 1);
}

// periodic mirror mapping with sign for antisymmetric boundaries
struct MirrorIndex { int idx; int sign; };
inline MirrorIndex mirror_index(
  std::int64_t k,
  int N, 
  bool symmetric) 
{
  if (N == 1) return {0, 1};
  const std::int64_t n = static_cast<std::int64_t>(N);
  const std::int64_t period_sym  = 2*n - 2;
  const std::int64_t period_asym = 2*n - 3;
  const std::int64_t P = symmetric ? period_sym : period_asym;

  const auto mod = [](std::int64_t a, std::int64_t m) {
    const std::int64_t r = a % m;
    return (r < 0) ? r + m : r;
  };
  const std::int64_t t = mod(k, P);

  if (symmetric) {
    const std::int64_t j = (t >= n) ? (period_sym - t) : t;
    return {static_cast<int>(j), 1};
  } else {
    if (t < n)       return {static_cast<int>(t),  1};
    if (t == n)      return {N - 2, -1};
    std::int64_t j = period_asym - t;
    const int sign = (((t - n) & 1) ? -1 : 1);
    if (j < 0) j = 0;
    if (j >= n) j = n - 1;
    return {static_cast<int>(j), sign};
  }
}

} // namespace lsresize
